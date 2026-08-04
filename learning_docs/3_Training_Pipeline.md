# HyMo Training Pipeline — Code Walkthrough

> **Prerequisite reading:** [`learning_docs/1_Model_Architecture.md`](1_Model_Architecture.md) for the
> model stack, [`learning_docs/4_Optimizations.md`](4_Optimizations.md) for the kernel + flags,
> [`learning_docs/6_Config_System.md`](6_Config_System.md) for `TrainingConfig` / `OptimizerConfig` /
> `SchedulerConfig`.
>
> **Files covered:**
> - `src/hymo/training/partition.py` — `ParameterPartition`, `partition_parameters`, `goes_to_adamw`
> - `src/hymo/training/optimizer.py` — `NorMuon`, `CautiousAdamW`, `Optimizers`, `build_optimizers`, `_newton_schulz_orthogonalize`
> - `src/hymo/training/scheduler.py` — `JointWSDScheduler` (3 decay kinds)
> - `src/hymo/training/fsdp.py` — `wrap_model_with_fsdp`, `fsdp_auto_wrap_policy`
> - `src/hymo/training/trainer.py` — `Trainer`, `train_step_result`, training loop
> - `src/hymo/training/checkpoint.py` — DCP-based `save_checkpoint`, `load_checkpoint`
> - `src/hymo/training/validation.py` — `compute_validation_loss`

> **No `hymo.training.train` module.** The trainer is a class
> (`Trainer`); you wire it from your driver script. See
> [`../SKILLS.md`](../SKILLS.md) §Skill 4.

---

## Table of Contents

1. [Training pipeline at a glance](#1-training-pipeline-at-a-glance)
2. [Parameter partitioning (`partition.py`)](#2-parameter-partitioning-partitionpy)
3. [Dual optimizer: NorMuon + CautiousAdamW (`optimizer.py`)](#3-dual-optimizer-normuon--cautiousadamw-optimizerpy)
4. [Joint WSD scheduler (`scheduler.py`)](#4-joint-wsd-scheduler-schedulerpy)
5. [FSDP-2 wrapping (`fsdp.py`)](#5-fsdp-2-wrapping-fsdppy)
6. [The training loop (`trainer.py`)](#6-the-training-loop-trainerpy)
7. [Checkpointing (`checkpoint.py`)](#7-checkpointing-checkpointpy)
8. [In-training validation (`validation.py`)](#8-in-training-validation-validationpy)
9. [End-to-end step trace](#9-end-to-end-step-trace)
10. [Interview Q&A](#10-interview-qa)

---

## 1. Training pipeline at a glance

```
                    ┌──────────────────────────────────────────────┐
                    │   Trainer (trainer.py)                       │
                    │   - threads the 4 opt flags                  │
                    │   - initializes W&B (rank 0)                 │
                    │   - builds optimizers + scheduler            │
                    └──────────────────────┬───────────────────────┘
                                           │
              ┌────────────────────────────┼──────────────────────────────┐
              ▼                            ▼                              ▼
   ┌────────────────────┐      ┌────────────────────────┐     ┌─────────────────────┐
   │   partition.py     │      │   optimizer.py         │     │   scheduler.py      │
   │ goes_to_adamw():   │      │ NorMuon (Newton-       │     │ JointWSDScheduler   │
   │  ndim<2, embed,    │      │  Schulz + cautious WD) │     │  warmup-stable-decay│
   │  MoE experts, A_log│──┬──▶│ CautiousAdamW (cautious│     │  get_factor(step)   │
   │  → AdamW           │  │   │  mask on WD)           │     │  decay ∈ linear,    │
   │ everything else    │  │   │ Optimizers(nor_muon,   │     │   cosine, sqrt      │
   │ → NorMuon          │  └──▶│  adamw) (partition     │     │ state_dict/         │
   │                    │      │  holder)               │     │  load_state_dict    │
   └────────────────────┘      │ build_optimizers()     │     └─────────────────────┘
                               └────────────────────────┘
                                           │
                                           ▼
                                   ┌────────────────────┐
                                   │   fsdp.py          │
                                   │ wrap_model_with_   │
                                   │  fsdp() — BF16     │
                                   │  mixed precision   │
                                   │  auto-wrap per     │
                                   │  GDN/MLA block     │
                                   └────────────────────┘
                                           │
                                           ▼
                                   ┌────────────────────┐
                                   │   checkpoint.py    │
                                   │ save_checkpoint()  │
                                   │ load_checkpoint()  │
                                   │  (DCP + RNG state) │
                                   └────────────────────┘
```

At runtime, `Trainer.train_step` orchestrates everything: forward
(with MTP), backward, gradient accumulation, grad clip, optimizer
step (NorMuon + AdamW), EMA gate-bias update, scheduler step.

---

## 2. Parameter partitioning (`partition.py`)

### 2.1 The rule — `goes_to_adamw(name, param)` (line 15)

A parameter goes to **AdamW** if and only if any of these hold:

| Condition | Why |
|---|---|
| `param.ndim < 2` (scalar / vector) | Newton–Schulz orthogonalization needs a 2D matrix. |
| `param.ndim > 2` (tensor with 3+ dims) | Same — Muon is a matrix-shaped optimizer. |
| `name.endswith("embed.weight")` or `"head.weight"` | Embeddings have one row per token; orthogonalization would scramble the row-meaning. |
| `name.endswith(".gate.weight")` or `".gate.bias"` | MoE gate (16-way softmax) is small and 2D-shape anyway, but it's a routing signal — let AdamW handle it as a regular 2D. |
| `name.endswith((".experts.w1.weight" / ".w2.weight" / ".w3.weight"))` (routed MoE) | Expert weights are 2D but very deep; NorMuon's orthogonalization isn't beneficial at this width. |
| `name.endswith((".shared_expert.w1.weight" / ...))` (shared MoE) | Same as routed. |
| `name.endswith((".A_log" / ".dt_bias" / ".D"))` (GDN scalars) | 1D, falls into the `ndim < 2` rule but listed for clarity. |
| `name.endswith("norm.weight")` | RMSNorm scale, 1D. |

**Everything else** (the big 2D dense matrices: attention `q_proj`,
`k_proj`, `v_proj`, `o_proj`, MLA's `q_a_proj` / `q_b_proj` / `kv_a_proj_with_mqa`
/ `kv_b_proj` / `wk` / `wv` / `wo`, and the GDN `in_proj` /
`out_proj`) goes to **NorMuon**.

```python
def goes_to_adamw(name: str, param: nn.Parameter) -> bool:
    if param.ndim < 2:
        return True
    if param.ndim > 2:
        return True
    if name.endswith("embed.weight") or name.endswith("head.weight"):
        return True
    if name.endswith(".gate.weight") or name.endswith(".gate.bias"):
        return True

    if ".experts." in name and (
        name.endswith(".w1.weight")
        or name.endswith(".w2.weight")
        or name.endswith(".w3.weight")
    ):
        return True
    if (
        name.endswith(".shared_expert.w1.weight")
        or name.endswith(".shared_expert.w2.weight")
        or name.endswith(".shared_expert.w3.weight")
    ):
        return True

    if name.endswith(".A_log") or name.endswith(".dt_bias") or name.endswith(".D"):
        return True

    return name.endswith("norm.weight")
```

The string match is exact — there's no fuzzy logic. A typo like
`".experts.0.w1.weight"` would still match (the leading `.experts.`
and trailing `.w1.weight`). A new parameter category needs an
explicit rule or it defaults to NorMuon.

### 2.2 `partition_parameters(model)` → `ParameterPartition`

`ParameterPartition` is the most boring class in the codebase —
two `list[nn.Parameter]` slots, one per optimizer:

```python
class ParameterPartition:
    __slots__ = ("adamw", "nor_muon")
    def __init__(self) -> None:
        self.adamw: list[nn.Parameter] = []
        self.nor_muon: list[nn.Parameter] = []
```

`partition_parameters(model)` walks `model.named_parameters()` and
funnels each parameter into the right bucket. It's a single pass;
no shuffling, no sorting.

### 2.3 Why this partition?

The intuition:

- **NorMuon** orthogonalizes 2D update matrices, which has been
  shown to converge faster than AdamW on dense attention-shaped
  matrices (the original Muon paper, then "NorMuon" by Apple
  Research for the cautious WD addition).
- **AdamW** is a robust fallback for everything else — scalars,
  embeddings, MoE experts. NorMuon's orthogonalization on a
  `(16, 896)` gate matrix would be wasteful; on `(vocab_size=64256,
  dim=896)` embeddings, it would be destructive (one row = one
  token's identity, orthogonalization randomizes rows).
- The **MoE expert exclusion** is HyMo-specific: at this width
  (`moe_inter_dim = 2304`), the experts are roughly the same shape
  as attention projections, but the goal of MoE training is *load
  balance*, and Muon's per-row update normalization can fight the
  EMA gate-bias adjustment. Keeping experts on AdamW + a manual
  bias update is the cleaner split.

---

## 3. Dual optimizer: NorMuon + CautiousAdamW (`optimizer.py`)

The two optimizers live in one file because they share the
`Optimizer` base, the cautious-WD mechanic, and the FP32-master-weight
pattern.

### 3.1 `_newton_schulz_orthogonalize(g, iterations=5)` (line 19)

Newton–Schulz iteration to approximate the matrix sign function:

```python
@torch.no_grad()
def _newton_schulz_orthogonalize(g: torch.Tensor, iterations: int = 5) -> torch.Tensor:
    """Newton-Schulz iterative orthogonalization to approximate matrix sign function."""
    norm = g.norm()
    if norm < 1e-12:
        return g
    g = g / norm
    for _ in range(iterations):
        g = 1.5 * g - 0.5 * g @ g.T @ g
    return g * norm
```

The iteration is `g ← 1.5 g − 0.5 g gᵀ g`. Starting from a
Frobenius-normalized matrix, 5 iterations is sufficient to get an
orthogonal "sign" matrix within ~1e-3 spectral error. The final
multiply by `g.norm()` restores the original scale (Muon keeps the
update magnitude).

The `_norm < 1e-12` early-return handles the "vanishing gradient"
edge case (returns the zero gradient unchanged; the optimizer will
then see a no-op update).

### 3.2 `NorMuon` (line 32)

`torch.optim.Optimizer` subclass with constructor:

```python
class NorMuon(Optimizer):
    def __init__(
        self,
        params: Iterable[nn.Parameter],
        lr: float = 0.02,
        momentum: float = 0.95,
        betas: tuple[float, float] = (0.95, 0.95),
        eps: float = 1e-8,
        weight_decay: float = 0.1,
        cautious_wd: bool = True,
        ns_iterations: int = 5,
    ) -> None:
```

All defaults match `OptimizerConfig` defaults. The
`ns_iterations=5` is the Newton–Schulz iteration count from
`goes_to_adamw` above.

The `step()` method (line 62) is roughly:

1. For each param group, for each parameter:
   1. Pull `g = param.grad` (already accumulated across
      micro-batches).
   2. Apply momentum (Polyak) — maintain `state["momentum_buffer"]`,
      `momentum_buffer *= momentum; m_b.add_(g)`.
   3. **Cautious mask** (when `cautious_wd=True`): for each
      coordinate, mask out the WD term unless `g * param > 0`
      (decay only where the gradient agrees with the parameter
      sign — Liang et al. 2024).
   4. Orthogonalize: `u = _newton_schulz_orthogonalize(m_b,
      iterations=ns_iterations)`.
   5. Decoupled WD: `param *= (1 - lr * wd)` (with the cautious
      mask).
   6. Update: `param.add_(u, alpha=-lr)`.
   7. **FP32 master weights**: the master copy is held in `state`
      and synced back. (In production with FSDP, BF16 parameters
      are what's read by forward; FP32 masters live on each rank.)

The cautious mask is the headline NorMuon-vs-Muon addition: it
removes the sign-disagreement problem where decoupled WD pulls
parameters in a direction the gradient is pushing against, undoing
the actual update.

### 3.3 `CautiousAdamW` (line 111)

A second `torch.optim.Optimizer` subclass that does the standard
AdamW update (m, v, bias correction) with the same cautious mask
on the WD term:

```python
class CautiousAdamW(Optimizer):
    def __init__(
        self, params, lr: float = 3e-4,
        betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8, weight_decay: float = 0.0,
        embed_weight_decay: float = 0.1,  # separate WD for embed/head
        cautious_wd: bool = True,
    ) -> None:
```

`weight_decay=0.0` is the AdamW default; embeddings + head
override to `embed_weight_decay = 0.1` (a separate param-group
distinction in `build_optimizers` below).

### 3.4 `Optimizers(nor_muon, adamw)` — the partition holder (line 188)

```python
class Optimizers:
    """The partitioned model parameter grouping lists."""
    def __init__(self, nor_muon: NorMuon | None, adamw: CautiousAdamW) -> None:
        self.nor_muon = nor_muon
        self.adamw = adamw
```

`nor_muon` can be `None` if the model has no NorMuon-eligible
parameters (e.g. a tiny surrogate in tests). `adamw` is always
present.

### 3.5 `build_optimizers(model, config)` (line 209)

```python
def build_optimizers(model, config) -> Optimizers: ...
```

The function:

1. Runs `partition_parameters(model)` to get the
   `ParameterPartition`.
2. Builds the NorMuon optimizer over `partition.nor_muon` (or
   `None` if empty), using `config.muon_lr`, `config.muon_momentum`,
   `config.muon_betas`, `config.muon_eps`, `config.muon_weight_decay`,
   `config.cautious_wd`.
3. Builds the CautiousAdamW optimizer over `partition.adamw`,
   using `config.adamw_lr`, `config.adamw_betas`,
   `config.adamw_eps`. Then it **further splits** the AdamW group
   by name: anything ending in `embed.weight` / `head.weight` gets
   `weight_decay = config.adamw_embed_weight_decay` (default `0.1`);
   everything else gets `config.adamw_weight_decay` (default `0.0`).
4. Returns the `Optimizers(nor_muon, adamw)` holder.

This is what `Trainer.__init__` calls. The trainer then holds
`self.optimizers.nor_muon` and `self.optimizers.adamw`.

### 3.6 FP32 master weights

`OptimizerConfig.master_weights_dtype: str = "float32"` (or
`"bfloat16"`) is read by both optimizers' `step()`. The pattern:

- Each parameter gets a `state["master"]` FP32 (or BF16) tensor
  initialized to the parameter's data on first step.
- The actual update is computed in master precision.
- The parameter's BF16 storage is overwritten from the master
  copy at the end of `step()`.

This is the standard "master weights for stability" pattern.
Total optimizer state per parameter ≈ 2× (master) + 4× (AdamW `m`,
`v`) = 6× in FP32. With `master_weights_dtype = "bfloat16"`, it's
2× (master) + 4× (AdamW state) = 6× in BF16, but in practice
master weights stay FP32 for the full 30 B run since the stability
benefit is the primary motivation.

---

## 4. Joint WSD scheduler (`scheduler.py`)

### 4.1 The shape — `JointWSDScheduler`

`JointWSDScheduler` is **not** a `torch.optim.lr_scheduler`; it's
a small class with its own `step()` and `get_factor(step)`. The
trainer reads `get_factor(step + 1)` before each optimizer step
and sets `param_group["lr"] = base_lr * factor` for every group.

```python
class JointWSDScheduler:
    def __init__(self, config: SchedulerConfig) -> None:
        self._config = config
        self.warmup_steps = config.warmup_steps   # 1144
        self.stable_steps = config.stable_steps   # 47_492
        self.decay_steps = config.decay_steps     # 8_583
        self.min_lr_ratio = config.min_lr_ratio   # 0.05
        self.decay_kind: DecaySchedule = config.decay  # 'linear' / 'cosine' / 'sqrt'
        self._step: int = 0

    def get_factor(self, step: int) -> float:
        """Return the LR multiplier for the given step."""
```

### 4.2 `get_factor(step)` — the three phases

```python
def get_factor(self, step: int) -> float:
    if step < self.warmup_steps:
        return step / max(self.warmup_steps, 1)           # (A) warmup: 0 → 1

    stable_end = self.warmup_steps + self.stable_steps
    if step < stable_end:
        return 1.0                                       # (B) stable: 1

    decay_progress = (step - stable_end) / max(self.decay_steps, 1)
    decay_progress = min(decay_progress, 1.0)
    decay_val = self._decay_factor(decay_progress, self.decay_kind)
    return self.min_lr_ratio + (1.0 - self.min_lr_ratio) * decay_val
    # (C) decay: 1 → min_lr_ratio, shaped by decay kind
```

With `total_steps = 57,220`, `warmup_frac = 0.02`, `stable_frac =
0.83`, `decay_frac = 0.15`, `min_lr_ratio = 0.05`:

- Phase A: steps `0..1143`, factor goes `0/1144 .. 1143/1144`.
- Phase B: steps `1144..48,635`, factor = `1.0`.
- Phase C: steps `48,636..57,219`, factor goes from `1.0` to
  `0.05` along the chosen decay shape (`linear` by default;
  `cosine` and `sqrt` are also available).

The `step < warmup_steps` branch returns `step / warmup_steps` so
step 0 is `0.0`, step `warmup_steps - 1` is `(warmup_steps - 1) /
warmup_steps`. Note the scheduler does not return `0` at step 0
in this version — it's `0/1144 = 0.0`, which is fine (the
optimizer sees a 0 LR for the first micro-batch).

### 4.3 Three decay shapes

`_decay_factor(progress, kind)` (line 67) is a `@staticmethod`:

```python
@staticmethod
def _decay_factor(progress: float, kind: DecaySchedule) -> float:
    if not 0.0 <= progress <= 1.0:
        raise ValueError(f"progress must be in [0, 1], got {progress}")
    if kind == "linear":
        return 1.0 - progress
    if kind == "cosine":
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    if kind == "sqrt":
        return math.sqrt(1.0 - progress)
    raise ValueError(f"Unknown decay kind: {kind!r}")
```

`progress=0` returns `1.0` in all three kinds (start of decay =
full LR); `progress=1` returns `0.0` (linear/cosine-sqrt) — which
gets shifted to `min_lr_ratio` by the caller.

### 4.4 `state_dict` / `load_state_dict`

```python
def state_dict(self) -> dict[str, int]:
    return {"step": self._step}

def load_state_dict(self, state: dict[str, int]) -> None:
    self._step = state.get("step", 0)
```

The scheduler's state is just its step counter. Saving and
restoring is two lines per direction.

### 4.5 Why WSD over cosine?

The classic cosine schedule decays from step 0 to `total_steps`;
WSD holds the peak LR for `stable_frac` (~83% of training here)
before the decay. Two practical wins:

1. **Continued pretraining**: if the 30 B run is extended to 50 B
   tokens at a later date, only the scheduler's `total_steps`,
   `stable_steps`, and `decay_steps` need updating — no LR
   retuning.
2. **Ablation comparability**: the same `warmup_frac` /
   `stable_frac` / `decay_frac` fractions apply to every
   7.5 B-token ablation run (see
   [`learning_docs/5_Evaluation_and_Ablations.md`](5_Evaluation_and_Ablations.md)
   §3.2). With cosine, the peak LR would shift to match run
   length — apples-to-oranges comparison.

See [`docs/concepts/08-wsd-scheduler.md`](../docs/concepts/08-wsd-scheduler.md) for the math derivation.

---

## 5. FSDP-2 wrapping (`fsdp.py`)

### 5.1 The surface

`fsdp.py` exposes two functions:

```python
def fsdp_auto_wrap_policy(module, recurse, non_blocking) -> bool:
    """FSDP auto-wrap policy: wrap per-layer blocks (GDN, MLA)."""
    from hymo.models.gdn import GatedDeltaNetBlock
    from hymo.models.mla import MLABlock
    return isinstance(module, (GatedDeltaNetBlock, MLABlock))


def wrap_model_with_fsdp(model, config, *, world_size=None,
                        auto_wrap_policy=None, **kwargs):
    """Wrap model inside FullyShardedDataParallel wrapper."""
```

### 5.2 Auto-wrap policy

`fsdp_auto_wrap_policy` returns `True` for `GatedDeltaNetBlock` and
`MLABlock` — meaning FSDP wraps each layer as its own FSDP unit
(so all-gather/reduce-scatter happens at the block boundary, not
at the parameter boundary). This is the right granularity: blocks
are big enough to amortize the comm overhead, small enough that no
single block's peak memory is unreasonable.

### 5.3 `wrap_model_with_fsdp`

```python
def wrap_model_with_fsdp(model, config, *, world_size=None,
                        auto_wrap_policy=None, **kwargs):
    try:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp import MixedPrecision
    except ImportError:
        return model
```

The function tries to import `torch.distributed.fsdp`; if not
available (e.g. a CPU-only debug run), it returns the model
unwrapped. With FSDP available:

1. Builds a `MixedPrecision` policy from
   `config.fsdp_mixed_precision` (`"bfloat16"` in production).
2. Wraps the model with `FSDP(...)`, passing the policy and the
   auto-wrap policy.
3. Returns the wrapped model.

`world_size` defaults to `None`, which then uses PyTorch's
distributed init to query it.

See [`docs/concepts/09-fsdp2.md`](../docs/concepts/09-fsdp2.md) for the ZeRO-3 vs FSDP-2 internals.

---

## 6. The training loop (`trainer.py`)

### 6.1 `Trainer.__init__(config, model)` (line 48)

```python
def __init__(self, config: HyMoConfig, model: HyMo) -> None:
    self._config = config
    self.model = model
    self._thread_optimization_flags()
    # ... W&B init (rank 0, master only)
    self.optimizers = build_optimizers(model, config.optimizer)
    self.scheduler = JointWSDScheduler(config.scheduler)
    # ... base LRs, step counters, MTP flag
```

Construction order matters:

1. **Store config + model.**
2. **Thread optimization flags** — walks `model.modules()` once
   and sets `use_triton`, `use_compile`, `use_mixed_precision`,
   `use_cuda_graphs` on the appropriate blocks. (See
   [`learning_docs/4_Optimizations.md`](4_Optimizations.md) §2.1.)
3. **Initialize W&B** — but only on rank 0 (or `WANDB_MODE` not
   disabled):
   ```python
   _wandb_disabled = os.environ.get("WANDB_MODE", "").lower() in ("disabled", ...)
   if not _wandb_disabled and (not dist.is_initialized() or dist.get_rank() == 0):
       wandb.init(project="HyMo", config=cfg_dict, resume="allow")
   ```
   The `resume="allow"` means a W&B run ID collision reconnects
   rather than rejecting — useful for resuming from a checkpoint.
4. **Build optimizers + scheduler.**
5. **Snapshot base LRs** (`_base_lr_muon`, `_base_lr_adamw`) — the
   per-step LR is `base * factor`, so the base is held here.
6. **Reset step counters** (`step = 0`, `micro_step = 0`,
   `token_count = 0`, `best_loss = inf`).
7. **Set `_has_mtp`** from `config.model.mtp_depth > 0`.

### 6.2 `train_step(tokens, targets)` (line 114)

The full forward + backward + (sometimes) optimizer step:

```python
def train_step(self, tokens, targets) -> train_step_result:
    self.model.train()

    # 1. Forward (with MTP if enabled)
    if self._has_mtp:
        mtp_module = getattr(self.model, "_mtp", None)
        if mtp_module is not None:
            logits, mtp_outputs = mtp_module.forward(tokens)
        else:
            logits = self.model.forward(tokens)
            mtp_outputs = []
    else:
        logits = self.model.forward(tokens)
        mtp_outputs = []

    # 2. Loss
    V = logits.size(-1)
    main_loss = F.cross_entropy(logits[:, :-1].reshape(-1, V), targets[:, :-1].reshape(-1))
    total_loss = main_loss
    mtp_details = {}
    for i, mtp_out in enumerate(mtp_outputs):
        mtp_loss = F.cross_entropy(mtp_out.logits.reshape(-1, V), mtp_out.targets.reshape(-1))
        weighted = mtp_loss * mtp_out.loss_weight
        total_loss = total_loss + weighted
        mtp_details[f"mtp_{i}_loss"] = mtp_loss.item()
        mtp_details[f"mtp_{i}_weighted"] = weighted.item()

    # 3. NaN skip
    if self._config.training.loss_nan_skip and (torch.isnan(total_loss) or torch.isinf(total_loss)):
        self.model.zero_grad(set_to_none=True)
        return train_step_result(
            loss=float("nan"),
            grad_norm=0.0,
            lr_muon=self._current_lr_muon(),
            lr_adamw=self._current_lr_adamw(),
            is_update=False, skipped=True,
            metrics={"main_loss": float("nan"), **mtp_details},
        )

    # 4. Backward (scaled)
    scaled_loss = total_loss / self._config.training.gradient_accumulation_steps
    scaled_loss.backward()

    self.micro_step += 1
    is_update = (self.micro_step % self._config.training.gradient_accumulation_steps == 0)
    grad_norm_val = 0.0

    # 5. Optimizer step (only when is_update)
    if is_update:
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), max_norm=self._config.training.grad_clip, norm_type=2.0
        )
        grad_norm_val = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else float(grad_norm)

        factor = self.scheduler.get_factor(self.step + 1)
        if self.optimizers.nor_muon is not None and self._base_lr_muon is not None:
            for g in self.optimizers.nor_muon.param_groups:
                g["lr"] = self._base_lr_muon * factor
        for g in self.optimizers.adamw.param_groups:
            g["lr"] = self._base_lr_adamw * factor

        if self.optimizers.nor_muon is not None:
            self.optimizers.nor_muon.step()
        self.optimizers.adamw.step()

        # EMA gate-bias update
        self._update_moe_gate_biases()

        self.scheduler.step()
        self.model.zero_grad(set_to_none=True)
        self.step += 1

    self.token_count += tokens.numel()
    return train_step_result(...)
```

Walk through this:

| Phase | Lines | What |
|---|---|---|
| Forward (with MTP) | 122-131 | `mtp_module.forward(tokens)` returns `(logits, [mtp_out, mtp_out])` for `mtp_depth=2`. Without MTP, `model.forward(tokens)` returns a single logits tensor. |
| Loss | 133-151 | `F.cross_entropy(logits[:, :-1] vs targets[:, :-1])` — next-token prediction. MTP losses are added with `mtp_loss * mtp_out.loss_weight` (default `[0.3, 0.1]`). |
| NaN skip | 153-163 | If `total_loss` is NaN/Inf and `loss_nan_skip=True`, zero grad and return early with `is_update=False, skipped=True`. |
| Backward (scaled) | 165-166 | `scaled_loss.backward()` — dividing by `gradient_accumulation_steps` is what makes the accumulated loss mean-correct. |
| Update gate | 168-169 | `is_update = (micro_step % grad_accum == 0)` — True every `grad_accum` micro-steps. |
| Grad clip | 173-178 | `clip_grad_norm_` with `grad_clip=1.0`. The `grad_norm` is captured for logging. |
| LR update | 180-185 | Read `factor = scheduler.get_factor(step + 1)` and set every param group's LR. |
| Optimizer step | 187-189 | Step both optimizers (NorMuon can be `None`). |
| EMA gate-bias | 191 | `_update_moe_gate_biases()` — walks MoE modules, calls `update_gate_bias()`. |
| Scheduler step | 193 | Increments `scheduler._step`. |
| Zero grad | 194 | `set_to_none=True` is faster than `set_to_zero`. |
| Step counter | 196 | `self.step += 1`. |
| Token count | 198 | Tracked for "tokens seen so far" reporting. |

### 6.3 `train(data_iter, max_steps=None)` (line 252)

The driver-loop wrapper:

```python
def train(self, data_iter, max_steps=None) -> None:
    if max_steps is None:
        max_steps = self._config.scheduler.total_steps
    for tokens, targets in data_iter:
        if self.step >= max_steps:
            break
        result = self.train_step(tokens, targets)

        if result.is_update:
            # W&B log every `log_interval` updates
            if (self.step % self._config.training.log_interval == 0) and not result.skipped:
                wandb.log({"train/loss": result.loss, ...}, step=self.step)
            # Eval every `eval_interval` updates
            if self._config.training.eval_interval > 0 and (self.step % self._config.training.eval_interval == 0):
                eval_metrics = self.evaluate()
                if eval_metrics.get("val_loss", inf) < self.best_loss:
                    self.best_loss = eval_metrics["val_loss"]
                    self.save(tag="best")
            # Save every `save_interval` updates
            if self._config.training.save_interval > 0 and (self.step % self._config.training.save_interval == 0):
                self.save()
```

The cadence knobs are all in `TrainingConfig`:
- `log_interval = 50`
- `eval_interval = 2_000`
- `save_interval = 4_000`
- `max_keep = 2` (old checkpoints pruned)

### 6.4 `evaluate(val_bin_path=None)` (line 297)

```python
def evaluate(self, val_bin_path=None) -> dict[str, float]:
    training_cfg = self._config.training
    model_cfg = self._config.model
    from hymo.training.validation import DEFAULT_VAL_BIN

    metrics: ValMetrics = compute_validation_loss(
        self.model,
        batch_size=training_cfg.micro_batch_size,
        seq_len=training_cfg.max_seq_len,
        vocab_size=model_cfg.vocab_size,
        num_batches=min(4, 32),
        device=next(self.model.parameters()).device,
        val_bin_path=Path(val_bin_path) if val_bin_path else DEFAULT_VAL_BIN,
    )
    wandb.log({...}, step=self.step)
    return {"val_loss": metrics.loss, "val_ppl": metrics.ppl}
```

`num_batches=min(4, 32)` is a small-N validation — 4 batches is
enough for a noise-level signal during training; the full 32-batch
validation is reserved for the 6-eval suite run.

### 6.5 `_update_moe_gate_biases` (line 329)

```python
def _update_moe_gate_biases(self) -> None:
    """Apply EMA load-balancing to every MoE gate (aux-loss-free routing)."""
    from hymo.models.moe import DeepSeekMoE
    for module in self.model.modules():
        if isinstance(module, DeepSeekMoE):
            module.update_gate_bias()
```

A single walk over model modules, calling `update_gate_bias()` on
each `DeepSeekMoE` (8 of them in the production model — one per
MLA layer). The actual EMA logic is in
[`src/hymo/models/moe.py`](https://github.com) — see
[`learning_docs/4_Optimizations.md`](4_Optimizations.md) §5.3 and
[`docs/concepts/03-mixture-of-experts.md`](../docs/concepts/03-mixture-of-experts.md).

### 6.6 `save(tag=None)`, `load(path)` — DCP checkpointing

`save(tag=None)` (line 210):

```python
def save(self, tag=None) -> Path:
    if tag is None:
        tag = f"step_{self.step}"
    output_dir = Path(self._config.run.output_dir)
    ckpt_dir = output_dir / tag
    state = CheckpointState(step=self.step, token_count=self.token_count, best_loss=self.best_loss)
    save_checkpoint(path=ckpt_dir, model=self.model, optimizers=self.optimizers,
                    scheduler=self.scheduler, state=state)
    return ckpt_dir
```

`load(path)` (line 234) is the reverse:

```python
def load(self, path) -> int:
    p = Path(path)
    state = load_checkpoint(path=p, model=self.model, optimizers=self.optimizers, scheduler=self.scheduler)
    self.step = state.step
    self.token_count = state.token_count
    self.best_loss = state.best_loss
    return self.step
```

Both go through `checkpoint.py` (next section). The "best" tag
wins: `Trainer.train` saves to `best/` whenever a new
`val_loss` is the running minimum.

### 6.7 `_current_lr_muon` / `_current_lr_adamw` (lines 337-345)

```python
def _current_lr_muon(self) -> float:
    if self.optimizers.nor_muon is not None and len(self.optimizers.nor_muon.param_groups) > 0:
        return float(self.optimizers.nor_muon.param_groups[0].get("lr", 0.0))
    return 0.0

def _current_lr_adamw(self) -> float:
    if len(self.optimizers.adamw.param_groups) > 0:
        return float(self.optimizers.adamw.param_groups[0].get("lr", 0.0))
    return 0.0
```

Read the current LR from group[0]. These are logged to W&B every
`log_interval` updates.

---

## 7. Checkpointing (`checkpoint.py`)

### 7.1 `CheckpointState` (line 30)

```python
@dataclass
class CheckpointState:
    step: int = 0
    token_count: int = 0
    best_loss: float = float("inf")
    rng_state: dict[str, Any] | None = None
    metrics_extra: dict[str, Any] | None = None
```

`step` and `token_count` are the canonical progress markers;
`best_loss` is the running minimum for "best" auto-snapshots;
`rng_state` and `metrics_extra` are populated on save for
deterministic resume.

### 7.2 RNG capture

`_capture_rng_state()` (line 50) snapshots:

- Python `random` (`py_state` — `version`, `internalstate`, `gauss`).
- NumPy `np.random.get_state()`.
- PyTorch CPU `torch.random.get_rng_state()`.
- PyTorch CUDA per-device states (only if `torch.cuda.is_available`).

This is what makes `resume_from` truly deterministic — replay the
same shuffled data order + the same dropout masks + the same
optimizer momentum buffers.

### 7.3 `save_checkpoint(path, model, optimizers, scheduler, state)`

The save sequence:

1. Create the checkpoint dir (`path`).
2. Use `torch.distributed.checkpoint` (DCP) to save the model and
   optimizer state dicts to `path/model/` and `path/optim/`.
3. Write `state` (plus the captured RNG state) to `path/hymo_meta.json`
   via the `_JsonEncoder` (handles NumPy arrays + scalars).
4. Prune older checkpoints: keep the latest `max_keep = 2`.

### 7.4 `load_checkpoint(path, model, optimizers, scheduler) → CheckpointState`

The reverse:

1. Read `path/hymo_meta.json` → restore `step`, `token_count`,
   `best_loss`, `rng_state`.
2. Use DCP to load `path/model/` and `path/optim/`.
3. Call `optimizers.load_state_dict(...)` and
   `scheduler.load_state_dict(...)`.
4. If `rng_state` is present, restore Python/NumPy/PyTorch RNG
   states (CUDA is best-effort).

### 7.5 Why DCP not `torch.save`?

DCP shards the state dict across ranks during write and
reassembles during read. For a 1.86 B-param model on 4 ranks,
each rank writes ~465 MB of BF16 parameters instead of one rank
writing 1.86 GB. That makes checkpoint I/O bandwidth-comparable
to the per-rank HBM bandwidth.

---

## 8. In-training validation (`validation.py`)

### 8.1 `DEFAULT_VAL_BIN`

The default validation binary path is set in
`src/hymo/data/prepare_validation.py` as the absolute path to
`data/tokens/val.bin` (the tokenized FineWeb-Edu held-out split).

### 8.2 `compute_validation_loss(model, batch_size, seq_len, vocab_size, num_batches, device, val_bin_path) → ValMetrics`

The validation loss routine:

1. Open `val_bin_path` with `np.memmap` (zero-copy read).
2. Sample `batch_size × seq_len` token windows in a sliding
   pattern across the validation binary.
3. Run `model(tokens)` (without MTP — too costly for in-training
   eval).
4. Compute `F.cross_entropy(logits[:, :-1] vs targets[:, :-1])`
   per batch; mean over `num_batches`.
5. Return `ValMetrics(loss, ppl, num_batches, num_tokens)`.

```python
@dataclass
class ValMetrics:
    loss: float
    ppl: float
    num_batches: int
    num_tokens: int
```

The 4-batch cap (`num_batches=min(4, 32)`) keeps in-training
validation cheap — 4 × 4 × 4096 = 65 k tokens vs. the eval
suite's 32 × 4 × 4096 = 524 k tokens.

`ppl = math.exp(loss)`. The README's target "≤ 2.10 FineWeb-Edu
PPL" is computed this way.

---

## 9. End-to-end step trace

A single optimizer step on a 4-GPU A100 pod:

1. **Data fetch.** The driver iterates `data_iter`; each item is
   `(tokens, targets)` of shape `(4, 4096)` per rank.
2. **Forward (rank-local).** `Trainer.train_step` is called on
   each rank independently; with FSDP-2, the per-block all-gather
   is fired before the block forward.
3. **MTP forward (if enabled).** `mtp_module.forward(tokens)`
   returns `(main_logits, [mtp_out_1, mtp_out_2])`.
4. **Loss.** `F.cross_entropy(main_logits[:, :-1] vs targets[:, :-1])`
   plus weighted MTP losses.
5. **NaN skip.** If `loss_nan_skip` and total is NaN/Inf → zero
   grad, return early.
6. **Backward (scaled).** `total_loss / grad_accum.backward()` —
   FSDP-2 reduces-scatters across ranks at each block boundary.
7. **Grad accumulation gate.** `is_update = (micro_step % grad_accum == 0)`.
8. **If `is_update`:**
   1. `clip_grad_norm_(...)` — global norm clip (FSDP-2 aware
      via `model.parameters()` post all-gather).
   2. Read `factor = scheduler.get_factor(step + 1)`.
   3. Set every param group LR.
   4. Step both optimizers (NorMuon optional).
   5. `module.update_gate_bias()` on every `DeepSeekMoE`.
   6. `scheduler.step()`.
   7. `model.zero_grad(set_to_none=True)`.
   8. `step += 1`.
9. **`token_count += B * T`**. (Per micro-step; aggregate is
   ~`B * T * grad_accum * world_size` per optimizer step.)
10. **Cadence hooks** (in `train`):
    - Every `log_interval`: W&B log.
    - Every `eval_interval`: `evaluate()` → if `val_loss < best`,
      save as `best/`.
    - Every `save_interval`: save as `step_N/`.

---

## 10. Interview Q&A

**Q1. Why partition parameters between NorMuon and AdamW by name
pattern?**

> A: Because the optimizer choice is coupled to the parameter's
> role in the model, not its shape alone. AdamW handles
> embeddings (1 row = 1 token, orthogonalization would scramble
> rows), MoE experts (orthogonalization isn't beneficial at this
> width and fights the EMA gate-bias), and 1D scalars (norms,
> gate biases, GDN `A_log`, `dt_bias`, `D`). NorMuon handles
> the dense attention-shaped 2D matrices where orthogonalization
> gives its signature convergence win. Naming keeps the rule
> explicit and reviewable; a new parameter category has to be
> added or it defaults to NorMuon.

**Q2. Explain Newton–Schulz iteration in plain language.**

> A: It's a 5-step iteration that converts any 2D matrix `g` into
> an approximately orthogonal one (think "matrix sign function"
> or "all singular values = 1") by repeatedly applying
> `g ← 1.5 g − 0.5 g gᵀ g`. The trick is that this is a
> polynomial in `g gᵀ` (no SVD), runs entirely on GPU, and
> converges quadratically. After 5 iterations starting from a
> normalized matrix, the spectral error is ~1e-3 — close enough
> to orthogonal for the optimizer's purposes. The final multiply
> by the original Frobenius norm restores the magnitude.

**Q3. Why cautious weight decay instead of decoupled WD?**

> A: Decoupled WD pulls every parameter toward 0 by a fixed
> fraction regardless of the gradient. If the gradient is
> pushing in the opposite direction (e.g. a coordinate is
> trying to *grow* past 0 to satisfy the loss), the WD term
> fights the gradient and adds noise. Cautious WD masks out
> the WD term unless `g * param > 0` — i.e. only decays where
> the gradient is *agreeing* with going smaller. The result is
> that coordinates that should grow are allowed to grow.

**Q4. Why WSD over cosine for continued pretraining?**

> A: Cosine's decay shape depends on `total_steps`; if you
> extend a 30 B run to 50 B, you have to retune the peak LR to
> match the new total. With WSD, the LR is held at peak for
> `stable_steps`, so an extended run just sees a longer stable
> phase — no retuning. The same property makes ablation runs
> (7.5 B each) comparable to the v1.0 30 B run, because the
> fractions (2% / 83% / 15%) and shape are identical.

**Q5. Walk through what happens when `train_step` is called.**

> A: 1) forward (with MTP if enabled) → logits + mtp_outputs;
> 2) cross-entropy main + weighted MTP losses;
> 3) NaN-skip branch if `loss_nan_skip` and total is non-finite;
> 4) `scaled_loss.backward()` where
> `scaled = total / gradient_accumulation_steps`;
> 5) `is_update = micro_step % grad_accum == 0`. If yes:
> `clip_grad_norm_`, set LR from `scheduler.get_factor(step+1)`,
> step both optimizers, fire EMA gate-bias update, step
> scheduler, zero grad (`set_to_none=True`), `step += 1`.
> The result is `train_step_result` with `is_update`,
> `skipped`, `loss`, `grad_norm`, `lr_muon`, `lr_adamw`,
> `metrics`.

**Q6. Why does the EMA gate-bias update fire only on optimizer
steps, not micro-steps?**

> A: The EMA averages load statistics over many batches; updating
> on every micro-batch makes the bias noisy. Tying the update to
> the optimizer step (i.e. the accumulated micro-batch) means
> the EMA sees the *averaged* dispatch per step.

**Q7. Why DCP for checkpointing and not `torch.save`?**

> A: With 4 ranks each holding 465 MB of BF16 parameters +
> ~700 MB of optimizer state, `torch.save` on rank 0 forces one
> rank to gather and write ~5 GB. DCP shards the write across
> ranks (~1.25 GB per rank), making checkpoint I/O bandwidth-
> comparable to per-rank HBM bandwidth.

**Q8. Why is `accumulation_steps` the divisor on the loss, not
the gradient?**

> A: Mathematically equivalent — dividing the loss by `k` and
> then calling `.backward()` gives the same parameter gradient
> as accumulating `k` micro-batch gradients and not dividing. But
> the loss-divide version only calls `.backward()` once, which is
> one autograd graph build instead of `k`. The throughput win is
> ~10–20% at `k=8`.

---

## 11. Cross-links

- Walkthrough: `learning_docs/1_Model_Architecture.md` §3 (model top-level).
- Concepts: `docs/concepts/07-muon-optimizer.md` (Muon lineage + Newton–Schulz),
  `docs/concepts/08-wsd-scheduler.md` (WSD phases),
  `docs/concepts/09-fsdp2.md` (FSDP-2 mechanics),
  `docs/concepts/03-mixture-of-experts.md` (EMA gate-bias derivation).
- Tests: `tests/unit/test_training.py` (optimizer + scheduler + trainer tests).
- Evaluation: `learning_docs/5_Evaluation_and_Ablations.md` (the 6-eval suite +
  ablations), `learning_docs/4_Optimizations.md` (the four optimization flags).
