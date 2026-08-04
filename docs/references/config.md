# HyMo — Configuration Reference

> **Prerequisite reading:** none. This is the first doc to read if you want
> to understand a `configs/hymo_750m.yaml` field.
>
> **Files covered:**
> - `src/hymo/core/config.py` — the 5 frozen dataclasses + load/save
> - `src/hymo/core/config_validation.py` — cross-field invariant checks
> - `src/hymo/core/types.py` — semantic newtypes
>
> **Companion concepts:** see [`optimization.md`](../concepts/optimization.md) for why every
> field exists, [`optimization.md`](../concepts/optimization.md) for the scheduler
> fractions, [`optimization.md`](../concepts/optimization.md) for the optimizer
> hyperparameters.

---

## 1. Why a config system

HyMo's primary config is a **6 KB YAML file** (`configs/hymo_750m.yaml`)
that drives everything: the model architecture, the optimizer partition,
the LR schedule, the distributed training settings, and the run
identity. The config system exists to:

1. **Be the single source of truth.** Every architectural knob is in
   one place; code reads it via a single typed object.
2. **Make mutation impossible by accident.** Every config class is
   `@dataclass(frozen=True)`, so the training loop cannot accidentally
   bump a `dim` or change an LR mid-run.
3. **Make variants easy.** Derive an ablation config via
   `dataclasses.replace` (or the higher-level `derive_config`); the
   base config is never mutated.
4. **Fail loud.** Cross-field invariants (e.g. `n_heads % n_kv_groups == 0`)
   are checked in `__post_init__` and again in
   `core/config_validation.py::validate_full_config`, so a wrong YAML
   surfaces a clear error before any GPU time is spent.

The dependency graph is `core ← utils ← {models, training, data, eval}`
(see `AGENTS.md` §Engineering rules). `core` is **PyTorch-free** — it
imports nothing from `torch.*` (except the `DType` alias in `types.py`).
This means you can import `hymo.core.config` from a CPU machine, a CI
runner, or a config-lint script without paying the torch import cost.

---

## 2. The five sub-configs

`src/hymo/core/config.py` defines five `@dataclass(frozen=True)` classes.
`HyMoConfig` aggregates them. Each sub-config has its own `__post_init__`
that validates its fields independently.

### 2.1 `ModelConfig` (line 23)

Architectural hyperparameters — the only place in the code that defines
*what the model is*.

```python
@dataclass(frozen=True)
class ModelConfig:
    # Token + sequence
    vocab_size: int = 64_256          # BPE-64k + 256-byte fallback
    max_seq_len: int = 4_096

    # Stack
    n_layers: int = 32                # 8 MLA + 24 GDN
    dim: int = 896                    # residual stream width
    tie_embeddings: bool = True       # share embed/head weights

    # MLA (full attention) — see concepts/01
    n_heads: int = 16
    n_kv_groups: int = 4              # MQA-4: 4 query heads per KV head
    q_lora_rank: int = 224
    kv_lora_rank: int = 128           # low-rank KV compression
    head_dim: int = 128
    qk_rope_head_dim: int = 32        # 25% partial RoPE
    qk_nope_head_dim: int = 96
    v_head_dim: int = 128
    rope_theta: float = 10_000.0

    # GDN (linear attention) — see concepts/02
    gdn_d_state: int = 32
    gdn_d_conv: int = 4
    gdn_headdim: int = 32
    gdn_d_inner: int = 1_280
    gdn_chunk_size: int = 64          # see concepts/10 (Triton kernel)

    # NoPE-hybrid — see concepts/04 + 11
    nope_hybrid_gdn_enabled: bool = False  # v1.0 OFF, v1.1 ablation

    # MoE — see concepts/03
    n_routed_experts: int = 16
    n_shared_experts: int = 1
    n_activated_experts: int = 2     # top-2 routing
    moe_inter_dim: int = 2_304
    moe_ema_alpha: float = 0.02      # EMA decay for gate-bias update
    moe_capacity_factor: float = 1.5

    # DenseFFN on GDN blocks
    inter_dim: int = 2_560

    # MTP — see concepts/05
    mtp_depth: int = 2
    mtp_loss_weights: tuple[float, ...] = (0.3, 0.1)
    mtp_inter_dim: int = 2_304

    # Logit softcap (PaLM-style)
    logit_softcap: float = 15.0

    # μP init — see concepts/06
    mup_init: bool = True
```

#### Validation in `__post_init__` (lines 78-122)

| Check | Error |
|---|---|
| `vocab_size > 0`, `n_layers > 0`, `dim > 0` | `ValueError` |
| `n_heads % n_kv_groups == 0` | `ValueError("n_heads must be a multiple of n_kv_groups")` |
| `qk_rope_head_dim + qk_nope_head_dim == head_dim` | `ValueError("partial RoPE dims must sum to head_dim")` |
| `gdn_d_inner % gdn_headdim == 0` | `ValueError("gdn_d_inner must be a multiple of gdn_headdim")` |
| `n_activated_experts <= n_routed_experts` | `ValueError` |
| `mtp_depth >= 0` | `ValueError` |
| If `mtp_depth == 0`: `mtp_loss_weights` must be empty | `ValueError` |
| If `mtp_depth > 0`: `len(mtp_loss_weights) == mtp_depth`, all `>= 0` | `ValueError` |
| `logit_softcap >= 0` | `ValueError` |

The `qk_rope + qk_nope == head_dim` check is what enforces the
"25% partial RoPE" invariant — there's no way to mis-configure it
silently.

#### Properties (lines 124-149)

```python
@property
def n_mla_layers(self) -> int:
    return self.n_layers // 4     # 8

@property
def n_gdn_layers(self) -> int:
    return self.n_layers - self.n_mla_layers  # 24

@property
def mla_positions(self) -> frozenset[int]:
    """{0, 4, 8, 12, 16, 20, 24, 28} — every 4th layer starting at 0."""
    return frozenset(i * 4 for i in range(self.n_mla_layers))

@property
def gdn_positions(self) -> frozenset[int]:
    return frozenset(i for i in range(self.n_layers) if i not in self.mla_positions)

@property
def nope_hybrid_gdn_positions(self) -> frozenset[int]:
    """When enabled, the 7 GDN layers immediately after each MLA position."""
    if not self.nope_hybrid_gdn_enabled:
        return frozenset()
    return frozenset(mla - 1 for mla in self.mla_positions if mla > 0)
```

`mla_positions` and `gdn_positions` are computed from `n_layers` so
the 3:1 ratio is **structural**, not a config field. A 28-layer
stack would still get a 3:1 split (`n_mla_layers = 7`, `n_gdn_layers = 21`).
The NoPE-hybrid subset is exactly `{3, 7, 11, 15, 19, 23, 27}` for the
default 32-layer stack.

### 2.2 `OptimizerConfig` (line 153)

```python
@dataclass(frozen=True)
class OptimizerConfig:
    # NorMuon — drives attention + GDN 2D matrices (excludes MoE experts).
    muon_lr: float = 0.02
    muon_momentum: float = 0.95
    muon_betas: tuple[float, float] = (0.95, 0.95)
    muon_eps: float = 1e-8
    muon_weight_decay: float = 0.1

    # AdamW — embed/head/norm/gate/scalars + MoE experts.
    adamw_lr: float = 3e-4
    adamw_betas: tuple[float, float] = (0.9, 0.95)
    adamw_eps: float = 1e-8
    adamw_weight_decay: float = 0.0
    adamw_embed_weight_decay: float = 0.1  # L2 only on the embed/head

    # Master weight precision
    master_weights_dtype: str = "float32"

    # Cautious weight decay (Liang et al. 2024)
    cautious_wd: bool = True
```

#### Validation in `__post_init__` (lines 176-189)

| Check | Error |
|---|---|
| `muon_lr > 0`, `adamw_lr > 0` | `ValueError` |
| `0 <= muon_momentum < 1` | `ValueError` |
| `0 <= beta < 1` for both `muon_betas` and `adamw_betas` | `ValueError` |
| `master_weights_dtype in {"float32", "bfloat16"}` | `ValueError` |

The `cautious_wd: bool = True` flag drives the cautious-mask logic in
`CautiousAdamW.step` (see `concepts/optimization.md`).

### 2.3 `SchedulerConfig` (line 193)

```python
@dataclass(frozen=True)
class SchedulerConfig:
    total_steps: Step = Step(57_220)   # NewType'd int
    warmup_frac: float = 0.02
    stable_frac: float = 0.83
    decay_frac: float = 0.15
    min_lr_ratio: float = 0.05          # final LR = min_lr_ratio * base LR
    decay: str = "linear"               # 'linear' | 'cosine' | 'sqrt'
```

#### Validation in `__post_init__` (lines 203-222)

| Check | Error |
|---|---|
| `total_steps > 0` | `ValueError` |
| `0 < warmup_frac < 1` (same for stable, decay) | `ValueError` |
| `warmup + stable + decay == 1.0` (within `1e-6`) | `ValueError` |
| `0 <= min_lr_ratio < 1` | `ValueError` |
| `decay in {"linear", "cosine", "sqrt"}` | `ValueError` |

#### Properties (lines 224-234)

```python
@property
def warmup_steps(self) -> int:  return int(self.total_steps * self.warmup_frac)   # 1144
@property
def stable_steps(self) -> int:  return int(self.total_steps * self.stable_frac)   # 47492
@property
def decay_steps(self) -> int:   return int(self.total_steps * self.decay_frac)    # 8583
```

The 57,220-step total is `30 B tokens / 524,288 tokens per step`
(see `TrainingConfig.per_step_tokens` below).

### 2.4 `TrainingConfig`

```python
@dataclass(frozen=True)
class TrainingConfig:
    # Batch
    micro_batch_size: int = 4
    gradient_accumulation_steps: int = 8
    max_seq_len: int = 4_096

    # Distributed
    world_size: int = 4
    fsdp: bool = True
    fsdp_mixed_precision: str = "bfloat16"

    # Gradient handling
    grad_clip: float = 1.0
    loss_nan_skip: bool = True

    # Checkpoint cadence
    save_interval: int = 4_000
    log_interval: int = 50
    eval_interval: int = 2_000

    # Optimizations — the 3 flags (default True)
    fused_gdn: bool = True
    moe_mixed_precision: bool = True
    torch_compile_gdn: bool = True
```

#### Validation in `__post_init__`

| Check | Error |
|---|---|
| `micro_batch_size > 0` | `ValueError` |
| `gradient_accumulation_steps > 0` | `ValueError` |
| `world_size > 0` | `ValueError` |
| `grad_clip > 0` | `ValueError` |
| `save_interval > 0`, `eval_interval > 0` | `ValueError` |
| `fsdp_mixed_precision in {"bfloat16", "float32", "float16"}` | `ValueError` |

#### Property: `per_step_tokens` (line 290)

```python
@property
def per_step_tokens(self) -> int:
    return (
        self.micro_batch_size        # 4
        * self.gradient_accumulation_steps  # 8
        * self.world_size            # 4
        * self.max_seq_len           # 4_096
    )                                # = 524_288
```

This is the per-optimizer-step token count across all ranks. The 30 B
training run is `30e9 / 524_288 ≈ 57,220` optimizer steps — which is
where `SchedulerConfig.total_steps = 57_220` comes from.

#### The three optimization flags

| Flag | Threads to | Default | See |
|---|---|---|---|
| `fused_gdn` | `GatedDeltaNetBlock.use_triton` | True | `optimization.md` §GDN kernel |
| `moe_mixed_precision` | `DeepSeekMoE.use_mixed_precision` | True | `optimization.md` §MoE |
| `torch_compile_gdn` | `GatedDeltaNetBlock.use_compile` | True | `optimization.md` §torch.compile |

All three are wired in `Trainer._thread_optimization_flags`
(`src/hymo/training/trainer.py`) at construction time. (The
`cuda_graphs_mla` flag and its `MLABlock.use_cuda_graphs` attr were removed
in the 2026-08-04 cleanup — no CUDA-graph capture path ever shipped.)

### 2.5 `RunConfig`

```python
@dataclass(frozen=True)
class RunConfig:
    name: str = "hymo-v1.0"
    output_dir: str = "checkpoints/pretrain"
```

#### Validation in `__post_init__`

| Check | Error |
|---|---|
| `name` non-empty | `ValueError` |

The earlier `seed` / `log_dir` / `eval_dir` / `distributed` /
`deterministic` / `resume_from` fields were removed in the 2026-08-04
cleanup — no production code read them (the trainer hardcodes
`torch.manual_seed(0)` in the smoke driver, writes to `run.output_dir`,
and never seeded deterministically).

### 2.6 `HyMoConfig` (line 322)

```python
@dataclass(frozen=True)
class HyMoConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    run: RunConfig = field(default_factory=RunConfig)
```

A `default_factory=…` is used (not a mutable default) so the frozen
instances are fresh per call.

---

## 3. Semantic newtypes — `core/types.py`

```python
TokenId = NewType("TokenId", int)       # token IDs (0..vocab_size-1)
LayerIndex = NewType("LayerIndex", int) # 0..31 for a 32-layer stack
ExpertIndex = NewType("ExpertIndex", int)  # 0..n_routed-1
MicroStep = NewType("MicroStep", int)   # 0..grad_accum-1 (per optimizer step)
Step = NewType("Step", int)             # global optimizer step

DType = torch.dtype          # alias
Device = torch.device | str  # alias
Shape = tuple[int, ...]      # alias
```

These are zero-cost at runtime — `NewType` is a function that returns
its argument unchanged — but they make the type checker catch
`layer_idx: int` vs `expert_idx: int` swaps. `MicroStep` and `Step` are
distinguished so a function expecting the global step cannot silently
receive a micro-step counter.

`Path` is re-exported from `pathlib` so a `Path | str` union is
unambiguous.

---

## 4. Load / save / derive

### 4.1 `load_config(path)` (line 348)

```python
def load_config(path: str | Path) -> HyMoConfig:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"Top-level YAML must be a mapping, got {type(raw).__name__}")
    return _build_config(raw)
```

File-not-found and YAML-parse errors are wrapped with the path in the
message, so a bad config surfaces immediately with the right context.

### 4.2 `_build_config(raw)` (line 373)

Builds each sub-config from the raw dict:

```python
def _build_config(raw: dict[str, Any]) -> HyMoConfig:
    model = ModelConfig(**_filter(raw.get("model", {}), ModelConfig))
    optimizer = OptimizerConfig(**_filter(raw.get("optimizer", {}), OptimizerConfig))
    scheduler = SchedulerConfig(**_filter(raw.get("scheduler", {}), SchedulerConfig))
    training = TrainingConfig(**_filter(raw.get("training", {}), TrainingConfig))
    run = RunConfig(**_filter(raw.get("run", {}), RunConfig))
    return HyMoConfig(model=model, optimizer=optimizer, scheduler=scheduler,
                      training=training, run=run)
```

A `TypeError` from a sub-config is wrapped as
`ValueError("Unknown / wrong-type config field: ...")`.

### 4.3 `_filter(raw, cls)` (line 403)

**Two things happen here**:

1. **Field-name filtering** — keys not in `cls.__dataclass_fields__` are
   silently dropped. This means a stale or typo'd field in the YAML
   doesn't fail; it just gets ignored. (Trade-off: typos are quiet,
   but old configs keep loading as fields evolve.)
2. **Tuple coercion** — if a field is declared as `tuple[...]`, the
   value is wrapped via `tuple(v)` so a YAML list `[0.3, 0.1]` becomes
   `(0.3, 0.1)`. This is what makes `mtp_loss_weights: [0.3, 0.1]`
   work in YAML.

```python
def _filter(raw: dict[str, Any], cls: type) -> dict[str, Any]:
    valid = {f.name for f in fields(cls)}
    out: dict[str, Any] = {}
    for k, v in raw.items():
        if k not in valid:
            continue
        f = next(f for f in fields(cls) if f.name == k)
        if f.type is tuple or (
            hasattr(f.type, "__origin__") and f.type.__origin__ is tuple
        ):
            v = tuple(v) if isinstance(v, (list, tuple)) else (v,)
        out[k] = v
    return out
```

### 4.4 `save_config(config, path)` (line 419)

Atomic YAML dump:

```python
def save_config(config: HyMoConfig, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(_to_dict(config), f, sort_keys=False, default_flow_style=False)
```

`sort_keys=False` keeps the YAML in declaration order so the saved
file is human-diffable against the original.

`_to_dict(obj)` (line 337) is a recursive dataclass → plain-dict
converter, with the same `tuple` coercion as the inverse `_filter`.

### 4.5 `derive_config(base, ...)` (line 427)

Higher-level wrapper over `dataclasses.replace`:

```python
def derive_config(base, *, model=None, optimizer=None, scheduler=None,
                  training=None, run=None) -> HyMoConfig:
    return replace(
        base,
        model=model if model is not None else base.model,
        # ... etc
    )
```

This is the function the eval/ablation framework was going to use; the
in-repo `ablations/` package was removed in the 2026-08-04 cleanup (see
`../training.md` §2).

---

## 5. Cross-field validation — `core/config_validation.py`

`__post_init__` checks fields **within a sub-config**. Cross-field
checks (one config vs another) live in `validate_full_config`:

```python
def validate_full_config(config: HyMoConfig) -> None:
    _validate_total_steps_consistency(config)
    _validate_layer_distribution(config.model)
    _validate_partial_rope_math(config.model)
    _validate_vram_budget(config.model, config.training)
```

| Helper | Check |
|---|---|
| `_validate_total_steps_consistency` (line 24) | `scheduler.total_steps * training.per_step_tokens ≈ intended tokens` (within a tolerance) |
| `_validate_layer_distribution` (line 32) | `n_layers % 4 == 0` (so 3:1 splits evenly) |
| `_validate_partial_rope_math` (line 51) | `qk_rope_head_dim + qk_nope_head_dim == head_dim` (also caught in `__post_init__`, but checked again here for the 25% rule) |
| `_validate_vram_budget` (line 59) | `model size × training.fp16 bytes ≤ approx 4× A100 80 GB` (rough heuristic; raises on gross violations) |

Call it after `load_config`:

```python
from hymo.core.config import load_config
from hymo.core.config_validation import validate_full_config

config = load_config("configs/hymo_750m.yaml")
validate_full_config(config)
```

---

## 6. The full load-and-validate sequence

```python
# 1. Load
config = load_config("configs/hymo_750m.yaml")

# 2. Cross-field validate
validate_full_config(config)

# 3. Build model
from hymo.models import build_hymo
model = build_hymo(config)

# 4. Build trainer (which calls _thread_optimization_flags)
from hymo.training import Trainer
trainer = Trainer(config, model)

# 5. Derive an ablation
from hymo.core.config import derive_config
ablation = derive_config(config, model=derive_config.model)  # actually use replace(ModelConfig, ...)
```

See `model-architecture.md` for step 3, and
`../training.md` for steps 4–5.

---

## 7. Worked example — reading a 750m config

The full default config is 92 lines. The most useful 10 lines to read
first:

```yaml
model:
  n_layers: 32            # 8 MLA + 24 GDN (3:1)
  dim: 896
  n_heads: 16             # 4 query heads per KV group → MQA-4
  n_kv_groups: 4
  qk_rope_head_dim: 32    # 25% of head_dim (128) → partial RoPE
  qk_nope_head_dim: 96
  n_routed_experts: 16    # 16 routed + 1 shared
  n_activated_experts: 2  # top-2 routing
  mtp_depth: 2            # 2 MTP heads
  mtp_loss_weights: [0.3, 0.1]
```

Everything else is either a derived value (e.g. `n_mla_layers = 8`),
a known-good default (e.g. `tie_embeddings: true`, `logit_softcap: 15.0`),
or an optimization knob (the 4 flags).

---

## 8. Interview Q&A

**Q1. Why are config classes `@dataclass(frozen=True)` and not regular
classes?**

> A: Two reasons. First, frozen instances are hashable, so they can be
> used as dict keys / in sets (useful for caching and identifying runs
> in W&B). Second, and more important: a training loop that mutates
> `config.scheduler.total_steps = …` mid-run should be a hard error,
> not a silent bug. `frozen=True` makes accidental mutation impossible
> — you must use `dataclasses.replace` to derive a new config.

**Q2. Why is `core` PyTorch-free?**

> A: It keeps the config system importable from a CPU-only CI runner, a
> config-lint script, or a notebook, without paying the ~1-second
> `import torch` cost. The dependency rule (`core ← utils ← …`) is
> enforced by `AGENTS.md`; a `from torch import …` in `core/` is a
> hard don't.

**Q3. Why does `qk_rope + qk_nope == head_dim` get checked twice
(`__post_init__` and `validate_full_config`)?**

> A: Defense in depth. `__post_init__` runs when any sub-config is
> constructed (e.g. in a test that builds a `ModelConfig()` directly).
> `validate_full_config` runs after `load_config` and adds the
> *invariant* check (the 25% ratio). The second check is a bit
> redundant for the `__post_init__` case, but it ensures the
> validation is part of the load sequence regardless of how the config
> was built.

**Q4. The plan called for `vocab_size = 64_256`. Why 64 k + 256?**

> A: 64 K is a standard BPE vocabulary for English + code; the
> additional 256 are byte-level fallback tokens. Any input can be
> losslessly encoded as a sequence of bytes, so the model can never
> produce an `OOV` (out-of-vocabulary) token. The combined 64,256 is
> what the tokenizer's `ExtendedTokenizer` emits (see
> `../training.md` §Tokenizer).

**Q5. Where does the `57,220` step count come from?**

> A: It's derived from the 30 B-token training budget divided by
> `per_step_tokens = 524,288`. `30e9 / 524_288 ≈ 57,220`. The
> relationship is asserted by `_validate_total_steps_consistency`.

**Q6. How do you add a new config field safely?**

> A: Add the field to the dataclass with a default value (so old YAML
> files still load). Add a `__post_init__` check if it has invariants.
> Add a unit test that builds the config with the new field and one
> without. The `key in valid` filter in `_filter` means a missing
> field just gets its default; an extra field gets silently dropped.

**Q7. Why `total_steps: Step = Step(57_220)` instead of `int`?**

> A: `Step` is a `NewType` over `int`, so it's zero-cost at runtime but
> the type checker knows the difference between a `Step` (global
> optimizer step) and a `MicroStep` (per-accumulation counter). In
> practice, this catches bugs like `scheduler.get_factor(micro_step)`
> (which should be `step + 1`) at static-analysis time.

---

## References

- [api.md](api.md) — the model + trainer API surface.
- [../concepts/model-architecture.md](../concepts/model-architecture.md) — the model walkthrough.
- [../concepts/optimization.md](../concepts/optimization.md) — optimizer/scheduler/FSDP mechanics and init status.
- [../training.md](../training.md) — the training pipeline.
- [../guides/quickstart.md](../guides/quickstart.md) — the 30-second start.
- Source: `src/hymo/core/config.py` (`ModelConfig`, `OptimizerConfig`, `SchedulerConfig`, `TrainingConfig`, `RunConfig`, `HyMoConfig`, `load_config`, `save_config`, `derive_config`), `src/hymo/core/config_validation.py` (`validate_full_config`), `src/hymo/core/types.py`.
