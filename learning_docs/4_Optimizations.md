# HyMo Optimizations — Code Walkthrough

> **Prerequisite reading:** [`learning_docs/1_Model_Architecture.md`](1_Model_Architecture.md) §5–§6
> for the GDN block and the forward trace,
> [`learning_docs/3_Training_Pipeline.md`](3_Training_Pipeline.md) for `Trainer`,
> [`docs/concepts/10-triton-kernels.md`](../docs/concepts/10-triton-kernels.md) for the GPU execution
> model.
>
> **Files covered:**
> - `src/hymo/models/gdn_triton.py` — hand-written Triton kernel for the GDN recurrence
> - `src/hymo/models/gdn.py` — eager GDN reference + `use_triton` / `use_compile` toggles
> - `src/hymo/models/moe.py` — aux-loss-free EMA gate-bias + mixed-precision dispatch
> - `src/hymo/models/mla.py` — CUDA-Graph capture path
> - `src/hymo/core/config.py` — `TrainingConfig`'s four optimization flags
> - `src/hymo/training/trainer.py` — `_thread_optimization_flags` (the wire-up)
>
> **Important:** the design doc discussed depending on the `fla` library
> (`fla.layers.gated_delta_net.chunk_gated_delta_rule`); the shipped
> implementation does **not** depend on `fla`. The only sanctioned
> custom kernel is the hand-written Triton kernel in
> `gdn_triton.py`. See the "Design vs. implementation" note at the top
> of [`../docs/HyMo-Design.md`](../docs/HyMo-Design.md).

---

## Table of Contents

1. [Optimization overview](#1-optimization-overview)
2. [The four optimization flags](#2-the-four-optimization-flags)
3. [Triton GDN kernel](#3-triton-gdn-kernel)
4. [`torch.compile` on GDN blocks](#4-torchcompile-on-gdn-blocks)
5. [MoE mixed precision and EMA gate-bias](#5-moe-mixed-precision-and-ema-gate-bias)
6. [CUDA Graphs on MLA](#6-cuda-graphs-on-mla)
7. [FSDP-2 + BF16 mixed precision (overview)](#7-fsdp-2--bf16-mixed-precision-overview)
8. [Gradient accumulation and NaN-skip](#8-gradient-accumulation-and-nan-skip)
9. [Memory budget analysis](#9-memory-budget-analysis)
10. [Interview Q&A](#10-interview-qa)

---

## 1. Optimization overview

The production target is sustained throughput on 4× A100 80 GB SXM,
with `per_step_tokens = 524,288` (see
[`learning_docs/6_Config_System.md`](6_Config_System.md) §2.4). The four
in-scope optimizations are toggled by flags in `TrainingConfig`; the
fifth, **FSDP-2**, is what makes the 1.86 B-param model fit across
4 GPUs at all.

| # | Optimization | Flag (TrainingConfig) | Where it lives | Speedup vs eager |
|---|---|---|---|---|
| 1 | Triton GDN kernel | `fused_gdn` | `gdn.py` + `gdn_triton.py` | 3–5× over the Python double-loop |
| 2 | `torch.compile` on GDN | `torch_compile_gdn` | `gdn.py::forward` | ~1.2× via small-op fusion |
| 3 | MoE mixed precision (BF16 dispatch) | `moe_mixed_precision` | `moe.py::forward` | ~1.3× via halved HBM bandwidth |
| 4 | CUDA Graphs on MLA | `cuda_graphs_mla` | `mla.py::forward` | ~1.1× via kernel-launch elimination |

The fifth optimization — **FSDP-2 full sharding** — is not optional at
production scale. It's gated by `training.fsdp = True` (the default).
See [`concepts/09-fsdp2.md`](../docs/concepts/09-fsdp2.md) for the
mechanics and `src/hymo/training/fsdp.py::wrap_model_with_fsdp` for the
wiring.

### Why these four?

The 3:1 GDN:MLA stack means ~75% of forward time is spent in GDN
blocks. A 3–5× speedup there dominates. MoE mixed precision matters
less per-call but fires on every MLA layer (8× per forward). `torch
.compile` on GDN captures cross-block fusion that
`inductor` can do automatically once the per-block compute graph is
simple. CUDA Graphs on MLA captures the kernel-launch overhead out of
the 8 MLA forward passes (108 SMs × 8 blocks × many kernels is real
time at batch size 4).

---

## 2. The four optimization flags

Defined in `src/hymo/core/config.py:265-269` as fields of
`TrainingConfig`:

```python
fused_gdn: bool = True
moe_mixed_precision: bool = True
torch_compile_gdn: bool = True
cuda_graphs_mla: bool = True
```

(All default to `True` because production training depends on them.
For a CPU smoke test or a debugging run, override to `False`.)

### 2.1 Wiring: `Trainer._thread_optimization_flags`

`src/hymo/training/trainer.py:91-112`:

```python
def _thread_optimization_flags(self) -> None:
    """Push training-config optimization toggles onto the model blocks."""
    from hymo.models.gdn import GatedDeltaNetBlock
    from hymo.models.mla import MLABlock
    from hymo.models.moe import DeepSeekMoE

    t = self._config.training
    for module in self.model.modules():
        if isinstance(module, GatedDeltaNetBlock):
            module.use_triton = t.fused_gdn
            module.use_compile = t.torch_compile_gdn
        elif isinstance(module, DeepSeekMoE):
            module.use_mixed_precision = t.moe_mixed_precision
        elif isinstance(module, MLABlock):
            module.use_cuda_graphs = t.cuda_graphs_mla
```

This is called **once**, in `Trainer.__init__` (line 55), before the
forward pass ever runs. Each block has its own `use_*` attribute
that gates the optimized code path; setting the flag from config
is what tells the block "production mode" vs "debug mode".

### 2.2 Setting a flag for a run

```yaml
# configs/hymo_750m.yaml
training:
  fused_gdn: true              # default
  moe_mixed_precision: true    # default
  torch_compile_gdn: true      # default
  cuda_graphs_mla: true        # default
```

To disable any optimization for a debugging run, pass `--config
debug.yaml` (where `debug.yaml` overrides one or more flags to
`false`); the trainer will see the new value on init and re-thread
it. There is no per-step toggling — the attribute is read on each
forward.

---

## 3. Triton GDN kernel

> **This is the headline optimization.** It is the only sanctioned
> custom kernel in the codebase
> ([`AGENTS.md`](../AGENTS.md) §Engineering rules). The previous
> implementation was a Python double-loop over the T axis; the
> current implementation fuses the recurrence into a single
> chunked Triton kernel.

### 3.1 Why not `fla`?

The `fla-org` library ships
`fla.layers.gated_delta_net.chunk_gated_delta_rule` — a reference
implementation of the same algorithm. The design doc suggested
depending on it (or vendoring a copy). The shipped implementation
**does neither**:

- `pyproject.toml:48` has `fla>=0.1` commented out with the note
  `fla-org GDN kernel (unavailable in standard registry)`.
- A repo-wide `grep "fla"` across `*.py` returns zero hits in
  source. (The only matching references are
  `pyproject.toml`'s `[[tool.mypy.overrides]]` `module = [...,
  "fla.*", ...]` ignored-module list and the unrelated
  `SKILLS.md` line "fla (GDN kernel)" which has been removed.)
- `src/hymo/models/gdn_triton.py` is the hand-written kernel; the
  Triton → C++ → PTX pipeline is owned by the project.

### 3.2 The recurrence, in math

The Gated Delta Net state update:

```
h_t = exp(g_t * A) * h_{t-1} + b_t ⊗ v_t
o_t = c_t · h_t
```

where

- `v, o : (B, T, H, D)` — value and output projections per head
- `b, c : (B, T, H, S)` — write and read keys (low-rank state)
- `g : (B, T, H)` — per-head scalar gate (sigmoid of `dt_proj`
  output, *positive*)
- `A_log : (H, S)` — log of decay eigenvalues (learnable,
  typically negative; the kernel exponentiates them)

The naive "unfused" version iterates `t = 0 .. T-1` in Python, doing
two einsums per step. For `T = 4096`, `B = 4`, `H = 40`,
`D = 32`, that's **65,536 Python-level iterations per GDN forward
per micro-batch** — and there are 24 GDN layers per forward. The
Python overhead alone, on a 4-GPU A100 pod, is the gap between
"tolerable" and "infeasible" wall-clock.

The hand-written kernel reduces this to one kernel launch per layer
per micro-batch, by chunking the sequence into `chunk_size = 64`
chunks and processing each chunk with intra-chunk parallelism
(within-SM Triton blocks) and inter-chunk serialism (the recurrence
state `h` carries).

### 3.3 Kernel layout — `gdn_triton.py`

The file is 288 lines and contains:

| Lines | Symbol | Role |
|---|---|---|
| 23-26 | import-time `HAS_TRITON` | computed from `try: import triton` |
| 30-41 | `_next_power_of_2(n)` | helper — pads to power-of-2 for Triton |
| 43-92 | `@triton.jit gdn_fwd_kernel` | forward, one block per `(B*H, chunk)` |
| 94-235 | `@triton.jit gdn_bwd_kernel` | backward (recompute-style, see concepts/10) |
| 237-288 | `triton_gated_delta_rule(v, b, c, g, A_log)` | the Python wrapper |
| 60+ | `class TritonGDNFunction(torch.autograd.Function)` | autograd `Function` wrapping forward + backward (uses `setup_context` + `backward`); checks `HAS_TRITON` in both |

#### The Python wrapper (line 237)

```python
def triton_gated_delta_rule(
    v: torch.Tensor, b: torch.Tensor, c: torch.Tensor,
    g: torch.Tensor, A_log: torch.Tensor,
) -> torch.Tensor:
    if not HAS_TRITON:
        raise ImportError(
            "Triton is required for triton_gated_delta_rule. "
            "Install it with: pip install triton"
        )

    B, T, H, D = v.shape
    S = b.shape[-1]

    D_pad = _next_power_of_2(D)
    S_pad = _next_power_of_2(S)

    def _pad(t: torch.Tensor, target_last: int) -> torch.Tensor:
        if t.shape[-1] == target_last:
            return t
        pad = [0] * (2 * t.ndim)
        pad[1] = target_last - t.shape[-1]
        return torch.nn.functional.pad(t, pad)

    v_p     = _pad(v.float().contiguous(),     D_pad)
    b_p     = _pad(b.float().contiguous(),     S_pad)
    c_p     = _pad(c.float().contiguous(),     S_pad)
    A_log_p = _pad(A_log.float().contiguous(), S_pad)

    g_p = g.float().contiguous()
    if g_p.ndim == 4:
        g_p = g_p.squeeze(-1)
    assert g_p.shape == (B, T, H), (
        f"triton_gated_delta_rule: g must be (B,T,H), got {g_p.shape}"
    )

    out_p = TritonGDNFunction.apply(v_p, b_p, c_p, g_p, A_log_p)

    out = out_p[..., :D]
    return out.to(v.dtype)
```

Two non-obvious things:

1. **Power-of-2 padding** — Triton's autotuner and block layouts
   assume powers of 2 along the inner dims. The wrapper pads `D` (32
   in production) and `S` (32 in production) to their next
   power-of-2 (already 32 in both cases, so usually a no-op), and
   slices the padding back off before returning (`out[..., :D]`).

2. **FP32 inside, cast back outside** — the kernel runs all math
   in `float()` (FP32), then `.to(v.dtype)` casts back to the
   caller's dtype (typically BF16 in production, FP32 in tests).
   This is "FP32 accumulator + caller-dtype storage" — what
   `torch.cuda.amp` would do for a matmul; we just do it manually.

### 3.4 The autograd `Function`

`TritonGDNFunction` extends `torch.autograd.Function` with
`forward` calling `gdn_fwd_kernel`, saving
`v_p, b_p, c_p, g_p, A_log_p, out_p` for backward, and
`backward` calling `gdn_bwd_kernel` with the saved tensors. The
backward kernel is **recompute-style** (not stored-activation
style) — the activation `h` is not kept across `forward`/`backward`,
but the inputs (`v, b, c, g, A_log`) are enough to recompute
locally within each chunk, halving the activation memory cost.

`tests/unit/test_triton_gdn_gpu.py` is the test surface:

- `test_triton_importable` and `test_pad_helper` are CPU-safe.
- `test_forward_parity_atol`,
  `test_forward_non_power_of_2_pads_correctly`, and
  `test_backward_grads_match_pytorch` are `@pytest.mark.skipif
  (not HAS_TRITON)` and additionally need a CUDA GPU.
- The tests gate **forward** and **backward** numerical parity
  vs. the eager PyTorch reference (the one in `gdn.py::forward`).

### 3.5 Wiring from `gdn.py::GatedDeltaNetBlock.forward`

`gdn.py:34` initializes:

```python
self.use_triton = True   # default — threaded from TrainingConfig.fused_gdn
self.use_compile = True  # default — threaded from TrainingConfig.torch_compile_gdn
```

The `forward` then dispatches on `self.use_triton`:

```python
if self.use_triton:
    o_inner = triton_gated_delta_rule(v, b, c, g, A_log)
else:
    o_inner = self._eager_recurrence(v, b, c, g, A_log)
```

There is **no silent fallback**: if `use_triton=True` and Triton
isn't installed, `triton_gated_delta_rule` raises `ImportError`.
This is mandated by [`AGENTS.md`](../AGENTS.md) §Hard don'ts:

> Don't let a Triton kernel silently fall back to raw PyTorch during
> a default-config training run. Opt-in is explicit; failures must
> surface a clear error.

### 3.6 Throughput

The conservative estimate (and what the design doc assumed for the
30 B run's wall-clock) is:

- Eager (`_eager_recurrence`): ~4.5 ms per GDN layer forward at the
  production shape — dominated by Python overhead.
- Triton (`triton_gated_delta_rule`): ~1 ms per GDN layer forward
  at the same shape — about 4.5× faster.
- Across 24 GDN layers per forward and forward+backward accounted
  for: ~3.5 ms × 24 × 2 = ~168 ms savings per forward+backward,
  or ~25% of the per-step wall-clock.

These are the conservative numbers; real numbers depend on A100 vs
H100, driver, and torch.compile interaction (the next optimization).

---

## 4. `torch.compile` on GDN blocks

`TrainingConfig.torch_compile_gdn: bool = True` threads through to
`GatedDeltaNetBlock.use_compile = True` and decorates the block's
`forward` with `@torch.compile(mode="reduce-overhead", dynamic=False)`.

What `torch.compile` does to a GDN block:

- **Fuses the pointwise ops** around the Triton kernel call —
  input projections, output projection, gating, residual addition
  — into a single compiled graph per block.
- **CUDA Graph capture** (`mode="reduce-overhead"`) — the first
  iteration captures a CUDA graph for the block's forward; subsequent
  iterations replay the graph, eliminating per-op kernel launch
  overhead.

Caveats:

- `dynamic=False` is set because the GDN block's shapes are static
  (batch, seq, dim, head_dim) at production; with `dynamic=True`,
  recompilation would defeat the purpose.
- The flag is named "torch_compile_gdn" because the GDN block is
  the largest non-attention compute; MLA blocks already benefit
  from inductor's defaults via the regular FSDP wrapping path.
- `torch.compile` and the Triton kernel interact well because
  inductor sees `triton_gated_delta_rule` as a single leaf node —
  it doesn't try to recompile what it can't see.

### 4.1 The interaction with `fused_gdn`

Both can be on. Order of dispatch in `gdn.py::forward`:

1. `torch.compile` wraps the entire `forward`, including the
   Triton call. So even with `fused_gdn=True`,
   `torch_compile_gdn=True` adds value (fused surrounding pointwise
   ops, plus CUDA-graph replay).
2. With `fused_gdn=False, torch_compile_gdn=True`, the eager
   recurrence runs inside a compiled graph. This is the
   "compile but don't use the kernel" debug mode — useful for
   isolating whether a numerical divergence is in the kernel
   or in the surrounding graph.

---

## 5. MoE mixed precision and EMA gate-bias

Two co-located optimizations in `src/hymo/models/moe.py::DeepSeekMoE.forward`:

### 5.1 `moe_mixed_precision` — the BF16 expert dispatch

```python
# mixed-precision dispatch (design §12a.2): cast the expert input to
# the expert weight dtype so matmuls run in that precision.
self.use_mixed_precision = True   # default (line 77); threaded from TrainingConfig.moe_mixed_precision

x_experts = (
    x_flat.to(self.experts[0].w1.weight.dtype)
    if self.use_mixed_precision
    else x_flat
)
```

Mechanics:

- Under FSDP-BF16, the expert weights (`expert.w1.weight`,
  `expert.w2.weight`, `expert.w3.weight`) are stored in BF16.
- The token activations come in as `x_flat` in the same dtype as
  the residual stream (typically BF16 in production).
- Without the cast: matmul runs in BF16; saves ~50% of activation
  bandwidth vs FP32 dispatch.
- With `use_mixed_precision=False`: the cast is skipped, and on
  CPU (where weights are FP32) we preserve the test-as-written
  dtype semantics.

The dispatch is a one-line `if`, but it fires 16 times per MLA
forward (once per routed expert).

### 5.2 The capacity cap and the FP32 router

Even before the cast, the MoE forward has two other correctness
features worth knowing:

**`capacity_factor` (line 62, 133-134)**

```python
capacity = int(self.capacity_factor * (B * T * k) / self.n_routed)
capacity = max(capacity, 1)
```

With `B*T*k = 4·4096·2 = 32,768` tokens to dispatch across
`n_routed = 16` experts, the average is 2048 tokens per expert.
With `capacity_factor = 1.5` (the default), the cap is 3072 tokens
per expert — so no expert gets more than 1.5× the mean. Over-capacity
indices are dropped (the `sel = sel[:capacity]` line at 154).

**`gate_forward` in FP32 (line 95)**

```python
def gate_forward(self, x: torch.Tensor) -> torch.Tensor:
    x_fp32 = x.float()
    w_fp32 = self.gate.weight.float()
    b_fp32 = self.gate.bias.float()
    logits = F.linear(x_fp32, w_fp32, b_fp32)
    return logits.to(x.dtype)
```

The gate (16-way softmax over hidden states) is computed in FP32
to avoid sigmoid/softmax underflow under BF16 (small gate logits
silently round to zero in BF16, collapsing routing to uniform).
The cost is one FP32 matmul of shape `(B*T, dim) × (dim,
n_routed)`, which is tiny relative to the expert matmuls.

### 5.3 EMA gate-bias — aux-loss-free load balancing

This is the **conceptual** counterpart to the **tactical**
mixed-precision flag — both live in MoE, but EMA gate-bias is not
a TrainingConfig flag; it's a `ModelConfig.moe_ema_alpha` knob.

The recurrence, once per optimizer step (called from
`Trainer._update_moe_gate_biases`, `trainer.py:329-335`):

```python
def update_gate_bias(self, speed: float = 0.001) -> None:
    if getattr(self, "_last_indices", None) is None:
        return
    counts = torch.bincount(
        self._last_indices.flatten(), minlength=self.n_routed
    ).float()
    ema = cast(torch.Tensor, self.ema_expert_counts)
    ema.mul_(1.0 - self.ema_alpha).add_(counts, alpha=self.ema_alpha)
    avg = ema.mean()
    over = ema > avg * 1.05
    under = ema < avg * 0.95
    with torch.no_grad():
        new_bias = self.gate.bias.clone()
        new_bias[over] -= speed
        new_bias[under] += speed
        self.gate.bias.copy_(new_bias)
```

`moe.py:103-119`. See [`docs/concepts/03-mixture-of-experts.md`](../docs/concepts/03-mixture-of-experts.md) for the
derivation. Quick intuition:

- The gate bias is **dynamically adjusted** to penalize experts
  that have been getting more than the average load and reward
  those getting less.
- The `over`/`under` threshold is `± 5%` of the running mean
  (`avg * 1.05`, `avg * 0.95`).
- The step size `speed = 0.001` is small enough to be stable over
  57 k steps but large enough to re-balance within ~1 k steps.
- `ema_alpha = 0.02` (the default `ModelConfig.moe_ema_alpha`)
  weights the most recent batch counts at 2% — averaging over
  ~50 batches.

This replaces a learned auxiliary load-balancing loss (the
"DeepSeekMoE aux_loss" in the original DeepSeek paper). The EMA
approach is **strictly better** for this scale: no extra gradient
signal, no additional backward, no hyperparameter collision with
the main loss.

---

## 6. CUDA Graphs on MLA

`TrainingConfig.cuda_graphs_mla: bool = True` threads through to
`MLABlock.use_cuda_graphs = True`. In `mla.py::forward`,
the MLA block:

1. Runs its standard forward (low-rank KV compression → split
   RoPE / NoPE heads → attention → output projection).
2. On the first call with a given shape signature, **captures** a
   CUDA graph via `torch.cuda.CUDAGraph()`.
3. On subsequent calls with the same signature, **replays** the
   graph instead of re-running the kernel launches.

The capture happens per-rank on each device; shape changes
(typically only on validation, where batch may be different from
training) trigger recapture. CUDA Graphs require:

- All tensors allocated via the CUDA caching allocator (default
  PyTorch behavior).
- No CPU–GPU sync inside the captured region (e.g. no
  `.item()` calls).
- Constant input shapes per capture.

The MLA block satisfies all three by design — no `.item()` calls,
fixed-shape `head_dim` and `n_heads`, and FSDP-2's parameter
sharding is forward-pass-agnostic.

### 6.1 Why MLA and not GDN?

GDN already gets ~4.5× speedup from the Triton kernel, which is the
bottleneck; the surrounding pointwise ops don't need CUDA-Graph
capture for MLA-style launch-overhead elimination. MLA has more
small ops per block (low-rank split, RoPE/Nope split, attention
softmax, output projection), so it benefits more from graph
capture.

---

## 7. FSDP-2 + BF16 mixed precision (overview)

Full coverage is in [`docs/concepts/09-fsdp2.md`](../docs/concepts/09-fsdp2.md) and
`src/hymo/training/fsdp.py`. The optimization-side notes:

- `training.fsdp = True` is the production default. Setting
  `fsdp = False` skips the `wrap_model_with_fsdp` call entirely
  and is reserved for the test suite (where we run on a single
  rank with the tiny config).
- `training.fsdp_mixed_precision = "bfloat16"` is the only
  production setting; `"float32"` is for ablation runs that
  compare precision sensitivity; `"float16"` is not supported
  (raises in `TrainingConfig.__post_init__`).

The FSDP wrapper keeps BF16 parameters and gradients on rank,
maintains FP32 master weights (per `optimizer.master_weights_dtype`),
and uses ZeRO-3-style all-gather for forward + reduce-scatter for
backward. The 4× A100 80 GB ranks each hold ~465 MB of parameters +
~465 MB of gradients + ~700 MB of optimizer state at peak.

---

## 8. Gradient accumulation and NaN-skip

In `src/hymo/training/trainer.py::Trainer.train_step` (line 114):

1. Forward pass (with MTP if enabled — routed through
   `_mtp.forward(tokens)` which calls `model.forward_with_hidden`).
2. Compute `main_loss` + MTP losses (depth=2 by default,
   weights `[0.3, 0.1]`).
3. If `loss_nan_skip` is true and the total loss is `nan`/`inf`:
   zero grad, return early with `is_update=False, skipped=True`.
4. `scaled_loss = total_loss / gradient_accumulation_steps`;
   `scaled_loss.backward()`.
5. `is_update = (micro_step % gradient_accumulation_steps == 0)`.
   If `is_update`:
   - `clip_grad_norm_(...)` with `grad_clip = 1.0`.
   - Update LRs from the scheduler.
   - Step the optimizers (NorMuon + AdamW).
   - Call `_update_moe_gate_biases()` to fire the EMA update.
   - Step the scheduler; zero grad; `step += 1`.

The "NaN-skip" behavior is the **only** place a non-finite loss is
tolerated. Default `consecutive_nan_limit = 5` is for higher-level
recovery (e.g. aborting after 5 consecutive NaN-steps) but is not
checked in `train_step` itself — it's checked by the calling
`Trainer.train` if a kill-switch is wired.

### 8.1 Why `is_update` lives on the `train_step` return

`train_step_result` (line 33) returns:

```python
@dataclass
class train_step_result:
    loss: float
    grad_norm: float
    lr_muon: float
    lr_adamw: float
    is_update: bool = True
    skipped: bool = False
    metrics: dict[str, float] = field(default_factory=dict)
```

The reason `is_update` is on the result, not just a side-effect:
`Trainer.train` (line 252) uses it to gate W&B logging and the
"every N steps eval / save" cadence. An accumulated micro-step's
loss is still logged (`result.metrics["main_loss"]`), but only an
update step counts toward the eval/save interval.

---

## 9. Memory budget analysis

Per-rank memory at the production shape (`B=4`, `T=4096`,
`dim=896`, `n_layers=32`, `world_size=4`):

| Component | Size |
|---|---|
| BF16 parameters (sharded) | ~465 MB |
| BF16 gradients (sharded) | ~465 MB |
| FP32 master weights (sharded) | ~700 MB |
| FP32 AdamW state (m, v) | ~1.4 GB |
| All-gather activations at peak | ~3.5 GB |
| Activations (computed once) | ~1.5 GB |
| **Peak total** | **~7.8 GB / rank** |

A100 80 GB has **> 10× headroom**. The dominant variable at this
scale is the **optimizer state** (AdamW's m, v vectors per
parameter). With FSDP-2, this scales linearly with `1 / world_size`
— doubling to 8 GPUs cuts the per-rank optimizer state in half.

Activation memory grows linearly with `B * T * dim`; bumping
`micro_batch_size` from 4 to 8 would push activations past 3 GB
without FSDP all-gather savings, but is still well within budget.

---

## 10. Interview Q&A

**Q1. Why is the Triton kernel in HyMo handwritten, not from `fla`?**

> A: Two reasons. First, dependency minimization — `fla` was
> unavailable in the project's standard registry at the time
> (`pyproject.toml:48` has it commented out). Vendoring would
> have meant carrying ~600 lines of third-party Triton code with
> our own integration path. Second, control: the hand-written
> kernel in `gdn_triton.py` is small (288 lines, ~80 of which
> are the actual `@triton.jit` body) and exactly matches HyMo's
> recurrence signature (`exp(g * A) * h + b ⊗ v`, `o = c · h`)
> with no general-purpose infrastructure we don't use.

**Q2. Why is `use_triton` an attribute on the block rather than a
global env var?**

> A: Because `_thread_optimization_flags` (`trainer.py:91`)
> threads the flag from `TrainingConfig` per-block at construction
> time. A global env var would need to be set before model
> construction and would require restart to flip; a config flag
> survives a config reload (e.g. for an ablation rerun).

**Q3. Why is there no silent Triton → eager fallback?**

> A: Because that hides regressions. If a config calls for the
> kernel, the kernel must run; if it can't, we want a loud
> `ImportError` so the operator either installs Triton or sets
> the flag off explicitly. The latter is a deliberate choice; an
> accidental fall-through would mask numerical divergence.

**Q4. What does `torch.compile(mode="reduce-overhead")` do for the
GDN block?**

> A: It captures a CUDA graph of the block's forward after the
> first warm-up call, then replays the graph on subsequent calls
> instead of re-running each kernel launch. Combined with the
> Triton kernel (`triton_gated_delta_rule`), the entire
> forward is roughly 3–5 small CUDA graph replays per layer
> instead of ~30+ kernel launches.

**Q5. Why aux-loss-free MoE instead of an auxiliary loss term?**

> A: An aux loss term has to compete with the main LM loss for
> gradient signal — there's a hyperparameter λ controlling the
> trade-off, and the right value depends on scale. The EMA
> gate-bias approach is gradient-free (it's a bias update, not a
> parameter update) and doesn't interact with the main loss's
> optimizer. Tuning reduces to one number: `speed = 0.001`.

**Q6. Why FP32 in the gate even when activations are BF16?**

> A: Softmax underflows in BF16 for small logits — a 16-bit
> float cannot represent probabilities below ~6e-8, so a
> near-uniform gate becomes exactly uniform after BF16 rounding.
> Doing the gate matmul + softmax in FP32 (then casting the
> output back) costs one FP32 matmul of shape `(B*T,
> dim) × (dim, n_routed)` — negligible relative to the expert
> matmuls.

**Q7. Why does the EMA gate-bias update fire only on optimizer
steps, not every micro-batch?**

> A: The EMA is meant to average load statistics over many
> batches; updating on every micro-batch would make the bias
> noisy. Tying the update to the optimizer step (i.e. the
> accumulated micro-batch) means the EMA sees the *averaged*
> dispatch distribution per step, which is exactly what we want.

---

## 11. Cross-links

- Walkthrough: `learning_docs/1_Model_Architecture.md` §5–6 (GDN block,
  Triton kernel call site), §8 (MoE), §4 (MLA), §3 (model top-level).
- Concepts: `docs/concepts/10-triton-kernels.md` (Triton + autograd
  patterns), `docs/concepts/03-mixture-of-experts.md` (EMA gate-bias
  derivation), `docs/concepts/09-fsdp2.md` (FSDP-2 mechanics).
- Tests: `tests/unit/test_triton_gdn_gpu.py` (kernel parity),
  `tests/unit/test_training.py::test_train_step_consumes_optimization_flags`
  (flag wiring).
