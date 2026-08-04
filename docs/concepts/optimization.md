# HyMo — Optimization

> Consolidates the optimization stack: the Muon-family optimizers (NorMuon + CautiousAdamW),
> the WSD learning-rate schedule, FSDP-2 parameter sharding, initialization status, and the
> training-time optimization flags (Triton GDN kernel, MoE mixed precision, `torch.compile`).
> The Triton kernel deep-dive lives in [kernels.md](kernels.md).


## Muon Optimizer: Newton–Schulz and Cautious WD

### Learning objectives

After this file, you can:

1. State the Muon optimizer and explain why it outperforms
   AdamW on dense 2D weight matrices.
2. Walk through Newton–Schulz iteration and derive the
   orthogonalization step.
3. Explain cautious weight decay (Liang et al. 2024) and why it
   matters at scale.
4. Defend HyMo's `NorMuon` (Muon + cautious WD + FP32 masters)
   variant.

### Intuition

AdamW treats every parameter as a vector of independent
entries; the update is `m_hat / (sqrt(v_hat) + eps)`. This is
robust (any shape, any scale) but ignores the **matrix
structure** of dense 2D weights — for an attention
projection `(in_dim, out_dim)`, the gradient is a matrix,
and the matrix's spectrum (singular values) carries signal
that AdamW throws away.

**Muon** (Jordan et al. 2024, "Muon: An optimizer for hidden
layers in neural networks") replaces the AdamW update with the
**matrix sign function** of the gradient:

```
Muon update ≈ sign(grad) · ||grad||_F
```

where `sign(grad)` is the orthogonal matrix whose singular
values are all 1. This:
- Maintains the gradient's *spectrum* (every direction in the
  matrix is treated equally).
- Preserves the gradient's *magnitude* (the `||grad||_F`
  factor).
- Is scale-invariant at the per-layer level (the update
  magnitude doesn't depend on the *shape* of the matrix).

In practice, Muon converges faster than AdamW on dense 2D
matrices — typically 30-50% fewer steps to the same loss.

**Cautious WD** (Liang et al. 2024,
"Cautious Optimizers: Improving Training with One Line of
Code") modifies the weight-decay step to mask out the
decay term unless the gradient agrees with the parameter
sign:

```python
mask = (grad * param > 0)           # 1 where they agree
update = grad + wd * param * mask
```

The line `update = grad + wd * param * mask` is the entire
contribution. It says: "only apply weight decay where the
gradient is *pushing* the parameter in the same direction
the decay would pull it." This eliminates the "WD fights the
gradient" pathology where decoupled WD pulls parameters toward
0 even when the gradient is trying to grow them.

**NorMuon** is HyMo's name for the combination: `Muon` +
`c cautious WD` + `FP32 master weights`.

### Math derivation

### The matrix sign function

For a matrix `G` with SVD `G = U Σ Vᵀ`, the matrix sign is
`sign(G) = U · I · Vᵀ` (replace singular values with 1, keep
the singular vectors). This is the "closest orthogonal matrix
to G" in the spectral norm.

In closed form, Newton's method on `f(X) = X²` initialized
at `G / ||G||₂` converges to it:

```
X_{k+1} = 1.5·X_k − 0.5·X_k³     (Newton update for f(X) = X²)
```

After 5 iterations starting from `G / ||G||_F`, `X_5` is
within ~1e-3 of `sign(G)`. The final multiply by `||G||_F`
restores the original magnitude.

Cost: 5 matmuls of `(out_dim, out_dim)` per layer per step. For
HyMo's attention projections (`out_dim ~ 2048`), that's 5 ×
2048³ ≈ 90 GFLOPs per optimizer step per projection — small
relative to the forward+backward (~5 GFLOPs per token).

### Caiguous WD derivation

Standard decoupled WD:

```
param ← param - lr · (grad + wd · param)
```

The WD term `wd · param` is **always** subtracted. If `grad`
is positive and `param` is positive, the WD term fights the
gradient.

Cautious WD adds a mask:

```
mask = (grad * param > 0) ? 1 : 0
param ← param - lr · (grad + wd · param * mask)
```

Now the WD term only applies where the gradient sign
*agrees* with the parameter sign. The "fight" is eliminated.

### Why Muon for dense 2D matrices

The orthogonalization step has three properties:

1. **Spectrum preservation**: every singular direction is
   treated equally, regardless of its singular value.
   This is good for transformers because the gradient of
   an attention projection tends to have a few large
   singular values (the "signal" directions) and many small
   ones (the "noise" directions); AdamW's `1/sqrt(v)` per-
   entry rescaling dampens the large values more than the
   small ones, which is the opposite of what we want.

2. **Magnitude preservation**: the `||grad||_F` factor
   preserves the gradient's scale. AdamW's update magnitude
   depends on the running `v` (second-moment estimate) which
   can drift up or down across layers.

3. **Shape invariance**: the step size is roughly independent
   of the matrix shape. A 2048×2048 layer and a 256×256
   layer get the same effective step size.

### Implementation in HyMo

- `src/hymo/training/optimizer.py:_newton_schulz_orthogonalize` — `_newton_schulz_orthogonalize
  (g, iterations=5)`: the 5-step NS iteration.
- `src/hymo/training/optimizer.py:NorMuon` — `class NorMuon(Optimizer)`.
- `src/hymo/training/optimizer.py:NorMuon.__init__` — `__init__` with
  `lr, momentum, betas, eps, weight_decay, cautious_wd,
  ns_iterations`.
- `src/hymo/training/optimizer.py:NorMuon.step` — `step()`: applies
  momentum, cautious mask, Newton–Schulz, decoupled WD.
- `src/hymo/training/optimizer.py:CautiousAdamW` — `class CautiousAdamW`.
- `src/hymo/training/optimizer.py:CautiousAdamW.__init__` — `__init__` with
  `embed_weight_decay` separate from `weight_decay`.
- `src/hymo/training/optimizer.py:CautiousAdamW.step` — `step()`: AdamW
  with cautious mask.
- `src/hymo/training/optimizer.py:Optimizers` — `class Optimizers`:
  the bundle `(nor_muon, adamw)`.
- `src/hymo/training/optimizer.py:build_optimizers` — `build_optimizers(model,
  config)`: the partition + dual build.

### Worked example

Production scale (LR = 0.02, momentum = 0.95, betas = (0.95, 0.95),
eps = 1e-8, weight_decay = 0.1, cautious_wd = True):

- For an MLA `q_a_proj` of shape `(dim=896, q_lora_rank=224)`:
  - Gradient shape `(896, 224)`; `||grad||_F` typically 0.5-5.
  - Newton–Schulz: 5 iterations of `(224, 224)` matmuls.
  - Total per-step: ~5K MAC per iteration × 5 = ~25K MAC for
    orthogonalization.
  - Plus the update: `param -= lr · sign(grad) · ||grad||_F`,
    ~800 × 224 = 180K MAC.

For a 32-layer model with ~12 NorMuon-eligible matrices per
layer, the total per-step NS cost is ~384 × 25K = 9.6M MAC
plus ~2M MAC for the updates. Vs. the forward pass's ~5 GFLOPs
per token (× 4 batch × 4096 seq = 80 GFLOPs), the optimizer
adds < 0.01% to the per-step wall-clock when the NS matmuls
are GPU-efficient.

### Interview Q&A

**Q1. Why Muon on 2D matrices but AdamW on everything else?**

> A: Muon's orthogonalization is matrix-shaped — it operates
> on singular vectors and singular values. For 1D scalars
> (norms, biases) or 3D+ tensors (embeddings, head), the
> "singular vector" notion doesn't apply, so we'd be wasting
> compute. AdamW is the universal fallback.

**Q2. Why cautious WD instead of standard decoupled WD?**

> A: Decoupled WD pulls every parameter toward 0 by a fixed
> fraction, regardless of the gradient. If the gradient is
> pushing in the opposite direction (a parameter is trying
> to *grow* to satisfy the loss), the WD term fights the
> gradient. Cautious WD masks the WD term unless the
> gradient and parameter agree, eliminating the fight.

**Q3. Why does Muon converge faster than AdamW on dense 2D
matrices?**

> A: Three reasons. (1) Spectrum preservation: every
> direction is treated equally, so the "signal" directions
> (large singular values) aren't dampened. (2) Magnitude
> preservation: the `||grad||_F` factor is the natural
> step size; AdamW's `1/sqrt(v)` can drift. (3) Shape
> invariance: the step size is roughly independent of
> matrix shape, so the LR transfers across width changes.

**Q4. Why FP32 master weights?**

> A: BF16 has 8 bits of mantissa; for accumulated updates
> (e.g. momentum buffers), the precision loss compounds over
> 57 k steps. FP32 has 23 bits of mantissa, which is
> effectively exact for the magnitudes we work with. The
> cost is 2× the optimizer state memory (per layer per
> parameter, ~2 KB extra at dim 896), well within the
> 80 GB A100 budget.

**Q5. Why `ns_iterations = 5` and not 3 or 10?**

> A: 5 iterations gives ~1e-3 spectral error for a
> Frobenius-normalized matrix. 3 iterations gives ~1e-2
> (10× worse); 10 iterations gives ~1e-5 (better, but
> doesn't change the convergence rate in practice). 5 is the
> empirical sweet spot.

**Q6. What happens to AdamW on a 2D matrix?**

> A: AdamW treats it as a vector of independent entries;
> the matrix structure is lost. The `m_hat / (sqrt(v_hat) +
> eps)` update dampens the entries with the largest
> gradients more than the entries with the smallest, which
> is the opposite of what we want for a meaningful gradient
> matrix.

**Q7. Why does `partition.py` exclude MoE experts from NorMuon?**

> A: MoE training is about *load balance* (the EMA gate-bias
> update), and Muon's orthogonalization can fight that
> balance by equalizing the update across all experts. AdamW
> treats each weight independently and lets the bias
> update do its job. The split is: NorMuon for attention + GDN
> 2D matrices, AdamW for embeddings, norms, gates, and MoE
> experts.

### Cross-links

- [`../training.md`](../training.md) §3
  (optimizer walkthrough).
- [`concepts/optimization.md`](optimization.md) — μP init
  is the partner scaling rule.
- [`concepts/gdn-and-mla.md`](gdn-and-mla.md) —
  why MoE experts are routed to AdamW.


## WSD: Warmup–Stable–Decay

### Learning objectives

After this file, you can:

1. State the WSD (warmup-stable-decay) schedule and why it
   beats cosine for large-scale pretraining.
2. Walk through the three decay kinds (linear, cosine, sqrt)
   and when each is appropriate.
3. Defend HyMo's defaults: 2% warmup, 83% stable, 15% decay,
   `min_lr_ratio = 0.05`, `decay = "linear"`.

### Intuition

The classic learning rate schedule is **cosine**: the LR
ramps up from 0 to peak over a short warmup, then decays as
a half-cosine to `min_lr_ratio` over the rest of training.

```
LR
peak ─────────┐
              ╲
               ╲
                ╲
                 ╲
                  ╲___
min ──────────────────────
       warmup        decay
```

For large-scale pretraining, cosine has two problems:

1. **Cannot extend without retuning**: if the run is extended
   to 50 B tokens, the LR trajectory shifts and the peak may
   need to be re-tuned.
2. **Decay is shape-changing**: the LR is never really "stable"
   at peak; it always starts decaying after warmup. This means
   the model never sees the full benefits of the peak LR.

**WSD** (warmup-stable-decay) splits the schedule into three
phases:

```
LR
peak ─────────────────────────────┐
                                  ╲
                                   ╲
                                    ╲
                                     ╲___
min ──────────────────────────────────────
       warmup      stable       decay
       (2%)        (83%)        (15%)
```

- **Warmup (2%)**: ramp 0 → peak linearly.
- **Stable (83%)**: hold at peak.
- **Decay (15%)**: drop from peak to `min_lr_ratio` along a
  chosen shape (linear, cosine, or sqrt).

### Why WSD beats cosine

Two reasons:

1. **Comparability across run lengths**: an ablation that runs
   for 7.5 B tokens with the same `warmup_frac / stable_frac /
   decay_frac` as the 30 B production run has the same LR
   shape — just shorter in each phase. Cosine would force the
   peak LR to scale with run length.
2. **Long stable phase**: the model sees the full peak LR for
   83% of training, which empirically gives ~5-10% better
   convergence at fixed tokens.

The cost: at the very end of training, the LR drops sharply
(linear). This can be mitigated by choosing `cosine` decay
instead of `linear`, but `linear` is the empirical default for
large-scale pre-training.

### Math derivation

### Phase A: warmup

```
factor(t) = t / warmup_steps            for t in [0, warmup_steps)
```

Smooth ramp from 0 to peak. Linearly in `t`, not based on
log-scale.

### Phase B: stable

```
factor(t) = 1.0                          for t in [warmup_steps, warmup_steps + stable_steps)
```

Hold at peak. Simple.

### Phase C: decay

```
progress = (t - warmup_steps - stable_steps) / decay_steps
                                             ∈ [0, 1]

linear:   f(p) = 1 - p
cosine:   f(p) = 0.5 · (1 + cos(π · p))
sqrt:     f(p) = sqrt(1 - p)

factor(t) = min_lr_ratio + (1 - min_lr_ratio) · f(progress)
```

The `min_lr_ratio + (1 - min_lr_ratio) · f(progress)` form
ensures that `factor = 1.0` at `progress = 0` and
`factor = min_lr_ratio` at `progress = 1`.

### Three decay shapes

| Shape | `f(p)` at `p = 0.5` | Sharpness at end |
|---|---|---|
| `linear` | 0.5 | Linear drop |
| `cosine` | 0.5 | Smooth tail |
| `sqrt` | 0.707 | Aggressive at end |

`linear` is the default; `cosine` is for transformer
fine-tuning where the long tail is valued; `sqrt` is rare
and used for fast end-of-training convergence.

### Why WSD's total_steps is the global optimizer step

The `total_steps` field on `SchedulerConfig` is the
**optimizer step** count, not the micro-batch count. So
`warmup_steps = total_steps × warmup_frac` is in optimizer
steps. With `gradient_accumulation_steps = 8`, the
warmup phase has `8 × warmup_steps` micro-batches.

This is what the trainer uses: `scheduler.get_factor(step + 1)`
is called with the optimizer step counter, not the
micro-step counter.

### Implementation in HyMo

- `src/hymo/training/scheduler.py:JointWSDScheduler` — `class JointWSDScheduler`.
- `src/hymo/training/scheduler.py:JointWSDScheduler.__init__` — `__init__`:
  `warmup_steps`, `stable_steps`, `decay_steps` from config
  properties; `min_lr_ratio`, `decay_kind` from config.
- `src/hymo/training/scheduler.py:JointWSDScheduler.get_factor` — `get_factor(step)`:
  the three-phase logic.
- `src/hymo/training/scheduler.py:JointWSDScheduler._decay_factor` — `_decay_factor(progress,
  kind)`: the static helper.
- `src/hymo/training/trainer.py:Trainer.train_step` — `factor = self.scheduler
  .get_factor(self.step + 1)`: the call site.
- `src/hymo/training/scheduler.py:JointWSDScheduler.state_dict` — `state_dict` /
  `load_state_dict`: the scheduler step counter.

### Worked example

Production scale (default `configs/hymo_750m.yaml`):

- `total_steps = 57_220`
- `warmup_frac = 0.02`, `stable_frac = 0.83`, `decay_frac = 0.15`
- `min_lr_ratio = 0.05`, `decay = "linear"`

Computed:

- `warmup_steps = 57220 × 0.02 = 1144`
- `stable_steps = 57220 × 0.83 = 47492`
- `decay_steps = 57220 × 0.15 = 8583`
- `stable_end = 1144 + 47492 = 48636`

LR trajectory (for `muon_lr = 0.02`):

```
step 0       → factor = 0/1144 = 0.0           → LR = 0.0000
step 572     → factor = 572/1144 = 0.5        → LR = 0.0100
step 1144    → factor = 1.0                     → LR = 0.0200 (peak)
step 1145..48635 → factor = 1.0                → LR = 0.0200 (stable)
step 52884   → factor = 0.05 + 0.95 × 0.5 = 0.525 → LR = 0.0105
step 57219   → factor = 0.05 + 0.95 × 0.0 = 0.05 → LR = 0.0010 (end)
```

The model spends ~83% of training at peak LR, then drops
sharply over the final 15% to 5% of peak.

For an ablation with 7.5 B tokens (per the ablations
framework), the same `warmup_frac = 0.02` gives
`warmup_steps = total_steps × 0.02 = (7.5e9 / 524288) × 0.02
= 286` — but the same proportion of training.

### Interview Q&A

**Q1. Why WSD over cosine?**

> A: WSD's stable phase holds the LR at peak for 83% of
> training, which empirically gives ~5-10% better convergence
> at fixed tokens. The cost is a sharp end-of-training drop;
> for pre-training, this is fine because the model has already
> converged by then.

**Q2. Why 2% warmup and not 5% or 0.5%?**

> A: 2% is the empirical default for large-scale
> pre-training. Shorter warmup (0.5%) risks an early spike
> in loss; longer warmup (5%) wastes optimizer steps that
> could be at peak LR. 2% is the sweet spot for 30 B tokens.

**Q3. Why 15% decay and not 10% or 30%?**

> A: 15% gives enough decay steps to bring the LR to
> `min_lr_ratio = 0.05` without being so long that the
> model effectively trains at low LR. 10% is too sharp; 30%
> wastes steps at sub-peak LR.

**Q4. Why `min_lr_ratio = 0.05` and not 0.0 or 0.1?**

> A: 0.05 is the Llama-3 / DeepSeek-V3 default for
> pre-training. 0.0 would let the optimizer make no
> progress at the very end. 0.1 would keep the LR
> artificially high, which can hurt final convergence.

**Q5. Why `decay = "linear"` and not "cosine"?**

> A: Linear decay gives a sharp end-of-training drop, which
> is fine for pre-training because the model has converged
> by then. Cosine decay would give a smoother tail but
> would still hold the LR at peak for the same 83% of
> training; the difference is only in the final 15%.

**Q6. Why is `Step` a `NewType` rather than `int`?**

> A: Type checking. `Step = NewType("Step", int)` is zero-cost
> at runtime but the type checker catches bugs like
> `scheduler.get_factor(micro_step)` (which should be the
> optimizer step). See `../references/config.md` §3.

**Q7. What happens if I extend the run to 50 B tokens?**

> A: Bump `total_steps` to `50e9 / 524288 = 95367`. The
> scheduler recomputes `warmup_steps = 1907`, `stable_steps =
> 79135`, `decay_steps = 14305`. The LR shape is the same; the
> training spends more steps at peak. No LR retuning needed.

### Cross-links

- [`../training.md`](../training.md) §4
  (scheduler walkthrough).
- [`../references/config.md`](../references/config.md) §2.3
  (SchedulerConfig).
- [`../training.md`](../training.md)
  §3.2 (how ablations inherit the schedule fractions).
- [`concepts/optimization.md`](optimization.md) —
  the optimizer that depends on the LR.


## FSDP-2: Full Parameter Sharding

### Learning objectives

After this file, you can:

1. State the lineage from DDP to ZeRO to FSDP-2.
2. Explain why full parameter sharding is necessary at 1 B+
   params.
3. Walk through HyMo's `wrap_model_with_fsdp` and the
   auto-wrap policy.
4. Compute per-rank memory at the production scale.

### Intuition

Standard **DDP** (Distributed Data Parallel) replicates the
model on every rank and synchronizes gradients via
all-reduce. Memory per rank: `O(model_size)` — every rank
holds the full model.

**ZeRO** (Zero Redundancy Optimizer, Rajbhandari et al.
2019) splits the optimizer state, gradients, and parameters
across ranks:

- **ZeRO-1**: optimizer state sharded.
- **ZeRO-2**: optimizer state + gradients sharded.
- **ZeRO-3**: optimizer state + gradients + parameters
  sharded.

**FSDP** (Fully Sharded Data Parallel, PyTorch's re-
implementation of ZeRO-3) is the modern API. The current
generation is **FSDP-2** (sometimes called "FSDP v2"),
which uses `fully_shard` and the new
`torch.distributed.fsdp.fully_shard` module.

The key operations:

- **All-gather** at the start of a forward pass: each rank
  reconstructs the full block of parameters it's about to
  compute on. Compute is local.
- **Reduce-scatter** at the end of the backward pass: each
  rank ends up with its shard of the gradients (already
  summed across ranks).
- **Re-shard** after the forward-backward: each rank frees
  the gathered parameters and goes back to its shard.

Memory per rank: `O(model_size / world_size)` — at 4 ranks,
each rank holds 1/4 of the model. This is what makes a 1.86 B
model fit on 4 × 80 GB A100s.

### Math derivation

### Memory per component

For a model with `P` parameters, `world_size = W`, and mixed
precision (BF16 params + FP32 master + FP32 AdamW state):

- **BF16 parameters** (sharded): `P / W × 2 bytes`.
- **BF16 gradients** (sharded): `P / W × 2 bytes`.
- **FP32 master weights** (sharded): `P / W × 4 bytes`.
- **AdamW state** (m, v, FP32): `P / W × 8 bytes`.
- **All-gather activations** (per-block): `O(block_size ×
  forward_activations)`.
- **Activations** (per-micro-batch): `O(B × T × dim ×
  L_block)` where `L_block` is the block count.

### Full-shard savings

At `P = 1.86 B`, `W = 4`:

- BF16 params: `1.86e9 / 4 × 2 = 930 MB`
- BF16 grads: `930 MB`
- FP32 master: `1.86e9 / 4 × 4 = 1.86 GB`
- AdamW state: `1.86e9 / 4 × 8 = 3.72 GB`
- All-gather activations: ~3.5 GB (variable, depends on
  block size)
- Total per rank: ~10 GB

A100 80 GB has 8× headroom.

### Communication cost

Each forward pass triggers `n_blocks` all-gathers; each
backward triggers `n_blocks` reduce-scatters. The total
communication volume per step is roughly:

```
volume = 2 × P × 2 bytes (BF16) = 7.4 GB   (one forward + one backward)
```

Across 4 ranks, this is ~1.85 GB per rank of all-gather
traffic. With NVLink at 600 GB/s, this is ~3 ms per step
— negligible relative to the ~5 s per step compute.

### BF16 mixed precision

FSDP-2 supports per-parameter mixed precision via
`MixedPrecision`. The recipe:

- Parameters stored as BF16 in the FSDP unit.
- Gradients computed as BF16 during all-gather.
- Master weights stored as FP32 in the optimizer (FSDP-2's internal
  `cast_param_meta`/mixed-precision contract; there is no
  `master_weights_dtype` config field).
- The optimizer step is computed in FP32.
- The master weights are cast back to BF16 for the next
  forward.

The `fsdp_mixed_precision = "bfloat16"` config field
selects this.

### Implementation in HyMo

- `src/hymo/training/fsdp.py:fsdp_auto_wrap_policy` — `fsdp_auto_wrap_policy
  (module, recurse, non_blocking)`: returns True for
  `GatedDeltaNetBlock` and `MLABlock`.
- `src/hymo/training/fsdp.py:wrap_model_with_fsdp` — `wrap_model_with_fsdp(model,
  config, *, world_size=None, auto_wrap_policy=None, **kwargs)`.
- `src/hymo/training/fsdp.py:wrap_model_with_fsdp` — try-import FSDP; if not
  available, return the model unwrapped (so CPU dev runs work).
- `src/hymo/training/fsdp.py:wrap_model_with_fsdp` — full wrapping with
  `MixedPrecision` from `config.fsdp_mixed_precision`.

The auto-wrap policy `fsdp_auto_wrap_policy` wraps each
`GatedDeltaNetBlock` and `MLABlock` as its own FSDP unit.
This means all-gather and reduce-scatter happen at the block
boundary (32 times per forward + 32 per backward), not at
the parameter boundary (hundreds of times per forward).

### Worked example

Production scale (`world_size = 4`, `fsdp = True`,
`fsdp_mixed_precision = "bfloat16"`):

Per-rank memory (`P = 1.86 B`):

| Component | Per rank |
|---|---|
| BF16 params (sharded) | 930 MB |
| BF16 grads (sharded) | 930 MB |
| FP32 master (sharded) | 1.86 GB |
| AdamW state (m, v) | 3.72 GB |
| All-gather activations (peak) | 3.5 GB |
| Activations (per micro-batch) | 1.5 GB |
| **Peak total** | **~10 GB / rank** |

A100 80 GB has 8× headroom — the dominant bottleneck is
**compute time**, not memory.

Communication:

```
Volume per optimizer step (forward + backward) = 2 × P × 2 bytes = 7.4 GB
Per rank (NVLink × 4) = 1.85 GB ≈ 3 ms
```

At 8 s per step (the production target), communication is
< 0.1% of the wall-clock.

### Interview Q&A

**Q1. Why FSDP and not DDP?**

> A: DDP replicates the full model on every rank. At
> `P = 1.86 B`, each rank holds 1.86 GB of BF16 parameters
> (plus 7.4 GB of optimizer state) — 9.2 GB just for the
> model and optimizer, plus activations. A100 80 GB can fit
> this, but it's tight. FSDP shards the model across ranks,
> so each rank holds only `P / W` of everything. With `W =
> 4`, each rank holds 1/4 of the model. Plus, the all-gather
> + reduce-scatter is bandwidth-cheap relative to the
> forward+backward.

**Q2. Why full sharding (ZeRO-3 / FSDP) instead of ZeRO-1 or 2?**

> A: At 1 B+ params, the optimizer state alone is 2× the
> parameters (FP32 master + AdamW `m` + `v`). For `P = 1.86 B`,
> that's 2 × 1.86 GB = 3.7 GB per rank just for the
> optimizer state. ZeRO-1 shaves that by `W`; ZeRO-3 shaves
> the parameters and gradients too. For 4 ranks, the savings
> are ~1.8 GB (ZeRO-1) vs ~7.4 GB (ZeRO-3).

**Q3. Why auto-wrap at the block level (not the parameter level)?**

> A: All-gather and reduce-scatter have fixed overhead per
> call (kernel launch, sync). At the parameter level, you'd
> have hundreds of all-gathers per forward. At the block
> level, you have 32. The trade-off is that each all-gather
> is larger (a whole block), but the per-call overhead is
> amortized.

**Q4. Why `world_size = 4` and not 8?**

> A: For the 1.86 B model on 4 × A100 80 GB, 4 ranks is
> enough to fit the model + activations + optimizer state
> with ~7× headroom. Doubling to 8 ranks halves the per-rank
> memory but halves the per-rank compute; the throughput
> per dollar is roughly the same. HyMo's 4-rank target is
> the empirical sweet spot for this model size.

**Q5. What happens if `torch.distributed.fsdp` is not
available?**

> A: `wrap_model_with_fsdp` returns the model unwrapped. This
> is the case for CPU dev runs and CI. The trainer still
> works (single-rank, no sharding), but the model uses ~9.2
> GB of RAM for parameters + AdamW state. The test suite
> uses the tiny config (~760 K params) so this isn't a
> problem.

**Q6. What is `fully_shard` (FSDP-2's new API)?**

> A: It's a module-level function that shards a module's
> parameters, gradients, and optimizer state. Called per
> block (or per parameter, depending on the granularity
> you want). The new FSDP-2 API is `torch.distributed.fsdp
> .fully_shard(module)` which in-place modifies the
> module to use FSDP. The legacy `FSDP` wrapper is what
> `wrap_model_with_fsdp` uses — both are supported in
> PyTorch 2.5+.

**Q7. Why does `MixedPrecision` need `bfloat16` and not
`float16`?**

> A: `float16` requires a `GradScaler` to handle
> underflow; `bfloat16` has the same exponent range as
> `float32` (no overflow under reasonable magnitudes) and
> no scaler is needed. HyMo's `training.fsdp_mixed_precision
> = "bfloat16"` skips the GradScaler entirely.

### Cross-links

- [`../training.md`](../training.md) §5
  (FSDP wrapping).
- [`optimization.md`](optimization.md) §7
  (overview of FSDP-2 + BF16).
- [`concepts/optimization.md`](optimization.md) —
  AdamW state is sharded by FSDP.
- [`concepts/optimization.md`](optimization.md) — μP init is
  applied before FSDP wrapping.


## Initialization (μP status)

### Learning objectives

After this file, you can:

1. State what initialization the shipped model actually applies.
2. Explain why the μP init described in the architecture doc was
   **never shipped**.
3. Identify where real init happens in the code.

### The honest status: μP init was designed, not shipped

The architecture doc (`design.md` §4) and the earlier draft of
this chapter describe a full **maximal-update parametrization** init:
`mup_init(model, config)` walking every parameter, zeroing gates/biases,
scaling 2D weights to `std = 1/dim`, and a `zero_init_predicate` keyword
set. That module existed at `src/hymo/models/init.py` and was removed in
the 2026-08-04 cleanup.

**It was never called.** `build_hymo` constructs `HyMo(config.model)`
and returns — no init pass. The trainer never invoked `mup_init` either.
The μP init was dead code: fully written, fully documented, and never
wired into the production path. The `mup_init: true` config flag was
inert.

The reasons it was safe to delete:

- `grep mup_init( src/` → only the definition; zero call sites outside tests.
- No test asserted production behavior depended on it (only unit tests of
  the predicate function itself).
- The actual init path (below) was already what every real run used.

### What the shipped model actually does

PyTorch defaults, plus two deliberate inline choices:

1. **`nn.Linear` / `nn.Embedding` defaults.** The constructor applies
   Kaiming-uniform / normal init to weights, zero bias. `HyMo.__init__`
   calls no custom init; `GatedDeltaNetBlock`, `MLABlock`, `DeepSeekMoE`,
   and `MultiTokenPrediction` rely on module defaults.

2. **MoE gate init (inline in `moe.py`).** `DeepSeekMoE.__init__` sets
   `nn.init.zeros_(self.gate.bias)` and `nn.init.normal_(self.gate.weight,
   std=0.006)` — the gate starts near-uniform so routing is neutral at
   step 0 and the EMA bias update specializes it over time.

3. **GDN `A_log`/`dt_bias`/`D` (inline in `gdn.py`).** `A_log` starts at
   `log(1..n_heads)` so `A = -exp(A_log)` is a gentle decay; `dt_bias` is
   zero; `D` is ones. These are the values the recurrence actually uses —
   no external init pass touches them.

### Implementation in HyMo

- `src/hymo/models/model.py:HyMo.__init__` — `HyMo.__init__`: no init pass; the
  constructor relies on module defaults.
- `src/hymo/models/moe.py:DeepSeekMoE.__init__` — the gate init: `bias = 0`,
  `weight ~ N(0, 0.006²)`.
- `src/hymo/models/gdn.py:GatedDeltaNetBlock.__init__` — `A_log`, `dt_bias`, `D` inline init.

### Interview Q&A

**Q1. Why is the MoE gate init so small (`N(0, 0.006²)`)?**

> A: Because the gate is the **routing signal**. A gate with
> `weight ~ N(0, 1/dim)` would produce logits `O(1)` and a near-uniform
> softmax over 16 experts — fine start routing, but the EMA gate-bias
> update has to do all the work to specialize. Starting with smaller
> weights keeps the softmax closer to uniform across more of the early
> training, giving the EMA update more time to react to actual load
> imbalance.

**Q2. Why is `gate.bias = 0`?**

> A: Same reason — start uniform routing. The bias is updated by the EMA
> to break the symmetry as training progresses. Zero is the neutral
> starting point.

**Q3. Why is `A_log` initialized to `log(1..n_heads)`?**

> A: `A = -exp(A_log)` is the per-head decay. Starting near `A = -1`
> gives `α = exp(g · A) ≈ 0.27` at typical sigmoid inputs — a moderate
> decay: recent writes are weighted heavily, but old state isn't
> completely forgotten. This is the balanced starting point for the
> recurrence.

**Q4. Was μP init ever active?**

> A: No. It was written and documented but never called from
> `build_hymo` or the trainer; it was removed in the 2026-08-04 cleanup.
> If μP scaling is wanted later, the design (architecture doc §4) is
> preserved in git history — but the LR schedule (NorMuon `0.02`, AdamW
> `3e-4`) was tuned on the *current* init, so enabling μP would require
> re-tuning.

### Cross-links

- [`model-architecture.md`](model-architecture.md) §10
  (model construction walkthrough).
- [`concepts/optimization.md`](optimization.md) — the optimizer.
- [`concepts/gdn-and-mla.md`](gdn-and-mla.md) — the
  gate init and EMA bias update.



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

### Optimization overview

The production target is sustained throughput on 4× A100 80 GB SXM,
with `per_step_tokens = 524,288` (see
[`../references/config.md`(../references/config.md) §2.4). The four
in-scope optimizations are toggled by flags in `TrainingConfig`; the
fifth, **FSDP-2**, is what makes the 1.86 B-param model fit across
4 GPUs at all.

| # | Optimization | Flag (TrainingConfig) | Where it lives | Speedup vs eager |
|---|---|---|---|---|
| 1 | Triton GDN kernel | `fused_gdn` | `gdn.py` + `gdn_triton.py` | 3–5× over the Python double-loop |
| 2 | `torch.compile` on GDN | `torch_compile_gdn` | `gdn.py::forward` | ~1.2× via small-op fusion |
| 3 | MoE mixed precision (BF16 dispatch) | `moe_mixed_precision` | `moe.py::forward` | ~1.3× via halved HBM bandwidth |
| ~~4~~ | CUDA Graphs on MLA | ~~`cuda_graphs_mla`~~ | — | Removed in the 2026-08-04 cleanup (never shipped) |

The fifth optimization — **FSDP-2 full sharding** — is not optional at
production scale. It's gated by `training.fsdp = True` (the default).
See [optimization.md](optimization.md) (FSDP-2 section) for the
mechanics and `src/hymo/training/fsdp.py:wrap_model_with_fsdp` for the
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

### The four optimization flags

Defined as fields of
`TrainingConfig`:

```python
fused_gdn: bool = True
moe_mixed_precision: bool = True
torch_compile_gdn: bool = True
```

(All default to `True` because production training depends on them.
For a CPU smoke test or a debugging run, override to `False`.)

### 2.1 Wiring: `Trainer._thread_optimization_flags`

`src/hymo/training/trainer.py:Trainer._thread_optimization_flags`:

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
```

To disable any optimization for a debugging run, pass `--config
debug.yaml` (where `debug.yaml` overrides one or more flags to
`false`); the trainer will see the new value on init and re-thread
it. There is no per-step toggling — the attribute is read on each
forward.

---

### Triton GDN kernel

> **This is the headline optimization.** It is the only sanctioned
> custom kernel in the codebase
> ([`AGENTS.md`](../../AGENTS.md) §Engineering rules). The previous
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

`gdn.py:GatedDeltaNetBlock.__init__` initializes:

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
This is mandated by [`AGENTS.md`](../../AGENTS.md) §Hard don'ts:

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

### `torch.compile` on GDN blocks

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

### MoE mixed precision and EMA gate-bias

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
`Trainer._update_moe_gate_biases`):

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

`moe.py:DeepSeekMoE.update_gate_bias`. See [`gdn-and-mla.md`(gdn-and-mla.md) for the
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

### CUDA Graphs on MLA — removed (2026-08-04)

The `cuda_graphs_mla` flag and `MLABlock.use_cuda_graphs` attribute were
**removed in the cleanup** — no CUDA-graph capture path ever shipped (the
attr was set but never read in `forward`; the code carried a
`ponytail:` comment acknowledging this). If kernel-launch overhead on the
MLA path ever becomes the training bottleneck, add explicit
`torch.cuda.CUDAGraph()` capture/replay (design §12a.4) and re-add the
flag then.

### FSDP-2 + BF16 mixed precision (overview)

Full coverage is in [`optimization.md`(optimization.md) and
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

### Gradient accumulation and NaN-skip

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

### Memory budget analysis

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

### Interview Q&A

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

> A: Because `_thread_optimization_flags` (`trainer.py:Trainer._thread_optimization_flags`)
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

### Cross-links

- Walkthrough: `model-architecture.md` §5–6 (GDN block,
  Triton kernel call site), §8 (MoE), §4 (MLA), §3 (model top-level).
- Concepts: `kernels.md` (Triton + autograd
  patterns), `gdn-and-mla.md` (EMA gate-bias
  derivation), `optimization.md` (FSDP-2 mechanics).
- Tests: `tests/unit/test_triton_gdn_gpu.py` (kernel parity),
  `tests/unit/test_training.py::test_train_step_consumes_optimization_flags`
  (flag wiring).

## References

- [model-architecture.md](model-architecture.md) — the model these optimizers train.
- [gdn-and-mla.md](gdn-and-mla.md) — MoE routing and the EMA gate-bias derivation.
- [kernels.md](kernels.md) — the Triton kernel and `torch.compile` interaction.
- [training.md](../training.md) — the trainer loop, parameter partition, and checkpointing.
- [config.md](../references/config.md) — the `OptimizerConfig` / `SchedulerConfig` / `TrainingConfig` fields.
- Source: `src/hymo/training/optimizer.py`, `src/hymo/training/scheduler.py`, `src/hymo/training/fsdp.py`, `src/hymo/training/partition.py`, `src/hymo/training/trainer.py`, `src/hymo/models/moe.py`, `src/hymo/models/gdn.py`, `src/hymo/models/model.py`.
