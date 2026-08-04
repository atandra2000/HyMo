# 06 — μP Initialization

> **Bridges to:** [`learning_docs/1_Model_Architecture.md`](../../learning_docs/1_Model_Architecture.md) §10
> (μP init)

## Learning objectives

After this file, you can:

1. State the μP (Maximal Update Parameterization) scaling
   rules.
2. Explain why standard init (PyTorch default) breaks at scale.
3. Walk through HyMo's μP init in `init.py`.
4. Defend `mup_init: true` as the v1.0 default.

## Intuition

Standard PyTorch init for `nn.Linear`: `weight ~ N(0, 1/fan_in)`,
`bias ~ N(0, 1/fan_in)` (or zero). This is the "canonical" init
that's been used since GPT-2.

The problem: at scale, the **update/parameter ratio** matters
more than the absolute magnitudes. If `update / param = O(1)` at
small scale but `O(1/sqrt(N))` at large scale, the model learns
nothing — the optimizer is too timid to move parameters that
are initialized to large magnitudes.

**μP** (Yang et al. 2021, arXiv 2011.14522) fixes this by
scaling init *and* learning rates so that the
update-to-parameter ratio is constant across model width. In
particular:

- The **last layer** (the head) is the canary: its init and
  learning rate are set so that the output of the residual
  stream has `O(1)` magnitude regardless of width.
- All other layers are scaled to match: their init and LR
  decrease as `1/sqrt(fan_in)` so that the update-to-parameter
  ratio at the last layer is constant.

Practically, this means you can tune the LR on a **tiny
model** (e.g. 100 M params) and the same LR works at the
**production scale** (750 M, 1 B, 100 B). The standard init
breaks this transfer.

## Math derivation

### Standard init

For an `nn.Linear(dim_in, dim_out)` without μP:

```
weight ~ N(0, σ²)       with σ² = 1 / dim_in          (Kaiming uniform)
bias   = 0                                          (or 0.01 in some init schemes)
```

The output magnitude is `O(1)` per element (because the
fan_in = dim_in). The gradient magnitude is `O(1)` per
element. The update-to-parameter ratio is `O(1)`.

But this assumes `dim_in` is fixed. If you scale
`dim_in → 4 × dim_in` (a wider model):

- `weight ~ N(0, 1/4 × dim_in)` → smaller init.
- The gradient through the layer is `O(1 / sqrt(dim_in))` → smaller.
- The update is `O(1 / sqrt(dim_in))` → smaller.
- The parameter is `O(1 / sqrt(dim_in))` → smaller.

The ratio `update / param` stays `O(1)`. So standard init *is*
approximately scale-invariant in the fan-in dimension. But
**depth scaling** is different — adding more layers (depth
`L → 4L`) compounds the magnitudes, and standard init doesn't
address that.

### μP's scaling

For depth scaling, μP says:

```
init variance per layer    = 1 / dim_in                         (as standard)
learning rate per layer    = base_lr / L                        (scales with 1/depth)
last-layer init             = 1 / dim_in                         (same as standard)
last-layer learning rate    = base_lr                            (constant)
```

The `base_lr` is dimension-free: it doesn't depend on
`dim` or `L`. The expected update-to-parameter ratio is
`O(1)` at every layer, regardless of width or depth.

### HyMo's specific implementation

HyMo's `mup_init` (in `src/hymo/models/init.py`) does:

1. **Per-parameter variance**: `weight ~ N(0, σ²)` where
   `σ² = 1 / fan_in` for dense layers, `σ² = 0.006²` for the
   MoE gate (small because the gate is the routing signal —
   we want it to *start* near-uniform).
2. **GPT-2-style residual projection**: MLP outputs and
   attention outputs are scaled by `1/sqrt(2L)` so that
   `||residual||`-magnitudes don't blow up over `L` layers.
3. **Tied embeddings**: `head.weight = embed.weight`. The
   head's init is the same as the embed's, so the initial
   logits are `O(1)` for any vocab size.
4. **Zero-init for stability params**: `gate.bias = 0` (start
   uniform routing), `A_log = 0` (start with `A = -1`, so
   `α = exp(g · A) = exp(-g) ≈ 0.27` when `g = 1`).

## Implementation in HyMo

- `src/hymo/models/init.py:39` — `def mup_init(model, config)`:
  the main entry point that walks the model and applies the
  init rules.
- `src/hymo/models/init.py:56` — `def zero_init_predicate
  (param_name)`: returns True for parameters that should be
  zero-initialized (norms, gate biases, MTP scalars).
- `src/hymo/models/gdn.py:34` — `use_triton = True` /
  `use_compile = True` defaults (separate concern).
- `src/hymo/models/moe.py:71-73` — `gate = Linear(dim,
  n_routed)` with `bias = 0`, `weight ~ N(0, 0.006²)`.

The `mup_init: true` flag is in `TrainingConfig`. When True,
`build_hymo` calls `mup_init(model, config)` after the
constructor.

## Worked example

Production scale (32 layers, dim 896):

- Standard init would set `weight ~ N(0, 1/896)` for each
  layer. After 32 layers, the residual stream has
  accumulated magnitudes that are `O(sqrt(32))` larger than
  the initial.
- μP adds the `1/sqrt(32)` projection at each layer so the
  residual stream magnitude stays `O(1)` throughout.
- The head's `weight ~ N(0, 1/896)`; logits magnitude
  `O(1/sqrt(896))` per vocab dim. With `vocab_size = 64_256`,
  the largest logit is `O(1)`, so the softmax is well-
  conditioned at init.

Without μP, the first 100 steps of training would have to
"unlearn" the random initialization — initial gradient
norms can be `O(100)` and the loss curve shows a brief
plateau. With μP, the loss decreases monotonically from
step 1.

## Interview Q&A

**Q1. Why does standard init break at scale?**

> A: Because depth scaling compounds the residual stream
> magnitude. With `L` layers, the residual norm grows like
> `sqrt(L)`; the gradient norms grow similarly. The
> optimizer can spend the first 1 k steps just "burning off"
> the random init magnitudes before any real learning
> happens. μP pre-compensates for this depth-scaling.

**Q2. Why is the MoE gate init so small (`N(0, 0.006²)`)?**

> A: Because the gate is the **routing signal**. A gate
> with `weight ~ N(0, 1/dim)` would produce logits `O(1)`
> and a near-uniform softmax over 16 experts — which is
> fine start routing, but it means the EMA gate-bias update
> has to do all the work to specialize. Starting with
> smaller weights keeps the softmax closer to uniform
> across more of the early training, giving the EMA update
> more time to react to actual load imbalance.

**Q3. Why is `gate.bias = 0` zeros?**

> A: Same reason — start uniform routing. The bias is
> updated by the EMA to break the symmetry as training
> progresses. Zero is the neutral starting point.

**Q4. Why is `A_log` initialized to 0?**

> A: `A = -exp(A_log) = -1` when `A_log = 0`. The decay
> `α = exp(g · A) = exp(-g)`. With `g ≈ 1` (typical sigmoid
> input), `α ≈ 0.27`. This is a moderate decay — recent
> writes are weighted heavily, but old state isn't
> completely forgotten. If `A_log` started at `-1`, decay
> would be `α ≈ 0.05` (almost no retention); if `A_log`
> started at `+1`, decay would be `α ≈ 2.7` (state grows).
> Zero is the "balanced starting point".

**Q5. Why tie embeddings?**

> A: Two reasons. First, parameter efficiency — the embed
> is `vocab_size × dim = 64_256 × 896 = 57.5 M` params, the
> same as the head. Tying means we save 57.5 M params. Second,
> well-trained embeddings and well-trained heads have
> similar column geometry (each row is a "token
> representation"), so tying is a useful regularization.

**Q6. Why is `mup_init: true` the default and not an
opt-in?**

> A: Because the recipe is calibrated for `mup_init =
> true`. The downstream scaling (NorMuon LR
> `0.02`, AdamW LR `3e-4`) was tuned with μP on. Switching
> to `mup_init = false` would require re-tuning the LRs and
> total steps. The flag exists for ablation but should
> not be flipped casually.

## Cross-links

- [`learning_docs/1_Model_Architecture.md`](../../learning_docs/1_Model_Architecture.md) §10
  (μP init walkthrough).
- [`concepts/07-muon-optimizer.md`](07-muon-optimizer.md) —
  the optimizer that depends on μP-style init.
- [`concepts/06-mup-init.md`](06-mup-init.md) (this file).
