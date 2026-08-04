# 07 — Muon Optimizer: Newton–Schulz and Cautious WD

> **Bridges to:** [`learning_docs/3_Training_Pipeline.md`](../../learning_docs/3_Training_Pipeline.md) §3
> (optimizer)

## Learning objectives

After this file, you can:

1. State the Muon optimizer and explain why it outperforms
   AdamW on dense 2D weight matrices.
2. Walk through Newton–Schulz iteration and derive the
   orthogonalization step.
3. Explain cautious weight decay (Liang et al. 2024) and why it
   matters at scale.
4. Defend HyMo's `NorMuon` (Muon + cautious WD + FP32 masters)
   variant.

## Intuition

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

## Math derivation

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

## Implementation in HyMo

- `src/hymo/training/optimizer.py:19` — `_newton_schulz_orthogonalize
  (g, iterations=5)`: the 5-step NS iteration.
- `src/hymo/training/optimizer.py:32` — `class NorMuon(Optimizer)`.
- `src/hymo/training/optimizer.py:35` — `__init__` with
  `lr, momentum, betas, eps, weight_decay, cautious_wd,
  ns_iterations`.
- `src/hymo/training/optimizer.py:62` — `step()`: applies
  momentum, cautious mask, Newton–Schulz, decoupled WD.
- `src/hymo/training/optimizer.py:111` — `class CautiousAdamW`.
- `src/hymo/training/optimizer.py:114` — `__init__` with
  `embed_weight_decay` separate from `weight_decay`.
- `src/hymo/training/optimizer.py:135` — `step()`: AdamW
  with cautious mask.
- `src/hymo/training/optimizer.py:188` — `class Optimizers`:
  the bundle `(nor_muon, adamw)`.
- `src/hymo/training/optimizer.py:209` — `build_optimizers(model,
  config)`: the partition + dual build.

## Worked example

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

## Interview Q&A

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

## Cross-links

- [`learning_docs/3_Training_Pipeline.md`](../../learning_docs/3_Training_Pipeline.md) §3
  (optimizer walkthrough).
- [`concepts/06-mup-init.md`](06-mup-init.md) — μP init
  is the partner scaling rule.
- [`concepts/03-mixture-of-experts.md`](03-mixture-of-experts.md) —
  why MoE experts are routed to AdamW.
