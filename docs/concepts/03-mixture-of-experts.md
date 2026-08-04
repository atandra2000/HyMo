# 03 — Mixture of Experts

> **Bridges to:** [`learning_docs/1_Model_Architecture.md`](../../learning_docs/1_Model_Architecture.md) §8
> (MoE block)

## Learning objectives

After this file, you can:

1. State the Mixture-of-Experts sparsity pattern and how
   top-k routing works.
2. Walk through DeepSeek-style MoE: 16 routed + 1 shared expert,
   top-2 routing.
3. Explain why aux-loss-free (EMA gate-bias) load balancing
   replaces the classical auxiliary loss.
4. Compute the FLOPs savings of MoE vs. dense FFN at the same
   parameter count.

## Intuition

A dense FFN (SwiGLU) at the production shape is
`dim × 3 × inter_dim = 896 × 3 × 2304 = 6.2 M` FLOPs *per token
per layer*. With 8 MLA layers and 32 total layers, that's
`8 × 6.2 M = 50 M` FLOPs per token just for the FFN on the MLA
blocks.

A sparse MoE with `n_routed = 16` experts and `top-2 = 2`
activation has the same per-expert cost (each expert is a
SwiGLU of the same `inter_dim`), but only **2 of 16** run per
token. So:

```
  dense FLOPs per token per MoE layer = 16 × 6.2 M = 99 M
  sparse FLOPs per token per MoE layer = 2 × 6.2 M  = 12 M
  savings                                              = 8.3×
```

at the cost of **16× the stored parameters** (the inactive
14 experts still exist, just don't run). That's the MoE
trade-off: more storage for less compute.

Two engineering problems:

1. **Routing**: how to choose which 2 of 16 experts each token
   goes to. Top-k routing with a learned gate (`Linear(dim,
   n_routed)`) is the standard. The gate produces a softmax
   over the 16 experts; top-2 are selected.
2. **Load balancing**: if the gate learns to send every token
   to the same 2 experts, the other 14 are wasted. The
   classical fix is an auxiliary loss that penalizes
   imbalance. HyMo's fix is **aux-loss-free**: an EMA-smoothed
   bias added to the gate logits, where the bias is increased
   for underused experts and decreased for overused ones.

The **shared expert** (1 of the 17 total MoE parameters) is a
separate, always-on SwiGLU. It captures "common patterns" that
should be available to every token, regardless of which
routed experts fire. This is the DeepSeek-V2/V3 design.

## Math derivation

### Top-k routing

Given hidden state `x ∈ ℝ^{dim}`:

```
g = softmax(Linear(dim, n_routed)(x))        ∈ ℝ^{n_routed}
indices, weights = topk(g, k=2)              # 2 of 16 experts
```

`indices[i]` is the expert chosen for token `i`'s `i`-th
activation; `weights[i]` is the gate probability (used as a
mixing coefficient).

### Sparse dispatch

The naive version iterates over all 16 experts and gathers
tokens assigned to each. With top-2 routing, each expert
receives ~ `1/16 × 2 = 1/8` of tokens on average (the
`capacity_factor = 1.5` caps the upper bound).

Output:

```
y = Σ_{i=1}^{k} weights[i] · expert_i(x)
```

### Capacity factor

Per-expert token cap:

```
capacity = int(capacity_factor * (B * T * k) / n_routed)
         = int(1.5 * (4 * 4096 * 2) / 16)      = 3072
```

With `B * T * k = 32,768` tokens to dispatch across 16 experts,
the average is 2048 tokens per expert. The cap of 3072 means
no expert sees more than 1.5× the mean — over-capacity tokens
are dropped (silently, since the per-expert forward skips them).

### EMA gate-bias update (the aux-loss-free trick)

Pseudo-code (matches `src/hymo/models/moe.py:103-119`):

```python
def update_gate_bias(self, speed=0.001):
    counts = bincount(self._last_indices, minlength=n_routed)
    ema = (1 - 0.02) * ema + 0.02 * counts
    avg = ema.mean()
    over  = ema > avg * 1.05
    under = ema < avg * 0.95
    bias[over]  -= speed
    bias[under] += speed
```

The decay constant `0.98` weights the previous EMA at 98%; the
new batch counts at 2%. This is a 50-batch effective averaging
window.

The thresholds `1.05` and `0.95` (±5% of the mean) define the
"balanced" zone — biases inside this band are not updated.
Outside it, the bias moves by `speed = 0.001` per step.

The intuition: an expert that has been getting more than 5%
above the mean should be slightly *less* attractive (bias
decreased), so future tokens are less likely to choose it. An
underused expert gets the opposite. The 0.001 step is small
enough to be stable over the 57 k training steps but large
enough to converge within ~1 k steps.

### Why not an auxiliary loss?

The classical DeepSeek-MoE adds `λ · Σ(per_expert_fraction ×
per_expert_mean)` to the training loss; `λ` is a hyperparameter
that has to be tuned to balance load balance vs. main loss.

The EMA approach has **no additional loss term** — the bias
update is a detached operation that doesn't contribute to the
gradient. The trade-off:

- Aux loss: one extra backward pass per step, one extra
  hyperparameter, potential gradient interference with the
  main loss.
- EMA bias: zero interaction with the main loss; one extra
  `update_gate_bias()` call per step; one hyperparameter
  (`speed`) that is robust across scale.

HyMo chose EMA. The DeepSeek-V3 paper does the same.

## Implementation in HyMo

- `src/hymo/models/moe.py:1-9` — module docstring
- `src/hymo/models/moe.py:18` — `__all__ = ["SwiGLUExpert",
  "DenseFFN", "DeepSeekMoE"]`
- `src/hymo/models/moe.py:21` — `class SwiGLUExpert`: a single
  SwiGLU with `w1` (gate), `w2` (down), `w3` (up).
- `src/hymo/models/moe.py:39` — `class DenseFFN`: same shape
  as SwiGLU expert, used on **GDN blocks** (which are dense, not
  MoE).
- `src/hymo/models/moe.py:57` — `class DeepSeekMoE`.
- `src/hymo/models/moe.py:60-69` — `__init__`:
  `n_routed, n_shared, n_activated, moe_inter_dim, ema_alpha,
  capacity_factor` from `ModelConfig`.
- `src/hymo/models/moe.py:71-73` — the `gate = Linear(dim,
  n_routed)` with bias zero-initialized and weight `N(0, 0.006)`.
- `src/hymo/models/moe.py:77` — `self.use_mixed_precision = True`
  (default; threaded from `TrainingConfig.moe_mixed_precision`).
- `src/hymo/models/moe.py:89-93` — `ema_expert_counts` registered
  as a non-persistent buffer (`persistent=False` so it doesn't
  save/load with checkpoints).
- `src/hymo/models/moe.py:95` — `gate_forward(x)`: the FP32
  gate computation.
- `src/hymo/models/moe.py:103` — `update_gate_bias(speed=0.001)`:
  the EMA logic.
- `src/hymo/models/moe.py:121` — `forward(x)`: full dispatch
  loop.

The wiring from `Trainer` is in
`src/hymo/training/trainer.py:329-335` (`_update_moe_gate_biases`):

```python
def _update_moe_gate_biases(self) -> None:
    from hymo.models.moe import DeepSeekMoE
    for module in self.model.modules():
        if isinstance(module, DeepSeekMoE):
            module.update_gate_bias()
```

This fires after every optimizer step (not every micro-batch) —
see [`learning_docs/3_Training_Pipeline.md`](../../learning_docs/3_Training_Pipeline.md) §6.2 for why.

## Worked example

Production scale:

- `n_routed = 16`, `n_shared = 1`, `n_activated = 2`,
  `moe_inter_dim = 2304`, `dim = 896`.
- Per-expert params: `3 × 896 × 2304 = 6.2 M`. With 16 routed + 1
  shared = `17 × 6.2 M = 105 M` MoE params per layer.
- Across 8 MLA layers: `8 × 105 M = 840 M` MoE params
  (most of the "stored" 1.86 B).
- Per-token FLOPs: `2 × 6.2 M = 12.4 M` per layer (top-2
  routing × expert FLOPs) — `8 × 12.4 M = 99 M` total.
- Per-token "saved" FLOPs vs. dense: `8 × (16-2)/16 × 6.2 M =
  43 M` (the 14 inactive experts * 16 - 2 = 14 × savings per
  dense equivalent).

The EMA gate-bias update fires **once per optimizer step**, so
over 57 k steps there are 57 k bias updates. With `speed =
0.001`, the bias moves by at most `57_220 × 0.001 = 57.2` units
in absolute terms — large enough to re-balance from a
pathological starting state, small enough to not destabilize a
balanced one.

## Interview Q&A

**Q1. Why DeepSeek-style MoE (routed + shared) instead of just
top-k routed?**

> A: The shared expert captures "common patterns" that every
> token needs regardless of routing. Without it, every token has
> to find a routed expert for common functionality; with it, the
> shared SwiGLU handles that and the routed experts specialize
> on the residual. This is empirically what the DeepSeek-V2/V3
> papers found.

**Q2. Why not an auxiliary loss for load balancing?**

> A: Three reasons. First, an auxiliary loss has to compete
> with the main loss for gradient signal — adding `λ × aux_loss`
> to the loss means the optimizer updates weights to satisfy
> the load-balance constraint *and* the language modeling
> objective, with the trade-off controlled by `λ`. Second, `λ`
> has to be tuned per scale; what's right at 1 B params might
> be wrong at 100 B. Third, the EMA approach is gradient-free —
> the bias update doesn't affect any backward pass. The single
> hyperparameter is `speed`, which is remarkably stable across
> scale (the 0.001 default works at 750 M and at 100 B+).

**Q3. Why is the gate computed in FP32 if the rest is BF16?**

> A: Softmax underflows in BF16 for small logits. A 16-bit float
> cannot represent probabilities below ~6e-8, so a near-uniform
> 16-way softmax becomes exactly uniform after BF16 rounding —
> which collapses routing to random. Doing the gate matmul +
> softmax in FP32 (then casting back) costs one FP32 matmul of
> shape `(B*T, dim) × (dim, n_routed)` — negligible relative to
> the expert SwiGLUs.

**Q4. Why `capacity_factor = 1.5` and not 1.0 or 2.0?**

> A: 1.0 means exactly the average — any imbalance drops tokens
> silently. 2.0 means twice the average — wasteful. 1.5 is the
> "moderate slack" sweet spot; experts can run 50% above their
> mean before tokens start dropping. For a 16-expert,
> top-2 routing, that means up to 3072 tokens per expert per
> layer per forward.

**Q5. Why does the EMA bias update run only on optimizer steps?**

> A: Because the EMA averages over many batches. Updating on
> every micro-batch would make the bias noisy. Tying the update
> to the optimizer step sees the *averaged* dispatch per step,
> which is what we want.

**Q6. Why does the mixed-precision flag cast `x_flat.to(weight.dtype)`?**

> A: Under FSDP-BF16, the expert weights are stored in BF16.
> The activations come in at the residual-stream dtype (also
> BF16 in production). The matmul `x @ wᵀ` runs in BF16 with
> this cast, halving the input bandwidth. On CPU (where weights
> are FP32), the cast is a no-op because the dtype is already
> FP32.

## Cross-links

- [`learning_docs/1_Model_Architecture.md`](../../learning_docs/1_Model_Architecture.md) §8
  (DeepSeekMoE block walkthrough).
- [`learning_docs/4_Optimizations.md`](../../learning_docs/4_Optimizations.md) §5 (mixed
  precision + EMA gate-bias).
- [`learning_docs/3_Training_Pipeline.md`](../../learning_docs/3_Training_Pipeline.md) §6.5
  (the trainer's `_update_moe_gate_biases` call).
- [`concepts/11-hybrid-architectures.md`](11-hybrid-architectures.md) —
  why MoE is **only** on MLA blocks, not GDN.
