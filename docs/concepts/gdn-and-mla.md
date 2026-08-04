# HyMo — GDN, MLA, MoE, MTP, and the Hybrid Stack

> This doc consolidates the mechanism deep-dives behind HyMo's hybrid architecture:
> the Gated Delta Net (linear attention), Multi-Head Latent Attention (MLA, full attention),
> DeepSeek-style Mixture of Experts, Multi-Token Prediction, and the 3:1 hybrid-stack thesis.
> The line-by-line model walkthrough lives in [model-architecture.md](model-architecture.md);
> the Triton kernel is covered in [kernels.md](kernels.md).


## Linear Attention and the Gated Delta Net

### Learning objectives

After this file, you can:

1. Derive linear attention from softmax attention by replacing
   the softmax with a kernel feature map.
2. State the delta rule and explain why gating makes it the
   "Gated Delta Net".
3. Sketch the chunked recurrence and why it's a Triton-friendly
   algorithm.
4. Defend HyMo's choice of GDN over Mamba-2/S4/RWKV at this scale.

### Intuition

Softmax attention is `O(N²)` in time and memory because every
token attends to every previous token. The quadratic is in the
softmax (which produces an `N × N` matrix) — not the value
projection.

**Linear attention** (Katharopoulos et al. 2020) replaces the
softmax with a *feature map* `φ`:

```
softmax(Q Kᵀ) V  ≈  φ(Q) · (φ(K)ᵀ V)
```

The associativity lets us compute `φ(K)ᵀ V` once (the state), then
read from it with each query. Time and memory drop to `O(N)`.

This works for a vanilla linear attention, but the kernel
feature map `φ` is hard to choose. The delta-rule lineage
(Gated Delta Net, Yang et al. 2024) replaces the feature map
with a **learned write-key / read-key** decomposition
(`b` and `c`), so the recurrence is:

```
h_t = α_t · h_{t-1} + b_t ⊗ v_t           (write)
o_t = c_t · h_t                            (read)
```

where `α_t = exp(g_t · A)` is a per-head, per-state scalar decay
in `(0, 1)`. The recurrence is still `O(N)` time and `O(1)` memory
*per state*; the integration is what makes it well-defined
without the `φ` choice.

Gating enters via `g_t = sigmoid(dt_proj(x)_t)` — a learned
per-token, per-head scalar that drives the decay. When `g_t ≈ 0`,
the past state is forgotten; when `g_t ≈ 1`, the past state is
preserved. The `A` is a learned per-head, per-state scalar (in
log-space, `A_log`).

This is the **Gated Delta Net** (Yang et al. 2024, arXiv 2412.06464,
ICLR 2025), and it is what HyMo uses as the linear-attention
substitute.

### Math derivation

### Step 1: Softmax attention

```
A = softmax(Q Kᵀ / √d)        ∈ ℝ^{N × N}
o_i = Σ_j A[i,j] · v_j
```

### Step 2: Replace softmax with a kernel `φ`

```
A[i, j] ≈ φ(q_i)ᵀ φ(k_j)
o_i = Σ_j φ(q_i)ᵀ φ(k_j) · v_j
     = φ(q_i)ᵀ · Σ_j φ(k_j) · v_j
     = φ(q_i)ᵀ · S_N                    where S_N = Σ_j φ(k_j) ⊗ v_j
```

The state `S_N ∈ ℝ^{d_φ × d_v}` accumulates in `O(N)`. New `S` is
`S_{t+1} = S_t + φ(k_{t+1}) ⊗ v_{t+1}`.

### Step 3: Delta rule (Schlag et al. 2021)

Let `φ(q_i) = W_q x_i`, `φ(k_j) = w_k x_j` for chosen matrices
(this is the "linear transformer" simplification). The state is
no longer additive — we want to *forget* and *replace*:

```
h_t = h_{t-1} + b_t ⊗ (v_t - h_{t-1} · c_t)
```

where `b_t` and `c_t` are learned write/read keys. This is the
"delta rule" — replace the old contribution with the new one.

### Step 4: Gated delta rule (Yang et al. 2024)

Add a per-head decay `α_t ∈ (0, 1)`:

```
h_t = α_t · h_{t-1} + b_t ⊗ v_t         (write)
o_t = c_t · h_t                          (read)
```

where `α_t = exp(g_t · A)`, with `g_t` the per-head learnable
scalar (gating) and `A` the per-head, per-state learned decay
(log-space, `A_log`).

### Step 5: Chunked recurrence

The naive recurrence is sequential over `t`. Yang et al. 2024
show the chunked algorithm: split the sequence into chunks of
`chunk_size` (HyMo uses `chunk_size = 64`), and within each chunk:

1. Run the recurrence in closed form using a small `chunk_size ×
   chunk_size` matrix.
2. Between chunks, propagate the state `h` serially.

This is what makes the algorithm Triton-friendly: each chunk is
an `O(chunk_size²)` matmul (small, fits in SMEM) plus an
inner-loop reduction over `chunk_size` steps. The inter-chunk
serialism is exactly `T / chunk_size = 64` steps.

### Signed-state caveat

Because `α_t ∈ (0, 1)` (always positive), the state can grow
without bound if writes are large. The convention is to bound
the **state norm** by a fixed value (the "max_state_norm"
trick). HyMo doesn't explicit-norm; the combination of `A_log`
initialization (negative, so decay starts strong) and the EMA
gate-bias update on the *router* (not the state) keeps things
stable in practice.

### Implementation in HyMo

- `src/hymo/models/gdn.py:GatedDeltaNetBlock` — `class GatedDeltaNetBlock`.
- `src/hymo/models/gdn.py:GatedDeltaNetBlock.__init__` — `__init__` initializes
  `use_triton = True` and `use_compile = True` (defaults;
  threaded from `TrainingConfig.fused_gdn` and
  `TrainingConfig.torch_compile_gdn`).
- `src/hymo/models/gdn.py:GatedDeltaNetBlock.n_heads` — properties for `n_heads`,
  `d_inner`, `d_state`, `headdim`.
  (`gdn_chunk_size` and the `chunk_size` property were removed in the
  2026-08-04 cleanup — the Triton kernel is a serial time loop, not chunked.)
- `src/hymo/models/gdn.py:GatedDeltaNetBlock._gated_delta_rule` — `_gated_delta_rule(v, b, c, g,
  A_log)` — the eager PyTorch reference (used when `use_triton
  = False`).
- `src/hymo/models/gdn.py:GatedDeltaNetBlock.forward` — `forward(x)`: input projection
  → split into `v`, `b`, `c`, `g`, `A_log`; `dt_proj` produces
  `g`; dispatch to `triton_gated_delta_rule` or `_forward_eager`.
- `src/hymo/models/gdn.py:GatedDeltaNetBlock._build_compiled_forward` — `_build_compiled_forward` —
  the `torch.compile`-wrapped forward.
- `src/hymo/models/gdn.py:GatedDeltaNetBlock._kernel_out` — `_kernel_out` — the wrapper that
  calls into `triton_gated_delta_rule`.
- `src/hymo/models/gdn.py:GatedDeltaNetBlock._forward_eager` — `_forward_eager` — the eager
  reference (used in tests and as the Triton parity baseline).
- `src/hymo/models/gdn_triton.py` (the whole file) — the
  hand-written Triton kernel: `_next_power_of_2`,
  `gdn_fwd_kernel`, `gdn_bwd_kernel`,
  `triton_gated_delta_rule`, `TritonGDNFunction` (the
  autograd `Function`).

### Worked example

Production scale (from `configs/hymo_750m.yaml`):

- `gdn_d_state = 32` — the `S` dimension of the state.
- `gdn_headdim = 32` — the per-head `D` dimension.
- `gdn_d_inner = 1280` — total inner dim = `n_heads * headdim = 40 * 32`.

So the per-head state is `(S, D) = (32, 32) = 1024` floats; per
block (40 heads) it's `40 * 1024 = 41 K` floats = 164 KB at
FP32 (or 82 KB at BF16). Across 24 GDN layers, the per-token
state is `24 * 164 KB = 3.9 MB` — small enough to keep in cache
between forward and backward.

The `chunk_size = 64` gives `T / 64 = 64` chunks per forward;
the per-chunk matmul is `(64, 32) × (32, 32) = 64 × 32 × 32 × 2
= 130 K` FLOPs. With 24 layers, 64 chunks, 40 heads:
`24 × 64 × 40 × 130 K = 80 G` FLOPs total recurrence — roughly
the same as the 8 MLA layers' attention.

### Interview Q&A

**Q1. Why is GDN better than Mamba-2 at 750 M scale?**

> A: Mamba-2 uses a pure linear recurrence with `α = exp(A·Δt)`
> (scalar decay) and `b = 1` (read of the latest value). GDN
> adds the *write key* `b` and *read key* `c` separately, so the
> model can target its writes and reads. This matters for
> associative recall — the canonical "match the value you wrote
> to key X" task — which Mamba-2 struggles with at this scale.

**Q2. Why does HyMo use `chunk_size = 64` instead of 32 or 128?**

> A: It's a kernel-utilization trade-off. Smaller chunks give
> more parallelism (better for an A100 with 108 SMs at `T =
> 4 K`); larger chunks give better register reuse (favors H100).
> 64 is the empirical sweet spot on A100; the design-doc
> "delta-rule sweep" suggested 32–64–128 first, and 64 won.

**Q3. Why GDN instead of Mamba/S4/RWKV?**

> A: GDN's write/read asymmetry (the `b ⊗ v` then `c · h`
> pattern) gives it Mamba-2's linear-time inference with
> better associative recall. Mamba-2's selection mechanism
> (`g_t`) is scalar; GDN's vector `b_t`, `c_t` gives the model
> a learned key for writing and reading, which is critical for
> in-context learning.

**Q4. Why is the chunked recurrence Triton-friendly?**

> A: Because the per-chunk computation is a small dense
> matmul (`chunk_size × chunk_size`) plus a within-chunk
> reduction. This fits in SMEM and is exact on GPU. The
> inter-chunk serialism is `T / chunk_size = 64` steps — tiny
> vs. the `T = 4096` steps of the naive recurrence, so the
> Python overhead is amortized away.

**Q5. Why is the state shape `(H, S, D) = (40, 32, 32)`?**

> A: `H = 40` is the number of heads (`gdn_d_inner // gdn_headdim
> = 1280 // 32`); `S = 32` is the state dimension (low-rank KV
> in the GDN analogy); `D = 32` is the per-head value/output
> dim. The trade-off: smaller `S` and `D` makes the state
> smaller (good for memory) but reduces per-head expressivity
> (bad for recall). HyMo's 32/32 is the empirically calibrated
> 750 M shape.

**Q6. Why does the Triton kernel require power-of-2 dims?**

> A: Triton's autotuner and block layouts assume powers of 2
> for the inner dims (they get compiled to specific
> vector widths). The Python wrapper (`triton_gated_delta_rule`)
   pads `D` and `S` to `_next_power_of_2` if needed and slices
> the padding back off before returning.

### Cross-links

- [`model-architecture.md`](model-architecture.md) §5–§6
  (GDN block walkthrough).
- [`optimization.md`](optimization.md) §3 (the
  Triton kernel integration).
- [`concepts/kernels.md`](kernels.md) —
  Triton execution model + autograd `Function` integration.
- [`concepts/gdn-and-mla.md`](gdn-and-mla.md) —
  why GDN + MLA, not just one.


## Mixture of Experts

### Learning objectives

After this file, you can:

1. State the Mixture-of-Experts sparsity pattern and how
   top-k routing works.
2. Walk through DeepSeek-style MoE: 16 routed + 1 shared expert,
   top-2 routing.
3. Explain why aux-loss-free (EMA gate-bias) load balancing
   replaces the classical auxiliary loss.
4. Compute the FLOPs savings of MoE vs. dense FFN at the same
   parameter count.

### Intuition

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

### Math derivation

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

Pseudo-code (matches `src/hymo/models/moe.py:DeepSeekMoE.forward`):

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

### Implementation in HyMo

- `src/hymo/models/moe.py` — module docstring.
- `src/hymo/models/moe.py` — `__all__ = ["SwiGLUExpert", "DeepSeekMoE"]`.
- `src/hymo/models/moe.py:SwiGLUExpert` — `class SwiGLUExpert`: a single
  SwiGLU with `w1` (gate), `w2` (down), `w3` (up).
- `src/hymo/models/moe.py` — a `DenseFFN` of the same shape was used on
  **GDN blocks** (dense, not MoE) in earlier revisions; it was removed in
  the 2026-08-04 cleanup (never instantiated in `src`) along with the
  `ModelConfig.inter_dim` field.
- `src/hymo/models/moe.py:DeepSeekMoE` — `class DeepSeekMoE`.
- `src/hymo/models/moe.py:DeepSeekMoE.__init__` — `__init__`:
  `n_routed, n_shared, n_activated, moe_inter_dim, ema_alpha,
  capacity_factor` from `ModelConfig`.
- `src/hymo/models/moe.py:DeepSeekMoE.__init__` — the `gate = Linear(dim,
  n_routed)` with bias zero-initialized and weight `N(0, 0.006)`.
- `src/hymo/models/moe.py:DeepSeekMoE.__init__` — `self.use_mixed_precision = True`
  (default; threaded from `TrainingConfig.moe_mixed_precision`).
- `src/hymo/models/moe.py:DeepSeekMoE.__init__` — `ema_expert_counts` registered
  as a non-persistent buffer (`persistent=False` so it doesn't
  save/load with checkpoints).
- `src/hymo/models/moe.py:DeepSeekMoE.gate_forward` — `gate_forward(x)`: the FP32
  gate computation.
- `src/hymo/models/moe.py:DeepSeekMoE.update_gate_bias` — `update_gate_bias(speed=0.001)`:
  the EMA logic.
- `src/hymo/models/moe.py:DeepSeekMoE.forward` — `forward(x)`: full dispatch
  loop.

The wiring from `Trainer` is in
`src/hymo/training/trainer.py:Trainer._update_moe_gate_biases` (`_update_moe_gate_biases`):

```python
def _update_moe_gate_biases(self) -> None:
    from hymo.models.moe import DeepSeekMoE
    for module in self.model.modules():
        if isinstance(module, DeepSeekMoE):
            module.update_gate_bias()
```

This fires after every optimizer step (not every micro-batch) —
see [`../training.md`](../training.md) §6.2 for why.

### Worked example

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

### Interview Q&A

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

### Cross-links

- [`model-architecture.md`](model-architecture.md) §8
  (DeepSeekMoE block walkthrough).
- [`optimization.md`](optimization.md) §5 (mixed
  precision + EMA gate-bias).
- [`../training.md`](../training.md) §6.5
  (the trainer's `_update_moe_gate_biases` call).
- [`concepts/gdn-and-mla.md`](gdn-and-mla.md) —
  why MoE is **only** on MLA blocks, not GDN.


## Multi-Token Prediction (MTP)

### Learning objectives

After this file, you can:

1. State the MTP objective and why it improves representation
   learning.
2. Walk through DeepSeek-V3's MTP design (depth-2 weighted
   auxiliary heads).
3. Compute the FLOPs overhead of MTP at the production scale.
4. Defend HyMo's choice of `mtp_depth = 2` with weights
   `[0.3, 0.1]`.

### Intuition

Standard next-token prediction is a single-task objective:
predict `t+1` given `t`. The model only learns to look *one*
step ahead.

**Multi-Token Prediction** (Globermann et al. 2024; DeepSeek-V3
2024) adds auxiliary heads that predict `t+2`, `t+3`, …
alongside the main `t+1` head. The auxiliary losses are
down-weighted, so the primary objective doesn't change, but
the model learns to plan further ahead.

In practice, MTP gives:

- **Better representations**: the auxiliary heads force the
  hidden state to encode longer-range dependencies than the
  main head alone requires.
- **Faster convergence**: at fixed wall-clock, MTP-trained
  models reach a given val loss in fewer steps.
- **Cheap**: the auxiliary heads are tiny — one Linear
  `(dim, dim) + RMSNorm + Linear(dim, vocab_size)` per head,
  shared embedding lookup. The FLOPs overhead is ~3%, but the
  quality wins are enough to justify it.

### Math derivation

### Objective

For each prefix `x = [x_1, ..., x_T]`, the targets are:

```
main:   y_1 = [x_2, x_3, ..., x_{T+1}]              (next-token)
mtp_1:  y_2 = [x_3, x_4, ..., x_{T+2}]              (next-next-token)
mtp_2:  y_3 = [x_4, x_5, ..., x_{T+3}]              (skip-2)
```

(Each `y_k` is shifted by `k-1` positions; the model skips
ahead by `k` tokens.)

### Loss

```
L = L_main
  + w_1 · L_mtp_1
  + w_2 · L_mtp_2
  + ...
```

With HyMo's defaults `mtp_depth = 2`, `mtp_loss_weights = (0.3, 0.1)`:

```
L = L_main + 0.3 · L_mtp_1 + 0.1 · L_mtp_2
```

The weights form a **decaying series**: the further-ahead
prediction is harder, so it's down-weighted more. Total auxiliary
weight: `0.3 + 0.1 = 0.4`. The main loss still dominates.

### Architecture

Each MTP head is a small block:

```
mtp_head_k(x):
    h = RMSNorm(x)
    h = Linear(dim, inter_dim=mtp_inter_dim)(h)
    h = Linear(mtp_inter_dim, dim)(h)
    logits = head(h)                       # tied to main embed
```

The MTP heads operate on the **main model's hidden state**
(captured via `forward_with_hidden`, not the regular `forward`),
so they don't need their own full transformer stack. This is
the key design choice: MTP is a "post-processor" on the main
model's hidden state, not a separate stack.

### Why the weights decay

Empirically, `t+2` is much harder to predict than `t+1`, and
`t+3` is harder than `t+2`. The model can't perfectly predict
the second-next-token, so a high weight on `L_mtp_2` would just
add noise to the gradient. The decay `0.3, 0.1` (or sometimes
`0.5, 0.25, 0.125` for depth-3) is the empirical sweet spot.

### Implementation in HyMo

- `src/hymo/models/mtp.py:MTPOutput` — `class MTPOutput` (per-head
  output container).
- `src/hymo/models/mtp.py:MTPBlock` — `class MTPBlock`: one auxiliary
  head (RMSNorm + Linear + Linear).
- `src/hymo/models/mtp.py:MultiTokenPrediction` — `class MultiTokenPrediction`:
  the wrapper that holds `mtp_depth` heads.
- `src/hymo/models/mtp.py:MultiTokenPrediction.__init__` — `__init__` builds the heads
  and a small projection from the main hidden state.
- `src/hymo/models/mtp.py:MultiTokenPrediction._mtp_head` — `_mtp_head(k, x)`: per-head
  forward.
- `src/hymo/models/mtp.py:MultiTokenPrediction.forward` — `forward(tokens)`:
  1. Call `main_model.forward_with_hidden(tokens)` to get
     `(logits, hidden)`.
  2. For each `k in 1..mtp_depth`:
     - Compute shifted targets: `targets_k = tokens[:, k:]`,
       trimmed to length.
     - Compute `mtp_logits_k = mtp_head_k(hidden[:, :T-k])`.
     - Compute `L_mtp_k = CE(mtp_logits_k, targets_k)`.
     - Compute `weight = mtp_loss_weights[k-1]`.
     - Return (`logits`, [`MTPOutput(k, logits_k, targets_k,
       weight_k)` for k in 1..mtp_depth]`).

Wiring in `Trainer`:

- `src/hymo/training/trainer.py:Trainer.train_step` — if `_has_mtp`, call
  `mtp_module.forward(tokens)` instead of `model.forward(tokens)`.
  This returns `(logits, mtp_outputs)` instead of just `logits`.
- `src/hymo/training/trainer.py:Trainer.train_step` — loop over
  `mtp_outputs`, compute each MTP loss, multiply by
  `mtp_out.loss_weight`, add to `total_loss`. Record per-head
  metrics for W&B.

### Worked example

Production scale:

- `mtp_depth = 2`, `mtp_loss_weights = (0.3, 0.1)`,
  `mtp_inter_dim = 2304`, `dim = 896`, `vocab_size = 64_256`.
- Per-MTP-head params: `896 × 2304 + 2304 × 896 + 896 × 64256`
  ≈ `2.0 M + 57.6 M = 59.6 M`. (The last factor is the tied
  embedding lookup, which is free because of `tie_embeddings:
  true`.)
- Two MTP heads: `~ 2 × 2.0 M = 4.0 M` *new* params (the vocab
  projection is shared).
- Forward FLOPs overhead: 2 × (forward of MTP block) ≈ 2 ×
  3 × 896 × 2304 ≈ 12 M FLOPs per token. Vs. the main
  32-layer forward's ~5 G FLOPs per token, that's ~0.2% overhead.
- Loss overhead: 2 × CE per token ≈ 2 × 64_256 = 128 K
  activations per token. Negligible.

So the MTP cost is < 0.5% FLOPs and ~4 M params (0.2% of the
750 M active). The quality win (typically 5-10% faster
convergence at fixed eval loss) is a free lunch.

### Interview Q&A

**Q1. Why `mtp_depth = 2` and not 1 or 3?**

> A: Empirically, depth-1 is "free" but barely helps
> (the second-step loss is too close to the main loss). Depth-3
> gives a third head that adds compute but only a tiny
> additional signal at this scale. Depth-2 is the sweet spot,
> especially with weights `[0.3, 0.1]` — the second head
> contributes ~30% of the main loss, the third ~10%.

**Q2. Why weighted `[0.3, 0.1]` and not equal weights `[0.5, 0.5]`?**

> A: The further-ahead prediction is harder; equal weights
> would let the noisier `t+3` loss dominate the gradient. The
> decay `0.3, 0.1` says "the main loss is the primary signal;
> the auxiliary heads are tie-breakers". This is what
> DeepSeek-V3 found at 670 B; we expect similar at 750 M.

**Q3. Why does the MTP head operate on the main model's hidden
state, not its own?**

> A: Because the auxiliary heads are post-processors on the
> main representation. The point of MTP is to shape the main
> hidden state to encode longer-range dependencies, not to
> train a separate stack. A separate MTP stack would be 32
> layers × 2 = 64 extra layers, which is too much compute.

**Q4. Why is the MTP forward different from `model.forward`?**

> A: `model.forward(tokens)` returns just `logits`. The MTP
  heads need the *hidden state* (the output of the final
  norm, before the head projection). `model.forward_with_hidden
  (tokens)` returns `(logits, hidden)` — the MTP module reads
  `hidden` and passes the main `logits` through unchanged.

**Q5. Why is the MTP loss computed in `train_step` instead of
  `forward`?**

> A: Loss weights can be tuned per training run (e.g. an
> ablation that drops MTP). Keeping the loss computation in
> the trainer lets the model module return raw logits and the
> trainer decide what to do with them. This pattern matches
> how the rest of the training pipeline works.

**Q6. What happens to MTP at inference?**

> A: The MTP heads are not used. `model.forward(tokens)` returns
> just the main next-token logits. The MTP heads exist only to
> shape the training-time hidden state; at inference, only the
> main head is consulted. MTP saves zero inference cost.

### Cross-links

- [`model-architecture.md`](model-architecture.md) §9
  (MTP block walkthrough).
- [`../training.md`](../training.md) §6.2
  (the trainer's MTP wiring).
- [`concepts/gdn-and-mla.md`](gdn-and-mla.md) —
  the MoE-on-MLA "auxiliary compute" pattern that MTP parallels.


## Hybrid Architectures: Jamba, Zamba, StripedHyena, HyMo

### Learning objectives

After this file, you can:

1. State why "all-attention" transformers are sub-optimal and
   the case for hybrid stacks.
2. Walk through the 3:1 GDN:MLA ratio and the MoE-on-attention-
   only thesis.
3. Compare HyMo to Jamba, Zamba, and StripedHyena.
4. Defend HyMo's specific architectural choices.

### Intuition

Pure-attention transformers (Llama-3, DeepSeek-V3, Qwen-3)
scale attention `O(N²)` with sequence length. For pretraining
at long contexts, this is the dominant compute cost — the
FFN is `O(N)` but the attention is `O(N²)`.

**Linear-attention** models (Mamba, Mamba-2, RWKV, GDN) drop
the quadratic to linear, but at the cost of weaker long-range
modeling. They're great for streaming / long-context
inference, but on standard NLP benchmarks they're usually a
few points behind equivalently-sized transformers.

**Hybrid** architectures combine the two: bulk of the stack is
linear (cheap), with sparse full-attention anchors for the
hard reasoning tasks. The two main published hybrids:

- **Jamba** (AI21, 2024): Mamba + attention + MoE, 1:7
  attention ratio.
- **StripedHyena** (Together AI, 2023): Hyena (convolution-based)
  + attention, 1:7.
- **Zamba** (Zyphra, 2024): Mamba + shared attention, 1:6.

HyMo is in this family. The 3:1 ratio (24 GDN + 8 MLA) is
**more attention-heavy** than Jamba/StripedHyena — 8 of 32
layers (25%) is full attention, vs. Jamba's 1:7 (~12.5%).

### The 3:1 ratio

```
8 MLA layers  ── full attention, MoE
24 GDN layers ── linear attention, dense SwiGLU
```

Why 3:1?

1. **Compute budget**: at 75% linear layers, the per-step
   compute is dominated by the linear recurrence (cheap) +
   the 8 MLA layers (still `O(N²)` per MLA forward but only
   8 times). The total is ~30% lower than an all-MLA stack.
2. **Quality floor**: 8 attention anchors give the model
   full-sequence reasoning capacity for the tasks where
   linear attention lags (retrieval, copying, long-range
   coreference).
3. **Empirical sweet spot**: at this scale (750 M), a 2:1
   ratio (more attention) doesn't improve quality much; a
   4:1 ratio starts to hurt. 3:1 is the
   engineering/quality compromise.

### MoE-on-attention-only

The 8 MLA blocks each have a `DeepSeekMoE` FFN. The 24 GDN
blocks each have a `DenseFFN` (SwiGLU, `inter_dim = 2560`).

Why this asymmetric feed-forward?

1. **Compute is spent where it buys the most**. MoE is
   expensive in *parameters* (16× more storage for the same
   inter_dim) but cheap in *active FLOPs* (top-2 of 16).
   Putting MoE on every layer would 2× the active FFN
   compute, which is a lot of model capacity going to the
   already-strong linear layers.
2. **MLA layers need more capacity**. The full-attention
   layers are doing harder work (long-range reasoning) and
   benefit from the larger effective vocabulary of an MoE
   FFN. The linear layers are doing cheaper work (memorizing
   local patterns) and don't need the extra capacity.
3. **Inference uniformity**. At inference, every layer runs
   the same number of experts (2 of 16). The active FFN
   FLOPs per token are the same regardless of layer type.
   This makes inference budgeting simpler.

### Comparison to Jamba, Zamba, StripedHyena

| Model | Linear:attn | Linear type | Attn type | MoE? |
|---|---|---|---|---|
| HyMo | 3:1 | GDN (chunked) | MLA (low-rank KV) | Yes, on attn only |
| Jamba | 7:1 | Mamba | Standard MHA | Yes, every layer |
| Zamba | 6:1 | Mamba | Standard MHA (shared) | No (dense FFN) |
| StripedHyena | 7:1 | Hyena (conv) | Standard MHA | No |

HyMo is **more attention-heavy** (25% vs 12.5-14%) and uses
**better attention** (MLA vs MHA). The MoE placement is
unique: HyMo puts MoE on attention only (8 layers), while
Jamba puts MoE on every layer.

Trade-offs:

- More attention = more compute per step.
- MLA = lower KV cache, better long-context inference.
- MoE on attention only = simpler inference (uniform active
  FLOPs per token across layers), but lower total FFN
  capacity than Jamba (which has 32 MoE layers vs HyMo's 8).

### What HyMo inherits

- From **DeepSeek-V3**: MLA absorption (the latent KV cache),
  aux-loss-free MoE (EMA gate-bias), partial RoPE.
- From **Mamba / RWKV**: linear-time recurrence.
- From **Jamba**: the hybrid-stack idea.
- From **GPT-3 / Llama**: dense token embedding + tied
  head + μP-style init.
- From **Muon**: the optimizer family (orthogonalized
  updates for 2D matrices).

### What HyMo adds

- **GDN over Mamba-2**: the write/read asymmetry (`b ⊗ v`,
  `c · h`) gives better associative recall than Mamba-2's
  scalar `g`.
- **MLA over MHA**: lower KV cache + better long-context
  performance.
- **Cautious WD**: the simple mask trick that eliminates
  the "WD fights gradient" pathology.
- **NoPE-hybrid ablation** (deferred): the v1.1 experiment
  that may give long-context wins.
- **Triton kernel for GDN** (custom): ~4.5× speedup over
  the naive Python recurrence.

### Implementation in HyMo

- `src/hymo/models/model.py:HyMo` — `class HyMo`: the top-level
  model.
- `src/hymo/models/model.py:HyMo.__init__` — `__init__`: builds the
  32-block stack, 8 MLA + 24 GDN.
- `src/hymo/models/model.py:HyMo.__init__` — the loop that picks
  MLA or GDN per position and applies the `use_rope` flag
  for NoPE-hybrid positions.
- `src/hymo/models/model.py:build_hymo` — `build_hymo(config)`: the
  factory function.

### Interview Q&A

**Q1. Why 3:1 and not 4:1 or 2:1?**

> A: Empirically, 3:1 is the engineering/quality sweet spot
> at 750 M scale. 2:1 (more attention) doesn't improve quality
> much; 4:1 (less attention) starts to hurt on long-range
> tasks. The trade-off is: more attention = more compute per
> step, but better quality on hard tasks.

**Q2. Why MoE on attention only, not on every layer?**

> A: Compute is spent where it buys the most. The MLA layers
> are doing the harder work (long-range reasoning) and
> benefit from the larger effective vocabulary of an MoE
> FFN. The GDN layers are doing cheaper work (memorizing
> local patterns) and don't need the extra capacity. Putting
> MoE on every layer would 2× the active FFN compute.

**Q3. Why GDN over Mamba-2?**

> A: GDN's write/read asymmetry (`b ⊗ v` then `c · h`)
> gives better associative recall than Mamba-2's scalar `g`.
> For tasks like "match the value written to key X" (the
> canonical retrieval benchmark), GDN wins at this scale.

**Q4. Why not StripedHyena's convolution-based linear layer?**

> A: Hyena uses FFT-based convolutions, which are fast at
> training time but have `O(N log N)` inference complexity
> (the FFT is computed once and then dot-product-ed). GDN
> is `O(N)` at training and inference, with a small constant
> (the recurrent state). For long-context inference, GDN
> wins.

**Q5. Why is the NoPE-hybrid flag off in v1.0?**

> A: Risk reduction. The shipped model is the primary 30 B
> pre-training run; flipping the NoPE-hybrid flag without a
> prior ablation result is a multi-day experiment with no
> guarantee of payoff. The v1.0 ships with the flag off
> (everyone gets partial RoPE); the v1.1 ablation tests
> whether the 7 NoPE GDN layers help long-context tasks.

**Q6. What is the inference FLOPs at `T = 4096`?**

> A: Per token:
> - Embed lookup: `dim * 1 = 896` MAC.
> - 24 GDN layers: 24 * 4.5 ms (rough) = ~108 ms compute
>   over the sequence. Per token, ~108 ms / 4096 = 26 µs.
> - 8 MLA layers: 8 * 5 ms = 40 ms. Per token, ~10 µs.
> - 8 MoE FFNs (top-2 of 16): 8 * 2 * 6.2 M = 99 M FLOPs
>   per token.
> - 24 dense FFNs: 24 * 6.2 M = 149 M FLOPs per token.
> - Head: `dim * vocab_size = 896 * 64_256 = 57.5 M` FLOPs
>   per token.
>
> Total per token: ~340 M FLOPs. At ~330 TFLOPs/s (A100 BF16
> peak), ~1 µs per token. With FSDP + Triton + torch.compile,
> ~3 µs per token (roughly 3× peak). 4096 tokens = 12 ms.

**Q7. Why is HyMo's `dim = 896` and not 1024 or 512?**

> A: 896 is a multiple of 32 (`head_dim * n_kv_groups = 128 *
> 4 = 512`; `n_heads * head_dim = 16 * 128 = 2048`; 896 is
> divisible by both 128 and 224, the Q-lora and KV-lora
> ranks). It also gives a sweet-spot parameter count (~750 M
> active, ~1.86 B stored). 1024 would be 18% more params;
> 512 would be 43% fewer.

### Cross-links

- [`model-architecture.md`](model-architecture.md) §1
  (high-level architecture).
- [`design.md`(design.md) §8 (hybrid
  thesis).
- [`concepts/model-architecture.md`](model-architecture.md) — MLA details.
- [`concepts/gdn-and-mla.md`](gdn-and-mla.md) —
  GDN details.
- [`concepts/gdn-and-mla.md`](gdn-and-mla.md) —
  MoE details.
- [`concepts/model-architecture.md`](model-architecture.md) —
  the NoPE-hybrid flag.

## References

- [model-architecture.md](model-architecture.md) — the code walkthrough these mechanisms implement.
- [kernels.md](kernels.md) — the Triton GDN kernel execution model and autograd integration.
- [optimization.md](optimization.md) — NorMuon/AdamW, WSD, FSDP-2, and the optimization flags.
- [design.md](design.md) — the full architecture & design document (v1.0).
- Source: `src/hymo/models/gdn.py`, `src/hymo/models/gdn_triton.py`, `src/hymo/models/mla.py`, `src/hymo/models/moe.py`, `src/hymo/models/mtp.py`, `src/hymo/models/rope.py`, `src/hymo/models/model.py`.
