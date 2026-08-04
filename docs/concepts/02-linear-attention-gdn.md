# 02 — Linear Attention and the Gated Delta Net

> **Bridges to:** [`learning_docs/1_Model_Architecture.md`](../../learning_docs/1_Model_Architecture.md) §5–§6 (GDN
> block + Triton kernel)

## Learning objectives

After this file, you can:

1. Derive linear attention from softmax attention by replacing
   the softmax with a kernel feature map.
2. State the delta rule and explain why gating makes it the
   "Gated Delta Net".
3. Sketch the chunked recurrence and why it's a Triton-friendly
   algorithm.
4. Defend HyMo's choice of GDN over Mamba-2/S4/RWKV at this scale.

## Intuition

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

## Math derivation

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

## Implementation in HyMo

- `src/hymo/models/gdn.py:23` — `class GatedDeltaNetBlock`.
- `src/hymo/models/gdn.py:26` — `__init__` initializes
  `use_triton = True` and `use_compile = True` (defaults;
  threaded from `TrainingConfig.fused_gdn` and
  `TrainingConfig.torch_compile_gdn`).
- `src/hymo/models/gdn.py:78-94` — properties for `n_heads`,
  `d_inner`, `d_state`, `headdim`, `chunk_size`.
- `src/hymo/models/gdn.py:97` — `_gated_delta_rule(v, b, c, g,
  A_log)` — the eager PyTorch reference (used when `use_triton
  = False`).
- `src/hymo/models/gdn.py:123` — `forward(x)`: input projection
  → split into `v`, `b`, `c`, `g`, `A_log`; `dt_proj` produces
  `g`; dispatch to `triton_gated_delta_rule` or `_forward_eager`.
- `src/hymo/models/gdn.py:130` — `_build_compiled_forward` —
  the `torch.compile`-wrapped forward.
- `src/hymo/models/gdn.py:142` — `_kernel_out` — the wrapper that
  calls into `triton_gated_delta_rule`.
- `src/hymo/models/gdn.py:162` — `_forward_eager` — the eager
  reference (used in tests and as the Triton parity baseline).
- `src/hymo/models/gdn_triton.py:288` (whole file) — the
  hand-written Triton kernel: `_next_power_of_2` (line 30),
  `gdn_fwd_kernel` (43), `gdn_bwd_kernel` (94),
  `triton_gated_delta_rule` (237), `TritonGDNFunction` (the
  autograd `Function`).

## Worked example

Production scale (from `configs/hymo_750m.yaml`):

- `gdn_d_state = 32` — the `S` dimension of the state.
- `gdn_headdim = 32` — the per-head `D` dimension.
- `gdn_d_inner = 1280` — total inner dim = `n_heads * headdim = 40 * 32`.
- `gdn_chunk_size = 64` — the chunking for the recurrence.

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

## Interview Q&A

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

## Cross-links

- [`learning_docs/1_Model_Architecture.md`](../../learning_docs/1_Model_Architecture.md) §5–§6
  (GDN block walkthrough).
- [`learning_docs/4_Optimizations.md`](../../learning_docs/4_Optimizations.md) §3 (the
  Triton kernel integration).
- [`concepts/10-triton-kernels.md`](10-triton-kernels.md) —
  Triton execution model + autograd `Function` integration.
- [`concepts/11-hybrid-architectures.md`](11-hybrid-architectures.md) —
  why GDN + MLA, not just one.
