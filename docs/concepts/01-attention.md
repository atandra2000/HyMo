# 01 — Attention: MHA → GQA → MQA → MLA

> **Bridges to:** [`learning_docs/1_Model_Architecture.md`](../../learning_docs/1_Model_Architecture.md) §4 (MLA)

## Learning objectives

After this file, you can:

1. Write down the multi-head attention operation and its
   `O(N²)` complexity in time and memory.
2. Derive MQA, GQA, and MLA from MHA, identifying exactly what
   each one compresses.
3. Explain why MLA's KV compression to a single low-rank vector
   per token beats MQA at long context.
4. Defend HyMo's choice of **MQA-4** (4 KV groups) over full
   MLA-style compression at the production scale.

## Intuition

Standard multi-head attention has every query head attend to every
key head. For inference with long context, the KV cache grows
linearly with sequence length and quadratically with head count —
the dominant memory cost of a transformer at decode time.

Three engineering responses, in order of increasing compression:

| Variant | What is shared | KV-cache size per token |
|---|---|---|
| **MHA** | nothing | `n_layers * n_heads * (head_dim + head_dim) * 2` |
| **GQA** | `n_kv_groups < n_heads` query heads share KV | MHA / `n_heads / n_kv_groups` |
| **MQA** | one KV head total | MHA / `n_heads` |
| **MLA** | a single low-rank `kv_lora_rank`-dim vector | `n_layers * kv_lora_rank * 2` (no `n_heads` factor) |

MLA is what DeepSeek-V2 introduced; it's the most aggressive —
it bypasses the head dimension entirely and stores one shared
low-rank latent per token, then per-head "absorption" matrices
reconstruct each query/key's view at attention time. The cost is
extra matmuls in the attention forward (which is cheap on GPU)
and a slightly different gradient signal (which is what made
absorbed MLA a careful research contribution).

HyMo uses **MQA-4** (4 KV groups for 16 query heads → 4 query
heads per KV group). It's the second-most aggressive of the four
options — significantly cheaper than MHA, but not as aggressive
as full MLA absorption. Why: at this scale (750 M active params),
the MQA-4 trade-off wins on quality-per-FLOPs, and the per-block
attention kernel is much simpler than the absorbed MLA kernel.

## Math derivation

### MHA

Given `X ∈ ℝ^{T × d}` (`d` = model dim), project to `Q, K, V`:

```
Q = X W_Q, K = X W_K, V = X W_V     ∈ ℝ^{T × n_heads × head_dim}
```

Attention output:

```
Y = softmax(Q Kᵀ / √head_dim) V     ∈ ℝ^{T × n_heads × head_dim}
```

Complexity per layer: `O(T² · head_dim · n_heads)` in time,
`O(T · n_heads · head_dim)` in KV cache size per token.

### MQA — `n_kv_groups = 1`

`K` and `V` are projected with a single shared head; `Q` keeps
`n_heads`. The KV cache size per token shrinks by `n_heads`,
but the attention math is unchanged.

### GQA — `n_kv_groups ∈ (1, n_heads)`

`K` and `V` are projected to `n_kv_groups` heads; `Q` is projected
to `n_heads` heads; each query head attends to one KV group (round-
robin assignment: query head `i` reads from KV group
`i // (n_heads // n_kv_groups)`).

KV cache size per token: `n_heads / n_kv_groups ×` MHA.

### MLA — DeepSeek-style

Project `X` once into a shared low-rank latent `c_t`:

```
c_t = X · W_KV_A                          ∈ ℝ^{T × kv_lora_rank}
k_t = c_t · W_K_B                         ∈ ℝ^{T × n_heads × qk_nope_head_dim}
v_t = c_t · W_V_B                         ∈ ℝ^{T × n_heads × v_head_dim}
```

Each query also has its own low-rank projection:

```
q_nope_t = X · W_Q_A · W_Q_B              ∈ ℝ^{T × n_heads × qk_nope_head_dim}
```

The "absorbed" trick: precompute `W_K_B` and `W_V_B` once, then
absorb them into the query projection, so the attention op sees
a single shared latent `c_t` plus per-head `q_t`. The KV cache
becomes `c_t` only — `T × kv_lora_rank` — not `T × n_heads ×
(qk_nope_head_dim + v_head_dim)`.

This is the innovation of MLA: the cache shrinks by another
factor of `n_heads × (qk_nope + v_head_dim) / kv_lora_rank`
(roughly `n_heads × 2 × head_dim / kv_lora_rank`).

### HyMo's choice — MQA-4

| Knob | Value |
|---|---|
| `n_heads` | 16 |
| `n_kv_groups` | 4 |
| `q_lora_rank` | 224 |
| `kv_lora_rank` | 128 |
| `head_dim` | 128 |
| `qk_rope_head_dim` | 32 |
| `qk_nope_head_dim` | 96 |
| `v_head_dim` | 128 |

These are **not** full MLA — there's no `W_K_B`/`W_V_B`
absorption. Instead, every query head reads from one of 4 KV
heads (round-robin: query heads {0,1,2,3} → KV head 0; query
heads {4,5,6,7} → KV head 1; …).

The `q_lora_rank = 224` and `kv_lora_rank = 128` *are* full MLA:
the query is first projected to 224 dim, then up-projected to
`n_heads × qk_nope_head_dim`; the KV is projected to a shared
128-dim latent per token and broadcast to the 4 KV heads. The
KV-cache win is from `kv_lora_rank = 128` instead of
`4 × 128 = 512`.

So HyMo is **"MQA-4 with low-rank KV compression"** — MQA-style
KV sharing plus the MLA-style latent. Pure MLA absorption (the
trick that lets you store *only* `c_t`) is what was deferred; at
this scale, the simpler kernel wins.

## Implementation in HyMo

- `src/hymo/models/mla.py:21` — `class MultiHeadLatentAttention`
  (the projection class).
- `src/hymo/models/mla.py:29` — `__init__` with the 8 MLA config
  fields used.
- `src/hymo/models/mla.py:72-88` — properties `n_heads`,
  `n_kv_groups`, `qk_rope_head_dim`, `qk_nope_head_dim`,
  `v_head_dim`.
- `src/hymo/models/mla.py:91` — `forward(x)`: low-rank Q → split
  into RoPE / NoPE parts; KV → 4 groups with partial RoPE on the
  first `qk_rope_head_dim = 32` of `head_dim`; attention; output
  projection.
- `src/hymo/models/mla.py:143` — `class MLABlock`: pre-norm +
  MultiHeadLatentAttention + residual.
- `src/hymo/models/mla.py:146` — `__init__(config, layer_idx)`:
  builds the block; the `use_cuda_graphs` flag is set as an
  attribute at `mla.py:160` (default True, threaded from
  `TrainingConfig.cuda_graphs_mla`).
- `src/hymo/models/mla.py:162` — `forward(x)`: full block
  forward, including the soft-cap (no — the softcap is on the
  logits, not the attention output; see `model.py:90`).

## Worked example

HyMo at production scale:

- 16 query heads × 128 head_dim = 2048 query projection width.
- 4 KV heads × 128 = 512 KV projection width (or via latent:
  128-dim shared latent broadcast to 4 heads).
- Attention FLOPs per layer per token:
  `2 * (q_dim * k_dim + q_dim * v_dim) ≈ 2 * (2048 * 512) = 2.1 M`
  FLOPs, plus the `O(T²)` softmax matmul.
- 8 MLA layers per forward, `T = 4096`, batch 4: ~`2.1 M * 4096 *
  8 = 69 G` FLOPs for QK + softmax.

The shared latent saves ~`4×` on KV cache size: from
`(4096 * 16 * 256 * 2)` (MHA, both K and V) bytes down to
`(4096 * 128 * 2) = 1 MB` per token — small but compounded
across the 8 MLA layers at inference time.

## Interview Q&A

**Q1. Why does MLA beat MQA at long context?**

> A: MQA collapses all KV heads into one; MLA collapses all KV
> heads into one **plus** projects through a low-rank bottleneck
> that drops the head dim entirely. For very long contexts
> (`T > 32 k`), the bottleneck gives another factor of ~`head_dim
> / kv_lora_rank` in cache size, which matters for inference
> memory.

**Q2. Why didn't HyMo go full MLA?**

> A: Two reasons. First, at 750 M active params the absorbed-MLA
> kernel (which needs extra matmuls to project through the
> low-rank bottleneck at attention time) is not a clear win on
> quality-per-FLOPs vs. MQA-4. Second, the absorbed-MLA kernel
> is harder to write in pure PyTorch + Triton than the MQA-4
> kernel; HyMo's kernel budget didn't include it. The
> `kv_lora_rank = 128` field is set up so a future v1.2 could
> flip to full MLA absorption without changing the config shape.

**Q3. Why is the partial RoPE only on the first 25% of
`head_dim`?**

> A: Empirically, RoPE on the full `head_dim` works but is
> over-parameterized; 25% gives the position-aware channels
> enough rope to learn relative-position patterns without
> burning 75% of the head dim on position encoding (which
> would otherwise compete with content attention). See
> `04-position-encoding.md`.

**Q4. What does "absorbed" mean in MLA?**

> A: The trick that the latent `c_t` can be the only thing
> stored in the KV cache (instead of per-head `k_t`, `v_t`).
> The per-head projections `W_K_B`, `W_V_B` are folded into the
> query projection at training time, so at inference, attention
> reads `c_t` directly and the per-head `k_t`, `v_t` are
> reconstructed on the fly. HyMo doesn't do this folding yet;
> see `learning_docs/4_Optimizations.md` for the future work.

**Q5. MQA-4 vs GQA-1.75 — what's the difference?**

> A: GQA-1.75 (an earlier draft of HyMo's config) means the
> ratio `n_heads / n_kv_groups = 1.75` — but `1.75` isn't an
> integer, so it was a placeholder for "not yet decided".
> MQA-4 is `n_kv_groups = 4`, so the ratio is `16 / 4 = 4`.
> Ablation family D (`D_mqa4_vs_gqa175`) compares these.

## Cross-links

- [`learning_docs/1_Model_Architecture.md`](../../learning_docs/1_Model_Architecture.md) §4 — the
  MLA block walkthrough.
- [`concepts/04-position-encoding.md`](04-position-encoding.md) —
  partial RoPE on the first 25% of `head_dim`.
- [`concepts/11-hybrid-architectures.md`](11-hybrid-architectures.md) —
  why MLA + GDN, not just MLA.
