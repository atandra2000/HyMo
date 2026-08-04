# 11 — Hybrid Architectures: Jamba, Zamba, StripedHyena, HyMo

> **Bridges to:** [`learning_docs/1_Model_Architecture.md`](../../learning_docs/1_Model_Architecture.md) §1
> (high-level architecture); [`docs/HyMo-Design.md`](../../docs/HyMo-Design.md) §8 (hybrid thesis)

## Learning objectives

After this file, you can:

1. State why "all-attention" transformers are sub-optimal and
   the case for hybrid stacks.
2. Walk through the 3:1 GDN:MLA ratio and the MoE-on-attention-
   only thesis.
3. Compare HyMo to Jamba, Zamba, and StripedHyena.
4. Defend HyMo's specific architectural choices.

## Intuition

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

## The 3:1 ratio

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

## MoE-on-attention-only

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

## Comparison to Jamba, Zamba, StripedHyena

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

## What HyMo inherits

- From **DeepSeek-V3**: MLA absorption (the latent KV cache),
  aux-loss-free MoE (EMA gate-bias), partial RoPE.
- From **Mamba / RWKV**: linear-time recurrence.
- From **Jamba**: the hybrid-stack idea.
- From **GPT-3 / Llama**: dense token embedding + tied
  head + μP-style init.
- From **Muon**: the optimizer family (orthogonalized
  updates for 2D matrices).

## What HyMo adds

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

## Implementation in HyMo

- `src/hymo/models/model.py:20` — `class HyMo`: the top-level
  model.
- `src/hymo/models/model.py:23` — `__init__`: builds the
  32-block stack, 8 MLA + 24 GDN.
- `src/hymo/models/model.py:42-48` — the loop that picks
  MLA or GDN per position and applies the `use_rope` flag
  for NoPE-hybrid positions.
- `src/hymo/models/model.py:97` — `build_hymo(config)`: the
  factory function.

## Interview Q&A

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

## Cross-links

- [`learning_docs/1_Model_Architecture.md`](../../learning_docs/1_Model_Architecture.md) §1
  (high-level architecture).
- [`docs/HyMo-Design.md`](../../docs/HyMo-Design.md) §8 (hybrid
  thesis).
- [`concepts/01-attention.md`](01-attention.md) — MLA details.
- [`concepts/02-linear-attention-gdn.md`](02-linear-attention-gdn.md) —
  GDN details.
- [`concepts/03-mixture-of-experts.md`](03-mixture-of-experts.md) —
  MoE details.
- [`concepts/04-position-encoding.md`](04-position-encoding.md) —
  the NoPE-hybrid flag.
