# HyMo — The 3:1 GDN:MLA Hybrid Ratio

> **Audience:** intermediate. Assumes you know what attention and a feed-forward
> block are; the mechanism deep-dives live in [`gdn-and-mla.md`](gdn-and-mla.md)
> and the line-by-line walkthrough in [`model-architecture.md`](model-architecture.md).
> This doc answers one question: *why is the layer mix 24 GDN to 8 MLA, and what
> does each half buy?*

## Where the ratio lives

The 3:1 mix is not a list written down in two places — it is *derived* from one
number. `src/hymo/core/config.py:ModelConfig` stores only `n_layers` and derives
the schedule through properties:

- `src/hymo/core/config.py:ModelConfig.n_mla_layers` returns `n_layers // 4`
  (32 // 4 = 8), and `src/hymo/core/config.py:ModelConfig.n_gdn_layers` is the
  remainder (24).
- `src/hymo/core/config.py:ModelConfig.mla_positions` places an MLA block at
  every 4th index — `frozenset(0, 4, 8, …, 28)`. Every other index is GDN
  (`src/hymo/core/config.py:ModelConfig.gdn_positions`).
- `src/hymo/models/model.py:HyMo.__init__` walks `i in range(config.n_layers)`
  and builds an `src/hymo/models/mla.py:MLABlock` when `i in mla_positions`,
  else an `src/hymo/models/gdn.py:GatedDeltaNetBlock`. There is no second,
  hand-maintained layer list to drift out of sync — the class docstring on
  `src/hymo/core/config.py:ModelConfig` says exactly this.

```python
# illustrative — the placement rule, simplified from the two methods above
mla_positions = {i * 4 for i in range(n_layers // 4)}   # 0, 4, …, 28
for i in range(n_layers):
    layer = MLABlock(...) if i in mla_positions else GatedDeltaNetBlock(...)
```

Because the derivation is arithmetically forced, `n_layers = 32` *is* the
3:1 ratio; changing one config number changes the mix coherently.

## What GDN buys

The 24 GDN blocks run a gated delta-rule recurrence
(`src/hymo/models/gdn.py:GatedDeltaNetBlock._gated_delta_rule` is the eager
reference; the CUDA path is the hand-written kernel in
[`kernels.md`](kernels.md)):

1. **Linear time, constant per-token state.** The recurrence keeps a
   fixed-size state per head — at the production shape 40 heads × (32 × 32)
   state — instead of a KV cache that grows with sequence length. Streaming
   and long-context decode do not slow down as context grows.
2. **Write/read asymmetry.** The state update writes with a learned key `b`
   and reads with a learned key `c`, with a learned per-head decay. This is
   what GDN adds over Mamba-2's scalar decay, and it is the mechanism behind
   associative recall (write once, read by key later).
3. **Cheap FLOPs per layer.** A GDN block is projections + a recurrence —
   no quadratic attention matrix, and (see below) no feed-forward tower.

A detail that matters for the ratio math: a GDN block is **recurrence-only**.
`src/hymo/models/gdn.py:GatedDeltaNetBlock.__init__` builds projections
(`in_proj`, `conv1d`, `b_proj`, `c_proj`, `dt_proj`, `g_proj`, `out_proj`,
`skip_proj`) and the learned `A_log` / `dt_bias` / `D` scalars — and nothing
else. There is no FFN or MoE on a GDN layer; all feed-forward capacity in
HyMo lives on the 8 MLA blocks. See
[`asymmetric-moe.md`](asymmetric-moe.md) for why.

## What MLA buys

The 8 MLA blocks give exact softmax attention with a compressed KV
representation (`src/hymo/models/mla.py:MultiHeadLatentAttention.forward`):
queries and keys are split into a RoPE slice and a NoPE slice, keys/values
come from a 128-dim latent (`kv_lora_rank`), and grouped SDPA
(`n_heads = 16` over `n_kv_groups = 4`) runs with `is_causal=True`. The block
residual structure is two pre-norm updates — attention, then MoE — in
`src/hymo/models/mla.py:MLABlock.forward`.

What exact attention buys that the recurrence does not: lossless access to
any earlier token. Linear-attention state is a fixed-size summary; retrieval
tasks ("find the token I defined three pages ago") are exactly where it
degrades. The MLA anchors are the retrieval hardware.

## Why 3:1 and not another ratio

The honest accounting has three parts:

1. **Compute placement.** At 25% full-attention layers, the quadratic term is
   confined to 8 of 32 blocks while the other 24 run in linear time. [INFERENCE]
   The corpus's working assumption (see [`gdn-and-mla.md`](gdn-and-mla.md),
   "Hybrid Architectures") is that this lands the per-step compute well below
   an all-MLA stack of the same depth; it has not been measured in this repo
   (no GPU run yet), so treat it as design intent, not a benchmark.
2. **Quality floor.** Published hybrids sit between 1:6 and 1:7
   attention:linear — Jamba 1:7, Zamba 1:6, StripedHyena 1:7. HyMo's 3:1 is
   deliberately *more attention-heavy* (25% vs their ~12–14%), spending the
   extra compute on retrieval capacity with the low-KV-cache MLA rather than
   plain MHA.
3. **Coupling with the FFN asymmetry.** Because GDN blocks carry no FFN, the
   ratio also decides where the model's feed-forward capacity sits: all of it
   is on the 8 attention anchors, where sequence-mixing and channel-mixing
   capacity land in the same blocks.

Two config facts complete the picture:

- `src/hymo/core/config.py:ModelConfig.__post_init__` rejects impossible
  stacks (e.g. `n_heads` not a multiple of `n_kv_groups`), so a ratio
  experiment cannot silently produce a non-buildable model.
- The NoPE-hybrid option
  (`src/hymo/core/config.py:ModelConfig.nope_hybrid_gdn_positions`, gated by
  `nope_hybrid_gdn_enabled`, off in v1.0) would drop RoPE from the GDN layer
  directly preceding each MLA anchor. It changes positional handling only —
  not the ratio.

## Interview quick answers

**Why not 7:1 like Jamba?** Fewer attention anchors, and with plain MHA rather
than MLA. HyMo trades more per-step compute for stronger exact-attention
retrieval on 25% of the stack. [INFERENCE] on the quality claim — no ablation
has been run in this repo.

**Why not all-MLA?** You give up linear-time long-context behavior and the
constant-size recurrent state, and (with MoE on every layer) you multiply
stored parameters by 4. The hybrid keeps the 1.13 B-stored budget in range of
the MoE layers only.

**Where is the ratio visible at runtime?** In the block schedule only —
`src/hymo/models/model.py:HyMo._run_layers` iterates the mixed `ModuleList`
uniformly; nothing downstream branches on layer type.

## Cross-links

- [`gdn-and-mla.md`](gdn-and-mla.md) — the mechanism deep-dives and the
  published-hybrid comparison table.
- [`asymmetric-moe.md`](asymmetric-moe.md) — why all FFN/MoE capacity sits on
  the MLA blocks.
- [`model-architecture.md`](model-architecture.md) — the line-by-line
  walkthrough of both block types.
- [`../references/config.md`](../references/config.md) — the full
  `ModelConfig` field table.

## References

- Source: `src/hymo/core/config.py`, `src/hymo/models/model.py`,
  `src/hymo/models/gdn.py`, `src/hymo/models/mla.py`.
- Numbers (`n_layers = 32`, `dim = 896`, `gdn_d_inner = 1280`,
  `gdn_headdim = 32`, `gdn_d_state = 32`) are from `configs/hymo_750m.yaml`.
