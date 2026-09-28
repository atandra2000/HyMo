# HyMo — Asymmetric Mixture of Experts

> **Audience:** intermediate. "Asymmetric" names two facts about HyMo's MoE:
> *where* it is placed (8 of 32 layers) and *how* the expert block is shaped
> (16 routed + 1 shared, top-2 activated). The routing mechanics are derived
> in [`gdn-and-mla.md`](gdn-and-mla.md) §Mixture of Experts; this doc is the
> configuration-and-rationale view.

## The two asymmetries

**1. Placement asymmetry.** `src/hymo/models/mla.py:MLABlock.__init__` wires a
`src/hymo/models/moe.py:DeepSeekMoE` as the block's second residual branch
(`src/hymo/models/mla.py:MLABlock.forward` applies attention, then the MoE —
two independent pre-norm updates). The 24 GDN blocks have **no feed-forward
path at all**: `src/hymo/models/gdn.py:GatedDeltaNetBlock.__init__` builds
only the recurrence projections. A `DenseFFN` module existed in
`src/hymo/models/moe.py` history for GDN blocks and was removed in the
2026-08-04 cleanup (never instantiated in `src`). Consequence: 100% of HyMo's
channel-mixing (FFN) capacity lives on the 8 full-attention anchors.

**2. Configuration asymmetry.** On those 8 layers, the expert block is
DeepSeek-style fine-grained sparsity, from `src/hymo/core/config.py:ModelConfig`
(via `configs/hymo_750m.yaml`):

| Field | Value | Meaning |
|---|---|---|
| `n_routed_experts` | 16 | routed experts per MoE layer |
| `n_shared_experts` | 1 | one always-on dense expert |
| `n_activated_experts` | 2 | top-2 routing per token |
| `moe_inter_dim` | 2304 | per-expert SwiGLU width (~2.6× `dim = 896`) |
| `moe_capacity_factor` | 1.5 | per-expert token cap = 1.5× mean load |
| `moe_ema_alpha` | 0.02 | EMA rate for the aux-loss-free balance update |

Each expert is a `src/hymo/models/moe.py:SwiGLUExpert` — `w2(silu(w1(x)) * w3(x))`,
no biases. The shared expert is the same class, computed for every token and
added after the routed dispatch (`src/hymo/models/moe.py:DeepSeekMoE.forward`).

## Why this shape

**Why MoE only on the attention anchors?** Stored parameters are cheap when
they are not activated; active FLOPs are what you pay per token. Top-2-of-16
routing means each MoE layer spends 2/16 of its routed SwiGLUs per token, so
the sparse branch's *active* cost stays at a dense-FFN-like level while its
*stored* capacity grows ~8.5× (17 experts vs 1 dense FFN of the same width).
Concentrating that stored capacity on the 8 MLA blocks buys specialized
channel-mixing exactly where the model does its hardest work (retrieval,
long-range reasoning), and leaves the 24 GDN layers as cheap linear-time
sequence mixers. Putting MoE on all 32 layers would quadruple stored FFN
parameters for capacity the linear blocks do not need.

**Why fine-grained (16 narrow experts) instead of few wide ones?** Top-2 over
16 experts gives 120 ordered pair combinations per layer — far more
composable specializations than top-2 over 4. This is the DeepSeek-V2/V3
fine-grained-expert thesis, scaled to a 750 M-class model.

**Why a shared expert at all?** Some processing is needed on every token; the
routed branch should spend its capacity on *residual* specialization. The
shared branch also bounds the damage of the capacity cap: tokens dropped from
an overloaded expert still get the dense path (the `forward` docstring says
exactly this).

## Derived parameter budget

Arithmetic from the config values above (per MoE layer):

- Per expert: `3 × 896 × 2304 ≈ 6.19 M` params.
- Routed + shared: `17 × 6.19 M ≈ 105 M` per layer; across 8 layers
  `≈ 840 M` stored — the bulk of the model's stored total.
- Active per token per layer: top-2 routed `2 × 6.19 M` + shared `6.19 M`
  `≈ 18.6 M`; across 8 layers `≈ 149 M` active MoE params.

These are derived multiplications, not measurements. The headline totals
(~434 M active / ~1.13 B stored) are the documented model numbers
([`../../README.md`](../../README.md)); they come from
`src/hymo/models/model.py:HyMo.num_parameters`, whose full-model verification
is heavy-gated (see [`../AUDIT.md`](../AUDIT.md) F4).

## Load balancing without an auxiliary loss

The router is a single `Linear(dim, 16)` initialized near-uniform (bias zero,
weight `N(0, 0.006)` in `src/hymo/models/moe.py:DeepSeekMoE.__init__`), and
its logits are computed in FP32 regardless of activation dtype
(`src/hymo/models/moe.py:DeepSeekMoE.gate_forward`) because a BF16 16-way
softmax rounds near-uniform probabilities to exactly uniform.

Balance is enforced by a bias update, not a loss term. After every optimizer
step the trainer calls `src/hymo/training/trainer.py:Trainer._update_moe_gate_biases`,
which invokes `src/hymo/models/moe.py:DeepSeekMoE.update_gate_bias`: bin the
last batch's routing indices, EMA-smooth the counts (`moe_ema_alpha = 0.02`,
≈50-batch window), and move the gate bias down for experts >5% over the mean
load and up for experts >5% under it, by `speed = 0.001` per step. No
auxiliary gradient competes with the LM loss, and there is no λ to tune.

Two more mechanics the dispatch loop enforces
(`src/hymo/models/moe.py:DeepSeekMoE.forward`):

- **Capacity cap.** `capacity = int(1.5 × (B·T·k) / 16)`; tokens beyond an
  expert's cap are dropped from that *routed* branch (shared branch still
  covers them).
- **Mixed-precision dispatch** (threaded from
  `src/hymo/core/config.py:TrainingConfig.moe_mixed_precision` via
  `src/hymo/training/trainer.py:Trainer._thread_optimization_flags`): token
  activations are cast to the expert weight dtype so expert matmuls run in
  BF16 under FSDP, halving dispatch bandwidth.

## Which optimizer sees the experts

None of the expert weights go to NorMuon.
`src/hymo/training/partition.py:goes_to_adamw` explicitly routes
`.experts.*.w1/w2/w3.weight` and the `.shared_expert.*` weights to AdamW, as
well as the router's `.gate.weight`/`.gate.bias`. The rationale (and the full
partition) is in [`dual-optimizers.md`](dual-optimizers.md); the short version:
Muon's orthogonalized update equalizes directions across the matrix, which is
the wrong inductive bias for weights that a load-balancing bias update is
actively trying to differentiate.

## Cross-links

- [`gdn-and-mla.md`](gdn-and-mla.md) — routing math, EMA derivation, Q&A.
- [`hybrid-ratio.md`](hybrid-ratio.md) — why only 8 layers are in scope at all.
- [`dual-optimizers.md`](dual-optimizers.md) — why experts ride AdamW.
- [`../training.md`](../training.md) — the trainer call site for the bias update.

## References

- Source: `src/hymo/models/moe.py`, `src/hymo/models/mla.py`,
  `src/hymo/training/partition.py`, `src/hymo/core/config.py`.
- Config values: `configs/hymo_750m.yaml` §model.
