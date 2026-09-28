# HyMo — Multi-Token Prediction (MTP)

> **Audience:** beginner-to-intermediate. MTP is HyMo's third "free lunch":
> extra learning signal for ~12 M parameters and a few extra matmuls per
> token. The block-level walkthrough is in
> [`model-architecture.md`](model-architecture.md) §9; the mechanism context
> alongside GDN/MLA/MoE is in [`gdn-and-mla.md`](gdn-and-mla.md).

## The idea

Next-token prediction supervises position `t` with exactly one target, `t+1`.
**Multi-Token Prediction** (Gloeckle et al. 2024; DeepSeek-V3) adds auxiliary
heads that predict *further* futures — `t+2`, `t+3`, … — from the same
representations, with down-weighted losses. The main objective is untouched;
the auxiliary gradients shape the hidden state to encode more than one step
of lookahead, which is cheap at train time and discarded at inference.

HyMo's objective, from `src/hymo/core/config.py:ModelConfig`
(`mtp_depth = 2`, `mtp_loss_weights = (0.3, 0.1)`):

```
L = L_next + 0.3 · L_{t+2} + 0.1 · L_{t+3}
```

The weights decay with distance because further futures are less predictable:
an equal-weight `t+3` head would mostly inject noise into the gradient. The
weights are validated at load time — length must equal `mtp_depth`, all
non-negative, and `mtp_depth = 0` requires an empty list
(`src/hymo/core/config.py:ModelConfig.__post_init__`) — so the loss shape and
the head count can never disagree.

## The design: chained heads on the main model

`src/hymo/models/mtp.py:MultiTokenPrediction` is constructed with a reference
to the main model (`src/hymo/models/model.py:HyMo.__init__` creates it when
`mtp_depth > 0`) and runs *after* the main stack:

1. Get the main result once:
   `main_logits, main_hidden = main.forward_with_hidden(tokens)`
   (`src/hymo/models/model.py:HyMo.forward_with_hidden` returns the softcapped
   logits and the post-`norm` hidden states).
2. For each depth `d` (0-based), let `usable = T − d − 1`. Take the target
   ids `tokens[:, d+1 : d+1+usable]`, embed them with the main model's own
   `embed`, and run head `d`:
   `src/hymo/models/mtp.py:MultiTokenPrediction._mtp_head` computes
   `fused = hidden + emb`, then a SwiGLU
   (`w2(silu(w1(fused)) * w3(fused))`) via
   `src/hymo/models/mtp.py:MTPBlock`, and projects to logits with the
   **main model's** `head` — which is tied to the embedding when
   `tie_embeddings` is true, so the vocab projection adds no new parameters.
3. **The chain is the part that differs from a flat multi-head reading:**
   head 1 consumes the *main* hidden state, but head 2 consumes head 1's
   output hidden state (`prev_hidden = new_hidden` in
   `src/hymo/models/mtp.py:MultiTokenPrediction.forward`). Each head fuses
   the previous module's representation with the embedding of the token it is
   being asked to predict — the DeepSeek-V3 sequential-MTP pattern. Each
   head's result is packaged as an `src/hymo/models/mtp.py:MTPOutput`
   (logits, target ids, loss weight) for the trainer.

```python
# illustrative — one chained head, condensed from _mtp_head
fused = hidden + emb            # prev representation + target-token embedding
h = F.silu(block.w1(fused)) * block.w3(fused)
out = block.w2(h)
logits = main.head(out)         # shared, tied vocab projection
```

The sequence shortens by one position per depth (`usable = T − d − 1`), so
every auxiliary logit has a target that is genuinely `d+1` tokens ahead in
the original sequence — no padding, no masking tricks.

## Cost

Derived from the config shapes (`dim = 896`, `mtp_inter_dim = 2304`,
`mtp_depth = 2`):

- Per head: `3 × 896 × 2304 ≈ 6.19 M` parameters (`w1`, `w2`, `w3`); both
  heads `≈ 12.4 M` new parameters — about 1% of the ~1.13 B stored total.
- The vocab projection is shared with the main head, so there is no
  `896 × 64,256` duplication per head.
- Compute: one extra SwiGLU plus one vocab matmul per depth per token
  position — a small single-digit-percent forward overhead, and the two CE
  evaluations on top of the main loss.

The documented headline totals (~434 M active / ~1.13 B stored,
[`../../README.md`](../../README.md)) come from
`src/hymo/models/model.py:HyMo.num_parameters`; its full-model verification
is heavy-gated (see [`../AUDIT.md`](../AUDIT.md) F4).

## How the trainer consumes it

`src/hymo/training/trainer.py:Trainer.train_step` branches on
`mtp_depth > 0`: when MTP is present it calls
`src/hymo/models/mtp.py:MultiTokenPrediction.forward` (which returns main
logits *plus* the auxiliary outputs) instead of
`src/hymo/models/model.py:HyMo.forward`. Then:

- the main CE is computed exactly as without MTP
  (`logits[:, :-1]` vs `targets[:, :-1]`);
- each `MTPOutput` contributes `CE(mtp_out.logits, mtp_out.targets) ×
  mtp_out.loss_weight` to `total_loss`, with per-head values logged as
  `mtp_{i}_loss` / `mtp_{i}_weighted` metrics;
- the NaN/inf skip, gradient-accumulation scaling, clip, and the dual
  optimizer step all see one combined `total_loss` — MTP gradients flow
  through the same backward as the main loss (the loss-computation split is
  documented in [`gdn-and-mla.md`](gdn-and-mla.md) Q5).

## What happens at inference

Nothing — by design. `src/hymo/models/model.py:HyMo.forward` runs
`forward_with_hidden` → `head` → `softcap` and never touches `self._mtp`.
The MTP heads exist to shape training-time representations; the deployed
model's logits path is byte-identical with or without them. (This is why
`mtp_depth = 0` is a legal config: the same code trains and serves either
way.)

## Cross-links

- [`gdn-and-mla.md`](gdn-and-mla.md) — MTP in the mechanism-lineup context.
- [`model-architecture.md`](model-architecture.md) — the line-by-line MTP
  walkthrough.
- [`hybrid-ratio.md`](hybrid-ratio.md) — the stack whose hidden states the
  heads consume.
- [`../references/config.md`](../references/config.md) — the `mtp_*` fields.

## References

- Source: `src/hymo/models/mtp.py`, `src/hymo/models/model.py`,
  `src/hymo/training/trainer.py`, `src/hymo/core/config.py`.
- Config values: `configs/hymo_750m.yaml` §model (`mtp_depth`,
  `mtp_loss_weights`, `mtp_inter_dim`).
