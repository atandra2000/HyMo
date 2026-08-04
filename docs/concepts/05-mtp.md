# 05 — Multi-Token Prediction (MTP)

> **Bridges to:** [`learning_docs/1_Model_Architecture.md`](../../learning_docs/1_Model_Architecture.md) §9
> (MTP block)

## Learning objectives

After this file, you can:

1. State the MTP objective and why it improves representation
   learning.
2. Walk through DeepSeek-V3's MTP design (depth-2 weighted
   auxiliary heads).
3. Compute the FLOPs overhead of MTP at the production scale.
4. Defend HyMo's choice of `mtp_depth = 2` with weights
   `[0.3, 0.1]`.

## Intuition

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

## Math derivation

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

## Implementation in HyMo

- `src/hymo/models/mtp.py:23` — `class MTPOutput` (per-head
  output container).
- `src/hymo/models/mtp.py:43` — `class MTPBlock`: one auxiliary
  head (RMSNorm + Linear + Linear).
- `src/hymo/models/mtp.py:53` — `class MultiTokenPrediction`:
  the wrapper that holds `mtp_depth` heads.
- `src/hymo/models/mtp.py:56` — `__init__` builds the heads
  and a small projection from the main hidden state.
- `src/hymo/models/mtp.py:69` — `_mtp_head(k, x)`: per-head
  forward.
- `src/hymo/models/mtp.py:81` — `forward(tokens)`:
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

- `src/hymo/training/trainer.py:122-131` — if `_has_mtp`, call
  `mtp_module.forward(tokens)` instead of `model.forward(tokens)`.
  This returns `(logits, mtp_outputs)` instead of just `logits`.
- `src/hymo/training/trainer.py:143-151` — loop over
  `mtp_outputs`, compute each MTP loss, multiply by
  `mtp_out.loss_weight`, add to `total_loss`. Record per-head
  metrics for W&B.

## Worked example

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

## Interview Q&A

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

## Cross-links

- [`learning_docs/1_Model_Architecture.md`](../../learning_docs/1_Model_Architecture.md) §9
  (MTP block walkthrough).
- [`learning_docs/3_Training_Pipeline.md`](../../learning_docs/3_Training_Pipeline.md) §6.2
  (the trainer's MTP wiring).
- [`concepts/03-mixture-of-experts.md`](03-mixture-of-experts.md) —
  the MoE-on-MLA "auxiliary compute" pattern that MTP parallels.
