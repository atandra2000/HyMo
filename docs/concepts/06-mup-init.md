# 06 — Initialization

> **Bridges to:** [`learning_docs/1_Model_Architecture.md`](../../learning_docs/1_Model_Architecture.md) §10
> (model construction).

## Learning objectives

After this file, you can:

1. State what initialization the shipped model actually applies.
2. Explain why the μP init described in the architecture doc was
   **never shipped**.
3. Identify where real init happens in the code.

## The honest status: μP init was designed, not shipped

The architecture doc (`docs/HyMo-Design.md` §4) and the earlier draft of
this chapter describe a full **maximal-update parametrization** init:
`mup_init(model, config)` walking every parameter, zeroing gates/biases,
scaling 2D weights to `std = 1/dim`, and a `zero_init_predicate` keyword
set. That module existed at `src/hymo/models/init.py` and was removed in
the 2026-08-04 cleanup.

**It was never called.** `build_hymo` constructs `HyMo(config.model)`
and returns — no init pass. The trainer never invoked `mup_init` either.
The μP init was dead code: fully written, fully documented, and never
wired into the production path. The `mup_init: true` config flag was
inert.

The reasons it was safe to delete:

- `grep mup_init( src/` → only the definition; zero call sites outside tests.
- No test asserted production behavior depended on it (only unit tests of
  the predicate function itself).
- The actual init path (below) was already what every real run used.

## What the shipped model actually does

PyTorch defaults, plus two deliberate inline choices:

1. **`nn.Linear` / `nn.Embedding` defaults.** The constructor applies
   Kaiming-uniform / normal init to weights, zero bias. `HyMo.__init__`
   calls no custom init; `GatedDeltaNetBlock`, `MLABlock`, `DeepSeekMoE`,
   and `MultiTokenPrediction` rely on module defaults.

2. **MoE gate init (inline in `moe.py`).** `DeepSeekMoE.__init__` sets
   `nn.init.zeros_(self.gate.bias)` and `nn.init.normal_(self.gate.weight,
   std=0.006)` — the gate starts near-uniform so routing is neutral at
   step 0 and the EMA bias update specializes it over time.

3. **GDN `A_log`/`dt_bias`/`D` (inline in `gdn.py`).** `A_log` starts at
   `log(1..n_heads)` so `A = -exp(A_log)` is a gentle decay; `dt_bias` is
   zero; `D` is ones. These are the values the recurrence actually uses —
   no external init pass touches them.

## Implementation in HyMo

- `src/hymo/models/model.py:23` — `HyMo.__init__`: no init pass; the
  constructor relies on module defaults.
- `src/hymo/models/moe.py:71-73` — the gate init: `bias = 0`,
  `weight ~ N(0, 0.006²)`.
- `src/hymo/models/gdn.py:57-62` — `A_log`, `dt_bias`, `D` inline init.

## Interview Q&A

**Q1. Why is the MoE gate init so small (`N(0, 0.006²)`)?**

> A: Because the gate is the **routing signal**. A gate with
> `weight ~ N(0, 1/dim)` would produce logits `O(1)` and a near-uniform
> softmax over 16 experts — fine start routing, but the EMA gate-bias
> update has to do all the work to specialize. Starting with smaller
> weights keeps the softmax closer to uniform across more of the early
> training, giving the EMA update more time to react to actual load
> imbalance.

**Q2. Why is `gate.bias = 0`?**

> A: Same reason — start uniform routing. The bias is updated by the EMA
> to break the symmetry as training progresses. Zero is the neutral
> starting point.

**Q3. Why is `A_log` initialized to `log(1..n_heads)`?**

> A: `A = -exp(A_log)` is the per-head decay. Starting near `A = -1`
> gives `α = exp(g · A) ≈ 0.27` at typical sigmoid inputs — a moderate
> decay: recent writes are weighted heavily, but old state isn't
> completely forgotten. This is the balanced starting point for the
> recurrence.

**Q4. Was μP init ever active?**

> A: No. It was written and documented but never called from
> `build_hymo` or the trainer; it was removed in the 2026-08-04 cleanup.
> If μP scaling is wanted later, the design (architecture doc §4) is
> preserved in git history — but the LR schedule (NorMuon `0.02`, AdamW
> `3e-4`) was tuned on the *current* init, so enabling μP would require
> re-tuning.

## Cross-links

- [`learning_docs/1_Model_Architecture.md`](../../learning_docs/1_Model_Architecture.md) §10
  (model construction walkthrough).
- [`concepts/07-muon-optimizer.md`](07-muon-optimizer.md) — the optimizer.
- [`concepts/03-mixture-of-experts.md`](03-mixture-of-experts.md) — the
  gate init and EMA bias update.
