# 08 — WSD: Warmup–Stable–Decay

> **Bridges to:** [`learning_docs/3_Training_Pipeline.md`](../../learning_docs/3_Training_Pipeline.md) §4
> (scheduler)

## Learning objectives

After this file, you can:

1. State the WSD (warmup-stable-decay) schedule and why it
   beats cosine for large-scale pretraining.
2. Walk through the three decay kinds (linear, cosine, sqrt)
   and when each is appropriate.
3. Defend HyMo's defaults: 2% warmup, 83% stable, 15% decay,
   `min_lr_ratio = 0.05`, `decay = "linear"`.

## Intuition

The classic learning rate schedule is **cosine**: the LR
ramps up from 0 to peak over a short warmup, then decays as
a half-cosine to `min_lr_ratio` over the rest of training.

```
LR
peak ─────────┐
              ╲
               ╲
                ╲
                 ╲
                  ╲___
min ──────────────────────
       warmup        decay
```

For large-scale pretraining, cosine has two problems:

1. **Cannot extend without retuning**: if the run is extended
   to 50 B tokens, the LR trajectory shifts and the peak may
   need to be re-tuned.
2. **Decay is shape-changing**: the LR is never really "stable"
   at peak; it always starts decaying after warmup. This means
   the model never sees the full benefits of the peak LR.

**WSD** (warmup-stable-decay) splits the schedule into three
phases:

```
LR
peak ─────────────────────────────┐
                                  ╲
                                   ╲
                                    ╲
                                     ╲___
min ──────────────────────────────────────
       warmup      stable       decay
       (2%)        (83%)        (15%)
```

- **Warmup (2%)**: ramp 0 → peak linearly.
- **Stable (83%)**: hold at peak.
- **Decay (15%)**: drop from peak to `min_lr_ratio` along a
  chosen shape (linear, cosine, or sqrt).

### Why WSD beats cosine

Two reasons:

1. **Comparability across run lengths**: an ablation that runs
   for 7.5 B tokens with the same `warmup_frac / stable_frac /
   decay_frac` as the 30 B production run has the same LR
   shape — just shorter in each phase. Cosine would force the
   peak LR to scale with run length.
2. **Long stable phase**: the model sees the full peak LR for
   83% of training, which empirically gives ~5-10% better
   convergence at fixed tokens.

The cost: at the very end of training, the LR drops sharply
(linear). This can be mitigated by choosing `cosine` decay
instead of `linear`, but `linear` is the empirical default for
large-scale pre-training.

## Math derivation

### Phase A: warmup

```
factor(t) = t / warmup_steps            for t in [0, warmup_steps)
```

Smooth ramp from 0 to peak. Linearly in `t`, not based on
log-scale.

### Phase B: stable

```
factor(t) = 1.0                          for t in [warmup_steps, warmup_steps + stable_steps)
```

Hold at peak. Simple.

### Phase C: decay

```
progress = (t - warmup_steps - stable_steps) / decay_steps
                                             ∈ [0, 1]

linear:   f(p) = 1 - p
cosine:   f(p) = 0.5 · (1 + cos(π · p))
sqrt:     f(p) = sqrt(1 - p)

factor(t) = min_lr_ratio + (1 - min_lr_ratio) · f(progress)
```

The `min_lr_ratio + (1 - min_lr_ratio) · f(progress)` form
ensures that `factor = 1.0` at `progress = 0` and
`factor = min_lr_ratio` at `progress = 1`.

### Three decay shapes

| Shape | `f(p)` at `p = 0.5` | Sharpness at end |
|---|---|---|
| `linear` | 0.5 | Linear drop |
| `cosine` | 0.5 | Smooth tail |
| `sqrt` | 0.707 | Aggressive at end |

`linear` is the default; `cosine` is for transformer
fine-tuning where the long tail is valued; `sqrt` is rare
and used for fast end-of-training convergence.

### Why WSD's total_steps is the global optimizer step

The `total_steps` field on `SchedulerConfig` is the
**optimizer step** count, not the micro-batch count. So
`warmup_steps = total_steps × warmup_frac` is in optimizer
steps. With `gradient_accumulation_steps = 8`, the
warmup phase has `8 × warmup_steps` micro-batches.

This is what the trainer uses: `scheduler.get_factor(step + 1)`
is called with the optimizer step counter, not the
micro-step counter.

## Implementation in HyMo

- `src/hymo/training/scheduler.py:15` — `class JointWSDScheduler`.
- `src/hymo/training/scheduler.py:18-27` — `__init__`:
  `warmup_steps`, `stable_steps`, `decay_steps` from config
  properties; `min_lr_ratio`, `decay_kind` from config.
- `src/hymo/training/scheduler.py:31` — `get_factor(step)`:
  the three-phase logic.
- `src/hymo/training/scheduler.py:57` — `_decay_factor(progress,
  kind)`: the static helper.
- `src/hymo/training/trainer.py:180` — `factor = self.scheduler
  .get_factor(self.step + 1)`: the call site.
- `src/hymo/training/scheduler.py:50-54` — `state_dict` /
  `load_state_dict`: the scheduler step counter.

## Worked example

Production scale (default `configs/hymo_750m.yaml`):

- `total_steps = 57_220`
- `warmup_frac = 0.02`, `stable_frac = 0.83`, `decay_frac = 0.15`
- `min_lr_ratio = 0.05`, `decay = "linear"`

Computed:

- `warmup_steps = 57220 × 0.02 = 1144`
- `stable_steps = 57220 × 0.83 = 47492`
- `decay_steps = 57220 × 0.15 = 8583`
- `stable_end = 1144 + 47492 = 48636`

LR trajectory (for `muon_lr = 0.02`):

```
step 0       → factor = 0/1144 = 0.0           → LR = 0.0000
step 572     → factor = 572/1144 = 0.5        → LR = 0.0100
step 1144    → factor = 1.0                     → LR = 0.0200 (peak)
step 1145..48635 → factor = 1.0                → LR = 0.0200 (stable)
step 52884   → factor = 0.05 + 0.95 × 0.5 = 0.525 → LR = 0.0105
step 57219   → factor = 0.05 + 0.95 × 0.0 = 0.05 → LR = 0.0010 (end)
```

The model spends ~83% of training at peak LR, then drops
sharply over the final 15% to 5% of peak.

For an ablation with 7.5 B tokens (per the ablations
framework), the same `warmup_frac = 0.02` gives
`warmup_steps = total_steps × 0.02 = (7.5e9 / 524288) × 0.02
= 286` — but the same proportion of training.

## Interview Q&A

**Q1. Why WSD over cosine?**

> A: WSD's stable phase holds the LR at peak for 83% of
> training, which empirically gives ~5-10% better convergence
> at fixed tokens. The cost is a sharp end-of-training drop;
> for pre-training, this is fine because the model has already
> converged by then.

**Q2. Why 2% warmup and not 5% or 0.5%?**

> A: 2% is the empirical default for large-scale
> pre-training. Shorter warmup (0.5%) risks an early spike
> in loss; longer warmup (5%) wastes optimizer steps that
> could be at peak LR. 2% is the sweet spot for 30 B tokens.

**Q3. Why 15% decay and not 10% or 30%?**

> A: 15% gives enough decay steps to bring the LR to
> `min_lr_ratio = 0.05` without being so long that the
> model effectively trains at low LR. 10% is too sharp; 30%
> wastes steps at sub-peak LR.

**Q4. Why `min_lr_ratio = 0.05` and not 0.0 or 0.1?**

> A: 0.05 is the Llama-3 / DeepSeek-V3 default for
> pre-training. 0.0 would let the optimizer make no
> progress at the very end. 0.1 would keep the LR
> artificially high, which can hurt final convergence.

**Q5. Why `decay = "linear"` and not "cosine"?**

> A: Linear decay gives a sharp end-of-training drop, which
> is fine for pre-training because the model has converged
> by then. Cosine decay would give a smoother tail but
> would still hold the LR at peak for the same 83% of
> training; the difference is only in the final 15%.

**Q6. Why is `Step` a `NewType` rather than `int`?**

> A: Type checking. `Step = NewType("Step", int)` is zero-cost
> at runtime but the type checker catches bugs like
> `scheduler.get_factor(micro_step)` (which should be the
> optimizer step). See `learning_docs/6_Config_System.md` §3.

**Q7. What happens if I extend the run to 50 B tokens?**

> A: Bump `total_steps` to `50e9 / 524288 = 95367`. The
> scheduler recomputes `warmup_steps = 1907`, `stable_steps =
> 79135`, `decay_steps = 14305`. The LR shape is the same; the
> training spends more steps at peak. No LR retuning needed.

## Cross-links

- [`learning_docs/3_Training_Pipeline.md`](../../learning_docs/3_Training_Pipeline.md) §4
  (scheduler walkthrough).
- [`learning_docs/6_Config_System.md`](../../learning_docs/6_Config_System.md) §2.3
  (SchedulerConfig).
- [`learning_docs/5_Evaluation_and_Ablations.md`](../../learning_docs/5_Evaluation_and_Ablations.md)
  §3.2 (how ablations inherit the schedule fractions).
- [`concepts/07-muon-optimizer.md`](07-muon-optimizer.md) —
  the optimizer that depends on the LR.
