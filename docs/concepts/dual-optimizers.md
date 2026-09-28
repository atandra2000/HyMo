# HyMo — Dual Optimizers: NorMuon + CautiousAdamW

> **Audience:** expert (optimizer mechanics). The user-facing summary is in
> [`optimization.md`](optimization.md) §Muon; [`../training.md`](../training.md)
> §3 walks the trainer loop. This doc is the ground-truth view of *which*
> optimizers HyMo runs, *which* parameters each one owns, and *why* the split
> is what it is.

## The two optimizers

HyMo trains with two disjoint optimizers over one model — never one optimizer
with two parameter groups:

1. **`src/hymo/training/optimizer.py:NorMuon`** — a Muon-family optimizer for
   dense 2-D weight matrices: momentum → Newton–Schulz orthogonalization →
   per-row RMS matching → an FP32 master-weight update with cautious weight
   decay. Production hyperparameters (from `src/hymo/core/config.py:OptimizerConfig`,
   mirrored in `configs/hymo_750m.yaml`): `muon_lr = 0.02`,
   `muon_momentum = 0.95`, `muon_betas = (0.95, 0.95)`,
   `muon_weight_decay = 0.1`, `ns_iterations = 5`.
2. **`src/hymo/training/optimizer.py:CautiousAdamW`** — textbook AdamW
   (bias-corrected, decoupled decay) plus the cautious-decay mask, holding
   its own FP32 master copies. Production: `adamw_lr = 3e-4`,
   `adamw_betas = (0.9, 0.95)`, `adamw_weight_decay = 0.0`.

The pair is bundled in `src/hymo/training/optimizer.py:Optimizers` (a
`__slots__` container with joint `state_dict` / `load_state_dict` — note
`nor_muon` is nullable, for a model whose partition leaves no 2-D matrices),
and built by `src/hymo/training/optimizer.py:build_optimizers`, which first
partitions the model and then constructs each optimizer over its own list.

## The partition: who owns what

`src/hymo/training/partition.py:partition_parameters` walks
`model.named_parameters()`, deduplicates tied parameters (the tied
embedding/head weights appear once — the `seen` id set), and sends every
parameter through `src/hymo/training/partition.py:goes_to_adamw`. The rules,
in order:

| Rule | Rationale |
|---|---|
| `ndim < 2` (biases, scalars) | orthogonalization is a matrix operation; vectors/scalars get none of its benefit |
| `ndim > 2` (e.g. the grouped `conv1d` kernel) | same — not a dense 2-D matrix |
| name ends `embed.weight` / `head.weight` | the vocab projection's geometry (and its tied head) is not what Muon conditions well; standard practice is AdamW |
| name ends `.gate.weight` / `.gate.bias` | the MoE router must stay free for the EMA bias update; see [`asymmetric-moe.md`](asymmetric-moe.md) |
| `.experts.*.w1/w2/w3.weight` and `.shared_expert.*` | Muon's spectrum-equalizing update fights expert *differentiation*; each expert needs independent entry-wise dynamics |
| `.A_log`, `.dt_bias`, `.D` | GDN's learned scalar controls; small, 1-D-ish, decay-sensitive |
| name ends `norm.weight` | norm gains are scalar-per-channel |
| everything else 2-D | **NorMuon** |

The leftover-2-D set is exactly what Muon is for: the attention projections
in `src/hymo/models/mla.py:MultiHeadLatentAttention` (`wq_a/wq_b/wkv_a/wkv_b/wo`)
and the GDN projections in `src/hymo/models/gdn.py:GatedDeltaNetBlock.__init__`
(`in_proj`, `b_proj`, `c_proj`, `dt_proj`, `g_proj`, `out_proj`, `skip_proj`).
`src/hymo/training/partition.py:ParameterPartition` just holds the two lists.

```python
# illustrative — the decision, condensed from goes_to_adamw + partition_parameters
if goes_to_adamw(name, param):
    p.adamw.append(param)
elif param.ndim >= 2:
    p.nor_muon.append(param)
```

## What NorMuon actually does per step

`src/hymo/training/optimizer.py:NorMuon.step` per parameter:

1. **Momentum buffer**: `buf = β·buf + (1−β)·grad` with `β = 0.95`.
2. **Orthogonalize** the buffer via
   `src/hymo/training/optimizer.py:_newton_schulz_orthogonalize`: normalize by
   the Frobenius norm, run 5 iterations of `X ← 1.5·X − 0.5·X·Xᵀ·X`, then
   restore the original norm. The result has (approximately) unit singular
   values — every gradient *direction* moves at the same rate, and no single
   large singular value dominates the step.
3. **RMS matching** — the "Nor" in NorMuon. The update's per-row norms are
   rescaled so each row's RMS is 1 (`row_norms / sqrt(row_len)`), which puts
   the update's per-entry magnitude on the same scale as an AdamW update.
   This is what lets a single `lr = 0.02` sit next to `adamw_lr = 3e-4`
   without per-tensor retuning, and it makes the LR transferable across layer
   shapes.
4. **Cautious decay on the master**: with `cautious_wd = True`
   (`src/hymo/core/config.py:OptimizerConfig.cautious_wd`), the weight-decay
   shrink `master ← master·(1 − lr·wd)` is masked to entries where
   `grad · master > 0` — decay only where it agrees with the gradient's push.
5. **FP32 master update, cast back**: `master.add_(update, alpha=−lr)`, then
   `p.data.copy_(master.to(p.dtype))`. The live parameter can be BF16 under
   FSDP while the accumulated state stays FP32.

## What CautiousAdamW does

`src/hymo/training/optimizer.py:CautiousAdamW.step` is AdamW with bias
correction (`denom = sqrt(exp_avg_sq / bias_corr2) + eps`,
`step_size = lr / bias_corr1`) plus the same cautious mask on decay — and both
optimizers keep a `state["master_weight"]` FP32 copy from step 0. One precise
consequence of the production config: `adamw_weight_decay = 0.0`, so the
`wd != 0` branch (and therefore the cautious mask) is **inactive** for AdamW
today; the cautious machinery is live on NorMuon, where
`muon_weight_decay = 0.1`. The flag is shared — flipping
`cautious_wd` re-arms decay masking on both.

## How the trainer drives both

`src/hymo/training/trainer.py:Trainer.train_step` is the single driver:

- Every micro-batch backprops `scaled_loss = total_loss /
  gradient_accumulation_steps`; the update block runs only when
  `micro_step % gradient_accumulation_steps == 0`.
- One schedule factor for both: `factor = scheduler.get_factor(step + 1)`
  (`src/hymo/training/scheduler.py:JointWSDScheduler.get_factor`), then both
  param-group LRs are set to `base × factor` — Muon and AdamW ride the same
  WSD curve at their different bases (0.02 and 3e-4).
- `nor_muon.step()` (when present) then `adamw.step()`, then the MoE gate-bias
  EMA update, then `scheduler.step()` and `zero_grad`.
- Both optimizers are checkpointed through
  `src/hymo/training/optimizer.py:Optimizers.state_dict` (momentum buffers,
  Adam moments, FP32 masters) so resume is exact.

## Why two optimizers instead of one

Muon's orthogonalization is the right operator for the bulk of the parameter
*count* in hidden layers (dense 2-D projections) and the wrong one for
everything whose geometry is not "a matrix whose spectrum is the signal":
embeddings, norms, scalars, router gates, and MoE experts. A single AdamW
would give up the convergence win on the projections; a single Muon would
have no defined, sane update for the rest. The split keeps each parameter
under the optimizer whose update rule matches its shape — at the cost of two
state machines, which `Optimizers` collapses into one checkpointable bundle.

## Cross-links

- [`optimization.md`](optimization.md) — the intuition/Q&A treatment of Muon,
  cautious WD, and the WSD schedule that scales both LRs.
- [`asymmetric-moe.md`](asymmetric-moe.md) — why the experts and router are
  AdamW-only.
- [`../training.md`](../training.md) — the loop, gradient accumulation, and
  NaN-skip that wrap the optimizer steps.

## References

- Source: `src/hymo/training/optimizer.py`, `src/hymo/training/partition.py`,
  `src/hymo/training/trainer.py`, `src/hymo/core/config.py`.
- Hyperparameters: `configs/hymo_750m.yaml` §optimizer.
