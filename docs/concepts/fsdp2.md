# HyMo — FSDP-2: Fully Sharded Data Parallel

> **Audience:** intermediate. This doc covers both the mechanism (what full
> parameter sharding means for memory) and HyMo's actual sharding surface
> (`src/hymo/training/fsdp.py`), including the parts that are wired and the
> part that is not. The optimization-side overview lives in
> [`optimization.md`](optimization.md) §FSDP-2.

## The mechanism in one cycle

Data parallelism replicates the model on every rank. Full sharding removes
that redundancy: each rank holds only `1/W` of the parameters, gradients, and
optimizer state (`W = world_size`), and reconstructs the pieces it needs
on demand:

1. **All-gather before use** — a module about to compute gathers the full
   values of its sharded parameters from all ranks.
2. **Compute locally** — forward and backward math is unchanged; it is
   ordinary per-module math on full tensors.
3. **Reduce-scatter after backward** — gradients are summed across ranks and
   immediately split so each rank keeps only its `1/W` shard.
4. **Re-shard** — gathered parameters are freed; steady-state memory per rank
   is `O(model / W)`.

This is ZeRO-3 (the DeepSpeed stage taxonomy: stage 1 shards optimizer state,
stage 2 adds gradients, stage 3 adds parameters) realized in PyTorch. The
"FSDP-2" label in HyMo's docs and `TrainingConfig` naming refers to this
fully-sharded design intent; the shipped module implements it with
PyTorch's `FullyShardedDataParallel` wrapper API (see the drift note below).

## What HyMo's code actually does

`src/hymo/training/fsdp.py:wrap_model_with_fsdp` is the entire integration
surface:

- **Sharding unit = whole blocks.** With no explicit policy, it installs
  `src/hymo/training/fsdp.py:fsdp_auto_wrap_policy`, which returns `True`
  only for `GatedDeltaNetBlock` and `MLABlock`. So the model is partitioned
  into its 32 top-level blocks (`HyMo.__init__`'s stack), not into individual
  projections. Fewer, larger FSDP units mean fewer all-gather/reduce-scatter
  calls per step — you trade finer-grained peak-memory savings for fewer
  communication launches. At 32 units over a 1.13 B-stored model, each unit
  averages ~35 M parameters.
- **Mixed precision from config.** `src/hymo/core/config.py:TrainingConfig.fsdp_mixed_precision`
  (production: `bfloat16`) sets `param_dtype`, `reduce_dtype`, and
  `buffer_dtype` together, so parameters are stored and communicated in BF16
  and reductions run in BF16 — no loss-scale/GradScaler machinery, which
  BF16's FP32-wide exponent range makes unnecessary. `"float32"` and
  `"float16"` are accepted values; `"float16"` keeps scaler-free semantics in
  this module (precision-sensitivity of that choice is untested here).
- **CPU-safe no-op.** If `torch.distributed.fsdp` cannot be imported, the
  wrapper returns the model unchanged, so CPU-only construction, inspection,
  and tests work — the unit suite exercises the policy on CPU
  (`tests/unit/test_fsdp_trainer.py`).

```python
# illustrative — the sharding unit rule, verbatim in intent
def fsdp_auto_wrap_policy(module, recurse, non_blocking) -> bool:
    return isinstance(module, (GatedDeltaNetBlock, MLABlock))
```

## Status versus the docs' design intent

> **Drift (as of 2026-09-21, code truth first).** Three facts the older
> optimization prose smoothed over:
>
> 1. **The trainer does not call the wrapper.** Nothing in
>    `src/hymo/training/trainer.py` references FSDP; `wrap_model_with_fsdp`
>    is exported (`src/hymo/training/__init__.py`) and unit-tested, but no
>    production call site exists.
> 2. **`src/hymo/core/config.py:TrainingConfig.fsdp` is currently a declared
>    but unread knob in `src/`** — the only consumed field is
>    `fsdp_mixed_precision`, read by `wrap_model_with_fsdp` itself. Turning
>    `fsdp: true` (the default in `configs/hymo_750m.yaml`) on or off does
>    not change any code path in this repository today.
> 3. **Naming.** The module docstring says "FSDP-2 integration helpers", and
>    the config/labels say FSDP-2, but the implementation uses the classic
>    `FullyShardedDataParallel` wrapper (pre-`fully_shard` API). The sharding
>    semantics — parameters, gradients, and optimizer state split per rank —
>    are the fully-sharded (ZeRO-3-style) semantics either way.

This is recorded as finding F5 in [`../AUDIT.md`](../AUDIT.md). The correct
reading of the corpus until the wiring lands: *full sharding is the designed
deployment mode for the 4-rank A100 run, and its building blocks exist and
are tested, but enabling it requires a call site in the trainer.*

## What it means for memory (derived, not measured)

For the stored model (`P ≈ 1.13 B`, from
`src/hymo/models/model.py:HyMo.num_parameters`, heavy-gated) at the
configured `world_size = 4`
(`src/hymo/core/config.py:TrainingConfig.world_size`), with the BF16 recipe
and FP32 master/state that the optimizers hold internally
([`dual-optimizers.md`](dual-optimizers.md)):

| Component (per rank, sharded) | Arithmetic | Value |
|---|---|---|
| BF16 parameters | `1.13 B / 4 × 2 B` | ≈ 565 MB |
| BF16 gradients | same | ≈ 565 MB |
| FP32 master weights | `1.13 B / 4 × 4 B` | ≈ 1.13 GB |
| AdamW `exp_avg` + `exp_avg_sq` (FP32) | `1.13 B / 4 × 8 B` | ≈ 2.26 GB |
| Momentum buffer (NorMuon, FP32) | `1.13 B / 4 × 4 B` | ≈ 565 MB |
| Activations + all-gather window | workload-dependent | not derived here |

These are [INFERENCE] numbers: arithmetic from config values and dtype widths,
measured nowhere in this repo (the audit machine is CPU-only macOS). Two
things the table makes obvious: sharding scales the entire column by `1/W`
(doubling ranks halves it), and the optimizer state — not the model — is the
largest resident block, which is the classic ZeRO motivation for sharding
parameters *and* state together. The per-step token budget that fills the
activation term is `src/hymo/core/config.py:TrainingConfig.per_step_tokens`
(4 × 8 × 4 × 4096 = 524,288 tokens per optimizer step across ranks).

## Cross-links

- [`dual-optimizers.md`](dual-optimizers.md) — the FP32 master weights the
  table above shards.
- [`optimization.md`](optimization.md) — the optimization-flags context and
  the older FSDP overview prose.
- [`../training.md`](../training.md) — the trainer loop that would host the
  wrapping call.

## References

- Source: `src/hymo/training/fsdp.py`, `src/hymo/core/config.py`,
  `src/hymo/models/model.py`.
- Config values: `configs/hymo_750m.yaml` §training (`world_size = 4`,
  `fsdp: true`, `fsdp_mixed_precision: bfloat16`).
