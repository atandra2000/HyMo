# 09 — FSDP-2: Full Parameter Sharding

> **Bridges to:** [`learning_docs/3_Training_Pipeline.md`](../../learning_docs/3_Training_Pipeline.md) §5
> (FSDP wrapper)

## Learning objectives

After this file, you can:

1. State the lineage from DDP to ZeRO to FSDP-2.
2. Explain why full parameter sharding is necessary at 1 B+
   params.
3. Walk through HyMo's `wrap_model_with_fsdp` and the
   auto-wrap policy.
4. Compute per-rank memory at the production scale.

## Intuition

Standard **DDP** (Distributed Data Parallel) replicates the
model on every rank and synchronizes gradients via
all-reduce. Memory per rank: `O(model_size)` — every rank
holds the full model.

**ZeRO** (Zero Redundancy Optimizer, Rajbhandari et al.
2019) splits the optimizer state, gradients, and parameters
across ranks:

- **ZeRO-1**: optimizer state sharded.
- **ZeRO-2**: optimizer state + gradients sharded.
- **ZeRO-3**: optimizer state + gradients + parameters
  sharded.

**FSDP** (Fully Sharded Data Parallel, PyTorch's re-
implementation of ZeRO-3) is the modern API. The current
generation is **FSDP-2** (sometimes called "FSDP v2"),
which uses `fully_shard` and the new
`torch.distributed.fsdp.fully_shard` module.

The key operations:

- **All-gather** at the start of a forward pass: each rank
  reconstructs the full block of parameters it's about to
  compute on. Compute is local.
- **Reduce-scatter** at the end of the backward pass: each
  rank ends up with its shard of the gradients (already
  summed across ranks).
- **Re-shard** after the forward-backward: each rank frees
  the gathered parameters and goes back to its shard.

Memory per rank: `O(model_size / world_size)` — at 4 ranks,
each rank holds 1/4 of the model. This is what makes a 1.86 B
model fit on 4 × 80 GB A100s.

## Math derivation

### Memory per component

For a model with `P` parameters, `world_size = W`, and mixed
precision (BF16 params + FP32 master + FP32 AdamW state):

- **BF16 parameters** (sharded): `P / W × 2 bytes`.
- **BF16 gradients** (sharded): `P / W × 2 bytes`.
- **FP32 master weights** (sharded): `P / W × 4 bytes`.
- **AdamW state** (m, v, FP32): `P / W × 8 bytes`.
- **All-gather activations** (per-block): `O(block_size ×
  forward_activations)`.
- **Activations** (per-micro-batch): `O(B × T × dim ×
  L_block)` where `L_block` is the block count.

### Full-shard savings

At `P = 1.86 B`, `W = 4`:

- BF16 params: `1.86e9 / 4 × 2 = 930 MB`
- BF16 grads: `930 MB`
- FP32 master: `1.86e9 / 4 × 4 = 1.86 GB`
- AdamW state: `1.86e9 / 4 × 8 = 3.72 GB`
- All-gather activations: ~3.5 GB (variable, depends on
  block size)
- Total per rank: ~10 GB

A100 80 GB has 8× headroom.

### Communication cost

Each forward pass triggers `n_blocks` all-gathers; each
backward triggers `n_blocks` reduce-scatters. The total
communication volume per step is roughly:

```
volume = 2 × P × 2 bytes (BF16) = 7.4 GB   (one forward + one backward)
```

Across 4 ranks, this is ~1.85 GB per rank of all-gather
traffic. With NVLink at 600 GB/s, this is ~3 ms per step
— negligible relative to the ~5 s per step compute.

### BF16 mixed precision

FSDP-2 supports per-parameter mixed precision via
`MixedPrecision`. The recipe:

- Parameters stored as BF16 in the FSDP unit.
- Gradients computed as BF16 during all-gather.
- Master weights stored as FP32 in the optimizer (per
  `optimizer.master_weights_dtype = "float32"`).
- The optimizer step is computed in FP32.
- The master weights are cast back to BF16 for the next
  forward.

The `fsdp_mixed_precision = "bfloat16"` config field
selects this.

## Implementation in HyMo

- `src/hymo/training/fsdp.py:19` — `fsdp_auto_wrap_policy
  (module, recurse, non_blocking)`: returns True for
  `GatedDeltaNetBlock` and `MLABlock`.
- `src/hymo/training/fsdp.py:26` — `wrap_model_with_fsdp(model,
  config, *, world_size=None, auto_wrap_policy=None, **kwargs)`.
- `src/hymo/training/fsdp.py:35-39` — try-import FSDP; if not
  available, return the model unwrapped (so CPU dev runs work).
- `src/hymo/training/fsdp.py:50-63` — full wrapping with
  `MixedPrecision` from `config.fsdp_mixed_precision`.

The auto-wrap policy `fsdp_auto_wrap_policy` wraps each
`GatedDeltaNetBlock` and `MLABlock` as its own FSDP unit.
This means all-gather and reduce-scatter happen at the block
boundary (32 times per forward + 32 per backward), not at
the parameter boundary (hundreds of times per forward).

## Worked example

Production scale (`world_size = 4`, `fsdp = True`,
`fsdp_mixed_precision = "bfloat16"`):

Per-rank memory (`P = 1.86 B`):

| Component | Per rank |
|---|---|
| BF16 params (sharded) | 930 MB |
| BF16 grads (sharded) | 930 MB |
| FP32 master (sharded) | 1.86 GB |
| AdamW state (m, v) | 3.72 GB |
| All-gather activations (peak) | 3.5 GB |
| Activations (per micro-batch) | 1.5 GB |
| **Peak total** | **~10 GB / rank** |

A100 80 GB has 8× headroom — the dominant bottleneck is
**compute time**, not memory.

Communication:

```
Volume per optimizer step (forward + backward) = 2 × P × 2 bytes = 7.4 GB
Per rank (NVLink × 4) = 1.85 GB ≈ 3 ms
```

At 8 s per step (the production target), communication is
< 0.1% of the wall-clock.

## Interview Q&A

**Q1. Why FSDP and not DDP?**

> A: DDP replicates the full model on every rank. At
> `P = 1.86 B`, each rank holds 1.86 GB of BF16 parameters
> (plus 7.4 GB of optimizer state) — 9.2 GB just for the
> model and optimizer, plus activations. A100 80 GB can fit
> this, but it's tight. FSDP shards the model across ranks,
> so each rank holds only `P / W` of everything. With `W =
> 4`, each rank holds 1/4 of the model. Plus, the all-gather
> + reduce-scatter is bandwidth-cheap relative to the
> forward+backward.

**Q2. Why full sharding (ZeRO-3 / FSDP) instead of ZeRO-1 or 2?**

> A: At 1 B+ params, the optimizer state alone is 2× the
> parameters (FP32 master + AdamW `m` + `v`). For `P = 1.86 B`,
> that's 2 × 1.86 GB = 3.7 GB per rank just for the
> optimizer state. ZeRO-1 shaves that by `W`; ZeRO-3 shaves
> the parameters and gradients too. For 4 ranks, the savings
> are ~1.8 GB (ZeRO-1) vs ~7.4 GB (ZeRO-3).

**Q3. Why auto-wrap at the block level (not the parameter level)?**

> A: All-gather and reduce-scatter have fixed overhead per
> call (kernel launch, sync). At the parameter level, you'd
> have hundreds of all-gathers per forward. At the block
> level, you have 32. The trade-off is that each all-gather
> is larger (a whole block), but the per-call overhead is
> amortized.

**Q4. Why `world_size = 4` and not 8?**

> A: For the 1.86 B model on 4 × A100 80 GB, 4 ranks is
> enough to fit the model + activations + optimizer state
> with ~7× headroom. Doubling to 8 ranks halves the per-rank
> memory but halves the per-rank compute; the throughput
> per dollar is roughly the same. HyMo's 4-rank target is
> the empirical sweet spot for this model size.

**Q5. What happens if `torch.distributed.fsdp` is not
available?**

> A: `wrap_model_with_fsdp` returns the model unwrapped. This
> is the case for CPU dev runs and CI. The trainer still
> works (single-rank, no sharding), but the model uses ~9.2
> GB of RAM for parameters + AdamW state. The test suite
> uses the tiny config (~760 K params) so this isn't a
> problem.

**Q6. What is `fully_shard` (FSDP-2's new API)?**

> A: It's a module-level function that shards a module's
> parameters, gradients, and optimizer state. Called per
> block (or per parameter, depending on the granularity
> you want). The new FSDP-2 API is `torch.distributed.fsdp
> .fully_shard(module)` which in-place modifies the
> module to use FSDP. The legacy `FSDP` wrapper is what
> `wrap_model_with_fsdp` uses — both are supported in
> PyTorch 2.5+.

**Q7. Why does `MixedPrecision` need `bfloat16` and not
`float16`?**

> A: `float16` requires a `GradScaler` to handle
> underflow; `bfloat16` has the same exponent range as
> `float32` (no overflow under reasonable magnitudes) and
> no scaler is needed. HyMo's `training.fsdp_mixed_precision
> = "bfloat16"` skips the GradScaler entirely.

## Cross-links

- [`learning_docs/3_Training_Pipeline.md`](../../learning_docs/3_Training_Pipeline.md) §5
  (FSDP wrapping).
- [`learning_docs/4_Optimizations.md`](../../learning_docs/4_Optimizations.md) §7
  (overview of FSDP-2 + BF16).
- [`concepts/07-muon-optimizer.md`](07-muon-optimizer.md) —
  AdamW state is sharded by FSDP.
- [`concepts/06-mup-init.md`](06-mup-init.md) — μP init is
  applied before FSDP wrapping.
