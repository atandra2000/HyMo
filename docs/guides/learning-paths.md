# HyMo — Learning Paths

> Three ordered routes through the corpus. Each row names a doc and what
> it adds; every path is cumulative — later rows assume the earlier ones.
> If you only have one hour, do the beginner path and stop after
> [`references/config.md`](../references/config.md).

## Beginner — "What is HyMo and how do I run it?" (1–2 h)

| # | Read | What you learn |
|---|---|---|
| 1 | [`README.md`](../../README.md) | The pitch: 32-layer 3:1 GDN:MLA hybrid, 434 M active / 1.13 B stored params, 30 B-token pretrain target. |
| 2 | [`guides/quickstart.md`](quickstart.md) | Install, load `configs/hymo_750m.yaml`, build the tiny fixture model, run the CPU test suite. |
| 3 | [`references/config.md`](../references/config.md) | How YAML becomes a frozen `src/hymo/core/config.py:HyMoConfig`; every field, every invariant (`src/hymo/core/config_validation.py:validate_full_config`). |
| 4 | [`concepts/model-architecture.md`](../concepts/model-architecture.md) | Line-by-line walkthrough of `src/hymo/models/`: `HyMo`, `MLABlock`, `GatedDeltaNetBlock`, `DeepSeekMoE`, `MultiTokenPrediction`. |
| 5 | [`guides/glossary.md`](glossary.md) | The vocabulary used everywhere else — keep it open while reading. |
| 6 | [`concepts/hybrid-ratio.md`](../concepts/hybrid-ratio.md) | Why every 4th layer is MLA: what GDN buys vs MLA, and how the 3:1 mix is derived from one config number. |

**Landmark check:** you can explain why every 4th layer is MLA (positions
0, 4, …, 28) and what `tie_embeddings` does to `src/hymo/models/model.py:HyMo`.

## Intermediate — "How is it trained?" (1 day)

| # | Read | What you learn |
|---|---|---|
| 1 | [`concepts/gdn-and-mla.md`](../concepts/gdn-and-mla.md) | Mechanism tiers: gated delta rule, latent KV compression, fine-grained MoE, MTP heads. |
| 2 | [`training.md`](../training.md) §Data Pipeline | BPE-64k + 256-byte `src/hymo/data/tokenizer.py:ExtendedTokenizer`, shards, `data/tokens/val.bin`. |
| 3 | [`concepts/optimization.md`](../concepts/optimization.md) | NorMuon vs CautiousAdamW, the WSD schedule, FSDP-2, initialization. |
| 4 | [`training.md`](../training.md) §Training Pipeline | `src/hymo/training/trainer.py:Trainer`, the parameter partition (`src/hymo/training/partition.py:ParameterPartition`), WSD phases, DCP checkpointing. |
| 5 | [`references/api.md`](../references/api.md) | The public surface at a glance — model factory, trainer, checkpoints. |
| 6 | [`concepts/asymmetric-moe.md`](../concepts/asymmetric-moe.md) | The MoE placement (8 of 32 layers) and 16+1/top-2 configuration, with the stored-vs-active budget. |
| 7 | [`concepts/dual-optimizers.md`](../concepts/dual-optimizers.md) | Ground truth on NorMuon + CautiousAdamW: the partition rules, RMS matching, FP32 masters, trainer wiring. |
| 8 | [`concepts/mtp.md`](../concepts/mtp.md) | The chained MTP heads: objective, loss weights, trainer wiring, zero inference cost. |

**Landmark check:** you can say which optimizer receives the embedding
matrix and why (`goes_to_adamw`), and what changes at `warmup_frac` /
`stable_frac` / `decay_frac` boundaries in
`src/hymo/training/scheduler.py:JointWSDScheduler`.

## Expert — "How would I extend or debug it?" (2–3 days)

| # | Read | What you learn |
|---|---|---|
| 1 | [`concepts/kernels.md`](../concepts/kernels.md) | The one sanctioned Triton kernel: `src/hymo/models/gdn_triton.py:triton_gated_delta_rule`, FA2-style recompute backward, CPU-reference test contract. |
| 2 | [`training.md`](../training.md) §Distributed | FSDP-2 wrapping (`src/hymo/training/fsdp.py:wrap_model_with_fsdp`), auto-wrap policy, mixed precision, 4-rank topology. |
| 3 | [`concepts/design.md`](../concepts/design.md) | The full v1.0 design document — the "why" behind every choice, including Phase-4 deferrals. |
| 4 | [`training.md`](../training.md) §Checkpointing | `src/hymo/training/checkpoint.py:CheckpointState` + DCP resumability, RNG capture. |
| 5 | [`references/config.md`](../references/config.md) §Deriving | `src/hymo/core/config.py:derive_config` for ablation variants without mutating the frozen base. |
| 6 | `tests/` | The testing rules: tiny-config defaults, `@pytest.mark.heavy` for the 1.13 B construct, gates (`mypy --strict`, `ruff`). |
| 7 | [`concepts/fsdp2.md`](../concepts/fsdp2.md) | Full-sharding mechanics, block-level sharding units, per-rank memory derivation — and the honest wiring status (F5). |

**Landmark check:** you can explain the Triton contract (opt-in,
`HAS_TRITON`, no silent fallback), where validation loss enters
`Trainer.train_step`, and what a checkpoint contains beyond weights.

## Pointers

- New to transformers entirely? Read [`concepts/model-architecture.md`](../concepts/model-architecture.md) §1 before the beginner path.
- Interview mode? The README's reading order 1 is a compressed version of the beginner + intermediate paths.
- After any code change: `uv run pytest tests/ -v`, then the doc gates (`python3 scripts/check_docs.py --coverage --links`).
