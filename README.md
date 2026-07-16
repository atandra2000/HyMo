# HyMo

A 750M-active hybrid language model combining **Gated Delta Net** (linear attention), **Multi-Head Latent Attention** (full attention), and an asymmetric feed-forward block (MoE on attention layers, dense SwiGLU on linear layers). Pre-trained from scratch on 30B tokens at 40× params-in-tokens.

## Status

- **Phase 1 (Repository Foundation):** ✅ Shipped — 47 Python files, full public API, frozen config, all interfaces. See [docs/PHASE_1_DELIVERY.md](docs/PHASE_1_DELIVERY.md).
- **Phase 2 (Algorithmic Model Implementation):** ✅ Completed — every `forward` in `hymo.models` is real (GDN, MLA, MoE, MTP, RoPE, μP init, 32-layer HyMo stack). See [Phase 2 delivery note](docs/HyMo-Roadmap.md#user-content-phase-2-delivery-note-algorithmic-model-implementation).
- **Phase 3 (Training Infrastructure):** ✅ Completed — `NorMuon`/`CautiousAdamW` with FP32 master weights, `JointWSDScheduler`, real held-out validation, DCP checkpoint save/load, `Trainer` with MTP loss, FSDP-aware grad norm, NaN-skip, EMA gate bias, eval every 2k steps. See [Phase 3 delivery note](docs/HyMo-Roadmap.md#phase-3-delivery-note-training-infrastructure).
- **Phase 4 (Data pipeline, eval, ablations):** ✅ Completed — 10 real source loaders, BPE-64k + byte-level tokenizer, shard writer/reader, 6-eval harness, ablation framework (4 families). See [Phase 4 delivery note](docs/HyMo-Roadmap.md#phase-4-delivery-note-data-pipeline--eval--ablations).
- **Phase 5 (Deployment, 30B run):** Pending — see [docs/HyMo-Roadmap.md](docs/HyMo-Roadmap.md).

## Layout

```
src/hymo/
├── core/        # config, types, exceptions, validation
├── models/      # GDN, MLA, MoE, MTP, fusion stack, RoPE, init
├── training/    # optimizer, scheduler, trainer, FSDP, checkpoint, validation
├── data/        # tokenizer, sources, sharding, loader
├── eval/        # lm-eval harness, baselines, comparison
├── ablations/   # v1.1 ablation framework (deferred)
├── registry/    # named-constructor registries
└── utils/       # logging, metrics, callbacks, registry, paths
```

## Install (editable)

```bash
pip install -e ".[dev,train]"
```

The `train` extra pulls `fla` (the GDN kernel) and `lm-eval` (the held-out
evaluation harness). The `dev` extra pulls pytest, mypy, and ruff.

## Quick check

```bash
pytest tests/        # default run: CPU-friendly, ~1 min, heavy tests auto-skipped
pytest tests/ --run-heavy   # opt-in: also runs tests that build the 1.86B model
mypy src/hymo
ruff check src/hymo
```

> **Tests stay cool by design.** The default `pytest` run never builds the
> 1.86 B-parameter production model — it uses a tiny (~760 K-param) config
> for everything. Tests that do need production scale are marked
> `@pytest.mark.heavy` and are skipped unless you pass `--run-heavy`
> (run those on CI / a GPU pod). See `AGENTS.md` for the full test
> rules.

## Documentation

- [docs/HyMo-Design.md](docs/HyMo-Design.md) — architecture (source of truth for design)
- [docs/HyMo-Roadmap.md](docs/HyMo-Roadmap.md) — implementation plan (source of truth for tasks)

## License

Apache-2.0.
