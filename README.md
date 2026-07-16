# HyMo

A 750M-active hybrid language model combining **Gated Delta Net** (linear attention), **Multi-Head Latent Attention** (full attention), and an asymmetric feed-forward block (MoE on attention layers, dense SwiGLU on linear layers). Pre-trained from scratch on 30B tokens at 40× params-in-tokens.

## Status

- **Phase 1 (Repository Foundation):** in progress — see [docs/HyMo-Design.md](docs/HyMo-Design.md) and [docs/HyMo-Roadmap.md](docs/HyMo-Roadmap.md).
- **Phase 2+ (Algorithmic implementation):** not started.

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
