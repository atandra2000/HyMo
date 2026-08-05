# HyMo — Quickstart

> How to install HyMo, run the test suite and gates, and make a first forward pass with the ~750 M-active model config. The canonical layout overview and reading orders live in [docs/README.md](../README.md).

## Installation

Requires Python 3.11+ and [uv](https://github.com/astral-sh/uv).

```bash
git clone https://github.com/atandra-bharati/hymo.git
cd hymo
uv sync --all-extras
```

Core dependencies: PyTorch ≥ 2.5, NumPy, PyYAML, HuggingFace `tokenizers` + `datasets`.

Optional extras:
- `train` — Triton (Linux), `lm-eval`, `wandb`
- `dev` — `pytest`, `mypy`, `ruff`

## First forward pass

```python
import torch
from hymo import load_config, build_hymo

config = load_config("configs/hymo_750m.yaml")
model = build_hymo(config)

x = torch.randint(0, config.model.vocab_size, (2, 128))
logits = model(x)  # HyMo.forward returns next-token logits only
print(logits.shape)  # (2, 128, 64256)

# For MTP (multi-token prediction) auxiliary heads, the trainer uses
# model.forward_with_hidden(tokens) which returns (logits, hidden).
# See src/hymo/training/trainer.py::train_step for the MTP loss wiring.
```

## Tests and gates

```bash
pytest tests/ -v                # ~1 min on CPU; heavy tests skipped (280 passed / 36 skipped; 316 collected as of 2026-08-04)
pytest tests/ --run-heavy       # includes full 1.86B model construction
mypy src/hymo                   # type gate
ruff check src/hymo             # lint gate
```

## Configuration

All hyperparameters live in YAML configs under `configs/`. The primary config is `configs/hymo_750m.yaml`, organized into 5 frozen dataclass groups (`ModelConfig`, `OptimizerConfig`, `SchedulerConfig`, `TrainingConfig`, `RunConfig`). See [references/config.md](../references/config.md) for the full field reference and [training.md](../training.md) for the training pipeline.

## References

- [../README.md](../README.md) — project overview and architecture table.
- [training.md](../training.md) — the training pipeline.
- [references/config.md](../references/config.md) — the config system.
- [concepts/model-architecture.md](../concepts/model-architecture.md) — the model walkthrough.
- Source: `src/hymo/__init__.py` (`load_config`, `build_hymo`), `src/hymo/models/model.py` (`build_hymo`), `src/hymo/core/config.py` (`load_config`).
