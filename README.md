<div align="center">

# HyMo

**A 750M-active / 1.86B-stored hybrid language model** — Gated Delta Networks (linear attention) × Multi-Head Latent Attention (full attention) with Asymmetric Mixture-of-Experts. Pre-trained from scratch on 30B tokens, targeting held-out FineWeb-Edu perplexity ≤ 2.10.

*The flagship model of the CoreProjects portfolio.*

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![PyTorch 2.5+](https://img.shields.io/badge/PyTorch-2.5%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000?logo=astral&logoColor=white)](https://github.com/astral-sh/ruff)
[![Type checked: mypy](https://img.shields.io/badge/mypy-checked-3178C6?logo=mypy&logoColor=white)](https://mypy-lang.org/)

</div>

---

## Documentation

The full corpus lives under [`docs/`](docs/README.md) — reading orders, the model walkthrough, mechanism deep-dives, the design doc, config/API references, and the training pipeline. Quick links:

- [`docs/guides/quickstart.md`](docs/guides/quickstart.md) — install, first forward pass, tests and gates
- [`docs/concepts/model-architecture.md`](docs/concepts/model-architecture.md) — the code walkthrough
- [`docs/concepts/design.md`](docs/concepts/design.md) — the v1.0 architecture & design document
- [`docs/references/config.md`](docs/references/config.md) — the typed-config reference
- [`docs/training.md`](docs/training.md) — the training pipeline

## Why HyMo

Transformer attention scales quadratically with sequence length — the dominant cost of pretraining at scale. **HyMo is a hybrid**: it processes the bulk of the sequence through *linear-complexity* recurrence (Gated Delta Net) and reserves sparse *full-attention* anchors for genuine long-range reasoning. The result is a model that trains and infers far cheaper than an all-attention transformer of equal quality, while keeping the expressivity where it matters.

The headline design choices:

- **3:1 GDN-to-MLA ratio** — 24 linear-attention layers interleaved with 8 full-attention layers, so 75% of the stack is sub-quadratic.
- **Asymmetric feed-forward** — MoE (sparse, expensive) lives only on the 8 full-attention MLA blocks; the 24 linear GDN blocks use cheap dense SwiGLU. Compute is spent where it buys the most.
- **Custom Triton GDN kernel** — a fused 1D selective scan with chunked recurrence (`chunk_size=64`), written by hand in `src/hymo/models/gdn_triton.py` for throughput and numerical parity with the eager reference. There is no `fla`-library dependency — the only sanctioned kernel path is this hand-written Triton kernel.

---

## Architecture

HyMo is a 32-layer stack with a **3:1 GDN-to-MLA ratio**.

| Component | Layers | Type | Description |
|---|---|---|---|
| **GDN** | 24 | Linear attention | Gated Delta Net with 1D selective scan, partial RoPE, dense SwiGLU FFN |
| **MLA** | 8 | Full attention | Multi-Head Latent Attention (DeepSeek-style low-rank KV compression, 4 KV heads) |
| **MoE** | On MLA layers | Sparse FFN | DeepSeekMoE (16 routed + 1 shared expert, top-2 routing) |
| **FFN** | On GDN layers | Dense | SwiGLU, `inter_dim = 2560` |
| **MTP** | 2 heads | Multi-token prediction | Auxiliary heads predicting next 2 tokens, weighted `[0.3, 0.1]` |

**Model footprint (v1.0 config):** `dim = 896`, `n_heads = 16`, `max_seq_len = 4096`, `vocab_size = 64,256` (BPE-64k + 256-byte tokenizer). ~750M active / ~1.86B stored parameters.

Key architectural invariants:

- **Asymmetric feed-forward** — MoE exclusively on MLA blocks; GDN blocks stay dense SwiGLU.
- **Partial RoPE** — applied to the first 25% of `head_dim` at every position across all 32 layers.
- **MQA-4** — MLA compresses to 4 KV groups for efficient inference.
- **FP32 master weights** — full numerical stability; optimizer state held in float32.
- **NorMuon / AdamW dual optimizer** — NorMuon drives attention + GDN 2D matrices; AdamW handles embeddings, norms, gates, and MoE experts. Cautious weight decay enabled.
- **Initialization** — PyTorch module defaults plus the inline MoE-gate init (`bias=0`, `std=0.006`) and the GDN recurrence init (`A_log`, `dt_bias`, `D`). The designed μP init was never wired into `build_hymo` and was removed in the 2026-08-04 cleanup (see [`docs/concepts/optimization.md`](docs/concepts/optimization.md)).
- **Logit softcap (15.0)** — bounds logits for training stability.

---

## Features

- **Custom Triton GDN kernel** — a hand-written fused 1D selective scan in `src/hymo/models/gdn_triton.py` (serial time loop, FP32 accumulation; the parallel-chunk algorithm from the GDN paper remains the design intent). Linux only — Triton does not ship on macOS/Windows; on those platforms the eager path in `src/hymo/models/gdn.py` is the reference and is what unit tests exercise.
- **FSDP-2 full parameter sharding** — BF16 mixed precision, gradient clipping by global norm, NaN-step skipping with configurable tolerance.
- **10-source data pipeline** — BPE-64k + 256-byte tokenizer and the held-out FineWeb-Edu validation-set builder remain in-repo; the 10 streaming loaders and shard writer moved to the workspace `LLM/shared_data/` package in the 2026-08-04 cleanup (the trainer consumes a raw `data_iter`).
- **Ablation framework** — 4 families of config derivation (GDN variants, MLA variants, MoE variants, optimizer variants) via `dataclasses.replace` on the frozen configs; the in-repo `ablations/` package was removed in the 2026-08-04 cleanup — the derivation helper `derive_config` lives in `hymo.core.config`.
- **Cool-by-design test suite** — the full 1.86B model is never built in default tests; a ~760K-param surrogate is used instead. Heavy tests (full model construction) are opt-in via `--run-heavy`. Default `pytest` finishes in ~1 minute on an M1 Air.
- **DCP checkpointing** — distributed checkpoint save/load with resume-from-arbitrary-step support.

---

## Installation and Quick Start

Install, run the first forward pass, and run the test suite / gates in
[`docs/guides/quickstart.md`](docs/guides/quickstart.md). The 30-second
version:

```bash
uv sync --all-extras
```

```python
import torch
from hymo import load_config, build_hymo

config = load_config("configs/hymo_750m.yaml")
model = build_hymo(config)

x = torch.randint(0, config.model.vocab_size, (2, 128))
logits = model(x)  # HyMo.forward returns next-token logits only
print(logits.shape)  # (2, 128, 64256)
```

```bash
pytest tests/ -v                # ~1 min on CPU; heavy tests skipped (191 passed / 35 skipped as of 2026-08-05)
pytest tests/ --run-heavy       # includes full 1.86B model construction
mypy src/hymo                   # type gate
ruff check src/hymo             # lint gate
```

---

## Configuration

All hyperparameters live in YAML configs under `configs/`. The primary config is [`configs/hymo_750m.yaml`](configs/hymo_750m.yaml), organized into 5 frozen dataclass groups:

| Group | Class | Key knobs |
|---|---|---|
| `model` | `ModelConfig` | 32 layers, dim 896, 16 heads, 16 MoE experts, MTP depth 2, seq 4096 |
| `optimizer` | `OptimizerConfig` | NorMuon LR 0.02, AdamW LR 3e-4, FP32 master weights, cautious WD |
| `scheduler` | `SchedulerConfig` | WSD schedule, ~57.2k total steps, 2% warmup, linear decay |
| `training` | `TrainingConfig` | Micro-batch 4, grad accum 8, FSDP BF16, eval every 2k steps |
| `run` | `RunConfig` | Name + output directory |

Every field, validation rule, and the derivation helper are documented in
[`docs/references/config.md`](docs/references/config.md). Derive config
variants via `hymo.core.config.derive_config()` (e.g. `dataclasses.replace` on sub-configs).

---

## Project Structure

```
hymo/
├── configs/                  # YAML configurations
│   ├── hymo_750m.yaml        # Primary v1.0 config
│   └── hymo_mixture.yaml     # Data mixture config
├── src/hymo/
│   ├── core/                 # Config dataclasses, types, exceptions, validation (PyTorch-free)
│   ├── models/               # GDN, MLA, MoE, MTP, RoPE, Triton kernel
│   ├── training/             # Trainer, dual optimizer, WSD scheduler, FSDP-2, checkpoint
│   └── data/                 # Tokenizer + held-out validation-set builder
├── tests/
│   ├── unit/                 # Module-level unit tests
│   ├── integration/          # Cross-module integration tests
│   └── conftest.py           # Tiny model fixtures (760K params)
```

---

## Project Status

| Phase | Status | Description |
|---|---|---|
| 1 — Repository foundation | ✅ Done | Clean architecture, public API, config system, CI gates |
| 2 — Algorithmic model | ✅ Done | GDN, MLA, MoE, MTP, RoPE, μP init — all forward/backward finite |
| 3 — Training infrastructure | ✅ Done | Trainer, dual optimizer, WSD scheduler, FSDP-2, DCP checkpointing |
| 4 — Data & eval pipelines | ✅ Done | 10-source loader, tokenizer, sharding, eval harness, ablation framework |
| 5 — Deployment & 30B run | ⏳ Pending | RunPod scripts, 30B-token pre-training on 4× A100 80GB SXM |

The architecture, training, data, evaluation, and ablation pipelines are fully implemented. The 30B-token pre-training run on 4× A100 80GB is the remaining milestone.

---

## Engineering Principles

- **Raw PyTorch first** — no HuggingFace `Trainer`, no Lightning. The loop, kernels, and distributed training are hand-written and deeply optimized (`torch.compile`, FSDP-2).
- **Strong typing** — every public function is fully annotated; `mypy --strict` is a gate.
- **No magic numbers** — all hyperparameters live in `configs/hymo_750m.yaml`; code references them via `hymo.core.config`.
- **No circular dependencies** — `core ← utils ← {models, training, data, eval}`; `models` and `training` share state only through config.
- **Fully implemented** — no `NotImplementedError` placeholders for core model logic.

---

## License

Apache 2.0.

---

<div align="center">

**Atandra Bharati** · [GitHub](https://github.com/atandra2000) · [LinkedIn](https://www.linkedin.com/in/atandrabharati) · [Portfolio](https://atandra2000.github.io/mycv)

</div>
