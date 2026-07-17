# HyMo

**A 750M-active / 1.86B-stored hybrid language model** combining Gated Delta Networks (linear attention), Multi-Head Latent Attention (full attention), and Asymmetric Mixture-of-Experts. Pre-trained from scratch on 30B tokens — targeting held-out FineWeb-Edu PPL ≤ 2.10.

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Type checked: mypy --strict](https://img.shields.io/badge/mypy--strict-3178C6)](https://mypy-lang.org/)
[![Pre-commit: none](https://img.shields.io/badge/pre--commit-none-lightgrey)](https://pre-commit.com/)

---

## Architecture

HyMo is a 32-layer stack with a **3:1 GDN-to-MLA ratio** — 24 linear-attention layers (Gated Delta Net) interleaved with 8 full-attention layers (Multi-Head Latent Attention). This hybrid design captures long-range context through sparse full-attention anchors while processing the bulk of the sequence through a linear-complexity recurrence.

| Component | Layers | Type | Description |
|---|---|---|---|
| **GDN** | 24 | Linear attention | Gated Delta Net with 1D selective scan, partial RoPE, and Dense SwiGLU FFN |
| **MLA** | 8 | Full attention | Multi-Head Latent Attention (DeepSeek-style low-rank KV compression, 4 KV heads) |
| **MoE** | On MLA layers | Sparse FFN | DeepSeekMoE (16 routed + 1 shared expert, top-2 routing) |
| **FFN** | On GDN layers | Dense | SwiGLU, `inter_dim = 2560` |
| **MTP** | 2 heads | Multi-token prediction | Auxiliary heads predicting next 2 tokens, weighted `[0.3, 0.1]` |

Key architectural invariants:
- **Asymmetric feed-forward**: MoE is deployed exclusively on attention layers (MLA blocks); GDN blocks use dense SwiGLU throughout
- **partial-RoPE**: Applied on the first 25% of `head_dim` at every position across all 32 layers
- **MQA-4**: MLA compresses to 4 KV groups for efficient inference
- **FP32 master weights**: Full numerical stability at the cost of 2× optimizer state
- **Muon/AdamW dual optimizer**: NorMuon drives attention + GDN matrices; AdamW handles embeddings, norms, gates, and MoE experts
- **μP initialization**: Maximal update parameterization for stable training at scale

---

## Features

- **Custom Triton GDN kernel** — Optimized 1D selective scan with fused recurrence, chunked with `chunk_size=64`. Falls back to native PyTorch implementation on non-Linux platforms.
- **FSDP-2 parameter sharding** — Full-sharding with BF16 mixed precision, gradient clipping by global norm, and NaN-step skipping with configurable tolerance.
- **10-source data pipeline** — BPE-64k + 256-byte tokenizer, `np.memmap` zero-copy lazy loading, multi-process prefetching via `ShardWriter`/`ShardDataset`.
- **Ablation framework** — 4 families of config derivation (GDN variants, MLA variants, MoE variants, optimizer variants) for systematic experimentation.
- **Cool-by-design test suite** — Full model is replaced by a ~760K-parameter surrogate in default tests. The 1.86B model is only constructed when `--run-heavy` is passed. Default `pytest` run finishes in ~1 minute on an M1 Air.
- **DCP checkpointing** — Distributed checkpoint save/load with resume-from-arbitrary-step support.

---

## Installation

Requires Python 3.11+ and [uv](https://github.com/astral-sh/uv).

```bash
git clone https://github.com/atandra-bharati/hymo.git
cd hymo
uv sync --all-extras
```

Core dependencies: PyTorch ≥2.5, NumPy, PyYAML, HuggingFace `tokenizers` + `datasets`, `safetensors`.

Optional extras:
- `train` — Triton (Linux), `lm-eval`, `wandb`
- `dev` — `pytest`, `mypy`, `ruff`

---

## Quick Start

```python
import torch
from hymo import load_config, build_hymo

config = load_config("configs/hymo_750m.yaml")
model = build_hymo(config)

x = torch.randint(0, config.model.vocab_size, (2, 128))
logits, aux_losses = model(x)
print(logits.shape)  # (2, 128, 64256)
```

Run the test suite:
```bash
pytest tests/ -v                # ~1 min on CPU, skips full model tests
pytest tests/ --run-heavy       # includes full 1.86B model construction
mypy src/hymo
ruff check src/hymo
```

---

## Configuration

All hyperparameters live in YAML configs under `configs/`. The primary config is [`configs/hymo_750m.yaml`](configs/hymo_750m.yaml), which defines 5 frozen dataclass groups:

| Group | Class | Key knobs |
|---|---|---|
| `model` | `ModelConfig` | 32 layers, dim 896, 16 heads, 16 MoE experts, MTP depth 2 |
| `optimizer` | `OptimizerConfig` | NorMuon LR 0.02, AdamW LR 3e-4, FP32 master weights |
| `scheduler` | `SchedulerConfig` | WSD schedule, 57k total steps, 2% warmup, linear decay |
| `training` | `TrainingConfig` | Micro-batch 4, grad accum 8, FSDP BF16, eval every 2k steps |
| `run` | `RunConfig` | Seed 42, deterministic, distributed |

Derive ablation configs via `hymo.core.config.derive_config()` — see the [ablation framework](src/hymo/ablations/__init__.py).

---

## Project Structure

```
hymo/
├── configs/                  # YAML configurations
│   ├── hymo_750m.yaml        # Primary v1.0 config
│   └── hymo_mixture.yaml     # Data mixture config
├── src/hymo/
│   ├── core/                 # Config dataclasses, types, exceptions, validation
│   ├── models/               # GDN, MLA, MoE, MTP, RoPE, μP init, Triton kernel
│   ├── training/             # Trainer, dual optimizer, WSD scheduler, FSDP-2, checkpoint
│   ├── data/                 # Tokenizer, 10 source loaders, sharding pipeline
│   ├── eval/                 # lm-eval harness, baselines, comparison runner
│   ├── ablations/            # Config derivation for systematic ablations
│   └── registry/             # Global plug-and-play component registry
├── tests/
│   ├── unit/                 # Module-level unit tests
│   ├── integration/          # Cross-module integration tests
│   └── conftest.py           # Tiny model fixtures (760K params)

```

---

## Project Status

| Phase | Status | Description |
|---|---|---|
| 1 — Repository foundation | ✅ **Done** | Clean architecture, public API, config system, CI gates |
| 2 — Algorithmic model | ✅ **Done** | GDN, MLA, MoE, MTP, RoPE, μP init — all forward/backward finite |
| 3 — Training infrastructure | ✅ **Done** | Trainer, dual optimizer, WSD scheduler, FSDP-2, DCP checkpointing |
| 4 — Data & eval pipelines | ✅ **Done** | 10-source loader, tokenizer, sharding, eval harness, ablation framework |
| 5 — Deployment & 30B run | ⏳ **Pending** | RunPod scripts, 30B-token pre-training on 4× A100 80GB SXM |

---

## License

Apache 2.0. See [LICENSE](LICENSE).
