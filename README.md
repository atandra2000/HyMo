<div align="center">
  
# 🧬 HyMo: Hybrid Language Model

**A next-generation, highly optimized 1.86B hybrid model architecture** <br/> 
combining Gated Delta Networks, Multi-Head Latent Attention, and Asymmetric MoE.

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Testing: pytest](https://img.shields.io/badge/testing-pytest-green.svg)](https://docs.pytest.org/en/stable/)

</div>

---

## 📖 Overview

**HyMo** is a state-of-the-art 1.86B parameter (750M-active) language model designed from the ground up for maximum inference efficiency without sacrificing representational capacity. 

It features a bespoke **hybrid architecture** that strategically fuses multiple cutting-edge sub-layers:
- **GDN (Gated Delta Net):** Lightning-fast linear attention mechanism for the majority of sequence processing.
- **MLA (Multi-Head Latent Attention):** Full attention heads placed at specific intervals to retain high-fidelity global context.
- **Asymmetric MoE:** Sparse Mixture of Experts deployed exclusively on attention layers, while linear layers utilize dense SwiGLU blocks.
- **MTP (Multi-Token Prediction):** Advanced predictive heads for enhanced sample efficiency.

Built to be pre-trained natively on 30B tokens at 40× params-in-tokens ratio.

---

## ✨ Key Features

- 🏗️ **Architectural Innovation**: Asymmetric feed-forward design combining dense layers with sparse experts (MoE).
- ⚡ **Optimized Kernels**: Integrated with custom `Triton` kernels for heavily optimized GDN recurrence (fallback to `fla` or native PyTorch).
- 🏋️ **Advanced Training Stack**: Features full **FSDP-2** parameter sharding, **Gradient Accumulation**, `torch.compile` graph optimization, `NorMuon` + `CautiousAdamW` dual-optimizer routing, and `JointWSDScheduler`.
- 📊 **Robust Data Pipeline**: Natively handles 10 data sources with a BPE-64k tokenizer, zero-copy `np.memmap` lazy loading, and persistent prefetched multi-processing to keep A100 GPUs saturated.
- 🧪 **Cool-by-Design Testing**: Our test suite dynamically mocks the 1.86B model with a tiny 760K-param surrogate so tests run lightning fast on laptops without GPUs (M1 friendly!).

---

## 🛠️ Installation

Requires Python 3.10+.

Clone the repository and use `uv` to install in editable mode with development and training extras:

```bash
git clone https://github.com/your-username/hymo.git
cd hymo

# Install core + train + dev dependencies
uv sync --all-extras
```

> **Note:** The `train` extra includes `triton` (for our custom GDN kernels) and `lm-eval` (for harness evaluation). 

---

## 🚀 Quick Start

### Model Construction

Easily build the model from a YAML configuration or a dataclass config:

```python
import torch
from hymo.core import load_config
from hymo.models import build_hymo

# Load configuration
config = load_config("configs/hymo_750m.yaml")

# Initialize the model (μP init is automatically handled)
model = build_hymo(config)
print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
```

### Running the Test Suite

HyMo has a rigorous, CPU-friendly test suite that you can run locally in seconds:

```bash
# Run the fast, default test suite (skips heavy model builds)
pytest tests/ -v

# Run type checking and linting
mypy src/hymo
ruff check src/hymo
```

To run tests that actually instantiate the full 1.86B model (typically reserved for GPU CI), use:
```bash
pytest tests/ --run-heavy
```

---

## 📂 Project Structure

```text
HyMo/
├── configs/             # YAML configurations (e.g., hymo_750m.yaml)
├── src/hymo/
│   ├── core/            # Configuration definitions, types, and exceptions
│   ├── models/          # Model definition: GDN, MLA, MoE, MTP, RoPE, and μP init
│   ├── training/        # Trainer, dual-optimizers, WSD scheduler, and FSDP
│   ├── data/            # Tokenization, source loaders, and data sharding
│   ├── eval/            # Eval harnesses and baselines
│   └── registry/        # Global registry for plug-and-play components
├── tests/               # Unit and integration tests
└── docs/                # Architectural designs and roadmap
```

---

## 📈 Project Status & Roadmap

- [x] **Phase 1: Repository Foundation** — Clean architecture, robust config, public API.
- [x] **Phase 2: Algorithmic Implementation** — Functional GDN, MLA, MoE, MTP, and Fusion stack implementations.
- [x] **Phase 3: Training Infrastructure** — Trainer loops, NorMuon + AdamW partitioning, validation pipelines, and DCP checkpointing.
- [x] **Phase 4: Data & Eval Pipelines** — 10 distinct data loaders, optimized tokenizers, fast bin sharding, and lm-eval harness.
- [ ] **Phase 5: Deployment & 30B Run** — Pending execution.

For a deep dive into the underlying architecture and future milestones, please refer to:
- 📐 [Architecture & Design Document](docs/HyMo-Design.md)
- 🗺️ [Implementation Roadmap](docs/HyMo-Roadmap.md)

---

## 📄 License

This project is licensed under the terms of the **Apache 2.0 License**.
