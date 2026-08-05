# HyMo — Quickstart

> How to install HyMo, run the test suite and quality gates, and execute model forward passes, training steps, checkpointing, and validation evaluations. For the reading order guide and corpus overview, see [docs/README.md](../README.md).

---

## 1. Installation

Requires **Python 3.11+** and [uv](https://github.com/astral-sh/uv).

```bash
git clone https://github.com/atandra-bharati/hymo.git
cd hymo
uv sync --all-extras
```

### Core dependencies
- `torch >= 2.5`
- `numpy >= 1.26`
- `pyyaml >= 6.0`
- `tokenizers >= 0.19`
- `datasets >= 2.20`

### Optional dependency groups
- `train`: `triton >= 3.0.0` (Linux), `wandb >= 0.17`
- `dev`: `pytest >= 8.0`, `pytest-cov >= 4.1`, `mypy >= 1.10`, `ruff >= 0.5`, `types-PyYAML`

---

## 2. Basic Workflows

### 2.1 First forward pass (~750M active / 1.86B total)

```python
import torch
from hymo import load_config, build_hymo

# 1. Load the production 750M config
config = load_config("configs/hymo_750m.yaml")

# 2. Instantiate the 32-layer hybrid stack
model = build_hymo(config)

# 3. Next-token forward pass
# Input: batch_size=2, seq_len=128
tokens = torch.randint(0, config.model.vocab_size, (2, 128))
logits = model(tokens)  # HyMo.forward returns next-token logits only

print(f"Logits shape: {logits.shape}")  # (2, 128, 64256)
```

### 2.2 CPU-friendly testing config (~760K params)

For local development or unit testing on laptop hardware without allocating the full 1.86B parameter graph:

```python
from hymo.core.config import ModelConfig, HyMoConfig, TrainingConfig, OptimizerConfig, SchedulerConfig
from hymo.models import HyMo

# Create a tiny model config for fast CPU execution
tiny_config = HyMoConfig(
    model=ModelConfig(
        vocab_size=1000,
        max_seq_len=64,
        n_layers=4,
        dim=64,
        n_heads=4,
        n_kv_groups=2,
        kv_lora_rank=16,
        q_lora_rank=32,
        head_dim=16,
        qk_rope_head_dim=4,
        qk_nope_head_dim=12,
        v_head_dim=16,
        gdn_d_state=8,
        gdn_headdim=8,
        gdn_d_inner=32,
        n_routed_experts=4,
        n_activated_experts=2,
        moe_inter_dim=64,
        mtp_depth=1,
        mtp_loss_weights=(0.3,),
        mtp_inter_dim=64,
    )
)

model = HyMo(tiny_config.model)
print(f"Tiny model parameter count: {model.num_parameters():,}")
```

### 2.3 Executing a training step

```python
import torch
from hymo import load_config, build_hymo
from hymo.training import Trainer

config = load_config("configs/hymo_750m.yaml")
model = build_hymo(config)
trainer = Trainer(config, model)

# Create input tokens (B=2, T=128) and target tokens
tokens = torch.randint(0, config.model.vocab_size, (2, 128))
targets = torch.randint(0, config.model.vocab_size, (2, 128))

# Execute a training step (forward + backward + optimizer step on grad accumulation boundary)
result = trainer.train_step(tokens, targets)

print(f"Step Loss: {result.loss:.4f}")
print(f"Grad Norm: {result.grad_norm:.4f}")
print(f"Muon LR: {result.lr_muon:.6f}, AdamW LR: {result.lr_adamw:.6f}")
```

### 2.4 Checkpoint saving & restoration

```python
from pathlib import Path
import torch
from hymo import load_config, build_hymo
from hymo.training import Trainer

config = load_config("configs/hymo_750m.yaml")
model = build_hymo(config)
trainer = Trainer(config, model)

# Save checkpoint to run output directory (e.g. checkpoints/pretrain/step_4000)
ckpt_dir = trainer.save(tag="step_4000")
print(f"Saved checkpoint to {ckpt_dir}")

# Restore training state into a new trainer instance
model2 = build_hymo(config)
trainer2 = Trainer(config, model2)
restored_step = trainer2.load(ckpt_dir)
print(f"Restored trainer at step {restored_step}")
```

### 2.5 Validation loss evaluation

```python
import torch
from hymo import load_config, build_hymo
from hymo.training.validation import compute_validation_loss

config = load_config("configs/hymo_750m.yaml")
model = build_hymo(config)

# Evaluate validation metrics over 10 batches of synthetic or FineWeb-Edu tokens
val_metrics = compute_validation_loss(
    model,
    batch_size=2,
    seq_len=128,
    vocab_size=config.model.vocab_size,
    num_batches=10,
    device="cpu",
)

print(f"Validation Loss: {val_metrics.loss:.4f}")
print(f"Validation Perplexity: {val_metrics.ppl:.2f}")
```

---

## 3. Tests and Quality Gates

```bash
# Default test suite (~1.8s on CPU; heavy 1.86B model tests auto-skipped)
pytest tests/ -v

# Include full 1.86B model construction and heavy memory allocation tests
pytest tests/ --run-heavy

# Validate doc-code symbol anchors and intra-repo markdown links
python3 tests/test_doc_refs.py --links

# Run static type checking gate
mypy src/hymo

# Run linting gate
ruff check src/hymo tests/
```

---

## 4. Configuration System

All hyperparameters live in YAML configuration files under `configs/`. The production configuration is `configs/hymo_750m.yaml`, structured into 5 frozen dataclass sub-configs (`ModelConfig`, `OptimizerConfig`, `SchedulerConfig`, `TrainingConfig`, `RunConfig`). See [references/config.md](../references/config.md) for the full field reference and [training.md](../training.md) for training pipeline mechanics.

---

## 5. Documentation References

- [../README.md](../README.md) — Project overview and architecture comparison.
- [references/api.md](../references/api.md) — Complete public API surface and symbol map.
- [references/config.md](../references/config.md) — Typed configuration system reference.
- [concepts/model-architecture.md](../concepts/model-architecture.md) — Line-by-line model walkthrough.
- [concepts/gdn-and-mla.md](../concepts/gdn-and-mla.md) — Theoretical derivations of GDN, MLA, MoE, and MTP.
- [concepts/kernels.md](../concepts/kernels.md) — Triton GDN kernel execution model.
- [training.md](../training.md) — Training loop, optimizers, scheduler, and FSDP-2 integration.
