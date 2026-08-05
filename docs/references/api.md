# HyMo — API Reference

> The comprehensive API reference for HyMo: top-level factories, model components, training managers, optimizers, schedulers, checkpointing utilities, validation functions, and tokenizer interfaces. All symbol anchors are machine-verified by `tests/test_doc_refs.py`.

---

## 1. Top-Level Entrypoints (`src/hymo/__init__.py`)

- `load_config(path: str | Path) -> HyMoConfig`
  - Loads a `HyMoConfig` from a YAML file (`src/hymo/core/config.py:load_config`).
- `load_config_from_dict(raw: dict[str, Any]) -> HyMoConfig`
  - Constructs a `HyMoConfig` from a plain Python dictionary (`src/hymo/core/config.py:load_config_from_dict`).
- `build_hymo(config: HyMoConfig) -> HyMo`
  - Instantiates a `HyMo` model from a `HyMoConfig` (`src/hymo/models/model.py:build_hymo`).
- Re-exported dataclasses: `HyMoConfig`, `ModelConfig`, `OptimizerConfig`, `SchedulerConfig`, `TrainingConfig`, `RunConfig`.

---

## 2. Model Architecture (`src/hymo/models/`)

| Class / Function | Anchor | Description |
|---|---|---|
| `HyMo` | `src/hymo/models/model.py:HyMo` | 32-layer 3:1 GDN:MLA hybrid language model |
| `build_hymo` | `src/hymo/models/model.py:build_hymo` | Factory function creating `HyMo` from `HyMoConfig` |
| `GatedDeltaNetBlock` | `src/hymo/models/gdn.py:GatedDeltaNetBlock` | Linear attention block implementing the Gated Delta Net recurrence |
| `MultiHeadLatentAttention` | `src/hymo/models/mla.py:MultiHeadLatentAttention` | Compressed Multi-Head Latent Attention with MQA-4 grouping & RoPE |
| `MLABlock` | `src/hymo/models/mla.py:MLABlock` | Block wrapping MLA attention, DeepSeekMoE, pre-norms, and residuals |
| `DeepSeekMoE` | `src/hymo/models/moe.py:DeepSeekMoE` | DeepSeek-style MoE with fine-grained routed experts & shared expert |
| `SwiGLUExpert` | `src/hymo/models/moe.py:SwiGLUExpert` | Single SwiGLU expert projection layer (`w1`, `w2`, `w3`) |
| `MultiTokenPrediction` | `src/hymo/models/mtp.py:MultiTokenPrediction` | Depth-$D$ speculative multi-token prediction heads |
| `RotaryEmbedding` | `src/hymo/models/rope.py:RotaryEmbedding` | Precomputed rotary position embedding tables (partial RoPE) |
| `TritonGDNFunction` | `src/hymo/models/gdn_triton.py:TritonGDNFunction` | PyTorch autograd `Function` wrapping the Triton GDN kernel |
| `triton_gated_delta_rule` | `src/hymo/models/gdn_triton.py:triton_gated_delta_rule` | Python host entrypoint for Triton-accelerated GDN forward+backward |

### 2.1 `HyMo` Class Details

- `__init__(config: ModelConfig) -> None`
  - Constructs embedding (`vocab_size` $\to$ `dim`), 32 hybrid layers (24 `GatedDeltaNetBlock` + 8 `MLABlock`), RMSNorm, logit softcapping scalar, tied/untied head, and optional `MultiTokenPrediction` module.
- `forward(tokens: torch.Tensor) -> torch.Tensor`
  - Input: `tokens` of shape `(B, T)` (integer token IDs).
  - Output: Next-token logits of shape `(B, T, vocab_size)` with logit softcapping applied.
- `forward_with_hidden(tokens: torch.Tensor, start_pos: int = 0) -> tuple[torch.Tensor, torch.Tensor]`
  - Returns `(softcapped_logits, pre_head_hidden_state)`. Used by `MultiTokenPrediction` and training pipeline.
- `softcap(logits: torch.Tensor) -> torch.Tensor`
  - Applies PaLM-style logit softcapping: `logit_softcap * tanh(logits / logit_softcap)`.
- `num_parameters(only_trainable: bool = False) -> int`
  - Returns total parameter count. Accurately accounts for weight tying (`head.weight` tied to `embed.weight`).
- `config` property: Returns the `ModelConfig`.

### 2.2 `GatedDeltaNetBlock` Class Details

- `__init__(config: ModelConfig, layer_idx: int, use_rope: bool = True) -> None`
  - Initializes projections for `v`, `b`, `c`, `g`, `A_log`, RMSNorm, and DenseFFN/SwiGLU feed-forward.
- `forward(x: torch.Tensor) -> torch.Tensor`
  - Maps `(B, T, dim)` to `(B, T, dim)`. Dispatches to `triton_gated_delta_rule` on CUDA when `use_triton=True`, falling back to `_gated_delta_rule` eager reference on CPU.
- `_gated_delta_rule(v, b, c, g, A_log) -> torch.Tensor`
  - Pure PyTorch eager reference recurrence: $h_t = \exp(g_t A) \odot h_{t-1} + b_t \otimes v_t, o_t = c_t \cdot h_t$.
- Properties: `n_heads`, `d_inner`, `d_state`, `headdim`.

### 2.3 `MultiHeadLatentAttention` Class Details

- `forward(x: torch.Tensor) -> torch.Tensor`
  - Projects `x` into latent query space ($d_c = 224$), latent KV space ($d_c = 128$), applies decoupled RoPE ($d_R = 32$), and executes `F.scaled_dot_product_attention(..., is_causal=True)` with MQA-4 grouping.

### 2.4 `DeepSeekMoE` Class Details

- `gate_forward(x: torch.Tensor) -> torch.Tensor`
  - Computes expert gate logits in FP32 to prevent softmax underflow/overflow under BF16.
- `update_gate_bias(speed: float = 0.001) -> None`
  - Executes EMA load-balancing bias adjustment across 16 routed experts without auxiliary loss terms.
- `forward(x: torch.Tensor) -> torch.Tensor`
  - Routes tokens to top-$k$ experts with capacity capping (`capacity_factor = 1.5`) and adds the shared expert output.

### 2.5 `MultiTokenPrediction` Class Details

- `forward(tokens: torch.Tensor, start_pos: int = 0) -> tuple[torch.Tensor, list[MTPOutput]]`
  - Executes main model forward pass and chains $D$ depth prediction heads over previous hidden states.

---

## 3. Training Loop & Distributed Infra (`src/hymo/training/`)

| Symbol | Anchor | Description |
|---|---|---|
| `Trainer` | `src/hymo/training/trainer.py:Trainer` | Main training loop manager handling forward/backward, optimization, validation, and DCP |
| `train_step_result` | `src/hymo/training/trainer.py:train_step_result` | Metrics dataclass returned by `Trainer.train_step` |
| `NorMuon` | `src/hymo/training/optimizer.py:NorMuon` | Muon optimizer with Newton-Schulz orthogonalization for 2D matrix weights |
| `CautiousAdamW` | `src/hymo/training/optimizer.py:CautiousAdamW` | AdamW with cautious weight decay for 1D vectors, biases, and expert weights |
| `Optimizers` | `src/hymo/training/optimizer.py:Optimizers` | Container holding `(nor_muon, adamw)` dual optimizer pair |
| `build_optimizers` | `src/hymo/training/optimizer.py:build_optimizers` | Builds dual optimizers from a partitioned model |
| `JointWSDScheduler` | `src/hymo/training/scheduler.py:JointWSDScheduler` | Warmup-Stable-Decay joint learning rate scheduler |
| `partition_parameters` | `src/hymo/training/partition.py:partition_parameters` | Partitions parameters into NorMuon (2D matrix) vs AdamW sets |
| `goes_to_adamw` | `src/hymo/training/partition.py:goes_to_adamw` | Predicate determining whether a parameter routes to AdamW |
| `wrap_model_with_fsdp` | `src/hymo/training/fsdp.py:wrap_model_with_fsdp` | FSDP-2 distributed wrapper with per-block unit wrapping |
| `fsdp_auto_wrap_policy` | `src/hymo/training/fsdp.py:fsdp_auto_wrap_policy` | Custom auto-wrap policy targeting `GatedDeltaNetBlock` and `MLABlock` |

### 3.1 `Trainer` Methods

- `__init__(config: HyMoConfig, model: HyMo) -> None`
  - Threads optimization flags (`fused_gdn`, `moe_mixed_precision`, `torch_compile_gdn`), partitions parameters, constructs `Optimizers` and `JointWSDScheduler`.
- `train_step(tokens: torch.Tensor, targets: torch.Tensor) -> train_step_result`
  - Runs forward pass (main model + MTP auxiliary heads), cross-entropy loss computation, scaled backward pass, gradient clipping, dual optimizer step (at `gradient_accumulation_steps` boundaries), MoE gate bias EMA update, and LR schedule step.
- `train(data_iter: Iterable[tuple[torch.Tensor, torch.Tensor]], max_steps: int | None = None) -> None`
  - Primary driver loop executing training steps, logging metrics to W&B, evaluating validation loss, and saving DCP checkpoints.
- `evaluate(val_bin_path: str | Path | None = None) -> dict[str, float]`
  - Computes validation loss and perplexity over held-out validation data.
- `save(tag: str | None = None) -> Path`
  - Writes Distributed Checkpoint (DCP) directory with tensor state and JSON metadata sidecar.
- `load(path: str | Path) -> int`
  - Restores model weights, optimizer states, scheduler state, and RNG states from a DCP checkpoint directory.

---

## 4. Checkpoint & Validation (`src/hymo/training/`)

- `save_checkpoint(path: str | Path, model: nn.Module, optimizers: Optimizers, scheduler: JointWSDScheduler, state: CheckpointState) -> None`
  - Writes atomic checkpoint directory (`src/hymo/training/checkpoint.py:save_checkpoint`).
- `load_checkpoint(path: str | Path, model: nn.Module, optimizers: Optimizers, scheduler: JointWSDScheduler) -> CheckpointState`
  - Restores state from atomic checkpoint directory (`src/hymo/training/checkpoint.py:load_checkpoint`).
- `compute_validation_loss(model: nn.Module, *, batch_size: int, seq_len: int, vocab_size: int, num_batches: int = 32, device: torch.device | str = "cpu", seed: int = 42, val_bin_path: Path = DEFAULT_VAL_BIN) -> ValMetrics`
  - Computes average cross-entropy loss and perplexity (`src/hymo/training/validation.py:compute_validation_loss`).
- `get_val_batch(batch_size: int, seq_len: int, device: torch.device | str = "cpu", seed: int = 42, path: Path = DEFAULT_VAL_BIN) -> tuple[torch.Tensor, torch.Tensor]`
  - Deterministically slices `(x, y)` validation batches (`src/hymo/training/validation.py:get_val_batch`).

---

## 5. Tokenizer Interface (`src/hymo/data/`)

- `ExtendedTokenizer` (`src/hymo/data/tokenizer.py:ExtendedTokenizer`)
  - `encode(text: str) -> list[int]`: Converts text string to token ID list with byte fallback.
  - `decode(ids: list[int]) -> str`: Decodes token ID list back to text string.
  - Properties: `vocab_size` (64,256), `eos_token_id`, `pad_token_id`.
- `train_bpe_tokenizer(texts: Iterable[str], *, vocab_size: int = 64_000, output_path: str | Path) -> ExtendedTokenizer`
  - Trains a byte-pair encoding (BPE) tokenizer over text corpora (`src/hymo/data/tokenizer.py:train_bpe_tokenizer`).
- `build_val_set(target_tokens: int = 450_000_000, output_path: str | Path = "data/tokens/val.bin") -> None`
  - Prepares held-out validation token binary file (`src/hymo/data/prepare_validation.py:build_val_set`).

---

## 6. References & Cross-Links

- [config.md](config.md) — Complete hyperparameter table and validation checks.
- [../concepts/model-architecture.md](../concepts/model-architecture.md) — Comprehensive structural model walkthrough.
- [../concepts/gdn-and-mla.md](../concepts/gdn-and-mla.md) — Architectural mechanism deep-dives.
- [../concepts/optimization.md](../concepts/optimization.md) — NorMuon, AdamW, WSD scheduler, and FSDP-2 mechanics.
- [../concepts/kernels.md](../concepts/kernels.md) — Triton GDN kernel implementation and autograd integration.
