# HyMo — API Reference

> The public model and trainer API surface: the top-level factory, the model
> classes, the trainer, and the optimization/checkpoint/validation entry
> points. Full config-field tables live in [config.md](config.md); the
> line-by-line walkthrough is [concepts/model-architecture.md](../concepts/model-architecture.md).

## Top-level package (`src/hymo/__init__.py`)

- `load_config(path)` — load a `HyMoConfig` from YAML
  (`src/hymo/core/config.py:load_config`).
- `build_hymo(config)` — construct a `HyMo` from a `HyMoConfig`
  (`src/hymo/models/model.py:build_hymo`).
- Config classes re-exported: `HyMoConfig`, `ModelConfig`, `OptimizerConfig`,
  `SchedulerConfig`, `TrainingConfig`, `RunConfig`.

## Model classes (`src/hymo/models/`)

| Class | File | Role |
|---|---|---|
| `HyMo` | `src/hymo/models/model.py:HyMo` | The 32-layer 3:1 GDN:MLA hybrid stack |
| `GatedDeltaNetBlock` | `src/hymo/models/gdn.py:GatedDeltaNetBlock` | Linear-attention block with the GDN recurrence |
| `MultiHeadLatentAttention` | `src/hymo/models/mla.py:MultiHeadLatentAttention` | MLA with MQA-4 grouping + low-rank KV |
| `MLABlock` | `src/hymo/models/mla.py:MLABlock` | MLA attention + DeepSeekMoE + pre-norms |
| `DeepSeekMoE` | `src/hymo/models/moe.py:DeepSeekMoE` | 16 routed + 1 shared expert, aux-loss-free routing |
| `SwiGLUExpert` | `src/hymo/models/moe.py:SwiGLUExpert` | Single expert SwiGLU (`w1`/`w2`/`w3`) |
| `MultiTokenPrediction` | `src/hymo/models/mtp.py:MultiTokenPrediction` | Depth-2 MTP heads on the main hidden state |
| `RotaryEmbedding` | `src/hymo/models/rope.py:RotaryEmbedding` | Precomputed cos/sin RoPE tables |
| `TritonGDNFunction` | `src/hymo/models/gdn_triton.py:TritonGDNFunction` | Autograd `Function` wrapping the Triton kernels |

### `HyMo` methods

- `forward(tokens)` — next-token logits only: `(B, T, vocab_size)`.
- `forward_with_hidden(tokens, start_pos=0)` — returns
  `(softcap(logits), hidden)`; the hidden state feeds the MTP heads.
- `softcap(logits)` — `logit_softcap * tanh(logits / logit_softcap)`.
- `num_parameters(only_trainable=False)` — parameter count.
- `config` property — the `ModelConfig`.

### `GatedDeltaNetBlock` methods

- `forward(x)` — full block forward; dispatches to the Triton kernel
  (`triton_gated_delta_rule`) when `use_triton` and CUDA, else the eager
  `_gated_delta_rule` recurrence. `torch.compile`-wrapped when
  `use_compile` and CUDA.
- `_gated_delta_rule(v, b, c, g, A_log)` — eager PyTorch reference.
- `_kernel_out(...)` — the wrapper into the Triton kernel.
- Properties: `n_heads`, `d_inner`, `d_state`, `headdim`.

### `DeepSeekMoE` methods

- `gate_forward(x)` — FP32 gate logits (avoids BF16 softmax underflow).
- `update_gate_bias(speed=0.001)` — the EMA load-balancing bias update
  (aux-loss-free).
- `forward(x)` — top-k dispatch with capacity capping + shared expert.

### `MultiTokenPrediction` methods

- `forward(tokens, start_pos=0)` — returns `(main_logits, [MTPOutput, ...])`
  with chained heads and per-depth `loss_weight`.

## Trainer (`src/hymo/training/trainer.py:Trainer`)

| Method | Role |
|---|---|
| `__init__(config, model)` | Threads optimization flags, builds optimizers + scheduler |
| `train_step(tokens, targets)` | One forward/backward + (every `grad_accum`) optimizer step; returns `train_step_result` |
| `train(data_iter, max_steps=None)` | Driver loop with W&B logging, eval, and save cadence |
| `evaluate(val_bin_path=None)` | In-training validation loss/PPL |
| `save(tag=None)` / `load(path)` | DCP checkpoint write/read |
| `_update_moe_gate_biases()` | Fires the MoE EMA update after each optimizer step |

`train_step_result` fields: `loss`, `grad_norm`, `lr_muon`, `lr_adamw`,
`is_update`, `skipped`, `metrics`.

## Optimizer / scheduler / FSDP / partition

- `build_optimizers(model, config)` — returns `Optimizers(nor_muon, adamw)`
  (`src/hymo/training/optimizer.py:build_optimizers`).
- `NorMuon` — Muon + Newton–Schulz + cautious WD
  (`src/hymo/training/optimizer.py:NorMuon`).
- `CautiousAdamW` — AdamW with the cautious WD mask
  (`src/hymo/training/optimizer.py:CautiousAdamW`).
- `JointWSDScheduler` — warmup/stable/decay LR factor
  (`src/hymo/training/scheduler.py:JointWSDScheduler`).
- `goes_to_adamw(name, param)` / `partition_parameters(model)` — the
  NorMuon/AdamW partition (`src/hymo/training/partition.py`).
- `wrap_model_with_fsdp(model, config, ...)` / `fsdp_auto_wrap_policy`
  (`src/hymo/training/fsdp.py`).

## Checkpoint and validation

- `save_checkpoint(path, model, optimizers, scheduler, state)` /
  `load_checkpoint(...)` — DCP-based with RNG capture
  (`src/hymo/training/checkpoint.py`).
- `compute_validation_loss(...)` / `get_val_batch(...)` / `ValMetrics`
  (`src/hymo/training/validation.py`).
- `build_val_set(target_tokens=450_000_000, ...)` — builds the held-out
  FineWeb-Edu validation binary (`src/hymo/data/prepare_validation.py`).

## Tokenizer

- `ExtendedTokenizer` — BPE-64k + 256-byte fallback wrapper with
  `encode`/`decode`/`vocab_size`/`eos_token_id`/`pad_token_id`
  (`src/hymo/data/tokenizer.py:ExtendedTokenizer`).
- `train_bpe_tokenizer(texts, *, vocab_size=64_000, output_path=...)` —
  BPE training entry point.

## References

- [config.md](config.md) — the typed-config system and every field table.
- [../concepts/model-architecture.md](../concepts/model-architecture.md) — the walkthrough this reference summarizes.
- [../concepts/optimization.md](../concepts/optimization.md) — optimizer/scheduler/FSDP mechanics.
- [../training.md](../training.md) — the trainer loop in detail.
- Source: the files listed in the tables above.
