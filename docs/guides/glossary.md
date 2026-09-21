# HyMo — Glossary

> Terms and canonical values used across the HyMo docs. Values are from
> `configs/hymo_750m.yaml` at HEAD; the [`references/config.md`](../references/config.md)
> field tables are the authoritative per-field source.

## Notation

| Symbol | Value | Meaning |
|---|---|---|
| N_active / N_stored | 434 M / 1.13 B | active vs stored parameters (MoE sparsity: only 2 routed experts activate per token; stored = all expert weights). Production construct ≈ 4.5 GB fp32. |
| L | 32 | layers (`n_layers`) |
| d_model | 896 | model width (`dim`) |
| V | 64,256 | vocab: BPE-64k + 256-byte fallback (`vocab_size`) |
| T | 4096 | context window (`max_seq_len`) |
| Schedule | 3:1 | GDN:MLA ratio — MLA at layers 0, 4, 8, …, 28 (`src/hymo/core/config.py:ModelConfig.mla_positions`) |
| tokens/step | 524,288 | micro_batch 4 × accum 8 × seq 4096 (`TrainingConfig.per_step_tokens`) |
| total_steps | 57,220 | 30 B tokens / 524,288 (`SchedulerConfig.total_steps`) |
| WSD shape | 2% / 83% / 15% | warmup / stable / decay fractions, linear decay to `min_lr_ratio` 0.05 |

## Hybrid architecture

| Term | Definition |
|---|---|
| **GDN (Gated Delta Net)** | Linear-attention recurrent block (`src/hymo/models/gdn.py:GatedDeltaNetBlock`): gated delta-rule state update, dims `gdn_d_state` 32, `gdn_d_conv` 4, `gdn_d_inner` 1280. 24 of 32 layers. |
| **MLA** | Multi-Head Latent Attention (`src/hymo/models/mla.py:MultiHeadLatentAttention`): low-rank KV compression (`q_lora_rank` 224, `kv_lora_rank` 128), 16 heads in 4 KV groups, head_dim 128 (qk_nope 96 + qk_rope 32). 8 of 32 layers, carrying the MoE. |
| **NoPE-hybrid GDN** | Optional variant dropping RoPE on GDN layers adjacent to MLA (`nope_hybrid_gdn_enabled`); disabled in v1.0. |
| **Hybrid thesis** | GDN gives O(1)-state sequence mixing at linear cost; MLA layers every 4th position provide precise retrieval anchors. See [`concepts/gdn-and-mla.md`](../concepts/gdn-and-mla.md). |
| **MTP** | Multi-token prediction (`src/hymo/models/mtp.py:MultiTokenPrediction`): depth-2 auxiliary heads on shared trunk, loss weights [0.3, 0.1], `mtp_inter_dim` 2304. |
| **Logit softcap** | Tanh-bounded logits (`logit_softcap` 15.0) via `src/hymo/models/model.py:HyMo.softcap`. |

## Sparse feed-forward (MoE)

| Term | Definition |
|---|---|
| **DeepSeekMoE** | Fine-grained expert split (`src/hymo/models/moe.py:DeepSeekMoE`): 16 routed + 1 shared expert, top-2 activation, `moe_inter_dim` 2304. Lives only on the 8 MLA blocks → 128 routed expert instances total. |
| **SwiGLUExpert** | The expert unit (`src/hymo/models/moe.py:SwiGLUExpert`): gated `w1/w3` projections + `w2` down-projection. |
| **Capacity factor** | Token-to-expert dispatch headroom (`moe_capacity_factor` 1.5). |
| **EMA balancing** | Auxiliary-loss-free load balancing via exponential moving average of expert load (`moe_ema_alpha` 0.02). |

## Optimization

| Term | Definition |
|---|---|
| **NorMuon** | Muon-family optimizer for 2-D attention/GDN matrices (`src/hymo/training/optimizer.py:NorMuon`): Newton-Schulz orthogonalization (5 iterations) + momentum 0.95, lr 0.02, wd 0.1. |
| **CautiousAdamW** | AdamW variant masking decay by gradient alignment (`src/hymo/training/optimizer.py:CautiousAdamW`): lr 3e-4, betas (0.9, 0.95), wd 0, `cautious_wd` true. |
| **Parameter partition** | The rule assigning each tensor to one optimizer (`src/hymo/training/partition.py:partition_parameters` → `src/hymo/training/partition.py:ParameterPartition`): 2-D dense → NorMuon; embeddings/head/norm/gate/scalars + experts → AdamW. |
| **Optimizers** | The dual-optimizer holder (`src/hymo/training/optimizer.py:Optimizers`), built by `src/hymo/training/optimizer.py:build_optimizers`; FP32 master copies internally. |
| **WSD** | Warmup-Stable-Decay LR schedule (`src/hymo/training/scheduler.py:JointWSDScheduler`) — one schedule driving both optimizers. |
| **FSDP-2** | PyTorch's fully-sharded data parallel wrapping (`src/hymo/training/fsdp.py:wrap_model_with_fsdp`, `src/hymo/training/fsdp.py:fsdp_auto_wrap_policy`); bf16 mixed precision, world_size 4. |
| **DCP** | PyTorch Distributed Checkpointing — the format behind `src/hymo/training/checkpoint.py:save_checkpoint` / `src/hymo/training/checkpoint.py:load_checkpoint` (sharded, resumable). |

## Data & tokenizer

| Term | Definition |
|---|---|
| **BPE-64k + 256-byte** | Custom tokenizer layout: 64,000 learned BPE merges + 256 byte-fallback IDs = V 64,256. |
| **`ExtendedTokenizer`** | The encode/decode wrapper (`src/hymo/data/tokenizer.py:ExtendedTokenizer`) with byte fallback; trained by `src/hymo/data/tokenizer.py:train_bpe_tokenizer`. |
| **`val.bin`** | 450 M-token FineWeb-Edu held-out shard built by `src/hymo/data/prepare_validation.py:build_val_set` (CLI: `src/hymo/data/prepare_validation.py:main`). |
| **Shared pipeline** | Corpus preparation lives in the workspace `LLM/shared_data/`; HyMo consumes it via `data/prepare_data.py` + its own tokenizer stage. |

## Validation & numerics

| Term | Definition |
|---|---|
| **In-training validation** | `src/hymo/training/validation.py:compute_validation_loss` over `val.bin`, returning `src/hymo/training/validation.py:ValMetrics`, at `eval_interval` 2000 steps. |
| **loss_nan_skip** | Skip micro-steps whose loss is non-finite (`loss_nan_skip` true) — divergence containment. |
| **grad_clip** | Global-norm clip 1.0 across the dual-optimizer step. |

## Acronyms

| Acronym | Expansion |
|---|---|
| GDN | Gated Delta Net |
| MLA | Multi-head Latent Attention |
| MTP | Multi-Token Prediction |
| MoE | Mixture of Experts |
| WSD | Warmup-Stable-Decay (schedule) |
| FSDP | Fully Sharded Data Parallel |
| DCP | Distributed Checkpointing |
| BPE | Byte-Pair Encoding |
| MFU | Model FLOPs Utilization |
| EMA | Exponential Moving Average |
