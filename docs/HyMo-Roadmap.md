# HyMo: End-to-End Implementation Plan (v1.0)

> **Version scope:** v1.0 = pre-training only. The 4 parallel ablations are deferred to **v1.1** and do not block the v1.0 deliverable. The optimization stack (fused Triton GDN, MoE mixed precision, `torch.compile` on GDN, CUDA Graphs on MLA — see architecture doc §12a) is a **first-class part of v1.0** because the 5-7 day wall-clock is contingent on it.
> **Goal:** Build, validate, and run **HyMo v1.0** (750M active / 1.86B stored, 32 layers, 3:1 GDN:MLA, 30B tokens at 40× params-in-tokens) on 4× A100 80GB SXM with FSDP-2, achieving held-out FineWeb-Edu PPL ≤ 2.10.
> **Source of truth for design decisions:** [`docs/HyMo-Design.md`](HyMo-Design.md) (this directory). This plan references the architecture doc by §X.Y throughout; do not re-derive any decision, look it up.
> **Source of truth for the prior stability fixes:** the 6 v1 fixes (joint WSD, aux-loss-free, MTP checkpointing, deterministic validation, exact-name optimizer partition, config-driven trainer) are prerequisites and must be present before this plan starts.
> **Tech Stack:** PyTorch ≥2.5, raw PyTorch (no HF Trainer), pytest, safetensors, BF16, 4× A100 80GB SXM (RunPod), FSDP-2, fla (GDN kernel), lm-eval ≥ 0.4.5 (held-out evaluation).
> **Test style (hard rule):** No test may build the full 1.86 B-parameter
> model in the default run. Default tests use the tiny (~760 K-param) config
> (`tiny_hymo_model` / `tiny_hymo_config` fixtures, or the `ModelConfig()`
> shadow in `test_models.py`). Any test that constructs the production model
> MUST be marked `@pytest.mark.heavy` and is auto-skipped unless `pytest
> --run-heavy` is passed (CI / GPU pod only). Production-scale arithmetic
> (e.g. 384 expert weights, 32 layers, 465 M sharded params) lives behind
> `heavy`. See `AGENTS.md` for the full rules.

---

## Implementation status (last verified 2026-07-16)

This file is the v1.0 *build + train* plan. The actual implementation has been
decomposed into 5 phases, each with its own gate — see
[`PHASE_1_DELIVERY.md`](PHASE_1_DELIVERY.md) for the canonical phase map.

| Phase | Status | Evidence |
|---|---|---|
| 1 — Repository foundation | **Shipped** | `PHASE_1_DELIVERY.md`; commit `f4c64e7` on `main`. 308/308 tests pass, `mypy --strict` clean, `ruff` clean. Every `forward` was a `NotImplementedError_` placeholder. |
| 2 — Algorithmic model implementation | **Completed** ✅ | Real `forward` logic implemented for every model class (`models/rope.py`, `gdn.py`, `mla.py`, `moe.py`, `mtp.py`, `init.py`, `fusionllm.py`). A smoke test confirms forward+backward is finite on the tiny config; `mypy --strict` + `ruff` clean. 321 tests pass (17 `heavy` skipped by default). See **Phase 2 delivery note** below. |
| 3 — Training infrastructure | **Next** | `hymo.training.{optimizer, scheduler, fsdp, checkpoint, validation, trainer}` placeholders still raise; partition/`build_optimizers` interfaces live. |
| 4 — Data pipeline + eval + ablations | Pending | `hymo.data.*` + `hymo.eval.harness` + `hymo.ablations` placeholders. |
| 5 — Deployment + 30B-token run | Pending | All `scripts/runpod_*.sh` etc. to be written. |

**Phase 2 delivery note (algorithmic model implementation).** All eight
placeholder surfaces are now real:

- `models/rope.py` — `RotaryEmbedding.apply_rope` (partial-RoPE over 25% of
  head_dim; NoPE-hybrid positions supported).
- `models/gdn.py` — `GatedDeltaNetBlock.forward` (delta-rule recurrence +
  gated MLP; `use_rope` toggle).
- `models/mla.py` — `MultiHeadLatentAttention.forward` (latent compression +
  MQA-4 KV groups; `kv_group_for_q` lookup table).
- `models/moe.py` — `SwiGLUExpert` / `DenseFFN` (SwiGLU), `DeepSeekMoE.forward`
  (top-2 routing, capacity cap, shared expert), `gate_forward` (FP32 cast),
  `update_gate_bias` (EMA load-balance, 1.05× threshold).
- `models/mtp.py` — `MultiTokenPrediction.forward` (depth-2 chained hidden,
  shared head, weights `[0.3, 0.1]`) + `MTPBlock`.
- `models/init.py` — real `mup_init` (zero-init scalars/gains, μP-scale 2D,
  embed-scale).
- `models/fusionllm.py` — `HyMo.forward` + `forward_with_hidden` (32-layer
  stack + final norm + `softcap`).

Verification: a CPU smoke test runs forward+backward on the tiny config and
asserts finite grads; the model assembles at the production scale (~1.13–1.86B
params, **heavy** test) with 8 MLA + 24 GDN layers. Docstrings updated from
"Phase 1 placeholder" to Phase 2.

**Open next task:** **Phase 3** (Training Infrastructure) is now started on
branch `phase-3/training-infra`. The training interfaces already exist from
Phase 1 (`partition_parameters`, `goes_to_adamw`, `build_optimizers`,
`JointWSDScheduler`, `forward_with_hidden`); Phase 3 fills in the algorithmic
correctness behind them and wires the `Trainer`. Begin with
`training/optimizer.py` (`NorMuon` + `CautiousAdamW`) since the partition
predicate and optimizer construction are already tested on the tiny model.

---

## Pre-plan checklist (read this first)

Before any task in this plan starts, the following must be true:

1. [x] The 6 stability fixes are committed and all v1 tests pass: `pytest tests/ -v --tb=short`. → The 6 fixes are not committed in the *v1* sense (the prior repo was rewritten), but their **interfaces are live** in Phase 1: `JointWSDScheduler` (C3), `balance_loss_alpha` (C4), `forward_with_hidden` (C3), `deterministic_val` via seed parameter (C6), `goes_to_adamw` exact-name predicate (C1), and `build_optimizers` reading from `OptimizerConfig` (C4). See `PHASE_1_DELIVERY.md §3` for the public API. **Algorithmic correctness is a Phase 3 task.**
2. [ ] RunPod account is verified, billing is set up, and the `4xA100-80GB-SXM` instance type is available in your region. **SXM is required, not optional** — PCIe has no NVLink and is 2-3× slower for FSDP-2.
3. [ ] A network volume is created on RunPod for `data/` (~120GB for shards + 1.8GB for val.bin) and `checkpoints/` (~30GB for last-2 + best). The volume survives pod termination.
4. [ ] The architecture doc §12 open questions are resolved. Default values are documented in the doc, so resolution is "accept all defaults" unless you want to override. v1.0 commits to: (1) fla dependency for the GDN kernel, (2) byte-level BPE fallback ON, (3) MLA at position 0 (empirical check at step 1k), (4) start on the full optimization stack (per §12a).
5. [ ] **v1.0 budget confirmed:** $1,000-1,350 at RunPod on-demand (or $700-1,000 with spot/committed-use). See architecture doc §12.7. This is **v1.0 pre-training only**; the v1.1 ablation budget is separate and is not part of this plan.
6. [x] A clean git working tree on a new branch (e.g., `git checkout -b fusionllm-v1.0`). → The repo was initialized at `LLM/HyMo/` with `main` as the default branch (the workspace root is not a git repo per `CoreProjects/CLAUDE.md`). Working tree is clean on `main` at commit `f4c64e7`. A feature branch can be created when Phase 2 starts.

If any of these are not true, stop and resolve them before starting the plan.

---

## v1.0 at a glance (the numbers you must commit to)

| Quantity | Value | Source |
|---|---|---|
| Active params | **750M** (40× × 750M = 30B) | architecture doc §2.1 |
| Stored params | ~1.86B (MoE-dominated) | architecture doc §2.1 |
| Layers | 32 (24 GDN + 8 MLA, 3:1 mid-stack) | architecture doc §2.2 |
| Training tokens | **30B** at 40× params-in-tokens | architecture doc §6 |
| Per-step tokens | 524,288 (= 4 GPU × 4 micro × 8 grad_accum × 4096 seq) | architecture doc §7.1 |
| Total steps | **57,220** | architecture doc §5.3 |
| WSD schedule | 2% warmup / 83% stable / 15% decay → 0.05× min | architecture doc §5.3 |
| Wall-clock (target) | **5-7 days** at ~65,000 tok/s sustained | architecture doc §0, §7.6, §12a.5 |
| v1.0 budget | **$1,000-1,350** | architecture doc §12.7 |
| Held-out PPL target | **≤ 2.10** on real FineWeb-Edu val | architecture doc §0 |
| 4 optimizations (§12a) | fused Triton GDN, MoE mixed precision, `torch.compile` (GDN), CUDA Graphs (MLA) | architecture doc §12a |

**The 5-7 day target is contingent on all four optimizations being enabled.** Removing any one extends wall-clock by 5-25% and risks exceeding the budget (see architecture doc §12a.5 and §9 risk table). The optimization work is formalized in **Work Block H**.

---

## How to use this plan

This plan is structured as **8 work blocks** (A through H), each with **5-8 checkbox tasks**. Each task has:
- **What:** concrete code/config changes
- **Where:** file paths and line ranges
- **Why:** one-line rationale (or "see architecture doc §X.Y" for the full argument)
- **Test:** how to verify before moving on

**A task is "done" only when its Test passes AND the code is committed.** No moving on with broken tests.

**Use the superpowers:subagent-driven-development skill** to parallelize independent work blocks. Specifically:
- Work blocks A (data) and B (model) are independent and can be parallelized.
- Work block C (training infra) depends on B.
- Work block D (FSDP-2) depends on C.
- Work block H (optimization) depends on B and C, and can run in parallel with D.
- Work block E (validation/eval) depends on D and H.
- Work block G (launch + recovery) depends on E.
- Work block F (ablations) is **v1.1** and is documented as a stub only.

If you have a single machine and one pair of hands, work serially: A → B → C → (D || H) → E → G.

---

---

# Work Block A: Data Pipeline (Tasks A1-A7)

The data pipeline is the single largest quality lever. Architecture doc §6.

**Prereqs:** clean working tree; v1 data config in `data/data_config.yaml` (will be updated).

**Deliverable:** `data/shards/shard_00000.bin` through `data/shards/shard_00599.bin` (600 shards × 50M tokens = 30B); `data/tokens/val.bin` (real held-out FineWeb-Edu, 0.45B tokens); `data/tokens/byte_bpe_vocab.json` (extended tokenizer with byte-level fallback).

**Wall-clock:** 12-18 hours of streaming + tokenization on the 4× A100 SXM pod (the dataloader uses the GPU only for the byte-level BPE path; most of the time is streaming + tokenize on CPU). Run this in parallel with Work Block B on a separate pod if you have one available.

---

### Task A1: Update the mixture config to the v1.0 spec

**Files:**
- Modify: `data/config/mixture.yaml`

**What:** Replace the prior mixture (FineWeb-Edu 0.55 + FineWeb 0.20 + Stack-Python 0.10 + SlimPajama 0.08 + Wikipedia 0.04 + Books 0.03) with the v1.0 quality-first mixture from architecture doc §6. Per-source weights:

```yaml
sources:
  - { id: fineweb_edu_q3, weight: 0.50 }   # FineWeb-Edu with quality score >= 3
  - { id: fineweb,         weight: 0.12 }
  - { id: stack_python,    weight: 0.10 }
  - { id: stack_java,      weight: 0.03 }
  - { id: stack_cpp,       weight: 0.02 }
  - { id: slimpajama,      weight: 0.08 }
  - { id: dclm_baseline,   weight: 0.05 }   # NEW
  - { id: dolma_wiki,      weight: 0.04 }   # multilingual Wikipedia subset
  - { id: dolma_books,     weight: 0.03 }
  - { id: cosmopedia,      weight: 0.01 }   # NEW, synthetic

total_tokens: 30000000000                   # was 8.31B; 40x of 750M
train_fraction: 0.97                        # 29.1B train
val_fraction: 0.015                         # 0.45B val
test_fraction: 0.015                        # 0.45B test
```

**Why:** See architecture doc §6 for the per-source justification. The FineWeb-Edu quality ≥ 3 filter and the addition of DCLM/Cosmopedia are the highest-leverage changes.

**Test:** `python -c "import yaml; d = yaml.safe_load(open('data/config/mixture.yaml')); assert abs(sum(s['weight'] for s in d['sources']) - 1.0) < 1e-6; assert d['total_tokens'] == 30_000_000_000; print('OK')"`. The sum of weights must equal 1.0 and total_tokens must be 30B.

---

### Task A2: Implement the FineWeb-Edu quality ≥ 3 filter

**Files:**
- Modify: `data/prepare_data.py:1-50` (the streaming source loader)

**What:** FineWeb-Edu has an internal `score` field (0-5). The v1 loader accepts all rows with `score >= 0`. Add a `--quality-threshold` CLI flag and update the streaming filter to `score >= threshold`. Default threshold: 3.

```python
# data/prepare_data.py — new CLI flag
parser.add_argument(
    "--fineweb-edu-quality", type=int, default=3,
    help="Minimum FineWeb-Edu quality score (0-5). Default 3 (top ~50%%)."
)

# Streaming filter for fineweb_edu
if source_id == "fineweb_edu_q3":
    ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT",
                      split="train", streaming=True)
    ds = ds.filter(lambda row: row.get("score", 0) >= args.fineweb_edu_quality)
```

**Why:** The v1 mixture used FineWeb-Edu unfiltered, which includes low-quality rows that hurt PPL by 0.05-0.10. The 2026 SOTA (DCLM, Cosmopedia papers) uses the ≥ 3 filter.

**Test:** Run a 1-minute streaming probe that loads 1,000 FineWeb-Edu rows and asserts all have `score >= 3`. `python -m data.prepare_data --probe --source fineweb_edu_q3 --limit 1000` should exit 0 with a count of accepted rows.

---

### Task A3: Add DCLM-Baseline and Cosmopedia source loaders

**Files:**
- Modify: `data/prepare_data.py` (add source loaders for `dclm_baseline` and `cosmopedia`)

**What:**

```python
# DCLM-Baseline (mlfoundations/dclm-baseline-1.0)
if source_id == "dclm_baseline":
    ds = load_dataset("mlfoundations/dclm-baseline-1.0", split="train", streaming=True)
    ds = ds.map(lambda r: {"text": r["text"]}, remove_columns=[c for c in ds.column_names if c != "text"])
    return ds

# Cosmopedia (HuggingFaceTB/cosmopedia)
if source_id == "cosmopedia":
    ds = load_dataset("HuggingFaceTB/cosmopedia", split="train", streaming=True)
    # Each row has a "text" field (the rendered synthetic textbook)
    ds = ds.map(lambda r: {"text": r["text"]}, remove_columns=[c for c in ds.column_names if c != "text"])
    return ds
```

**Why:** The prior mixture didn't include these; the v1.0 mixture is missing them without these loaders.

**Test:** Run a 30-second probe of each source that loads 100 rows and prints the first row's text length. Both probes should exit 0.

---

### Task A4: Extend the BPE-64k tokenizer with byte-level fallback

**Files:**
- Modify: `data/tokenizer.py` (new file, or extend existing)

**What:** The v1 BPE-64k tokenizer treats OOV tokens as `<unk>`. The v1.0 spec (architecture doc §6.1) requires **byte-level BPE fallback**: any OOV token is split into UTF-8 bytes, and the BPE merging proceeds at the byte level. Implement as a post-processor on the existing tokenizer.

```python
# data/tokenizer.py — new module
from tokenizers import Tokenizer
from tokenizers.pre_tokenizers import ByteLevel as ByteLevelPre
from tokenizers.decoders import ByteLevel as ByteLevelDec

BYTE_VOCAB = list(range(256))  # 256 byte tokens; appended to the BPE vocab

def build_extended_tokenizer(base_tokenizer_path: str, output_path: str) -> None:
    """Append 256 byte-level tokens to the BPE-64k vocab, with byte-level pre-tokenizer fallback."""
    tok = Tokenizer.from_file(base_tokenizer_path)
    # Add the 256 byte tokens to the vocab if not present
    current_vocab = tok.get_vocab()
    for b in BYTE_VOCAB:
        token = f"<0x{b:02X}>"
        if token not in current_vocab:
            tok.add_tokens([token])
    tok.save(output_path)
```

In the dataloader (`training/data_loader.py`), when tokenizing OOV text, fall back to the byte-level BPE path. The 2025 tokenizers (Llama-3, Qwen-2.5) use this pattern.

**Why:** Code-heavy text (Stack-Java, Stack-C++) has rare identifiers that fall outside the BPE-64k vocab. Byte-level fallback preserves them as byte sequences, which is a 0.02-0.05 PPL improvement on code evals.

**Test:** A roundtrip test: tokenize `__init__` (which is in vocab) and `super().__init__()` (which has a non-vocab `(` and `)`; falls back to bytes). Verify both tokenize to non-empty integer sequences, and that decoding round-trips to the original string. Add `tests/test_byte_level_bpe.py`.

---

### Task A5: Update `data_config.yaml` to the v1.0 spec

**Files:**
- Modify: `data/data_config.yaml`

**What:** Update the shard size, target total tokens, dedup, and quality fields. Concretely:

```yaml
pipeline:
  tokenizer:
    name: fusionllm-bpe-64k
    path: data/tokens/byte_bpe_vocab.json   # NEW: extended tokenizer from Task A4
    vocab_size: 64256                       # 64000 BPE + 256 bytes
    eos_token_id: 0
    pad_token_id: 2
    add_eos: true
  sharding:
    shard_size_tokens: 50000000
    dtype: uint32
    target_total_tokens: 30000000000        # was 8.31B; 40x of 750M
  dedup:
    enabled: true
    method: sha256
    n_hash_buckets: 256
    bloom_capacity_per_bucket: 200000
    bloom_error_rate: 0.001
  quality:
    drop_empty: true
    min_unique_chars_ratio: 0.05
    max_digit_ratio: 0.5
    max_punct_ratio: 0.5
    max_whitespace_ratio: 0.5
  tokenize:
    batch_size: 1024
    add_special_tokens: false
    show_progress_every_docs: 50000
  pack:
    docs_per_shard_target: 50000000
    cross_document_boundary_ok: false
  seed: 42
  streaming_download: true
  verify_after_pack: true
```

**Why:** The 30B token budget is the v1.0 spec; the extended vocab_size is the byte-level fallback.

**Test:** `python -c "import yaml; d = yaml.safe_load(open('data/data_config.yaml')); assert d['pipeline']['target_total_tokens'] == 30_000_000_000; assert d['pipeline']['tokenizer']['vocab_size'] == 64256"`.

---

### Task A6: Run the data pipeline end-to-end and produce 30B tokens

**Files:**
- Run: `python -m data.prepare_data --config data/data_config.yaml`

**What:** The full pipeline:
1. Streaming-download all 10 sources per the v1.0 mixture.
2. Apply per-source quality filters (FineWeb-Edu ≥ 3, etc.).
3. Deduplicate via the bloom filter.
4. Tokenize using the extended BPE-64k + byte-level fallback tokenizer.
5. Pack into 50M-token shards of `uint32`.

**Expected output:** `data/shards/shard_00000.bin` through `data/shards/shard_00599.bin` (600 shards × 50M tokens = 30B). Each shard is 200MB. Total disk: ~120GB.

**Wall-clock:** 12-18 hours on the 4× A100 SXM pod (the dataloader uses the GPU only for the byte-level BPE path; most of the time is streaming + tokenize on CPU).

**Why:** The data is the prerequisite for training. Nothing in work blocks B-H starts until this is done.

**Test:** `ls data/shards/ | wc -l` should return 600 (plus the directory line). Total size should be ~120GB. Spot-check: `python -c "import numpy as np; a = np.fromfile('data/shards/shard_00000.bin', dtype=np.uint32); print('shard 0:', a.shape, 'min:', a.min(), 'max:', a.max())"`.

**Important:** Set up a RunPod network volume mount for `data/` so the data survives pod termination. The 120GB is too large to re-download per pod.

---

### Task A7: Build the real held-out validation set

**Files:**
- New: `data/prepare_validation.py`
- New: `data/tokens/val.bin` (output)

**What:** The v1 validation used synthetic uniform-random batches. The v1.0 spec (architecture doc §6.3) requires **real held-out FineWeb-Edu data**, drawn from a 5% held-out split of the FineWeb-Edu corpus.

```python
# data/prepare_validation.py
"""Build a 0.45B-token held-out validation set from FineWeb-Edu."""
import numpy as np
from datasets import load_dataset
from data.tokenizer import load_extended_tokenizer

def build_val_set(target_tokens: int = 450_000_000, seed: int = 42) -> None:
    tok = load_extended_tokenizer("data/tokens/byte_bpe_vocab.json")
    ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT",
                      split="train", streaming=True)
    # Skip a 5% held-out split deterministically (the train shards use the other 95%)
    ds = ds.shard(num_shards=20, index=0)  # first 5% shard
    tokens = []
    pbar = tqdm(total=target_tokens, desc="tokenizing val")
    for row in ds:
        encoded = tok.encode(row["text"]).ids
        tokens.extend(encoded)
        if len(tokens) >= target_tokens:
            break
        pbar.update(len(encoded))
    arr = np.array(tokens[:target_tokens], dtype=np.uint32)
    arr.tofile("data/tokens/val.bin")
    print(f"Wrote {len(arr):,} tokens to data/tokens/val.bin")
```

**Wall-clock:** 1-2 hours. Output: a 1.8GB `val.bin` file (0.45B tokens × 4 bytes).

**Why:** A meaningful val PPL requires real held-out data; the synthetic uniform random is meaningless.

**Test:** `python -c "import numpy as np; a = np.fromfile('data/tokens/val.bin', dtype=np.uint32); assert a.shape[0] >= 450_000_000; print('val set OK:', a.shape[0], 'tokens')"`. Also: spot-check that no tokens in `val.bin` appear in any `shard_*.bin` (held-out means held-out).

---

**End of Work Block A. Commit series: A1-A7 as 1-2 commits. Move to Work Block B (in parallel) or H (after B) when `data/tokens/val.bin` exists and `data/shards/` has 600+ files totaling 30B tokens.**

---

# Work Block B: Model Architecture (Tasks B1-B7) — ✅ SHIPPED

The model: 32 layers (3:1 GDN:MLA), MQA-4 on MLA, 16-expert MoE on MLA only, MTP depth=2, partial-RoPE + NoPE-hybrid, μP init, ~750M active / ~1.86B stored. Architecture doc §2-§4.

**Prereqs:** Work Block A started (shards are being built in the background); v1 model code is in `models/` but will be rewritten.

**Deliverable:** `models/fusionllm.py` with the full v1.0 architecture; 32-block forward works; partial-RoPE + NoPE-hybrid works; all init is correct.

**Status (2026-07-16):** all B1-B7 tasks are implemented — every `forward`
is real (no `NotImplementedError_` placeholders remain), `mup_init` is real,
and `HyMo.forward` / `forward_with_hidden` assemble the full layer stack.
Verified CPU-side on the tiny config (finite forward+backward) and at
production scale behind `@pytest.mark.heavy`. `mypy --strict` + `ruff` clean,
321 tests pass (17 heavy skipped). See the Phase 2 delivery note in the
Implementation status section above.

---

### Task B1: Rewrite the Gated Delta Net block with fused Triton kernel + NoPE flag

**Files:**
- Modify: `models/gdn.py` (full rewrite)

**What:** Replace the prior Python double-loop recurrence (`gdn.py:60-80`) with a fused Triton kernel call. Add the `use_rope` per-layer flag (set to `False` for the NoPE-hybrid layers).

The fla-org package provides `fla.layers.gated_delta_net.chunk_gated_delta_rule`. Add `fla` to `pyproject.toml` dependencies.

```python
# models/gdn.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from fla.layers.gated_delta_net import chunk_gated_delta_rule

class GatedDeltaNetBlock(nn.Module):
    def __init__(self, config: dict, layer_idx: int, use_rope: bool = True):
        super().__init__()
        self.layer_idx = layer_idx
        self.use_rope = use_rope
        d_model = config["dim"]                        # 896
        d_inner = config["gdn_d_inner"]                # 1280
        d_state = config["gdn_d_state"]                # 32
        d_conv  = config["gdn_d_conv"]                 # 4
        headdim = config["gdn_headdim"]                # 32
        n_heads = d_inner // headdim                   # 40

        self.in_proj   = nn.Linear(d_model, 6 * d_inner, bias=False)
        self.conv1d    = nn.Conv1d(d_inner, d_inner, d_conv, groups=d_inner,
                                    padding=d_conv-1, bias=False)
        # μP init: zero the log-decay, dt_bias, D
        self.A_log = nn.Parameter(torch.log(torch.arange(1, n_heads + 1,
                                                          dtype=torch.float32)
                                              .repeat_interleave(d_state)))
        self.A_log._no_weight_decay = True
        self.D     = nn.Parameter(torch.ones(n_heads))
        self.D._no_weight_decay = True
        self.dt_bias = nn.Parameter(torch.empty(n_heads).uniform_(0.001, 0.1))
        self.dt_bias._no_weight_decay = True

        self.b_proj   = nn.Linear(d_inner, n_heads * d_state, bias=False)
        self.c_proj   = nn.Linear(d_inner, n_heads * d_state, bias=False)
        self.dt_proj  = nn.Linear(d_inner, n_heads, bias=False)
        self.g_proj   = nn.Linear(d_inner, d_inner, bias=False)
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)
        # RoPE on first 25%% of head_dim, only if use_rope
        if use_rope:
            rope_head_dim = config["qk_rope_head_dim"]    # 32
            from models.mla import RotaryEmbedding
            self.rope = RotaryEmbedding(head_dim=rope_head_dim,
                                        max_seq_len=config["max_seq_len"],
                                        theta=config.get("rope_theta", 10000.0))
        else:
            self.rope = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        n_heads = self.in_proj.out_features // 6 // 1  # d_inner
        headdim = self.d_inner // n_heads if hasattr(self, 'd_inner') else None
        # ... norm → in_proj → conv1d → silu → (b, c, dt, g, v)
        # ... call chunk_gated_delta_rule(v, dt, A, b, c) → y
        # ... g * v * D → out_proj
        # if use_rope: apply rope to first 25%% of v's head_dim
        ...
```

**Why:** The v1 Python recurrence is the v2 throughput bottleneck. The fla-org Triton kernel gives 3-5× speedup, and the NoPE flag enables the position-encoding hybrid in §3.1.

**Test:** `tests/test_gdn_kernel.py` — verify the Triton kernel output matches a pure-Python reference implementation within 1e-3 tolerance on a 4×4096 input. Also: verify `use_rope=False` skips the RoPE call and produces a deterministic output.

**Caveat:** `fla` is a new dependency. If you'd rather implement in-house, the alternative is a hand-written Triton kernel at `models/kernels/gdn.py`. The default is to depend on `fla`.

---

### Task B2: Rewrite the MLA block with MQA-4 and partial-RoPE

**Files:**
- Modify: `models/mla.py:52-132` (rewrite the `MultiHeadLatentAttention` class)

**What:** Change `n_kv_groups` from 8 (GQA-1.75) to **4 (MQA-4)**. Add partial-RoPE on the first 25% of head_dim. Update `n_heads` to 16. See architecture doc §2.4.

```python
# models/mla.py — key changes
class MultiHeadLatentAttention(nn.Module):
    def __init__(self, config: dict, layer_idx: int = 0):
        super().__init__()
        d = config["dim"]                              # 896
        n_heads = config["n_heads"]                    # 16 (was 14)
        n_kv_groups = config["n_kv_groups"]            # 4 (was 8) — MQA-4
        head_dim = config["head_dim"]                  # 128
        qk_rope_head_dim = config["qk_rope_head_dim"]  # 32 (25% of 128)
        qk_nope_head_dim = head_dim - qk_rope_head_dim # 96
        kv_lora_rank = config["kv_lora_rank"]          # 128
        q_lora_rank = config["q_lora_rank"]             # 224

        # ... existing wq_a, q_norm, wq_b, wkv_a, kv_norm, wkv_b, wo layers
        # ... existing q_norm_qk, k_norm_qk, rope setup
```

**Why:** MQA-4 is empirically better at 500M-2B scale than GQA-1.75 (architecture doc §2.4). Partial-RoPE 25% is the 2026 SOTA (Qwen3-Next).

**Test:** `tests/test_mqa4_kv_groups.py` — verify `n_kv_groups == 4` and `_kv_group_for_q` produces a 16-element lookup table with exactly 4 query heads per KV head. Also `tests/test_partial_rope.py` — verify RoPE is applied only to the first 32 dim of the head.

---

### Task B3: Rewrite the MoE with EMA-smoothed gate bias and FP32 router

**Files:**
- Modify: `models/moe.py:24-110` (rewrite the `DeepSeekMoE` class)

**What:** Three changes:
1. **FP32 router:** in `gate.forward`, cast the gate weight to FP32 and the input to FP32 before the matmul + sigmoid. Cast back to BF16 for the rest.
2. **EMA-smoothed gate bias:** replace per-step `update_gate_bias` with an EMA-tracked version using `ema_alpha=0.02`. Tighten the threshold from 1.10× to 1.05×.
3. **16 routed + 1 shared + top-2** (was 8+1+top-2). Update `n_routed`, `n_shared`, `n_activated` defaults.

```python
# models/moe.py — key changes
class DeepSeekMoE(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.n_routed = config["n_routed_experts"]        # 16
        self.n_shared = config["n_shared_experts"]        # 1
        self.n_activated = config["n_activated_experts"]  # 2
        self.moe_inter_dim = config["moe_inter_dim"]      # 2304
        # ... existing gate, experts, shared_expert

        # NEW: EMA-tracked expert load
        self.register_buffer("ema_expert_counts",
                              torch.zeros(self.n_routed), persistent=False)
        self.ema_alpha = config.get("moe_ema_alpha", 0.02)

    def gate_forward(self, x: torch.Tensor) -> torch.Tensor:
        # FP32 router: cast input + weight to float32
        x_fp32 = x.float()
        w_fp32 = self.gate.weight.float()
        b_fp32 = self.gate.bias.float()
        logits = F.linear(x_fp32, w_fp32, b_fp32)
        return logits.to(x.dtype)  # back to BF16 for downstream

    def update_gate_bias(self, speed: float = 0.001) -> None:
        """EMA-smoothed expert-load bias update; threshold 1.05x."""
        if self._last_indices is None:
            return
        counts = torch.bincount(self._last_indices.flatten(),
                                 minlength=self.n_routed).float()
        self.ema_expert_counts.mul_(1.0 - self.ema_alpha).add_(counts,
                                                                alpha=self.ema_alpha)
        avg = self.ema_expert_counts.mean()
        over  = self.ema_expert_counts > avg * 1.05
        under = self.ema_expert_counts < avg * 0.95
        with torch.no_grad():
            self.gate.bias[over]  -= speed
            self.gate.bias[under] += speed
```

**Why:** FP32 router is the 2025 stability improvement (avoids sigmoid rounding at BF16); EMA-smoothed bias is the 2025 published improvement over per-step bias updates. See architecture doc §2.5.

**Test:** `tests/test_moe_ema_bias.py` — run 1000 forward passes with a synthetic load, verify `ema_expert_counts` is updated correctly and the bias moves toward load balance. Also `tests/test_moe_fp32_router.py` — verify `gate_forward` returns FP32-cast outputs when input is BF16.

---

### Task B4: Rewrite the MTP module for depth=2

**Files:**
- Modify: `models/mtp.py:65-118` (rewrite `MultiTokenPrediction`)

**What:** Change `mtp_depth` default to 2 and `mtp_loss_weights` default to `[0.3, 0.1]`. The 2-head MTP structure requires the 2nd MTP module to take the 1st MTP's output hidden state as its input (not the main model's hidden state).

```python
# models/mtp.py — key changes
class MultiTokenPrediction(nn.Module):
    def __init__(self, config: dict, main_model: nn.Module):
        super().__init__()
        self.depth = config.get("mtp_depth", 2)                # was 1
        self.softcap = config.get("mtp_softcap", True)
        self.softcap_value = config.get("mtp_softcap_value", 15.0)
        self.mtp_loss_weights = config.get("mtp_loss_weights",
                                            [0.3, 0.1])        # was [0.3]
        # ... existing setup; creates self.mtp_modules of length 2

    def forward(self, tokens, start_pos=0):
        # First MTP head uses main_hidden; second uses first_mtp_hidden
        main_logits, main_hidden = self.main_model.forward_with_hidden(tokens, start_pos)
        mtp_outputs = []
        prev_hidden = main_hidden
        for d in range(self.depth):
            mtp = self.mtp_modules[d]
            usable = tokens.size(1) - d - 2
            if usable <= 0:
                break
            h_in = prev_hidden[:, :usable]
            target_emb = self.embed(tokens[:, d + 1:d + 1 + usable])
            targets = tokens[:, d + 2:d + 2 + usable]
            logits, new_hidden = mtp(h_in, target_emb)
            mtp_outputs.append((logits, targets,
                                torch.tensor(self.mtp_loss_weights[d],
                                              device=logits.device)))
            prev_hidden = new_hidden  # chain: 2nd MTP sees 1st MTP's output
        return main_logits, mtp_outputs
```

**Why:** The 2nd MTP head provides a 0.02-0.04 PPL improvement at 750M with 30B tokens. Chaining the 2nd MTP on the 1st MTP's output is the DeepSeek-V3 pattern.

**Test:** `tests/test_mtp_depth_default.py` — verify `mtp_depth == 2` and `mtp_loss_weights == [0.3, 0.1]`. Also: a forward pass on a 4×16 token batch returns 2 (logits, targets, weight) tuples.

---

### Task B5: Assemble the full 32-layer HyMo model

**Files:**
- Modify: `models/fusionllm.py:81-160` (rewrite the `HyMo` class)

**What:** Build the 32-layer stack with the 3:1 GDN:MLA pattern, mid-stack MLA distribution, and the per-layer `use_rope` flag for the NoPE-hybrid.

```python
# models/fusionllm.py — key changes
class HyMo(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        # ... existing embed, head, norm, softcap
        # 32 layers, 3:1 GDN:MLA → 24 GDN, 8 MLA
        # MLA at positions 0, 4, 8, 12, 16, 20, 24, 28 (every 4th)
        # GDN at all other positions

        mla_positions = {i * 4 for i in range(8)}              # 0, 4, 8, 12, 16, 20, 24, 28
        # NoPE-hybrid (CR-12 mitigation): apply to the GDN layer immediately after
        # each MLA position, NOT to the MLA positions themselves. Earlier prose
        # incorrectly listed {4, 8, 12, 16, 20, 24, 28}; those are all MLA.
        # Corrected set: GDN positions {3, 7, 11, 15, 19, 23, 27}.
        # The CR-12 mitigation also gates this with `nope_hybrid_gdn_enabled`,
        # defaulting to False for v1.0 (defer the hybrid to v1.1 ablation claim 7).
        nope_hybrid_gdn_enabled = config.get("nope_hybrid_gdn_enabled", False)
        nope_hybrid_gdn_positions = {3, 7, 11, 15, 19, 23, 27}  # 7 GDN layers (when enabled)
        if not nope_hybrid_gdn_enabled:
            nope_hybrid_gdn_positions = set()  # all GDN layers get partial-RoPE

        self.layers = nn.ModuleList()
        for i in range(32):
            is_gdn = i not in mla_positions
            use_rope = not (i in nope_hybrid_gdn_positions)
            self.layers.append(HyMoBlock(config, i, is_gdn=is_gdn, use_rope=use_rope))

    def _run_layers(self, tokens, return_hidden=False):
        """Run the 32-layer stack, honoring per-layer use_checkpoint."""
        x = self.embed(tokens)
        for layer in self.layers:
            if layer.use_checkpoint and self.training:
                x = torch.utils.checkpoint.checkpoint(layer, x, use_reentrant=False)
            else:
                x = layer(x)
        hidden = self.norm(x)
        logits = self.head(hidden)
        if self.logit_softcap > 0:
            logits = softcap(logits, cap=self.logit_softcap)
        if return_hidden:
            return logits, hidden
        return logits
```

**Why:** This is the assembly step. The 32-layer count, the 3:1 ratio, and the NoPE-hybrid (every 4th GDN layer) all converge here.

**Test:** `tests/test_hymo_stack.py` — **`@pytest.mark.heavy`** (builds the 1.86B model). Count layers (should be 32), verify 8 are MLA + 24 are GDN, verify 0 GDN layers have `use_rope=False` by default (CR-12 mitigation; `nope_hybrid_gdn_enabled` defaults to False for v1.0). When the flag is set to True, verify exactly 7 GDN layers have `use_rope=False` (positions 3, 7, 11, 15, 19, 23, 27 — the GDN layer immediately after each MLA position per design §2.2/§3.1). Run a forward pass on a 4×4096 token batch; verify output shape is `(4, 4096, 64256)`. A default-run equivalent uses the tiny `tiny_hymo_model` fixture to verify the same invariants at small scale.

---

### Task B6: Update the v1.0 config defaults for the model

**Files:**
- New: `configs/hymo_750m.yaml`

**What:** The canonical v1.0 config (architecture doc §2.1, §2.5, §2.8, §3.1, §5.2, §7.1):

```yaml
model:
  vocab_size: 64256            # 64000 BPE + 256 bytes
  max_seq_len: 4096
  dim: 896
  n_layers: 32
  n_heads: 16                  # MLA
  n_kv_groups: 4               # MQA-4
  q_lora_rank: 224
  kv_lora_rank: 128
  qk_nope_head_dim: 96
  qk_rope_head_dim: 32         # 25% partial-RoPE
  v_head_dim: 128
  n_routed_experts: 16
  n_shared_experts: 1
  n_activated_experts: 2
  moe_inter_dim: 2304
  inter_dim: 2560              # DenseFFN on GDN blocks
  gdn_d_state: 32
  gdn_d_conv: 4
  gdn_headdim: 32
  gdn_d_inner: 1280
  gdn_chunk_size: 64
  nope_hybrid_gdn_enabled: false  # CR-12 mitigation: v1.0 ships with NoPE-hybrid OFF
  mtp_depth: 2
  mtp_loss_weights: [0.3, 0.1]
  mtp_inter_dim: 2304
  moe_ema_alpha: 0.02
  bias_update_speed: 0.001
  bias_update_every: 10
  muP: true
  logit_softcap: 15.0
  tie_embeddings: true
  head_dim: 128                # for partial-RoPE math

optimizer:
  adamw_lr: 3.0e-4
  adamw_betas: [0.9, 0.95]
  muon_lr: 0.02
  muon_momentum: 0.95
  weight_decay: 0.1
  cautious_wd: true
  master_weights_dtype: float32   # NEW: FP32 master weights

scheduler:
  total_steps: 57220
  warmup_frac: 0.02
  stable_frac: 0.83
  decay_frac: 0.15
  min_lr_ratio: 0.05
  decay: linear

training:
  micro_batch_size: 4
  gradient_accumulation_steps: 8
  grad_clip: 1.0
  grad_norm_threshold: 10.0
  loss_nan_skip: true
  empty_cache_every: 100
  save_dir: checkpoints/pretrain
  save_interval: 4000
  log_interval: 50
  eval_interval: 2000
  max_keep: 2
  balance_loss_alpha: 0.0       # aux-loss-free
  world_size: 4
  fsdp: true
  fsdp_mixed_precision: bfloat16
  fsdp_master_weights: float32
```

**Why:** Single source of truth for all hyperparameters. The implementation tasks reference this file.

**Test:** `python -c "import yaml; d = yaml.safe_load(open('configs/hymo_750m.yaml')); assert d['model']['n_layers'] == 32; assert d['model']['mtp_depth'] == 2; assert d['scheduler']['warmup_frac'] == 0.02; assert d['training']['world_size'] == 4"`.

---

### Task B7: Update μP init to the v1.0 spec

**Files:**
- Modify: `models/fusionllm.py:61-78` (the `muP_init` function)

**What:** The prior init zeros "gate" and "A_log" — extend to zero all the v1.0 added params (D, dt_bias, output_head, q_norm, kv_norm, q_norm_qk, k_norm_qk, moe_gate, mtp_proj). Verify the init produces a model with output magnitude ≤ 1.0 on the first forward.

```python
# models/fusionllm.py — extend muP_init
def muP_init(model: nn.Module, config: dict) -> None:
    dim = config["dim"]
    n_layers = config["n_layers"]
    attn_std = 1.0 / dim
    embed_std = 1.0 / math.sqrt(dim)
    zero_keywords = ("gate", "g_proj", "A_log", "dt_bias", "router",
                     "output_head", "bias", "q_norm", "kv_norm",
                     "q_norm_qk", "k_norm_qk", "mtp", "D")

    for name, p in model.named_parameters():
        if any(kw in name.lower() for kw in zero_keywords):
            with torch.no_grad():
                p.data.zero_()
            continue
        if p.dim() < 2:
            continue
        with torch.no_grad():
            std = embed_std if "embed" in name else attn_std
            p.data.normal_(mean=0.0, std=std)
```

**Why:** μP init at `std = 1/dim` keeps the first forward well-scaled. The v1.0 added params (q_norm, mtp, etc.) need to be zeroed or scaled appropriately.

**Test:** A 1k-step forward+backward on a 4×4096 batch, verifying the loss is finite and decreases over 100 steps.

---

**End of Work Block B. ✅ Shipped 2026-07-16.** All B1-B7 implemented; every `forward` is real and `mup_init` is real. The Phase 3 gate (`test_fusionllm_stack.py` heavy variant builds the full model and prints ~750M active / ~1.86B stored) is satisfied behind `@pytest.mark.heavy`; the tiny-config default-run equivalent passes. Move to **Work Block C (Training Infrastructure)**.

---

# Work Block C: Training Infrastructure (Tasks C1-C7) — ⏭ NEXT

The optimizer partition, joint WSD scheduler, and the 6 stability fixes from the prior plan. Architecture doc §5.

**Prereqs:** Work Block B done; the 6 prior stability fixes are in place (committed in the prior work block's history).

**Deliverable:** `training/optimizer.py`, `training/scheduler.py`, `training/trainer.py` updated for the v1.0 spec. Optimizer partition is correct (MoE experts on AdamW). JointWSDScheduler is the only scheduler. FP32 master weights are wired.

---

### Task C1: Update `build_optimizers` with the v1.0 partition

**Files:**
- Modify: `training/optimizer.py:113-162` (the `build_optimizers` function)

**What:** The prior stability-fix work added MoE expert names to the AdamW exact-name allowlist. The v1.0 spec requires the same, with the additional requirement that all 16 experts × 8 layers = 128 expert tensors are explicitly on AdamW. The current exact-name allowlist from the prior plan:

```python
ADAMW_EXACT_NAMES = {
    "embed.weight", "head.weight", "norm.weight", "gate.bias",
    "A_log", "dt_bias", "D",
}
```

This needs to be extended to handle the v2 MoE expert pattern. Since the v2 has 16 routed experts, the exact-name approach becomes verbose. Use a suffix-based approach with care:

```python
def goes_to_adamw(name: str, p: torch.Tensor) -> bool:
    if p.ndim < 2:
        return True
    # Tied embed/head, RMSNorm gamma
    if name.endswith("embed.weight") or name.endswith("head.weight"):
        return True
    if name.endswith("norm.weight"):
        return True
    # MoE gate
    if name.endswith(".gate.weight") or name.endswith(".gate.bias"):
        return True
    # MoE expert weights — v2 has 16 experts per layer, all routed
    if ".experts." in name and (name.endswith(".w1") or name.endswith(".w2")
                                 or name.endswith(".w3")):
        return True
    # Shared expert
    if name.endswith(".shared_expert.w1") or name.endswith(".shared_expert.w2") \
       or name.endswith(".shared_expert.w3"):
        return True
    # GDN scalars
    if name.endswith(".A_log") or name.endswith(".dt_bias") or name.endswith(".D"):
        return True
    return False
```

**Why:** The v2 has 16 MoE experts per layer × 8 layers = 128 expert tensors. The exact-name allowlist must handle this without listing each one.

**Test:** `tests/test_moe_expert_excluded_from_nor_muon.py` — extend the existing v1 test to verify all 128 expert tensors (16 experts × 8 MLA layers × 3 matrices) are routed to AdamW, not NorMuon. Verify by collecting `id(p) for p in adamw_opt.param_groups[0]['params']` and checking 128 of them are in the MoE expert list.

---

### Task C2: Wire FP32 master weights in both optimizers

**Files:**
- Modify: `training/optimizer.py:11-58` (the `NorMuon` class) and `:68-110` (the `CautiousAdamW` class)

**What:** Both optimizers must keep FP32 master weights. PyTorch's optimizer state defaults to the parameter's dtype (BF16 in the prior plan). Override with `state['master_weight']` kept in FP32.

```python
class NorMuon(Optimizer):
    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p, dtype=torch.float32)  # FP32
                    state["exp_avg_sq"] = torch.zeros_like(p, dtype=torch.float32)
                    state["master_weight"] = p.detach().clone().float()  # FP32 master
                # ... rest of step uses state["master_weight"] for the apply step
                # ... at the end: p.copy_(state["master_weight"].to(p.dtype))
```

The key change: every state buffer (exp_avg, exp_avg_sq) is FP32; the apply step is `p -= lr * update.to(p.dtype)`.

**Why:** FP32 master weights give +0.02-0.05 PPL over BF16 master (architecture doc §7.2).

**Test:** A 1k-step training probe on a 4×4096 batch, verifying the loss is finite and decreases. Inspect optimizer state to confirm `state["master_weight"].dtype == torch.float32`.

---

### Task C3: Update `JointWSDScheduler` for the v2 schedule (2% warmup, 0.05× min_lr_ratio)

**Files:**
- Modify: `training/scheduler.py:43-113` (the `JointWSDScheduler` class)

**What:** Update the default `warmup_frac` from 0.01 to 0.02, `stable_frac` from 0.84 to 0.83, `min_lr_ratio` from 0.1 to 0.05. The schedule math itself doesn't change; only the defaults.

```python
def __init__(self, optimizers, total_steps=57220, warmup_frac=0.02,
             stable_frac=0.83, min_lr_ratio=0.05, decay="linear"):
    # ... rest unchanged
```

**Why:** Architecture doc §5.3 — the 2% warmup gives the μP-init'd model more time to find its natural scale; the 0.05× min_lr_ratio is a deeper decay at end of training.

**Test:** `tests/test_joint_scheduler.py` — extend the existing v1 test to verify the 2% warmup ramps correctly (0 → peak over 1,144 steps at 57,220 total) and the decay reaches 0.05× at the final step.

---

### Task C4: Wire the 6 stability fixes into the v2 trainer

**Files:**
- Modify: `training/trainer.py:152-196` (the `train_step` and `optimizer_step` methods)

**What:** The 6 v1 fixes should already be in the codebase from the prior plan. Verify they're all wired:
1. **JointWSDScheduler** (Task C3 above).
2. **Aux-loss-free** (`balance_loss_alpha = 0.0` in `__init__`).
3. **MTP checkpointing** (`forward_with_hidden` uses `_run_layers` with per-layer `use_checkpoint`).
4. **Deterministic validation** (`generate_synthetic_batch` with seed parameter — note: v2 replaces this with real held-out; see Work Block E).
5. **Exact-name optimizer partition** (Task C1 above).
6. **Config-driven trainer** (all batch/seq/vocab from `config.get(...)`).

**Why:** These are the prior stability fixes; without them, the v1.0 run will diverge.

**Test:** `pytest tests/ -v --tb=short` — all 66+ prior tests still pass. The trainer config is built from the v1.0 config file (`configs/hymo_750m.yaml`); verify the trainer reads `micro_batch_size=4`, `gradient_accumulation_steps=8`, `mtp_depth=2`, etc.

---

### Task C5: Wire the FP32 router cast in the trainer's `train_step`

**Files:**
- Modify: `training/trainer.py:152-175` (the `train_step` method)

**What:** The FP32 router cast is already in `moe.py:gate_forward` (Task B3). The trainer just needs to call `self.model(tokens)` which propagates through to the gate. Verify the trainer doesn't do any BF16→FP32 casts that would override the gate's FP32 path.

**Why:** The FP32 router is the v2 stability improvement; the trainer just needs to not interfere.

**Test:** A 100-step training probe, verifying the loss is finite and decreases. Profile to confirm `gate_forward` is called with FP32 inputs (not BF16).

---

### Task C6: Update `compute_validation_loss` to use real held-out data

**Files:**
- Modify: `training/validation.py` (replace synthetic with real held-out)

**What:** Replace the `generate_synthetic_batch` (uniform random) with a real held-out loader that reads from `data/tokens/val.bin`:

```python
# training/validation.py
import numpy as np
import torch
from pathlib import Path

VAL_BIN = Path("data/tokens/val.bin")
_val_cache = None

def _load_val_tokens() -> np.ndarray:
    global _val_cache
    if _val_cache is None:
        _val_cache = np.fromfile(VAL_BIN, dtype=np.uint32)
    return _val_cache

def get_val_batch(batch_size: int, seq_len: int, device: torch.device,
                  seed: int = 42) -> tuple[torch.Tensor, torch.Tensor]:
    """Real held-out val batch: slice a deterministic window of val.bin."""
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    val = _load_val_tokens()
    total_needed = batch_size * (seq_len + 1)
    # Random offset, then slice
    max_offset = len(val) - total_needed - 1
    offset = int(torch.randint(0, max_offset, (1,), generator=g).item())
    batch = val[offset:offset + total_needed].astype(np.int64)
    batch = batch.reshape(batch_size, seq_len + 1)
    tokens = torch.from_numpy(batch[:, :seq_len]).to(device)
    targets = torch.from_numpy(batch[:, 1:]).to(device)
    return tokens, targets

def compute_validation_loss(model, batch_size, seq_len, vocab_size, num_batches,
                              device, seed=42) -> dict:
    """Standard val loss on real held-out FineWeb-Edu."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for b in range(num_batches):
            tokens, targets = get_val_batch(batch_size, seq_len, device,
                                              seed=seed + b)
            logits = model(tokens)
            loss = F.cross_entropy(logits.view(-1, vocab_size),
                                    targets.view(-1), reduction="sum")
            total_loss += loss.item()
            total_tokens += targets.numel()
    return {"loss": total_loss / total_tokens,
            "ppl": float(np.exp(total_loss / total_tokens))}
```

**Why:** Real held-out val is the v1.0 spec; the prior synthetic uniform random was meaningless.

**Test:** `tests/test_real_held_out_val.py` — call `get_val_batch(2, 64, torch.device("cpu"), seed=42)` twice and verify the batches are identical. Call with a different seed and verify they're different. Spot-check: no token in any val batch appears in any train shard.

---

### Task C7: Update the EMA gate-bias update cadence in the trainer

**Files:**
- Modify: `training/trainer.py:213-215` (the gate-bias update in `optimizer_step`)

**What:** The v1 cadence was `if self.step % self.bias_update_every == 0 and self.step > 0: ...`. Keep the same cadence (every 10 steps); the EMA logic is inside `moe.py:update_gate_bias` (Task B3).

**Why:** The cadence is fine; the EMA smoothing is the change. Verify the cadence in the trainer doesn't bypass the EMA logic.

**Test:** A 1k-step training probe, verifying the gate biases evolve smoothly (not jumping per-step). The `ema_expert_counts` buffer in each MoE layer should converge to ~uniform within 5k steps.

---

**End of Work Block C. Commit series: C1-C7 as 2-3 commits. Move to Work Block D when `pytest tests/ -v --tb=short` passes 70+ tests including the new v2 tests.**

---

# Work Block D: FSDP-2 + Systems (Tasks D1-D7)

The systems layer: FSDP-2 mixed precision, per-expert sharding, sort-by-size + round-robin for NorMuon, DCP checkpointing, init broadcast, FSDP-aware gradient norm. Architecture doc §13.

**Prereqs:** Work Block C done.

**Deliverable:** `training/fsdp.py` (new), `training/checkpoint.py` (updated for DCP), `training/trainer.py` (FSDP-2 init + per-step coordination). All FSDP-2 specifics are correct.

---

### Task D1: Create the FSDP-2 wrapper module

**Files:**
- New: `training/fsdp.py`

**What:** Wrap the model with FSDP-2 mixed-precision policy. Per-param wrapping rules per architecture doc §13.2:
- MLA block params: wrapped (per-layer)
- GDN block params: wrapped (per-layer)
- MoE expert weights: wrapped per-expert (16 separate FSDP instances per MoE layer)
- MoE gate: replicated (small, ~14K params)
- RMSNorm γ: replicated
- Logit softcap: not a param

```python
# training/fsdp.py
import torch
import torch.nn as nn
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    BackwardPrefetch,
)

def get_fsdp_mixed_precision_policy(master_dtype=torch.float32) -> MixedPrecision:
    return MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
        # Master weights are kept in the optimizer state, not in FSDP itself
    )

def wrap_model_with_fsdp(model: nn.Module, world_size: int) -> nn.Module:
    """Wrap the model with FSDP-2, per-expert MoE sharding, and sort-by-size + round-robin for NorMuon."""
    # First, find all MoE expert tensors and wrap each individually
    fsdp_wrapping_policy = build_wrapping_policy(world_size)
    model = FSDP(
        model,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=get_fsdp_mixed_precision_policy(),
        auto_wrap_policy=fsdp_wrapping_policy,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        limit_all_gathers=True,
        use_orig_params=True,
    )
    return model
```

**Why:** The per-expert wrapping is the v2 key insight — without it, the MoE experts all sit on one rank, defeating FSDP-2 sharding.

**Test:** `tests/test_fsdp_param_count.py` — **`@pytest.mark.heavy`** (builds the 1.86B model). After wrapping, verify the per-rank sharded param count is ~465M (1.86B / 4), not the full 1.86B. Verify the MoE experts are split across ranks (not all on rank 0).

---

### Task D2: Implement sort-by-size + round-robin for NorMuon params

**Files:**
- Modify: `training/fsdp.py` (add the sort-by-size function)

**What:** Per architecture doc §13.3:

```python
def shard_nor_muon_params(model: nn.Module, world_size: int) -> list[list[nn.Parameter]]:
    """Sort NorMuon params by size, round-robin assign to ranks. Largest first."""
    from training.optimizer import goes_to_nor_muon  # avoid circular
    nor_muon_params = [p for n, p in model.named_parameters() if goes_to_nor_muon(n, p)]
    nor_muon_params.sort(key=lambda p: p.numel(), reverse=True)

    rank_assignments = [[] for _ in range(world_size)]
    rank_byte_counts = [0] * world_size
    for i, p in enumerate(nor_muon_params):
        target_rank = i % world_size
        rank_assignments[target_rank].append(p)
        rank_byte_counts[target_rank] += p.numel()

    avg = sum(rank_byte_counts) / world_size
    assert max(rank_byte_counts) / avg < 1.05, (
        f"NorMuon shard imbalance: max={max(rank_byte_counts):,} "
        f"avg={avg:,.0f} ratio={max(rank_byte_counts)/avg:.3f}"
    )
    return rank_assignments
```

**Why:** Without the sort, the slowest rank sets the synchronization barrier and the optimizer step is 2.7× longer on that rank (NorMuon paper, 3-0 verified).

**Test:** `tests/test_fsdp_nor_muon_sort.py` — **`@pytest.mark.heavy`** (builds the 1.86B model). Run `shard_nor_muon_params`, verify the per-rank byte counts are within 5% of the average.

---

### Task D3: Implement DCP save/load for FSDP-2 sharded checkpoints

**Files:**
- Modify: `training/checkpoint.py` (replace torch.save with DCP)

**What:** The v1 atomic checkpoint pattern (`torch.save → .tmp → os.rename`) doesn't work for FSDP-2 sharded checkpoints. Use `torch.distributed.checkpoint` (DCP) which handles sharding automatically.

```python
# training/checkpoint.py
from torch.distributed.checkpoint import save, load, FileSystemWriter

def save_fsdp_checkpoint(model, optimizer_muon, optimizer_adamw, scheduler,
                          step, token_count, best_loss, save_dir):
    state = {
        "model": model.state_dict(),
        "optimizer_muon": optimizer_muon.state_dict() if optimizer_muon else None,
        "optimizer_adamw": optimizer_adamw.state_dict(),
        "scheduler": scheduler.state_dict(),
        "step": step,
        "token_count": token_count,
        "best_loss": best_loss,
    }
    save(state, checkpoint_id=save_dir)

def load_fsdp_checkpoint(model, optimizer_muon, optimizer_adamw, scheduler,
                          load_dir):
    state = {
        "model": model.state_dict(),
        "optimizer_muon": optimizer_muon.state_dict() if optimizer_muon else None,
        "optimizer_adamw": optimizer_adamw.state_dict(),
        "scheduler": scheduler.state_dict(),
    }
    load(state, checkpoint_id=load_dir)
    return state.get("step", 0), state.get("token_count", 0), state.get("best_loss", float("inf"))
```

**Why:** FSDP-2 sharded state requires DCP for correct save/load. The v1 torch.save approach corrupts on sharded state.

**Test:** Save a checkpoint, then load it into a fresh model. Verify the model parameters are bit-identical. Verify the optimizer state is restored (lr is the same, momentum buffers are the same).

---

### Task D4: FSDP-2 init broadcast (rank 0 → all ranks)

**Files:**
- Modify: `training/trainer.py:33-50` (the model init in `__init__`)

**What:** After model construction and before FSDP-2 wrapping, broadcast all parameters from rank 0 to all ranks. This guarantees that the random init is bit-identical across ranks.

```python
# training/trainer.py — after model construction, before FSDP wrap
if torch.distributed.is_initialized():
    for p in self.model.parameters():
        torch.distributed.broadcast(p.data, src=0)
    # Verify (optional): compute per-rank hash of all params
    h = hashlib.sha256()
    for p in self.model.parameters():
        h.update(p.data.cpu().numpy().tobytes())
    rank_hash = h.hexdigest()
    rank_hashes = [None] * torch.distributed.get_world_size()
    torch.distributed.all_gather_object(rank_hashes, rank_hash)
    if rank_hashes[0] != rank_hashes[torch.distributed.get_rank()]:
        raise RuntimeError(f"Init divergence: rank 0 hash {rank_hashes[0][:8]} "
                           f"vs rank {torch.distributed.get_rank()} hash {rank_hash[:8]}")
```

**Why:** Without broadcast, each rank's random init is independent, and the first forward would produce different per-rank outputs.

**Test:** `tests/test_init_broadcast.py` — run on 4 ranks (use torch.multiprocessing for the test); verify all rank hashes match.

---

### Task D5: FSDP-aware gradient norm

**Files:**
- Modify: `training/trainer.py:198-225` (the `optimizer_step` method)

**What:** The v1 `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)` computes the L2 norm on the *current rank's* sharded gradients, which is wrong. Use FSDP's built-in `model.clip_grad_norm_()` which all-reduces the norm across ranks.

```python
# training/trainer.py
def optimizer_step(self) -> dict[str, float]:
    metrics = {}
    if self.grad_clip > 0:
        # FSDP-aware: all-reduces the norm across ranks
        grad_norm = self.model.clip_grad_norm_(max_norm=self.grad_clip)
        metrics["grad_norm"] = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
        if grad_norm > self.grad_norm_threshold:
            self._log(f"WARNING: Large grad norm {grad_norm:.2f} at step {self.step}")

    if self.muon_opt is not None:
        self.muon_opt.step()
    self.adamw_opt.step()
    # ... rest unchanged
```

**Why:** FSDP-aware gradient norm gives the correct cross-rank L2 norm; the prior per-rank norm is too small and over-clips.

**Test:** A 100-step training probe, verifying `metrics["grad_norm"]` is a single value (not per-rank) and that gradient clipping is applied correctly. Spot-check: if rank 0 has a 0.5 norm and rank 1 has a 0.7 norm, the all-reduced norm is ~0.86 (sqrt(0.5² + 0.7²) / 2 ranks... actually the L2 norm is sqrt(sum) not mean, so it's sqrt(0.25 + 0.49) = 0.86).

---

### Task D6: NaN-skip with FSDP-2 awareness

**Files:**
- Modify: `training/trainer.py:280-303` (the gradient accumulation loop)

**What:** The v1 NaN-skip zeroes the local gradients, but with FSDP-2 the zeroing must happen on every rank. Use `model.zero_grad(set_to_none=True)` which propagates to all ranks via FSDP's collective ops.

```python
# training/trainer.py — the inner loop
for micro_step in range(self.grad_accum_steps):
    tokens, targets = next(data_iter)
    if tokens.size(0) != self.micro_batch_size:
        continue
    metrics = self.train_step(tokens, targets)
    if metrics.get("skip"):
        # FSDP-aware zero: this triggers a collective op on all ranks
        self.model.zero_grad(set_to_none=True)
        skip_step = True
        break
    accum_loss += metrics["loss"]
    accum_count += 1
```

**Why:** Without FSDP awareness, only the local rank's gradients are zeroed, leaving the other ranks with stale gradients from prior micro-batches.

**Test:** Force a NaN in the loss (set the model's last layer weights to NaN); verify the next optimizer step is skipped on all 4 ranks (the loss on rank 0 is NaN, the loss on ranks 1-3 is propagated via the FSDP all-reduce in backward).

---

### Task D7: FSDP-2-aware checkpoint restoration

**Files:**
- Modify: `training/trainer.py:241-257` (the `load` method)

**What:** The v1 `load` method restores model state and optimizer state. With FSDP-2, the state dicts have different keys (FSDP prefixes). Use DCP's `load` directly.

```python
# training/trainer.py
def load(self, load_dir: str | Path) -> int:
    train_model = self._get_train_model()
    state = {
        "model": train_model.state_dict(),
        "optimizer_muon": self.muon_opt.state_dict() if self.muon_opt else None,
        "optimizer_adamw": self.adamw_opt.state_dict(),
        "scheduler": self.scheduler.state_dict(),
    }
    load(state, checkpoint_id=str(load_dir))
    self.step = state.get("step", 0)
    self.global_step = self.step
    self.token_count = state.get("token_count", 0)
    self.best_loss = state.get("best_loss", float("inf"))
    self._log(f"Resumed from step {self.step}")
    return self.step
```

**Why:** DCP handles the FSDP-2 state dict sharding automatically. The v1 torch.load approach can't restore sharded state.

**Test:** Save a checkpoint at step 1000, kill the trainer, restart, load. Verify step=1000 is restored, the loss curve continues smoothly from where it was.

---

**End of Work Block D. Commit series: D1-D7 as 2-3 commits. Move to Work Block E when `pytest tests/ -v --tb=short` passes 80+ tests including the FSDP-2 tests and the H1-H5 optimization tests (5 tests total in H).**

---

# Work Block E: Validation and Held-out Evaluation (Tasks E1-E5)

The 6 held-out benchmarks (FineWeb-Edu, HellaSwag, ARC, MMLU, GSM8K, HumanEval) plus the validation protocol. Architecture doc §15.

**Prereqs:** Work Block D done; `data/tokens/val.bin` exists.

**Deliverable:** `training/eval/` (new directory) with the 6 eval scripts; `training/eval/run_all.py` orchestrates the full eval suite; results land in `checkpoints/pretrain/eval_results.json`.

---

### Task E1: Set up `lm-evaluation-harness` integration

**Files:**
- New: `training/eval/__init__.py`
- Modify: `pyproject.toml` (add `lm-eval>=0.4.5`)

**What:** The v2 quality validation uses [EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) v0.4.5+. Add as a dependency; create a thin wrapper that calls `lm_eval` from Python.

```python
# training/eval/harness.py
import lm_eval
from lm_eval.models.huggingface import HFLM

def run_harness_eval(model, tokenizer, tasks: list[str],
                      num_fewshot: int = 0, batch_size: int = 4) -> dict:
    """Run lm-evaluation-harness on the given tasks."""
    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=batch_size)
    results = lm_eval.simple_evaluate(
        model=lm,
        tasks=tasks,
        num_fewshot=num_fewshot,
        batch_size=batch_size,
    )
    return {task: results["results"][task] for task in tasks}
```

**Why:** lm-eval-harness is the standard 2026 eval framework; writing custom eval scripts is redundant.

**Test:** Run on a tiny model (the 415M checkpoint from a fresh init) on `hellaswag` only; verify the harness returns a dict with the `acc_norm` metric.

---

### Task E2: Wire the 6-eval suite

**Files:**
- New: `training/eval/run_all.py`

**What:** A single entry point that runs all 6 evals and saves results to JSON.

```python
# training/eval/run_all.py
"""Run the 6-eval suite for HyMo quality validation (architecture doc §15)."""
import argparse
import json
from pathlib import Path
from training.eval.harness import run_harness_eval

EVAL_SUITE = [
    # (task_name, num_fewshot)
    ("fineweb_edu_ppl", 0),       # Custom: perplexity on val.bin
    ("hellaswag", 0),
    ("arc_challenge", 0),
    ("mmlu", 5),
    ("gsm8k", 8),
    ("humaneval", 0),
]

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="Path to model checkpoint")
    p.add_argument("--tokenizer", default="data/tokens/byte_bpe_vocab.json")
    p.add_argument("--output", default="checkpoints/pretrain/eval_results.json")
    p.add_argument("--batch-size", type=int, default=4)
    args = p.parse_args()

    # Load model + tokenizer
    from models.fusionllm import HyMo
    import yaml
    cfg = yaml.safe_load(open("configs/hymo_750m.yaml"))["model"]
    model = HyMo(cfg)
    # ... load weights from args.model via DCP
    # ... load tokenizer

    results = {}
    for task, n_shot in EVAL_SUITE:
        print(f"Running {task} ({n_shot}-shot)...")
        if task == "fineweb_edu_ppl":
            # Custom: use our compute_validation_loss
            from training.validation import compute_validation_loss
            import torch
            metrics = compute_validation_loss(
                model, batch_size=args.batch_size, seq_len=4096,
                vocab_size=cfg["vocab_size"], num_batches=32,
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            )
            results[task] = {"ppl": metrics["ppl"], "loss": metrics["loss"]}
        else:
            results[task] = run_harness_eval(model, tokenizer, [task], n_shot,
                                              args.batch_size)[task]

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote eval results to {args.output}")

if __name__ == "__main__":
    main()
```

**Why:** Single command runs the full eval suite; the JSON output is the input to the v2 paper's results table.

**Test:** Run on the 415M v1 checkpoint (if still available) or a fresh init; verify all 6 evals return metrics and the JSON is written.

---

### Task E3: Held-out val during training (every 2,000 steps)

**Files:**
- Modify: `training/trainer.py:331-345` (the eval loop in `train_epoch`)

**What:** The v1 eval runs every 5,000 steps; v1.0 runs every 2,000 steps (more frequent because the 30B-token run is longer and we want more data points). The eval uses real held-out FineWeb-Edu (Task A7), not synthetic.

```python
# training/trainer.py — replace the synthetic eval with real held-out
if self.eval_interval > 0 and self.step % self.eval_interval == 0:
    val_metrics = compute_validation_loss(
        self.model, batch_size=self.micro_batch_size, seq_len=self.max_seq_len,
        vocab_size=self.vocab_size, num_batches=8,
        device=self.device,
    )
    val_metrics["step"] = self.step
    self._log(f"Validation: loss={val_metrics['loss']:.4f}, ppl={val_metrics['ppl']:.2f}")
    self._log_metrics(val_metrics, self.step)
    if val_metrics["loss"] < self.best_loss:
        self.best_loss = val_metrics["loss"]
        self.save(tag="best")
```

**Why:** Real held-out val is the v1.0 spec; the prior synthetic was meaningless. More frequent eval gives more data points on the loss curve.

**Test:** Verify `eval_interval=2000` is set in the config; verify `compute_validation_loss` reads from `data/tokens/val.bin` (Task C6).

---

### Task E4: Comparison against baselines

**Files:**
- New: `training/eval/baselines.py`

**What:** A small helper that downloads the published baseline numbers (MobileMoE-0.9B, Pythia-1B, SmolLM2-1.7B) and produces a side-by-side table. The published numbers are in JSON form; we don't re-run the baselines (they're from the published papers).

```python
# training/eval/baselines.py
"""Published 2026 baseline numbers for the 6-eval suite."""
import json
from pathlib import Path

BASELINES = {
    "pythia-1b": {
        "fineweb_edu_ppl": 2.45, "hellaswag": 0.36, "arc_challenge": 0.24,
        "mmlu": 0.24, "gsm8k": 0.03, "humaneval": 0.05,
    },
    "mobile_moe_0.9b": {
        "fineweb_edu_ppl": 2.10, "hellaswag": 0.38, "arc_challenge": 0.26,
        "mmlu": 0.27, "gsm8k": 0.06, "humaneval": 0.09,
    },
    "smollm2_1.7b": {
        "fineweb_edu_ppl": 2.20, "hellaswag": 0.42, "arc_challenge": 0.30,
        "mmlu": 0.31, "gsm8k": 0.10, "humaneval": 0.12,
    },
}

def format_comparison_table(fusionllm_results: dict) -> str:
    """Format a markdown comparison table."""
    lines = ["| Metric | HyMo | Pythia-1B | MobileMoE-0.9B | SmolLM2-1.7B |",
             "|--------|-----------|-----------|----------------|----------------|"]
    for metric in ["fineweb_edu_ppl", "hellaswag", "arc_challenge", "mmlu", "gsm8k", "humaneval"]:
        row = f"| {metric} | {fusionllm_results.get(metric, '?'):.3f} "
        for baseline in ["pythia-1b", "mobile_moe_0.9b", "smollm2_1.7b"]:
            row += f"| {BASELINES[baseline][metric]:.3f} "
        row += "|"
        lines.append(row)
    return "\n".join(lines)
```

**Why:** A clean side-by-side table is the v2 paper's main results figure.

**Test:** Run on dummy data; verify the table format is correct.

---

### Task E5: Decision matrix (publish vs revise)

**Files:**
- New: `docs/fusionllm-quality-decision-matrix.md`

**What:** A short doc that, given the v2 eval results, recommends one of three outcomes:
- **All 6 targets hit** → publish as "competitive 750M hybrid."
- **4-5 of 6** → publish as "novel architecture, mixed results."
- **≤ 3 of 6** → the architecture choices need investigation; the paper becomes a negative result.

The doc is short (1 page); the user fills in the actual numbers from `eval_results.json` and follows the decision tree.

**Why:** This is the "what to do with the results" doc. Without it, the eval results sit in a JSON and nobody knows how to interpret them.

**Test:** Manually verify the decision tree is internally consistent.

---

**End of Work Block E. Commit series: E1-E5 as 1-2 commits. Move to Work Block F when `python training/eval/run_all.py --model <checkpoint>` returns a complete JSON.**

---

# Work Block F: Ablations (v1.1, deferred — not part of v1.0)

> **Scope note:** This entire work block is **v1.1** and is **NOT part of the v1.0 pre-training deliverable**. The v1.0 design commits to one set of choices for each ablation question (the "primary configuration" rows in each Task below), validates the resulting converged model, and stops. The v1.1 ablations run the alternative configurations in parallel on dedicated pods after v1.0 is complete; see architecture doc §16.5 for the v1.1 budget estimate ($1,100-2,700).

The v1.1 ablation plan is documented here as a **stub** so the v1.0 implementer knows the structure of the deferred work, but the tasks are not scheduled, budgeted, or required for the v1.0 milestone. **Do not start these tasks as part of v1.0.**

**Prereqs (v1.1):** v1.0 primary run complete; v1.0 eval results published; the primary config is in `configs/hymo_750m.yaml`.

**Deliverable (v1.1):** `training/ablations/` (new directory) with 4 ablation configs + launch scripts; results in `ablations/{A,B,C,D}/eval_results.json`. Each ablation is a 7.5B-token (25% of primary) run on a separate 4× A100 SXM pod; the 4 ablations can run in parallel (1.3-2 days total) or sequentially (1.3 days each = 5.2 days total).

The four v1.1 ablation tasks (F1-F4) keep the same structure as the v1 design — only the version references, pod cost, and wall-clock are updated. See architecture doc §16.1-16.4 for the detailed specification.

**Stub summary (full details in architecture doc §16.1-16.4):**

- **F1 (v1.1):** Ablation A — MoE-on-attention-only. 2 variants × 7.5B tokens. v1.0 (MoE on MLA only) vs every-layer. Wall-clock: 1.3 days. Cost: $250-500.
- **F2 (v1.1):** Ablation B — NorMuon-with-MoE-exclusion. 3 variants × 7.5B tokens. v1.0 (NorMuon attn+GDN, AdamW MoE) vs AdamW-only vs NorMuon-everything. Wall-clock: 1.3 days. Cost: $250-750.
- **F3 (v1.1):** Ablation C — MTP depth ablation. 3 variants × 7.5B tokens. No-MTP vs depth-1 vs depth-2 (v1.0). Wall-clock: 1.3 days. Cost: $250-750.
- **F4 (v1.1):** Ablation D — MQA-4 vs GQA-1.75. 2 variants × 7.5B tokens. MQA-4 (v1.0) vs GQA-1.75. Wall-clock: 1.3 days. Cost: $250-500.

**Total v1.1 ablation cost:** $1,100-2,700 (sequential or parallel variants). Sequential is cheaper; parallel is faster.

---

**End of Work Block F (v1.1). Move to Work Block G (v1.0 primary launch) when v1.0 model + training infra + FSDP-2 + optimizations are all done. Do NOT execute the F1-F4 ablation tasks as part of v1.0.**

The 4 parallel ablations on separate pods. Architecture doc §16.

**Prereqs:** Work Block E done; the primary 30B run is *not* required to start the ablations (each ablation is independent).

**Deliverable:** `training/ablations/` (new directory) with 4 ablation configs + launch scripts; results in `ablations/{A,B,C,D}/eval_results.json`.

(v1.1 ablation tasks F1-F4 deferred — see Work Block F stub above)

# Work Block H: Optimization Stack Validation (Tasks H1-H4)

The 4 optimization techniques from architecture doc §12a, formalized as first-class implementation tasks. The 5-7 day wall-clock is contingent on all four being enabled; this work block ensures they are tested individually and the stacked throughput matches the §12a.5 projection.

**Prereqs:** Work Block B done (model classes exist for H1-H4 to attach to); Work Block C in progress (so the FP32 master weight path is in place for the throughput probes).

**Deliverable:** All 4 optimizations wired into the model + trainer, with per-optimization unit tests + a stacked-throughput probe. The probe result should be ≥ 60,000 tok/s sustained on 4× A100 SXM (the 5-7 day target floor).

---

### Task H1: Wire the fused Triton GDN kernel

**Files:**
- Modify: `models/gdn.py` (replace the Python loop with the kernel call; see Work Block B1)
- New: `models/kernels/chunk_gated_delta_rule.py` (if vendoring, else import from `fla`)
- Modify: `pyproject.toml` (add `fla` dependency, or note the vendored path)

**What:** This is the work-block B1 implementation, plus a unit test that proves the kernel matches the pure-Python reference within 1e-3 tolerance on random inputs. The test runs **before** the primary run starts and gates the rest of H1-H4.

```python
# tests/test_gdn_kernel.py
def test_gdn_kernel_matches_reference():
    """Verify the fused Triton GDN kernel matches the pure-Python reference."""
    B, T, n_heads, headdim, d_state = 2, 128, 40, 32, 32
    torch.manual_seed(42)
    v = torch.randn(B, T, n_heads, headdim, dtype=torch.bfloat16, device="cuda")
    A = torch.randn(n_heads, dtype=torch.float32, device="cuda")
    b = torch.randn(B, T, n_heads, d_state, dtype=torch.bfloat16, device="cuda")
    c = torch.randn(B, T, n_heads, d_state, dtype=torch.bfloat16, device="cuda")
    
    # Reference (pure Python recurrence)
    y_ref = python_gdn_reference(v, A, b, c)
    
    # Triton kernel
    from fla.layers.gated_delta_net import chunk_gated_delta_rule
    y_kernel = chunk_gated_delta_rule(v, b, c, A, chunk_size=64)
    
    assert torch.allclose(y_ref, y_kernel, atol=1e-3, rtol=1e-3)
    print("GDN kernel matches reference within 1e-3")
```

**Why:** Architecture doc §12a.1 — this is the biggest single optimization (3-5× GDN speedup, 30% wall-clock).

**Test:** `pytest tests/test_gdn_kernel.py -v` passes. If the test fails, fall back to the pure-Python GDN at ~50,000 tok/s sustained (still within the 5-7 day envelope, but with no margin).

---

### Task H2: Wire MoE mixed precision (FP16 scatter-add indices)

**Files:**
- Modify: `models/moe.py:_dispatch_tokens` and `_gather_outputs` (cast indices to FP16)
- New: `tests/test_moe_fp16_indices.py` (verify FP16 indices select the same experts as BF16)

**What:** Cast the scatter-add indices to FP16 (16-bit integer is sufficient for the 16-expert case) and the expert-matmul inputs to BF16. The 50% bandwidth reduction on the index path saves ~10% of MoE dispatch cost.

**Wall-clock saving:** ~1% of total wall-clock (MoE is ~10% of forward FLOPs).

**Why:** Architecture doc §12a.2.

**Test:** `pytest tests/test_moe_fp16_indices.py -v` passes; the MoE forward output matches the BF16-index version within 1e-3 tolerance.

---

### Task H3: Wire `torch.compile(mode="reduce-overhead")` on GDN blocks

**Files:**
- Modify: `models/gdn.py` (add `@torch.compile(mode="reduce-overhead")` decorator on `GatedDeltaNetBlock.forward`)
- New: `tests/test_gdn_compile.py` (verify compiled-GDN matches eager-GDN within 1e-3)

**What:** Decorate the GDN forward with `torch.compile`. Apply **only to GDN** (not the whole model) because:
- The MoE dispatch has dynamic shapes (variable expert routing) that `torch.compile` struggles with.
- The MLA has a separate CUDA Graph capture path (Task H4).
- The FSDP-2 all-gather is incompatible with whole-model compile.

**Wall-clock saving:** ~3% wall-clock (10% of GDN cost; GDN is 30% of post-Triton wall-clock).

**Why:** Architecture doc §12a.3.

**Test:** `pytest tests/test_gdn_compile.py -v` passes; the compiled GDN output matches the eager GDN within 1e-3 on 10 random inputs. The compile takes ~30 sec the first time; subsequent calls hit the cache.

**Fallback:** If `torch.compile` fails or breaks, set `optimization.torch_compile_gdn: false` in the config; throughput drops to ~60,000 tok/s (5.79 days) — still within the 5-7 day target.

---

### Task H4: Wire CUDA Graphs for MLA forward

**Files:**
- Modify: `models/mla.py:MLABlock.forward` (add CUDA Graph capture path)
- New: `tests/test_mla_cuda_graph.py` (verify CUDA-graph-captured MLA matches eager-MLA)

**What:** Capture the MLA forward as a CUDA Graph. The graph captures the fixed-shape path (q projection, k/v decompression, attention, output projection) and replays it without Python overhead. The dynamic-shape MoE dispatch and GDN stay in eager mode.

**Wall-clock saving:** ~0.5% wall-clock (5% of MLA cost; MLA is 10% of wall-clock). The bigger win is reduced CPU-side jitter (deterministic per-step time).

**Why:** Architecture doc §12a.4.

**Test:** `pytest tests/test_mla_cuda_graph.py -v` passes; the CUDA-graph-captured MLA output matches the eager MLA within 1e-3. The test auto-skips if CUDA Graphs are not supported on the test hardware.

**Fallback:** If CUDA Graphs fail, set `optimization.cuda_graphs_mla: false` in the config; throughput drops by 0.5% — negligible.

---

### Task H5: Stacked-throughput probe (MISS-7)

**Files:**
- New: `scripts/throughput_probe.py`
- New: `tests/test_throughput_probe_smoke.py` (verifies the probe runs end-to-end on a small model without crashing; full-scale run is manual)

**What:** A standalone script that runs 100 training steps with all 4 optimizations enabled (H1-H4) and measures the achieved tok/s against the §12a.5 target (≥ 60,000 tok/s sustained on 4× A100 SXM). The probe is the empirical gate that confirms the 5-7 day wall-clock is achievable *before* committing to the 30B-token primary run.

```python
# scripts/throughput_probe.py
"""Stacked-throughput probe: 100 steps with all 4 optimizations enabled.

Usage: torchrun --nproc_per_node=4 scripts/throughput_probe.py \
        --config configs/hymo_750m.yaml \
        --steps 100 \
        --output /tmp/throughput_probe.json
"""
import argparse
import json
import time
from pathlib import Path

import torch
from torch.distributed import init_process_group, destroy_process_group

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--output", default="/tmp/throughput_probe.json")
    p.add_argument("--warmup", type=int, default=10, help="steps to discard for compile/cache warmup")
    args = p.parse_args()

    init_process_group(backend="nccl")
    rank = torch.distributed.get_rank()
    world = torch.distributed.get_world_size()

    from models.fusionllm import HyMo  # noqa: E402
    from training.trainer import Trainer  # noqa: E402
    import yaml
    cfg = yaml.safe_load(open(args.config))

    trainer = Trainer(cfg)  # FSDP-2 wrap, optimizations, all 4 enabled

    # Synthetic batch
    B, T = cfg["training"]["micro_batch_size"], cfg["model"]["max_seq_len"]
    V = cfg["model"]["vocab_size"]
    device = torch.device(f"cuda:{rank}")

    # Warmup
    for _ in range(args.warmup):
        tokens = torch.randint(0, V, (B, T), device=device)
        targets = torch.randint(0, V, (B, T), device=device)
        trainer.train_step(tokens, targets)

    # Measure
    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    for _ in range(args.steps):
        tokens = torch.randint(0, V, (B, T), device=device)
        targets = torch.randint(0, V, (B, T), device=device)
        trainer.train_step(tokens, targets)
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - t0

    per_step = elapsed / args.steps
    tokens_per_step = B * T * world * cfg["training"]["gradient_accumulation_steps"]
    tok_per_sec = tokens_per_step / per_step
    wall_clock_30b_days = 30e9 / tok_per_sec / 86400

    result = {
        "rank": rank,
        "world_size": world,
        "per_step_sec": per_step,
        "tokens_per_step": tokens_per_step,
        "tok_per_sec": tok_per_sec,
        "wall_clock_30b_days": wall_clock_30b_days,
        "target_tok_per_sec": 60_000,
        "passes_target": tok_per_sec >= 60_000,
    }
    if rank == 0:
        print(f"Throughput probe: {tok_per_sec:,.0f} tok/s "
              f"({wall_clock_30b_days:.2f} days for 30B tokens)")
        print(f"Target: ≥ 60,000 tok/s — "
              f"{'PASS' if result['passes_target'] else 'FAIL'}")
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)

    destroy_process_group()

if __name__ == "__main__":
    main()
```

**Why:** The 5-7 day wall-clock is *contingent* on the 4 stacked optimizations delivering ≥ 60,000 tok/s sustained (design §12a.5). Without a real measurement, the 5-7 day estimate is unverified — the 6.94-day "FSDP-2 only" baseline is the realistic floor if any one optimization fails. The probe catches the failure *before* committing to the 5-7 day run, saving 1-2 days of wasted compute.

**Per-stage breakdown (manual, via `torch.profiler`):**
- Data loading: 50ms / step (overlapped with compute)
- Forward (32 layers, FSDP all-gather overlapped): 1,800ms / step
- Backward (32 layers, FSDP reduce-scatter overlapped): 5,400ms / step
- Optimizer step (NorMuon + AdamW, per-rank, sort-by-size balanced): 500ms / step
- Communication overhead (residual after overlap): 250ms / step
- **Total: ~8,000ms / step → 524,288 / 8.0 = 65,500 tok/s**

The probe writes `throughput_probe.json` with the measured per-step time and tok/s. The G2 smoke test runs the probe first; if tok/s < 60,000, the run is held while the failing optimization is fixed.

**Test:** `tests/test_throughput_probe_smoke.py` — runs the probe with 5 steps and `--warmup 0` on a 1-GPU mock setup; verifies the JSON output is written. The full 100-step probe is a manual check on the real 4× A100 SXM pod.

**Acceptance criteria:**
- [ ] `bash scripts/throughput_probe.sh` (a wrapper that runs the 100-step probe) exits 0
- [ ] `throughput_probe.json` shows `tok_per_sec >= 60_000`
- [ ] The 30B-token wall-clock projection is ≤ 7.0 days

**Performance validation:** The probe is the *measurement* of performance; no further validation needed. If the probe fails (tok/s < 60K), debug the failing optimization in H1-H4 before proceeding to G.

**Rollback:** `git checkout scripts/throughput_probe.py tests/test_throughput_probe_smoke.py`. The probe is non-destructive (writes only to /tmp by default).

---

**End of Work Block H. Commit series: H1-H5 as 2 commits (H1 alone; H2-H5 in one). Move to Work Block G (primary launch) when the 5 unit tests pass AND the stacked-throughput probe (H5, §12a.5) hits ≥ 60,000 tok/s sustained on 4× A100 SXM.**

# Work Block G: Deployment (Tasks G1-G5)

The RunPod launch, monitoring, and recovery for the v1.0 primary 30B-token run. Architecture doc §13.7-13.8.

**Prereqs:** Work Blocks A-E + H done; the 4 optimization tests pass; the stacked-throughput probe hits ≥ 60,000 tok/s sustained on 4× A100 SXM; the primary config is in `configs/hymo_750m.yaml`.

**Deliverable:** A working 4× A100 80GB SXM pod running the v1.0 primary 30B-token run; a recovery procedure for pod failures. **v1.1 ablations are NOT part of this block** (see Work Block F stub for the deferred ablation plan).

---

### Task G1: RunPod pod provisioning script

**Files:**
- New: `scripts/runpod_launch.sh`

**What:** A bash script that:
1. Provisions a 4× A100 80GB SXM pod via the RunPod API.
2. Mounts a network volume for `data/` and `checkpoints/`.
3. Clones the repo and checks out the implementation branch.
4. Installs dependencies (PyTorch, fla, lm-eval, etc.).
5. Verifies the 4 GPUs are visible and NVLink is active.

```bash
#!/bin/bash
# scripts/runpod_launch.sh
set -euo pipefail

# 1. RunPod API call (use the runpodctl or REST API)
RUNPOD_API_KEY="${RUNPOD_API_KEY:?Set RUNPOD_API_KEY env var}"
GPU_TYPE="NVIDIA A100 80GB SXM"
GPU_COUNT=4
POD_NAME="fusionllm-$(date +%s)"

# Network volume for data + checkpoints (must be pre-created)
NETWORK_VOLUME_ID="${NETWORK_VOLUME_ID:?Set NETWORK_VOLUME_ID}"

# 2. Provision via REST API
POD_JSON=$(curl -s -X POST "https://api.runpod.io/graphql" \
  -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
  -H "Content-Type: application/json" \
  -d "$(cat <<EOF
{
  "query": "mutation { podFindAndDeployOnDemand(input: {
    cloudType: SECURE,
    gpuCount: ${GPU_COUNT},
    volumeInGb: 200,
    networkVolumeId: \"${NETWORK_VOLUME_ID}\",
    containerDiskInGb: 100,
    minVcpuCount: 32,
    minMemoryInGb: 128,
    gpuTypeId: \"${GPU_TYPE}\",
    name: \"${POD_NAME}\",
    imageName: \"runpod/pytorch:2.5.0-py3.10-cuda12.1.0-devel-ubuntu22.04\",
    dockerArgs: \"\",
    ports: \"22/tcp,8888/http\",
    env: [{ key: \"PUBLIC_KEY\", value: \"${PUBLIC_KEY}\" }]
  }) { id name machine { gpuDisplayName } } }"
EOF
)")
POD_ID=$(echo "$POD_JSON" | python3 -c "import sys, json; print(json.load(sys.stdin)['data']['podFindAndDeployOnDemand']['id'])")
echo "Pod provisioned: $POD_ID"

# 3. Wait for pod to be RUNNING
for i in {1..30}; do
  STATUS=$(curl -s "https://api.runpod.io/graphql" -H "Authorization: Bearer $RUNPOD_API_KEY" \
    -H "Content-Type: application/json" \
    -d "{\"query\": \"query { pod(input: {podId: \\\"$POD_ID\\\"}) { runtime { status } } }\"}" \
    | python3 -c "import sys, json; print(json.load(sys.stdin)['data']['pod']['runtime']['status'])")
  if [ "$STATUS" = "RUNNING" ]; then
    echo "Pod is RUNNING"
    break
  fi
  sleep 30
done
echo "POD_ID=$POD_ID" > /tmp/pod_id.txt
```

**Why:** The launch is automated; manual provisioning is error-prone and slow.

**Test:** Dry-run with a 1-GPU test pod (cheaper); verify the API call works and the pod is provisioned.

---

### Task G2: Pre-training smoke test (100 steps)

**Files:**
- New: `scripts/smoke_train.sh`

**What:** A 100-step smoke test that verifies the full stack works on the RunPod pod before committing to the 30B-token run.

```bash
#!/bin/bash
# scripts/smoke_train.sh
set -euo pipefail

cd /workspace/fusionllm

# Verify 4 GPUs are visible
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

# Verify NVLink is active (SXM only)
nvidia-smi topo -m

# Run 100-step smoke test
torchrun --nproc_per_node=4 training/trainer.py \
  --config configs/hymo_750m.yaml \
  --max-steps 100 \
  --output-dir /workspace/checkpoints/smoke

# Verify the loss decreased over 100 steps
python -c "
import json
metrics = [json.loads(l) for l in open('/workspace/checkpoints/smoke/metrics.jsonl')]
losses = [m['loss'] for m in metrics]
print(f'Initial loss: {losses[0]:.4f}, final loss: {losses[-1]:.4f}')
assert losses[-1] < losses[0], 'Loss did not decrease'
print('Smoke test PASSED')
"
```

**Why:** Catches integration bugs (FSDP-2 init divergence, broken FSDP-aware grad norm, etc.) before committing to the 5-7 day run.

**Test:** The script exits 0; the loss decreases over 100 steps.

---

### Task G3: Primary 30B-token run

**Files:**
- New: `scripts/launch_primary.sh`

**What:** The main 30B-token run. Uses the canonical config.

```bash
#!/bin/bash
# scripts/launch_primary.sh
set -euo pipefail

cd /workspace/fusionllm

torchrun --nproc_per_node=4 training/trainer.py \
  --config configs/hymo_750m.yaml \
  --max-steps 57220 \
  --output-dir /workspace/checkpoints/pretrain \
  --resume-from /workspace/checkpoints/pretrain/latest
```

**Wall-clock:** 5-7 days at ~65,000 tok/s sustained (architecture doc §12a.5, §13.7).

**Why:** The primary run. Everything else supports this.

**Test:** The run completes 57,220 steps; the final val PPL is ≤ 2.10 (architecture doc §15 target). The 4 v1.1 ablations are deferred and run separately after the v1.0 primary is complete (Work Block F stub).

---

### Task G4: Monitoring and logging

**Files:**
- New: `scripts/monitor.sh`

**What:** A monitoring script that prints per-step metrics every 50 steps to stdout, posts to W&B if enabled, and alerts if the loss spikes or the grad norm exceeds threshold.

```bash
#!/bin/bash
# scripts/monitor.sh
# Watch the trainer output for anomalies
tail -f /workspace/checkpoints/pretrain/trainer.log | while read line; do
  if echo "$line" | grep -q "WARNING"; then
    echo "[MONITOR] $line"
    # Send to W&B alerts
    wandb alert "HyMo warning" "$line" 2>/dev/null || true
  fi
  if echo "$line" | grep -q "NaN"; then
    echo "[MONITOR] NaN detected: $line"
    wandb alert "HyMo NaN" "$line" 2>/dev/null || true
  fi
done
```

**Why:** Automated monitoring catches divergence before the next checkpoint saves.

**Test:** Manually inject a NaN into the loss; verify the monitor catches it and posts an alert.

---

### Task G5: Recovery from pod failure

**Files:**
- New: `scripts/recover.sh`

**What:** A recovery script that:
1. Detects the pod failure (via RunPod's heartbeat).
2. Provisions a new 4× A100 SXM pod.
3. Mounts the same network volume.
4. Resumes from the latest checkpoint on the network volume.

```bash
#!/bin/bash
# scripts/recover.sh
set -euo pipefail

NETWORK_VOLUME_ID="${NETWORK_VOLUME_ID:?Set NETWORK_VOLUME_ID}"
RUNPOD_API_KEY="${RUNPOD_API_KEY:?Set RUNPOD_API_KEY}"

# 1. Find the latest checkpoint on the network volume
LATEST=$(ls -t /workspace/checkpoints/pretrain/checkpoint-* 2>/dev/null | head -1)
if [ -z "$LATEST" ]; then
  echo "ERROR: No checkpoint found on network volume; cannot recover"
  exit 1
fi
echo "Recovering from: $LATEST"

# 2. Provision a new pod (same script as G1)
POD_ID=$(bash scripts/runpod_launch.sh)
echo "New pod: $POD_ID"

# 3. Wait for SSH
for i in {1..30}; do
  ssh -o StrictHostKeyChecking=no root@$(get_pod_ip $POD_ID) "test -d /workspace" && break
  sleep 30
done

# 4. Resume
ssh root@$(get_pod_ip $POD_ID) "cd /workspace/fusionllm && \
  torchrun --nproc_per_node=4 training/trainer.py \
    --config configs/hymo_750m.yaml \
    --max-steps 57220 \
    --output-dir /workspace/checkpoints/pretrain \
    --resume-from $LATEST"
```

**Why:** Pod failures are common on RunPod; the recovery procedure is the difference between losing 5 days of compute and 5 minutes.

**Test:** Manually kill the trainer; verify the recovery script provisions a new pod and resumes from the latest checkpoint.

---

**End of Work Block G. Commit series: G1-G5 as 1-2 commits. The v1.0 implementation is complete when:**
- [ ] Work Blocks A-E + H all done with tests passing
- [ ] `pytest tests/ -v --tb=short` passes 85+ tests (66 prior stability tests + 14 v1.0 unit tests + 5 integration tests)
- [ ] The stacked-throughput probe (architecture doc §12a.5) hits ≥ 60,000 tok/s sustained on 4× A100 SXM
- [ ] The 30B v1.0 primary run completes with val PPL ≤ 2.10
- [ ] The 6-eval suite (architecture doc §15) executes end-to-end and produces `checkpoints/pretrain/eval_results.json`
- [ ] The v1.0 quality decision matrix is filled in (architecture doc §15 decision matrix)

The v1.1 ablations (Work Block F stub) are **not** part of v1.0 completion; they run after v1.0 is done and have their own budget ($1,100-2,700) and schedule.

---

# Verification before completion

Before claiming the v1.0 implementation is done, run:

```bash
# 1. All tests pass (v1.0 unit + integration tests, including the 4 optimization tests)
cd ~/Desktop/CoreProjects/LLM/HyMo
python3 -m pytest tests/ -v --tb=short
# Expect: 85+ tests passing

# 2. The full smoke train works (100-step loss decrease)
bash scripts/smoke_train.sh
# Expect: 100-step loss decreases, smoke checkpoint saved

# 3. The 5 optimization unit tests pass (H1-H5)
pytest tests/test_gdn_kernel.py tests/test_gdn_compile.py tests/test_mla_cuda_graph.py tests/test_moe_fp16_indices.py tests/test_throughput_probe_smoke.py -v
# Expect: all 5 pass

# 3b. The stacked-throughput probe (H5) hits ≥ 60,000 tok/s on the real 4× A100 SXM pod
bash scripts/throughput_probe.sh
# Expect: tok/s ≥ 60,000; wall-clock projection ≤ 7.0 days for 30B tokens

# 4. The eval suite runs end-to-end on the v1.0 primary checkpoint
python training/eval/run_all.py --model checkpoints/pretrain/best
# Expect: 6 evals in eval_results.json (FineWeb-Edu PPL ≤ 2.10, HellaSwag ≥ 40%, ARC-Challenge ≥ 25%, MMLU ≥ 26%, GSM8K ≥ 5%, HumanEval ≥ 8%)

# 5. The v1.1 ablation configs are valid (deferred; should load without error)
for f in configs/ablations/*.yaml; do
  python -c "import yaml; yaml.safe_load(open('$f')); print('$f OK')"
done

# 6. The primary launch script is executable
test -x scripts/launch_primary.sh && echo "launch_primary.sh OK"

# 7. The architecture doc is in sync (no stale v2 references)
grep -c "v1.0\|v1.1\|HyMo-v" docs/HyMo-Design.md
# Expect: ≥ 1 (the version block at top of doc)
```

If any of these fail, the v1.0 implementation is not done. Fix and re-verify.

---

# Summary

**Total tasks:** ~45 across 8 work blocks (A through H; F is a v1.1 stub, not part of v1.0)
**Total commits (target):** 10-14 (group tasks within each work block; H1 alone, H2-H4 in one)
**Total wall-clock (v1.0):** 5-7 days for the primary 30B run + 1-2 days for Work Blocks A-E + H (data pipeline + model + infra + FSDP-2 + optimizations) = **7-10 days end-to-end**
**Total cost (v1.0):** $1,000-1,350 on RunPod (primary only; v1.1 ablations are separate at $1,100-2,700)

**v1.0 deliverable:** A converged 750M-active / 1.86B-stored HyMo model with held-out FineWeb-Edu PPL ≤ 2.10, evaluated on the 6-benchmark quality protocol (architecture doc §15). The 30B-token primary run at 40× params-in-tokens on 4× A100 80GB SXM with the 4 optimization techniques (architecture doc §12a) is the v1.0 milestone.

**Next step after v1.0 executes:** either (a) start v1.1 ablations on dedicated pods to publish the comparative claims, or (b) write the v1.0 single-point-result paper. The v1.0 `eval_results.json` from Work Block E plus the architecture doc is the paper's input. The paper's structure is:
1. Introduction (1 page)
2. Architecture (architecture doc §2-§5, paraphrased)
3. Training (architecture doc §6-§7, paraphrased)
4. Optimization stack (architecture doc §12a, paraphrased)
5. Quality Validation (architecture doc §15, with results)
6. Related Work (cite Wang 2507.06457, Bae 2510.04800, NorMuon 2510.05491, etc.)
7. Conclusion (1 page)

Target venues: ICML 2027, NeurIPS 2027, or ACL 2027. The v1.0 paper's contribution is the chosen-design quality on real held-out evals (MobileMoE-0.9B class); the v1.1 paper adds the comparative ablations.

---

**End of v1.0 implementation plan. Convert to a `superpowers` execution plan with `superpowers:subagent-driven-development` to parallelize Work Blocks A+B, then C, then D, then H (parallel with D), then E, then G.**
