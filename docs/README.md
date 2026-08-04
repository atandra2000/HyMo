# HyMo — Documentation Map

> Reading-order guide for the HyMo documentation. Pick the path that matches
> what you need; cross-references in every file use `file:line` anchors into
> `src/hymo/` so a reader can `Read` the source from any doc.

**Audience for every doc in this directory and `learning_docs/`:**
the author (interview preparation + self-mastery), with a motivated reader
who has linear algebra + calculus but no transformer background as the
secondary audience. Docs are **concept-first, code-anchored, formula-driven**;
no hand-waving, no fabricated paths.

**Source of truth:** the code in `src/hymo/` at the HEAD commit. When the
code changes a fact a doc cites, the doc is updated in the same commit
(no stale doc commits).

---

## Reading orders

### 1. Interview prep (2–3 hours)

This is the order to read if you need to defend every design decision from
memory. Each doc ends with an **Interview Q&A** section — work through those
last if you have time pressure.

1. [`README.md`](../README.md) — the 30-second elevator pitch.
2. [`concepts/01-attention.md`](concepts/01-attention.md) — MHA → MQA → MLA
   lineage, low-rank KV compression, complexity ladder.
3. [`concepts/02-linear-attention-gdn.md`](concepts/02-linear-attention-gdn.md) —
   linear-attention lineage, delta rule, gated delta rule, chunked recurrence.
4. [`concepts/03-mixture-of-experts.md`](concepts/03-mixture-of-experts.md) —
   DeepSeekMoE, top-2 routing, aux-loss-free EMA gate-bias.
5. [`concepts/10-triton-kernels.md`](concepts/10-triton-kernels.md) — GPU
   execution model, autograd `Function` + recompute, the HyMo GDN kernel.
6. [`concepts/06-mup-init.md`](concepts/06-mup-init.md) — μP scaling rules.
7. [`concepts/07-muon-optimizer.md`](concepts/07-muon-optimizer.md) — Newton–Schulz.
8. [`concepts/08-wsd-scheduler.md`](concepts/08-wsd-scheduler.md) — why WSD
   beats cosine for continued pretraining.
9. [`concepts/11-hybrid-architectures.md`](concepts/11-hybrid-architectures.md) —
   the 3:1 thesis; Jamba / Zamba / StripedHyena context.
10. [`learning_docs/6_Config_System.md`](../learning_docs/6_Config_System.md) —
    so you can read any `configs/hymo_750m.yaml` field in isolation.

### 2. "From scratch" (1–2 days, full read)

This is the order to write a tutorial from.

1. [`concepts/01-attention.md`](concepts/01-attention.md) — attention math.
2. [`concepts/02-linear-attention-gdn.md`](concepts/02-linear-attention-gdn.md) —
   linear attention.
3. [`concepts/03-mixture-of-experts.md`](concepts/03-mixture-of-experts.md) —
   MoE.
4. [`concepts/04-position-encoding.md`](concepts/04-position-encoding.md) —
   partial-RoPE + NoPE-hybrid.
5. [`concepts/05-mtp.md`](concepts/05-mtp.md) — multi-token prediction.
6. [`concepts/12-tokenization-data.md`](concepts/12-tokenization-data.md) —
   BPE + FineWeb-Edu.
7. [`concepts/06-mup-init.md`](concepts/06-mup-init.md), `07-muon-optimizer.md`,
   `08-wsd-scheduler.md`, `09-fsdp2.md` — the optimization quartet.
8. [`concepts/10-triton-kernels.md`](concepts/10-triton-kernels.md) — the
   hand-written kernel.
9. [`learning_docs/1_Model_Architecture.md`](../learning_docs/1_Model_Architecture.md) —
   line-by-line walkthrough of `src/hymo/models/`.
10. [`learning_docs/2_Data_Pipeline.md`](../learning_docs/2_Data_Pipeline.md) →
    [`learning_docs/3_Training_Pipeline.md`](../learning_docs/3_Training_Pipeline.md) →
    [`learning_docs/4_Optimizations.md`](../learning_docs/4_Optimizations.md) →
    [`learning_docs/5_Evaluation_and_Ablations.md`](../learning_docs/5_Evaluation_and_Ablations.md).

### 3. Engineering (touch the code)

1. [`learning_docs/6_Config_System.md`](../learning_docs/6_Config_System.md) —
   how a YAML becomes a `HyMoConfig`.
2. [`learning_docs/1_Model_Architecture.md`](../learning_docs/1_Model_Architecture.md) —
   the model layer, block-by-block.
3. [`learning_docs/3_Training_Pipeline.md`](../learning_docs/3_Training_Pipeline.md) —
   `Trainer`, the two optimizers, the WSD scheduler.
4. [`learning_docs/4_Optimizations.md`](../learning_docs/4_Optimizations.md) —
   the 4 optimization flags, the Triton kernel, throughput numbers.
5. [`learning_docs/5_Evaluation_and_Ablations.md`](../learning_docs/5_Evaluation_and_Ablations.md) —
   `run_all` + `build_ablation_config`.
6. [`learning_docs/2_Data_Pipeline.md`](../learning_docs/2_Data_Pipeline.md) —
   if you're touching data.

---

## Cross-reference: concepts ↔ walkthroughs

Each concept file says "Bridgesto (walkthrough)" — that points to a section
in a learning doc. The reverse mapping (walkthrough-section → concept):

| Walkthrough section | Concept |
|---|---|
| 1_Model §4 (MLA) | `01-attention.md` |
| 1_Model §5–6 (GDN, Triton) | `02-linear-attention-gdn.md`, `10-triton-kernels.md` |
| 1_Model §7 (RoPE) | `04-position-encoding.md` |
| 1_Model §8 (MoE) | `03-mixture-of-experts.md` |
| 1_Model §9 (MTP) | `05-mtp.md` |
| 1_Model §10 (μP) | `06-mup-init.md` |
| 1_Model §1 + Design §8 (hybrid) | `11-hybrid-architectures.md` |
| 2_Data (pipeline) | `12-tokenization-data.md` |
| 3_Training §3 (optimizer) | `07-muon-optimizer.md` |
| 3_Training §4 (scheduler) | `08-wsd-scheduler.md` |
| 3_Training §5 (FSDP-2) | `09-fsdp2.md` |
| 4_Opt (optimizations) | `10-triton-kernels.md` |

---

## Glossary

| Term | Definition | First explained in |
|---|---|---|
| **MLA** | Multi-Head Latent Attention — full attention with low-rank KV compression (DeepSeek-V2). | `concepts/01-attention.md` |
| **GDN** | Gated Delta Net — linear attention via the gated delta rule; recurrent state shape `(H, S)`. | `concepts/02-linear-attention-gdn.md` |
| **MQA-4** | Multi-Query Attention with 4 KV groups — the HyMo MLA uses 4 KV heads shared by `n_heads / n_kv_groups = 4` query heads. | `concepts/01-attention.md` |
| **Partial RoPE** | Rotary position embedding applied to the first 25% of `head_dim` (`qk_rope_head_dim = 32` of `head_dim = 128`). | `concepts/04-position-encoding.md` |
| **NoPE-hybrid** | Every GDN layer immediately after an MLA layer gets **no** position encoding (positions {3,7,11,15,19,23,27} for 8 MLA layers). v1.0 ships with this disabled (`nope_hybrid_gdn_enabled: false`); v1.1 ablation. | `concepts/04-position-encoding.md`, `concepts/11-hybrid-architectures.md` |
| **Aux-loss-free MoE** | MoE load balancing via EMA-smoothed bias updates on the gate (no auxiliary loss term in the loss). | `concepts/03-mixture-of-experts.md` |
| **MTP** | Multi-Token Prediction — auxiliary heads predicting tokens `t+2` and `t+3`, weighted `[0.3, 0.1]`. | `concepts/05-mtp.md` |
| **WSD** | Warmup–Stable–Decay LR schedule (2% warmup, 83% stable, 15% decay). Beats cosine for continued pretraining. | `concepts/08-wsd-scheduler.md` |
| **NorMuon** | Muon optimizer variant for non-attention 2D matrices (e.g. MoE expert weights excluded) with cautious weight decay and FP32 master weights. | `concepts/07-muon-optimizer.md` |
| **CautiousAdamW** | AdamW where the weight-decay mask zeros the update on coordinates where the gradient disagrees with the parameter sign (Liang et al. 2024). | `concepts/07-muon-optimizer.md` |
| **FSDP-2** | PyTorch's `fully_shard` API — full parameter + gradient + optimizer-state sharding. | `concepts/09-fsdp2.md` |
| **μP (mup_init)** | Maximal Update Parameterization — scale-invariant init that lets you tune hyperparameters on a tiny model and transfer to the full scale. | `concepts/06-mup-init.md` |
| **Logit softcap** | `15 * tanh(logits / 15)` — bounds logits for training stability (PaLM). | `learning_docs/1_Model_Architecture.md` §3 |
| **DCP** | Distributed Checkpoint — PyTorch's async-checkpoint API for sharded save/load. | `learning_docs/3_Training_Pipeline.md` §Checkpoint |
| **Tiny config** | The ~760 K-param config used by the default test suite; never build the production 1.86 B-param model in a non-`heavy` test. | [`../AGENTS.md`](../AGENTS.md) §Testing rules |

---

## Conventions

- **`file:line` anchors** in every doc are verified against the code at
  the HEAD commit. The `audit` pass (the final task in the docs expansion
  plan) re-verifies them.
- **Blockquotes** highlight plan-vs-implementation drift — if a design doc
  says one thing and the code does another, the blockquote says so.
- **Code excerpts** are minimal — they show the line under discussion, not
  the whole file. The `file:line` anchor is the canonical reference.
- **No fabricated paths.** A doc that references `data/prepare_data.py`
  is a bug; it does not exist.
- **No `fla`.** The only sanctioned custom kernel is the hand-written
  Triton kernel in `src/hymo/models/gdn_triton.py`. The `fla` library is
  commented out in `pyproject.toml:48`.

---

## Corpus size (measured 2026-08-04)

| Tree | Words |
|---|---|
| `docs/` total (incl. concepts) | 54,373 |
| — `docs/concepts/` (12 files) | 17,115 |
| — `docs/HyMo-Design.md` | 18,911 |
| — `docs/HyMo-Roadmap.md` | 12,653 |
| — `docs/PHASE_1_DELIVERY.md` | 4,620 |
| — `docs/README.md` | 1,074 |
| `learning_docs/` (6 chapters) | 33,894 |
| **Corpus total** (incl. top-level `README.md` / `AGENTS.md` / `SKILLS.md`) | **90,933** |

---

## Test counts (live, not historical)

**316 tests collected (2026-08-04): 280 passed / 36 skipped.** Default `pytest` skips the
GPU-gated tests (heavy model construction, CUDA-required, Triton-not-available);
`pytest --run-heavy` runs all 316.

| Run | Pass | Skip | Notes |
|---|---|---|---|
| `pytest` (default) | 280 | 36 | 13 heavy model-construction + 2 CUDA-required + 5 Triton-not-available + … |
| `pytest --run-heavy` | 316 | 0 | All collected tests run; needs the full 1.86B model on a GPU pod. |

Re-run command: `cd /Users/atandrabharati/Desktop/CoreProjects/LLM/HyMo && python3 -m pytest -q --tb=no 2>&1 | tail -3`.
