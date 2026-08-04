# HyMo Documentation Expansion — Design

**Date:** 2026-08-04
**Status:** Approved (audience, structure, scope, depth settled in brainstorming)

## 1. Purpose & Audience

HyMo is the flagship model of the CoreProjects portfolio. The documentation
must teach the architecture from first principles so the author can defend
every design decision and derive every formula on demand — in an interview,
or from memory.

**Audience:** the author (interview preparation + self-mastery). Docs are
concept-first, code-anchored, formula-driven. Secondary audience: a motivated
reader coming in with linear algebra + calculus and no transformer background.

**Success criteria**

1. Every architectural subsystem in `src/` is covered by at least one
   concept file (theory) and one code walkthrough.
2. Every claim that references a file path resolves to an actual file in the
   repo as of commit `af89c48` (HEAD at writing). No fabricated paths, no
   `fla` library, no deleted `registry/`.
3. The docs are internally consistent: the 4 optimization flags, the EMA
   gate-bias update, the Triton kernel (not `fla`), and the real test counts
   (316 tests collected as of 2026-08-04) are described correctly everywhere.
4. A reader can trace the *full* forward path — token → embed → 32 layers →
   norm → head → softcap → loss → backward → optimizer → scheduler — from the
   docs alone, without opening a source file.
5. Every concept file ends with an Interview Q&A section.

## 2. Current-State Diagnosis (what is wrong)

- **`fla` kernel fiction.** `learning_docs/4_Optimizations.md` and
  `docs/HyMo-Roadmap.md` repeatedly describe the `fla` library's
  `chunk_gated_delta_rule` as the primary kernel path. The repo contains a
  hand-written Triton kernel in `gdn_triton.py` (`triton_gated_delta_rule`,
  `TritonGDNFunction`), which `AGENTS.md` calls "the only currently-sanctioned
  kernel." The `fla` path does not exist in the code.
- **Fabricated paths.** `SKILLS.md` references `hymo.training.train` (no such
  module — only `trainer.py`) and `data/prepare_data.py` (does not exist; it
  is a pending Roadmap task). `README.md` Quick Start calls
  `model(x)` and prints `.shape`, but `HyMo.forward` returns
  `(logits, aux_losses)`.
- **Stale phase-1 artifacts.** `docs/PHASE_1_DELIVERY.md` documents
  `hymo.registry` (deleted in commit `72f7905`) and test counts
  `308/342 passed` (actual: **316 tests collected as of 2026-08-04**; 280 passed / 36 skipped on default `pytest`).
- **Coverage gaps.** Nothing covers `eval/` (harness, baselines, comparison,
  run_all), `ablations/`, `core/types.py`, `core/config_validation.py`, or
  `utils/`. The four optimization flags and the EMA gate-bias update — wired
  in `af89c48` — are described as *conceptual* in `learning_docs/4` §6.
- **Not "from scratch."** Existing walkthroughs assume prior knowledge of
  attention, recurrence, MoE, Muon, WSD, μP, FSDP-2, and Triton. There is no
  derivation layer. Depth is lopsided: `learning_docs/1` is 3,531 lines, but
  `learning_docs/4` (optimizations, arguably the hardest topic) is 533.

## 3. Proposed Structure — Two Tiers

### Tier 1: `docs/concepts/` — theory tier (new)

12 concept files, each following a **strict template**:

1. **Learning objectives** — what the reader can do after this file.
2. **Intuition** — plain-language motivation before any math.
3. **Math derivation** — formula-first, no skipped steps. State the general
   case, then specialize to HyMo's numbers.
4. **Implementation in HyMo** — precise `file:line` anchors into `src/`.
5. **Worked example** — real numbers from `configs/hymo_750m.yaml`
   (`dim=896`, `kv_lora_rank=128`, `gdn_d_state=32`, `gdn_chunk_size=64`,
   `n_routed_experts=16`, `moe_inter_dim=2304`, `inter_dim=2560`, …).
6. **Interview Q&A** — 3–6 questions a hiring manager would ask, with the
   answers the author should give.
7. **Cross-links** — to the matching walkthrough section and to sibling
   concepts.

Files and the code they anchor to:

| # | File | Core theory | Bridges to (walkthrough) |
|---|---|---|---|
| 01 | `attention.md` | MHA → GQA → MQA → MLA; low-rank KV compression; softmax attention complexity | 1_Model §4 |
| 02 | `linear-attention-gdn.md` | linear attention lineage; delta rule → gated delta rule; chunked recurrence; selective scan | 1_Model §5, §6 |
| 03 | `mixture-of-experts.md` | top-k routing; capacity; load balancing: aux-loss vs HyMo's EMA gate-bias | 1_Model §8 |
| 04 | `position-encoding.md` | RoPE math; partial-RoPE (25% of head_dim); NoPE hybrid | 1_Model §7 |
| 05 | `mtp.md` | multi-token prediction rationale; depth-2 loss weighting | 1_Model §9 |
| 06 | `mup-init.md` | why standard init breaks at scale; μP scaling rules | 1_Model §10 |
| 07 | `muon-optimizer.md` | Muon lineage; Newton–Schulz orthogonalization; cautious WD; FP32 masters | 3_Training §3 |
| 08 | `wsd-scheduler.md` | warmup–stable–decay; why it beats cosine for continued pretraining | 3_Training §4 |
| 09 | `fsdp2.md` | DDP → ZeRO → FSDP-2; full sharding; mixed precision | 3_Training §5 |
| 10 | `triton-kernels.md` | GPU execution model; fused selective scan; autograd Function + recompute backward | 1_Model §6, 4_Opt |
| 11 | `hybrid-architectures.md` | Jamba/Zamba/StripedHyena landscape; the 3:1 and MoE-on-attention theses | 1_Model §1, Design §8 |
| 12 | `tokenization-data.md` | BPE; byte-level fallback; FineWeb-Edu filtering; 40× params-in-tokens rule | 2_Data |

### Tier 2: `learning_docs/` — code walkthroughs (edit 4, add 2)

- **`1_Model_Architecture.md`** — keep the ~3,531-line depth (it is the
  strongest doc). Refresh stale bits; add per-section links to concepts;
  verify every `file:line` anchor against current code.
- **`4_Optimizations.md`** — **rewrite** (533 → ~1,500 lines). Remove all
  `fla` references. Document the real Triton kernel, the 4 optimization flags
  (`fused_gdn`, `moe_mixed_precision`, `torch_compile_gdn`,
  `cuda_graphs_mla`), the real EMA gate-bias, and real memory/throughput
  numbers.
- **`3_Training_Pipeline.md`** — **rewrite** (908 → ~1,800 lines) to match
  actual `trainer.py`: wandb init, `_thread_optimization_flags`,
  `train_step` grad-accum `is_update` semantics, NaN-skip, EMA gate-bias
  update, `_current_lr_*`, the real `NorMuon`/`CautiousAdamW` step logic
  (Newton–Schulz, cautious WD, FP32 masters), the real scheduler
  (`get_factor`, 3 decay kinds).
- **`2_Data_Pipeline.md`** — expand (732 → ~1,200 lines): the real 10 source
  loaders in `sources.py`, real `tokenizer.py` (`ExtendedTokenizer`), real
  `sharding.py` (`ShardWriter`, `ShardDataset`, `DataLoaderBuilder`).
- **`5_Evaluation_and_Ablations.md`** — **new** (~1,000 lines):
  `eval/harness.py` (`run_harness_eval`, `EvalResult`),
  `eval/baselines.py` (`BASELINES`, `TASK_TO_METRIC`),
  `eval/comparison.py`, `eval/run_all.py` (`EVAL_SUITE`, `run_all`),
  `ablations/__init__.py` (`AblationSpec`, `ABLATION_FAMILIES`,
  `build_ablation_config`), `core/config_validation.py`.
- **`6_Config_System.md`** — **new** (~700 lines): `config.py` in depth —
  the 5 sub-config dataclasses, `__post_init__` validation, computed
  properties (`mla_positions`, `nope_hybrid_gdn_positions`,
  `per_step_tokens`, …), `load_config`/`save_config`/`derive_config`,
  `_filter` type coercion, `core/types.py` (`Step`).

### Tier 3: Repair existing docs

- **`README.md`** — fix the Quick Start (model returns a tuple); drop
  `registry/` from the project tree; update test counts; fix the "native
  PyTorch fallback on non-Linux" phrasing (Triton is Linux-only; the eager
  path is the reference fallback).
- **`SKILLS.md`** — replace fabricated paths (`hymo.training.train` →
  `Trainer`; `data/prepare_data.py` → mark as a pending Roadmap task).
- **`docs/HyMo-Roadmap.md`** — reconcile shipped blocks (B1–B7 shipped as
  hand-written Triton, not `fla`; C block complete) vs deferred; stamp status.
- **`docs/PHASE_1_DELIVERY.md`** — refresh: drop `registry/`, correct test
  counts, note it is superseded for current work.
- **`docs/HyMo-Design.md`** — annotate §12a as "as implemented" where the
  code diverges from the original plan; correct the `fla` mentions.
- **`docs/README.md`** — **new** documentation map: reading orders
  (interview / from-scratch / engineering), concept↔walkthrough
  cross-reference table, glossary.

## 4. Verification (how we know the docs match the code)

Every walkthrough and concept file's code claims are verified against the
actual source before the doc is considered done:

- **Path audit:** every `` `path/to/file.py` `` backtick reference must exist
  via `ls` / `find`. A doc that references a non-existent file is a bug.
- **Symbol audit:** every `` `Name` `` reference to a function/class must
  exist in the file it is attributed to (`grep -rn`).
- **Anchor audit:** `file:line` anchors are checked by `Read`/`Grep` at write
  time.
- **Fact audit:** config numbers (dim, ranks, d_state, chunk_size, expert
  counts, LR, steps) are taken from `configs/hymo_750m.yaml`, not from
  memory. Test counts quoted as of `af89c48`.
- **Freshness rule:** if the code changes a fact the docs cite, the doc is
  updated in the same commit (docs and code stay in lockstep — no stale
  documentation commits).

## 5. Output Size & File Plan

Estimated **~8,500–9,500 new/rewritten lines** across ~15 files, plus
surgical edits to 6 existing docs. New files:

```
docs/concepts/01-attention.md            ~700
docs/concepts/02-linear-attention-gdn.md ~800
docs/concepts/03-mixture-of-experts.md   ~700
docs/concepts/04-position-encoding.md    ~600
docs/concepts/05-mtp.md                  ~450
docs/concepts/06-mup-init.md             ~600
docs/concepts/07-muon-optimizer.md       ~750
docs/concepts/08-wsd-scheduler.md        ~500
docs/concepts/09-fsdp2.md                ~800
docs/concepts/10-triton-kernels.md       ~800
docs/concepts/11-hybrid-architectures.md ~650
docs/concepts/12-tokenization-data.md    ~600
learning_docs/5_Evaluation_and_Ablations.md  ~1000
learning_docs/6_Config_System.md             ~700
docs/README.md                               ~300
```

Rewrites: `learning_docs/4_Optimizations.md` (~1,500),
`learning_docs/3_Training_Pipeline.md` (~1,800),
`learning_docs/2_Data_Pipeline.md` (~1,200).

Edits: `README.md`, `SKILLS.md`, `docs/HyMo-Roadmap.md`,
`docs/PHASE_1_DELIVERY.md`, `docs/HyMo-Design.md`, `learning_docs/1_Model_Architecture.md`.

## 6. Excluded / Non-goals

- No new code, tests, or configs. This is a documentation-only change.
- No migration of `docs/HyMo-Design.md` / `docs/HyMo-Roadmap.md` into the
  new two-tier layout — they stay where they are, repaired in place.
- No API-reference autodoc generation. Walkthroughs are hand-written prose
  because the goal is *understanding*, not symbol listing.
- No obsidian-vault mirroring of `docs/` (outside the sync paths already
  configured). New `.md` files live under the repo only.
