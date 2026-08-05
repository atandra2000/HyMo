# HyMo — Documentation Index

> Reading-order guide for the HyMo documentation (canonical layout: `concepts/` for theory and architecture, `references/` for API + config, `guides/` for how-tos, `training.md` for the training pipeline).

**Audience:** the author (interview preparation + self-mastery), with a motivated reader who has linear algebra + calculus but no transformer background as the secondary audience. Docs are **concept-first, code-anchored, formula-driven**; no hand-waving, no fabricated paths.

**Source of truth:** the code in `src/hymo/` at the HEAD commit. When the code changes a fact a doc cites, the doc is updated in the same commit (no stale doc commits).

---

## Layout

| Path | Content |
|---|---|
| [`README.md`](../README.md) | Project overview, architecture table, quickstart pointer |
| [`concepts/model-architecture.md`](concepts/model-architecture.md) | Line-by-line code walkthrough of `src/hymo/models/` (model, GDN, MLA, MoE, MTP, RoPE) |
| [`concepts/gdn-and-mla.md`](concepts/gdn-and-mla.md) | Mechanism deep-dives: GDN, MLA, MoE, MTP, hybrid-stack thesis |
| [`concepts/optimization.md`](concepts/optimization.md) | NorMuon/AdamW, WSD, FSDP-2, initialization status, optimization flags |
| [`concepts/kernels.md`](concepts/kernels.md) | GPU execution model + the hand-written Triton GDN kernel |
| [`concepts/design.md`](concepts/design.md) | The full v1.0 architecture & design document |
| [`references/config.md`](references/config.md) | The typed-config system: every sub-config, field table, validation rule |
| [`references/api.md`](references/api.md) | Model + trainer public API surface |
| [`guides/quickstart.md`](guides/quickstart.md) | Install, first forward pass, tests and gates |
| [`training.md`](training.md) | Data pipeline, trainer loop, checkpointing, in-training validation, eval scope |

## Reading orders

### 1. Interview prep (2–3 hours)

1. [`README.md`](../README.md) — the 30-second elevator pitch.
2. [`concepts/model-architecture.md`](concepts/model-architecture.md) — the full model walkthrough (attention lineage, MLA, GDN, MoE, MTP, RoPE, μP status).
3. [`concepts/gdn-and-mla.md`](concepts/gdn-and-mla.md) — GDN/MLA/MoE/MTP mechanism details.
4. [`concepts/kernels.md`](concepts/kernels.md) — the Triton GDN kernel and autograd integration.
5. [`concepts/optimization.md`](concepts/optimization.md) — NorMuon, WSD, FSDP-2.
6. [`references/config.md`](references/config.md) — so you can read any `configs/hymo_750m.yaml` field in isolation.

### 2. "From scratch" (1–2 days, full read)

1. [`concepts/model-architecture.md`](concepts/model-architecture.md) — model + attention + position encoding.
2. [`concepts/gdn-and-mla.md`](concepts/gdn-and-mla.md) — the mechanism tiers.
3. [`concepts/optimization.md`](concepts/optimization.md) — the optimization quartet (optimizer, scheduler, FSDP, init).
4. [`concepts/kernels.md`](concepts/kernels.md) — the hand-written kernel.
5. [`training.md`](training.md) — data → trainer → checkpoint → validation.
6. [`concepts/design.md`](concepts/design.md) — the design rationale end to end.

### 3. Engineering (touch the code)

1. [`references/config.md`](references/config.md) — how a YAML becomes a `HyMoConfig`.
2. [`concepts/model-architecture.md`](concepts/model-architecture.md) — the model layer, block-by-block.
3. [`training.md`](training.md) — `Trainer`, the two optimizers, the WSD scheduler, DCP.
4. [`references/api.md`](references/api.md) — the API surface at a glance.

## Conventions

- **`file.py:Symbol` anchors** in every doc are verified against the code at
  the HEAD commit by `tests/test_doc_refs.py` (symbols resolve via import; line-number anchors are not used).
- **Blockquotes** highlight plan-vs-implementation drift — if a design doc
  says one thing and the code does another, the blockquote says so.
- **No fabricated paths.** A doc that references `data/prepare_data.py`
  is a bug; it does not exist.
- **No `fla`.** The only sanctioned custom kernel is the hand-written
  Triton kernel in `src/hymo/models/gdn_triton.py`.

## History

- **2026-08-05 — canonical documentation layout.** `learning_docs/`
  (6 chapters) and the process docs (`PHASE_1_DELIVERY.md`, `HyMo-Roadmap.md`, `docs/superpowers/`) were removed. The corpus was consolidated into the canonical layout above. The roadmap's 2-line status: Phase 1–4 implementation shipped (with the 2026-08-04 cleanup trimming test-only `eval/`/`ablations/`/data-pipeline modules), the 30B-token pre-training run remains the v1.0 milestone.

## Test counts (live, not historical)

**226 tests collected (2026-08-05): 191 passed / 35 skipped.** Default `pytest` skips the GPU-gated tests (heavy model construction, CUDA-required, Triton-not-available); `pytest --run-heavy` runs all 226.

Re-run command: `cd /Users/atandrabharati/Desktop/CoreProjects/LLM/HyMo && python3 -m pytest -q --tb=no 2>&1 | tail -3`.
