# HyMo — Documentation Audit

> Status as of the Wave-1 docs upgrade. Every claim below was measured on
> this machine (macOS, CPU-only, no Triton/CUDA) at HEAD; "pending" items
> are named as pending, not estimated.

## Verification runs

| Check | Command | Result |
|---|---|---|
| Anchor resolution + coverage + links | `python3 scripts/check_docs.py --coverage --links` | PASS — 17 docs, 170 anchors; resolution PASS, coverage PASS (all public symbols in the 19 shipped modules cited), links PASS |
| pytest gate variant | `python3 tests/test_doc_refs.py --links` | PASS (silent on success) |
| Test suite | `python3 -m pytest --tb=no` | **203 passed / 35 skipped / 4 warnings in 2.24 s** (35 = heavy/CUDA/Triton-gated skips, by design) |
| Stale-count sweep | grep for `N passed` / `N collected` across `docs/`, `README.md`, `AGENTS.md`, `SKILLS.md` | clean after F1 fix |
| Anchor resolution + coverage + links (2026-09-21, concepts cluster) | `python3 scripts/check_docs.py --coverage --links` | PASS — 22 docs, 249 anchors; resolution PASS, coverage PASS (all public symbols in the 19 shipped modules cited), links PASS |
| pytest gate variant (2026-09-21) | `python3 tests/test_doc_refs.py --links` | PASS (silent on success) |
| Test suite (2026-09-21) | `python3 -m pytest --tb=no -q` | **203 passed / 35 skipped / 4 warnings in 2.54 s** — unchanged from the 2026-08-20 run (docs-only wave) |

The 4 pytest warnings are deprecation notices from third-party deps (not
test failures); tracked, no action this wave.

## Findings

| ID | Severity | Finding | Disposition |
|---|---|---|---|
| F1 | doc-rot | `docs/README.md` claimed "226 tests collected: 191 passed / 35 skipped (2026-08-05)"; live run gives **238 collected / 203 passed**. Root `README.md` had been refreshed (commit `5497e61`) but the docs nav page was missed. | **Fixed** — count refreshed to the measured 238/203/35. |
| F2 | gap | No `scripts/check_docs.py` gate, and 18 public symbols in `core/`, `data/`, `training/` were uncited anywhere (all five sub-configs, the semantic newtypes, `CheckpointState`, `ParameterPartition`, `ValMetrics`, `prepare_validation.main`, `derive_config`, `save_config`, `validate_full_config`). | **Fixed** — gate ported from the DiffusionGemma-Lite template (HyMo dialect: `src/hymo/...:Symbol` anchors, line ranges resolved against EOF, `JIT_SYMBOLS` skip); gaps closed via a symbol index in [`references/config.md`](references/config.md) and anchors in [`training.md`](training.md). Coverage now PASS. |
| F3 | accepted | [`references/config.md`](references/config.md) uses line-number headings ("`(line 23)`") alongside symbol citations. HyMo's citation contract permits line anchors and the gate verifies each against EOF — but they rot after refactors. | **Accepted** for this corpus; new prose sticks to symbols. |
| F4 | pending | Headline numbers (434 M active / 1.13 B stored, 30 B-token pretrain) come from `src/hymo/models/model.py:HyMo.num_parameters` behind `@pytest.mark.heavy` and the un-started pretraining run. No new measurements were made in this audit; none are claimed. | **Accepted** — heavy-gated verification documented, run remains the v1.0 milestone. |
| F5 (2026-09-21) | drift | FSDP wiring: `src/hymo/training/fsdp.py:wrap_model_with_fsdp` and `src/hymo/training/fsdp.py:fsdp_auto_wrap_policy` exist and are unit-tested, but nothing in `src/hymo/training/trainer.py` calls the wrapper; `src/hymo/core/config.py:TrainingConfig.fsdp` is declared but unread in `src/` (only `fsdp_mixed_precision` is consumed); the "FSDP-2" label describes design intent while the shipped wrapper uses the `FullyShardedDataParallel` API. Older optimization prose implied the config flag gates an active code path. | **Documented** — honest accounting in [`concepts/fsdp2.md`](concepts/fsdp2.md) (§Status), dated correction appended to [`concepts/optimization.md`](concepts/optimization.md). Wiring the trainer call site is code work, out of docs scope. |
| F6 (2026-09-21) | doc-rot | [`concepts/gdn-and-mla.md`](concepts/gdn-and-mla.md) claimed the 24 GDN blocks each carry a `DenseFFN` (module removed in the 2026-08-04 cleanup; `src/hymo/models/gdn.py:GatedDeltaNetBlock.__init__` builds recurrence projections only) and described MTP heads as flat per-head projections, while `src/hymo/models/mtp.py:MultiTokenPrediction.forward` chains each head on the previous head's hidden state. | **Fixed** — dated Corrections section appended to [`concepts/gdn-and-mla.md`](concepts/gdn-and-mla.md); corrected stories in [`concepts/asymmetric-moe.md`](concepts/asymmetric-moe.md) and [`concepts/mtp.md`](concepts/mtp.md). |

## Codebase map (condensed)

| Package | Modules | Docs that cover them |
|---|---|---|
| `src/hymo/core/` | `config.py`, `config_validation.py`, `types.py` (PyTorch-free) | [`references/config.md`](references/config.md) |
| `src/hymo/models/` | `model.py`, `mla.py`, `gdn.py`, `gdn_triton.py`, `moe.py`, `mtp.py`, `rope.py` | [`concepts/model-architecture.md`](concepts/model-architecture.md), [`concepts/gdn-and-mla.md`](concepts/gdn-and-mla.md), [`concepts/kernels.md`](concepts/kernels.md), [`references/api.md`](references/api.md) |
| `src/hymo/training/` | `trainer.py`, `optimizer.py`, `partition.py`, `scheduler.py`, `fsdp.py`, `checkpoint.py`, `validation.py` | [`training.md`](training.md), [`concepts/optimization.md`](concepts/optimization.md) |
| `src/hymo/data/` | `tokenizer.py`, `prepare_validation.py` | [`training.md`](training.md) §Data Pipeline |
| `configs/` | `hymo_750m.yaml` (primary), `hymo_mixture.yaml` | [`references/config.md`](references/config.md) |

`src/hymo/{eval,ablations,registry,utils}/` are empty after the
2026-08-04 cleanup — see [`training.md`](training.md) §Evaluation scope
note; `eval/`/`ablations/` content is Phase-4 design intent recorded in
[`concepts/design.md`](concepts/design.md).

## Modification plan (next changes touch docs here)

1. Any change to `src/hymo/core/config.py` fields → update [`references/config.md`](references/config.md) field tables and the glossary notation table in the same commit.
2. Any new public symbol in a shipped module → cite it once in the appropriate doc; `check_docs --coverage` fails otherwise.
3. Optimizer/scheduler changes → [`concepts/optimization.md`](concepts/optimization.md) + [`training.md`](training.md) §3/§4 together (they share the mechanism/prose split).
4. Kernel work → [`concepts/kernels.md`](concepts/kernels.md) + the CPU-reference test contract; cite only `triton_gated_delta_rule` for GPU-side behavior.
5. Test-count claims → only with a fresh `pytest` run pasted into the audit table above.

## Acceptance criteria

- [x] Every anchor in every doc resolves (resolution PASS).
- [x] Every public symbol in the 19 shipped modules is cited at least once (coverage PASS).
- [x] No broken intra-repo markdown links (links PASS).
- [x] `test_doc_refs.py` and `scripts/check_docs.py` agree on semantics and both pass.
- [x] Live test counts in docs match the measured run (203/35, F1 fixed).
- [x] Learning paths + glossary exist and are linked from the nav map.
- [ ] 30 B-token pretrain run (v1.0 milestone) — not started; out of docs scope.
