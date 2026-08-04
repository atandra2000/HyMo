# HyMo Evaluation & Ablations — Code Walkthrough

> **Prerequisite reading:** [`learning_docs/1_Model_Architecture.md`](1_Model_Architecture.md) for the
> model shape, [`learning_docs/3_Training_Pipeline.md`](3_Training_Pipeline.md) §Evaluate for the in-training
> val loop, [`learning_docs/6_Config_System.md`](6_Config_System.md) for `HyMoConfig` and `derive_config`.
>
> **Files covered:**
> - `src/hymo/eval/harness.py` — `lm-evaluation-harness` wrapper
> - `src/hymo/eval/baselines.py` — published baseline numbers + task→metric map
> - `src/hymo/eval/comparison.py` — comparison-table formatter
> - `src/hymo/eval/run_all.py` — the 6-eval suite runner
> - `src/hymo/ablations/__init__.py` — 4 ablation families + `build_ablation_config`
> - `src/hymo/core/config_validation.py` — cross-field config checks
>
> **Companion concepts:** none directly (this is the workflow layer above the
> concepts); the closest is [`docs/concepts/08-wsd-scheduler.md`](../docs/concepts/08-wsd-scheduler.md) for
> why ablations share the v1.0 schedule fractions.

---

## 1. Where eval sits in the pipeline

There are three evaluation surfaces in HyMo:

1. **In-training validation** — every `eval_interval` (default 2,000
   steps), `Trainer.evaluate` runs `compute_validation_loss` over a
   held-out FineWeb-Edu binary (`data/tokens/val.bin`). This is
   training-loop health monitoring, not a benchmark. See
   `learning_docs/3_Training_Pipeline.md` §Evaluate.

2. **The 6-eval suite** — at training end (or after a heavy eval
   checkpoint), `eval/run_all.py::run_all` runs the 6 standard
   benchmarks + the FineWeb-Edu PPL. This is the **primary
   quality signal** cited in the README's `≤ 2.10 PPL` target.

3. **Ablations** — `ablations/__init__.py::build_ablation_config`
   derives a frozen-dataclass config variant for one of the 4
   ablation families (A/B/C/D) so a side-experiment uses the same
   `Trainer`, the same data, and the same evaluation — only the
   ablation knob differs.

This document covers (2) and (3). The in-training val is covered in
`learning_docs/3_Training_Pipeline.md`.

---

## 2. Eval + ablations — scope note (2026-08-04)

The `src/hymo/eval/` package (`harness.py`, `baselines.py`,
`comparison.py`, `run_all.py`) and `src/hymo/ablations/` were **removed
in the cleanup**. They were consumed only by tests — the production path
(`tools/a100_smoke.py` → `load_config` → `build_hymo` → `Trainer`) never
imported them. The 6-task eval suite (HellaSwag, ARC, MMLU, GSM8K,
HumanEval, FineWeb-Edu PPL) and the ablation families (GDN/MLA/MoE/
optimizer config derivation via `derive_config`) remain design intent for
Phase 4, recorded in `docs/HyMo-Roadmap.md` and `docs/HyMo-Design.md` §8/§16.

The live evaluation surface in-repo:
- `src/hymo/training/validation.py` — `compute_validation_loss`,
  `get_val_batch`, `ValMetrics` (used by `Trainer` at `eval_interval`).
- `src/hymo/data/prepare_validation.py` — builds the held-out FineWeb-Edu
  validation binary.

## 4. `core/config_validation.py` — cross-field safety net

`__post_init__` checks fields **inside a sub-config**. Cross-field
checks live in `validate_full_config`:

```python
def validate_full_config(config: HyMoConfig) -> None:
    _validate_total_steps_consistency(config)
    _validate_layer_distribution(config.model)
    _validate_partial_rope_math(config.model)
    _validate_vram_budget(config.model, config.training)
```

| Helper | What it checks |
|---|---|
| `_validate_total_steps_consistency` (line 24) | `per_step_tokens > 0` (a guard, not a real cross-check today) |
| `_validate_layer_distribution` (line 32) | `n_layers % 4 == 0` AND `n_mla_layers == n_layers // 4` AND `mla_positions == {0,4,…,4·(n_mla-1)}` |
| `_validate_partial_rope_math` (line 51) | `0 ≤ qk_rope_head_dim / head_dim ≤ 1` |
| `_validate_vram_budget` (line 59) | Estimated peak VRAM ≤ 80 GB (per rank, A100 SXM 80 GB target) |

The VRAM estimator uses rough per-layer parameter
budgets (`gdn_per = 25_000_000`, `mla_attn_per = 5_800_000`,
`moe_active_per = 9_000_000`, `moe_stored_per = 145_000_000`) so the
validator runs **without** instantiating the actual model. It catches
gross over-runs (e.g. accidentally bumping `n_routed_experts` to 160);
it is not a substitute for measuring real GPU memory after a smoke
run.

Call it after `load_config`:

```python
from hymo.core.config import load_config
from hymo.core.config_validation import validate_full_config

config = load_config("configs/hymo_750m.yaml")
validate_full_config(config)
```

The ablation builder (`build_ablation_config`) does **not** call
`validate_full_config` itself — the caller is expected to validate
the result before passing to `build_hymo`.

---

## 5. Interview Q&A

**Q1. Why two eval paths (in-training val + 6-eval suite)?**

> A: Different jobs. In-training val answers "is the model still
> learning?" — cheap, every 2 k steps, just PPL on held-out
> FineWeb-Edu. The 6-eval suite answers "is the model *good*?" —
> expensive (lm-eval on 5 tasks + PPL), run at training end (or at a
> saved checkpoint for an interim answer). They share the same
> validation binary but different metrics and cadences.

**Q2. Why does `run_all` handle ImportError on `lm_eval` gracefully?**

> A: Optional dep. `lm_eval` is in `[project.optional-dependencies]
> train`, not the core install. A developer with only the dev
> extras can still call `run_all` and get NaN for the harness tasks
> plus a real PPL — useful for `validate_full_config` + a smoke
> forward pass without paying for the lm-eval install.

**Q3. Why is the 6-eval suite hard-coded as a tuple, not a list of
config-driven entries?**

> A: It's a **constant** — the 6 tasks and their fewshot counts are
> part of the design commitment (the ≤ 2.10 PPL target is for
> `fineweb_edu_ppl` in the suite; the headline accuracy claims are
> for `mmlu` and `gsm8k`). Making them mutable config would let
> cherry-picking happen silently. If a future v1.1 wants a 7th task,
> edit `EVAL_SUITE` and bump the test fixture count.

**Q4. Why do all 4 ablation families share the 7.5 B budget?**

> A: Comparability. With identical training budgets, identical
> data, identical schedule shape, the only difference between
> variants **is** the ablation knob. If family A used 5 B and
> family B used 10 B, you couldn't tell whether family A "lost"
> because MoE-everywhere is bad, or because 5 B tokens wasn't
> enough.

**Q5. Why is `_validate_vram_budget` approximate?**

> A: It can't afford to instantiate the model — that would defeat
> the purpose of a load-time validator. Instead it uses per-layer
> parameter approximations. This catches "I changed
> `n_routed_experts` from 16 to 160" but not "I changed `dim` from
> 896 to 1024". A real OOM check happens after the first
> micro-batch in the trainer; the load-time check is a coarse
> guardrail.

**Q6. Why is `Step(int(7_500_000_000 / cfg.training.per_step_tokens))` inside `build_ablation_config` and not a field on `AblationSpec`?**

> A: `AblationSpec.tokens = 7_500_000_000` is the **human-friendly**
> token budget. The `Step` value depends on `cfg.training.per_step_tokens`,
> which depends on the derived config — so the conversion happens
> *after* the per-family builder runs. Putting it in the spec would
> require re-deriving the spec every time the base config changed.

**Q7. Why do `_build_ablation_config_*` functions raise `ValueError`
on unknown variants rather than returning a sentinel?**

> A: Fail loud. The ablation runner is mechanical; it iterates
> over `AblationSpec.variants`. If a new variant gets added to
> `AblationSpec.variants` but the builder's `if` chain doesn't
> handle it, the runner crashes with a clear message instead of
> silently running the v1.0 default.

---

## 6. Cross-links

- Walkthrough: `learning_docs/1_Model_Architecture.md` §3 (model),
  `learning_docs/3_Training_Pipeline.md` §Evaluate.
- Concepts: `docs/concepts/08-wsd-scheduler.md` (schedule fractions
  used by `build_ablation_config`), `docs/concepts/06-mup-init.md`
  (μP knobs the ablations hold constant).
- Tests: `tests/unit/test_eval.py` for harness + baselines,
  `tests/unit/test_ablations.py` for the four families.
- Validation: `src/hymo/core/config_validation.py` for the
  cross-field checks.
