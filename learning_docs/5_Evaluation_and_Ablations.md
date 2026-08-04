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

## 2. The eval package — `src/hymo/eval/`

### 2.1 `harness.py` — `run_harness_eval` (line 28)

A thin wrapper over the `lm-evaluation-harness` library, adapted for a
HyMo model that is **not** an HF `PreTrainedModel` (so it uses
`HFLM(pretrained=model, ...)` to expose HyMo through the harness's
HF-shaped API):

```python
def run_harness_eval(
    model: Any,
    tokenizer: Any,
    tasks: list[str],
    *,
    num_fewshot: int = 0,
    batch_size: int = 4,
) -> dict[str, EvalResult]:
    import lm_eval
    from lm_eval.models.huggingface import HFLM

    lm = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        batch_size=batch_size,
        device=next(model.parameters()).device,
    )
    results = lm_eval.simple_evaluate(
        model=lm, tasks=tasks,
        num_fewshot=num_fewshot, batch_size=batch_size,
    )
    ...
```

`EvalResult` (line 14) is a frozen-shaped dataclass:

```python
@dataclass
class EvalResult:
    task: str
    metric: str
    value: float
    stderr: float | None = None
```

The function looks up the right metric key per task via
`TASK_TO_METRIC` (from `baselines.py`); if the task isn't in the map,
it falls back to `next(iter(task_data))` (the first metric the
harness returns).

### 2.2 `baselines.py` — published numbers

Three sibling models from 1B-class published results, plus the task → metric mapping:

```python
BASELINES: dict[str, dict[str, float]] = {
    "pythia-1b": { "fineweb_edu_ppl": 2.45, "hellaswag": 0.36, ... },
    "mobile_moe_0.9b": { "fineweb_edu_ppl": 2.10, ... },
    "smollm2_1.7b": { "fineweb_edu_ppl": 2.20, ... },
}

TASK_TO_METRIC: dict[str, str] = {
    "hellaswag": "acc_norm,none",
    "arc_challenge": "acc_norm,none",
    "mmlu": "acc,none",
    "gsm8k": "exact_match,flexible-extract",
    "humaneval": "pass@1",
    "fineweb_edu_ppl": "ppl",
}

BASELINE_DISPLAY_NAMES = { ... }
```

The 2.10 PPL on `mobile_moe_0.9b` is the **anchoring baseline** — HyMo's
README quality target is "on par with MobileMoE-0.9B class" which
means ≤ 2.10 on FineWeb-Edu held-out.

### 2.3 `run_all.py` — the 6-eval suite runner

#### The suite — `EVAL_SUITE` (line 17)

```python
EVAL_SUITE: tuple[tuple[str, int], ...] = (
    ("hellaswag",       0),   # 0-shot
    ("arc_challenge",   0),   # 0-shot
    ("mmlu",            5),   # 5-shot
    ("gsm8k",           8),   # 8-shot (chain-of-thought)
    ("humaneval",       0),   # 0-shot
    ("fineweb_edu_ppl", 0),   # perplexity, not an LM-eval task
)
```

The first five go through `lm-evaluation-harness`; the sixth is computed
internally by `compute_validation_loss` over `data/tokens/val.bin`.

#### `run_all(...)` (line 44)

```python
def run_all(
    model: nn.Module,
    tokenizer: Any,
    output_path: str | Path = "checkpoints/pretrain/eval/eval_results.json",
    *,
    batch_size: int = 4,
    val_bin_path: str | Path = "data/tokens/val.bin",
    seq_len: int = 4_096,
    vocab_size: int = 64_256,
) -> EvalSuiteResult:
```

The flow:

1. Split `EVAL_SUITE` into `lm_eval_tasks` (everything except
   `fineweb_edu_ppl`) and the internal PPL task.
2. Call `run_harness_eval(...)` for the 5 LM-eval tasks. If
   `lm_eval` is not installed, fall back to a NaN per task (don't
   crash the whole run).
3. Call `compute_validation_loss(...)` for the FineWeb-Edu PPL.
4. Build a `dict[str, dict[str, Any]]` of per-task results, plus a
   summary dict.
5. Return an `EvalSuiteResult` and **atomically** write the result
   JSON to `output_path` (with `mkdir(parents=True, exist_ok=True)`
   first).

The output JSON shape:

```json
{
  "results": {
    "hellaswag":       { "metric": "acc_norm,none", "value": 0.42 },
    "fineweb_edu_ppl": { "loss": 0.74, "ppl": 2.10 }
  },
  "summary": { "hellaswag": 0.42, "fineweb_edu_ppl": 2.10 },
  "suite":   [["hellaswag", 0], ["arc_challenge", 0], ...]
}
```

### 2.4 `comparison.py` — table formatter

`format_comparison_table(...)` (line 20) takes the eval results dict
and produces a markdown table that compares HyMo against each entry in
`BASELINES`, with rows for each of the 6 tasks. `METRIC_ORDER` (also
exported) is the column order — tasks sorted by importance for the
interview/portfolio reading.

This is what gets pasted into the README's "Results" section after a
30 B run completes.

---

## 3. Ablations — `src/hymo/ablations/__init__.py`

### 3.1 Why four families

The v1.0 design doc defers 4 ablation sweeps to v1.1 (so they don't
block the primary 30 B run). Each family is **one structural
decision** with **two or three variants** (so the comparison is tight):

| Family | Question | Variants |
|---|---|---|
| **A — MoE on attention only?** | Should MoE live on every layer or stay on MLA blocks only (v1.0)? | `v1_0_mla_only` vs `every_layer` |
| **B — Optimizer partition?** | NorMuon-excluding-MoE (v1.0) vs AdamW-only vs NorMuon-everything. | `nor_muon_excl_moe`, `adamw_only`, `nor_muon_all` |
| **C — MTP depth?** | `no_mtp` vs `mtp_depth=1` vs `mtp_depth=2` (v1.0). | `no_mtp`, `depth_1`, `depth_2` |
| **D — MQA-4 vs GQA-1.75?** | Earlier draft used GQA-1.75; v1.0 ships MQA-4. | `mqa4`, `gqa_1_75` |

Each family defines an `AblationSpec`:

```python
@dataclass(frozen=True)
class AblationSpec:
    name: str
    description: str
    variants: tuple[str, ...]
    tokens: int = 7_500_000_000         # 7.5 B per ablation run
    pod_count: int = 1                  # GPU pods reserved
```

`ABLATION_FAMILIES: dict[str, AblationSpec]` (line 36) registers all
four.

### 3.2 `build_ablation_config(family, variant, base, output_dir)` (line 185)

```python
def build_ablation_config(
    family: str, variant: str, base: HyMoConfig, output_dir: str | Path,
) -> HyMoConfig:
    output = Path(output_dir)
    builder = _BUILDERS.get(family)
    if builder is None:
        raise ValueError(f"Unknown ablation family {family!r}. Valid: {list(_BUILDERS)}")
    cfg = builder(variant, base, output)
    return dc_replace(
        cfg,
        run=dc_replace(cfg.run, output_dir=str(output)),
        scheduler=SchedulerConfig(
            total_steps=Step(int(7_500_000_000 / cfg.training.per_step_tokens)),
            warmup_frac=base.scheduler.warmup_frac,
            stable_frac=base.scheduler.stable_frac,
            decay_frac=base.scheduler.decay_frac,
            min_lr_ratio=base.scheduler.min_lr_ratio,
            decay=base.scheduler.decay,
        ),
    )
```

Three things happen:

1. **Per-family builder** — `_build_ablation_config_{A,B,C,D}`
   applies the variant-specific knob (e.g. for family A
   `every_layer`, `inter_dim = moe_inter_dim` so the FFN becomes MoE,
   not dense SwiGLU). Builders are registered in the `_BUILDERS` dict
   (line 177).
2. **Output dir override** — the ablation's `run.output_dir` is
   pinned to the supplied `output_dir` so each variant writes to its
   own folder and doesn't collide with v1.0.
3. **Schedule re-derivation** — the ablation runs for **7.5 B
   tokens** (a quarter of v1.0's 30 B) with the *same* warmup/stable/
   decay fractions. `total_steps = 7.5e9 / per_step_tokens`.

Notice that the schedule is derived once `per_step_tokens` is known;
if you change `TrainingConfig.micro_batch_size` in the base config,
the ablation total steps change accordingly — but the fractions are
inherited from `base.scheduler`, which is what keeps every ablation
shape-comparable.

### 3.3 Worked example — run family B, variant `adamw_only`

```python
from pathlib import Path
from hymo import load_config
from hymo.ablations import ABLATION_FAMILIES, build_ablation_config

base = load_config("configs/hymo_750m.yaml")

print(ABLATION_FAMILIES["B_optimizer_partition"])
# AblationSpec(name='B_optimizer_partition',
#   description='NorMuon-with-MoE-exclusion (v1.0) vs AdamW-only vs NorMuon-everything.',
#   variants=('nor_muon_excl_moe', 'adamw_only', 'nor_muon_all'),
#   pod_count=3)

cfg = build_ablation_config(
    family="B_optimizer_partition",
    variant="adamw_only",
    base=base,
    output_dir="checkpoints/pretrain/ablation/B_adamw_only",
)
```

The returned `cfg` is a fresh, frozen `HyMoConfig`. You can pass it to
`build_hymo` and `Trainer` exactly as you would `base`, and the
optimizer partition will route everything through `CautiousAdamW`.

### 3.4 What is NOT an ablation (today)

Ablation *families* are pre-registered; an *ad-hoc* sweep (e.g. flip
`logit_softcap` from `15.0` to `30.0`) is done by `derive_config`:

```python
from hymo.core.config import derive_config
cfg = derive_config(base, model=replace(base.model, logit_softcap=30.0))
```

`build_ablation_config` is for the 4 planned sweeps with their
specific output dirs and 7.5 B budgets; everything else uses
`derive_config` (or plain `dataclasses.replace`).

---

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
