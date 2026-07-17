"""Published baseline numbers for the 6-eval suite (architecture doc §15)."""

from __future__ import annotations

__all__ = ["BASELINES", "TASK_TO_METRIC", "BASELINE_DISPLAY_NAMES"]

BASELINES: dict[str, dict[str, float]] = {
    "pythia-1b": {
        "fineweb_edu_ppl": 2.45,
        "hellaswag": 0.36,
        "arc_challenge": 0.24,
        "mmlu": 0.24,
        "gsm8k": 0.03,
        "humaneval": 0.05,
    },
    "mobile_moe_0.9b": {
        "fineweb_edu_ppl": 2.10,
        "hellaswag": 0.38,
        "arc_challenge": 0.26,
        "mmlu": 0.27,
        "gsm8k": 0.06,
        "humaneval": 0.09,
    },
    "smollm2_1.7b": {
        "fineweb_edu_ppl": 2.20,
        "hellaswag": 0.42,
        "arc_challenge": 0.30,
        "mmlu": 0.31,
        "gsm8k": 0.10,
        "humaneval": 0.12,
    },
}

TASK_TO_METRIC: dict[str, str] = {
    "hellaswag": "acc_norm,none",
    "arc_challenge": "acc_norm,none",
    "mmlu": "acc,none",
    "gsm8k": "exact_match,flexible-extract",
    "humaneval": "pass@1",
    "fineweb_edu_ppl": "ppl",
}

BASELINE_DISPLAY_NAMES: dict[str, str] = {
    "pythia-1b": "Pythia-1B",
    "mobile_moe_0.9b": "MobileMoE-0.9B",
    "smollm2_1.7b": "SmolLM2-1.7B",
}
