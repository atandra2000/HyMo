"""Public API of :mod:`hymo.eval`."""

from __future__ import annotations

from hymo.eval.baselines import BASELINE_DISPLAY_NAMES, BASELINES, TASK_TO_METRIC
from hymo.eval.comparison import METRIC_ORDER, format_comparison_table
from hymo.eval.harness import EvalResult, run_harness_eval
from hymo.eval.run_all import EVAL_SUITE, EvalSuiteResult, run_all

__all__ = [
    # Baselines
    "BASELINES",
    "BASELINE_DISPLAY_NAMES",
    "TASK_TO_METRIC",
    # Comparison
    "METRIC_ORDER",
    "format_comparison_table",
    # Harness
    "EvalResult",
    "run_harness_eval",
    # Suite
    "EVAL_SUITE",
    "EvalSuiteResult",
    "run_all",
]
