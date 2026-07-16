"""Tests for the eval placeholders and the comparison table."""

from __future__ import annotations

import pytest

from hymo.core.exceptions import NotImplementedError_
from hymo.eval import (
    BASELINES,
    EVAL_SUITE,
    METRIC_ORDER,
    TASK_TO_METRIC,
    EvalResult,
    EvalSuiteResult,
    format_comparison_table,
    run_all,
    run_harness_eval,
)


class TestBaselines:
    def test_three_baselines(self) -> None:
        assert set(BASELINES.keys()) == {"pythia-1b", "mobile_moe_0.9b", "smollm2_1.7b"}

    def test_six_metrics_per_baseline(self) -> None:
        for name, metrics in BASELINES.items():
            assert set(metrics.keys()) == {
                "fineweb_edu_ppl",
                "hellaswag",
                "arc_challenge",
                "mmlu",
                "gsm8k",
                "humaneval",
            }, f"Baseline {name!r} has wrong metrics"

    def test_mobile_moe_ppl_target(self) -> None:
        """The v1.0 PPL target is ≤ 2.10 (architecture doc §15)."""
        assert BASELINES["mobile_moe_0.9b"]["fineweb_edu_ppl"] == 2.10


class TestTaskToMetric:
    def test_all_six_tasks_mapped(self) -> None:
        assert set(TASK_TO_METRIC.keys()) == {
            "hellaswag",
            "arc_challenge",
            "mmlu",
            "gsm8k",
            "humaneval",
            "fineweb_edu_ppl",
        }


class TestMetricOrder:
    def test_six_metrics_in_order(self) -> None:
        assert len(METRIC_ORDER) == 6
        assert METRIC_ORDER[0] == "fineweb_edu_ppl"
        assert METRIC_ORDER[-1] == "humaneval"


class TestFormatComparisonTable:
    def test_produces_markdown(self) -> None:
        results = {
            "fineweb_edu_ppl": 2.05,
            "hellaswag": 0.40,
            "arc_challenge": 0.26,
            "mmlu": 0.27,
            "gsm8k": 0.06,
            "humaneval": 0.08,
        }
        table = format_comparison_table(results)
        # Header.
        assert "| Metric | HyMo |" in table
        assert "| Pythia-1B |" in table
        assert "| MobileMoE-0.9B |" in table
        assert "| SmolLM2-1.7B |" in table
        # 6 rows + 1 header + 1 separator = 8 lines.
        assert len(table.split("\n")) == 8
        # Each row should contain the HyMo value formatted to 3 decimals.
        assert "2.050" in table  # fineweb_edu_ppl
        assert "0.400" in table  # hellaswag

    def test_subset_of_baselines(self) -> None:
        results = {
            "fineweb_edu_ppl": 2.05,
            "hellaswag": 0.40,
            "arc_challenge": 0.26,
            "mmlu": 0.27,
            "gsm8k": 0.06,
            "humaneval": 0.08,
        }
        table = format_comparison_table(results, baselines=("pythia-1b",))
        assert "| Pythia-1B |" in table
        # The other two baselines should not appear.
        assert "MobileMoE" not in table
        assert "SmolLM" not in table


class TestEvalResult:
    def test_construct(self) -> None:
        r = EvalResult(task="hellaswag", metric="acc_norm,none", value=0.40)
        assert r.task == "hellaswag"
        assert r.metric == "acc_norm,none"
        assert r.value == 0.40
        assert r.stderr is None

    def test_with_stderr(self) -> None:
        r = EvalResult(task="arc", metric="acc", value=0.25, stderr=0.01)
        assert r.stderr == 0.01

    def test_repr(self) -> None:
        r = EvalResult(task="x", metric="m", value=0.5)
        assert "x" in repr(r)
        assert "0.5000" in repr(r)


class TestRunHarnessEval:
    def test_raises(self) -> None:
        with pytest.raises(NotImplementedError_):
            run_harness_eval(None, None, ["hellaswag"])


class TestRunAll:
    def test_six_tasks_in_suite(self) -> None:
        assert len(EVAL_SUITE) == 6
        task_names = [t for t, _ in EVAL_SUITE]
        assert "hellaswag" in task_names
        assert "arc_challenge" in task_names
        assert "mmlu" in task_names
        assert "gsm8k" in task_names
        assert "humaneval" in task_names

    def test_raises(self) -> None:
        with pytest.raises(NotImplementedError_):
            run_all(None, None)


class TestEvalSuiteResult:
    def test_construct(self) -> None:
        r = EvalSuiteResult(
            results={"hellaswag": {"acc_norm": 0.40}},
            summary={"hellaswag_acc": 0.40},
        )
        assert r.results == {"hellaswag": {"acc_norm": 0.40}}
        assert r.summary == {"hellaswag_acc": 0.40}
