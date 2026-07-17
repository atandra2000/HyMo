"""The 6-eval suite runner (Phase 4 implementation)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from torch import nn

from hymo.eval.harness import EvalResult, run_harness_eval
from hymo.training.validation import compute_validation_loss

__all__ = ["EVAL_SUITE", "run_all", "EvalSuiteResult"]

# The 6-eval suite (task_name, num_fewshot)
EVAL_SUITE: tuple[tuple[str, int], ...] = (
    ("hellaswag", 0),
    ("arc_challenge", 0),
    ("mmlu", 5),
    ("gsm8k", 8),
    ("humaneval", 0),
    ("fineweb_edu_ppl", 0),
)


class EvalSuiteResult:
    """The result of running the 6-eval suite."""

    __slots__ = ("results", "summary")

    def __init__(
        self,
        results: dict[str, dict[str, float]],
        summary: dict[str, float] | None = None,
    ) -> None:
        self.results = results
        self.summary = summary or {}

    def __repr__(self) -> str:
        return f"EvalSuiteResult(tasks={list(self.results)}, summary={self.summary})"


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
    """Run the 6-eval suite and write results to JSON."""
    device = next(model.parameters()).device
    lm_eval_tasks: list[str] = []
    for task, _n_shot in EVAL_SUITE:
        if task != "fineweb_edu_ppl":
            lm_eval_tasks.append(task)

    harness_results: dict[str, EvalResult] = {}
    if lm_eval_tasks:
        try:
            harness_results = run_harness_eval(
                model=model,
                tokenizer=tokenizer,
                tasks=lm_eval_tasks,
                batch_size=batch_size,
            )
        except ImportError:
            for task in lm_eval_tasks:
                harness_results[task] = EvalResult(
                    task=task, metric="error", value=float("nan"),
                )

    val_metrics = compute_validation_loss(
        model=model,
        batch_size=batch_size,
        seq_len=seq_len,
        vocab_size=vocab_size,
        num_batches=32,
        device=device,
        val_bin_path=Path(val_bin_path),
    )

    results: dict[str, dict[str, Any]] = {}
    for task in lm_eval_tasks:
        er = harness_results.get(task)
        if er is not None:
            results[task] = {
                "metric": er.metric,
                "value": er.value,
            }
            if er.stderr is not None:
                results[task]["stderr"] = er.stderr

    results["fineweb_edu_ppl"] = {
        "loss": val_metrics.loss,
        "ppl": val_metrics.ppl,
    }

    summary: dict[str, float] = {}
    for task, data in results.items():
        if "value" in data:
            summary[task] = data["value"]
        elif "ppl" in data:
            summary[task] = data["ppl"]

    suite_result = EvalSuiteResult(results=results, summary=summary)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as f:
        json.dump(
            {
                "results": results,
                "summary": summary,
                "suite": [(t, ns) for t, ns in EVAL_SUITE],
            },
            f,
            indent=2,
        )

    return suite_result
