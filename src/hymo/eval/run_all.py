"""The 6-eval suite runner (Phase 1 placeholder).

The real implementation (architecture doc §15, roadmap E2) is a single
entry point that runs the 6 evals (FineWeb-Edu PPL, HellaSwag, ARC,
MMLU, GSM8K, HumanEval) and writes ``eval_results.json``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from hymo.core.exceptions import NotImplementedError_

__all__ = ["EVAL_SUITE", "run_all", "EvalSuiteResult"]


# The 6-eval suite (architecture doc §15).
# (task_name, num_fewshot, metric_key_in_results)
EVAL_SUITE: tuple[tuple[str, int], ...] = (
    ("hellaswag", 0),
    ("arc_challenge", 0),
    ("mmlu", 5),
    ("gsm8k", 8),
    ("humaneval", 0),
    ("fineweb_edu_ppl", 0),  # custom eval (uses our compute_validation_loss)
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
    model: Any,
    tokenizer: Any,
    output_path: str | Path = "checkpoints/pretrain/eval/eval_results.json",
    *,
    batch_size: int = 4,
    val_bin_path: str | Path = "data/tokens/val.bin",
) -> EvalSuiteResult:
    """Run the 6-eval suite and write JSON.

    Phase 1 placeholder — raises :class:`NotImplementedError_`. The
    real implementation lands in Phase 4 (design §15, roadmap E2).
    """
    raise NotImplementedError_(
        "run_all is a Phase 1 placeholder; the real implementation "
        "lands in Phase 4 (design §15, roadmap E2)."
    )
