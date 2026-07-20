"""lm-evaluation-harness wrapper (Phase 4 implementation)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from hymo.eval.baselines import TASK_TO_METRIC

__all__ = ["run_harness_eval", "EvalResult"]


@dataclass
class EvalResult:
    """The result of a single task eval run."""
    task: str
    metric: str
    value: float
    stderr: float | None = None

    def __repr__(self) -> str:
        s = f"{self.value:.4f}"
        if self.stderr is not None:
            s += f" ± {self.stderr:.4f}"
        return f"EvalResult({self.task!r}, {self.metric!r}={s})"


def run_harness_eval(
    model: Any,
    tokenizer: Any,
    tasks: list[str],
    *,
    num_fewshot: int = 0,
    batch_size: int = 4,
) -> dict[str, EvalResult]:
    """Run lm-evaluation-harness on the given tasks for a HyMo model."""
    import lm_eval
    from lm_eval.models.huggingface import HFLM

    lm = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        batch_size=batch_size,
        device=next(model.parameters()).device,
    )
    results = lm_eval.simple_evaluate(
        model=lm,
        tasks=tasks,
        num_fewshot=num_fewshot,
        batch_size=batch_size,
    )
    task_results: dict[str, EvalResult] = {}
    for task in tasks:
        task_data = results["results"].get(task, {})
        metric_key = TASK_TO_METRIC.get(task, next(iter(task_data)))
        value = task_data.get(metric_key, float("nan"))
        stderr = task_data.get(f"{metric_key}_stderr", None)
        task_results[task] = EvalResult(
            task=task,
            metric=metric_key,
            value=float(value) if isinstance(value, (int, float)) else float("nan"),
            stderr=float(stderr) if stderr is not None else None,
        )
    return task_results
