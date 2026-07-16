"""lm-evaluation-harness wrapper (Phase 4 implementation).

Architecture doc §15, roadmap E1. Thin wrapper around ``lm_eval``
using ``lm_eval.models.huggingface.HFLM`` to wrap a HyMo model.
"""

from __future__ import annotations

from typing import Any

from hymo.eval.baselines import TASK_TO_METRIC

__all__ = ["run_harness_eval", "EvalResult"]


class EvalResult:
    """The result of a single eval run.

    Attributes
    ----------
    task : str
        Task name (e.g. ``"hellaswag"``).
    metric : str
        The metric key from lm-eval (e.g. ``"acc_norm,none"``).
    value : float
        The metric value.
    stderr : float or None
        The stderr (if reported by lm-eval).
    """

    __slots__ = ("task", "metric", "value", "stderr")

    def __init__(
        self,
        task: str,
        metric: str,
        value: float,
        stderr: float | None = None,
    ) -> None:
        self.task = task
        self.metric = metric
        self.value = value
        self.stderr = stderr

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
    """Run lm-evaluation-harness on the given tasks.

    Parameters
    ----------
    model : nn.Module
        The HyMo model instance.
    tokenizer : ExtendedTokenizer
        The tokenizer (used for encoding/decoding).
    tasks : list of str
        Task names understood by lm-eval (e.g. ``["hellaswag"]``).
    num_fewshot : int
        Number of few-shot examples (default 0).
    batch_size : int
        Batch size for evaluation.

    Returns
    -------
    dict[str, EvalResult]
        Mapping from task name to :class:`EvalResult`.
    """
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
