"""lm-evaluation-harness wrapper (Phase 1 placeholder).

The real implementation (architecture doc §15, roadmap E1) is a thin
wrapper around :mod:`lm_eval`. It uses
``lm_eval.models.huggingface.HFLM`` to wrap a HyMo model and runs the
6-eval suite.

This placeholder defines the public surface; the body raises
:class:`NotImplementedError_`.
"""

from __future__ import annotations

from typing import Any

from hymo.core.exceptions import NotImplementedError_

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

    Phase 1 placeholder — raises :class:`NotImplementedError_`. The
    real implementation lands in Phase 4 (design §15, roadmap E1).
    """
    raise NotImplementedError_(
        "run_harness_eval is a Phase 1 placeholder; the real "
        "implementation lands in Phase 4 (design §15, roadmap E1)."
    )
