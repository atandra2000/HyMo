"""In-memory metric tracking (counters, rolling means, etc.).

Distinct from :mod:`hymo.utils.logging` (which is the on-disk writer).
:class:`Metric` is a single named scalar with a reduce function. Used
by the trainer to track per-step loss, grad norm, expert-load entropy,
etc.

The :class:`MetricsLogger` in :mod:`hymo.utils.logging` is the on-disk
sink; this module is the in-memory accumulator.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

__all__ = ["Metric", "MetricCollection"]


@dataclass
class Metric:
    """A single named scalar with a reduce function.

    Reduce functions
    ----------------
    - ``"mean"`` (default): arithmetic mean of all updates.
    - ``"sum"``: cumulative sum.
    - ``"last"``: most recent value.
    - ``"max"``: running max.
    - ``"min"``: running min.
    """

    name: str
    reduce: str = "mean"
    _values: list[float] = field(default_factory=list)

    def update(self, value: float) -> None:
        self._values.append(value)

    def value(self) -> float:
        if not self._values:
            return 0.0
        if self.reduce == "mean":
            return sum(self._values) / len(self._values)
        if self.reduce == "sum":
            return sum(self._values)
        if self.reduce == "last":
            return self._values[-1]
        if self.reduce == "max":
            return max(self._values)
        if self.reduce == "min":
            return min(self._values)
        raise ValueError(f"Unknown reduce: {self.reduce!r}")

    def reset(self) -> None:
        self._values.clear()

    def as_dict(self) -> dict[str, float]:
        return {self.name: self.value()}


class MetricCollection:
    """A keyed collection of :class:`Metric` objects.

    Usage
    -----
    .. code-block:: python

        metrics = MetricCollection()
        metrics.add("loss", reduce="mean")
        metrics.add("grad_norm", reduce="last")
        metrics.update("loss", 11.06)
        metrics.update("grad_norm", 5.2)
        d = metrics.as_dict()  # {"loss": 11.06, "grad_norm": 5.2}
    """

    def __init__(self) -> None:
        self._metrics: dict[str, Metric] = {}

    def add(self, name: str, reduce: str = "mean") -> None:
        if name in self._metrics:
            raise ValueError(f"Metric {name!r} already in collection")
        self._metrics[name] = Metric(name=name, reduce=reduce)

    def update(self, name: str, value: float) -> None:
        try:
            self._metrics[name].update(value)
        except KeyError as e:
            raise KeyError(
                f"Metric {name!r} not registered; call add() first"
            ) from e

    def get(self, name: str) -> Metric:
        return self._metrics[name]

    def value(self, name: str) -> float:
        return self._metrics[name].value()

    def as_dict(self) -> dict[str, float]:
        return {name: m.value() for name, m in self._metrics.items()}

    def reset(self, names: Iterable[str] | None = None) -> None:
        if names is None:
            for m in self._metrics.values():
                m.reset()
        else:
            for name in names:
                self._metrics[name].reset()

    def names(self) -> list[str]:
        return sorted(self._metrics)

    def __contains__(self, name: str) -> bool:
        return name in self._metrics

    def __len__(self) -> int:
        return len(self._metrics)
