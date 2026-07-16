"""Logging utilities.

Two surfaces:

- :func:`get_logger` — a project-wide logger configured once, with
  optional W&B integration.
- :class:`MetricsLogger` — append-only JSON-lines writer for structured
  metrics. One line per training step; every metric is a key in the
  JSON object.

The :class:`MetricsLogger` is the canonical format for HyMo training
metrics. The trainer writes one line per step; downstream tooling
(notebooks, eval scripts) reads the JSONL line-by-line.
"""

from __future__ import annotations

import json
import logging
import sys
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, TextIO

_LOGGER_NAME = "hymo"
_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
_DATE_FORMAT = "%Y-%m-%dT%H:%M:%S"


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a project logger.

    The root ``hymo`` logger is configured once with a stderr handler
    and the standard format. Subsequent calls return child loggers.

    Parameters
    ----------
    name : str or None
        If None, returns the root ``hymo`` logger. If a dotted suffix
        is given, returns a child logger (e.g. ``"trainer"`` →
        ``hymo.trainer``).
    """
    root = logging.getLogger(_LOGGER_NAME)
    if not root.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATE_FORMAT))
        root.addHandler(handler)
        root.setLevel(logging.INFO)
    if name is None:
        return root
    return root.getChild(name)


# ----------------------------------------------------------------------
# Structured metrics logger (JSON-lines)
# ----------------------------------------------------------------------


@dataclass
class MetricsRecord:
    """A single step's worth of metrics.

    ``step`` is the global optimizer step. ``timestamp`` is set on
    construction. ``metrics`` is the free-form dict of scalar values.
    Extra fields can be added at construction; this dataclass is
    intentionally permissive.
    """

    step: int
    metrics: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class MetricsLogger:
    """Append-only JSON-lines metrics writer.

    Usage
    -----
    .. code-block:: python

        logger = MetricsLogger(Path("logs/metrics.jsonl"))
        logger.log(step=0, loss=11.06, lr=0.0)
        logger.log(step=1, loss=10.92, lr=1e-5)
        logger.close()

        # Or as a context manager:
        with MetricsLogger(Path("logs/metrics.jsonl")) as logger:
            logger.log(step=0, loss=11.06)
    """

    def __init__(self, path: str | Path, *, flush_every: int = 10) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._flush_every = flush_every
        self._n_since_flush = 0
        self._fp: TextIO = self.path.open("a", encoding="utf-8")
        self._log = get_logger("metrics")

    def log(self, step: int, **metrics: Any) -> None:
        """Write one JSON line for the given step + metric kwargs."""
        rec = MetricsRecord(step=step, metrics=metrics)
        line = json.dumps(asdict(rec), default=_json_default)
        self._fp.write(line + "\n")
        self._n_since_flush += 1
        if self._n_since_flush >= self._flush_every:
            self._fp.flush()
            self._n_since_flush = 0

    def iter_records(self) -> Iterator[MetricsRecord]:
        """Yield every record from the file (re-reads from the start)."""
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                yield MetricsRecord(
                    step=d["step"],
                    metrics=d.get("metrics", {}),
                    timestamp=d.get("timestamp", ""),
                )

    def last_step(self) -> int | None:
        """Return the step of the last record, or None if empty."""
        last: int | None = None
        for rec in self.iter_records():
            last = rec.step
        return last

    def close(self) -> None:
        if not self._fp.closed:
            self._fp.flush()
            self._fp.close()

    def __enter__(self) -> MetricsLogger:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


def _json_default(obj: Any) -> Any:
    """JSON encoder default for non-trivial types."""
    if hasattr(obj, "item"):
        # NumPy / torch scalars
        return obj.item()
    if hasattr(obj, "tolist"):
        return obj.tolist()
    return str(obj)


__all__ = ["MetricsLogger", "MetricsRecord", "get_logger"]
