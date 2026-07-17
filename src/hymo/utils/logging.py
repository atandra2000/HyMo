"""Logging utilities for project logging and structured JSONL metrics logging."""

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
    """Return or initialize a project Logger instance."""
    root = logging.getLogger(_LOGGER_NAME)
    if not root.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATE_FORMAT))
        root.addHandler(handler)
        root.setLevel(logging.INFO)
    if name is None:
        return root
    return root.getChild(name)


@dataclass
class MetricsRecord:
    """A single training step's structured metrics log record."""

    step: int
    metrics: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class MetricsLogger:
    """Append-only JSON-lines structured metrics writer."""

    def __init__(self, path: str | Path, *, flush_every: int = 10) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._flush_every = flush_every
        self._n_since_flush = 0
        self._fp: TextIO = self.path.open("a", encoding="utf-8")
        self._log = get_logger("metrics")

    def log(self, step: int, **metrics: Any) -> None:
        """Write a metrics record JSON line for the given step."""
        rec = MetricsRecord(step=step, metrics=metrics)
        line = json.dumps(asdict(rec), default=_json_default)
        self._fp.write(line + "\n")
        self._n_since_flush += 1
        if self._n_since_flush >= self._flush_every:
            self._fp.flush()
            self._n_since_flush = 0

    def iter_records(self) -> Iterator[MetricsRecord]:
        """Yield every MetricsRecord from the file (re-reads from start)."""
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
        """Return the step number of the last logged record, if any."""
        last: int | None = None
        for rec in self.iter_records():
            last = rec.step
        return last

    def close(self) -> None:
        """Flush and close the open metrics file handle."""
        if not self._fp.closed:
            self._fp.flush()
            self._fp.close()

    def __enter__(self) -> MetricsLogger:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


def _json_default(obj: Any) -> Any:
    """JSON encoder fallback for NumPy and PyTorch scalar/array types."""
    if hasattr(obj, "item"):
        return obj.item()
    if hasattr(obj, "tolist"):
        return obj.tolist()
    return str(obj)


__all__ = ["MetricsLogger", "MetricsRecord", "get_logger"]
