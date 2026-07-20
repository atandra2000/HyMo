"""Public API of :mod:`hymo.utils`."""

from __future__ import annotations

import os
import random
import numpy as np
import torch
import json
import logging
import sys

from collections.abc import Generator
from contextlib import contextmanager
from collections.abc import Callable
from pathlib import Path
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, TextIO
from hymo.core.types import DType

__all__ = [
    "set_seed", "seed_for_rank",
    "BF16", "FP32", "resolve_dtype", "autocast_disabled", "bf16_forward", "fp32_master_weights",]







def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch global random number generators."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_for_rank(base_seed: int, rank: int) -> int:
    """Derive a deterministic per-rank seed from a base seed and rank index."""
    return base_seed + rank

"""Precision and dtype utility definitions and context managers."""




from hymo.core.types import DType


# Canonical aliases
BF16: DType = torch.bfloat16
FP32: DType = torch.float32


def resolve_dtype(name: str) -> DType:
    """Resolve a string representation to a torch.dtype."""
    name = name.lower()
    if name in ("bfloat16", "bf16"):
        return torch.bfloat16
    if name in ("float32", "fp32"):
        return torch.float32
    if name in ("float16", "fp16", "half"):
        return torch.float16
    raise ValueError(f"Unknown dtype name: {name!r}")


@contextmanager
def autocast_disabled() -> Generator[None, None, None]:
    """Disable autocast context temporarily."""
    with torch.no_grad():
        if torch.cuda.is_available():
            with torch.amp.autocast(device_type="cuda", enabled=False):
                yield
        else:
            yield


@contextmanager
def bf16_forward() -> Generator[None, None, None]:
    """Execute forward pass under bfloat16 autocast context."""
    if torch.cuda.is_available():
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            yield
    else:
        yield


@contextmanager
def fp32_master_weights() -> Generator[None, None, None]:
    """Execute context for FP32 master weight updates."""
    yield

"""Atomic file-write helpers implementing tmp -> rename semantics."""






class CheckpointIOError(Exception):
    """Atomic-write helper failed."""


def atomic_write_bytes(path: str | Path, data: bytes) -> None:
    """Write bytes to path atomically using temporary files and os.replace."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        with open(tmp, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except OSError as e:
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass
        raise CheckpointIOError(f"Failed to write {path}: {e}") from e


def atomic_write_with(
    path: str | Path, writer: Callable[[Path], None]
) -> None:
    """Write data to path atomically, calling writer(tmp_path) to write."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        writer(tmp)
        fd = os.open(str(path.parent), os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        os.replace(tmp, path)
    except OSError as e:
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass
        raise CheckpointIOError(f"Failed to write {path}: {e}") from e
    except Exception:
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass
        raise

"""Logging utilities for project logging and structured JSONL metrics logging."""



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


