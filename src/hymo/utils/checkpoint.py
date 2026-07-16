"""Atomic checkpoint write helper (low-level, model-agnostic).

The DCP-based checkpoint save/load lives in :mod:`hymo.training.checkpoint`
(Phase 3). This module is the *low-level* atomic-write helper used by
the JSONL metrics logger and the YAML config dumper — anywhere we want
``tmp → rename`` semantics without DCP.

Pattern
-------
1. Write to ``path.with_suffix(path.suffix + ".tmp")``.
2. :func:`os.replace` the tmp to the final path (atomic on POSIX and
   Windows).
3. :func:`os.fsync` before the replace to ensure the data is on disk.

This pattern is what the design doc §7.4 calls out as "atomic
checkpoint: torch.save → .tmp → os.rename" — generalized to any file
type.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

from hymo.core.exceptions import HyMoError

__all__ = ["CheckpointIOError", "atomic_write_bytes", "atomic_write_with"]


class CheckpointIOError(HyMoError):
    """Atomic-write helper failed (permission, disk full, etc.)."""


def atomic_write_bytes(path: str | Path, data: bytes) -> None:
    """Write ``data`` to ``path`` atomically.

    See module docstring for the pattern.
    """
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
        # Clean up the tmp file on failure (best-effort).
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass
        raise CheckpointIOError(f"Failed to write {path}: {e}") from e


def atomic_write_with(
    path: str | Path, writer: Callable[[Path], None]
) -> None:
    """Write to ``path`` atomically, using ``writer`` to produce the data.

    ``writer`` receives the tmp path and is expected to write the data
    to it. The atomic rename happens after ``writer`` returns.

    Example
    -------
    .. code-block:: python

        def writer(tmp):
            with open(tmp, "w") as f:
                yaml.dump(data, f)
        atomic_write_with("out.yaml", writer)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        writer(tmp)
        # fsync the directory entry so the rename is durable.
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
