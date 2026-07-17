"""Atomic file-write helpers implementing tmp -> rename semantics."""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

from hymo.core.exceptions import HyMoError

__all__ = ["CheckpointIOError", "atomic_write_bytes", "atomic_write_with"]


class CheckpointIOError(HyMoError):
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
