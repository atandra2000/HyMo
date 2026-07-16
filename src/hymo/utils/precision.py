"""Precision utilities: dtype enums and context managers.

The two relevant dtypes in HyMo are:

- **BF16** — the compute dtype. Forward and backward in BF16; no
  GradScaler (BF16 does not need one).
- **FP32** — the optimizer-state and master-weight dtype (architecture
  doc §7.2, the "FP32 master weights throughout" choice).

The context managers :func:`autocast_disabled` and :func:`bf16_forward`
are convenience wrappers around :func:`torch.cuda.amp.autocast` and the
manual dtype-conversion paths the trainer uses.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import torch

from hymo.core.types import DType

__all__ = [
    "BF16",
    "FP32",
    "resolve_dtype",
    "autocast_disabled",
    "bf16_forward",
    "fp32_master_weights",
]


# Canonical aliases.
BF16: DType = torch.bfloat16
FP32: DType = torch.float32


def resolve_dtype(name: str) -> DType:
    """Map a string dtype name to a :class:`torch.dtype`.

    Accepts the names used in the config: ``"bfloat16"``, ``"float32"``,
    ``"float16"``.

    Raises
    ------
    ValueError
        If the name is unknown.
    """
    name = name.lower()
    if name in ("bfloat16", "bf16"):
        return torch.bfloat16
    if name in ("float32", "fp32"):
        return torch.float32
    if name in ("float16", "fp16", "half"):
        return torch.float16
    raise ValueError(f"Unknown dtype name: {name!r}")


@contextmanager
def autocast_disabled() -> Iterator[None]:
    """Disable autocast within the ``with`` block.

    Used by the trainer to ensure the backward path is computed in plain
    BF16 (no FP32 autocast) — see architecture doc §7.2.
    """
    with torch.no_grad():  # placeholder; the real work is the cast
        # PyTorch's autocast is the canonical way; we just need a
        # context where autocast(dtype=...) is bypassed.
        if torch.cuda.is_available():
            with torch.amp.autocast(device_type="cuda", enabled=False):
                yield
        else:
            yield


@contextmanager
def bf16_forward() -> Iterator[None]:
    """Run the forward pass under BF16 autocast (no-op on CPU)."""
    if torch.cuda.is_available():
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            yield
    else:
        yield


@contextmanager
def fp32_master_weights() -> Iterator[None]:
    """Context manager for the optimizer step (FP32 master path)."""
    # No-op context; the actual FP32 path is implemented in the
    # optimizer's ``step`` method (Phase 3).
    yield
