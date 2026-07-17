"""Precision and dtype utility definitions and context managers."""

from __future__ import annotations

from collections.abc import Generator
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
