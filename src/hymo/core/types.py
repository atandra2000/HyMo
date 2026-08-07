"""Shared type vocabulary for configuration, model, and training modules.

NewTypes distinguish integer concepts such as token IDs and optimizer steps to
make accidental interchange visible to static type checkers.
"""

from __future__ import annotations

from pathlib import Path as Path
from typing import NewType

import torch

# Domain-specific integer types retain runtime compatibility with ``int``.
TokenId = NewType("TokenId", int)
LayerIndex = NewType("LayerIndex", int)
ExpertIndex = NewType("ExpertIndex", int)
MicroStep = NewType("MicroStep", int)
Step = NewType("Step", int)

# Torch aliases are centralized here so public signatures use one vocabulary.
DType = torch.dtype
Device = torch.device | str
Shape = tuple[int, ...]

__all__ = [
    "DType",
    "Device",
    "ExpertIndex",
    "LayerIndex",
    "MicroStep",
    "Path",
    "Shape",
    "Step",
    "TokenId",
]
