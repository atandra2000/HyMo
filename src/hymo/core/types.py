"""Shared type aliases and semantic newtypes."""

from __future__ import annotations

from pathlib import Path as Path
from typing import NewType

import torch

# Semantic newtypes
TokenId = NewType("TokenId", int)
LayerIndex = NewType("LayerIndex", int)
ExpertIndex = NewType("ExpertIndex", int)
MicroStep = NewType("MicroStep", int)
Step = NewType("Step", int)

# Re-exports & Aliases
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
