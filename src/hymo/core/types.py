"""Shared type aliases and semantic newtypes.

This module is PyTorch-free except for the ``torch.dtype`` re-export
under :data:`DType`. The :class:`NewType` aliases are pure type-checker
constructs — zero runtime cost.

Semantic newtypes
-----------------
- :class:`TokenId` — int in ``[0, vocab_size)``.
- :class:`LayerIndex` — int in ``[0, n_layers)``.
- :class:`ExpertIndex` — int in ``[0, n_experts)``.
- :class:`MicroStep` — int, the gradient-accumulation step counter.
- :class:`Step` — int, the global optimizer step.
- :class:`Path` — :class:`pathlib.Path` (alias, not newtype).

Type aliases
------------
- :data:`DType` — :class:`torch.dtype`.
- :data:`Device` — :class:`torch.device` or :class:`str`.
- :data:`Shape` — :class:`tuple` of :class:`int`.
"""

from __future__ import annotations

from pathlib import Path as Path
from typing import NewType

import torch

# Semantic newtypes — int at runtime, distinct at type-check time.
TokenId = NewType("TokenId", int)
LayerIndex = NewType("LayerIndex", int)
ExpertIndex = NewType("ExpertIndex", int)
MicroStep = NewType("MicroStep", int)
Step = NewType("Step", int)

# Re-exports.
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
