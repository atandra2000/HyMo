"""Seeding utilities for Python, NumPy, and PyTorch."""

from __future__ import annotations

import os
import random

import numpy as np
import torch

__all__ = ["set_seed", "seed_for_rank"]


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
