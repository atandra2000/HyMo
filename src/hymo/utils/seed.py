"""Seeding and per-rank seed broadcast.

The trainer seeds every rank from a base seed + rank index for the
random ops, and broadcasts the same model parameters from rank 0 to
all ranks (architecture doc §4 and §13.4). This module provides the
seeding utilities; the FSDP-side broadcast is in
:mod:`hymo.training.fsdp` (Phase 3).
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch

__all__ = ["set_seed", "seed_for_rank"]


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch (CPU + all visible CUDA devices).

    Idempotent. Safe to call multiple times.

    Parameters
    ----------
    seed : int
        The base seed. Per-rank seeds are derived as
        ``seed + rank`` (see :func:`seed_for_rank`).
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_for_rank(base_seed: int, rank: int) -> int:
    """Derive a deterministic per-rank seed from a base seed + rank index.

    Parameters
    ----------
    base_seed : int
        The base seed (e.g. ``RunConfig.seed``).
    rank : int
        The rank index (0..world_size-1).

    Returns
    -------
    int
        ``base_seed + rank`` — different per rank but deterministic
        across runs.
    """
    return base_seed + rank
