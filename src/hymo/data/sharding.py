"""Shard writer and dataset (Phase 1 placeholders).

The real implementation (architecture doc §6.2, roadmap A6):

- :class:`ShardWriter` writes 50M-token ``uint32`` shards; cross-document
  boundary is not allowed within a shard.
- :class:`ShardDataset` reads shards as a :class:`torch.utils.data.Dataset`,
  yielding ``(tokens, targets)`` windows of ``max_seq_len`` tokens.
- :class:`DataLoaderBuilder` is a thin wrapper around
  :class:`torch.utils.data.DataLoader` with prefetch + multi-worker
  settings (architecture doc §13.7: 4 workers per rank, prefetch
  overlapped with compute).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from torch.utils.data import Dataset

from hymo.core.config import TrainingConfig
from hymo.core.exceptions import NotImplementedError_

__all__ = ["ShardWriter", "ShardDataset", "DataLoaderBuilder"]


class ShardWriter:
    """Write 50M-token ``uint32`` shards (Phase 1 placeholder).

    Architecture doc §6.2. Phase 1 placeholder.
    """

    def __init__(
        self,
        output_dir: str | Path,
        shard_size_tokens: int = 50_000_000,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.shard_size_tokens = shard_size_tokens

    def write_shard(self, shard_idx: int, tokens: npt.NDArray[np.uint32]) -> Path:
        """Write one shard to ``{output_dir}/shard_{shard_idx:05d}.bin``."""
        raise NotImplementedError_(
            "ShardWriter.write_shard is a Phase 1 placeholder; the real "
            "implementation lands in Phase 4 (design §6.2, roadmap A6)."
        )


class ShardDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Read shards and yield (tokens, targets) windows (Phase 1 placeholder).

    Architecture doc §6.2 / §7.1. Phase 1 placeholder.

    Parameters
    ----------
    shards_dir : str or Path
        Directory containing ``shard_*.bin`` files.
    max_seq_len : int
        Window size.
    """

    def __init__(self, shards_dir: str | Path, max_seq_len: int = 4_096) -> None:
        self.shards_dir = Path(shards_dir)
        self.max_seq_len = max_seq_len
        self._shard_paths: list[Path] = []
        self._total_tokens: int = 0

    def __len__(self) -> int:
        raise NotImplementedError_(
            "ShardDataset.__len__ is a Phase 1 placeholder."
        )

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError_(
            "ShardDataset.__getitem__ is a Phase 1 placeholder."
        )


class DataLoaderBuilder:
    """Build a :class:`torch.utils.data.DataLoader` (Phase 1 placeholder).

    Architecture doc §13.7: 4 workers per rank, prefetch overlapped
    with compute.
    """

    def __init__(
        self,
        dataset: ShardDataset,
        config: TrainingConfig,
    ) -> None:
        self.dataset = dataset
        self.config = config

    def build(self) -> Any:
        """Build a :class:`torch.utils.data.DataLoader`."""
        raise NotImplementedError_(
            "DataLoaderBuilder.build is a Phase 1 placeholder; the real "
            "implementation lands in Phase 4 (design §6.2, roadmap A6)."
        )
