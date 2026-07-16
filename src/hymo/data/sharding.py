"""Shard writer, dataset, and dataloader builder (Phase 4 implementation).

Architecture doc §6.2, roadmap A6.

- :class:`ShardWriter` writes 50M-token ``uint32`` shards.
- :class:`ShardDataset` reads shards as a :class:`torch.utils.data.Dataset`,
  yielding ``(tokens, targets)`` windows of ``max_seq_len`` tokens.
- :class:`DataLoaderBuilder` is a thin wrapper around
  :class:`torch.utils.data.DataLoader` with prefetch + multi-worker
  settings.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from torch.utils.data import DataLoader, Dataset

from hymo.core.config import TrainingConfig

__all__ = ["ShardWriter", "ShardDataset", "DataLoaderBuilder"]

_DEFAULT_SHARD_DIR = Path("data/shards")


class ShardWriter:
    """Write 50M-token ``uint32`` shards.

    Architecture doc §6.2.
    """

    def __init__(
        self,
        output_dir: str | Path = _DEFAULT_SHARD_DIR,
        shard_size_tokens: int = 50_000_000,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.shard_size_tokens = shard_size_tokens

    def write_shard(self, shard_idx: int, tokens: npt.NDArray[np.uint32]) -> Path:
        """Write one shard to ``{output_dir}/shard_{shard_idx:05d}.bin``."""
        path = self.output_dir / f"shard_{shard_idx:05d}.bin"
        tokens.astype(np.uint32).tofile(path)
        return path

    def write_batched(
        self, token_stream: npt.NDArray[np.uint32]
    ) -> list[Path]:
        """Write a flat token array into shards of ``shard_size_tokens``.

        Returns the list of written shard paths.
        """
        paths: list[Path] = []
        shard_idx = 0
        offset = 0
        while offset < len(token_stream):
            chunk = token_stream[offset : offset + self.shard_size_tokens]
            if len(chunk) < self.shard_size_tokens:
                pad = self.shard_size_tokens - len(chunk)
                chunk = np.pad(chunk, (0, pad), constant_values=0)
            path = self.write_shard(shard_idx, chunk)
            paths.append(path)
            shard_idx += 1
            offset += self.shard_size_tokens
        return paths


def _discover_shards(shards_dir: Path) -> list[Path]:
    """Return sorted list of shard_*.bin paths."""
    return sorted(shards_dir.glob("shard_*.bin"))


class ShardDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Read shards and yield ``(tokens, targets)`` windows.

    Architecture doc §6.2 / §7.1.

    Parameters
    ----------
    shards_dir : str or Path
        Directory containing ``shard_*.bin`` files.
    max_seq_len : int
        Window size.
    """

    def __init__(
        self,
        shards_dir: str | Path = _DEFAULT_SHARD_DIR,
        max_seq_len: int = 4_096,
    ) -> None:
        self.shards_dir = Path(shards_dir)
        self.max_seq_len = max_seq_len
        self._shard_paths: list[Path] = _discover_shards(self.shards_dir)
        self._shard_data: list[npt.NDArray[np.uint32]] = []
        self._total_tokens: int = 0
        self._load_all()

    def _load_all(self) -> None:
        """Load all shards into memory and compute total examples."""
        for sp in self._shard_paths:
            data = np.fromfile(sp, dtype=np.uint32)
            self._shard_data.append(data)
            self._total_tokens += len(data)

    def _locate(self, idx: int) -> tuple[int, int]:
        """Return (shard_index, token_offset) for the idx-th example.

        Each example consumes ``max_seq_len + 1`` tokens (inputs + targets).
        """
        target = idx * (self.max_seq_len + 1)
        for si, data in enumerate(self._shard_data):
            if target < len(data):
                return si, target
            target -= len(data)
        raise IndexError(f"Index {idx} out of range")

    def __len__(self) -> int:
        if self._total_tokens == 0:
            return 0
        return self._total_tokens // (self.max_seq_len + 1)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        si, offset = self._locate(idx)
        data = self._shard_data[si]
        chunk = data[offset : offset + self.max_seq_len + 1]
        if len(chunk) < self.max_seq_len + 1:
            remainder = (self.max_seq_len + 1) - len(chunk)
            # Cycle to the next shard for filler tokens.
            next_si = (si + 1) % len(self._shard_data)
            chunk = np.concatenate(
                [chunk, self._shard_data[next_si][:remainder]]
            )
        tokens = torch.from_numpy(chunk[: self.max_seq_len].astype(np.int64))
        targets = torch.from_numpy(chunk[1 : self.max_seq_len + 1].astype(np.int64))
        return tokens, targets


class DataLoaderBuilder:
    """Build a :class:`torch.utils.data.DataLoader` for training.

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

    def build(self) -> DataLoader[Any]:
        """Build a :class:`torch.utils.data.DataLoader`."""
        effective_batch = self.config.micro_batch_size
        sampler = torch.utils.data.RandomSampler(
            self.dataset,
            replacement=True,
            num_samples=(
                self.config.gradient_accumulation_steps
                * self.config.world_size
                * effective_batch
                * 100
            ),
        )
        num_workers = min(4, self.config.world_size) if self.config.world_size > 1 else 0
        return DataLoader(
            self.dataset,
            batch_size=effective_batch,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
            prefetch_factor=2 if num_workers > 0 else None,
            drop_last=True,
        )
