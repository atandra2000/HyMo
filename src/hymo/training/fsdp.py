"""FSDP-2 wrapper placeholders for Phase 1/Phase 3."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from torch import nn

from hymo.core.config import TrainingConfig
from hymo.core.exceptions import NotImplementedError_

__all__ = [
    "wrap_model_with_fsdp",
    "fsdp_auto_wrap_policy",
    "shard_nor_muon_params",
    "RankedParamShard",
]


def fsdp_auto_wrap_policy(module: nn.Module, recurse: bool, non_blocking: bool) -> bool:
    """FSDP auto-wrap policy function (Phase 1 placeholder)."""
    raise NotImplementedError_(
        "fsdp_auto_wrap_policy is a Phase 1 placeholder; the real "
        "implementation lands in Phase 3 (design §13.2, roadmap D1)."
    )


class RankedParamShard:
    """The parameter partitioning shard assignments across ranks."""

    __slots__ = ("rank_assignments", "rank_byte_counts")

    def __init__(
        self,
        rank_assignments: list[list[nn.Parameter]],
        rank_byte_counts: list[int],
    ) -> None:
        self.rank_assignments = rank_assignments
        self.rank_byte_counts = rank_byte_counts

    def __repr__(self) -> str:
        n = len(self.rank_byte_counts)
        max_bytes = max(self.rank_byte_counts)
        avg = sum(self.rank_byte_counts) / n if n else 0
        return (
            f"RankedParamShard(rank_count={n}, "
            f"max_bytes={max_bytes:,}, avg_bytes={avg:,.0f}, "
            f"imbalance={max_bytes / avg if avg else float('nan'):.3f})"
        )


def shard_nor_muon_params(
    model: nn.Module,
    world_size: int,
) -> RankedParamShard:
    """NorMuon parameter shard optimizer balancer (Phase 1 placeholder)."""
    raise NotImplementedError_(
        "shard_nor_muon_params is a Phase 1 placeholder; the real "
        "implementation lands in Phase 3 (design §13.3, roadmap D2)."
    )


def wrap_model_with_fsdp(
    model: nn.Module,
    config: TrainingConfig,
    *,
    world_size: int | None = None,
    auto_wrap_policy: Callable[..., bool] | None = None,
    **kwargs: Any,
) -> nn.Module:
    """Wrap model module inside FullyShardedDataParallel wrapper (Phase 1 placeholder)."""
    raise NotImplementedError_(
        "wrap_model_with_fsdp is a Phase 1 placeholder; the real "
        "implementation lands in Phase 3 (design §13.1, roadmap D1)."
    )
