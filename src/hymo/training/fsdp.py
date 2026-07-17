"""FSDP-2 wrapper placeholders for Phase 1/Phase 3."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from torch import nn

from hymo.core.config import TrainingConfig

__all__ = [
    "wrap_model_with_fsdp",
    "fsdp_auto_wrap_policy",
    "shard_nor_muon_params",
    "RankedParamShard",
]


def fsdp_auto_wrap_policy(module: nn.Module, recurse: bool, non_blocking: bool) -> bool:
    """FSDP auto-wrap policy function."""
    from hymo.models.gdn import GatedDeltaNetBlock
    from hymo.models.mla import MLABlock
    return isinstance(module, (GatedDeltaNetBlock, MLABlock))


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
    """NorMuon parameter shard optimizer balancer."""
    # Collect all parameters requiring grad that are suitable for NorMuon (e.g. 2D matrices)
    params = [p for p in model.parameters() if p.requires_grad and p.ndim == 2]
    
    # Sort by size descending
    params.sort(key=lambda p: p.numel(), reverse=True)
    
    rank_assignments: list[list[nn.Parameter]] = [[] for _ in range(world_size)]
    rank_byte_counts = [0 for _ in range(world_size)]
    
    for p in params:
        # Find rank with minimum bytes
        min_rank = min(range(world_size), key=lambda r: rank_byte_counts[r])
        rank_assignments[min_rank].append(p)
        rank_byte_counts[min_rank] += p.numel() * p.element_size()
        
    return RankedParamShard(rank_assignments, rank_byte_counts)


def wrap_model_with_fsdp(
    model: nn.Module,
    config: TrainingConfig,
    *,
    world_size: int | None = None,
    auto_wrap_policy: Callable[..., bool] | None = None,
    **kwargs: Any,
) -> nn.Module:
    """Wrap model module inside FullyShardedDataParallel wrapper."""
    try:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp import MixedPrecision
    except ImportError:
        return model  # Fallback if distributed is not available

    if auto_wrap_policy is None:
        auto_wrap_policy = fsdp_auto_wrap_policy

    mp_dtype = torch.float32
    if config.fsdp_mixed_precision == "bfloat16":
        mp_dtype = torch.bfloat16
    elif config.fsdp_mixed_precision == "float16":
        mp_dtype = torch.float16
        
    mixed_precision = MixedPrecision(
        param_dtype=mp_dtype,
        reduce_dtype=mp_dtype,
        buffer_dtype=mp_dtype,
    )
    
    device_id = torch.cuda.current_device() if torch.cuda.is_available() else None
    
    return FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=mixed_precision,
        device_id=device_id,
        **kwargs,
    )
