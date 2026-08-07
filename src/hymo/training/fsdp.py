"""FSDP-2 integration helpers for layer-wise model sharding."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from torch import nn

from hymo.core.config import TrainingConfig

__all__ = [
    "wrap_model_with_fsdp",
    "fsdp_auto_wrap_policy",
]   # ponytail: speculative NorMuon parameter-balancer removed; FSDP does the sharding.


def fsdp_auto_wrap_policy(module: nn.Module, recurse: bool, non_blocking: bool) -> bool:
    """Select complete GDN and MLA blocks as FSDP wrapping units."""
    from hymo.models.gdn import GatedDeltaNetBlock
    from hymo.models.mla import MLABlock
    return isinstance(module, (GatedDeltaNetBlock, MLABlock))


def wrap_model_with_fsdp(
    model: nn.Module,
    config: TrainingConfig,
    *,
    world_size: int | None = None,
    auto_wrap_policy: Callable[..., bool] | None = None,
    **kwargs: Any,
) -> nn.Module:
    """Wrap a model with FSDP using configured mixed-precision dtypes.

    If FSDP is unavailable, return the original module so CPU-only tooling can
    still construct and inspect a model.
    """
    try:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp import MixedPrecision
    except ImportError:
        return model

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
