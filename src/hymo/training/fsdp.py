"""FSDP-2 wrapper (Phase 1 placeholder).

The real implementation (architecture doc §13, roadmap D1, D2):

- Wraps the model with :class:`torch.distributed.fsdp.FullyShardedDataParallel`
  using a per-expert MoE wrapping policy (16 separate FSDP instances
  per MoE layer).
- Mixed-precision policy: ``bfloat16`` for params / reduce / buffer.
- Per-expert NorMuon sharding with sort-by-size + round-robin
  (architecture doc §13.3).

The :func:`wrap_model_with_fsdp` function signature is stable; the
body raises :class:`NotImplementedError_` in Phase 1.
"""

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
    """The FSDP auto-wrap policy for HyMo.

    Returns True iff ``module`` should be wrapped as its own FSDP
    instance. The rules (architecture doc §13.2):

    - Per-layer: every GDN block and every MLA block is wrapped.
    - Per-expert: every :class:`hymo.models.moe.SwiGLUExpert` is
      wrapped (16 experts × 8 MLA layers = 128 FSDP instances).
    - Replicated (NOT wrapped): the MoE gate, RMSNorm γ, softcap
      (which isn't a parameter).
    """
    raise NotImplementedError_(
        "fsdp_auto_wrap_policy is a Phase 1 placeholder; the real "
        "implementation lands in Phase 3 (design §13.2, roadmap D1)."
    )


class RankedParamShard:
    """The result of :func:`shard_nor_muon_params`.

    Attributes
    ----------
    rank_assignments : list[list[nn.Parameter]]
        Per-rank lists of parameters. ``rank_assignments[r]`` is the
        list of parameters assigned to rank ``r``.
    rank_byte_counts : list[int]
        Per-rank total bytes. Used for the 5% balance check.
    """

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
    """Sort-by-size + round-robin assignment of NorMuon params to ranks.

    Architecture doc §13.3. Phase 1 placeholder.

    The 5% balance invariant: ``max(rank_byte_counts) / avg < 1.05``.
    """
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
    """Wrap the model with FSDP-2.

    Phase 1 placeholder — the real implementation lands in Phase 3
    (design §13.1, roadmap D1).
    """
    raise NotImplementedError_(
        "wrap_model_with_fsdp is a Phase 1 placeholder; the real "
        "implementation lands in Phase 3 (design §13.1, roadmap D1)."
    )
