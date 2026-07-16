"""DCP-based checkpoint save/load (Phase 1 placeholder).

The real implementation (architecture doc §13.6, roadmap D3, D7) uses
``torch.distributed.checkpoint`` to save FSDP-2 sharded state. Each
rank writes its own shard; the load reassembles them.

This placeholder defines the public surface; the body raises
:class:`NotImplementedError_`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from torch import nn

from hymo.core.exceptions import NotImplementedError_
from hymo.training.optimizer import Optimizers
from hymo.training.scheduler import JointWSDScheduler

__all__ = [
    "save_checkpoint",
    "load_checkpoint",
    "CheckpointState",
]


@dataclass
class CheckpointState:
    """The state carried by every checkpoint.

    Attributes
    ----------
    step : int
        Global optimizer step.
    token_count : int
        Trained tokens (resumed for metrics continuity).
    best_loss : float
        Best validation loss seen so far.
    rng_state : dict
        Per-rank RNG state (CPU + CUDA).
    metrics_extra : dict
        Free-form dict for forward-compat (e.g. EMA bias stats, expert
        load entropy).
    """

    step: int = 0
    token_count: int = 0
    best_loss: float = float("inf")
    rng_state: dict[str, Any] | None = None
    metrics_extra: dict[str, Any] | None = None


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizers: Optimizers,
    scheduler: JointWSDScheduler,
    state: CheckpointState,
) -> None:
    """Save an FSDP-2-aware checkpoint to ``path`` via DCP.

    Phase 1 placeholder — raises :class:`NotImplementedError_`.
    The real implementation uses ``torch.distributed.checkpoint.save``
    so per-rank sharding is automatic.
    """
    raise NotImplementedError_(
        "save_checkpoint is a Phase 1 placeholder; the real "
        "implementation lands in Phase 3 (design §13.6, roadmap D3)."
    )


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizers: Optimizers,
    scheduler: JointWSDScheduler,
) -> CheckpointState:
    """Load an FSDP-2-aware checkpoint from ``path`` via DCP.

    Phase 1 placeholder — raises :class:`NotImplementedError_`.
    """
    raise NotImplementedError_(
        "load_checkpoint is a Phase 1 placeholder; the real "
        "implementation lands in Phase 3 (design §13.6, roadmap D3)."
    )
