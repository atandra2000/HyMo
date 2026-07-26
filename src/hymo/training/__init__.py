"""Public API of :mod:`hymo.training`."""

from __future__ import annotations

from hymo.training.checkpoint import (
    CheckpointState,
    load_checkpoint,
    save_checkpoint,
)
from hymo.training.fsdp import (
    fsdp_auto_wrap_policy,
    wrap_model_with_fsdp,
)
from hymo.training.optimizer import (
    CautiousAdamW,
    NorMuon,
    Optimizers,
    build_optimizers,
)
from hymo.training.partition import (
    ParameterPartition,
    goes_to_adamw,
    partition_parameters,
)
from hymo.training.scheduler import DecaySchedule, JointWSDScheduler
from hymo.training.trainer import Trainer, train_step_result
from hymo.training.validation import (
    ValMetrics,
    compute_validation_loss,
    get_val_batch,
)

__all__ = [
    # Checkpoint
    "CheckpointState",
    "load_checkpoint",
    "save_checkpoint",
    # FSDP
    "fsdp_auto_wrap_policy",
    "wrap_model_with_fsdp",
    # Optimizer
    "CautiousAdamW",
    "NorMuon",
    "Optimizers",
    "build_optimizers",
    # Partition
    "ParameterPartition",
    "goes_to_adamw",
    "partition_parameters",
    # Scheduler
    "DecaySchedule",
    "JointWSDScheduler",
    # Trainer
    "Trainer",
    "train_step_result",
    # Validation
    "ValMetrics",
    "compute_validation_loss",
    "get_val_batch",
]
