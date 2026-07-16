"""The HyMo training loop (Phase 1 placeholder).

The real implementation (architecture doc §7, roadmap C4, D5, D6, D7,
E3, G2) ties together:

- :class:`hymo.training.optimizer.build_optimizers` (NorMuon + AdamW).
- :class:`hymo.training.scheduler.JointWSDScheduler`.
- :class:`hymo.training.fsdp.wrap_model_with_fsdp` for FSDP-2 wrapping.
- :class:`hymo.training.checkpoint.save_checkpoint` /
  :func:`load_checkpoint` for DCP-based save/load.
- :class:`hymo.training.validation.compute_validation_loss` for
  real held-out val.
- :class:`hymo.utils.callbacks.CallbackList` for the event hook.

This placeholder defines the public surface (``Trainer.__init__``,
``train_step``, ``save``, ``load``, ``train``); the bodies raise
:class:`NotImplementedError_`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import nn

from hymo.core.config import HyMoConfig
from hymo.core.exceptions import NotImplementedError_
from hymo.utils.callbacks import CallbackList, TrainerState

__all__ = ["Trainer", "TrainerConfig", "train_step_result"]


@dataclass
class TrainerConfig:
    """Trainer-only config knobs (subset of :class:`TrainingConfig`).

    These are the knobs the Trainer reads directly at runtime. Most
    training config lives in :class:`hymo.core.config.TrainingConfig`
    and is passed in via the :class:`HyMoConfig`.
    """

    log_interval: int = 50
    save_interval: int = 4_000
    eval_interval: int = 2_000
    grad_clip: float = 1.0
    grad_norm_threshold: float = 10.0
    loss_nan_skip: bool = True
    consecutive_nan_limit: int = 5
    max_keep: int = 2


@dataclass
class train_step_result:
    """The result of a single :meth:`Trainer.train_step` call.

    Attributes
    ----------
    loss : float
        The cross-entropy loss for this step (after MTP contribution).
    grad_norm : float
        The L2 norm of the gradients (after clip).
    lr_muon : float
        The current NorMuon learning rate.
    lr_adamw : float
        The current AdamW learning rate.
    skipped : bool
        True if the step was skipped (NaN-skip).
    metrics : dict
        Free-form dict for additional metrics.
    """

    loss: float
    grad_norm: float
    lr_muon: float
    lr_adamw: float
    skipped: bool = False
    metrics: dict[str, float] = field(default_factory=dict)


class Trainer:
    """The main HyMo training loop (Phase 1 placeholder).

    Architecture doc §7, §13. Phase 1 placeholder.

    Parameters
    ----------
    config : HyMoConfig
        The top-level config. The Trainer reads every sub-config.
    model : nn.Module
        The HyMo model (already constructed and μP-init'd).
    callbacks : CallbackList or None
        Optional callback list.
    """

    def __init__(
        self,
        config: HyMoConfig,
        model: nn.Module,
        callbacks: CallbackList | None = None,
    ) -> None:
        self._config = config
        self.model = model
        # Explicit None check: CallbackList implements __len__ and an empty
        # instance is falsy, so ``callbacks or CallbackList()`` would
        # silently replace a real (empty) CallbackList with a fresh one.
        self.callbacks = callbacks if callbacks is not None else CallbackList()
        # Public state — the callbacks read this via the TrainerState.
        self.step: int = 0
        self.token_count: int = 0
        self.best_loss: float = float("inf")
        self.state = TrainerState()

    # ---- Public API -----------------------------------------------------

    def train_step(
        self,
        tokens: torch.Tensor,
        targets: torch.Tensor,
    ) -> train_step_result:
        """Run a single optimizer step.

        Phase 1 placeholder — raises :class:`NotImplementedError_`.
        The real implementation runs the forward, backward, optimizer
        step, scheduler step, and returns the metrics.
        """
        raise NotImplementedError_(
            "Trainer.train_step is a Phase 1 placeholder; the real "
            "implementation lands in Phase 3 (design §7, roadmap C4)."
        )

    def save(self, tag: str | None = None) -> Path:
        """Save a checkpoint.

        Phase 1 placeholder — raises :class:`NotImplementedError_`.
        The real implementation writes to
        ``{output_dir}/{tag}/`` via DCP (roadmap D3, D7).
        """
        raise NotImplementedError_(
            "Trainer.save is a Phase 1 placeholder; the real "
            "implementation lands in Phase 3 (design §13.6, roadmap D3)."
        )

    def load(self, path: str | Path) -> int:
        """Load a checkpoint and resume.

        Phase 1 placeholder — raises :class:`NotImplementedError_`.
        Returns the step count to resume from.
        """
        raise NotImplementedError_(
            "Trainer.load is a Phase 1 placeholder; the real "
            "implementation lands in Phase 3 (design §13.6, roadmap D7)."
        )

    def train(self, max_steps: int | None = None) -> None:
        """Run the main training loop.

        Phase 1 placeholder — raises :class:`NotImplementedError_`.
        """
        raise NotImplementedError_(
            "Trainer.train is a Phase 1 placeholder; the real "
            "implementation lands in Phase 3 (design §7, roadmap C4)."
        )

    def evaluate(self) -> dict[str, float]:
        """Run a single validation pass.

        Phase 1 placeholder — raises :class:`NotImplementedError_`.
        """
        raise NotImplementedError_(
            "Trainer.evaluate is a Phase 1 placeholder; the real "
            "implementation lands in Phase 3 (design §6.3, roadmap E3)."
        )

    def _make_state(self) -> TrainerState:
        """Build a fresh :class:`TrainerState` for callback dispatch."""
        return TrainerState(
            step=self.step,
            token_count=self.token_count,
            metrics={},
        )
