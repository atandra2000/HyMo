"""Joint WSD scheduler (Phase 3 implementation)."""

from __future__ import annotations

import math
from typing import Literal

from hymo.core.config import SchedulerConfig

__all__ = ["JointWSDScheduler", "DecaySchedule"]

DecaySchedule = Literal["linear", "cosine", "sqrt"]


class JointWSDScheduler:
    """Piecewise scheduler that warms up, holds, then decays the base LR."""

    def __init__(self, config: SchedulerConfig) -> None:
        self._config = config
        self.warmup_steps = config.warmup_steps
        self.stable_steps = config.stable_steps
        self.decay_steps = config.decay_steps
        self.min_lr_ratio = config.min_lr_ratio
        self.decay_kind: DecaySchedule = config.decay  # type: ignore[assignment]
        self._step: int = 0

    @property
    def config(self) -> SchedulerConfig:
        return self._config

    def get_factor(self, step: int) -> float:
        """Return the multiplicative factor for a zero-based optimizer step.

        Steps after the configured schedule remain at the minimum ratio rather
        than continuing below the requested floor.
        """
        if step < self.warmup_steps:
            return step / max(self.warmup_steps, 1)

        stable_end = self.warmup_steps + self.stable_steps
        if step < stable_end:
            return 1.0

        decay_progress = (step - stable_end) / max(self.decay_steps, 1)
        decay_progress = min(decay_progress, 1.0)

        decay_val = self._decay_factor(decay_progress, self.decay_kind)
        return self.min_lr_ratio + (1.0 - self.min_lr_ratio) * decay_val

    def step(self) -> None:
        """Advance the serialized optimizer-step counter by one."""
        self._step += 1

    def state_dict(self) -> dict[str, int]:
        return {"step": self._step}

    def load_state_dict(self, state: dict[str, int]) -> None:
        self._step = state.get("step", 0)

    @staticmethod
    def _decay_factor(progress: float, kind: DecaySchedule) -> float:
        """Map normalized decay progress to the selected curve."""
        if not 0.0 <= progress <= 1.0:
            raise ValueError(f"progress must be in [0, 1], got {progress}")
        if kind == "linear":
            return 1.0 - progress
        if kind == "cosine":
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        if kind == "sqrt":
            return math.sqrt(1.0 - progress)
        raise ValueError(f"Unknown decay kind: {kind!r}")
