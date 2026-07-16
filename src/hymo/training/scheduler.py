"""Joint WSD scheduler (Phase 3 implementation).

The multiplicative factor on both optimizers' learning rates. The factor
is 0 at step 0, ramps linearly to 1.0 over the warmup phase, stays
at 1.0 over the stable phase, and decays (linear/cosine/sqrt) to
``min_lr_ratio`` over the decay phase.

Preserves ``lr_muon / lr_adamw = 66.7`` by returning a single factor.
"""

from __future__ import annotations

import math
from typing import Literal

from hymo.core.config import SchedulerConfig

__all__ = ["JointWSDScheduler", "DecaySchedule"]


DecaySchedule = Literal["linear", "cosine", "sqrt"]


class JointWSDScheduler:
    """Joint WSD scheduler (warmup-stable-decay).

    Architecture doc §5.3. Returns a multiplicative factor applied
    to both optimizers' base LRs externally.
    """

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
        """Return the multiplicative factor for the given step.

        - step < warmup_steps: linear ramp 0 → 1
        - warmup ≤ step < warmup + stable: 1.0
        - decay phase: decay 1 → min_lr_ratio
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
        """Advance the internal step counter (call after each optimizer step)."""
        self._step += 1

    def state_dict(self) -> dict[str, int]:
        return {"step": self._step}

    def load_state_dict(self, state: dict[str, int]) -> None:
        self._step = state.get("step", 0)

    @staticmethod
    def _decay_factor(progress: float, kind: DecaySchedule) -> float:
        """Pure function: given progress in [0, 1], return decay factor in
        ``[0, 1]``.

        Exposed as a public static method so tests can verify the
        decay shape without instantiating the full scheduler.
        """
        if not 0.0 <= progress <= 1.0:
            raise ValueError(f"progress must be in [0, 1], got {progress}")
        if kind == "linear":
            return 1.0 - progress
        if kind == "cosine":
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        if kind == "sqrt":
            return math.sqrt(1.0 - progress)
        raise ValueError(f"Unknown decay kind: {kind!r}")
