"""Joint WSD scheduler (Phase 1 placeholder).

The real implementation (architecture doc §5.3, roadmap C3) is a
multiplicative factor on both optimizers' learning rates. The factor
is 0 at step 0, ramps linearly to 1.0 over the warmup phase, stays
at 1.0 over the stable phase, and decays (linearly, cosine, or sqrt)
to ``min_lr_ratio`` over the decay phase.

The scheduler preserves the ``lr_muon / lr_adamw = 66.7`` ratio by
returning a single multiplicative factor; both optimizers apply it to
their base LR externally.

Phase 1 placeholder: the :class:`JointWSDScheduler` class has the
right API and computes the warmup / stable / decay step counts from
the :class:`SchedulerConfig`. The :meth:`get_factor` body raises
:class:`NotImplementedError_`.
"""

from __future__ import annotations

import math
from typing import Literal

from hymo.core.config import SchedulerConfig
from hymo.core.exceptions import NotImplementedError_

__all__ = ["JointWSDScheduler", "DecaySchedule"]


DecaySchedule = Literal["linear", "cosine", "sqrt"]


class JointWSDScheduler:
    """Joint WSD scheduler (warmup-stable-decay).

    Architecture doc §5.3. The multiplicative factor returned by
    :meth:`get_factor` is applied to both optimizers' base LRs
    externally, preserving the ``lr_muon / lr_adamw`` ratio.

    Parameters
    ----------
    config : SchedulerConfig
    """

    def __init__(self, config: SchedulerConfig) -> None:
        self._config = config
        self.warmup_steps = config.warmup_steps
        self.stable_steps = config.stable_steps
        self.decay_steps = config.decay_steps
        self.min_lr_ratio = config.min_lr_ratio
        self.decay_kind: DecaySchedule = config.decay  # type: ignore[assignment]

    @property
    def config(self) -> SchedulerConfig:
        return self._config

    def get_factor(self, step: int) -> float:
        """Return the multiplicative factor for the given step.

        Phase 1 placeholder — raises :class:`NotImplementedError_`.
        The real implementation:
        - step < warmup_steps: linear ramp 0 → 1
        - warmup_steps ≤ step < warmup + stable: return 1.0
        - decay phase: linear / cosine / sqrt decay 1 → min_lr_ratio
        """
        raise NotImplementedError_(
            "JointWSDScheduler.get_factor is a Phase 1 placeholder; "
            "the real implementation lands in Phase 3 (design §5.3, "
            "roadmap C3)."
        )

    def state_dict(self) -> dict[str, int]:
        return {"step": 0}  # placeholder; the real impl tracks step.

    def load_state_dict(self, state: dict[str, int]) -> None:
        # Placeholder: accept the state without effect.
        return None

    @staticmethod
    def _decay_factor(progress: float, kind: DecaySchedule) -> float:
        """Pure function: given progress in [0, 1], return decay factor in
        ``[min_lr_ratio, 1]``.

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
