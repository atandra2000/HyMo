"""Dual optimizer: NorMuon + AdamW (Phase 1 placeholders).

The real implementation (architecture doc §5.2, roadmap C1, C2) has:

- :class:`NorMuon` — the NorMuon optimizer (arXiv 2510.05491) for
  2D dense weights. Newton-Schulz orthogonalization + row-wise RMS
  normalization. Cautious weight decay (Lion-style mask).
- :class:`CautiousAdamW` — AdamW with the same cautious mask; FP32
  master weights; β2 = 0.95 (the 2026 norm for small LLM training).

Both are subclasses of :class:`torch.optim.Optimizer`. The phase-1
placeholders are real subclasses (so the partition test can construct
them and verify the API) but ``step`` raises
:class:`hymo.core.exceptions.NotImplementedError_`.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer

from hymo.core.config import OptimizerConfig
from hymo.core.exceptions import NotImplementedError_

__all__ = ["NorMuon", "CautiousAdamW", "build_optimizers", "Optimizers"]


class NorMuon(Optimizer):
    """NorMuon optimizer (placeholder).

    Architecture doc §5.2. Phase 1 placeholder — the ``step`` method
    raises :class:`NotImplementedError_`.

    Defaults: lr=0.02, momentum=0.95, betas=(0.95, 0.95), eps=1e-8,
    weight_decay=0.1, cautious_wd=True.
    """

    def __init__(
        self,
        params: Iterable[nn.Parameter],
        lr: float = 0.02,
        momentum: float = 0.95,
        betas: tuple[float, float] = (0.95, 0.95),
        eps: float = 1e-8,
        weight_decay: float = 0.1,
        cautious_wd: bool = True,
    ) -> None:
        if lr <= 0:
            raise ValueError(f"lr must be > 0, got {lr}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"momentum must be in [0, 1), got {momentum}")
        defaults = {
            "lr": lr,
            "momentum": momentum,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
            "cautious_wd": cautious_wd,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Any = None) -> None:  # type: ignore[override]
        raise NotImplementedError_(
            "NorMuon.step is a Phase 1 placeholder; the real "
            "implementation lands in Phase 3 (design §5.2, roadmap C2)."
        )


class CautiousAdamW(Optimizer):
    """AdamW with cautious weight decay and FP32 master weights (placeholder).

    Architecture doc §5.2. Phase 1 placeholder — the ``step`` method
    raises :class:`NotImplementedError_`.

    Defaults: lr=3e-4, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0
    (most params; embed/head use 0.1), cautious_wd=False.
    """

    def __init__(
        self,
        params: Iterable[nn.Parameter],
        lr: float = 3e-4,
        betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        cautious_wd: bool = False,
    ) -> None:
        if lr <= 0:
            raise ValueError(f"lr must be > 0, got {lr}")
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
            "cautious_wd": cautious_wd,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Any = None) -> None:  # type: ignore[override]
        raise NotImplementedError_(
            "CautiousAdamW.step is a Phase 1 placeholder; the real "
            "implementation lands in Phase 3 (design §5.2, roadmap C2)."
        )


# ----------------------------------------------------------------------
# Builder
# ----------------------------------------------------------------------


class Optimizers:
    """Container for the two optimizers + the WSD scheduler.

    Attributes
    ----------
    nor_muon : NorMuon or None
        The NorMuon optimizer, or None if no 2D dense params.
    adamw : CautiousAdamW
        The AdamW optimizer.
    """

    __slots__ = ("nor_muon", "adamw")

    def __init__(self, nor_muon: NorMuon | None, adamw: CautiousAdamW) -> None:
        self.nor_muon = nor_muon
        self.adamw = adamw

    def state_dict(self) -> dict[str, Any]:
        return {
            "nor_muon": self.nor_muon.state_dict() if self.nor_muon else None,
            "adamw": self.adamw.state_dict(),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        if self.nor_muon and state_dict.get("nor_muon") is not None:
            self.nor_muon.load_state_dict(state_dict["nor_muon"])
        self.adamw.load_state_dict(state_dict["adamw"])


def build_optimizers(
    model: nn.Module,
    config: OptimizerConfig,
) -> Optimizers:
    """Build the dual optimizer pair from a partitioned model.

    Phase 1 placeholder — partitions correctly but the optimizer
    classes themselves are placeholders.
    """
    from hymo.training.partition import partition_parameters

    partition = partition_parameters(model)
    nor_muon = (
        NorMuon(
            partition.nor_muon,
            lr=config.muon_lr,
            momentum=config.muon_momentum,
            betas=config.muon_betas,
            eps=config.muon_eps,
            weight_decay=config.muon_weight_decay,
            cautious_wd=config.cautious_wd,
        )
        if partition.nor_muon
        else None
    )
    adamw = CautiousAdamW(
        partition.adamw,
        lr=config.adamw_lr,
        betas=config.adamw_betas,
        eps=config.adamw_eps,
        weight_decay=config.adamw_weight_decay,
        cautious_wd=config.cautious_wd,
    )
    return Optimizers(nor_muon=nor_muon, adamw=adamw)
