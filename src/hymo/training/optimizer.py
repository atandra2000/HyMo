"""Dual optimizer: NorMuon + AdamW (Phase 3 implementation)."""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer

from hymo.core.config import OptimizerConfig

__all__ = ["NorMuon", "CautiousAdamW", "build_optimizers", "Optimizers"]


@torch.no_grad()
def _newton_schulz_orthogonalize(
    g: torch.Tensor, iterations: int = 5
) -> torch.Tensor:
    """Newton-Schulz iterative orthogonalization to approximate matrix sign function."""
    norm = g.norm()
    if norm < 1e-12:
        return g
    g = g / norm
    for _ in range(iterations):
        g = 1.5 * g - 0.5 * g @ g.T @ g
    return g * norm  # type: ignore[no-any-return]


class NorMuon(Optimizer):
    """NorMuon optimizer with FP32 master weights (architecture doc §5.2)."""

    def __init__(
        self,
        params: Iterable[nn.Parameter],
        lr: float = 0.02,
        momentum: float = 0.95,
        betas: tuple[float, float] = (0.95, 0.95),
        eps: float = 1e-8,
        weight_decay: float = 0.1,
        cautious_wd: bool = True,
        ns_iterations: int = 5,
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
            "ns_iterations": ns_iterations,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Any = None) -> torch.Tensor | None:  # type: ignore[override]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta = group["betas"][0]
            eps = group["eps"]
            wd = group["weight_decay"]
            cautious = group["cautious_wd"]
            ns_iter = group["ns_iterations"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad

                state = self.state[p]

                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(p)
                    state["master_weight"] = p.data.clone()

                buf = state["momentum_buffer"]
                master = state["master_weight"]

                if wd != 0:
                    if cautious:
                        mask = (grad * master) > 0
                        master.mul_(1.0 - lr * wd * mask.to(master.dtype))
                    else:
                        master.mul_(1.0 - lr * wd)

                buf.mul_(beta).add_(grad, alpha=1.0 - beta)

                update = _newton_schulz_orthogonalize(buf, ns_iter)

                row_norms = update.norm(dim=1, keepdim=True)
                rms = row_norms / math.sqrt(update.size(1))
                scale = rms.clamp(min=eps)
                update = update / scale

                master.add_(update, alpha=-lr)
                p.data.copy_(master.to(p.dtype))
        return loss


class CautiousAdamW(Optimizer):
    """AdamW with cautious weight decay and FP32 master weights (architecture doc §5.2)."""

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
    def step(self, closure: Any = None) -> torch.Tensor | None:  # type: ignore[override]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]
            cautious = group["cautious_wd"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad

                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                    state["master_weight"] = p.data.clone()

                state["step"] += 1
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                master = state["master_weight"]
                step_t = state["step"]

                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                bias_corr1 = 1.0 - beta1 ** step_t
                bias_corr2 = 1.0 - beta2 ** step_t

                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_corr2)).add_(eps)

                if wd != 0:
                    if cautious:
                        mask = (grad * master) > 0
                        master.mul_(1.0 - lr * wd * mask.to(master.dtype))
                    else:
                        master.mul_(1.0 - lr * wd)

                step_size = lr / bias_corr1
                master.addcdiv_(exp_avg, denom, value=-step_size)
                p.data.copy_(master.to(p.dtype))
        return loss


class Optimizers:
    """Container for the dual optimizers."""

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
    """Build the dual optimizer pair from a partitioned model."""
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
