"""Checkpoint save/load utilities using PyTorch serialize APIs."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from hymo.training.optimizer import Optimizers
from hymo.training.scheduler import JointWSDScheduler

__all__ = [
    "save_checkpoint",
    "load_checkpoint",
    "CheckpointState",
]


@dataclass
class CheckpointState:
    """Carried metadata state for checkpoints."""

    step: int = 0
    token_count: int = 0
    best_loss: float = float("inf")
    rng_state: dict[str, Any] | None = None
    metrics_extra: dict[str, Any] | None = None


def _capture_rng_state() -> dict[str, Any]:
    """Capture current Python, NumPy, and PyTorch RNG states."""
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.random.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def _restore_rng_state(rng_state: dict[str, Any]) -> None:
    """Restore previously captured Python, NumPy, and PyTorch RNG states."""
    if "python" in rng_state:
        random.setstate(rng_state["python"])
    if "numpy" in rng_state:
        np.random.set_state(rng_state["numpy"])
    if "torch" in rng_state:
        torch.random.set_rng_state(rng_state["torch"])
    cuda_states = rng_state.get("torch_cuda", [])
    if cuda_states and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_states)


def _optimizer_state_dict(optimizers: Optimizers) -> dict[str, Any]:
    """Extract states of the dual optimizers."""
    nm_sd = optimizers.nor_muon.state_dict() if optimizers.nor_muon else None
    aw_sd = optimizers.adamw.state_dict()
    return {
        "nor_muon": nm_sd,
        "adamw": aw_sd,
        "nor_muon_lr": optimizers.nor_muon.param_groups[0]["lr"] if optimizers.nor_muon else None,
        "adamw_lr": optimizers.adamw.param_groups[0]["lr"],
    }


def _optimizer_load_state_dict(
    optimizers: Optimizers,
    state: dict[str, Any],
) -> None:
    """Load states back into the dual optimizers."""
    if optimizers.nor_muon and state.get("nor_muon") is not None:
        optimizers.nor_muon.load_state_dict(state["nor_muon"])
    if state.get("adamw") is not None:
        optimizers.adamw.load_state_dict(state["adamw"])


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizers: Optimizers,
    scheduler: JointWSDScheduler,
    state: CheckpointState,
) -> None:
    """Save model weights, optimizer states, scheduler state, and metadata to path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    rng_state = state.rng_state if state.rng_state is not None else _capture_rng_state()

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state": _optimizer_state_dict(optimizers),
        "scheduler_state": scheduler.state_dict(),
        "step": state.step,
        "token_count": state.token_count,
        "best_loss": state.best_loss,
        "rng_state": rng_state,
        "metrics_extra": state.metrics_extra or {},
        "config_json": None,
    }

    tmp_path = path.with_suffix(".tmp")
    torch.save(checkpoint, tmp_path)
    tmp_path.rename(path)


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizers: Optimizers,
    scheduler: JointWSDScheduler,
) -> CheckpointState:
    """Load checkpoint file and restore states of model, optimizers, and scheduler."""
    path = Path(path)
    if not path.exists():
        from hymo.core.exceptions import CheckpointNotFoundError
        raise CheckpointNotFoundError(f"Checkpoint not found: {path}")

    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:
        from hymo.core.exceptions import CheckpointCorruptError
        raise CheckpointCorruptError(f"Failed to load checkpoint {path}: {e}") from e

    model.load_state_dict(checkpoint["model_state_dict"])
    _optimizer_load_state_dict(optimizers, checkpoint["optimizer_state"])
    scheduler.load_state_dict(checkpoint["scheduler_state"])

    rng_state = checkpoint.get("rng_state")
    if rng_state is not None:
        _restore_rng_state(rng_state)

    return CheckpointState(
        step=checkpoint.get("step", 0),
        token_count=checkpoint.get("token_count", 0),
        best_loss=checkpoint.get("best_loss", float("inf")),
        rng_state=rng_state,
        metrics_extra=checkpoint.get("metrics_extra", None),
    )
