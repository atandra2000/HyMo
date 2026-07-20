"""Checkpoint save/load utilities using DCP for tensors and JSON for metadata."""

from __future__ import annotations

import json
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

_METADATA_FILE = "hymo_meta.json"


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
    py_state = random.getstate()  # (version: int, internalstate: tuple[int,...], gaussflag: float|None)
    return {
        "python": {"version": py_state[0], "internalstate": list(py_state[1]), "gauss": py_state[2]},
        "numpy": np.random.get_state(),
        "torch": torch.random.get_rng_state().tolist(),
        "torch_cuda": [t.tolist() for t in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else [],
    }


class _JsonEncoder(json.JSONEncoder):
    """JSON encoder that handles NumPy arrays and scalars."""
    def default(self, o: Any) -> Any:
        if isinstance(o, np.ndarray):
            return {"__ndarray__": True, "data": o.tolist(), "dtype": str(o.dtype)}
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        return super().default(o)


def _json_decode_hook(d: dict[str, Any]) -> Any:
    """JSON object hook that reconstructs NumPy arrays."""
    if d.get("__ndarray__"):
        return np.array(d["data"], dtype=d["dtype"])
    return d


def _restore_rng_state(rng_state: dict[str, Any]) -> None:
    """Restore previously captured Python, NumPy, and PyTorch RNG states."""
    if "python" in rng_state:
        py = rng_state["python"]
        # Reconstruct the 3-tuple (version, internalstate_tuple, gaussflag)
        random.setstate((py["version"], tuple(py["internalstate"]), py["gauss"]))
    if "numpy" in rng_state:
        np.random.set_state(rng_state["numpy"])
    if "torch" in rng_state:
        torch.random.set_rng_state(torch.tensor(rng_state["torch"], dtype=torch.uint8))
    cuda_states = rng_state.get("torch_cuda", [])
    if cuda_states and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([torch.tensor(s, dtype=torch.uint8) for s in cuda_states])


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
    """Save a DCP checkpoint directory with tensors + a JSON metadata sidecar."""
    import torch.distributed.checkpoint as dcp

    ckpt_dir = Path(path)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    rng_state = state.rng_state if state.rng_state is not None else _capture_rng_state()

    # --- 1. Save tensors (model, optimizers, scheduler) via DCP ---
    tensor_state: dict[str, Any] = {
        "model": model.state_dict(),
        "optimizer": _optimizer_state_dict(optimizers),
        "scheduler": scheduler.state_dict(),
    }
    dcp.save(tensor_state, checkpoint_id=str(ckpt_dir))

    # --- 2. Save scalar metadata as JSON (DCP cannot handle arbitrary Python objects) ---
    meta: dict[str, Any] = {
        "step": state.step,
        "token_count": state.token_count,
        "best_loss": state.best_loss,
        "rng_state": rng_state,
        "metrics_extra": state.metrics_extra or {},
    }
    meta_path = ckpt_dir / _METADATA_FILE
    tmp = meta_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(meta, cls=_JsonEncoder), encoding="utf-8")
    tmp.replace(meta_path)


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizers: Optimizers,
    scheduler: JointWSDScheduler,
) -> CheckpointState:
    """Load a DCP checkpoint directory and restore tensors + JSON metadata."""
    import torch.distributed.checkpoint as dcp

    ckpt_dir = Path(path)
    if not ckpt_dir.exists() or not ckpt_dir.is_dir():
        from hymo.core.exceptions import FileNotFoundError
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")

    # --- 1. Load tensors via DCP ---
    tensor_state: dict[str, Any] = {
        "model": model.state_dict(),
        "optimizer": _optimizer_state_dict(optimizers),
        "scheduler": scheduler.state_dict(),
    }
    try:
        dcp.load(tensor_state, checkpoint_id=str(ckpt_dir))
    except Exception as e:
        from hymo.core.exceptions import RuntimeError
        raise RuntimeError(f"Failed to load checkpoint {ckpt_dir}: {e}") from e

    model.load_state_dict(tensor_state["model"])
    _optimizer_load_state_dict(optimizers, tensor_state["optimizer"])
    scheduler.load_state_dict(tensor_state["scheduler"])

    # --- 2. Load scalar metadata from JSON sidecar ---
    meta_path = ckpt_dir / _METADATA_FILE
    meta: dict[str, Any] = {}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"), object_hook=_json_decode_hook)

    rng_state = meta.get("rng_state")
    if rng_state is not None:
        _restore_rng_state(rng_state)

    return CheckpointState(
        step=meta.get("step", 0),
        token_count=meta.get("token_count", 0),
        best_loss=meta.get("best_loss", float("inf")),
        rng_state=rng_state,
        metrics_extra=meta.get("metrics_extra", None),
    )
