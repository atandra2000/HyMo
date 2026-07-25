"""A100 pre-flight gate for the 30B-token pretraining run.

Loads the production HyMo v1.0 config, builds the model on the active CUDA
device, runs one micro-batch through forward + backward + optimizer step, and
fails loudly (exit 1) on any of:
    - OOM (CUDA out of memory)
    - NaN / Inf in forward output, parameter gradients, or optimizer step
    - Triton kernel "OutOfResources" (shared memory exceeded)
    - torch.compile / dynamo failure
    - model-to-device transfer failure

Run on the GPU pod before kicking off the production run:

    python tools/a100_smoke.py [--config configs/hymo_750m.yaml]

Exits 0 on success.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Sandbox-safe: never import wandb in smoke
os.environ.setdefault("WANDB_MODE", "disabled")

import torch
from torch.nn import functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from hymo.core.config import load_config  # noqa: E402
from hymo.models import HyMo  # noqa: E402
from hymo.training import Trainer  # noqa: E402


def _ok(msg: str) -> None:
    print(f"  [OK]   {msg}", flush=True)


def _fail(msg: str) -> None:
    print(f"  [FAIL] {msg}", flush=True)


def _section(title: str) -> None:
    print(f"\n=== {title} ===", flush=True)


def smoke(config_path: Path, micro_batch: int, seq_len: int) -> int:
    print(f"HyMo A100 smoke gate")
    print(f"  config:     {config_path}")
    print(f"  device:     {torch.cuda.get_device_name(0)}")
    print(
        f"  capability: sm_{torch.cuda.get_device_capability(0)[0]}"
        f"{torch.cuda.get_device_capability(0)[1]}"
    )

    if not torch.cuda.is_available():
        _fail("CUDA is not available")
        return 1

    _section("Config load")
    t0 = time.time()
    try:
        config = load_config(str(config_path))
    except Exception as e:
        _fail(f"failed to load config: {e}")
        return 1
    _ok(f"loaded in {time.time() - t0:.2f}s")

    _section("Model build (on CPU first to catch shape errors fast)")
    t0 = time.time()
    try:
        model = HyMo(config.model)
    except Exception as e:
        _fail(f"model construction failed: {e}")
        return 1
    n_params = model.num_parameters()
    n_train = model.num_parameters(only_trainable=True)
    _ok(
        f"built in {time.time() - t0:.2f}s — "
        f"{n_params / 1e9:.2f}B total params, {n_train / 1e9:.2f}B trainable"
    )

    _section("Move to CUDA")
    t0 = time.time()
    try:
        model = model.to("cuda")
    except torch.cuda.OutOfMemoryError as e:
        _fail(f"OOM moving model to CUDA: {e}")
        return 1
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            _fail(f"OOM moving model to CUDA: {e}")
            return 1
        raise
    _ok(f"moved in {time.time() - t0:.2f}s, peak alloc {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

    _section("Build trainer + optimizers")
    try:
        trainer = Trainer(config, model)
    except Exception as e:
        _fail(f"trainer construction failed: {e}")
        return 1
    _ok("trainer built (dual optimizer, WSD scheduler)")

    _section("Synthetic batch (one micro-batch, no real data needed)")
    vocab = config.model.vocab_size
    torch.manual_seed(0)
    try:
        tokens = torch.randint(0, vocab, (micro_batch, seq_len), device="cuda")
        targets = torch.randint(0, vocab, (micro_batch, seq_len), device="cuda")
    except torch.cuda.OutOfMemoryError as e:
        _fail(f"OOM allocating batch: {e}")
        return 1
    _ok(
        f"tokens={tuple(tokens.shape)} targets={tuple(targets.shape)} "
        f"on {tokens.device}"
    )

    _section("Forward + backward + optimizer step")
    t0 = time.time()
    try:
        result = trainer.train_step(tokens, targets)
    except torch.cuda.OutOfMemoryError as e:
        _fail(f"OOM during train_step: {e}")
        return 1
    except RuntimeError as e:
        msg = str(e).lower()
        if "out of memory" in msg:
            _fail(f"OOM during train_step: {e}")
            return 1
        if "outofresources" in msg or "shared memory" in msg:
            _fail(f"Triton OOR (shared memory exceeded): {e}")
            return 1
        _fail(f"train_step raised RuntimeError: {e}")
        return 1
    except Exception as e:
        _fail(f"train_step raised {type(e).__name__}: {e}")
        return 1
    elapsed = time.time() - t0

    if result.skipped:
        _fail("train_step was skipped (NaN-skip path triggered)")
        return 1
    if not torch.isfinite(torch.tensor(result.loss)):
        _fail(f"loss is not finite: {result.loss}")
        return 1
    if not torch.isfinite(torch.tensor(result.grad_norm)):
        _fail(f"grad_norm is not finite: {result.grad_norm}")
        return 1
    _ok(
        f"step in {elapsed:.2f}s — loss={result.loss:.4f} "
        f"grad_norm={result.grad_norm:.4f} trainer.step={trainer.step}"
    )

    _section("Checkpoint save (DCP)")
    try:
        ckpt_dir = trainer.save(tag="smoke")
    except Exception as e:
        _fail(f"checkpoint save failed: {e}")
        return 1
    if not ckpt_dir.exists() or not ckpt_dir.is_dir():
        _fail(f"checkpoint directory not created at {ckpt_dir}")
        return 1
    _ok(f"saved to {ckpt_dir}")

    _section("Smoke gate PASSED")
    print(
        f"  Peak GPU memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB\n"
        f"  Total time:      {elapsed:.2f}s for one train step\n"
        f"  All forward / backward / optimizer / checkpoint paths green.",
        flush=True,
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="HyMo A100 pre-flight gate")
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "hymo_750m.yaml",
        help="Path to a HyMo YAML config (default: production v1.0)",
    )
    parser.add_argument(
        "--micro-batch",
        type=int,
        default=1,
        help="Micro-batch size for the smoke step (default: 1)",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=512,
        help="Sequence length for the smoke step (default: 512)",
    )
    args = parser.parse_args()
    if not args.config.exists():
        print(f"config not found: {args.config}", file=sys.stderr)
        return 1
    return smoke(args.config, args.micro_batch, args.seq_len)


if __name__ == "__main__":
    raise SystemExit(main())
