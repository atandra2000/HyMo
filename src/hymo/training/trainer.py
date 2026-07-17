"""The HyMo training loop (Phase 3 implementation)."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch.nn import functional as F

from hymo.core.config import HyMoConfig
from hymo.models import HyMo
from hymo.training.checkpoint import (
    CheckpointState,
    load_checkpoint,
    save_checkpoint,
)
from hymo.training.optimizer import build_optimizers
from hymo.training.scheduler import JointWSDScheduler
from hymo.training.validation import (
    ValMetrics,
    compute_validation_loss,
)

__all__ = ["Trainer", "TrainerConfig", "train_step_result"]

log = logging.getLogger(__name__)


@dataclass
class TrainerConfig:
    """Trainer-only configuration settings."""

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
    """The result metrics of a single training step."""

    loss: float
    grad_norm: float
    lr_muon: float
    lr_adamw: float
    skipped: bool = False
    metrics: dict[str, float] = field(default_factory=dict)


class Trainer:
    """Trainer manager class for the HyMo pre-training loop."""

    def __init__(
        self,
        config: HyMoConfig,
        model: HyMo,
    ) -> None:
        self._config = config
        self.model = model

        self.optimizers = build_optimizers(model, config.optimizer)
        self.scheduler = JointWSDScheduler(config.scheduler)

        self._base_lr_muon: float | None = (
            config.optimizer.muon_lr if self.optimizers.nor_muon else None
        )
        self._base_lr_adamw: float = config.optimizer.adamw_lr

        self.step: int = 0
        self.token_count: int = 0
        self.best_loss: float = float("inf")

        if config.model.mtp_depth > 0:
            self._has_mtp = True
        else:
            self._has_mtp = False

    def train_step(
        self,
        tokens: torch.Tensor,
        targets: torch.Tensor,
    ) -> train_step_result:
        """Run a single forward-backward pass, optimizer step, and scheduler step."""
        self.model.train()

        if self._has_mtp:
            mtp_module = getattr(self.model, "_mtp", None)
            if mtp_module is not None:
                logits, mtp_outputs = mtp_module.forward(tokens)
            else:
                logits = self.model.forward(tokens)
                mtp_outputs = []
        else:
            logits = self.model.forward(tokens)
            mtp_outputs = []

        V = logits.size(-1)

        main_loss = F.cross_entropy(
            logits[:, :-1, :].reshape(-1, V),
            targets[:, :-1].reshape(-1),
        )

        total_loss = main_loss
        mtp_details: dict[str, float] = {}

        for i, mtp_out in enumerate(mtp_outputs):
            mtp_loss = F.cross_entropy(
                mtp_out.logits.reshape(-1, V),
                mtp_out.targets.reshape(-1),
            )
            weighted = mtp_loss * mtp_out.loss_weight
            total_loss = total_loss + weighted
            mtp_details[f"mtp_{i}_loss"] = mtp_loss.item()
            mtp_details[f"mtp_{i}_weighted"] = weighted.item()

        if self._config.training.loss_nan_skip and (torch.isnan(total_loss) or torch.isinf(total_loss)):
            self.model.zero_grad(set_to_none=True)
            return train_step_result(
                loss=float("nan"),
                grad_norm=0.0,
                lr_muon=self._current_lr_muon(),
                lr_adamw=self._current_lr_adamw(),
                skipped=True,
                metrics={"main_loss": float("nan"), **mtp_details},
            )

        total_loss.backward()  # type: ignore[no-untyped-call]

        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            max_norm=self._config.training.grad_clip,
            norm_type=2.0,
        )
        grad_norm_val = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else float(grad_norm)

        factor = self.scheduler.get_factor(self.step + 1)
        if self.optimizers.nor_muon is not None and self._base_lr_muon is not None:
            for g in self.optimizers.nor_muon.param_groups:
                g["lr"] = self._base_lr_muon * factor
        for g in self.optimizers.adamw.param_groups:
            g["lr"] = self._base_lr_adamw * factor

        if self.optimizers.nor_muon is not None:
            self.optimizers.nor_muon.step()
        self.optimizers.adamw.step()

        self.scheduler.step()
        self.model.zero_grad(set_to_none=True)

        self.step += 1
        self.token_count += tokens.numel()

        return train_step_result(
            loss=total_loss.item(),
            grad_norm=grad_norm_val,
            lr_muon=self._current_lr_muon(),
            lr_adamw=self._current_lr_adamw(),
            skipped=False,
            metrics={"main_loss": main_loss.item(), **mtp_details},
        )

    def save(self, tag: str | None = None) -> Path:
        """Save a training checkpoint file to output directory."""
        if tag is None:
            tag = f"step_{self.step}"
        output_dir = Path(self._config.run.output_dir)
        ckpt_dir = output_dir / tag
        ckpt_path = ckpt_dir / "model.pt"

        state = CheckpointState(
            step=self.step,
            token_count=self.token_count,
            best_loss=self.best_loss,
        )

        save_checkpoint(
            path=ckpt_path,
            model=self.model,
            optimizers=self.optimizers,
            scheduler=self.scheduler,
            state=state,
        )

        log.info("Checkpoint saved to %s (step=%d)", ckpt_path, self.step)
        return ckpt_path

    def load(self, path: str | Path) -> int:
        """Load a checkpoint file to resume training state."""
        p = Path(path)
        if p.is_dir():
            p = p / "model.pt"

        state = load_checkpoint(
            path=p,
            model=self.model,
            optimizers=self.optimizers,
            scheduler=self.scheduler,
        )

        self.step = state.step
        self.token_count = state.token_count
        self.best_loss = state.best_loss

        log.info("Checkpoint loaded from %s (step=%d)", p, self.step)
        return self.step

    def train(
        self, data_iter: Iterable[tuple[torch.Tensor, torch.Tensor]], max_steps: int | None = None
    ) -> None:
        """Execute the primary training loop iteration."""
        if max_steps is None:
            max_steps = self._config.scheduler.total_steps

        for tokens, targets in data_iter:
            if self.step >= max_steps:
                break

            result = self.train_step(tokens, targets)

            if (
                self.step % self._config.training.log_interval == 0
                and not result.skipped
            ):
                log.info(
                    "step=%d loss=%.4f grad_norm=%.4f lr_muon=%.6f lr_adamw=%.6f",
                    self.step, result.loss, result.grad_norm,
                    result.lr_muon, result.lr_adamw,
                )

            if (
                self._config.training.eval_interval > 0
                and self.step % self._config.training.eval_interval == 0
            ):
                eval_metrics = self.evaluate()
                if eval_metrics.get("val_loss", float("inf")) < self.best_loss:
                    self.best_loss = eval_metrics["val_loss"]
                    self.save(tag="best")

            if (
                self._config.training.save_interval > 0
                and self.step % self._config.training.save_interval == 0
            ):
                self.save()

    def evaluate(self, val_bin_path: str | Path | None = None) -> dict[str, float]:
        """Compute evaluation metrics over the validation subset."""
        training_cfg = self._config.training
        model_cfg = self._config.model

        from hymo.training.validation import DEFAULT_VAL_BIN

        metrics: ValMetrics = compute_validation_loss(
            self.model,
            batch_size=training_cfg.micro_batch_size,
            seq_len=training_cfg.max_seq_len,
            vocab_size=model_cfg.vocab_size,
            num_batches=min(4, 32),
            device=next(self.model.parameters()).device,
            val_bin_path=Path(val_bin_path) if val_bin_path else DEFAULT_VAL_BIN,
        )

        log.info(
            "eval step=%d val_loss=%.4f val_ppl=%.4f batches=%d tokens=%d",
            self.step, metrics.loss, metrics.ppl,
            metrics.num_batches, metrics.num_tokens,
        )

        return {
            "val_loss": metrics.loss,
            "val_ppl": metrics.ppl,
        }

    def _current_lr_muon(self) -> float:
        if self.optimizers.nor_muon is not None and len(self.optimizers.nor_muon.param_groups) > 0:
            return float(self.optimizers.nor_muon.param_groups[0].get("lr", 0.0))
        return 0.0

    def _current_lr_adamw(self) -> float:
        if len(self.optimizers.adamw.param_groups) > 0:
            return float(self.optimizers.adamw.param_groups[0].get("lr", 0.0))
        return 0.0
