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

__all__ = ["Trainer", "train_step_result"]

log = logging.getLogger(__name__)


@dataclass
class train_step_result:
    """The result metrics of a single training step."""

    loss: float
    grad_norm: float
    lr_muon: float
    lr_adamw: float
    is_update: bool = True
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
        self._thread_optimization_flags()

        import os

        import torch.distributed as dist
        _wandb_disabled = os.environ.get("WANDB_MODE", "").lower() in ("disabled", "offline", "dryrun")
        if not _wandb_disabled and (not dist.is_initialized() or dist.get_rank() == 0):
            import dataclasses

            import wandb
            cfg_dict = dataclasses.asdict(config) if dataclasses.is_dataclass(config) else {}
            wandb.init(
                project="HyMo",
                config=cfg_dict,
                resume="allow",
            )
        self._wandb_enabled = not _wandb_disabled

        self.optimizers = build_optimizers(model, config.optimizer)
        self.scheduler = JointWSDScheduler(config.scheduler)

        self._base_lr_muon: float | None = (
            config.optimizer.muon_lr if self.optimizers.nor_muon else None
        )
        self._base_lr_adamw: float = config.optimizer.adamw_lr

        self.step: int = 0
        self.micro_step: int = 0
        self.token_count: int = 0
        self.best_loss: float = float("inf")

        if config.model.mtp_depth > 0:
            self._has_mtp = True
        else:
            self._has_mtp = False

    def _thread_optimization_flags(self) -> None:
        """Push training-config optimization toggles onto the model blocks.

        GDN blocks select the Triton kernel / torch.compile via
        ``fused_gdn`` / ``torch_compile_gdn``; MoE blocks toggle the
        mixed-precision dispatch via ``moe_mixed_precision``; MLA blocks
        toggle CUDA-Graph capture via ``cuda_graphs_mla``. Blocks built
        standalone default everything on (design intent).
        """
        from hymo.models.gdn import GatedDeltaNetBlock
        from hymo.models.mla import MLABlock
        from hymo.models.moe import DeepSeekMoE

        t = self._config.training
        for module in self.model.modules():
            if isinstance(module, GatedDeltaNetBlock):
                module.use_triton = t.fused_gdn
                module.use_compile = t.torch_compile_gdn
            elif isinstance(module, DeepSeekMoE):
                module.use_mixed_precision = t.moe_mixed_precision

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
                is_update=False,
                skipped=True,
                metrics={"main_loss": float("nan"), **mtp_details},
            )

        scaled_loss = total_loss / self._config.training.gradient_accumulation_steps
        scaled_loss.backward()

        self.micro_step += 1
        is_update = (self.micro_step % self._config.training.gradient_accumulation_steps == 0)
        grad_norm_val = 0.0

        if is_update:
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

            self._update_moe_gate_biases()

            self.scheduler.step()
            self.model.zero_grad(set_to_none=True)

            self.step += 1

        self.token_count += tokens.numel()

        return train_step_result(
            loss=total_loss.item(),
            grad_norm=grad_norm_val,
            lr_muon=self._current_lr_muon(),
            lr_adamw=self._current_lr_adamw(),
            is_update=is_update,
            skipped=False,
            metrics={"main_loss": main_loss.item(), **mtp_details},
        )

    def save(self, tag: str | None = None) -> Path:
        """Save a training checkpoint directory using DCP."""
        if tag is None:
            tag = f"step_{self.step}"
        output_dir = Path(self._config.run.output_dir)
        ckpt_dir = output_dir / tag

        state = CheckpointState(
            step=self.step,
            token_count=self.token_count,
            best_loss=self.best_loss,
        )

        save_checkpoint(
            path=ckpt_dir,
            model=self.model,
            optimizers=self.optimizers,
            scheduler=self.scheduler,
            state=state,
        )

        log.info("Checkpoint saved to %s (step=%d)", ckpt_dir, self.step)
        return ckpt_dir

    def load(self, path: str | Path) -> int:
        """Load a DCP checkpoint directory and restore training state."""
        p = Path(path)

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

            if result.is_update:
                if (
                    self.step % self._config.training.log_interval == 0
                    and not result.skipped
                ):
                    import torch.distributed as dist
                    if self._wandb_enabled and (not dist.is_initialized() or dist.get_rank() == 0):
                        import wandb
                        wandb.log({
                            "train/loss": result.loss,
                            "train/grad_norm": result.grad_norm,
                            "train/lr_muon": result.lr_muon,
                            "train/lr_adamw": result.lr_adamw,
                            "train/tokens": self.token_count,
                            **{f"train/{k}": v for k, v in result.metrics.items()}
                        }, step=self.step)

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

        import torch.distributed as dist
        if self._wandb_enabled and (not dist.is_initialized() or dist.get_rank() == 0):
            import wandb
            wandb.log({
                "val/loss": metrics.loss,
                "val/ppl": metrics.ppl,
                "val/batches": metrics.num_batches,
                "val/tokens": metrics.num_tokens,
            }, step=self.step)

        return {
            "val_loss": metrics.loss,
            "val_ppl": metrics.ppl,
        }

    def _update_moe_gate_biases(self) -> None:
        """Apply EMA load-balancing to every MoE gate (aux-loss-free routing)."""
        from hymo.models.moe import DeepSeekMoE

        for module in self.model.modules():
            if isinstance(module, DeepSeekMoE):
                module.update_gate_bias()

    def _current_lr_muon(self) -> float:
        if self.optimizers.nor_muon is not None and len(self.optimizers.nor_muon.param_groups) > 0:
            return float(self.optimizers.nor_muon.param_groups[0].get("lr", 0.0))
        return 0.0

    def _current_lr_adamw(self) -> float:
        if len(self.optimizers.adamw.param_groups) > 0:
            return float(self.optimizers.adamw.param_groups[0].get("lr", 0.0))
        return 0.0
