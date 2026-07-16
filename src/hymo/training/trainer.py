"""The HyMo training loop (Phase 3 implementation).

Ties together:

- :class:`hymo.training.optimizer.build_optimizers` (NorMuon + AdamW).
- :class:`hymo.training.scheduler.JointWSDScheduler`.
- :class:`hymo.training.checkpoint.save_checkpoint` /
  :func:`load_checkpoint`.
- :class:`hymo.training.validation.compute_validation_loss` for
  real held-out val.
- :class:`hymo.utils.callbacks.CallbackList` for the event hook.
"""

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
from hymo.utils.callbacks import CallbackEvent, CallbackList, TrainerState

__all__ = ["Trainer", "TrainerConfig", "train_step_result"]

log = logging.getLogger(__name__)


@dataclass
class TrainerConfig:
    """Trainer-only config knobs (subset of :class:`TrainingConfig`)."""

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
    """The result of a single :meth:`Trainer.train_step` call.

    Attributes
    ----------
    loss : float
        The cross-entropy loss for this step (after MTP contributions).
    grad_norm : float
        The L2 norm of the gradients (after clip).
    lr_muon : float
        The current NorMuon learning rate.
    lr_adamw : float
        The current AdamW learning rate.
    skipped : bool
        True if the step was skipped (NaN-skip).
    metrics : dict
        Free-form dict for additional metrics (e.g. MTP losses).
    """

    loss: float
    grad_norm: float
    lr_muon: float
    lr_adamw: float
    skipped: bool = False
    metrics: dict[str, float] = field(default_factory=dict)


class Trainer:
    """The main HyMo training loop.

    Parameters
    ----------
    config : HyMoConfig
        The top-level config.
    model : HyMo
        The HyMo model (already constructed and μP-init'd).
    callbacks : CallbackList or None
        Optional callback list.
    """

    def __init__(
        self,
        config: HyMoConfig,
        model: HyMo,
        callbacks: CallbackList | None = None,
    ) -> None:
        self._config = config
        self.model = model

        self.callbacks = callbacks if callbacks is not None else CallbackList()

        self.optimizers = build_optimizers(model, config.optimizer)
        self.scheduler = JointWSDScheduler(config.scheduler)

        # Store base LRs so the scheduler factor is applied multiplicatively.
        self._base_lr_muon: float | None = (
            config.optimizer.muon_lr if self.optimizers.nor_muon else None
        )
        self._base_lr_adamw: float = config.optimizer.adamw_lr

        # Public state — the callbacks read this via the TrainerState.
        self.step: int = 0
        self.token_count: int = 0
        self.best_loss: float = float("inf")
        self.state = TrainerState()

        if config.model.mtp_depth > 0:
            self._has_mtp = True
        else:
            self._has_mtp = False

    # ---- Public API -----------------------------------------------------

    def train_step(
        self,
        tokens: torch.Tensor,
        targets: torch.Tensor,
    ) -> train_step_result:
        """Run a single optimizer step.

        Performs a forward pass, computes the cross-entropy loss
        (including MTP if applicable), backpropagates, clips gradients,
        applies the optimizers, and advances the scheduler.

        Parameters
        ----------
        tokens : torch.Tensor
            Input token ids of shape ``(B, T)``.
        targets : torch.Tensor
            Target token ids of shape ``(B, T)``.

        Returns
        -------
        train_step_result
        """
        self.model.train()

        # ---- forward + loss ----
        if self._has_mtp:
            mtp_module = getattr(self.model, "_mtp", None)
            if mtp_module is not None:
                logits, mtp_outputs = mtp_module.forward(tokens)
                # logits are raw (pre-softcap) from forward_with_hidden
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

        # ---- NaN-skip check ----
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

        # ---- backward ----
        total_loss.backward()  # type: ignore[no-untyped-call]

        # ---- gradient clipping ----
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            max_norm=self._config.training.grad_clip,
            norm_type=2.0,
        )
        grad_norm_val = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else float(grad_norm)

        # ---- scheduler factor → effective LR ----
        factor = self.scheduler.get_factor(self.step + 1)
        if self.optimizers.nor_muon is not None and self._base_lr_muon is not None:
            for g in self.optimizers.nor_muon.param_groups:
                g["lr"] = self._base_lr_muon * factor
        for g in self.optimizers.adamw.param_groups:
            g["lr"] = self._base_lr_adamw * factor

        # ---- optimizer step ----
        if self.optimizers.nor_muon is not None:
            self.optimizers.nor_muon.step()
        self.optimizers.adamw.step()

        # ---- post-step cleanup ----
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
        """Save a checkpoint to ``{output_dir}/{tag}/model.pt``.

        Parameters
        ----------
        tag : str or None
            Checkpoint tag (e.g. ``"step_1000"`` or ``"best"``).
            Defaults to ``f"step_{self.step}"``.

        Returns
        -------
        Path
            The path to the saved checkpoint.
        """
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
        """Load a checkpoint and resume.

        Parameters
        ----------
        path : str or Path
            Path to the checkpoint file (``.pt``) or directory
            containing ``model.pt``.

        Returns
        -------
        int
            The step count to resume from.
        """
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

        self.callbacks.dispatch(
            CallbackEvent.CHECKPOINT_LOAD,
            self._make_state(),
        )

        log.info("Checkpoint loaded from %s (step=%d)", p, self.step)
        return self.step

    def train(
        self, data_iter: Iterable[tuple[torch.Tensor, torch.Tensor]], max_steps: int | None = None
    ) -> None:
        """Run the main training loop.

        Parameters
        ----------
        data_iter : iterable of (tokens, targets)
            An iterable yielding ``(tokens, targets)`` tensors of shape
            ``(B, T)`` each. Typically a :class:`torch.utils.data.DataLoader`.
        max_steps : int or None
            Maximum number of optimizer steps. Defaults to the config's
            ``scheduler.total_steps``.
        """
        if max_steps is None:
            max_steps = self._config.scheduler.total_steps

        self.callbacks.dispatch(CallbackEvent.TRAIN_BEGIN, self._make_state())

        for tokens, targets in data_iter:
            if self.step >= max_steps:
                break

            self.callbacks.dispatch(CallbackEvent.STEP_BEGIN, self._make_state())

            result = self.train_step(tokens, targets)

            # Update shared state for callbacks.
            self.state.loss = result.loss
            self.state.grad_norm = result.grad_norm
            self.state.lr_muon = result.lr_muon
            self.state.lr_adamw = result.lr_adamw
            self.state.metrics = result.metrics

            if (
                self.step % self._config.training.log_interval == 0
                and not result.skipped
            ):
                log.info(
                    "step=%d loss=%.4f grad_norm=%.4f lr_muon=%.6f lr_adamw=%.6f",
                    self.step, result.loss, result.grad_norm,
                    result.lr_muon, result.lr_adamw,
                )

            # Validation.
            if (
                self._config.training.eval_interval > 0
                and self.step % self._config.training.eval_interval == 0
            ):
                self.callbacks.dispatch(CallbackEvent.EVAL_BEGIN, self._make_state())
                eval_metrics = self.evaluate()
                self.state.metrics.update(eval_metrics)
                if eval_metrics.get("val_loss", float("inf")) < self.best_loss:
                    self.best_loss = eval_metrics["val_loss"]
                    self.save(tag="best")
                self.callbacks.dispatch(CallbackEvent.EVAL_END, self._make_state())

            # Checkpoint save.
            if (
                self._config.training.save_interval > 0
                and self.step % self._config.training.save_interval == 0
            ):
                self.callbacks.dispatch(
                    CallbackEvent.CHECKPOINT_SAVE, self._make_state()
                )
                self.save()

            self.callbacks.dispatch(CallbackEvent.STEP_END, self._make_state())

            if self.state.stop_training:
                break

        self.callbacks.dispatch(CallbackEvent.TRAIN_END, self._make_state())

    def evaluate(self, val_bin_path: str | Path | None = None) -> dict[str, float]:
        """Run a single validation pass on the held-out ``val.bin``.

        Parameters
        ----------
        val_bin_path : str or Path or None
            Path to the validation binary. Defaults to
            ``data/tokens/val.bin``.

        Returns
        -------
        dict[str, float]
            With keys ``val_loss`` and ``val_ppl``.
        """
        training_cfg = self._config.training
        model_cfg = self._config.model

        from hymo.training.validation import DEFAULT_VAL_BIN

        metrics: ValMetrics = compute_validation_loss(
            self.model,
            batch_size=training_cfg.micro_batch_size,
            seq_len=training_cfg.max_seq_len,
            vocab_size=model_cfg.vocab_size,
            num_batches=min(4, 32),  # quick partial eval by default
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

    # ---- Internal helpers ------------------------------------------------

    def _current_lr_muon(self) -> float:
        if self.optimizers.nor_muon is not None and len(self.optimizers.nor_muon.param_groups) > 0:
            return float(self.optimizers.nor_muon.param_groups[0].get("lr", 0.0))
        return 0.0

    def _current_lr_adamw(self) -> float:
        if len(self.optimizers.adamw.param_groups) > 0:
            return float(self.optimizers.adamw.param_groups[0].get("lr", 0.0))
        return 0.0

    def _make_state(self) -> TrainerState:
        """Build a fresh :class:`TrainerState` for callback dispatch."""
        return TrainerState(
            step=self.step,
            token_count=self.token_count,
            loss=self.state.loss,
            grad_norm=self.state.grad_norm,
            lr_muon=self._current_lr_muon(),
            lr_adamw=self._current_lr_adamw(),
            metrics=dict(self.state.metrics),
        )
