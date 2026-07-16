"""Callback system for the training loop.

A :class:`Callback` is a hook point that the trainer calls at well-defined
events (start of training, end of step, end of epoch, end of training).
Callbacks are the standard way to add side effects — logging, validation,
checkpointing, NaN-skip, expert-load entropy alerting — without
polluting the main training loop.

Pattern
-------
The trainer emits events as :class:`TrainerState` snapshots. Callbacks
read the state and act. There are no return values; the state is
mutated by the trainer before each event.

Inspired by Keras's ``keras.callbacks.Callback``.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any, Protocol

__all__ = [
    "Callback",
    "CallbackList",
    "TrainerState",
    "CallbackEvent",
]


class CallbackEvent:
    """The set of trainer events a callback can hook into.

    Constants
    ---------
    TRAIN_BEGIN : str
        Before the first step of training.
    TRAIN_END : str
        After the last step of training.
    STEP_BEGIN : str
        Before each optimizer step.
    STEP_END : str
        After each optimizer step.
    EVAL_BEGIN : str
        Before a validation pass.
    EVAL_END : str
        After a validation pass.
    CHECKPOINT_SAVE : str
        Before a checkpoint save.
    CHECKPOINT_LOAD : str
        After a checkpoint load.
    EXCEPTION : str
        When an unhandled exception is raised in the training loop.
    """

    TRAIN_BEGIN = "on_train_begin"
    TRAIN_END = "on_train_end"
    STEP_BEGIN = "on_step_begin"
    STEP_END = "on_step_end"
    EVAL_BEGIN = "on_eval_begin"
    EVAL_END = "on_eval_end"
    CHECKPOINT_SAVE = "on_checkpoint_save"
    CHECKPOINT_LOAD = "on_checkpoint_load"
    EXCEPTION = "on_exception"


@dataclass
class TrainerState:
    """A snapshot of trainer state, passed to callbacks at each event.

    The trainer mutates this in place between events. Callbacks read
    from it and may *request* state changes (e.g. setting
    ``stop_training = True``), but the trainer is the only thing that
    actually mutates the underlying training state.
    """

    step: int = 0
    token_count: int = 0
    epoch: int = 0
    global_step: int = 0
    loss: float = 0.0
    grad_norm: float = 0.0
    lr_muon: float = 0.0
    lr_adamw: float = 0.0
    metrics: dict[str, float] = field(default_factory=dict)
    # Requested by callbacks
    stop_training: bool = False
    skip_step: bool = False
    save_checkpoint: bool = False


class Callback(Protocol):
    """Protocol for trainer callbacks.

    Implement any subset of the on_* methods. Methods take the
    :class:`TrainerState` and may set request flags on it.
    """

    def on_train_begin(self, state: TrainerState) -> None: ...
    def on_train_end(self, state: TrainerState) -> None: ...
    def on_step_begin(self, state: TrainerState) -> None: ...
    def on_step_end(self, state: TrainerState) -> None: ...
    def on_eval_begin(self, state: TrainerState) -> None: ...
    def on_eval_end(self, state: TrainerState) -> None: ...
    def on_checkpoint_save(self, state: TrainerState) -> None: ...
    def on_checkpoint_load(self, state: TrainerState) -> None: ...
    def on_exception(self, state: TrainerState, exc: BaseException) -> None: ...


class CallbackList:
    """A list of :class:`Callback` objects, dispatched by event.

    Iterates over the callbacks in registration order at each event.
    Failures in one callback are logged and isolated — the remaining
    callbacks still run.
    """

    def __init__(self, callbacks: Iterable[Callback] | None = None) -> None:
        self._callbacks: list[Callback] = list(callbacks or [])

    def add(self, callback: Callback) -> None:
        self._callbacks.append(callback)

    def remove(self, callback: Callback) -> None:
        try:
            self._callbacks.remove(callback)
        except ValueError as e:
            raise ValueError("Callback not in list") from e

    def __iter__(self) -> Iterator[Callback]:
        return iter(self._callbacks)

    def __len__(self) -> int:
        return len(self._callbacks)

    def dispatch(self, event: str, state: TrainerState, *args: Any) -> None:
        """Dispatch ``event`` to every callback that defines it.

        Exceptions in individual callbacks are logged and isolated.
        """
        for cb in self._callbacks:
            handler = getattr(cb, event, None)
            if handler is None:
                continue
            try:
                handler(state, *args)
            except Exception:  # pragma: no cover — defensive
                from hymo.utils.logging import get_logger

                get_logger("callbacks").exception(
                    "Callback %s raised in %s; continuing", cb, event
                )
