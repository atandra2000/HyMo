"""Tests for the :mod:`hymo.utils.callbacks` module."""

from __future__ import annotations

import pytest

from hymo.utils.callbacks import (
    CallbackEvent,
    CallbackList,
    TrainerState,
)


class CountingCallback:
    """A test callback that counts how many times each method was called."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def on_train_begin(self, state: TrainerState) -> None:
        self.calls.append("train_begin")

    def on_train_end(self, state: TrainerState) -> None:
        self.calls.append("train_end")

    def on_step_begin(self, state: TrainerState) -> None:
        self.calls.append("step_begin")

    def on_step_end(self, state: TrainerState) -> None:
        self.calls.append("step_end")

    def on_eval_begin(self, state: TrainerState) -> None:
        self.calls.append("eval_begin")

    def on_eval_end(self, state: TrainerState) -> None:
        self.calls.append("eval_end")

    def on_checkpoint_save(self, state: TrainerState) -> None:
        self.calls.append("ckpt_save")

    def on_checkpoint_load(self, state: TrainerState) -> None:
        self.calls.append("ckpt_load")

    def on_exception(self, state: TrainerState, exc: BaseException) -> None:
        self.calls.append(f"exception:{type(exc).__name__}")


class TestTrainerState:
    def test_defaults(self) -> None:
        s = TrainerState()
        assert s.step == 0
        assert s.token_count == 0
        assert s.loss == 0.0
        assert s.stop_training is False
        assert s.skip_step is False
        assert s.save_checkpoint is False
        assert s.metrics == {}

    def test_can_mutate(self) -> None:
        s = TrainerState()
        s.step = 100
        s.metrics["loss"] = 11.06
        s.stop_training = True
        assert s.step == 100
        assert s.metrics["loss"] == 11.06
        assert s.stop_training is True


class TestCallbackList:
    def test_empty(self) -> None:
        cl = CallbackList()
        assert len(cl) == 0
        cl.dispatch(CallbackEvent.STEP_END, TrainerState())  # no-op

    def test_dispatch_calls_all_callbacks(self) -> None:
        cb1 = CountingCallback()
        cb2 = CountingCallback()
        cl = CallbackList([cb1, cb2])
        cl.dispatch(CallbackEvent.STEP_END, TrainerState())
        assert cb1.calls == ["step_end"]
        assert cb2.calls == ["step_end"]

    def test_dispatch_in_registration_order(self) -> None:
        seen: list[str] = []

        class A:
            def on_step_end(self, state: TrainerState) -> None:
                seen.append("A")

        class B:
            def on_step_end(self, state: TrainerState) -> None:
                seen.append("B")

        cl = CallbackList([A(), B()])
        cl.dispatch(CallbackEvent.STEP_END, TrainerState())
        assert seen == ["A", "B"]

    def test_callback_without_method_is_skipped(self) -> None:
        class WithoutStepEnd:
            def on_train_begin(self, state: TrainerState) -> None:
                pass

        cl = CallbackList([WithoutStepEnd()])
        # No error even though the callback doesn't define on_step_end.
        cl.dispatch(CallbackEvent.STEP_END, TrainerState())

    def test_callback_exception_is_isolated(self) -> None:
        class BadCallback:
            def on_step_end(self, state: TrainerState) -> None:
                raise RuntimeError("boom")

        good = CountingCallback()
        cl = CallbackList([BadCallback(), good])
        # Should not raise; good callback still runs.
        cl.dispatch(CallbackEvent.STEP_END, TrainerState())
        assert good.calls == ["step_end"]

    def test_add_and_remove(self) -> None:
        cl = CallbackList()
        cb = CountingCallback()
        cl.add(cb)
        assert len(cl) == 1
        cl.remove(cb)
        assert len(cl) == 0

    def test_remove_unknown_raises(self) -> None:
        cl = CallbackList()
        with pytest.raises(ValueError):
            cl.remove(CountingCallback())
