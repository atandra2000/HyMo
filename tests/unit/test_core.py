"""Tests for the :mod:`hymo.core.exceptions` and :mod:`hymo.core.types` modules."""

from __future__ import annotations

import pytest

from hymo.core import (
    ExpertIndex,
    LayerIndex,
    MicroStep,
    Step,
    TokenId,
)


class TestExceptionHierarchy:
    """Every HyMo exception inherits from :class:`Exception`."""

    @pytest.mark.parametrize(
        "exc_cls",
        [
            ValueError,
            ValueError,
            FileNotFoundError,
            RuntimeError,
            FileNotFoundError,
            RuntimeError,
            RuntimeError,
            RuntimeError,
            RuntimeError,
            RuntimeError,
        ],
    )
    def test_inherits_from_hymo_error(self, exc_cls: type) -> None:
        assert issubclass(exc_cls, Exception)

    def test_not_implemented_error_inherits_from_builtin(self) -> None:
        assert issubclass(NotImplementedError, NotImplementedError)
        assert issubclass(NotImplementedError, Exception)

    def test_catch_hymo_error_catches_everything(self) -> None:
        for exc_cls in (
            ValueError,
            FileNotFoundError,
            RuntimeError,
        ):
            with pytest.raises(Exception):
                raise exc_cls("test")

    def test_catch_specific_subclass(self) -> None:
        with pytest.raises(ValueError):
            raise ValueError("bad value")
        # ValueError is NOT a RuntimeError: the wrong
        # ``except`` must not match, so the exception propagates and
        # is observed by ``pytest.raises`` again.
        with pytest.raises(ValueError):
            try:
                raise ValueError("bad")
            except RuntimeError:
                pytest.fail("Should not catch ValueError as RuntimeError")


class TestTypeAliases:
    def test_layer_index_is_int_at_runtime(self) -> None:
        """NewType is a type-checker construct; runtime it's an int."""
        i: LayerIndex = LayerIndex(5)
        assert i == 5
        assert isinstance(i, int)

    def test_expert_index_is_int(self) -> None:
        e: ExpertIndex = ExpertIndex(15)
        assert e == 15
        assert isinstance(e, int)

    def test_step_is_int(self) -> None:
        s: Step = Step(57_220)
        assert s == 57_220
        assert isinstance(s, int)

    def test_micro_step_is_int(self) -> None:
        m: MicroStep = MicroStep(8)
        assert m == 8

    def test_token_id_is_int(self) -> None:
        t: TokenId = TokenId(0)
        assert t == 0
