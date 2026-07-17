"""Tests for the :mod:`hymo.core.exceptions` and :mod:`hymo.core.types` modules."""

from __future__ import annotations

import pytest

from hymo.core.exceptions import (
    CheckpointCorruptError,
    CheckpointError,
    CheckpointNotFoundError,
    ConfigError,
    ConfigNotFoundError,
    ConfigValidationError,
    DataError,
    DistributedError,
    HyMoError,
    NotImplementedError_,
    ShapeError,
    TokenizerError,
)
from hymo.core import (
    ExpertIndex,
    LayerIndex,
    MicroStep,
    Step,
    TokenId,
)


class TestExceptionHierarchy:
    """Every HyMo exception inherits from :class:`HyMoError`."""

    @pytest.mark.parametrize(
        "exc_cls",
        [
            ConfigError,
            ConfigValidationError,
            ConfigNotFoundError,
            CheckpointError,
            CheckpointNotFoundError,
            CheckpointCorruptError,
            DataError,
            DistributedError,
            ShapeError,
            TokenizerError,
        ],
    )
    def test_inherits_from_hymo_error(self, exc_cls: type) -> None:
        assert issubclass(exc_cls, HyMoError)

    def test_not_implemented_error_inherits_from_builtin(self) -> None:
        assert issubclass(NotImplementedError_, NotImplementedError)
        assert issubclass(NotImplementedError_, HyMoError)

    def test_catch_hymo_error_catches_everything(self) -> None:
        for exc_cls in (
            ConfigValidationError,
            CheckpointNotFoundError,
            DataError,
        ):
            with pytest.raises(HyMoError):
                raise exc_cls("test")

    def test_catch_specific_subclass(self) -> None:
        with pytest.raises(ConfigValidationError):
            raise ConfigValidationError("bad value")
        # ConfigValidationError is NOT a CheckpointError: the wrong
        # ``except`` must not match, so the exception propagates and
        # is observed by ``pytest.raises`` again.
        with pytest.raises(ConfigValidationError):
            try:
                raise ConfigValidationError("bad")
            except CheckpointError:
                pytest.fail("Should not catch ConfigValidationError as CheckpointError")


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
