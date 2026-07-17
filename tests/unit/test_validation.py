"""Tests for the :mod:`hymo.core.config_validation` module."""

from __future__ import annotations

import pytest

from hymo.core.config import (
    HyMoConfig,
    ModelConfig,
    TrainingConfig,
)
from hymo.core.config_validation import validate_full_config
from hymo.core.exceptions import ConfigValidationError


class TestValidateFullConfig:
    def test_default_config_validates(self) -> None:
        """The v1.0 default config passes validation."""
        validate_full_config(HyMoConfig())

    def test_per_step_tokens_must_be_positive(self) -> None:
        with pytest.raises(ConfigValidationError):
            c = HyMoConfig(
                training=TrainingConfig(
                    micro_batch_size=0,
                    gradient_accumulation_steps=8,
                    max_seq_len=4096,
                    world_size=4,
                ),
            )
            validate_full_config(c)

    def test_n_layers_must_be_multiple_of_4(self) -> None:
        with pytest.raises(ConfigValidationError):
            c = HyMoConfig(model=ModelConfig(n_layers=5))
            validate_full_config(c)

    def test_mla_positions_match_3_1_distribution(self) -> None:
        # ``mla_positions`` is a derived property (``{0, 4, 8, ...}`` for
        # the v1.0 3:1 GDN:MLA distribution), so the only way to break it
        # is to violate ``n_layers % 4 == 0`` — already covered above.
        # This test confirms the default 3:1 distribution is consistent.
        m = ModelConfig()
        assert m.n_mla_layers == m.n_layers // 4
        assert m.mla_positions == frozenset(i * 4 for i in range(m.n_mla_layers))
        # And the derived config validates cleanly.
        validate_full_config(HyMoConfig(model=m))
