"""Cross-field runtime config validation.

Contains validation rules that check relationships between different configurations.
"""

from __future__ import annotations

import math

from hymo.core.config import HyMoConfig, ModelConfig, TrainingConfig

# Peak budget memory limit per GPU rank (A100 SXM 80GB)
A100_80GB_BYTES = 80 * 1024 * 1024 * 1024


def validate_full_config(config: HyMoConfig) -> None:
    """Run all cross-field configuration sanity checks."""
    _validate_total_steps_consistency(config)
    _validate_layer_distribution(config.model)
    _validate_partial_rope_math(config.model)
    _validate_vram_budget(config.model, config.training)


def _validate_total_steps_consistency(config: HyMoConfig) -> None:
    per_step = config.training.per_step_tokens
    if per_step <= 0:
        raise ValueError(
            f"per_step_tokens must be > 0, got {per_step}"
        )


def _validate_layer_distribution(model: ModelConfig) -> None:
    if model.n_layers % 4 != 0:
        raise ValueError(
            f"n_layers ({model.n_layers}) must be a multiple of 4 for the "
            f"3:1 GDN:MLA distribution; got {model.n_layers}"
        )
    expected_mla = model.n_layers // 4
    if model.n_mla_layers != expected_mla:
        raise ValueError(
            f"n_mla_layers ({model.n_mla_layers}) != n_layers // 4 "
            f"({expected_mla}); the 3:1 distribution is broken"
        )
    if model.mla_positions != frozenset(i * 4 for i in range(expected_mla)):
        raise ValueError(
            f"mla_positions ({sorted(model.mla_positions)}) does not match "
            f"expected {{0, 4, ..., {4 * (expected_mla - 1)}}}"
        )


def _validate_partial_rope_math(model: ModelConfig) -> None:
    rope_frac = model.qk_rope_head_dim / model.head_dim
    if not 0.0 <= rope_frac <= 1.0:
        raise ValueError(
            f"partial-RoPE fraction {rope_frac} out of [0, 1]"
        )


def _validate_vram_budget(model: ModelConfig, training: TrainingConfig) -> None:
    """Estimate parameters, grads, master weights, and optimizer state VRAM to warn if rank budget is exceeded."""
    embed = model.vocab_size * model.dim
    gdn_per = 25_000_000
    mla_attn_per = 5_800_000
    moe_active_per = 9_000_000
    moe_stored_per = 145_000_000
    norm_softcap = 1_000
    n_mla = model.n_mla_layers
    n_gdn = model.n_gdn_layers

    active = (
        embed
        + n_gdn * gdn_per
        + n_mla * (mla_attn_per + moe_active_per)
        + norm_softcap
    )
    stored = (
        embed
        + n_gdn * gdn_per
        + n_mla * (mla_attn_per + moe_stored_per)
        + norm_softcap
    )

    bytes_per_rank_params_bf16 = (stored / training.world_size) * 2
    bytes_per_rank_grads_bf16 = (stored / training.world_size) * 2
    bytes_per_rank_master_fp32 = (active / training.world_size) * 4
    bytes_per_rank_adamw_fp32 = (active / training.world_size) * 8
    bytes_all_gather = stored * 2
    bytes_activations = (
        training.micro_batch_size
        * training.max_seq_len
        * model.dim
        * 12
    )

    total = (
        bytes_per_rank_params_bf16
        + bytes_per_rank_grads_bf16
        + bytes_per_rank_master_fp32
        + bytes_per_rank_adamw_fp32
        + bytes_activations
    )
    peak = total + bytes_all_gather

    if peak > A100_80GB_BYTES:
        raise ValueError(
            f"Estimated per-rank peak VRAM ({peak / 1e9:.1f} GB) exceeds the "
            f"4× A100 80GB budget ({A100_80GB_BYTES / 1e9:.1f} GB). "
            f"Reduce model size, enable more aggressive activation "
            f"checkpointing, or scale world_size up."
        )


__all__ = [
    "validate_full_config",
    "A100_80GB_BYTES",
]

_ = math
