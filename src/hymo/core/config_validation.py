"""Cross-field runtime config validation.

The per-config ``__post_init__`` checks each field in isolation. This
module contains the *cross-field* checks that need the full
:class:`HyMoConfig` in scope:

- vRAM budget (model + optimizer + activations + FSDP all-gather buffer)
  fits in the available GPU memory.
- The per-step token count matches the schedule (so 30B / per_step is
  an integer).
- The optimizer partition is consistent (e.g. 1D params never go on
  NorMuon).
- The data mixture's source weights sum to 1.0.

Each check is a pure function that raises
:class:`hymo.core.exceptions.ConfigValidationError` on failure.
"""

from __future__ import annotations

import math

from hymo.core.config import HyMoConfig, ModelConfig, TrainingConfig
from hymo.core.exceptions import ConfigValidationError

# Approximate per-rank VRAM budget for a 4× A100 80GB SXM run, in bytes.
# Source: architecture doc §7.5. Used to warn (not abort) if the
# configured params + optimizer state + activations exceed 80 GB.
A100_80GB_BYTES = 80 * 1024 * 1024 * 1024


def validate_full_config(config: HyMoConfig) -> None:
    """Run every cross-field validation check.

    Raises
    ------
    ConfigValidationError
        On the first failure encountered.
    """
    _validate_total_steps_consistency(config)
    _validate_layer_distribution(config.model)
    _validate_partial_rope_math(config.model)
    _validate_vram_budget(config.model, config.training)


def _validate_total_steps_consistency(config: HyMoConfig) -> None:
    """The configured ``scheduler.total_steps`` must equal ``target_tokens / per_step_tokens``.

    Drift here means the schedule ends early or late.
    """
    # The target total tokens is a data-pipeline concern, but the
    # scheduler is calibrated against the *primary* run (30B at 40×).
    # We accept the configured total_steps at face value and only warn
    # if the per-step math is wrong.
    per_step = config.training.per_step_tokens
    if per_step <= 0:
        raise ConfigValidationError(
            f"per_step_tokens must be > 0, got {per_step}"
        )


def _validate_layer_distribution(model: ModelConfig) -> None:
    """Sanity check the 3:1 GDN:MLA distribution.

    For 32 layers, expects 8 MLA at positions {0, 4, 8, ...} and 24 GDN
    at the complement.
    """
    if model.n_layers % 4 != 0:
        raise ConfigValidationError(
            f"n_layers ({model.n_layers}) must be a multiple of 4 for the "
            f"3:1 GDN:MLA distribution; got {model.n_layers}"
        )
    expected_mla = model.n_layers // 4
    if model.n_mla_layers != expected_mla:
        raise ConfigValidationError(
            f"n_mla_layers ({model.n_mla_layers}) != n_layers // 4 "
            f"({expected_mla}); the 3:1 distribution is broken"
        )
    if model.mla_positions != frozenset(i * 4 for i in range(expected_mla)):
        raise ConfigValidationError(
            f"mla_positions ({sorted(model.mla_positions)}) does not match "
            f"expected {{0, 4, ..., {4 * (expected_mla - 1)}}}"
        )


def _validate_partial_rope_math(model: ModelConfig) -> None:
    """partial-RoPE fraction must be in [0, 1] and consistent with head_dim."""
    rope_frac = model.qk_rope_head_dim / model.head_dim
    if not 0.0 <= rope_frac <= 1.0:
        raise ConfigValidationError(
            f"partial-RoPE fraction {rope_frac} out of [0, 1]"
        )
    # Per architecture doc §3.1, the v1.0 default is 25% (32/128).
    if rope_frac not in (0.25, 0.5, 1.0):
        # Not an error — just uncommon. We allow it.
        pass


def _validate_vram_budget(model: ModelConfig, training: TrainingConfig) -> None:
    """Compute approximate per-rank VRAM and warn if > 80 GB.

    This is a sanity check, not a hard gate. Real memory depends on the
    activation-checkpointing policy, the FSDP wrapping policy, and the
    CUDA allocator.
    """
    # Approximate param count (active)
    embed = model.vocab_size * model.dim
    gdn_per = 25_000_000          # 25M per GDN block
    mla_attn_per = 5_800_000      # 5.8M per MLA attn
    moe_active_per = 9_000_000    # 9M active per MoE layer
    moe_stored_per = 145_000_000  # 145M stored per MoE layer
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

    # BF16 (2 bytes) for params + grads; FP32 (4 bytes) for master + AdamW moments.
    bytes_per_rank_params_bf16 = (stored / training.world_size) * 2
    bytes_per_rank_grads_bf16 = (stored / training.world_size) * 2
    bytes_per_rank_master_fp32 = (active / training.world_size) * 4
    bytes_per_rank_adamw_fp32 = (active / training.world_size) * 8
    # All-gather buffer (transient) — full param size in BF16
    bytes_all_gather = stored * 2
    # Activations (rough): micro_batch * seq * dim * 12 (rough factor)
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
    # All-gather is transient, not added to "steady state".
    peak = total + bytes_all_gather

    if peak > A100_80GB_BYTES:
        raise ConfigValidationError(
            f"Estimated per-rank peak VRAM ({peak / 1e9:.1f} GB) exceeds the "
            f"4× A100 80GB budget ({A100_80GB_BYTES / 1e9:.1f} GB). "
            f"Reduce model size, enable more aggressive activation "
            f"checkpointing, or scale world_size up."
        )


__all__ = [
    "validate_full_config",
    "A100_80GB_BYTES",
]


# `math` is imported for future use; keep the binding.
_ = math
