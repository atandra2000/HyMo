"""Public model-building API for the HyMo architecture.

Exports include the complete stack plus individual blocks for focused inspection,
ablation, and component-level integration.
"""

from __future__ import annotations

from hymo.models.gdn import GatedDeltaNetBlock
from hymo.models.mla import MLABlock, MultiHeadLatentAttention
from hymo.models.model import HyMo, build_hymo
from hymo.models.moe import DeepSeekMoE, SwiGLUExpert
from hymo.models.mtp import MTPOutput, MultiTokenPrediction
from hymo.models.rope import RotaryEmbedding

__all__ = [
    # Stack
    "HyMo",
    "build_hymo",
    # GDN
    "GatedDeltaNetBlock",
    # MLA
    "MLABlock",
    "MultiHeadLatentAttention",
    # MoE / FFN
    "DeepSeekMoE",
    "SwiGLUExpert",
    # MTP
    "MTPOutput",
    "MultiTokenPrediction",
    # RoPE
    "RotaryEmbedding",
]
