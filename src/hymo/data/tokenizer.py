"""Extended tokenizer (BPE-64k with byte-level fallback, Phase 1 placeholder).

The real implementation (architecture doc §6.1, roadmap A4) trains a
BPE-64k tokenizer and adds 256 byte-level tokens for OOV handling.
Code tokens like ``__init__`` are in the BPE vocab; rare identifiers
fall back to byte sequences.

This placeholder defines the :class:`ExtendedTokenizer` class with
the right API; the load / encode / decode bodies raise
:class:`NotImplementedError_`.
"""

from __future__ import annotations

from pathlib import Path

from hymo.core.exceptions import NotImplementedError_
from hymo.registry import TOKENIZERS

__all__ = ["ExtendedTokenizer", "BYTE_VOCAB_SIZE"]


BYTE_VOCAB_SIZE = 256  # 256 byte tokens; appended to the BPE vocab.


@TOKENIZERS.register("hymo-bpe-64k")
class ExtendedTokenizer:
    """BPE-64k tokenizer with byte-level fallback (Phase 1 placeholder).

    Architecture doc §6.1. Phase 1 placeholder.

    Parameters
    ----------
    path : str or Path
        Path to the saved tokenizer JSON.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._loaded = False

    def load(self) -> ExtendedTokenizer:
        """Load the tokenizer from disk.

        Phase 1 placeholder — raises :class:`NotImplementedError_`.
        """
        raise NotImplementedError_(
            "ExtendedTokenizer.load is a Phase 1 placeholder; the real "
            "implementation lands in Phase 4 (design §6.1, roadmap A4)."
        )

    def encode(self, text: str) -> list[int]:
        """Encode text to a list of token IDs.

        Phase 1 placeholder — raises :class:`NotImplementedError_`.
        """
        raise NotImplementedError_(
            "ExtendedTokenizer.encode is a Phase 1 placeholder; the real "
            "implementation lands in Phase 4 (design §6.1, roadmap A4)."
        )

    def decode(self, ids: list[int]) -> str:
        """Decode a list of token IDs back to text.

        Phase 1 placeholder — raises :class:`NotImplementedError_`.
        """
        raise NotImplementedError_(
            "ExtendedTokenizer.decode is a Phase 1 placeholder; the real "
            "implementation lands in Phase 4 (design §6.1, roadmap A4)."
        )

    @property
    def vocab_size(self) -> int:
        """The vocab size: 64,000 BPE + 256 bytes = 64,256."""
        return 64_000 + BYTE_VOCAB_SIZE

    @property
    def eos_token_id(self) -> int:
        return 0

    @property
    def pad_token_id(self) -> int:
        return 2
