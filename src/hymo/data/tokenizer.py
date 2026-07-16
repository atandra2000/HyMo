"""Extended tokenizer (BPE-64k with byte-level fallback, Phase 4 implementation).

Architecture doc §6.1, roadmap A4. Trains a BPE-64k tokenizer and adds
256 byte-level tokens for OOV handling. Code tokens like ``__init__`` are
in the BPE vocab; rare identifiers fall back to byte sequences.
"""

from __future__ import annotations

from pathlib import Path

from tokenizers import Tokenizer, pre_tokenizers
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

from hymo.core.exceptions import TokenizerError
from hymo.registry import TOKENIZERS

__all__ = ["ExtendedTokenizer", "BYTE_VOCAB_SIZE", "train_bpe_tokenizer"]


BYTE_VOCAB_SIZE = 256
_BYTE_TOKENS = [f"<0x{b:02X}>" for b in range(256)]
_BASE_VOCAB_SIZE = 64_000
_TOTAL_VOCAB_SIZE = _BASE_VOCAB_SIZE + BYTE_VOCAB_SIZE


def train_bpe_tokenizer(
    texts: list[str],
    *,
    vocab_size: int = _BASE_VOCAB_SIZE,
    output_path: str | Path = "data/tokens/byte_bpe_vocab.json",
) -> Tokenizer:
    """Train a BPE-64k tokenizer from text samples.

    Parameters
    ----------
    texts : list of str
        Training samples (typically a random subset of the data).
    vocab_size : int
        BPE vocab size (default 64,000).
    output_path : str or Path
        Where to save the trained tokenizer.

    Returns
    -------
    Tokenizer
        The trained tokenizer.
    """
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<unk>", "<s>", "</s>", "<pad>", "<mask>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tokenizer.train_from_iterator(texts, trainer=trainer)
    tokenizer.save(str(output_path))
    return tokenizer


def _byte_fallback_encode(
    base_tokenizer: Tokenizer, text: str
) -> tuple[list[int], list[str]]:
    """Encode text with byte-level fallback for OOV tokens.

    Returns
    -------
    ids : list of int
        Token IDs including byte-fallback tokens (> 64,000).
    tokens : list of str
        Corresponding token strings for debugging.
    """
    encoding = base_tokenizer.encode(text)
    ids: list[int] = []
    tokens: list[str] = []
    for token_id, token_str in zip(encoding.ids, encoding.tokens, strict=False):
        if token_id == base_tokenizer.token_to_id("<unk>"):
            for b in text.encode("utf-8"):
                byte_token = _BYTE_TOKENS[b]
                byte_id = _BASE_VOCAB_SIZE + b
                ids.append(byte_id)
                tokens.append(byte_token)
        else:
            ids.append(token_id)
            tokens.append(token_str)
    return ids, tokens


@TOKENIZERS.register("hymo-bpe-64k")
class ExtendedTokenizer:
    """BPE-64k tokenizer with byte-level fallback.

    Architecture doc §6.1. Wraps a ``tokenizers.Tokenizer`` and extends
    it with 256 byte-level tokens (IDs 64,000-64,255) for OOV handling.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._base: Tokenizer | None = None

    def load(self) -> ExtendedTokenizer:
        """Load the tokenizer from disk."""
        if not self.path.exists():
            raise TokenizerError(f"Tokenizer file not found: {self.path}")
        self._base = Tokenizer.from_file(str(self.path))
        self._base.add_special_tokens(
            [f"<{s}>" for s in ("unk", "s", "/s", "pad", "mask")]
        )
        byte_tokens = [f"<0x{b:02X}>" for b in range(256)]
        self._base.add_tokens(byte_tokens)
        return self

    def encode(self, text: str) -> list[int]:
        """Encode text to a list of token IDs with byte-level fallback."""
        if self._base is None:
            self.load()
        ids, _ = _byte_fallback_encode(self._base, text)
        return ids

    def decode(self, ids: list[int]) -> str:
        """Decode a list of token IDs back to text."""
        if self._base is None:
            self.load()
        assert self._base is not None
        chunks: list[str] = []
        for token_id in ids:
            if token_id >= _BASE_VOCAB_SIZE:
                byte_val = token_id - _BASE_VOCAB_SIZE
                if 0 <= byte_val < 256:
                    chunks.append(bytes([byte_val]).decode("utf-8", errors="replace"))
            else:
                token_str = self._base.id_to_token(token_id)
                if token_str is not None:
                    chunks.append(token_str)
        return "".join(chunks).replace("</s>", "").replace("<s>", "")

    @property
    def vocab_size(self) -> int:
        return _TOTAL_VOCAB_SIZE

    @property
    def eos_token_id(self) -> int:
        return 0

    @property
    def pad_token_id(self) -> int:
        return 2
