"""Extended tokenizer (BPE-64k with byte-level fallback, Phase 4 implementation).

Trains a BPE-64k tokenizer and adds 256 byte-level fallback tokens for out-of-vocabulary handling.
"""

from __future__ import annotations

from pathlib import Path

from tokenizers import Tokenizer, pre_tokenizers
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer


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
    """Train a BPE-64k tokenizer from a list of text samples."""
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
    """Encode text to token IDs and strings with byte-level fallback for <unk>."""
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


class ExtendedTokenizer:
    """BPE-64k tokenizer with byte-level fallback for out-of-vocabulary tokens (IDs 64,000-64,255)."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._base: Tokenizer | None = None

    def load(self) -> ExtendedTokenizer:
        """Load tokenizer vocabulary and add special/fallback byte tokens."""
        if not self.path.exists():
            raise RuntimeError(f"Tokenizer file not found: {self.path}")
        base = Tokenizer.from_file(str(self.path))
        base.add_special_tokens(
            [f"<{s}>" for s in ("unk", "s", "/s", "pad", "mask")]
        )
        byte_tokens = [f"<0x{b:02X}>" for b in range(256)]
        base.add_tokens(byte_tokens)
        self._base = base
        return self

    def encode(self, text: str) -> list[int]:
        """Encode text to a list of token IDs with byte-fallback support."""
        if self._base is None:
            self.load()
        assert self._base is not None
        ids, _ = _byte_fallback_encode(self._base, text)
        return ids

    def decode(self, ids: list[int]) -> str:
        """Decode token IDs back to a string, converting byte tokens back to characters."""
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
