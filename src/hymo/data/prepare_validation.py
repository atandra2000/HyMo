"""Build the real held-out validation set (architecture doc §6.3).

Produces a flat binary shard from a held-out split of FineWeb-Edu.
"""

from __future__ import annotations

import argparse
import typing
from pathlib import Path

import numpy as np

from hymo.data.tokenizer import ExtendedTokenizer

if typing.TYPE_CHECKING:
    from tqdm import tqdm
else:
    try:
        from tqdm import tqdm
    except ImportError:
        class tqdm:
            def __init__(self, *args: typing.Any, **kwargs: typing.Any) -> None:
                pass

            def update(self, n: int = 1) -> None:
                pass

            def close(self) -> None:
                pass


def build_val_set(
    target_tokens: int = 450_000_000,
    seed: int = 42,
    tokenizer_path: str | Path = "data/tokens/byte_bpe_vocab.json",
    output_path: str | Path = "data/tokens/val.bin",
) -> None:
    """Build the held-out validation set by tokenizing FineWeb-Edu to a flat binary file."""
    from datasets import load_dataset

    tok = ExtendedTokenizer(tokenizer_path)
    tok.load()

    ds = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
        split="train",
        streaming=True,
        trust_remote_code=True,
    )
    ds = ds.shard(num_shards=20, index=0)
    ds = ds.map(lambda r: {"text": r["text"]}, remove_columns=[c for c in (ds.features or {}) if c != "text"])

    tokens: list[int] = []
    pbar = tqdm(total=target_tokens, desc="Tokenizing val set")
    for row in ds:
        encoded = tok.encode(row["text"])
        tokens.extend(encoded)
        pbar.update(len(encoded))
        if len(tokens) >= target_tokens:
            break
    pbar.close()

    arr = np.array(tokens[:target_tokens], dtype=np.uint32)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    arr.tofile(out)
    print(f"Wrote {len(arr):,} tokens to {out}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a held-out validation set from FineWeb-Edu."
    )
    parser.add_argument(
        "--target-tokens",
        type=int,
        default=450_000_000,
        help="Target number of validation tokens (default: 450M).",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="RNG seed for deterministic shard selection."
    )
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        default="data/tokens/byte_bpe_vocab.json",
        help="Path to the tokenizer JSON.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="data/tokens/val.bin",
        help="Output path for the validation binary.",
    )
    args = parser.parse_args()
    build_val_set(
        target_tokens=args.target_tokens,
        seed=args.seed,
        tokenizer_path=args.tokenizer_path,
        output_path=args.output_path,
    )


if __name__ == "__main__":
    main()
