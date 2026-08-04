# 12 — Tokenization and the 40× Params-in-Tokens Rule

> **Bridges to:** [`learning_docs/2_Data_Pipeline.md`](../../learning_docs/2_Data_Pipeline.md) (entire)

## Learning objectives

After this file, you can:

1. State BPE tokenization and why it's the modern default.
2. Explain byte-level fallback and the 64,256 vocab choice.
3. State the 40× params-in-tokens rule and its provenance.
4. Defend HyMo's data mixture (10 sources, 30 B tokens at
   750 M active).

## Intuition

A neural network operates on **integer token IDs**, not on
text. The tokenization step maps text to a sequence of
integers in a fixed vocabulary.

Three common approaches:

| Method | Vocab size | Pros | Cons |
|---|---|---|---|
| **Word-level** | 100 k–1 M | Simple, interpretable | Huge vocab; rare-word OOV. |
| **BPE** (byte-pair encoding) | 32 k–64 k | Compromise; handles rare words via subwords | Pre-tokenizer matters |
| **Byte-level** | 256 + special | No OOV ever | Long sequences |

HyMo uses **BPE-64k** (a BPE vocabulary of 64 k tokens) plus
**256 byte-level fallback tokens** for OOV. Total vocab:
`64,256`.

### BPE basics

BPE starts from a character-level vocab and iteratively
merges the most-frequent adjacent pairs into new tokens.
After `V - 256` merges, you have a `V`-sized vocab.

```
text:  "the cat sat on the mat"
tokens: ["the", "cat", "sat", "on", "the", "mat"]
       (each is a BPE token, learned from data)
```

The most common short words become single tokens; rarer or
longer words get split into subwords. With `V = 64_000`,
the median word is a single token and the long tail of
rarer words is multi-token.

### Byte-level fallback

A BPE tokenizer can only emit tokens it knows. If it sees
an unknown word (e.g. a foreign script), it maps to `<unk>`
and the information is lost.

**Byte-level fallback** adds 256 tokens, one per UTF-8 byte
value (`<0x00>`, ..., `<0xFF>`). When the BPE tokenizer
would emit `<unk>`, the encoder falls back to the byte tokens
for the original UTF-8 bytes.

With BPE-64k + 256 byte tokens, **every UTF-8 string is
losslessly encodable**. The model never sees `<unk>`.

## Math derivation

### BPE merge count

For a corpus of `C` characters, the merge algorithm runs
until `|V|` merges are done. Each merge replaces two adjacent
tokens with a new one, halving the sequence length. The
final sequence length is `~C / log_{|V|}(C / |V|)` — much
shorter than character-level.

### Compression ratio

A 64 k BPE typically achieves ~4 characters per token on
English text. For code, ~3 characters per token (because
of long identifiers and whitespace patterns). For multilingual
text, ~2 characters per token.

With `vocab_size = 64_256` and `max_seq_len = 4_096`, the
"characters per context window" is `~ 4 * 4096 = 16 K`
characters of English text.

### Tokens-to-params ratio

The standard Chinchilla rule (Hoffmann et al. 2022) was
**20 tokens per parameter** at training compute optimum.
Modern frontier practice (Llama-3, DeepSeek-V3) uses
**40 tokens per parameter** — over-training, on the
assumption that more tokens = better quality, even at the
expense of compute.

For HyMo at 750 M active params:

- 20× Chinchilla: `15 B tokens`
- 40× over-training: `30 B tokens`

The v1.0 ships with 30 B tokens — the over-training budget.
Quality wins from extra tokens: ~5-10% better than 20× at
the same architecture (Llama-3's published comparison).

### The mixture (10 sources)

HyMo trains on 10 source corpora with weighted mixing:

| Source | Role |
|---|---|
| FineWeb-Edu (filtered `score >= 3`) | High-quality English web |
| FineWeb (non-edu) | English web breadth |
| Stack v2 Python | Code |
| Stack v2 Java | Code |
| Stack v2 C++ | Code |
| SlimPajama | Multi-source (books, wiki, web) |
| DCLM Baseline | High-quality web |
| Dolma Wiki | Encyclopedic |
| Dolma Books | Long-form text |
| Cosmopedia | Synthetic textbook |

Weights are tuned in `configs/hymo_mixture.yaml`. Roughly:

- ~50% FineWeb-Edu (the high-quality backbone).
- ~10% code (Stack v2 sum).
- ~10% books + wiki.
- ~30% other (FineWeb non-edu, SlimPajama, DCLM,
  Cosmopedia).

The mixture is loaded by `DataConfig` (see
`learning_docs/2_Data_Pipeline.md` §2).

## Implementation in HyMo

- `src/hymo/data/tokenizer.py:16-19` — module-level constants:
  `BYTE_VOCAB_SIZE = 256`, `_BASE_VOCAB_SIZE = 64_000`,
  `_TOTAL_VOCAB_SIZE = 64_256`.
- `src/hymo/data/tokenizer.py:22` — `train_bpe_tokenizer`:
  the BPE training entry point.
- `src/hymo/data/tokenizer.py:41` — `_byte_fallback_encode`:
  the OOV fallback.
- `src/hymo/data/tokenizer.py:61` — `class ExtendedTokenizer`:
  the user-facing wrapper.
- `src/hymo/data/sources.py` — 10 streaming loaders.
- `src/hymo/data/sharding.py:25` — `ShardWriter`: writes
  50 M-token flat shards.
- `src/hymo/data/sharding.py:66` — `ShardDataset`: lazy
  memmap dataset.
- `src/hymo/data/sharding.py:116` — `DataLoaderBuilder`:
  builds the PyTorch DataLoader.
- `src/hymo/data/data_config.py:29` — `SourceSpec`.
- `src/hymo/data/data_config.py:128` — `DataConfig`.
- `configs/hymo_mixture.yaml` — the mixture file.

## Worked example

Production scale:

- 30 B tokens × 4 bytes/token (uint32) = 120 GB of
  shard data.
- 30 B tokens × ~4 characters/token (English) = 120 B
  characters = ~240 GB of raw text (avg 1 byte per
  character).
- ~600 shards at 50 M tokens each.
- Train time: 30 B / 524_288 tokens/step = 57,220
  optimizer steps.
- Wall-clock: 57,220 × 8 s/step ≈ 5-7 days on 4× A100
  80 GB SXM.

Per-token FLOPs (forward + backward, dense + MoE):

- MLA: ~70 GFLOPs / 4096 tokens = 17 MFLOPs/token.
- GDN: ~80 GFLOPs / 4096 tokens = 20 MFLOPs/token.
- MoE: 8 layers × 12 MFLOPs = 96 MFLOPs/token.
- Dense FFN: 24 layers × 6.2 MFLOPs = 149 MFLOPs/token.
- Head: 57.5 MFLOPs/token.
- Embed: 896 MAC/token (negligible).
- Total: ~340 MFLOPs/token.

Per-token wall-clock: 340 MFLOPs / 330 TFLOPs/s = ~1 µs.
With FSDP + Triton + torch.compile, real per-token: ~3 µs.
Per micro-batch (B=4, T=4096): ~50 ms. Per optimizer step:
~8 s. Per 57,220 steps: ~127 hours = ~5.3 days.

## Interview Q&A

**Q1. Why 64 k BPE + 256 byte fallback?**

> A: 64 k is the modern standard for English + code +
> multilingual coverage. The 256 byte tokens cover every
> UTF-8 byte; with byte fallback, every string is
> losslessly encodable. Total 64,256 is what
> `ExtendedTokenizer` emits.

**Q2. Why 30 B tokens and not 15 B or 60 B?**

> A: 30 B is 40× params-in-tokens at 750 M active — the
> over-training budget used by Llama-3 and DeepSeek-V3.
> 15 B (20×) would be Chinchilla-optimal; 60 B would be
> 80×, which over-trains at the cost of compute. 40× is
> the empirical sweet spot.

**Q3. Why is the FineWeb-Edu quality threshold 3?**

> A: FineWeb-Edu scores 0-5 per document; threshold 3 keeps
> the upper half of quality. Higher thresholds (4-5) lose
> too much data; lower thresholds (1-2) admit more noise.
> 3 is the Llama-3 / DeepSeek-V3 default.

**Q4. Why 10 sources and not 1?**

> A: Diversity. Different sources contribute different
> capabilities: FineWeb-Edu is general English; Stack v2 is
> code; Dolma Books is long-form; Cosmopedia is synthetic
> instructional text. A mixture of 10 covers all the major
> capabilities a 750 M model can absorb.

**Q5. Why 50 M tokens per shard?**

> A: 50 M tokens = 200 MB per shard (uint32 = 4 bytes).
> This is large enough that file I/O overhead is amortized
> but small enough that memmap pages can be paged in fast.
> At 4096 tokens per `__getitem__`, each access reads ~16 KB
> = 1 page on most filesystems.

**Q6. Why `np.memmap` instead of loading all shards into RAM?**

> A: 30 B tokens × 4 bytes = 120 GB. RAM is typically
> 100 GB on a single-host dev machine. `np.memmap` lets the
> OS page in only the 16 KB chunk per `__getitem__` and
> discard it immediately. The cost is per-`__getitem__` page
> faults, which is fast enough to keep up with A100
> consumption.

**Q7. Why byte-level fallback rather than a larger BPE
vocab?**

> A: A larger BPE vocab (e.g. 128 k) would cover more
> Unicode scripts but at the cost of a bigger embedding
> table (each row is `dim` floats). The byte fallback
> covers all of UTF-8 with only 256 extra tokens — much
> cheaper than a 64 k-vocab expansion.

## Cross-links

- [`learning_docs/2_Data_Pipeline.md`](../../learning_docs/2_Data_Pipeline.md) (entire
  walkthrough).
- [`docs/HyMo-Design.md`](../../docs/HyMo-Design.md) §6 (data
  mixture rationale).
- [`concepts/01-attention.md`](01-attention.md) — attention
  FLOPs per token.
- [`concepts/02-linear-attention-gdn.md`](02-linear-attention-gdn.md) —
  linear-attention FLOPs per token.
