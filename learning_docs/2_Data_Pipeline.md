# HyMo Data Pipeline — Code Walkthrough

> **Prerequisite reading:** none (data pipeline is logically first).
>
> **Files covered:**
> - `src/hymo/data/sources.py` — 10 source loaders
> - `src/hymo/data/tokenizer.py` — `ExtendedTokenizer`, byte-level fallback
> - `src/hymo/data/sharding.py` — `ShardWriter`, `ShardDataset`, `DataLoaderBuilder`
> - `src/hymo/data/data_config.py` — `DataConfig`, `SourceSpec`, `load_data_config`
> - `src/hymo/data/prepare_validation.py` — `build_val_set` (held-out val binary)
>
> **Companion concepts:** [`docs/concepts/12-tokenization-data.md`](../docs/concepts/12-tokenization-data.md)
> for the BPE / byte-fallback derivation and the 40× params-in-tokens
> rule.

> **No `data/prepare_data.py`** — that path was referenced in earlier
> docs as a placeholder. The shipped data pipeline is constructed
> from the modules above; there is no monolithic `prepare_data.py`
> CLI for the v1.0 primary run.

---

## Table of Contents

1. [Pipeline at a glance](#1-pipeline-at-a-glance)
2. [Data config (`data_config.py`)](#2-data-config-data_configpy)
3. [Source loaders (`sources.py`)](#3-source-loaders-sourcespy)
4. [Tokenizer (`tokenizer.py`)](#4-tokenizer-tokenizerpy)
5. [Sharding (`sharding.py`)](#5-sharding-shardingpy)
6. [Validation set (`prepare_validation.py`)](#6-validation-set-prepare_validationpy)
7. [End-to-end flow](#7-end-to-end-flow)
8. [Interview Q&A](#8-interview-qa)

---

## 1. Pipeline at a glance

```
   ┌─────────────────────────────┐
   │   10 HF source loaders      │   sources.py
   │   (streaming, filtered)     │   ─────────────
   │   FineWeb-Edu, FineWeb,     │   load_fineweb_edu(),
   │   Stack v2 (py/java/cpp),   │   load_stack_python(),
   │   SlimPajama, DCLM,         │   load_slimpajama(), …
   │   Dolma wiki+books,         │
   │   Cosmopedia                │
   └────────────┬────────────────┘
                │  stream of {text: ...}
                ▼
   ┌─────────────────────────────┐
   │   ExtendedTokenizer         │   tokenizer.py
   │   (BPE-64k + 256 byte       │   ─────────────
   │   fallback tokens)          │   encode(text) → list[int]
   └────────────┬────────────────┘
                │  stream of token IDs (uint32)
                ▼
   ┌─────────────────────────────┐
   │   ShardWriter               │   sharding.py
   │   (50M-token flat shards)   │   ─────────────
   │                             │   write_batched(token_stream)
   │                             │   → shard_00000.bin, …
   └────────────┬────────────────┘
                │  uint32 binary files
                ▼
   ┌─────────────────────────────┐
   │   ShardDataset + DataLoaderBuilder  │
   │   (zero-copy memmap,        │
   │    sliding 4k windows,      │
   │    multi-worker prefetch)   │
   └────────────┬────────────────┘
                │  (tokens, targets) batches
                ▼
          ┌──────────┐
          │ Trainer  │
          └──────────┘
```

For **validation**, a parallel `build_val_set()` produces
`data/tokens/val.bin` — a 450 M-token FineWeb-Edu held-out shard
read by `compute_validation_loss`.

---

## 2. Data config (`data_config.py`)

The data pipeline has its own config separate from the model
config. It's a single tree:

```python
class SourceSpec:        # one source: name, kind, weight, …
class ShardingConfig:    # shard_size, directory, format
class TokenizationConfig # tokenizer path, vocab size, byte fallback
class DedupConfig:       # minhash / exact / none
class QualityConfig:     # fineweb-edu score threshold, etc.
class DataConfig:        # aggregate (sources, sharding, …)
```

`load_data_config(path)` and `load_data_config_from_dict(raw)` are
the YAML loaders. The data config is **separate** from the model
config (which lives in `configs/hymo_750m.yaml`) because data
preparation is a phase-independent offline step — the trainer
loads only the model config at runtime; the data config drives
the offline shard build.

### 2.1 `SourceSpec` (line 29)

```python
@dataclass(frozen=True)
class SourceSpec:
    name: str                 # source display name
    kind: str                 # "fineweb_edu" | "fineweb" | "stack_python" | …
    weight: float             # mixture weight (sums to 1.0 across all)
    quality_threshold: int = 0
```

A single source in the mixture with its weight. The 10 `load_*`
functions in `sources.py` are paired one-to-one with `kind`
strings; passing a `SourceSpec` to the orchestrator routes it to
the right loader.

### 2.2 `DataConfig` (line 128)

```python
@dataclass(frozen=True)
class DataConfig:
    sources: tuple[SourceSpec, ...]   # 10 entries with weights
    sharding: ShardingConfig
    tokenization: TokenizationConfig
    dedup: DedupConfig
    quality: QualityConfig
```

`load_data_config("configs/hymo_mixture.yaml")` builds one of
these.

---

## 3. Source loaders (`sources.py`)

10 streaming loaders, all using `datasets.load_dataset(..., streaming=True)`.
Each yields `dict` rows with a single key: `"text"`.

| Loader (line) | Source | HF Dataset | Filter / Map |
|---|---|---|---|
| `load_fineweb_edu` (28) | FineWeb-Edu | `HuggingFaceFW/fineweb-edu`, `sample-10BT`, `train` | `score >= quality_threshold` (default 3); rename → `text` |
| `load_fineweb` (45) | FineWeb (non-edu) | `HuggingFaceFW/fineweb`, `sample-10BT`, `train` | rename → `text` |
| `load_stack_python` (58) | Stack v2 Python | `bigcode/the-stack-v2-dedup`, `data/python`, `train` | rename `content` → `text` |
| `load_stack_java` (71) | Stack v2 Java | `bigcode/the-stack-v2-dedup`, `data/java`, `train` | rename `content` → `text` |
| `load_stack_cpp` (84) | Stack v2 C++ | `bigcode/the-stack-v2-dedup`, `data/cpp`, `train` | rename `content` → `text` |
| `load_slimpajama` (97) | SlimPajama | `cerebras/SlimPajama-627B`, `train` | rename → `text` |
| `load_dclm_baseline` (109) | DCLM | `mlfoundations/dclm-baseline-1.0`, `train` | rename → `text` |
| `load_dolma_wiki` (121) | Dolma wiki | `allenai/dolma`, `data/wikipedia`, `train` | rename → `text` |
| `load_dolma_books` (134) | Dolma books | `allenai/dolma`, `data/books`, `train` | rename → `text` |
| `load_cosmopedia` (147) | Cosmopedia | `HuggingfaceTB/cosmopedia`, `train` | rename → `text` |

All ten use the same skeleton:

```python
def load_<name>(**kwargs) -> Iterator[dict[str, Any]]:
    ds = load_dataset("<repo>", name=<variant>, split="train",
                      streaming=True, trust_remote_code=True)
    ds = ds.map(lambda r: {"text": r[...]}, remove_columns=[...])
    yield from ds
```

The `filter` (only `load_fineweb_edu` has one) and the `map` are
both **lazy** — they apply per row as the data streams. With
`streaming=True`, the loader never holds more than one batch in
memory; that's what lets us buffer 30 B tokens on a single CPU.

The `trust_remote_code=True` is required for `fineweb-edu` (it has
a custom loading script); the other repos are pure data and
don't need it but it's set consistently.

---

## 4. Tokenizer (`tokenizer.py`)

A BPE-64k tokenizer with **byte-level fallback** for OOV.

### 4.1 Vocab layout

```
 ┌─────────────────────────────────┬────────────────────────────┐
 │  IDs 0..63,999                   │  IDs 64,000..64,255         │
 │  ──────                          │  ──────                     │
 │  BPE tokens learned from text    │  256 byte-level tokens     │
 │  + 5 special tokens:             │  <0x00>, <0x01>, ...,      │
 │    <unk>=0, <s>=1, </s>=2,       │  <0xFF>                    │
 │    <pad>=3, <mask>=4             │  (one per byte value)      │
 └─────────────────────────────────┴────────────────────────────┘
                                          64,256 = vocab_size
```

Constants in `tokenizer.py`:

```python
BYTE_VOCAB_SIZE = 256
_BYTE_TOKENS    = [f"<0x{b:02X}>" for b in range(256)]
_BASE_VOCAB_SIZE = 64_000
_TOTAL_VOCAB_SIZE = _BASE_VOCAB_SIZE + BYTE_VOCAB_SIZE  # 64_256
```

The 5 special tokens (`<unk>`, `<s>`, `</s>`, `<pad>`, `<mask>`)
are added after training and take 5 of the 64 k BPE slots; the
rest are content tokens.

### 4.2 `train_bpe_tokenizer(texts, *, vocab_size=64_000, output_path=...)` (line 22)

Trains a BPE tokenizer from `texts` using HuggingFace
`tokenizers`:

```python
def train_bpe_tokenizer(texts, *, vocab_size=_BASE_VOCAB_SIZE,
                       output_path="data/tokens/byte_bpe_vocab.json"):
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
```

The `ByteLevel` pre-tokenizer splits text on Unicode byte boundaries
before BPE sees it — this is what makes BPE deterministic across
spaces, tabs, etc. (the GPT-2 trick). `add_prefix_space=True`
matches how GPT-2's pretokenizer handles leading spaces.

### 4.3 `_byte_fallback_encode(base_tokenizer, text) → (ids, tokens)` (line 41)

Walk the BPE encoding; for any `<unk>`, fall back to 1–4 byte-level
tokens:

```python
def _byte_fallback_encode(base_tokenizer, text):
    encoding = base_tokenizer.encode(text)
    ids, tokens = [], []
    for token_id, token_str in zip(encoding.ids, encoding.tokens):
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
```

This guarantees **every** UTF-8 string is losslessly encodable as a
sequence of token IDs in `[0, 64_256)`. The model can never emit an
"OOV" output. HyMo uses ID 0 (`<unk>`) rarely in practice — BPE
training produces 64 k content tokens that cover English + code
+ multilingual text.

### 4.4 `ExtendedTokenizer` (line 61)

The user-facing wrapper:

```python
class ExtendedTokenizer:
    """BPE-64k tokenizer with byte-level fallback for OOV tokens (IDs 64,000-64,255)."""

    def __init__(self, path):
        self.path = Path(path)
        self._base = None

    def load(self):
        base = Tokenizer.from_file(str(self.path))
        base.add_special_tokens([f"<{s}>" for s in ("unk","s","/s","pad","mask")])
        base.add_tokens([f"<0x{b:02X}>" for b in range(256)])
        self._base = base
        return self

    def encode(self, text) -> list[int]:
        if self._base is None:
            self.load()
        ids, _ = _byte_fallback_encode(self._base, text)
        return ids

    def decode(self, ids) -> str:
        if self._base is None:
            self.load()
        chunks = []
        for tid in ids:
            if tid >= _BASE_VOCAB_SIZE:
                byte_val = tid - _BASE_VOCAB_SIZE
                if 0 <= byte_val < 256:
                    chunks.append(bytes([byte_val]).decode("utf-8", errors="replace"))
            else:
                s = self._base.id_to_token(tid)
                if s is not None:
                    chunks.append(s)
        return "".join(chunks).replace("</s>","").replace("<s>","")

    @property
    def vocab_size(self):
        return _TOTAL_VOCAB_SIZE         # 64_256
    @property
    def eos_token_id(self):
        return 0                          # <unk>, by convention
    @property
    def pad_token_id(self):
        return 2                          # </s>, by convention
```

Properties (`vocab_size`, `eos_token_id`, `pad_token_id`) make the
class compatible with HF `tokenizer` API expectations.

(Note: `eos_token_id=0` is `<unk>` in this scheme — slightly
unconventional. The model is fine to use 0 as EOS; the tokenizer
falls back gracefully if it sees `<unk>` during decoding. Don't
confuse this with the auto-tokens used in `models/hymo/...` — the
model doesn't have an explicit EOS in the architecture; it's a
training-side convention.)

---

## 5. Sharding (`sharding.py`)

Three classes, all centering on the **flat uint32 binary shard**
format.

### 5.1 `ShardWriter` (line 25)

A writer for one or more shards, each containing up to
`shard_size_tokens` (default **50 M**) token IDs.

```python
class ShardWriter:
    def __init__(self, output_dir="data/shards", shard_size_tokens=50_000_000):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.shard_size_tokens = shard_size_tokens

    def write_shard(self, shard_idx, tokens) -> Path:
        path = self.output_dir / f"shard_{shard_idx:05d}.bin"
        tokens.astype(np.uint32).tofile(path)
        return path

    def write_batched(self, token_stream) -> list[Path]:
        paths, shard_idx, offset = [], 0, 0
        while offset < len(token_stream):
            chunk = token_stream[offset : offset + self.shard_size_tokens]
            if len(chunk) < self.shard_size_tokens:
                pad = self.shard_size_tokens - len(chunk)
                chunk = np.pad(chunk, (0, pad), constant_values=0)
            paths.append(self.write_shard(shard_idx, chunk))
            shard_idx += 1
            offset += self.shard_size_tokens
        return paths
```

Two design points:

1. **Fixed-size shards** — every shard is exactly `shard_size_tokens`
   uint32s (= 200 MB at 4 bytes/token, for 50 M). The trailing
   shard is **zero-padded** to keep the file size uniform, which
   simplifies `np.memmap` reads in `ShardDataset`.
2. **Filenames are zero-padded** — `shard_00000.bin`,
   `shard_00001.bin`, … — so `glob("shard_*.bin")` returns sorted
   order without any string compare.

### 5.2 `ShardDataset` (line 66)

A PyTorch `Dataset` that lazily maps over all shards and yields
`(tokens, targets)` sliding windows of `max_seq_len + 1` (so
`tokens[:-1]` and `targets = tokens[1:]` give the next-token
training pair).

```python
class ShardDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, shards_dir="data/shards", max_seq_len=4_096):
        self.shards_dir = Path(shards_dir)
        self.max_seq_len = max_seq_len
        self._shard_paths: list[Path] = _discover_shards(self.shards_dir)
        self._shard_data: list[np.memmap] = []
        self._total_tokens: int = 0
        self._load_all()

    def _load_all(self):
        for sp in self._shard_paths:
            data = np.memmap(sp, dtype=np.uint32, mode="r")
            self._shard_data.append(data)
            self._total_tokens += len(data)
```

`np.memmap` is **zero-copy**: it tells NumPy to treat the file as
a length-`_total_tokens` uint32 array, but the OS doesn't actually
load the file into RAM. Pages are faulted in on access. This is
what makes 30 B tokens fit on a single CPU: the OS pages in
`max_seq_len + 1` uint32s at a time, ~16 KB at the production
shape.

The dataset is **circular across shards** — if a window crosses
a shard boundary, the next shard's data is concatenated (line
107-110). This means `__len__() = total_tokens // (max_seq_len + 1)`
gives the number of full windows available; partial trailing
windows are silently absorbed into the wrap-around.

`__getitem__(idx)`:

```python
def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
    si, offset = self._locate(idx)
    data = self._shard_data[si]
    chunk = data[offset : offset + self.max_seq_len + 1]
    if len(chunk) < self.max_seq_len + 1:
        # Cross-shard wrap-around
        remainder = (self.max_seq_len + 1) - len(chunk)
        next_si = (si + 1) % len(self._shard_data)
        chunk = np.concatenate([chunk, self._shard_data[next_si][:remainder]])
    tokens = torch.from_numpy(chunk[: self.max_seq_len].astype(np.int64))
    targets = torch.from_numpy(chunk[1 : self.max_seq_len + 1].astype(np.int64))
    return tokens, targets
```

Two notes:

- **Linear scan to locate the shard** — `_locate(idx)` walks
  `self._shard_data` subtracting sizes. For 600 shards (30 B
  tokens / 50 M per shard), this is 600 element compares per
  `__getitem__`, which PyTorch DataLoader workers eat for breakfast
  (a million `__getitem__` calls / second is typical). If a larger
  scale needed binary search, swap the linear scan for
  `bisect.bisect_left` on a cumulative-length array.
- The cast to `np.int64` is what PyTorch expects (`torch.long` ==
  `int64`). The underlying file is uint32 (we don't need negative
  IDs), but token IDs in `[0, 64_256)` fit in 17 bits.

### 5.3 `DataLoaderBuilder` (line 116)

Builds a `torch.utils.data.DataLoader` with HyMo's specific
multi-worker settings:

```python
class DataLoaderBuilder:
    def __init__(self, dataset: ShardDataset, config: TrainingConfig):
        self.dataset = dataset
        self.config = config

    def build(self) -> DataLoader:
        effective_batch = self.config.micro_batch_size
        sampler = torch.utils.data.RandomSampler(
            self.dataset, replacement=True,
            num_samples=(
                self.config.gradient_accumulation_steps
                * self.config.world_size
                * effective_batch * 100
            ),
        )
        cpu_count = os.cpu_count() or 0
        num_workers = max(0, min(8, cpu_count // max(1, self.config.world_size)))
        return DataLoader(
            self.dataset,
            batch_size=effective_batch,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
            prefetch_factor=4 if num_workers > 0 else None,
            drop_last=True,
            persistent_workers=num_workers > 0,
        )
```

Two interesting design choices:

1. **`replacement=True`** — the dataloader samples with replacement.
   A non-replacement sampler would preclude the `num_samples`
   budget above being larger than `len(dataset)`; with replacement,
   we can size the per-step sample count to exactly
   `grad_accum * world_size * micro_batch * 100` so each dataloader
   yields ~100 effective batches before reshuffling.
2. **`num_workers = min(8, cpu_count // world_size)`** — the
   scaling is *per-GPU*, not total. With 4 ranks on a 32-CPU box,
   each rank gets `min(8, 8) = 8` workers; on a 4-CPU sandboxed
   box, each rank gets `min(8, 1) = 1` worker. The
   `num_workers=0` fallback (single-process loading) is what
   runs in CI without shared memory.

`pin_memory=True` enables faster CPU→GPU copies for the typical
disjoint-page pattern of next-token training windows.

---

## 6. Validation set (`prepare_validation.py`)

A separate one-shot CLI for building `data/tokens/val.bin`:

```python
def build_val_set(
    target_tokens: int = 450_000_000,
    seed: int = 42,
    tokenizer_path: str | Path = "data/tokens/byte_bpe_vocab.json",
    output_path: str | Path = "data/tokens/val.bin",
) -> None:
```

Pipeline:

1. Load the `ExtendedTokenizer`.
2. Stream `fineweb-edu/sample-10BT/train`.
3. `ds.shard(num_shards=20, index=0)` — take the first 5% as the
   **held-out** split (this is the v1.0 convention; the other 19
   shards are training data).
4. Tokenize each row and append to the running token list.
5. Stop at `target_tokens` (= 450 M default; the v1.0 value).
6. Save as a flat `np.uint32` binary file at `output_path`.

The result is consumed by `compute_validation_loss` in the
training pipeline (see `learning_docs/3_Training_Pipeline.md` §8).

---

## 7. End-to-end flow

A typical sequence at v1.0:

1. **Mix config:** edit `configs/hymo_mixture.yaml` if needed
   (weights for the 10 sources).
2. **Tokenize shard:** for each `SourceSpec`, call the corresponding
   `load_*` to stream documents, run them through `ExtendedTokenizer`
   in batches, accumulate tokens until `shard_size_tokens = 50 M`,
   write a shard via `ShardWriter.write_batched`.
3. **Validate val.bin exists:** `python -m
   hymo.data.prepare_validation` builds it.
4. **Wire the trainer:** `ShardDataset(shards_dir=...)` →
   `DataLoaderBuilder(dataset, training_config).build()` →
   pass the loader to `Trainer.train(data_iter)` (see
   `learning_docs/3_Training_Pipeline.md` §6.3).
5. **Run:** the dataloader serves `(tokens, targets)` windows;
   the trainer consumes them in `train_step`.

The 30 B-token run takes ~600 shards at 50 M each. With 4 A100
ranks and `num_workers=8` per rank, the loader reads ~16 KB per
`__getitem__` from memmap and the GPU never idles.

---

## 8. Interview Q&A

**Q1. Why BPE-64k + 256 byte-fallback = 64,256 vocab?**

> A: BPE-64k is the standard for English + code + multilingual
> training. The 256 byte tokens (IDs 64,000..64,255) cover every
> UTF-8 byte; the model can losslessly encode any input. Without
> byte-fallback, an OOV would map to `<unk>` and lose information.

**Q2. Why `np.memmap` instead of loading all shards into RAM?**

> A: 30 B tokens × 4 bytes/token = 120 GB. RAM is typically
> 100 GB on a single-host dev machine and 1+ TB on a training
> pod, but cheap dev machines don't have it. `np.memmap` lets
> the OS page in only the `max_seq_len + 1` token chunk we
> actually need (`~16 KB` at `max_seq_len=4096`) and discard it
> immediately. The cost is per-`__getitem__` page faults, which
> is fast enough that the data loader keeps up with A100
> consumption.

**Q3. Why fixed-size shards with zero-padding instead of variable-size?**

> A: Fixed shard size makes `np.memmap` reads trivial (every
> shard is the same length). Variable-size would require a
> length table per shard or a global index; both add complexity
> without throughput benefit. The zero-pad adds ~200 bytes to
> the last shard; the alternative (a small trailing shard) would
> require special-casing in `_locate`.

**Q4. Why `replacement=True` on the `RandomSampler`?**

> A: It avoids `num_samples > len(dataset)` errors when the
> budget per epoch is larger than the dataset. With
> `replacement=True`, the same `(tokens, targets)` window can
> be sampled more than once across an epoch — fine for
> training, where the model sees the same data many times
> anyway.

**Q5. Why does `ShardDataset.__getitem__` wrap across shards
(line 107-110)?**

> A: Without wrap-around, an example that lands at offset
> `(len(shard) - 10)` to `(len(shard) + 4086)` would have to be
> silently truncated to `max_seq_len` and the model would train
> on a partial window. Wrap-around reads the missing 10 tokens
> from the next shard; the model always gets a full
> `max_seq_len + 1` window.

**Q6. Why is the validation binary built from a `shard(index=0)` of
FineWeb-Edu rather than a held-out dataset?**

> A: FineWeb-Edu is the **only** dataset HyMo trains on that's
> clean enough to be a held-out set. The first 5% (shard 0 of 20)
> is reserved at training-corpus build time; the other 19 shards
> go into training. This guarantees the validation set has zero
> overlap with training.

**Q7. Why does `ShardWriter.write_batched` not preserve document
boundaries across shards?**

> A: It does preserve them *within* a shard (one document flows
> into the next), but the shard boundary is wherever the 50 M
> token count falls. There's no document alignment at the
> boundary because byte-packed token streams are inherently
> position-based; sampling a 4 k-token window straddling the
> boundary is fine because the model treats the stream as
> position-based anyway (positions are reset per-rank by FSDP,
> not per-document).

---

## 9. Cross-links

- Walkthrough: `learning_docs/3_Training_Pipeline.md` (trainer
  consumes the `DataLoader`), `learning_docs/1_Model_Architecture.md`
  §2 (model config), `learning_docs/5_Evaluation_and_Ablations.md`
  §6.1 (the validation binary is read by `compute_validation_loss`).
- Concepts: `docs/concepts/12-tokenization-data.md` (BPE / byte
  fallback derivation, 40× params-in-tokens rule).
- Tests: `tests/unit/test_data.py` (tokenizer round-trip, shard
  round-trip, dataset slicing).
- Config: `src/hymo/data/data_config.py` (`SourceSpec`,
  `ShardingConfig`, etc.); `configs/hymo_mixture.yaml` (the
  mixture file).
