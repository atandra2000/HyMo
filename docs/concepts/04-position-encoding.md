# 04 — Position Encoding: RoPE, Partial RoPE, NoPE-Hybrid

> **Bridges to:** [`learning_docs/1_Model_Architecture.md`](../../learning_docs/1_Model_Architecture.md) §7
> (RoPE)

## Learning objectives

After this file, you can:

1. State Rotary Position Embeddings (RoPE) and how they preserve
   relative position through a rotation.
2. Explain HyMo's "partial RoPE" choice (25% of `head_dim`).
3. Describe the NoPE-hybrid ablation (no PE on select GDN
   layers) and why it's deferred to v1.1.

## Intuition

The attention score between query `q_i` and key `k_j` depends on
the *content* of `q_i` and `k_j`. Without position information,
the model cannot distinguish "the cat sat on the mat" from
"mat the on sat cat the" — same tokens, different order.

Two main approaches:

| Approach | What it stores | Compute cost |
|---|---|---|
| **Absolute position embeddings** | A learned vector per position, added to the token embedding. | One extra `vocab_size + max_seq_len` embedding table. |
| **RoPE (Su et al. 2021)** | A learned rotation per head, applied to `q` and `k` per position. | Per-position rotation; no extra parameters. |
| **NoPE (no position encoding)** | None — the model has no explicit position signal. | Zero overhead. |

RoPE is the modern default. It encodes relative position
through an angle: at position `p`, query/key vectors are
rotated by an angle `p · θ_k` where `θ_k` is a per-dim
frequency. The attention score after rotation depends only on
the relative position `(i - j)`, not on the absolute
positions.

**Partial RoPE**: instead of applying RoPE to the full
`head_dim`, apply it to only the first `qk_rope_head_dim` of it.
The remaining `qk_nope_head_dim = head_dim - qk_rope_head_dim`
carries *no* position information.

```python
# HyMo at production scale:
# qk_rope_head_dim = 32
# qk_nope_head_dim = 96
# head_dim = 128 (= 32 + 96)
# 25% RoPE, 75% NoPE per head
```

### Why 25%?

Empirically (DeepSeek-V2, V3; also some Llama ablation work):
applying RoPE to fewer dimensions improves long-context
performance. The intuition: the head-dim budget is shared
between content and position; giving 75% of it to content lets
each head specialize more. The 25% is enough rope to learn
relative position patterns.

### The NoPE-hybrid (deferred)

An ablation that disables position encoding on **select GDN
layers** — specifically, the GDN layer immediately after each
MLA layer. The 7 affected positions for an 8-MLA stack are
`{3, 7, 11, 15, 19, 23, 27}`.

Why? A GDN layer with no position information relies entirely
on its recurrence to track "where" tokens are. The
MLA-then-GDN sandwich is a natural place to test this — the
MLA layer has its own position info (rotated by partial RoPE),
and the GDN that follows can either reuse it (via the hidden
state) or have no position info at all. The ablation family
`D_mqa4_vs_gqa175` and the NoPE-hybrid flag
(`nope_hybrid_gdn_enabled: false` in v1.0) are what gate this.

The risk was that "no position" hurts the model. The mitigation
in v1.0 is simple: ship with the flag **off** (everyone gets
partial RoPE), defer the test to v1.1.

## Math derivation

### RoPE

A 2D rotation by angle `θ`:

```
R(θ) = [cos θ, -sin θ]
       [sin θ,  cos θ]
```

For a `head_dim`-dim vector, RoPE applies a different angle to
each pair of dimensions. Standard convention:

```
θ_k = 1 / (rope_theta^(2k / head_dim))     k = 0, 1, ..., head_dim/2 - 1
```

At position `p`, the rotation per pair is `p · θ_k`. Query `q`
at position `p`:

```
q_rot = RoPE(q, p) = [q_0 cos(p θ_0) - q_1 sin(p θ_0),
                       q_0 sin(p θ_0) + q_1 cos(p θ_0),
                       q_2 cos(p θ_1) - q_3 sin(p θ_1),
                       ...]
```

Same for `k` at position `j`. The attention inner product
`q_rot · k_rot` depends on `(p - j)` (modulo `2π`), so the
model learns relative position patterns.

### Partial RoPE

Apply RoPE only to the first `qk_rope_head_dim` of `q` and `k`:

```
q_rot = [RoPE(q[:32], p),  q[32:]]         # 32 dims rotated, 96 dims not
k_rot = [RoPE(k[:32], j),  k[32:]]
```

The remaining `qk_nope_head_dim = 96` dims are vanilla content
attention. The attention inner product is:

```
q_rot · k_rot = RoPE(q[:32], p) · RoPE(k[:32], j)
              + q[32:128] · k[32:128]
```

The first term has relative-position dependence; the second
doesn't. This mixing is what gives partial RoPE its quality
advantage over full RoPE.

### NoPE

Set `qk_rope_head_dim = 0`. The full `head_dim` carries no
position information. The model must rely on:
- The previous MLA layer's hidden state (which has its own
  partial RoPE).
- The recurrence state of the GDN (which carries implicit
  position through the cumulative decay).
- The MoE router (which sees the same hidden state).

Empirically, partial NoPE-hybrid (mixing RoPE and NoPE across
layers) is competitive with full RoPE on most tasks but
*better* on long-context benchmarks. The v1.1 ablation will
test this.

## Implementation in HyMo

- `src/hymo/models/rope.py:17` — `class RotaryEmbedding`.
- `src/hymo/models/rope.py:20` — `__init__`: precomputes the
  `cos`, `sin` tables for `max_seq_len` positions and
  `head_dim / 2` freq pairs.
- `src/hymo/models/rope.py:47` — `apply_rope(x, positions, *,
  start_pos=0)`: applies the rotation to the first
  `qk_rope_head_dim` of `x`; leaves the rest unchanged.
- `src/hymo/models/rope.py:81` — `extra_repr`.

Wiring:

- `src/hymo/models/mla.py:91` — `forward(x)` calls
  `apply_rope(q_rope, positions)` for the first
  `qk_rope_head_dim = 32` of `q` and `k`.
- `src/hymo/models/gdn.py:123` — `forward(x)` calls
  `apply_rope` on the GDN's `b` and `c` keys (which have the
  same shape as MLA's `q_rope`).
- `src/hymo/models/model.py:45` — for each layer, if
  `i in nope_hybrid_gdn_positions`, the GDN block is built
  with `use_rope=False` (no `apply_rope` call).

The `use_rope` flag on `GatedDeltaNetBlock` is set in
`model.py:41-48`:

```python
for i in range(config.n_layers):
    if i in mla_positions:
        self.layers.append(MLABlock(config, layer_idx=i))
    else:
        use_rope = i not in nope_hybrid
        self.layers.append(
            GatedDeltaNetBlock(config, layer_idx=i, use_rope=use_rope)
        )
```

So `nope_hybrid_gdn_enabled: true` means the 7 GDN layers at
positions `{3, 7, 11, 15, 19, 23, 27}` are GDN-without-RoPE,
the other 17 GDN layers get partial RoPE.

## Worked example

Production scale (`configs/hymo_750m.yaml`):

- `head_dim = 128`, `qk_rope_head_dim = 32`, `qk_nope_head_dim = 96`
- `rope_theta = 10_000.0`
- 32 layers, 8 MLA + 24 GDN
- `nope_hybrid_gdn_enabled: false` (v1.0 default)

Per-layer positional info:

| Layer | Type | Pos info |
|---|---|---|
| 0 | MLA | 32-dim RoPE on `q, k` |
| 1, 2 | GDN | 32-dim RoPE on `b, c` |
| 3 | GDN | 32-dim RoPE on `b, c` (NoPE would be here if `nope_hybrid_gdn_enabled: true`) |
| 4 | MLA | 32-dim RoPE on `q, k` |
| ... | ... | ... |
| 28 | MLA | 32-dim RoPE on `q, k` |
| 29, 30, 31 | GDN | 32-dim RoPE on `b, c` |

If the NoPE-hybrid flag were flipped, layers 3, 7, 11, 15, 19,
23, 27 would not apply RoPE — the rest of the model looks
identical.

## Interview Q&A

**Q1. Why partial RoPE (25%) instead of full RoPE?**

> A: Empirically, full RoPE on 128-dim heads over-allocates
> capacity to position encoding. 25% gives enough rotational
> channels to learn relative positions while leaving 75% of
> the head dim for content signal. This is the DeepSeek-V2/V3
> choice.

**Q2. Why does the NoPE-hybrid defer to v1.1?**

> A: Risk reduction. The shipped model is the primary 30 B
> pre-training run; flipping the NoPE-hybrid flag without a
> prior ablation result is a multi-day experiment with no
> guarantee of payoff. The v1.0 ships with the flag off
> (everyone gets partial RoPE); the v1.1 ablation tests
> whether the 7 NoPE GDN layers help long-context tasks.

**Q3. Why is `rope_theta = 10_000` and not 500_000 (Llama-3)?**

> A: 10_000 is the original RoPE default; it gives reasonable
> extrapolation up to ~32 k context. 500_000 is the Llama-3
> "extended context" choice, which extrapolates to 128 k
> but adds no quality at the trained context. HyMo trains
> at `max_seq_len = 4_096`; 10_000 is sufficient for that
> range without wasted parameter capacity.

**Q4. Why apply RoPE to `b` and `c` in GDN, not just to `q` and
`k` in MLA?**

> A: The GDN writes-to-state via `b` and reads-from-state via
> `c`. If `b` and `c` don't carry position information, the
> model can't write a "memory at position p" or read "memory
> from position j" — the recurrence becomes position-agnostic,
> which hurts associative recall. Applying RoPE to `b, c`
> lets the GDN track timing through the recurrence.

**Q5. Why does `apply_rope` take a `start_pos` argument?**

> A: For inference with KV-cache reuse. When decoding at
> position `p`, the model only needs the rotation at `p`, not
> all positions. `start_pos` lets the kernel slice the cos/sin
> tables without recomputing. Currently `start_pos=0` always
> (no incremental inference in v1.0), but the argument is
> there for the v1.1 inference refactor.

**Q6. What's the cost of partial RoPE vs. full RoPE?**

> A: `apply_rope` always slices and rotates the first
> `qk_rope_head_dim` of the input. Whether that's 32 or 128
> dims, the cost is one rotation per dim pair (~30 ops per
> pair). The 75% NoPE portion is "free" — no rotation
> compute. Net: partial RoPE is *cheaper* than full RoPE.

## Cross-links

- [`learning_docs/1_Model_Architecture.md`](../../learning_docs/1_Model_Architecture.md) §7
  (RoPE walkthrough).
- [`concepts/01-attention.md`](01-attention.md) — how RoPE is
  applied to MLA's `q` and `k`.
- [`concepts/02-linear-attention-gdn.md`](02-linear-attention-gdn.md) —
  how RoPE is applied to GDN's `b` and `c`.
- [`concepts/11-hybrid-architectures.md`](11-hybrid-architectures.md) —
  the NoPE-hybrid as an ablation.
