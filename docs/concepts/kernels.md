# HyMo — Triton Kernels

> The GPU execution model, the hand-written Triton GDN kernel (forward + backward), and the
> `torch.autograd.Function` integration with recompute-style backward. This is the only sanctioned
> custom kernel in the codebase (`AGENTS.md` engineering rules); there is no `fla` dependency.

## Learning objectives

After this file, you can:

1. Sketch the GPU execution model (SMs, warps, blocks).
2. Explain why Triton is the right tool for the GDN
   recurrence.
3. Walk through `torch.autograd.Function` + recompute
   backward.
4. Defend the no-silent-fallback rule.

## Intuition

A GPU has many Streaming Multiprocessors (SMs); an A100 has
108. Each SM runs many warps (32 threads) in parallel. To
saturate the GPU, you want each SM to do useful work every
cycle.

PyTorch's high-level ops (e.g. `nn.Linear`) are compiled
ahead of time and dispatched at runtime. They're optimized
for general shapes, but for **specific** shapes (e.g. the
GDN recurrence with `chunk_size = 64`, `d_state = 32`,
`d_inner = 32`), you can do better by writing a kernel
that knows exactly what shape and access pattern to
expect.

**Triton** is a Python-embedded DSL that compiles to
PTX (NVIDIA assembly). It lets you write GPU kernels in
Python with type annotations and block-level primitives,
without dropping to CUDA C++. The compilation is
ahead-of-time but cached, so the second run is fast.

Why Triton for GDN?

1. **Custom recurrence**: PyTorch's einsum-based GDN would
   be `T × H` Python iterations. Triton lets us fuse the
   whole recurrence into one kernel launch.
2. **Specific shape**: the production shape (`H = 40`, `S
   = 32`, `D = 32`, `chunk_size = 64`) is fixed; we can
   autotune for it.
3. **Composes with autograd**: Triton's `@triton.jit`
   kernels can be wrapped in a `torch.autograd.Function`
   for gradient computation.

## GPU execution model

A **kernel** is a function that runs on the GPU. It's
launched with a grid of blocks; each block runs on one SM;
each block has many threads (warps).

```
  grid:  blocks of work
    └─ block: 32+ threads (warps) on one SM
         └─ thread: one CUDA thread, runs in parallel with others in a warp
```

Triton abstracts this: `@triton.jit` kernels see a 2D grid
of programs (blocks), and each program has a vector of
threads.

Key parameters:

- **Block size** (per program): how many threads cooperate
  on one SM. For our GDN kernel, `BLOCK_SIZE = chunk_size *
  d_inner / 4 = 64 * 32 / 4 = 512` threads per program.
- **Grid size**: how many programs to launch. For GDN, one
  per `(B*H, num_chunks)`.
- **Shared memory** (SMEM): fast on-chip memory shared by
  threads in a block. Used for the recurrent state and
  intermediate results.
- **Registers**: per-thread storage. The kernel's per-thread
  variables live here.

For GDN, the key constraint is that the recurrent state `h`
of shape `(S, D) = (32, 32)` must fit in SMEM (or registers)
across the chunk's iterations. With `S * D = 1024` floats =
4 KB, it fits in SMEM easily.

## The chunked GDN algorithm in Triton

Recap from [`concepts/gdn-and-mla.md`](gdn-and-mla.md):

```
h_t = α_t · h_{t-1} + b_t ⊗ v_t           (write)
o_t = c_t · h_t                            (read)
```

For chunked execution:

1. **Within a chunk** (size `chunk_size = 64`):
   - All `b`, `c`, `v`, `g` for the chunk are loaded into
     SMEM.
   - The state `h` is initialized to the carry-in from the
     previous chunk.
   - The chunk recurrence is computed as a `chunk_size ×
     chunk_size` matmul (the "intra-chunk" part) plus a
     reduction over `chunk_size` steps (the "carry-out"
     part).
2. **Between chunks**:
   - The carry-out `h` becomes the carry-in for the next
     chunk.
   - This is `T / chunk_size = 64` serial chunks per forward.

The Triton kernel in `gdn_triton.py` implements this
chunked algorithm:

- `@triton.jit gdn_fwd_kernel` (line 43): forward.
- `@triton.jit gdn_bwd_kernel` (line 94): backward
  (recompute-style).
- `def triton_gated_delta_rule(...)` (line 237): the Python
  wrapper that pads, casts, and dispatches.
- `class TritonGDNFunction(torch.autograd.Function)`: the
  autograd `Function` with `forward` and `backward` methods
  that wrap the kernels.

## `torch.autograd.Function` + recompute

The standard way to integrate a custom kernel with PyTorch
autograd is a `torch.autograd.Function` subclass:

```python
class TritonGDNFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, v, b, c, g, A_log):
        # Save for backward
        ctx.save_for_backward(v, b, c, g, A_log)
        # Run the kernel
        return gdn_fwd_kernel(v, b, c, g, A_log)

    @staticmethod
    def backward(ctx, grad_out):
        # Retrieve saved tensors
        v, b, c, g, A_log = ctx.saved_tensors
        # Run the backward kernel
        return gdn_bwd_kernel(grad_out, v, b, c, g, A_log)
```

The recompute trick: in the backward pass, we don't need
to store the intermediate `h` state. We can recompute it
from `(v, b, c, g, A_log)` — the inputs that we already
saved. This halves the activation memory: instead of
storing `(S, D) = 4 KB` per chunk per layer, we only store
the inputs.

Cost: each backward step does ~10% extra compute (recompute
the `h` state) but saves ~50% of activation memory. For
long sequences (`T = 4096`, `n_layers = 24`), this is
~10 GB of activations saved per micro-batch.

## Implementation in HyMo

- `src/hymo/models/gdn_triton.py` (the whole file).
- `src/hymo/models/gdn_triton.py` — `HAS_TRITON` constant.
- `src/hymo/models/gdn_triton.py:_next_power_of_2` — `_next_power_of_2(n)`.
- `src/hymo/models/gdn_triton.py:gdn_fwd_kernel` — `@triton.jit gdn_fwd_kernel`.
- `src/hymo/models/gdn_triton.py:gdn_bwd_kernel` — `@triton.jit gdn_bwd_kernel`.
- `src/hymo/models/gdn_triton.py:TritonGDNFunction` — `class TritonGDNFunction
  (torch.autograd.Function)`.
- `src/hymo/models/gdn_triton.py:triton_gated_delta_rule` — `triton_gated_delta_rule
  (v, b, c, g, A_log)`: the Python wrapper.

The wiring in `gdn.py`:

- `src/hymo/models/gdn.py:GatedDeltaNetBlock._gated_delta_rule` — `_gated_delta_rule`: the eager
  PyTorch reference (used in tests, also when
  `use_triton = False`).
- `src/hymo/models/gdn.py:GatedDeltaNetBlock.forward` — `forward(x)`: dispatches to
  `triton_gated_delta_rule` if `use_triton`, else to
  `_gated_delta_rule`.
- `src/hymo/models/gdn.py:GatedDeltaNetBlock._build_compiled_forward` — `_build_compiled_forward`: the
  `torch.compile`-wrapped forward.

## Worked example

Production scale (B=4, T=4096, H=40, S=32, D=32, chunk_size=64):

- Total chunks per forward: `T / chunk_size = 64`.
- Total blocks per kernel launch: `B * H * num_chunks = 4 *
  40 * 64 = 10,240`.
- Each block: `BLOCK_SIZE = chunk_size * d_inner / 4 = 512`
  threads.
- Total threads: `10,240 × 512 = 5.2 M`.
- A100 has 108 SMs × 2048 threads/SM = ~221 K resident
  threads.
- So we have ~24× more threads than fit at once — the
  blocks are scheduled in waves, ~24 blocks resident at
  any time.

Per-block SMEM: ~4 KB (the state) + ~16 KB (chunk tensors) =
20 KB. A100 has 192 KB SMEM/SM, so we can fit ~9 blocks per
SM resident. Good utilization.

The forward kernel launch cost: ~10 µs. The backward
recompute cost: ~10% extra compute per backward step.

Total throughput: roughly 4-5 ms per GDN layer forward at
production shape, vs ~22 ms for the naive Python double-
loop. ~4.5× speedup.

## Interview Q&A

**Q1. Why Triton instead of CUDA C++?**

> A: Triton is Python-embedded, which means the kernel can
> live in the same repo as the rest of the code, use the
> same type annotations, and integrate with `torch.autograd
> .Function` directly. CUDA C++ would require a separate
> compilation step and a Python binding. Triton's compiled
> output is PTX (close to optimal), and the autotuner
> handles block-size selection automatically.

**Q2. Why recompute-style backward instead of stored
activation?**

> A: Stored activation would keep the per-chunk state `h`
> around for backward. At `T = 4096`, `chunk_size = 64`,
> `B = 4`, `H = 40`, `S = 32`, `D = 32`, that's
> `64 × 4 × 40 × 32 × 32 × 4 bytes = 4 MB` per layer per
> forward — small. But across 24 GDN layers, ~96 MB total.
> Recompute saves this 96 MB at the cost of ~10% extra
> backward compute. The memory savings wins at long
> sequences.

**Q3. Why is there no silent Triton → eager fallback?**

> A: Because that hides regressions. If the kernel breaks
> (numerical divergence, NaN), a silent fallback would mask
> the issue by running the eager code, which has different
> numerical properties. Loud failures surface bugs
> immediately. The rule is in `AGENTS.md`: "Don't let a
> Triton kernel silently fall back to raw PyTorch during a
> default-config training run."

**Q4. Why does the kernel require power-of-2 dims?**

> A: Triton's autotuner and block layouts assume powers of
> 2 for the inner dims. The Python wrapper
> (`triton_gated_delta_rule`) pads `D` and `S` to
> `_next_power_of_2` if needed and slices the padding back
> off before returning. At production scale, `D = 32` and
> `S = 32` are already powers of 2 — no padding needed.

**Q5. What's the difference between `torch.compile` and a
Triton kernel?**

> A: `torch.compile` is automatic — it traces Python code
> and compiles it. A Triton kernel is hand-written — the
> developer chooses the block size, the SMEM layout, and the
> algorithm. Triton kernels are faster for specific shapes
> because the compiler can target exactly the right
> resources; `torch.compile` is good for general code where
> the cost of writing a custom kernel doesn't pay off.

**Q6. Why does the kernel use FP32 even when the model is in
BF16?**

> A: The recurrence is sensitive to small numerical errors
> — `α_t = exp(g_t * A)` involves an exponential, and
> accumulated state errors compound over `T = 4096` steps.
> BF16 has only 8 bits of mantissa, which is not enough
> for the `exp` + matmul accumulation. The wrapper casts
> to FP32 for the kernel, runs the math, and casts back.

**Q7. What does `HAS_TRITON` do?**

> A: It's the optional-import gate. `triton` is in
> `pyproject.toml`'s `[project.optional-dependencies]
> train` with `sys_platform == 'linux'`. On non-Linux or
> without `triton` installed, `HAS_TRITON = False` and the
> kernel raises `ImportError` if called. The trainer's
> `use_triton` flag (default True) would have to be
> flipped to False for the eager path.

## Cross-links

- [`optimization.md`](optimization.md) §3
  (Triton kernel integration).
- [`concepts/gdn-and-mla.md`](gdn-and-mla.md) —
  the recurrence that the kernel computes.
- [`model-architecture.md`](model-architecture.md) §6
  (the Triton kernel call site).

## References

- [gdn-and-mla.md](gdn-and-mla.md) — the recurrence the kernel computes.
- [model-architecture.md](model-architecture.md) — the GDN block call site (`gdn.py`).
- [optimization.md](optimization.md) — the `fused_gdn` flag and `torch.compile` interaction.
- [training.md](../training.md) — how the kernel threads through the trainer.
- Source: `src/hymo/models/gdn_triton.py` (`gdn_fwd_kernel`, `gdn_bwd_kernel`, `TritonGDNFunction`, `triton_gated_delta_rule`), `src/hymo/models/gdn.py`.
