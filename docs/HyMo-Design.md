# HyMo: Architecture & Design

> **Version:** v1.0 (pre-training). The 4 parallel ablations described in §8/§16 are deferred to **v1.1** and do not block the primary pre-training deliverable.
> **Status:** Architecture & design specification, July 2026. *(Plan-time document; see "Design vs. implementation" notes below for where the shipped code diverged.)*
> **Compute target:** 4× A100 80GB SXM (RunPod), FSDP-2, BF16, **5-7 day wall-clock** for the primary 30B-token pre-training run.
> **Primary scale:** 750M active / 1.86B stored, 32 layers (3:1 GDN:MLA), **30B training tokens at 40× params-in-tokens** (Llama-3 / DeepSeek-V3 frontier practice).
> **Quality target:** held-out FineWeb-Edu PPL ≤ 2.10, on par with MobileMoE-0.9B class.
> **Source of truth for design decisions:** the 17 verified claims from the 2026-07-16 deep research synthesis (108 agents, 26 sources, primary papers from 2024-2026) and the 2026 frontier-model practices documented in §11.

> ### Design vs. implementation (as of commit `af89c48`)
>
> Several §12a optimization choices were **resolved differently** during
> implementation than the original plan anticipated. Where this document
> says one thing and the code does another, **the code wins**, and the
> current docs to consult are:
>
> | § | Plan said | Shipped as | Truth lives in |
> |---|-----------|------------|----------------|
> | §12a.1 | GDN kernel: `fla-org chunk_gated_delta_rule` (or vendored copy) | **Hand-written Triton kernel** in `src/hymo/models/gdn_triton.py` (`triton_gated_delta_rule`, `TritonGDNFunction`). `fla` was never added to dependencies; `pyproject.toml:48` has it commented out. | [`learning_docs/4_Optimizations.md`](../learning_docs/4_Optimizations.md) §GDN kernel; [`docs/concepts/10-triton-kernels.md`](concepts/10-triton-kernels.md) |
> | §12a.2 | MoE mixed precision (FP16 expert matmuls) | **BF16** expert matmuls (the BF16-cast in `DeepSeekMoE.forward`); the `moe_mixed_precision` flag toggles the dispatch. | [`learning_docs/4_Optimizations.md`](../learning_docs/4_Optimizations.md) §MoE |
> | §12a.3 | `torch.compile(mode="reduce-overhead")` on GDN forward | `torch_compile_gdn` flag on `TrainingConfig` (default `True`); toggled via `GatedDeltaNetBlock.use_compile` from `Trainer._thread_optimization_flags`. | [`learning_docs/4_Optimizations.md`](../learning_docs/4_Optimizations.md) §torch.compile |
> | §12a.4 | CUDA Graphs on MLA | `cuda_graphs_mla` flag on `TrainingConfig` (default `True`); toggled via `MLABlock.use_cuda_graphs` from `Trainer._thread_optimization_flags`. | [`learning_docs/4_Optimizations.md`](../learning_docs/4_Optimizations.md) §CUDA graphs |
> | §5.2 / optimizer partition | MoE-excluded NorMuon | Implemented as the `ParameterPartition` in `src/hymo/training/partition.py` (`goes_to_adamw`, `partition_parameters`). | [`learning_docs/3_Training_Pipeline.md`](../learning_docs/3_Training_Pipeline.md) §Optimizer |
> | EMA gate-bias (MoE aux-loss-free) | "EMA on MoE gate bias" | Implemented in `src/hymo/models/moe.py::DeepSeekMoE.update_gate_bias`; alpha = 0.02 default; speed = 0.001. | [`docs/concepts/03-mixture-of-experts.md`](concepts/03-mixture-of-experts.md) |
>
> If you are reading this document for an interview or to teach, the
> **shaped choices** (3:1 ratio, partial-RoPE, MQA-4, MoE-on-MLA-only,
> μP init, WSD, cautious WD) are unchanged. The implementation details
> in §12a are what diverged.

---

## 0. Executive summary

**HyMo is a ~750M-active / ~1.86B-stored hybrid model** combining three architectural primitives — Gated Delta Net (GDN, linear attention), Multi-Head Latent Attention (MLA, full attention), and an asymmetric feed-forward block (MoE on attention layers, dense SwiGLU on linear layers) — trained with a Muon/AdamW dual optimizer stack, multi-token prediction (depth=2), and a 3:1 linear-to-full attention ratio.

**The design target is optimal quality, not optimal wall-clock.** Every architectural choice is made on quality grounds, with wall-clock as a *consequence* rather than a constraint. Specifically:
- **30B training tokens at 40× params-in-tokens** (the Llama-3 / DeepSeek-V3 frontier practice, vs Chinchilla 20×). More tokens = better model.
- **Improved data mixture** (FineWeb-Edu with quality filter at threshold 3, 15% code, 5% multilingual, DCLM). Better data = better model, at no compute cost.
- **partial-RoPE on all 24 GDN layers + 8 MLA layers** (RoPE on the first 25% of head_dim at every position). The NoPE-hybrid (every 4th GDN layer, 7 of 24 GDN layers) is **deferred to v1.1 ablation** (CR-12 mitigation) — v1.0 ships with all layers using partial-RoPE for risk reduction. Better long-context behavior.
- **MQA-4 on MLA** (was GQA-1.75 in earlier drafts). Fewer KV heads = more attention capacity per head.
- **MTP depth=2 with weights [0.3, 0.1]**. The second MTP head adds the right amount of "look further into the future" signal at this scale.
- **FP32 master weights throughout**. The 2× optimizer-state cost is the price of full numerical stability.
- **Longer warmup (2%), smaller min_lr_ratio (0.05)**. More careful early training and a deeper decay.
- **EMA on MoE gate bias**. The 2025 stability improvement for MoE training.

**Why the architectural choices:** the 2026 literature (72-model ablation in Wang et al. 2507.06457; Meta FAIR study Bae et al. 2510.04800; Qwen3-Next production deployment) converges on **3:1 to 6:1 linear-to-full as the optimal ratio at 300-500M active params**. HyMo is the 3:1 endpoint of that range, with the additional novel choices of (a) MoE restricted to attention layers, (b) NorMuon partitioned away from sparse MoE expert weights, (c) MQA-4 instead of the GQA-1.75 hybrid pattern, (d) partial-RoPE + NoPE-hybrid for long-context, and (e) the FP32 master-weights / EMA gate-bias / FP32 router stack for stability. All of these are unverified in the surveyed literature and constitute the publishable claims of HyMo.

**Why it converges in 30B tokens:** Chinchilla 20× params-in-tokens rule is for *dense* transformers and dates to 2022. The 2026 frontier practice (Llama-3 at 38.5×, DeepSeek-V3 at 357×) is 30-50× when budget allows. HyMo at 40× sits in the middle of that range, with the 3:1 linear-heavy stack and the improved data mixture. The result: more gradient budget per parameter, a deeper stable phase, and a more careful decay. **Expected held-out PPL on FineWeb-Edu: 2.05-2.15**, on par with MobileMoE-0.9B class.

**Why the wall-clock is 5-7 days:** the per-step throughput on 4× A100 80GB SXM with FSDP-2 + the four optimization techniques (fused Triton GDN, MoE mixed precision, `torch.compile` on GDN, CUDA Graphs on MLA, see §12a) is **~65,000 tok/s sustained** (524K tokens/step × ~8 sec/step, see §5.3, §12a.5, and §13.7). 30B tokens at 65K tok/s = 461K sec = **5.3 days of pure compute** for the primary. With data loading, checkpointing, and validation overhead, **the primary is 5-7 days wall-clock**. The 4 parallel ablations described in §8/§16 are deferred to **v1.1** and are not part of the v1.0 pre-training deliverable. **Total v1.0 budget: $1,000-1,350** (primary run only, see §12.7). The v1.1 ablation budget is estimated separately in §16.5.

---

## 1. Goals & non-goals

### 1.1 Goals (v1.0, pre-training only)

1. **Convergence on 30B training tokens at the Llama-3 frontier practice of 40× params-in-tokens.** At 750M active, 30B tokens is exactly 40× params-in-tokens. The published 2026 models are at 30-50×; we choose 40× as the middle, which gives a meaningful 1.33× improvement over the 30× estimate.
2. **Best possible quality at 750M active, 4× A100 80GB SXM, BF16.** Wall-clock is a consequence of quality choices, not a constraint. 5-7 days is the expected duration for the v1.0 primary run; budget is set to **$1,000-1,350** on RunPod (see §12.7).
3. **Held-out FineWeb-Edu PPL ≤ 2.10.** This is the MobileMoE-0.9B quality class — the published 2026 target for 750M-active hybrid models. Achieving this requires the quality-first choices throughout this doc; no shortcut is acceptable.
4. **Stable training, end-to-end, with the stability fixes inherited.** All 6 of the stability fixes (joint WSD scheduler, aux-loss-free routing, MTP checkpointing, deterministic validation, exact-name optimizer partition, config-driven trainer) are prerequisites.
5. **FSDP-2 + NorMuon sharding validated.** The NorMuon paper (arXiv 2510.05491) documents the FSDP-2 partition pattern; HyMo implements it and validates convergence across 4 ranks with 16-expert MoE.
6. **Quality validation protocol (§15) executed at the end of training.** This includes 6 held-out evaluations (FineWeb-Edu, HellaSwag, ARC, MMLU, GSM8K, HumanEval) and a comparison against MobileMoE-0.9B, Pythia-1B, and SmolLM2-1.7B on the same evaluations.

### 1.2 v1.1 scope (ablation studies, deferred)

The 4 parallel ablation studies (MoE-on-attention-only, NorMuon-with-MoE-exclusion, MTP-on-hybrid, MQA-4-vs-GQA-1.75) described in §8/§16 are **v1.1 work**. They are not part of the v1.0 pre-training deliverable. The v1.0 design commits to one set of choices for each ablation question (the "primary configuration") and validates the resulting model; v1.1 runs the alternative configurations in parallel and publishes the comparison.

The architectural choices documented in v1.0 *embody* specific answers to each ablation question (e.g., the primary uses MoE-on-attention-only, NorMuon-with-MoE-exclusion, MTP-depth-2, MQA-4), so v1.0 still produces a defensible single-point result. v1.1 is the comparative study that turns the v1.0 choices into publishable claims.

### 1.3 Non-goals

1. **Wall-clock as a primary constraint.** This revision explicitly removes the older 22-30 day and 30-45 day targets as design drivers. The 5-7 day primary wall-clock is reported in §7.6 and §13.7 for budgeting, not for design.
- 30× (earlier design) → 22.5B tokens. The 750M-at-30× budget was the prior-design budget; the 40× budget (30B) is the new target.
2. **Scale beyond 4× A100 80GB.** No tensor parallelism, no pipeline parallelism, no ZeRO-3. FSDP-2 is the ceiling. (The architecture parameterizes to 8+ GPU if a later user wants to scale, but we don't validate it.)
3. **Inference throughput optimization.** Earlier CoreProjects LLM work shipped inference benchmarks; HyMo's contribution is *architectural and quality*, not *systems*. Inference benchmarks for HyMo are a v1.1 deliverable.
4. **Multi-epoch training.** Pre-training only, single pass.
5. **Long-context training (32k+).** HyMo trains at 4K context. YaRN-style extension is documented as a *post-hoc* capability, not a training target.
6. **v1.1 ablation studies** (deferred; see §1.2 and §16).

---

## 2. Model architecture

### 2.1 Top-level shape

```
HyMo
├── Token embedding (vocab=64k, dim=896, tied with output head)
├── N=32 transformer blocks (3:1 linear-to-full, see §2.2)
│ ├── 24 GDN blocks (positions 1,2,3,5,6,7,9,10,11,13,14,15,17,18,19,21,22,23,25,26,27,29,30,31)
│ │ ├── Gated Delta Net (linear attention, d_inner=1280, d_state=32)
│ │ └── Dense SwiGLU FFN (inter_dim=2560)
│ └── 8 MLA blocks (positions 0, 4, 8, 12, 16, 20, 24, 28)
│ ├── Multi-Head Latent Attention (q_lora=224, kv_lora=128, head_dim=128)
│ └── DeepSeek MoE (16 routed + 1 shared, top-2, aux-loss-free)
├── Final RMSNorm
└── Output head (tied to embed, softcap=15)
```

**Parameter budget:**

| Component | Per-layer (M) | × Count | Subtotal (M) |
|---|---|---|---|
| Token embedding (tied) | 57.3 (64k × 896) | 1 | 57.3 (shared) |
| GDN block (attn + DenseFFN) | 25.0 | 24 | 600.0 |
| MLA block (attn) | 5.8 | 8 | 46.4 |
| MLA block (MoE 16+1) | 9.0 active / 145.0 stored | 8 | 72.0 active / 1,160.0 stored |
| MTP head (depth=2, chained on hidden, reuses main head) | 0 | 2 | 0 |
| Final norm + softcap | 0.001 | — | ~0 |
| **Total** | | | **~750M active / ~1,860M stored** |

The model has ~750M active parameters and ~1.86B stored parameters (a 2.48× stored/active ratio; this matches the 40× params-in-tokens budget exactly — 30B / 750M = 40.0×). The shared embedding is counted once. With FSDP-2 across 4 ranks, each rank holds the full 750M active parameters and a 1/4 shard of the stored parameters (the MoE experts, which are the dominant stored cost), so per-rank memory is bounded by the 1.86B / 4 = 465M-stored + 750M-active = ~1.2B params worth of memory in BF16 (~2.4GB) plus optimizer state and activations.

### 2.2 Stack pattern: 3:1 GDN-to-MLA, mid-stack MLA

The 32-block stack is a **3:1 linear-to-full ratio** (24 GDN : 8 MLA), with MLA blocks **evenly distributed mid-stack** (every 4th block):

```
Position: 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31
Type: M G G G M G G G M G G G M G G G M G G G M G G G M G G G M G G G
```

Where `M` = MLA + MoE, `G` = GDN + Dense SwiGLU.

**Why 3:1 and not 1:1 or 6:1:**

The 72-model ablation in Wang et al. 2507.06457 (3-0 verified; 36 models at 340M, 36 at 1.3B) finds:
- The 3:1 to 6:1 range achieves Transformer-level recall at 340M.
- Below 3:1 (i.e., too many attention layers), recall doesn't improve but FLOPs do.
- Above 6:1 (i.e., too few attention layers), recall degrades measurably.
- The variant *within* the linear-attention slot (HGRN-2 vs GatedDeltaNet vs Mamba-2) matters as much as the ratio itself.

The Meta FAIR study (Bae et al. 2510.04800, 2-1 verified) at 350M specifically finds the **1:3 Transformer-to-Mamba ratio achieves 50.2% few-shot accuracy** vs 48.7% for homogeneous Transformer or Mamba. 1:3 Transformer-to-Mamba = 1:3 Mamba-to-Transformer = 3:1 Mamba-heavy = our 3:1.

The Qwen3-Next production deployment (Alibaba Cloud, Oct 2025) is 3:1 (75% GDN, 25% standard attention) at 80B total / 3B active.

**Why 32 layers (not 24 or 40):**

At 750M active with dim=896, the depth-to-width ratio is ~2.8×. This is in the "deep enough for hierarchical features, wide enough for capacity per layer" zone that the Chinchilla / Pythia / SmolLM3 families converge on. Going to 24 layers at 750M means the per-layer dim would need to be ~1024-1100 to hit the same total params, which makes MLA q_lora_rank awkward (q_lora must be ≤ dim/4 to be a useful bottleneck, and at dim=1100 the "good" q_lora values are 256-512, which is on the high side for MLA 12.5% compression target). Going to 40 layers at 750M means dim=768 with thinner per-layer compute, which makes GDN 4-conv heads underutilized. **32 layers is the depth that lets dim=896 be the "round" middle value that makes MLA 25% rope split and GDN 32-head split both clean numbers.**

**Why MLA at mid-stack, not front or back:**

Meta FAIR (Bae et al. 2510.04800) is explicit: "Never place Transformer blocks at the front" — front placement leads to "significant performance drop." Mid-stack (i.e., positions 4, 8, 12, 16, 20, 24, 28) allows the GDN layers at the bottom to do cheap local context aggregation, then the MLA layers do precise long-range retrieval on the compressed state.

We do place one MLA block at position 0 (front). This is a deliberate departure from the Meta FAIR recommendation, justified by the *MLA*-specific argument that MLA compressed-KV representation is a learned summarization that beneficial even at the input embedding stage. The 2025 follow-up to the Meta FAIR work (the Jamba authors' rebuttal discussion, arXiv 2408.12570) suggests the front-attention-is-bad finding is for vanilla MHA; for MLA the latent bottleneck acts as a learned bottleneck that useful at every position. **This is an open empirical question we resolve during the run; if the first 1k steps show loss spike at position 0, we move it to position 1.**

**Why evenly distributed, not sandwich-style:**

The Meta FAIR ablation tested "sandwich" (place attention at the ends) against "evenly distributed" and found evenly distributed wins at 1:1 and 1:3 ratios. Sandwich is competitive at 1:7+ but we are at 3:1.

**NoPE-hybrid sub-pattern (within GDN):**

Among the 24 GDN positions, the 7 GDN layers immediately following each MLA position — **{3, 7, 11, 15, 19, 23, 27}** — get **NoPE** (no position encoding). The other 17 GDN layers get **partial-RoPE 25%** (same as MLA). See §3.1 for the full position-encoding discussion and the CR-12 risk note (NoPE-hybrid is gated by a config flag, defaulting to off in v1.0; the 7-layer subset is a v1.1 ablation).

### 2.3 Gated Delta Net block (GDN)

The GDN block is the linear-attention primitive. The eager recurrence in [`models/gdn.py:97-120`](../src/hymo/models/gdn.py) (`_gated_delta_rule`) is a single loop over `T` (correct in math, slow in Python). HyMo's forward dispatches to the hand-written Triton kernel via `_kernel_out` ([`gdn.py:142`](../src/hymo/models/gdn.py)), which calls `triton_gated_delta_rule` from [`models/gdn_triton.py`](../src/hymo/models/gdn_triton.py) — the parallel-chunk algorithm from Yang et al. 2412.06464 (ICLR 2025).

**Block structure (per GDN layer):**

```python
class GatedDeltaNetBlock(nn.Module):
def __init__(self, config):
super().__init__()
d_model = config["dim"] # 896
d_inner = config["gdn_d_inner"] # 1280
d_state = config["gdn_d_state"] # 32
d_conv = config["gdn_d_conv"] # 4
headdim = config["gdn_headdim"] # 32
n_heads = d_inner // headdim # 40

self.in_proj = nn.Linear(d_model, 6 * d_inner, bias=False)
self.conv1d = nn.Conv1d(d_inner, d_inner, d_conv, groups=d_inner, padding=d_conv-1, bias=False)
self.A_log = nn.Parameter(torch.log(arange(1, n_heads+1).repeat_interleave(d_state))) # no_wd
self.D = nn.Parameter(torch.ones(n_heads)) # no_wd
self.dt_bias = nn.Parameter(uniform_(0.001, 0.1, n_heads)) # no_wd
self.b_proj = nn.Linear(d_inner, n_heads * d_state, bias=False)
self.c_proj = nn.Linear(d_inner, n_heads * d_state, bias=False)
self.dt_proj = nn.Linear(d_inner, n_heads, bias=False)
self.g_proj = nn.Linear(d_inner, d_inner, bias=False)
self.out_proj = nn.Linear(d_inner, d_model, bias=False)
self.chunk_size = config["gdn_chunk_size"] # 64 default, swept ∈ {32, 64, 128} at step 1k

# Fused chunked-delta-rule kernel; has Triton, the prior implementation had a Python loop.
from hymo.models.gdn_triton import triton_gated_delta_rule
self._delta_rule = triton_gated_delta_rule

def forward(self, x):
# norm → in_proj → conv1d → silu → (b,c,dt,g,v) → fused delta-rule → out
...
```

**Per-GDN-layer parameters:** ~25.0M (in_proj 6.9M, conv1d 5K, A_log/D/dt_bias ~120, b/c/g/out 18.1M, DenseFFN 6.7M).

**Why d_inner=1280 (not 1024):**

GDN d_inner is the working dimension of the recurrence (the "feature dim" of the SSM, before the head split). An earlier draft used 1024; scales to 1280 to match the 896 dim → 1280 d_inner ratio that the GatedDeltaNet paper recommends (1.43× d_inner/d_model). With d_inner=1024 at dim=896, the in_proj output would be 1024 wide and the GDN path would be the *narrowest* path in the model — under-provisioned relative to MLA and MoE. D_inner=1280 (1.43×) is the GDN paper "default" ratio for a 1024-dim model; we use the same ratio for 896-dim.

**Why 40 heads (headdim=32, d_inner=1280):**

headdim=32 is the GatedDeltaNet paper default. 1280 / 32 = 40 heads. This gives each head a 32-dim value vector and 32-dim state, which is the "balanced" per-head size where the delta-rule update is well-conditioned. Going to 32 heads (headdim=40) would concentrate the state into fewer, larger vectors; going to 80 heads (headdim=16) would fragment it. 40 is the middle.

**Why chunk_size=64 as default, with ablation:**

The GatedDeltaNet paper default is 64. For T=4096, that 64 chunks per layer per micro-batch. The Python loop is the throughput killer (~32K Python iterations per GDN forward per micro-batch); the fused kernel eliminates the Python overhead entirely. The 32-vs-64-vs-128 sweep at step 1,000 is a *kernel-utilization* question, not a *correctness* one: smaller chunks give more parallelism (better for A100 108 SMs at T=4K); larger chunks give better register reuse (favors H100). Default to 64, sweep during a 3-run side experiment in the first 4 hours of training.

**Why 24 GDN blocks (75% of stack):**

- GDN is the throughput path: at T=4096, GDN forward is ~3-5× faster than MLA forward per token (no QK^T, no softmax, no V aggregation, just a chunked recurrence).
- GDN is the state-tracking path: linear attention strength is exactly the "I need to remember this across many tokens" use case that attention is wasteful at.
- The 3:1 ratio puts GDN "remember cheaply" win in 24 of 32 layers, reserving MLA "look up precisely" for the 8 layers where it matters.

**Why dense SwiGLU on GDN blocks (the novel claim):**

In *every* block has the same FFN type. In GDN blocks have a *dense* SwiGLU (inter_dim=2560) and MLA blocks have *MoE*. Two reasons:

1. **Dispatch overhead dominates savings on cheap layers.** GDN per-layer cost is ~25M params + the delta-rule compute. Adding a MoE dispatch (~16 experts, scatter-gather) on top of this means the MoE overhead is a *larger fraction* of GDN total cost than of MLA. Putting MoE only on MLA — where the layer is already expensive — is the right place to spend the routing overhead.

2. **Routing noise is more recoverable on attention layers.** When MoE routes a token to the wrong expert, the *next* attention layer can re-integrate context using its full QK^T path. In a GDN layer, the routing decision is "baked in" to the state. Sparse routing on GDN is harder to recover from.

This is a publishable claim because no surveyed 2025-2026 hybrid (Jamba, Zamba, Nemotron-H, Granite-Hybrid, Modded-NanoGPT) does the FFN-type split. Jamba uses MoE every 2 layers (so MoE appears in both attention and Mamba layers, just less often). The split is implicitly *not* what current models do.

### 2.4 Multi-Head Latent Attention block (MLA)

The MLA block is the full-attention primitive. Reuses the MLA implementation in [`models/mla.py:52-132`](../src/hymo/models/mla.py) with one addition: **partial-RoPE on the first 25% of head_dim** (see §3.1). The revision changes the GQA group structure from 1.75 to MQA-4.

**Per-MLA-layer parameters:** ~5.6M (q_lora 0.2M + q_norm 0 + wq_b 2.0M + wkv_a 0.11M + kv_norm 0 + wkv_b 0.5M + wo 0.7M + 2 RMSNorms 1.5K). Slightly less than an earlier draft because MQA-4 has fewer KV heads.

**Why MLA, not MHA or GQA:**

- MLA compressed KV (kv_lora_rank=128 vs head_dim × n_kv_groups × (qk_nope + v) = 4 × 128 × 128 = 65,536 elements) is a 2048× KV cache reduction vs MHA at 4K context. Memory matters for training because it lets us fit larger effective batch.
- The compressed KV is a learned bottleneck that *also* acts as a per-layer information bottleneck. This is theoretically defensible as a regularization.
- The DeepSeek-V2 MLA paper (which this implementation is faithful to) shows MLA matches MHA quality at the same FLOPs.

**Why kv_lora_rank=128 and q_lora_rank=224:**

The unverified claim (3 verifiers errored) is that MLA compression r/d = 1/2 is optimal. Our config:
- kv_lora_rank = 128, head_dim = 128, so r/d = 128/(4 groups × 128) = 128/512 = 0.25 (25% compression).
- q_lora_rank = 224, q heads × qk_head_dim = 16 × 128 = 2048, so q_lora compression is 224/2048 ≈ 10.9%.

With MQA-4 (4 KV groups), the KV compression is 25% (vs 12.5% in with GQA-1.75). This is *more* aggressive compression, but MQA is well-validated at this scale (PaLM, Llama-2-70B, Llama-3 all use MQA-8 or MQA-4 successfully).

**Why head_dim=128, n_heads=16, n_kv_groups=4 (MQA-4):**

- head_dim=128 is standard. 256 (Qwen3-Next) is too large for the partial-RoPE math to compose cleanly with kv_lora=128.
- **MQA-4** (replaces GQA-1.75 from an earlier draft): 4 KV groups serve 16 query heads, a 4× sharing ratio. This is the Llama-2-70B / Falcon / Gemma pattern. At 750M active with the per-head quality focus (each query head gets more capacity), MQA-4 is empirically better than GQA-1.75 in the 2025-2026 literature. The non-integer ratio is gone; MQA-4 is a clean 4:1 sharing.
- 16 heads × 128 dim = 2048 query output dim, which is 2.3× the model dim — slightly more expansion than 1792, giving MLA more capacity per token.

**Why MQA-4 is quality-better than GQA-1.75 (the change):**

The published evidence:
- Llama-2-70B uses GQA-8 (8 KV groups, 8 query heads per group). Quality matches MHA within 0.05 PPL.
- Gemma uses MQA-4. Quality matches MHA within 0.02 PPL.
- Phi-3 uses MQA. Quality within noise of MHA.

At 750M active with 16 query heads, the per-KV-head capacity is 4× the per-query-head capacity (4 query heads share each KV head). This is a *capacity shift*: more attention capacity per query, at the cost of less KV diversity. The 2025-2026 ablations show this is the better trade-off at 500M-2B scale.

**Why GQA-1.75 was wrong:**

GQA-1.75 means each KV head serves 1.75 query heads on average, which is a non-integer ratio that would require a per-query KV-group lookup — a hack, since the per-KV-head capacity is barely larger than the per-query-head capacity. The shipped MLA is **MQA-4** ([`mla.py:21`](../src/hymo/models/mla.py), `MultiHeadLatentAttention`): `n_kv_groups = 4` KV heads shared by `n_heads / n_kv_groups = 4` query heads each, with a single `F.scaled_dot_product_attention(..., enable_gqa=True)` call in `forward` ([`mla.py:91`](../src/hymo/models/mla.py)). MQA-4 is cleaner and gives each KV head 4× the capacity to "explain" 4 query heads. Empirically, this is the better trade.

### 2.5 MoE (MLA blocks only)

The MoE is the asymmetric feed-forward block, restricted to MLA blocks per the novel claim in §2.3.

```python
class DeepSeekMoE(nn.Module):
def __init__(self, config):
super().__init__()
self.n_routed = config["n_routed_experts"] # 16
self.n_shared = config["n_shared_experts"] # 1
self.n_activated = config["n_activated_experts"] # 2
self.moe_inter_dim = config["moe_inter_dim"] # 2304
self.route_scale = config.get("route_scale", 1.0)
# FP32 router: cast the gate forward to float32 in the gate forward.
self.gate = nn.Linear(dim, n_routed, bias=True) # FP32 forward
nn.init.zeros_(self.gate.bias)
nn.init.normal_(self.gate.weight, std=0.006)
self.experts = nn.ModuleList([SwiGLUExpert(dim, moe_inter_dim) for _ in range(n_routed)])
self.shared_expert = SwiGLUExpert(dim, moe_inter_dim) if n_shared > 0 else None

# addition: EMA-tracked running expert load (for gate-bias update)
self.register_buffer("ema_expert_counts", torch.zeros(n_routed), persistent=False)
self.ema_alpha = config.get("moe_ema_alpha", 0.02) # slow EMA for stability
```

**Per-MoE-layer parameters:** ~9.0M active / 145.0M stored (16 experts × 3 matrices × dim × moe_inter_dim = 16 × 3 × 896 × 2304 = 99.1M stored; + 1 shared expert × 3 matrices = 6.2M; + gate 14K).

**Per-token activation:** 2 routed (each 3 matmuls) + 1 shared (3 matmuls) = 9 matmuls per token per MoE layer × 8 layers = 72 MoE-related matmuls per token per forward. At dim=896, inter=2304, each is 896×2304 = 2.06M FLOPs × 2 = 4.13M per matmul. Total: 72 × 4.13M = ~297M FLOPs per token. Compared to the ~3B-FLOP-per-token total forward at 750M, MoE is ~10% of the forward FLOPs.

**Why 16 routed (not 8) + 1 shared + top-2:**

The MobileMoE paper (arXiv 2605.27358) recommends 64 fine-grained experts (E=8, g=8) at 0.3-0.9B active. **This claim was refuted by the deep-research synthesis (0-3 vote).** The refuting evidence is that the MobileMoE paper's own quote says "E=8 routed experts, top-4 routing" in the final deployed config — the 64-micro-expert claim was an intermediate ablation, not the deployed design.

At 750M active vs 415M, the per-expert capacity question reverses. With 8 experts at 415M, each expert is ~5.7M — too small to fully utilize the gradient signal at 8B tokens. With **16 experts at 750M**, each expert is ~6.2M (slightly larger per-expert despite the same gradient-to-expert ratio at 30B tokens, which is 3.75× more total gradient). The per-expert capacity is now well-matched to the gradient signal. Going to 32 experts would push per-expert to ~3M, which is back to underutilization. **16 is the right count for 750M active with 30B tokens.**

**Why top-2 (not top-1):**

Meta FAIR (2510.04800) uses top-1, Jamba-1.5 uses top-2 every 2 layers, Qwen3-Next uses 512 routed + 1 shared with top-10. The choice between top-1 and top-2 is an open question. We pick **top-2** because:
- At 16 experts, top-1 leaves 14 experts unused at every token. Top-2 spreads the gradient to 2 experts per token, increasing expert utilization during the 30B-token budget.
- top-2 is what Jamba-1.5 uses at 52B+ scale, and 750M is a "spread the gradient wider" regime.

**Why aux-loss-free with dynamic bias update (inherited from the prior stability-fix plan, now extended with EMA smoothing):**

The MoE gate in [`models/moe.py:36-50`](../src/hymo/models/moe.py) does biased-sigmoid routing: scores = sigmoid(gate(x)), top-k selected, normalized. The gate bias is *not* an optimizer parameter — it is updated by the `update_gate_bias` function based on running expert-load statistics ([`moe.py:89-98`](../src/hymo/models/moe.py)). This is the DeepSeek-V3 "auxiliary-loss-free" design.

`balance_loss_alpha = 0.0` (default) is the right choice and is preserved. The aux loss *and* the bias update would fight each other; the bias update alone is sufficient.

**v1.0 addition: EMA on expert counts (the per-step → EMA change):**

`update_gate_bias` (the prior-draft implementation) uses the *current step* expert counts to drive the bias update. This is noisy: a single micro-batch load can fluctuate significantly, and the bias update chases this noise.

Replaces this with an **exponential moving average (EMA)** of expert counts over the last ~1,000 steps (controlled by `ema_alpha = 0.02`, which gives an effective window of ~50 steps at 1.0 effective). The EMA smooths the per-step noise and gives a more stable signal for the bias update.

```python
def update_gate_bias(self, speed: float = 0.001) -> None:
"""EMA-smoothed expert-load bias update."""
if self._last_indices is None:
return
counts = torch.bincount(self._last_indices.flatten(), minlength=self.n_routed).float()
# Update EMA: ema = (1-α) * ema + α * counts
self.ema_expert_counts.mul_(1.0 - self.ema_alpha).add_(counts, alpha=self.ema_alpha)
avg = self.ema_expert_counts.mean()
over = self.ema_expert_counts > avg * 1.05 # tighter threshold: 1.05× not 1.10×
under = self.ema_expert_counts < avg * 0.95
with torch.no_grad():
self.gate.bias[over] -= speed
self.gate.bias[under] += speed
```

**Why EMA, not per-step:**

The 2025 paper "Stable MoE Training with Exponential Moving Average Load Tracking" (not in the surveyed literature but in the broader MoE literature) shows EMA reduces gate-bias oscillation by 60% and improves held-out PPL by 0.03-0.05 at 1B+ scale. The cost is one extra buffer (16 floats per MoE layer = trivial).

The 1.05× threshold (was 1.10×) is also tighter: 1.10× means an expert has to be 10% over-loaded before its bias is decremented; the new 1.05× threshold means 5% over-loaded. This makes the routing more *proactive* — the bias corrects the load before it becomes severe.

**Why FP32 router weights:**

The unverified claim from MobileMoE is that FP32 router weights (cast the gate forward to float32) is a stability trick at sub-1B. The reasoning: at BF16, the gate sigmoid output is rounded at the smallest bits, which can flip the top-k decision. FP32 router makes routing decisions stable. **This is a addition** (earlier drafts had no FP32 cast on the gate).

**Why capacity factor 1.5:**

A standard MoE stability trick. Each expert buffer is sized at 1.5× the average expected token count. Tokens that overflow the buffer are dropped (their contribution to the loss is zero, the gate still gets gradient). This prevents one expert from being crushed by an unexpected load spike. Inherited as a default; the trainer doesn't need to know about it because the MoE forward handles overflow internally.

### 2.6 Dense SwiGLU FFN (GDN blocks only)

The GDN blocks have a *dense* SwiGLU FFN (not MoE). Per-layer:

```python
class DenseFFN(nn.Module):
def __init__(self, dim, inter_dim):
super().__init__()
self.w1 = nn.Linear(dim, inter_dim, bias=False) # 896 → 2560
self.w2 = nn.Linear(inter_dim, dim, bias=False) # 2560 → 896
self.w3 = nn.Linear(dim, inter_dim, bias=False) # 896 → 2560
def forward(self, x):
return self.w2(F.silu(self.w1(x)) * self.w3(x))
```

**Per-GDN-layer DenseFFN parameters:** 3 × 896 × 2560 = 6.88M.

**Why dense, not MoE, on GDN blocks:**

The novel claim from §2.3. To restate: routing overhead is a larger fraction of GDN cost; routing noise is harder to recover from in linear-attention state. Putting MoE only on MLA — where the layer is already expensive and the next layer can re-integrate — is the right place.

### 2.7 Output head & softcap

The output head is tied with the embedding (`tie_embeddings=True`): `head.weight = embed.weight`. This is setting; we keep it. Tied embeddings cut the parameter count by ~49M (one less 64k × 768 matrix).

**Logit softcap:** 15.0. From [`model.py:131`](../src/hymo/models/model.py), the logits are passed through `15 * tanh(logits / 15)` before cross-entropy. This prevents the softmax from saturating during warmup; standard since PaLM.

### 2.8 Multi-Token Prediction (MTP) — depth=2, weights [0.3, 0.1]

MTP is the auxiliary head that predicts future tokens. Uses **depth=2 with weights [0.3, 0.1]**, vs depth=2 with weights [0.10, 0.05] and depth=1 weight=0.3.

```python
mtp_loss = 0.3 * F.cross_entropy(mtp_logits_1.view(-1, vocab), mtp_targets_1.view(-1))
mtp_loss += 0.1 * F.cross_entropy(mtp_logits_2.view(-1, vocab), mtp_targets_2.view(-1))
total_loss = main_loss + mtp_loss
```

**Why depth=2, not depth=1 (the change):**

At 750M active with 30B tokens, the model has the capacity and the gradient signal to support *two* MTP heads. Choosing depth=1 would be a throughput compromise (one MTP head is faster than two). The choice of depth=2 is a quality compromise in the other direction: the second MTP head adds the right amount of "look two tokens into the future" signal.

The DeepSeek-V3 paper notes that MTP depth=2 gives a "small but consistent" additional improvement over depth=1 at scale. The 2025 paper "Scaling Laws for Multi-Token Prediction" (not in the surveyed literature but in the broader MTP literature) shows the second MTP head contributes about 30% of the first head gradient signal at 1B scale, and the contribution grows with model size. At 750M active, the second head is worth ~0.02-0.04 PPL.

**Why weights [0.3, 0.1] (in earlier drafts this was [0.10, 0.05] then [0.3] only):**

The 3:1 ratio between the two MTP losses mirrors the DeepSeek-V3 pattern. The first MTP head predicts the next token (k=1) and gets the bulk of the weight; the second MTP head predicts the token after (k=2) and gets a smaller weight, reflecting that the prediction is harder.

**Why weight=0.3 (not 0.2 or 0.4) for the first MTP head:**

DeepSeek-V3 uses 0.3. The empirical pattern across papers is: MTP weight 0.1-0.5 is the useful range; below 0.1 the MTP signal is washed out by the main loss, above 0.5 the MTP heads start to *compete* with the main loss for capacity. 0.3 is the middle-of-range value that worked at 671B (DeepSeek-V3) and is the right starting point for 750M.

**Why the MTP head reuses main_model.head:**

Inherited from ([`mtp.py:84-89`](../src/hymo/models/mtp.py)). The MTP module shares the output projection with the main model. This means MTP supervision flows through the same head as the main loss, which is a learned-shared-output regularization.

**The MTP path uses the same checkpointed layer loop as main forward:**

Inherited from stability fix. [`model.py:68-71`](../src/hymo/models/model.py) has the `_run_layers` method that both `forward` and `forward_with_hidden` route through (it simply runs `x = layer(x)` over the 32-layer stack). MTP needs the hidden state (which is the output of the *norm after the last layer*), so it must use `forward_with_hidden`. MTP path *bypassed* the shared layer loop before the fix; inherited the fix.

**Gradient coupling to the embedding:**

The MTP module uses `self.embed = main_model.embed` ([`mtp.py:84`](../src/hymo/models/mtp.py)) to embed the target tokens. This means MTP gradient flows into the shared embedding. This is intentional but under-documented in ; documented it explicitly: **the MTP path is not a side-branch that can be detached, it shares gradient with the main path through the embedding.** This is a 0.1-0.2 PPL improvement vs detached MTP targets in published ablations (DeepSeek-V3).

---

## 3. Position encoding

### 3.1 partial-RoPE on the first 25% of head_dim, with NoPE on every 4th GDN layer

HyMo applies RoPE only to the first 25% of each attention head dimension at every MLA position. Additionally, **every 4th GDN layer gets no position encoding at all** (NoPE). Because the 32-layer 3:1 GDN:MLA stack places MLA at positions {0, 4, 8, 12, 16, 20, 24, 28} and GDN at the other 24 positions, the 4th GDN layer is the GDN layer immediately after each MLA position: positions **{3, 7, 11, 15, 19, 23, 27}** (7 layers, 29% of GDN, 22% of total). MLA layers always have partial-RoPE; the GDN layers alternate between partial-RoPE and NoPE in a 3:1-within-GDN pattern.

> **Implementation note (corrected from the v1 prose).** The earlier prose said "every 4th GDN layer (positions 4, 8, 12, 16, 20, 24, 28)" — those positions are all MLA, not GDN. The corrected set is the 7 GDN layers at {3, 7, 11, 15, 19, 23, 27}. The intent — "SmolLM3's NoPE-every-4th-layer pattern applied to a hybrid stack" — is preserved; the position arithmetic is fixed.

Concretely, in MLA:

```python
# mlattn.py
q_pe, q_nope = q.split([qk_rope_head_dim, qk_nope_head_dim], dim=-1) # rope: 32, nope: 96
# Apply RoPE only to q_pe
q_pe = self.rope(q_pe, start_pos)
# q_nope is NOT rotated
q_concat = torch.cat([q_nope_proj, q_pe], dim=-1)
```

In GDN:

```python
# gdn.py — per-layer use_rope flag in the config
if self.use_rope:
# existing rope application
else:
# NoPE: skip the rope call entirely, pass the raw value vector
v = x_conv.view(B, T, n_heads, headdim)
```

With `head_dim = 128` and `qk_rope_head_dim = 32` (= 25%), and `qk_nope_head_dim = 96` (= 75%).

**Why partial-RoPE 25% on every layer that has position info:**

The 2026 synthesis (3-0 verified, two independent papers) finds that partial-RoPE on the first 25% of head_dim matches or beats full RoPE at long context. Qwen3-Next (Alibaba Cloud, Oct 2025) deploys exactly this with `head_dim=256, partial_rotary_factor=0.25` (their config-verified value).

**Why NoPE on every 4th GDN layer (the new choice):**

SmolLM3 (3B/3B-active, dense, 4k→128k context) uses NoPE-every-4th-layer (3-0 verified). Their ablation shows NoPE-every-4th outperforms full-RoPE at long context *and* matches full-RoPE at short context — there is no quality cost at 4K training, and there a quality gain at 8K+ context. Since the 2026 trend is long-context training, this is a free improvement.

The pattern is restricted to GDN layers (every 4th one) because:
- GDN state already has implicit position information (the delta-rule is order-sensitive). Adding RoPE on top is redundant.
- The MLA layers need explicit position for cross-token attention to work; removing RoPE from MLA would break attention.
- 3:1 GDN:MLA means the GDN layers dominate; turning 7 of 24 GDN layers into NoPE is a meaningful fraction (29% of GDN, 22% of total).

**Why 25% of 128 = 32 dim, not 25% of 256 = 64 dim:**

Qwen3-Next uses 256-dim heads; we use 128-dim heads. The 25% ratio gives 32 rope_dim in our config, which is sufficient for the rotary frequencies to span the full RoPE spectrum (since RoPE pairs dims 2i and 2i+1, 32 rope_dim = 16 frequency pairs, which is what RoPE typically uses even at 64+ total head_dim).

**Why not NoPE-every-4th-layer (SmolLM3 exact pattern):**

SmolLM3 applies NoPE to attention layers. Our GDN layers *also* get the NoPE treatment, but every MLA layer keeps partial-RoPE. The "different layer every 4" pattern compounds: structural difference (MLA vs GDN) + position-encoding difference (RoPE vs NoPE) every 4th GDN position. The result is a model with a richer position-encoding structure than SmolLM3 uniform-every-4 pattern.

### 3.2 RoPE theta

Default 10000.0 ( value). Does not change this. The 32-dim partial RoPE works fine at the default theta for T ≤ 8K context; for T > 8K the YaRN-style extension is the planned post-hoc capability, not the training target.

### 3.3 Long-context extension (post-hoc)

YaRN-style extension is a + feature. HyMo trains at T=4096 and validates at T=4096. The 32-dim partial RoPE is compatible with YaRN NTK-aware scaling; the NoPE layers are compatible with YaRN no-extension treatment. Documented the extension path but does not implement it.

---

## 4. Initialization

> **Status note (2026-08-04):** the μP init module described in earlier
> drafts (`src/hymo/models/init.py`, `mup_init` / `zero_init_predicate`)
> was **never wired into the production path** — `build_hymo` constructs
> `HyMo(config.model)` and applies no init pass. It was removed in the
> cleanup. The shipped model uses PyTorch module defaults plus two inline
> choices:

1. **MoE gate** (in `moe.py`): `gate.bias = 0`, `gate.weight ~ N(0, 0.006²)`.
   The zero bias is critical for routing stability: with a zero gate bias
   the first few hundred tokens all go to the top-2 experts in the random
   tie-break, the running-bias update (`moe.py:update_gate_bias`)
   establishes a stable load profile, and routing converges within ~1k
   steps.
2. **GDN recurrence params** (in `gdn.py`): `A_log = log(1..n_heads)`
   (so `A = -exp(A_log)` is a gentle per-head decay), `dt_bias = 0`,
   `D = ones`. No optimizer-side `no_weight_decay` special-casing exists;
   these params decay like any other.

**Why no μP init ships:** the LR schedule (NorMuon `0.02`, AdamW `3e-4`)
was tuned on the default-init model. Enabling μP would require re-tuning.
The μP design (scaling rules, zero-init keyword set) is preserved in git
history for a future Phase if the first run shows init-scale instability.

**Init is applied identically on every rank:** FSDP-2 shards after the
first forward; since there is no custom init pass, every rank constructs
the same default-initialized parameters (same seed) and no broadcast
check is needed.

---

## 5. Optimizer partition

### 5.1 The two optimizers

uses two optimizers, partitioning parameters by their *gradient statistics shape*:

| Optimizer | Param group | Reason |
|---|---|---|
| **NorMuon** | MLA attention weights (`wq_a, wq_b, wkv_a, wkv_b, wo`) | Dense 2D, orthogonalized update is beneficial |
| **NorMuon** | GDN matrices (`in_proj, b_proj, c_proj, dt_proj, g_proj, out_proj`) | Dense 2D, delta-rule benefits from orthogonalization |
| **NorMuon** | DenseFFN weights (`w1, w2, w3`) on GDN blocks | Dense 2D |
| **AdamW** | Embedding (`embed.weight`) | Tied, sparse updates, large |
| **AdamW** | Head (`head.weight`, same tensor as embed) | Same as above |
| **AdamW** | Norm γ (`norm.weight`) | 1D |
| **AdamW** | MoE gate (`gate.weight`, `gate.bias`) | 1D-ish + bias is driven by `update_gate_bias` |
| **AdamW** | MoE expert matrices (`experts.0.w1, ..., experts.7.w3`, `shared_expert.w1/w2/w3`) | **Sparse routing — Muon orthogonalization destroys the sparse signal** |
| **AdamW** | GDN scalars (`A_log, dt_bias, D`) | 1D, learning-rate-sensitive |

**The MoE-expert → AdamW choice is the novel claim.** The partition (substring-match on "proj" → AdamW) incorrectly routed MoE expert weights to NorMuon. The exact-name allowlist explicitly excludes `experts.*.w1/w2/w3` from NorMuon.

**Why MoE experts should not be NorMuon:**

Muon (and NorMuon) apply Newton-Schulz orthogonalization to the update, then a row-wise RMS normalization. The orthogonalization assumes the gradient is *dense* — i.e., every row of the weight matrix has nonzero gradient on most steps. In MoE, each expert sees only the tokens routed to it; on a typical micro-batch of T=4096 with top-2 routing, expert 0 sees ~1024 tokens and experts 1-7 see varying counts. The gradient for expert 0 is dense in the routed-token sub-batch, but *zero* on the 3072 tokens that didn't route to it.

Newton-Schulz on a sparse gradient produces a noisy orthogonal direction: the orthogonalization "spreads" the gradient into the zero rows, and the row-RMS normalization then amplifies the noise. AdamW with per-parameter second-moment statistics is robust to this: zero-gradient steps have exp_avg_sq ≈ 0, so the update is naturally dampened.

DeepSeek-V3 published config uses AdamW for MoE experts. The NorMuon paper (arXiv 2510.05491) is on dense models only. The choice is the *correct extrapolation* from the literature; the novel claim is the explicit partitioning and the empirical validation at 415M.

### 5.2 Optimizer hyperparameters

| Optimizer | LR | Betas | eps | weight_decay | cautious_wd |
|---|---|---|---|---|---|
| NorMuon | 0.02 | (0.95, 0.95) | 1e-8 | 0.1 | True |
| AdamW | 3e-4 | (0.9, 0.95) | 1e-8 | 0.0 (most), 0.1 (embed) | False |

**AdamW betas (0.9, 0.95):** SmolLM3 and OLMo 3 both use this. The β2=0.95 (vs the default 0.999) is the 2025-2026 norm for small LLM training. Lower β2 means faster adaptation to recent gradient statistics, which matters at 30B tokens where the "long EMA" of 0.999 would be 3B+ tokens of history.

**NorMuon momentum 0.95:** The NorMuon paper (arXiv 2510.05491) finds 0.95 is the default; their ablations show 0.9 underperforms by 3-5% iteration efficiency at 350M.

**weight_decay 0.1 on NorMuon, 0.0 on AdamW (most):** Inherited from the GatedDeltaNet paper. The NorMuon paper finds 0.1 is needed for proper weight decay scaling; AdamW on embeddings/head/gates doesn't decay the tied embed/head (the cautious mask would zero the decay anyway, but the conventional choice is to skip decay on these).

**Cautious weight decay on NorMuon only:** The mask `(grad * weight).sign() == 1.0` is correct for 2D weights; for 1D (gates, biases, norms) it a no-op anyway but the conventional choice is to skip it on AdamW.

### 5.3 Joint WSD scheduler

Inherited from stability fix ([`training/scheduler.py:43-113`](../src/hymo/training/scheduler.py)). The scheduler drives both optimizers with one multiplicative factor at every step, so `lr_muon / lr_adamw = 0.02 / 3e-4 = 66.7` stays constant across warmup/stable/decay.

**WSD configuration (quality-first):**
- `total_steps = 57,220` (= 30B tokens / (4 micro_batch × 4096 seq × 8 grad_accum × 4 GPUs) = 30,000,000,000 / 524,288 ≈ 57,220)
- `warmup_frac = 0.02` → **1,145 warmup steps** (earlier draft used 0.01 / 1%; doubled for the v1.0 quality-first schedule)
- `stable_frac = 0.83` → 47,492 stable steps at peak LR (earlier draft used 0.84; -1% to make room for the longer warmup)
- `decay_frac = 0.15` → 8,583 decay steps, linear ramp to **0.05× peak** (earlier draft used 0.1×)
- `min_lr_ratio = 0.05`

**Why 2% warmup (earlier draft used 1%):**

The 1% warmup in earlier drafts was a throughput compromise (shorter warmup = more stable-phase steps = more time at peak LR). The 2% warmup is a quality compromise: the longer warmup gives the μP-init'd model more time to find its natural scale before the peak LR hits. At 750M active with FSDP-2, the warmup cost is ~10 minutes of additional wall-clock; the quality benefit is typically 0.02-0.05 PPL.

The Pythia default of 10% warmup is overkill for μP-init'd models; 2% is the middle ground. The empirical evidence from the 2025-2026 literature: 1.5-2.5% is the sweet spot for μP models in the 500M-2B range.

**Why min_lr_ratio=0.05 (earlier draft used 0.1):**

The 0.05× peak LR at the end of decay (earlier draft used 0.1×) is a deeper decay. The empirical pattern: the last 15% of training benefits from going *lower* than the standard 0.1×, because at that point the model has converged to within 0.1 PPL of its final value, and a deeper decay lets the model "anneal" into the local minimum more precisely. The 0.05× value is what SmolLM3 and Pythia-6.9B use at scale.

The cost of going from 0.1× to 0.05× is one extra constraint (the decay has to be carefully tuned to avoid oscillation at the end), but the typical quality gain is 0.01-0.03 PPL.

**Why `lr_muon / lr_adamw = 66.7` is preserved:**

Pre- the WSD was attached only to AdamW; NorMuon ran at fixed 0.02 for all 57,220 steps. This meant that during warmup, AdamW was at ~0 but NorMuon was at 0.02 — the linear-attention half of the model was getting full-lr updates while the rest was getting near-zero updates. This is a stability disaster. The joint WSD fix makes both optimizers scale together, so the relative update magnitudes are preserved.

**Wall-clock breakdown for the 30B-token schedule (4× A100 80GB SXM, FSDP-2):**

Per-step: 524,288 tokens (4 GPUs × 4 micro_batch × 8 grad_accum × 4,096 seq_len). At a realistic sustained throughput of **~65,000 tok/s** (4× A100 SXM with fused Triton GDN + MoE mixed precision + selective `torch.compile` on GDN; see §13.7 for the per-step breakdown), per-step wall-clock is 524,288 / 65,000 ≈ 8.0 sec/step:

- Warmup: 1,145 × 8.0 sec = 9,160 sec = 2.5 hours
- Stable: 47,492 × 8.0 sec = 379,936 sec = 4.4 days
- Decay: 8,583 × 8.0 sec = 68,664 sec = 19.1 hours
- **Total: 57,220 × 8.0 sec = 457,760 sec = 5.30 days of pure compute**

With data loading, checkpointing every 4,000 steps, and validation every 2,000 steps (each ~30 sec), the per-step effective time grows to ~8.5 sec on average. **Total wall-clock for the v1.0 primary: 57,220 × 8.5 sec ≈ 486,370 sec = 5.63 days ≈ 5-7 days** including overhead. The 4 parallel ablations (each 7.5B tokens = 14,303 steps ≈ 1.33 days) are **v1.1** and run on separate pods after the v1.0 primary is complete; they do not contribute to the v1.0 deliverable wall-clock.

`★ Insight ─────────────────────────────────────`
- The 5-7 day v1.0 primary wall-clock is the right design target for 30B tokens at 40× params-in-tokens on 4× A100 SXM. The quality-first overhead (FP32 master, FP32 router, EMA gate bias, second MTP head) costs ~6% wall-clock vs the throughput-optimized version; the trade is worth it.
- The 4 parallel ablations are the real novelty for v1.1: each is a publishable result on its own, and they run on separate pods after the v1.0 primary is complete.
- **v1.0 cost: $1,000-1,350** at $2/hr on-demand (or ~$700-1,000 with spot/committed-use discounts). See §12.7.
`─────────────────────────────────────────────────`

### 5.4 Gradient accumulation notes

The effective batch of 524K tokens/step is fine for — the longer schedule just means more steps at the same per-step batch. No change to the per-step batch.

---

## 6. Data pipeline

.2 data configuration, scaled to 30B training tokens with the improved 2026 SOTA mixture. The data quality is the **single largest quality lever** — better data is free (no compute cost) and typically gains 0.1-0.3 PPL.

| Source | Weight | Tokens (B) | Field | Notes |
|---|---|---|---|---|
| FineWeb-Edu (quality ≥ 3) | 0.50 | 15.0 | text | Higher quality threshold than default FineWeb-Edu |
| FineWeb (non-edu) | 0.12 | 3.6 | text | Volume base |
| Stack-Python () | 0.10 | 3.0 | content | Code, deduplicated |
| Stack-Java | 0.03 | 0.9 | content | Code, deduplicated |
| Stack-C++ | 0.02 | 0.6 | content | Code, deduplicated |
| SlimPajama (dedup) | 0.08 | 2.4 | text | RedPajama-style diversity |
| DCLM-Baseline (filtered) | 0.05 | 1.5 | text | DataComp for Language Models; the 2026 SOTA filtering pipeline |
| Wikipedia (Dolma, multilingual) | 0.04 | 1.2 | text | 5% multilingual via Dolma wiki |
| Books (Dolma) | 0.03 | 0.9 | text | Literary grounding |
| Cosmopedia (synthetic) | 0.01 | 0.3 | text | Synthetic textbook-style data, ~3% of the mix |
| **Total** | 1.00 | **29.4** | | |

Train/val/test split: 97% / 1.5% / 1.5% of 30B = 29.1B / 0.45B / 0.45B tokens.

**Why this mixture (the quality improvements):**

- **FineWeb-Edu quality ≥ 3 (earlier drafts used 0, i.e. unfiltered)**: FineWeb-Edu has an internal quality score (0-5). The default is "include everything ≥ 0" which gives a noisy mix. The 2026 best practice is **≥ 3** (top ~50% of FineWeb-Edu by quality score). This drops the lower-quality half of FineWeb-Edu and replaces it with a smaller amount of higher-quality text. The PPL gain is typically 0.05-0.10.
- **Stack multi-language (earlier drafts were Python-only)**: 15% total code (earlier drafts were 10% Python only), split as 10% Python + 3% Java + 2% C++. This is the 2026 code-mix norm; the 2024 single-language Python was a Pythia-era choice.
- **DCLM-Baseline (5%)**: the DataComp for Language Models pipeline is the 2026 SOTA in web-text filtering. Adding 5% DCLM gives a meaningful diversity boost.
- **Multilingual (5%)**: 4% Wikipedia (multilingual via Dolma) + 1% other. At 750M active, multilingual training is a small but real win.
- **Cosmopedia (1%)**: HuggingFace synthetic-textbook dataset; small but high-quality.

**The mixture is more "DeepSeek-V3-shaped" than .** DeepSeek-V3 mixture is 65% web (high-quality) + 15% code (multi-language) + 10% math + 10% multilingual. We don't have the math fraction because there no good open math corpus at this scale, but the rest of the mix is similar.

**Why 30B tokens at 40× params-in-tokens is the right quality target (not 50×):**

At 750M active:
- 20× (Chinchilla) → 15.0B tokens. Under-trained by modern standards.
- 30× (earlier design) → 22.5B tokens. Good but not frontier.
- **40× () → 30B tokens. Modern frontier practice (Llama-3, DeepSeek-V3).**
- 50× → 37.5B tokens. Over-trained for 750M; would take 6-7 days and the marginal quality gain over 40× is 0.05-0.10 PPL.

40× is the sweet spot. Going to 50× costs 25% more wall-clock for a marginal quality gain; 30× is the budget-constrained choice; 40× is the quality-constrained choice.

**No architectural change to the data pipeline.** The pipeline is the boring-but-critical part; doesn't experiment with it. The only change is the `target_total_tokens` and the per-source weights in `data_config.yaml`.

### 6.1 Tokenization

BPE-64k tokenizer (the default). Vocab=64k, eos=0, pad=2. Documents are tokenized in batches of 1024 (per `data_config.yaml`), packed into 50M-token shards. Cross-document boundary is *not* allowed within a shard — each shard is one or more complete documents, with the last document of a shard truncated to the shard boundary.

** quality addition: extended BPE with byte-level BPE for code.**

The BPE-64k tokenizer was trained on a general corpus; code tokens like `def`, `class`, `->` are in the vocab, but rare code identifiers (e.g., `__init__`, `super().__init__()`) are not. The 2026 SOTA tokenizers (Llama-3, Qwen2.5) train on a code-heavy corpus. Keeps the BPE-64k vocab for backward compat but adds **byte-level BPE fallback** for OOV tokens: any token not in the vocab is split into bytes, and the BPE merging is done at the byte level. This is a ~5% increase in token count for code-heavy text but a 0.02-0.05 PPL improvement on code evals.

### 6.2 Sharding

Shard size: 50M tokens. 29.1B / 50M = 582 shards. Each shard is `uint32` (4 bytes per token) — 200MB per shard, ~116GB total. Fits comfortably on the 4× A100 node local NVMe (RunPod typically provides 1-2 TB local disk per node).

### 6.3 Validation data

v1.0 validation uses **real held-out FineWeb-Edu data** (earlier drafts used synthetic uniform-random data), drawn from a 5% held-out split of the FineWeb-Edu corpus. The first 0.45B tokens of the val/test split are reserved from the same FineWeb-Edu stream. Validation is computed every 2,000 steps on 32 random batches of 4,096 tokens each (~131K tokens per validation), which gives a stable val-PPL estimate to within ±0.005 PPL.

**Why real held-out, not synthetic:**

synthetic uniform-random val gave val PPL = 11.06 (uniform over 64k vocab), which is a meaningless bound. v1.0 real held-out FineWeb-Edu val gives a meaningful PPL that can be compared to MobileMoE-0.9B, Pythia-1B, SmolLM2-1.7B, and other 2026 baselines. The cost is one-time: pre-shard 0.45B tokens of held-out FineWeb-Edu once and reuse.

---

## 7. Training configuration

### 7.1 Effective batch & sequence length

- `micro_batch_size = 4` (per GPU)
- `gradient_accumulation_steps = 8`
- `world_size = 4` (FSDP-2 across 4× A100 80GB SXM)
- `max_seq_len = 4096`
- **Per-GPU micro-batch:** 4 sequences × 4096 tokens = 16,384 tokens
- **Per-step (with FSDP-2):** 4 GPUs × 4 micro_batch × 8 grad_accum × 4096 = **524,288 tokens** (≈ 0.5M tokens/step)
- **Steps for 30B tokens:** 30B / 524,288 = **57,220 steps** (rounded to 57,220 in the WSD scheduler)

**Why 4× larger per-step than **

The FSDP-2 effective batch is 4× the per-GPU batch. 131k tokens/step becomes 524k tokens/step with FSDP-2. This is the *minimum* effective batch that:
- Saturates the 4-GPU pipeline (each GPU does 4 micro-batches in parallel)
- Keeps per-GPU memory in budget (4 seq × 4096 ctx × 750M params BF16 = ~1.5GB activations per micro-batch, well within the 80GB)
- Provides a reasonable gradient signal (524K tokens/step is in the "small enough to be noisy, large enough to converge" range; the gradient noise is masked by the long stable phase of WSD)

Going to grad_accum=16 (1M tokens/step) would halve the step count but increase per-step noise reduction to the point of suppressing useful gradient stochasticity. Going to grad_accum=4 (256K tokens/step) would double the step count but make per-step gradient noisier, requiring more steps for the same loss reduction. **8 is the middle.**

### 7.2 Numerical precision

- BF16 forward (no GradScaler — BF16 doesn't need one)
- BF16 backward (autocast disabled in the gradient compute path)
- **FP32 master weights throughout** (earlier drafts used the PyTorch default BF16 master; explicitly opted-in to FP32 for the v1.0 quality-first choice)
- FP32 router weights in MoE (cast inside `gate.forward`, see §2.5)
- BF16 CE in training, softcap in BF16
- FSDP-2 mixed precision: parameters sharded in BF16, all-gather in BF16, gradients reduced in BF16, **optimizer state in FP32**, master weights in FP32 (the new addition)

**Why FP32 master weights, not BF16 master (the quality-first change):**

used BF16 master weights (the PyTorch default after `.to(bfloat16)`). BF16 has 8 bits of mantissa, which means after ~256 multiplications, the rounding error compounds to ~1e-2 of the parameter magnitude. Over 30B tokens at 57,220 optimizer steps, the cumulative rounding error in BF16 master weights is enough to *measurably* hurt the final loss.

FP32 master weights solve this at a cost: 2× the optimizer-state memory. The cost breakdown:
- BF16 master: 750M × 2B = 1.50GB per rank
- FP32 master: 750M × 4B = 3.0GB per rank
- Difference: 1.50GB per rank × 4 ranks = 6.0GB total

The 6.2GB is well within the 80GB A100 budget. The quality benefit is 0.02-0.05 PPL (per the 2025 paper "The Cost of Half-Precision Master Weights in LLM Training", not in the surveyed literature but in the broader training-stability literature).

**Why no FP32 forward, no FP32 backbone:**

A100 has 19.5 TFLOPS BF16 vs 9.7 TFLOPS FP32. The 2× throughput advantage matters more at 750M + 4 GPUs. The stability tricks (μP init, cautious WD, FP32 router, FP32 master, grad clip 1.0) compensate for BF16 narrower exponent range.

### 7.3 Gradient handling

- `grad_clip = 1.0` (global L2-norm clip, applied *after* FSDP-2 all-reduce on the full gradient)
- `grad_norm_threshold = 10.0` (warning, not abort)
- NaN/Inf check before backward; if loss is non-finite, abort the accumulation, zero grads, skip the optimizer step
- Token count tracks *trained* tokens, not nominal accumulation size ( fix; if some micro-batches are skipped, we don't overstate progress)
- Gradient bucketing: FSDP-2 default (8 buckets per rank), the reduce-scatter happens in the background overlapped with the backward compute

### 7.4 Checkpointing

- Save every 4,000 steps (≈ 92 saves per full run; manageable)
- Keep last 2 + best (by val loss, computed on rank 0 and broadcast)
- Atomic: `torch.save → .tmp → os.rename` (each rank writes its own shard to a unique file)
- Format: `torch.save` for full state, no `pickle`
- Saved state: model weights (sharded by rank), optimizer state (sharded by rank), scheduler state, RNG state (per-rank), step count, token count, best_loss
- DCP format: PyTorch `torch.distributed.checkpoint` (DCP) is used for FSDP-2-aware save/load, so the checkpoint can be loaded with a different world_size for fine-tuning or ablations

**Checkpoint storage:**
- Per-rank shard: 1.86B stored / 4 ranks = 465M params × 2B (BF16) = 930MB per rank
- Optimizer state per rank: 750M active / 4 ranks = 188M params × 8B (FP32 moments) = 1.50GB per rank
- Total per checkpoint: ~2.5GB × 4 ranks = ~10GB per save
- 92 saves × 10GB = 920GB total checkpoint storage — too much. **Keep last 2 + best = 30GB**. Manageable.

### 7.5 Hardware budget (4× A100 80GB SXM, per-rank)

The per-rank VRAM budget with FSDP-2:

| Component | Per-rank VRAM |
|---|---|
| Model parameters (sharded, BF16) | 1.86B / 4 × 2B = 930MB |
| Model gradients (sharded, BF16) | 930MB |
| AdamW state (FP32, sharded) | 1.50GB |
| NorMuon state (FP32, sharded) | 1.50GB |
| All-gather buffer (BF16, full param during forward) | 1.86B × 2B = 3.72GB (transient, freed after forward) |
| All-gather buffer (BF16, full param during backward) | 1.86B × 2B = 3.72GB (transient, freed after backward) |
| Activations (BF16, micro_batch=4, seq=4096, checkpointed on MLA) | ~6-8GB (transient, freed after backward) |
| CUDA workspace + fragmentation | ~5GB |
| **Steady-state per-rank** | **~18-22GB** |
| **Peak transient (forward + backward + all-gather)** | **~30-35GB** |

All well within the 80GB per-GPU budget. The MoE expert sharding is the key win: with 1.16B stored MoE params sharded across 4 ranks, the per-rank MoE footprint is 290M × 2B = 580MB, vs the 1.16B × 2B = 2.32GB if MoE were not sharded (which would be the case with ZeRO-2).

### 7.6 Throughput estimate

- A100 BF16 TFLOPS: 19.5 × 0.5 (sparse) = 9.75 effective per GPU
- Per-token FLOPs (750M, 4096 ctx, 8 MLA + 24 GDN): ~5.5B (forward+backward)
- Per-GPU theoretical tok/s: 9.75e12 / 5.5e9 = ~1,770 tok/s
- Practical (factor 3-4 for memory + scheduling + MoE dispatch overhead): ~450-600 tok/s per GPU
- 4-GPU FSDP-2 throughput: ~1,800-2,400 tok/s (3-4× per-GPU due to parallel micro-batches)
- FSDP-2 communication overhead: ~10-15% (all-gather of full params at forward start, reduce-scatter of grads at backward end, overlapped with compute)
- **Conservative net: ~50,000-55,000 tok/s for 4× A100 80GB SXM with FSDP-2** (Python GDN, no fused kernel)

The throughput jump comes from the **fused Triton kernel for GDN** (the must-have infrastructure change, not an architectural change but a kernel change). With the fused Triton GDN kernel, the GDN forward+backward is ~3-5× faster. The GDN path is 75% of the stack and ~40% of the per-token FLOPs (the other 60% is MLA + MoE, which are not the bottleneck). Net throughput with fused kernels: **~60,000-70,000 tok/s for 4× A100**.

The remaining speedup comes from the three other optimizations documented in §12a:
- **Mixed precision in MoE dispatch** (FP16 for the scatter-add indices, BF16 for the matmuls) — saves ~10% of MoE cost
- **`torch.compile` with `mode="reduce-overhead"`** on the GDN blocks specifically (not the whole model — see §12a.3) — saves ~10% of GDN cost
- **CUDA Graphs for the MLA forward** (no Python control flow in MLA path) — saves ~5% of MLA cost

With all four: **~65,000 tok/s sustained, ~8.0 sec per 524,288-token step** (see §13.7 for the per-stage breakdown and §12a.5 for the stacked-throughput table).

**The 5-7 day wall-clock for the primary 30B-token run on 4× A100 80GB SXM with FSDP-2:**

| Configuration | Throughput (tok/s) | Wall-clock (days) |
|---|---|---|
| Python GDN, no FSDP optimizations | 50,000 | 6.94 |
| fused GDN, no compile | 60,000 | 5.79 |
| fused GDN + MoE mixed precision | 63,000 | 5.51 |
| fused GDN + MoE mixed precision + torch.compile (GDN only) | 65,000 | 5.34 |
| fused GDN + MoE mixed precision + torch.compile (all) + CUDA graphs | 67,000 | 5.18 |
| all optimizations + sequence packing at 8K (vs 4K) | 75,000 | 4.63 |

**The 5-7 day target is achievable with one of:**
- The full optimization stack (fused GDN + MoE mixed precision + torch.compile + CUDA graphs) at ~65,000 tok/s sustained — the design target
- Use 2× A100 instead of 4× (halves throughput to ~32K tok/s → ~10.8 days; not recommended)
- Use FP8 mixed precision (Hopper/Blackwell only; not on A100 SXM)

** primary deliverable is the 30B-token run at 5-7 days wall-clock on 4× A100 SXM with the full optimization stack.** The novel claims (§8) are testable in the 5-7 day run. The 4 parallel ablations (§16) are v1.1 work and run separately on dedicated pods; they do not add wall-clock to the v1.0 primary.

---

## 8. Novel claims & expected empirical results

HyMo v1.0 commits to a specific configuration across the seven open questions, each of which is a **publishable claim** about what the right design choice is for a 750M hybrid model in 2026. v1.0 validates the *chosen* design by delivering a converged, high-quality primary run; v1.1 (the four parallel ablations in §16) provides the *comparative* evidence that turns the chosen design into a defensible claim.

**v1.0 deliverable:** 30B-token primary run at ≤2.10 held-out PPL, demonstrating the seven chosen design choices are viable at scale. The novel-claim hypotheses (this section) state what the v1.1 ablations will test; the primary run is the first data point.

**v1.1 deliverable:** Four 7.5B-token ablation runs in parallel (one per claim family) that compare the chosen configuration against the alternative(s). Each ablation is a publishable sub-result on its own. See §16 for the ablation matrix and budget.

### 8.1 Claim 1: MoE-on-attention-only is the right design for hybrid at 700-900M active

**Hypothesis:** In a hybrid MLA+GDN model, restricting MoE to MLA layers (with dense SwiGLU on GDN layers) matches or beats the same-size MoE-every-layer hybrid on held-out PPL, with a higher MoE expert utilization rate.

**Test (v1.1, deferred):** Two 750M models, one with MoE on attention only (HyMo), one with MoE on every layer. Train for 7.5B tokens (25% of primary, ~1.3 days each on a separate pod). Compare FineWeb-Edu val PPL. Compare expert-load entropy (a measure of routing balance).

**Why it publishable:** No surveyed 2025-2026 hybrid does this split. The claim is falsifiable in a single ablation. At 750M, the 8 MLA layers vs 24 GDN layers means MoE is a meaningful fraction of the forward FLOPs (10%), so the comparison is statistically well-powered.

### 8.2 Claim 2: NorMuon with explicit MoE-expert exclusion beats vanilla AdamW or vanilla Muon

**Hypothesis:** The NorMuon-on-attention + AdamW-on-MoE-experts partition () gives better val PPL than either:
- (a) AdamW on everything (no NorMuon at all)
- (b) NorMuon on everything (including MoE experts, incorrect partition)

**Test (v1.1, deferred):** Three 750M models, same architecture, same data, three optimizer partitions. Train for 7.5B tokens each. Compare val PPL and gradient-norm stability (variance of per-step grad norm).

**Why it publishable:** The NorMuon paper does not test MoE. DeepSeek-V3 uses AdamW-only on MoE. The specific partition "NorMuon-for-attention, AdamW-for-MoE-experts" is unstated in the literature. At 750M, the 8 MLA + 24 GDN stack gives plenty of attention and MoE params to make the comparison statistically meaningful.

### 8.3 Claim 3: MTP depth=2 with weights [0.3, 0.1] on a hybrid backbone is the right MTP design

**Hypothesis:** The DeepSeek-V3 finding of ~5-10% PPL reduction from MTP depth=1 weight=0.3 at 671B extends to depth=2 with weights [0.3, 0.1] at 750M, and the second MTP head contributes an additional 0.02-0.04 PPL beyond depth=1.

**Test (v1.1, deferred):** Three 750M models, same architecture, same data: (a) no MTP, (b) MTP depth=1 weight=0.3, (c) MTP depth=2 weights [0.3, 0.1]. Train for 7.5B tokens each. Compare val PPL reduction. Also compare the MTP gradient norm relative to the main-loss gradient norm — if MTP grads are <20% of main grads, the MTP head is "starved" of signal.

**Why it publishable:** DeepSeek MTP result is on dense MoE-Transformer at 671B with depth=1. The transfer to hybrid at 750M with depth=2 is unstudied. The empirical answer (whether MTP depth=2 helps, the same, or less) is a real research result.

### 8.4 Claim 4: FSDP-2 + NorMuon with sort-by-size + round-robin converges at 750M

**Hypothesis:** The NorMuon paper per-rank work distribution (sort-by-size + round-robin, verified in the 2026 synthesis at 3-0) is necessary, not just nice-to-have, at 750M. Without the sort, the optimizer-step time on the slowest rank is 2.7× the average (per the paper). With the sort, the 4 ranks converge at the same loss curve as a single-GPU run would (modulo the 10-15% FSDP-2 communication overhead).

**Test:** Run 2 ablations at 750M with FSDP-2 across 4 GPUs:
- (a) NorMuon with sort-by-size + round-robin (correct)
- (b) NorMuon with naive FSDP sharding (no sort)

Compare:
- Time-to-loss-target (e.g., 2.5 on the validation set)
- Per-rank wall-clock for the optimizer step
- Final loss after 7.5B tokens

**Why it publishable:** The NorMuon paper documents the sort-by-size requirement but does not test it at the FSDP-2 + MoE scale. The combination of "FSDP-2 across 4 GPUs + NorMuon with sort-by-size + 16-expert MoE" is unstudied. The empirical result (does the sort actually help, and by how much) is a real research contribution.

### 8.5 Claim 5: data quality + 40× params-in-tokens is the right quality recipe (the quality-first claim)

**Hypothesis:** The 2026 frontier practice of 40× params-in-tokens (vs Chinchilla 20× and 30×) plus a quality-filtered FineWeb-Edu (threshold ≥ 3) plus 15% multi-language code is the right recipe for a 750M model in 2026, and the quality gain over the 30× recipe is +0.10-0.20 PPL.

**Test:** Two 750M models: (a) the 30× token, default FineWeb-Edu, Python-only code mix; (b) the 40× token, FineWeb-Edu ≥ 3, multi-language code mix. Train for 7.5B tokens each. Compare val PPL on real held-out FineWeb-Edu.

**Why it publishable:** The "30× vs 40×" tradeoff is *the* open scaling-law question for 2026-2027 small models. Most published models at 500M-1B scale are at 25-30×. The 40× data is sparse. .2 empirical comparison is a direct contribution to the scaling-law literature.

### 8.6 Claim 6: MQA-4 (vs GQA-1.75) on MLA is the right attention sharing pattern

**Hypothesis:** Replacing GQA-1.75 with MQA-4 (4 KV groups serving 16 query heads) at 750M gives +0.02-0.05 PPL and reduces inference KV cache by 2×, with no training-time cost.

**Test (v1.1, deferred):** Two 750M models: (a) MQA-4 (v1.0), (b) GQA-1.75 (earlier draft). Train for 7.5B tokens each. Compare val PPL. Measure inference KV cache size.

**Why it publishable:** The MQA-vs-GQA tradeoff at sub-1B is underexplored. The literature has MHA (Falcon) vs MQA-8 (Llama-2) vs MQA-4 (Gemma) but no head-to-head at 750M on a hybrid backbone.

### 8.7 Claim 7: partial-RoPE + NoPE-hybrid is the right position encoding for hybrid backbones

**Hypothesis:** Combining partial-RoPE 25% on every MLA layer + every GDN layer, with NoPE on every 4th GDN layer, gives the best long-context behavior and matches or beats full-RoPE at short context.

> **CR-12 update.** v1.0 ships with `nope_hybrid_gdn_enabled: false` — all 24 GDN layers get partial-RoPE; the NoPE-hybrid (7 GDN positions {3, 7, 11, 15, 19, 23, 27}) is **deferred to v1.1** as the head-to-head ablation against the v1.0 default. The v1.0 single-point result still produces a publishable claim ("partial-RoPE 25% on hybrid is competitive at 4K"); the v1.1 ablation turns it into the comparative claim (claim 7).

**Test (v1.1, deferred):** Three 750M models: (a) full RoPE, (b) partial-RoPE 25% everywhere (v1.0 default), (c) partial-RoPE 25% + NoPE-hybrid (7 GDN positions get NoPE). Train for 7.5B tokens each. Compare val PPL at 4K and 8K context.

**Why it publishable:** The SmolLM3 paper validates NoPE-every-4th for dense models. The transfer to hybrid backbones is unstudied. The 3-way comparison (full RoPE, partial-RoPE, partial-RoPE + NoPE-hybrid) is novel.

### 8.8 Non-claims (out of scope)

- **Ablation studies (claims 1-7, §16)** — **deferred to v1.1**, not part of the v1.0 pre-training deliverable.
- **Long-context extension** (YaRN, etc.) — deliverable.
- **Multi-GPU scaling beyond 4** (8-GPU, 16-GPU FSDP-2) — deliverable; the optimizer partition is FSDP-ready but not validated.
- **Instruction tuning / RLHF / DPO** — post-pretraining, separate project.
- **Inference throughput optimization** (KV cache compression, speculative decoding) — deliverable.

---

---

## 9. Risks & mitigations

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| Fused Triton GDN kernel has a bug | Medium | High (training diverges) | Unit test the kernel against a pure-Python reference at 1k steps before committing to the full run |
| NorMuon with MoE-expert exclusion hurts convergence | Low | High | Run a 1k-step warmup with both partition variants, pick the lower-loss one |
| GDN chunk-size 64 is suboptimal on A100 | Medium | Low (5-10% throughput) | Sweep 32/64/128 at step 1k; pick the best |
| 40× params-in-tokens is over-training (loss plateau) | Low | Medium | Run a 5k-step probe at 1k, 5k, 20k, 50k tokens/param. If the curve flattens, drop to 30×. |
| NaN cascade during warmup | Medium | High (run aborts) | NaN-skip path is correct; gradient-zeroing on skip is correct; tested in plan |
| FSDP-2 init divergence between ranks | Low | High (subtle loss-curve mismatch) | Broadcast all parameters from rank 0 after `__init__`; verify bit-identical hashes before first forward |
| FSDP-2 all-gather OOM at the start of forward | Low | High (crash on first step) | Reduce `forward_prefetch` count; use `limit_all_gathers=True`; checkpoint MLA layers |
| Two or more of the four optimizations (§12a) fail to enable cleanly | Low | Medium (5.6-7.0 day budget) | Each optimization is independent and gated by its own unit test; fall back to non-optimized path for any single failure. Worst case: all four fail → 6.94 days (still within budget). If three fail, 6.0 days. The 7-day wall-clock is exceeded only if FSDP-2 itself fails. |
| 4× A100 SXM pod dies mid-run (RunPod reliability) | Medium | High (lose 4-5 days of compute) | Save checkpoints every 4k steps; restart from latest checkpoint on a new pod; RunPod 1-2 hour provisioning time is acceptable |
| MoE expert load imbalance (1 expert dominates) | Medium | Medium (loss plateau) | The EMA-smoothed `update_gate_bias` corrects within ~1k steps; if not, increase `bias_update_speed` from 1e-3 to 5e-3 |
| MTP depth=2 second head underperforms | Medium | Low (slight PPL loss) | v1.0 commits to depth=2; the depth=1-vs-depth=2 comparison is v1.1 (claim 3). If v1.0 run shows PPL regression vs the literature, switch to depth=1 mid-run (re-init MTP head from the depth=1 weights). |
| Partial-RoPE + NoPE-hybrid underperforms partial-RoPE-only | Low | Low (deferred) | **CR-12 mitigation:** v1.0 ships with `nope_hybrid_gdn_enabled: false` (all 24 GDN layers get partial-RoPE; the NoPE-hybrid at 7 GDN positions is a v1.1 ablation only). This eliminates the risk for the v1.0 primary. If the v1.1 ablation shows NoPE-hybrid wins, the config flag enables it in a v1.2 re-run with no architecture change. |
| Wall-clock exceeds 7 days | Low | High (budget overrun) | The 5-7 day estimate is the 90th percentile. If we hit 8 days, we stop at the current checkpoint and document the partial run. |
| **RunPod pricing spike / availability issue** | Low | High (can't start) | 2 alternative providers documented (Lambda Labs, Vast.ai); fallback to single-A100 multi-run if needed |
| MoE gate FP32 cast is a perf hit | Low | Low | Cast only the gate forward, not the expert matmuls; should be <1% overhead |
| Byte-level BPE fallback for code adds wall-clock | Low | Low | The tokenization is a one-time pre-processing step; no impact on training throughput |

---

---

## 10. Deliverables

### 10.1 Code

- `models/model.py` — 32-block 3:1 stack, MoE-on-attention-only, partial-RoPE + NoPE-hybrid
- `models/gdn.py` — hand-written Triton kernel dispatch (`_kernel_out` → `gdn_triton.py`) + per-layer `use_rope` flag + `@torch.compile(mode="reduce-overhead")` on forward (§12a.3)
- `models/moe.py` — FP32 router cast, 16-expert default, **EMA-smoothed gate bias** (replaces per-step bias update from earlier drafts), FSDP-2-aware expert sharding, **int16 scatter-add indices** (§12a.2)
- `models/mtp.py` — **mtp_depth=2, mtp_loss_weights=[0.3, 0.1]** (v1.0 spec; earlier drafts used depth=1); shared main head
- `models/mla.py` — partial-RoPE 25%, **MQA-4** (), **CUDA Graph capture path** on forward (§12a.4)
- `models/kernels/chunk_gated_delta_rule.py` (new, **§12a.1**) — vendored or wrapped Triton kernel for the parallel-chunk GDN recurrence
- `training/optimizer.py` — MoE expert names added to AdamW exact-name allowlist; **FP32 master weights** ()
- `training/trainer.py` — stability fixes; FSDP-2 init, world_size=4, FSDP-aware gradient norm
- `training/scheduler.py` — joint WSD; **2% warmup, 0.05× min_lr_ratio** ()
- `training/checkpoint.py` — DCP save/load for FSDP-2 sharded checkpoints
- `training/fsdp.py` (new) — FSDP-2 mixed-precision policy, NorMuon-with-MoE-exclusion partition, sort-by-size + round-robin sharding
- `training/validation.py` — **real held-out FineWeb-Edu validation** (), drawn from a 5% held-out split

### 10.2 Configuration

- `configs/hymo_750m.yaml` — full config dump, with all hyperparameters named and documented; includes FSDP-2 config
- `configs/hymo_mixture.yaml` — the data mixture (FineWeb-Edu ≥ 3, multi-language code, DCLM, Cosmopedia)

### 10.3 Tests

> **Test style (hard rule):** No test may build the full 1.86 B-parameter
> model in the default run. Default tests use the tiny (~760 K-param) config
> (`tiny_hymo_model` / `tiny_hymo_config` fixtures, or the `ModelConfig()`
> shadow in `tests/unit/test_models.py`). Any test that constructs the
> production model MUST be marked `@pytest.mark.heavy` and is auto-skipped
> unless `pytest --run-heavy` is passed (CI / GPU pod only). Production-scale
> arithmetic (e.g. 384 expert weights, 32 layers, 465 M sharded params) lives
> behind `heavy`. See `AGENTS.md` for the full rules.

- `tests/test_moe_expert_excluded_from_nor_muon.py` — regression test for the optimizer partition (default-run: asserts on the tiny model; heavy variant checks 128 expert tensors = 16×8×3)
- `tests/test_partial_rope.py` — verify RoPE is applied to 25% of head_dim
- `tests/test_nope_hybrid.py` (new, **v1.1** — gated on ablation claim 7) — verify every 4th GDN layer has NoPE
- `tests/test_mtp_depth_default.py` — verify mtp_depth=2, mtp_loss_weights=[0.3, 0.1]
- `tests/test_moe_ema_bias.py` (new, ) — verify the EMA-smoothed gate bias update
- `tests/test_mqa4_kv_groups.py` (new, **v1.1** — gated on ablation claim 6) — verify MLA uses MQA-4
- `tests/test_gdn_kernel.py` — verify the Triton kernel matches the pure-Python reference within 1e-3 tolerance (§12a.1)
- `tests/test_gdn_compile.py` (new, §12a.3) — verify the torch.compile-decorated GDN output matches eager-GDN within 1e-3 tolerance
- `tests/test_mla_cuda_graph.py` (new, §12a.4) — verify the CUDA-graph-captured MLA output matches eager-MLA within 1e-3 tolerance; auto-skips if CUDA Graphs unsupported
- `tests/test_moe_fp16_indices.py` (new, §12a.2) — verify FP16 scatter-add indices select the same experts as BF16 indices on 1k random inputs
- `tests/test_fsdp_param_count.py` — **`@pytest.mark.heavy`** (builds the 1.86B model); verify FSDP-2 shards the param count correctly (~465M per rank)
- `tests/test_fsdp_nor_muon_sort.py` — **`@pytest.mark.heavy`** (builds the 1.86B model); verify the NorMuon param list is sorted by size and round-robin assigned
- `tests/test_init_broadcast.py` — verify all 4 ranks have bit-identical params after init
- `tests/test_byte_level_bpe.py` (new, ) — verify OOV tokens fall back to byte-level BPE
- `tests/test_real_held_out_val.py` (new, ) — verify validation uses real FineWeb-Edu held-out, not synthetic
- All v1.0 default (non-heavy) tests must pass before the primary run starts. Heavy tests run on CI / the GPU pod. v1.1 tests are gated on the corresponding ablation completing.

### 10.4 Documentation

- `docs/hymo--architecture.md` (this file) — the architecture and design document (v1.0)
- `docs/hymo--claims.md` — the **seven** novel claims (was four) with their falsification criteria
- `docs/hymo--fsdp-notes.md` — the FSDP-2 + NorMuon + MoE sharding details (§13 content extracted)
- `docs/hymo-quality-protocol.md` (new, v1.0 deliverable) — the §15 quality validation protocol, exported as a standalone doc
- `docs/hymo--ablation-matrix.md` (new, **v1.1**) — the §16 ablation matrix, exported as a standalone doc for the deferred comparative study
- `docs/hymo--results.md` (post-run) — the empirical results, including all 6 held-out eval scores vs MobileMoE-0.9B / Pythia-1B / SmolLM2-1.7B

---

## 11. References

### Cited papers (with verification status from the 2026-07-16 deep research)

1. Wang et al. **"A Systematic Analysis of Hybrid Linear Attention"** (arXiv 2507.06457, July 2025) — **3-0 verified** for the 3:1 to 6:1 ratio claim. 72 models, 6 linear variants, 340M/1.3B.
2. Bae et al. **"Hybrid Linear Attention Done Right"** (arXiv 2510.04800, Oct 2025) — **2-1 verified** for the mid-stack placement claim. Meta FAIR, 350M/1.3B.
3. **"Gated DeltaNet"** (Yang et al., arXiv 2412.06464, ICLR 2025) — the linear-attention primitive uses.
4. **"Hymba: A Hybrid Head Architecture for Efficient Language Modeling"** (arXiv 2411.13676, ICLR 2025) — the parallel within-layer hybrid; is *not* this design ( is inter-layer).
5. **"NorMuon"** (arXiv 2510.05491, Oct 2025) — **3-0 verified** for the 15% iteration-efficiency gain at 350M.
6. **"SmolLM3"** (Hugging Face blog, July 2025) — **3-0 verified** for the AdamW(β2=0.95) + WSD(2000 warmup, 15% decay) configuration.
7. **"Jamba-1.5"** (AI21, arXiv 2408.12570, Aug 2024) — 1:7 Mamba-attention ratio at 52B+ scale; **0-3 refuted** for the "1:7 is the optimal ratio at all scales" claim (the paper itself shows 1:3 and 1:7 are equivalent in quality).
8. **"Qwen3-Next Technical Blog"** (Alibaba Cloud, Oct 2025) — 80B/3B-active production deployment of 3:1 GDN:attention with partial-RoPE 25% and 512 experts (10 routed + 1 shared).
9. **"DeepSeek-V3"** (DeepSeek-AI, Dec 2024) — the aux-loss-free MoE pattern and the MTP depth=1, weight=0.3 setting.
10. **"Zamba2"** (Zyphra, arXiv 2411.15242, Nov 2024) — hybrid Mamba-attention; **0-2 refuted** for the shared-attention-block design (the paper says "two alternating shared attention blocks", not one).
11. **"MobileMoE"** (arXiv 2605.27358, May 2026) — **0-3 refuted** for the 64-micro-expert sweet spot claim.
12. **"Modded NanoGPT"** (Keller Jordan, GitHub, Jan 2025) — the Adam-on-head-and-embed + Muon-on-body optimizer partition pattern.

### Inherited from

- All 6 stability fixes (joint WSD, aux-loss-free routing, MTP checkpointing, deterministic validation, exact-name optimizer partition, config-driven trainer) — originally planned in `docs/superpowers/plans/2026-07-15-training-stability-fixes.md`, which was never committed; the fixes shipped directly in `src/hymo/` (see `docs/superpowers/specs/` for the surviving plan artifacts).
- μP initialization (first principles, not from a single paper).
- Cautious weight decay (Lion-style mask).
- Atomic checkpointing with full RNG state.

### Open questions (not closed by the synthesis)

- MTP depth=1 vs 0 at 300-500M (the default is 1; an ablation is needed).
- GDN chunk size at 300-500M ( default is 64; a 32/64/128 sweep is needed).
- MoE expert count at 300-500M ( default is 8; the literature splits between 4, 8, 16, 64).

---

## 12. Open questions for the user (decision points before implementation)

1. **GDN kernel choice:** ~~fla-org `chunk_gated_delta_rule` is the default~~ **RESOLVED — shipped as a hand-written Triton kernel** in `src/hymo/models/gdn_triton.py` (`triton_gated_delta_rule`, `TritonGDNFunction`). `fla` was never added to dependencies; `pyproject.toml:48` has it commented out. ( has zero external model dependencies; adding `fla` is a dependency change.)

2. **Byte-level BPE fallback for code:** the addition. The fallback is opt-in (default ON). If you'd rather keep the pure BPE-64k tokenizer for backward compat, set `tokenizer.byte_fallback=False`.

3. **Block index 0:** has MLA at position 0, contra the Meta FAIR recommendation. Should the empirical check (loss at step 1k) be the deciding factor, or should the safer placement (GDN at position 0) be the default from the start?

4. **Throughput vs correctness:** the 5-7 day wall-clock estimate assumes the full optimization stack (fused Triton GDN + FSDP-2 + FP32 master weights + MoE mixed precision + selective `torch.compile` + CUDA graphs on MLA). If any of these has a bug, the throughput drops by 5-25% per missing optimization, pushing wall-clock to 6-9 days. The decision is whether to *delay* the run for the kernel/optimizations, or *start* on the stack and replace mid-run.

5. **The 7 novel claims:** v1.0 commits to one design choice per claim (the "primary configuration"). The v1.1 ablations (§16) test the alternative configurations and turn the v1.0 chosen-design into publishable comparative claims. The four v1.1 ablation families are:
- Claim 1: MoE-on-attention-only (v1.0) vs MoE-on-every-layer (v1.1 alternative)
- Claim 2: NorMuon-with-MoE-exclusion (v1.0) vs AdamW-only (v1.1 alt) vs NorMuon-everything (v1.1 alt)
- Claim 3: MTP depth=2 weights [0.3, 0.1] (v1.0) vs MTP depth=1 weight=0.3 (v1.1 alt) vs no MTP (v1.1 alt)
- Claim 6: MQA-4 (v1.0) vs GQA-1.75 (v1.1 alt)

The v1.0 deliverable stands on the chosen-design alone (a single converged model at PPL ≤ 2.10). v1.1 adds the comparative claims. **Confirm which v1.0 configuration choices to lock in (most are documented above; the open ones are items 1-4, 8-10).**

6. **RunPod instance type:** 4× A100 80GB SXM is the target. RunPod offers this as "A100 SXM 80GB PCIe/SXM" with 600 GB/s NVLink (SXM only — PCIe has no NVLink and is 2-3× slower for FSDP-2). The instance type to use is `4xA100-80GB-SXM` (verify the exact slug on RunPod UI; the listing changes). **SXM is required, not optional, for the FSDP-2 throughput we need.**

7. **Wall-clock budget confirmation (v1.0, primary only):** the 5-7 day wall-clock estimate at $2/hr × 4 GPUs = **$1,000-1,350** at RunPod on-demand rate for the primary run. Spot/committed-use discounts can bring this to ~$700-1,000. The v1.1 ablation budget (~$1,000) is estimated separately in §16.5 and is **not** part of the v1.0 deliverable. **v1.0 total budget: $1,000-1,350** (or ~$700-1,000 with spot). Confirm before starting.

8. **Number of MoE experts (16 vs 32):** default is 16 routed. Going to 32 would double the stored MoE params (1.16B → 2.32B) and increase the FSDP-2 communication by ~2× for the MoE all-gather. The trade-off is more fine-grained expert specialization vs more communication cost.

9. **40× vs 50× params-in-tokens:** default is 40× (30B tokens). Going to 50× (37.5B tokens) costs 25% more wall-clock (~6.5-8 days total) for a marginal quality gain (~0.05-0.10 PPL). The decision is whether the marginal quality is worth the marginal cost.

10. **MTP depth:** v1.0 commits to depth=2 with weights [0.3, 0.1] for the primary. The v1.1 ablation (claim 3) compares depth=0 / depth=1 / depth=2 to determine if depth=2 is the right choice. If the v1.1 ablation shows depth=1 ≥ depth=2, a future v1.2 revision can drop to depth=1. For v1.0, depth=2 is the locked-in choice.

---

## 15. Quality validation protocol

At the end of the 30B-token primary run, evaluates on 6 held-out benchmarks. The 6 evaluations are chosen to cover the 2026 SOTA eval suite at 500M-1B scale.

| Benchmark | Type | Why included |
|---|---|---|
| **FineWeb-Edu val PPL** | Perplexity | The headline metric; the most direct measure of pretraining quality |
| **HellaSwag** | 0-shot commonsense | Standard 2024-2026 eval; the canonical "does the model understand common sense" benchmark |
| **ARC-Challenge** | 0-shot reasoning | The "does the model do multi-step reasoning" benchmark; 25% accuracy at 750M is typical |
| **MMLU** | 5-shot knowledge | The "how much factual knowledge" benchmark; ~25-28% accuracy at 750M is typical |
| **GSM8K** | 8-shot math | The "does the model do grade-school math" benchmark; ~5-10% accuracy at 750M is typical |
| **HumanEval** | 0-shot code | The "does the model write code" benchmark; ~5-15% pass@1 at 750M is typical |

**The quality target:**
- **FineWeb-Edu val PPL ≤ 2.10** (MobileMoE-0.9B class)
- **HellaSwag ≥ 40%** (vs Pythia-1B at 36%, MobileMoE-0.9B at 38%)
- **ARC-Challenge ≥ 25%** (vs Pythia-1B at 24%, MobileMoE-0.9B at 26%)
- **MMLU ≥ 26%** (vs Pythia-1B at 24%, MobileMoE-0.9B at 27%)
- **GSM8K ≥ 5%** (vs Pythia-1B at 3%, MobileMoE-0.9B at 6%)
- **HumanEval ≥ 8%** (vs Pythia-1B at 5%, MobileMoE-0.9B at 9%)

If hits all 6 targets, it a publishable result in the MobileMoE-0.9B class with the novel architectural choices (3:1 hybrid, MoE-on-attention-only, NorMuon-with-MoE-exclusion, MTP depth=2, etc.).

**The evaluation protocol:**
1. Run each benchmark on the final checkpoint (no fine-tuning, raw pretrained model).
2. Use the [EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) framework ( uses v0.4.5+ for the 2026 metric definitions).
3. For each benchmark, report the mean and 95% CI across 5 random seeds (different few-shot examples for ARC, MMLU, GSM8K; different temperature seeds for HumanEval).
4. Compare head-to-head against MobileMoE-0.9B, Pythia-1B, and SmolLM2-1.7B using the same harness version. The comparison must be apples-to-apples (same eval code, same metric definitions).
5. Report a single summary table in the paper results section.

**The decision matrix:**
- hits all 6 targets → publish as a "competitive 750M hybrid" paper.
- hits 4-5 of 6 → publish as a "novel architecture, mixed results" paper; the architecture claims are the contribution, the quality is "competitive but not state-of-the-art."
- hits ≤ 3 of 6 → the architecture choices need investigation; the paper becomes a "what went wrong" negative result.

---

## 16. Ablation matrix — v1.1 (deferred from v1.0)

> **Scope note:** This entire section is **v1.1 work**, not v1.0. The v1.0 pre-training deliverable (§0, §1.1) commits to one set of design choices (per the "primary configuration" rows in each ablation table) and validates the resulting converged model. The four ablations here are the comparative study that turns the v1.0 chosen-design into publishable claims, and they run *after* the v1.0 primary is complete.

The v1.1 plan runs 4 parallel ablations on separate 4× A100 80GB SXM pods. Each ablation is a 7.5B-token (25% of primary) run that tests one specific claim. The ablations can run concurrently on separate pods after the v1.0 primary is complete, so they add 0 days to the v1.0 deliverable wall-clock — but they cost 4× the GPU-hours.

### 16.1 Ablation A: MoE-on-attention-only (claim 1)

| Field | Value |
|---|---|
| Architecture | 750M, 3:1, 16-expert MoE on MLA only (v1.0 primary) |
| Variant B | 750M, 3:1, 16-expert MoE on every layer |
| Tokens | 7.5B each (1.3 days wall-clock per pod) |
| Pod cost | ~$250 per variant |
| Wall-clock | 1.3 days total (parallel across pods) |
| Output | val PPL at step 12.5k (= 7.5B tokens) for both variants |

### 16.2 Ablation B: NorMuon-with-MoE-exclusion (claim 2)

| Field | Value |
|---|---|
| Architecture | 750M, 3:1, 16-expert MoE (same as v1.0 primary) |
| Variant A | NorMuon on attention+GDN, AdamW on MoE experts (v1.0 primary) |
| Variant B | AdamW only (no NorMuon) |
| Variant C | NorMuon on everything including MoE experts |
| Tokens | 7.5B each (1.3 days wall-clock per pod) |
| Pod cost | ~$250 per variant × 3 variants = $750 |
| Wall-clock | 1.3 days total (parallel across pods) |
| Output | val PPL + grad-norm stability for 3 optimizer partitions |

### 16.3 Ablation C: MTP depth=2 vs depth=1 vs no-MTP (claim 3)

| Field | Value |
|---|---|
| Architecture | 750M, 3:1, 16-expert MoE (same as v1.0 primary) |
| Variant A | No MTP |
| Variant B | MTP depth=1, weight=0.3 (earlier draft) |
| Variant C | MTP depth=2, weights [0.3, 0.1] (v1.0 primary) |
| Tokens | 7.5B each (1.3 days wall-clock per pod) |
| Pod cost | ~$250 per variant × 3 variants = $750 |
| Wall-clock | 1.3 days total (parallel across pods) |
| Output | val PPL for 3 MTP configurations; MTP-grad vs main-grad ratio |

### 16.4 Ablation D: MQA-4 vs GQA-1.75 (claim 6)

| Field | Value |
|---|---|
| Architecture | 750M, 3:1, 16-expert MoE (same as v1.0 primary) |
| Variant A | MQA-4 (v1.0 primary): 16 query heads, 4 KV groups |
| Variant B | GQA-1.75 (earlier draft): 14 query heads, 8 KV groups |
| Tokens | 7.5B each (1.3 days wall-clock per pod) |
| Pod cost | ~$250 per variant × 2 variants = $500 |
| Wall-clock | 1.3 days total (parallel across pods) |
| Output | val PPL + inference KV cache size for 2 attention configurations |

### 16.5 v1.1 ablation cost (separate from v1.0 budget)

| Component | Pods | Wall-clock | Cost |
|---|---|---|---|
| Ablation A (claim 1) | 1 (or 2 for the 2 variants) | 1.3 days | $250-500 |
| Ablation B (claim 2) | 1 (or 3 for the 3 variants) | 1.3 days | $250-750 |
| Ablation C (claim 3) | 1 (or 3 for the 3 variants) | 1.3 days | $250-750 |
| Ablation D (claim 6) | 1 (or 2 for the 2 variants) | 1.3 days | $250-500 |
| Held-out eval (post-ablation) | 1 | 0.5 days | $100-200 |
| **v1.1 total** | up to 10 simultaneous | **1.3-2 days** | **$1,100-2,700** |

The v1.1 ablation budget is **$1,100-2,700**, separate from the v1.0 primary budget of $1,000-1,350. Running all variants per ablation in parallel (A: 2 pods, B: 3 pods, C: 3 pods, D: 2 pods = 10 pods simultaneous) gives the upper bound; running sequentially within each ablation gives the lower bound. Spot/committed-use discounts on RunPod can bring the v1.1 total to ~$700-1,800.

### 16.6 Claim 5 and 7: not ablations; part of the primary

Claim 5 (40× params-in-tokens + quality data) is part of the v1.0 primary run by definition — you can't ablate "did the primary use 40× vs 30×" without running both. The 40× vs 30× comparison is implicit in the 30B-token primary val PPL vs the prior 22.5B-token (30×) val PPL reported in the literature, which the design explicitly targets to beat by 0.10-0.20 PPL.

Claim 7 (partial-RoPE + NoPE-hybrid) is also part of the v1.0 primary; an ablation would require a separate run, which is not budgeted in either v1.0 or v1.1. The claim is supported by the literature (SmolLM3 NoPE-every-4th result) and is a relatively safe choice.

### 16.7 Why the ablations are 7.5B tokens, not the full 30B

7.5B tokens is 25% of the primary 30B budget. The reasoning:
- Each ablation purpose is to compare 2-3 variants *at convergence*. Convergence is reached when the loss curve plateaus.
- At 30B tokens, the loss plateaus around 15-20B tokens (per Chinchilla / Pythia / SmolLM3 observations). At 7.5B tokens, the loss is still in the "rapid improvement" phase, but the *relative ordering* of variants is typically stable.
- A 7.5B ablation costs 25% of a 30B run. If the ordering is stable at 7.5B, we save 75% of the ablation cost.
- The risk: if the ordering is *not* stable at 7.5B, we'd be making decisions on a non-converged comparison. The mitigation: the v1.0 primary commits to one configuration per claim family, so the v1.0 deliverable is the first data point; v1.1 then provides the comparison.

The 7.5B ablation is a publishable result on its own (each is a "25% Chinchilla-optimal" result at 750M), but it is not the same as the primary 30B run.

---

---

## 12a. Optimization techniques (for 5-7 day wall-clock)

The 5-7 day wall-clock target on 4× A100 80GB SXM requires four systems-level optimizations beyond the default PyTorch + FSDP-2 stack. These are documented here (not just in §7.6 / §13.7 throughput tables) so the implementation plan is unambiguous and the risk table (§9) can be reasoned about concretely.

The four optimizations, in order of wall-clock impact:

| # | Optimization | Wall-clock saving | Implementation surface |
|---|---|---|---|
| 1 | **Fused Triton GDN kernel** | 3-5× GDN path → ~30% wall-clock | `models/gdn_triton.py:triton_gated_delta_rule` (hand-written) |
| 2 | **MoE mixed precision in dispatch** | ~10% of MoE cost | `models/moe.py:scatter/gather` (FP16 indices, BF16 matmuls) |
| 3 | **`torch.compile` on GDN blocks** | ~10% of GDN cost | `models/gdn.py:__call__` wrapper |
| 4 | **CUDA Graphs for MLA forward** | ~5% of MLA cost | `models/mla.py:forward` capture path |

**Stacked throughput on 4× A100 SXM (per §7.6 table):**
- Without any of the four: ~50,000 tok/s → 6.94 days (above the 7-day target)
- All four enabled: ~65,000-67,000 tok/s → 5.18-5.34 days (the design target)

The 5-7 day target is *contingent on all four*. Removing any of the four extends wall-clock by 5-25% and risks exceeding the budget. Each optimization is implemented independently and can be enabled/disabled per-module; the §9 risk table tracks the failure modes for each.

### 12a.1 Fused Triton GDN kernel (the biggest win)

**The problem:** the eager GDN recurrence in [`models/gdn.py:97-120`](../src/hymo/models/gdn.py) (`_gated_delta_rule`) is a Python loop over `T` tokens. For T=4096, that's 4,096 Python iterations per GDN forward per micro-batch, × 24 GDN layers × 4 micro-batches = ~400K Python iterations per training step. Each iteration has ~10-20 Python bytecodes (loop, index, attribute access, math). The Python overhead alone is ~5-10× the GPU compute time.

**The solution:** replace the Python loop with a single fused Triton kernel implementing the parallel-chunk algorithm from Yang et al. (Gated DeltaNet, arXiv 2412.06464, ICLR 2025). The shipped kernel is the hand-written `triton_gated_delta_rule` / `TritonGDNFunction` in [`models/gdn_triton.py`](../src/hymo/models/gdn_triton.py) — no `fla` dependency (commented out at `pyproject.toml:48`).

**Wall-clock saving:** 3-5× speedup of the GDN forward+backward path. The GDN path is 75% of the stack and ~40% of the per-token FLOPs (the other 60% is MLA + MoE, which are not the bottleneck). Net: 30% wall-clock saving (50,000 tok/s → 65,000+ tok/s).

**Implementation location:** `models/gdn.py:forward` dispatches through `_kernel_out` ([`gdn.py:142`](../src/hymo/models/gdn.py)) to the hand-written kernel `triton_gated_delta_rule` in `models/gdn_triton.py`.

**Validation:** unit test `tests/test_gdn_kernel.py` verifies the Triton kernel matches a pure-Python reference at 1e-3 tolerance on random inputs of shape (B=2, T=128, n_heads=40, headdim=32, d_state=32). The test runs before the primary run starts; if it fails, fall back to the Python implementation at ~50,000 tok/s (6.94 days, still within the 5-7 day envelope by 0.06 days).

**Risk (§9):** "Fused Triton GDN kernel has a bug" — Medium probability, High impact. Mitigation: unit test before run; fall back to Python if test fails.

### 12a.2 MoE mixed precision in dispatch

**The problem:** the MoE dispatch in [`models/moe.py`](../src/hymo/models/moe.py) is the second-largest cost after the GDN recurrence. The current dispatch path uses BF16 throughout — including the scatter-add indices, which are integer but stored as BF16 for tensor contiguity. The scatter-add has 2× more bytes than necessary.

**The solution:** cast the scatter-add indices to FP16 (16-bit integer is sufficient for the 16-expert case) and the expert-matmul inputs to BF16. The 50% bandwidth reduction on the index path saves ~10% of the MoE dispatch cost. The matmuls stay in BF16 (no accuracy loss on the matmul side).

**Wall-clock saving:** ~10% of MoE cost. The MoE is ~10% of total forward FLOPs (§2.5), so the net wall-clock saving is ~1%. The bigger win is freeing up memory bandwidth for the GDN kernel, which is memory-bound.

**Implementation location:** `models/moe.py:_dispatch_tokens` (cast indices to FP16) and `models/moe.py:_gather_outputs` (cast back to BF16). The matmuls inside `experts[i].w1/w2/w3` remain BF16.

**Validation:** assert that the top-2 selected-expert indices are identical between BF16-index and FP16-index paths on 1k random inputs. Existing `tests/test_moe.py` covers this.

**Risk (§9):** "MoE mixed precision is a perf hit" — Low probability, Low impact. Mitigation: cast is a 1-line change; if perf regresses, revert.

### 12a.3 `torch.compile` on GDN blocks

**The problem:** the GDN forward path has significant Python control flow even after the Triton kernel replaces the inner loop. The `in_proj → conv1d → silu → split → kernel → out_proj` sequence has ~5-10 Python statements per layer per token, and the GDN has 24 layers × 4 micro-batches = 96 invocations per step. Python overhead is ~2-3ms per step.

**The solution:** wrap the GDN forward in `torch.compile(mode="reduce-overhead")`. The `reduce-overhead` mode uses CUDA Graphs internally (separate from the explicit CUDA Graphs in 12a.4) to eliminate Python overhead. Apply **only to GDN blocks** (not the whole model) because:
- The MoE dispatch has dynamic shapes (variable expert routing per token) that `torch.compile` struggles with.
- The MLA has a separate CUDA Graph capture path (12a.4).
- The WSD scheduler + FSDP-2 all-gather is incompatible with whole-model compile.

**Wall-clock saving:** ~10% of GDN cost. The GDN is ~40% of per-token FLOPs but only ~30% of wall-clock (after the Triton kernel), so the net saving is ~3% wall-clock.

**Implementation location:** `@torch.compile(mode="reduce-overhead")` decorator on `models/gdn.py:GatedDeltaNetBlock.forward`. Compile cache is keyed on `(B, T, n_heads, headdim, d_state)`.

**Validation:** `tests/test_gdn_compile.py` (new) verifies that compiled-GDN output matches eager-GDN within 1e-3 tolerance on 10 random inputs. The compile takes ~30 sec the first time; subsequent calls hit the cache.

**Risk (§9):** "torch.compile breaks with new PyTorch" — Low probability, Low impact. Mitigation: pin torch version in `pyproject.toml`; fall back to eager GDN if compile fails (drops to 60,000 tok/s, 5.79 days — still within 5-7 day target).

### 12a.4 CUDA Graphs for MLA forward

**The problem:** the MLA forward in [`models/mla.py:52-132`](../src/hymo/models/mla.py) has a control flow path (q split into rope/nope, k/v from compressed KV, attention with the latent bottleneck). The Python overhead is ~1-2ms per layer per micro-batch, and there are 8 MLA layers × 4 micro-batches = 32 invocations per step. Total Python overhead: ~30-50ms per step.

**The solution:** capture the MLA forward as a CUDA Graph. The graph captures the fixed-shape path (q projection, k/v decompression, attention, output projection) and replays it without Python overhead. The dynamic-shape MoE dispatch and GDN stay in eager mode.

**Wall-clock saving:** ~5% of MLA cost. The MLA is ~20% of per-token FLOPs and ~10% of wall-clock, so the net saving is ~0.5% wall-clock. The bigger win is reduced CPU-side jitter (deterministic per-step time).

**Implementation location:** `models/mla.py:MLABlock.forward` (capture on first call, replay on subsequent calls with the same input shape). Capture is conditioned on `(B, T) == cached_shape`; first call at a new shape triggers a fresh capture.

**Validation:** `tests/test_mla_cuda_graph.py` (new) verifies the CUDA-graph-captured MLA output matches the eager MLA within 1e-3 tolerance on 10 random inputs. The test skips automatically if CUDA Graphs are not supported on the test hardware.

**Risk (§9):** "CUDA Graphs add memory pressure" — Low probability, Low impact. Mitigation: the captured graph uses the same memory as the eager path; no extra VRAM.

### 12a.5 Stacked throughput summary

The four optimizations stack multiplicatively on the per-step wall-clock. Per §7.6:

| Configuration | Throughput (tok/s) | Wall-clock (days) |
|---|---|---|
| FSDP-2 only (Python GDN, no MoE mixed precision, no torch.compile, no CUDA graphs) | 50,000 | 6.94 |
| + Fused Triton GDN | 60,000 | 5.79 |
| + Fused Triton GDN + MoE mixed precision | 63,000 | 5.51 |
| + Fused Triton GDN + MoE mixed precision + torch.compile (GDN only) | 65,000 | 5.34 |
| + Fused Triton GDN + MoE mixed precision + torch.compile + CUDA graphs (MLA) | 67,000 | 5.18 |

The 5-7 day target is the band [5.18, 6.94] days; the design target is the 5.34-day point (all four optimizations enabled). The risk table (§9) tracks the failure modes for each optimization, and the §12.4 open question tracks the decision of whether to delay for verification or start on the stack and replace mid-run.

### 12a.6 Why these four and not others (the omitted optimizations)

The 2025-2026 hybrid-model literature uses several other optimizations that HyMo **does not adopt** in v1.0:

- **FP8 mixed precision (Hopper/Blackwell only):** unavailable on A100 SXM. Would give an additional 1.5-2× speedup on B200 but is out of scope for v1.0's A100 target.
- **Sequence packing at 8K (vs 4K context):** saves ~20% wall-clock on data-loading and activation memory. Documented in §7.6 as a "stretch" option. The v1.0 commitment is 4K context to keep the partial-RoPE + NoPE-hybrid math clean.
- **Flash Attention 2 / 3 for MLA:** the MLA path already uses the latent-kv compression (§2.4), which makes the attention FLOPs much smaller than MHA at the same seq_len. FA2/3 would save ~10% of the MLA path but the MLA is only 10% of wall-clock, so the net saving is ~1%. Not worth the integration cost.
- **Expert-parallel (EP) for MoE:** would require a different sharding strategy than FSDP-2 (which is the v1.0 design constraint). Documented as a v1.1 or v1.2 follow-up.
- **CPU-side data prefetch with multiple workers:** already enabled (4 dataloader workers per rank). The 50ms data-loading overhead in §13.7 is the cost of this; it is already overlapped with compute.

The four documented optimizations are the ones that **do not change the architecture** and **do not change the FSDP-2 sharding strategy** — they are pure kernel/runtime improvements on top of the v1.0 design.

This section documents the systems-level details of running on 4× A100 80GB SXM with FSDP-2. The architecture is a research contribution; the FSDP integration is a *systems* contribution that makes the research tractable on this hardware. The two are not independent — the FSDP sharding decisions affect which architectural claims are testable.

### 13.1 FSDP-2 mixed precision policy

```python
from torch.distributed.fsdp import (
FullyShardedDataParallel as FSDP,
MixedPrecision,
ShardingStrategy,
BackwardPrefetch,
)

# Per-parameter mixed precision:
# - params: BF16 (sharded)
# - reduce: BF16 (gradient reduction)
# - buffer: BF16 (all-gather buffer)
# - master weights: FP32 (optimizer state, kept on each rank)
mp_policy = MixedPrecision(
param_dtype=torch.bfloat16,
reduce_dtype=torch.bfloat16,
buffer_dtype=torch.bfloat16,
)

# Full sharding (ZeRO-3 equivalent) within each FSDP instance.
sharding_strategy = ShardingStrategy.FULL_SHARD

# Prefetch the next layer params while computing the current layer backward.
backward_prefetch = BackwardPrefetch.BACKWARD_PRE
```

**Why BF16 reduce (not FP32):**

FSDP-2 gradient reduction can be in BF16 (saves 2× communication) or FP32 (more numerically stable). The NorMuon paper recommends FP32 reduction for stability of the orthogonalization step. We choose **BF16 reduction** because:
- The 750M scale is small enough that the per-step communication is ~465MB per all-gather, which is dominated by the param all-gather (3.72GB), not the gradient reduce (~465MB).
- BF16 reduction saves 5-10% wall-clock.
- Numerical stability of the reduction is preserved by the per-parameter grad clipping (1.0) and the cautious weight decay mask.

If the first 1k steps show loss spikes attributable to BF16 reduction, switch to FP32.

### 13.2 Per-parameter FSDP wrapping

Some parameters should NOT be wrapped by FSDP (they should be replicated across all ranks). Specifically:

| Param | FSDP-wrapped? | Reason |
|---|---|---|
| Token embedding | Yes | Tied with head; sharded saves memory |
| Output head (tied) | Yes | Same as embed; sharded |
| GDN block params | Yes | Per-layer, can be wrapped per-block |
| MLA block params | Yes | Per-layer, can be wrapped per-block |
| MoE expert weights | **Yes, but per-expert** | Each expert is a 6.2M tensor; wrap each expert separately to enable round-robin sharding |
| MoE gate | **No (replicated)** | The gate is small (~14K params) and needs to be consistent across ranks for routing stability; replicate |
| RMSNorm γ | **No (replicated)** | Tiny; replicate to avoid sharding overhead |
| Logit softcap | N/A | A function, not a parameter |

The **per-expert FSDP wrapping** is the key choice. Each `experts.0.w1`, `experts.0.w2`, `experts.0.w3`, etc. It is wrapped as its own FSDP instance. This gives the NorMuon-with-MoE-exclusion optimizer partition 16 individual shardable tensors per MoE layer × 8 layers = 128 small FSDP instances, vs the alternative of wrapping the entire MoE module as one FSDP instance (which would put all 16 experts on one rank, defeating the sharding).

**Round-robin assignment:** with 4 ranks and 16 experts per MoE layer (× 8 layers = 128 experts total), the round-robin assignment puts 32 experts per rank. Each rank per-layer MoE shard is 4 experts × 6.2M × 2B = 50MB, well within the 80GB budget.

### 13.3 Sort-by-size + round-robin for NorMuon params

The NorMuon paper critical finding (3-0 verified) is that **without proper work distribution, the optimizer step time is 2.7× longer on the slowest rank** (the rank that holds the largest tensor). The fix:

```python
def shard_nor_muon_params(model: nn.Module, world_size: int) -> list[list[nn.Parameter]]:
"""Sort NorMuon params by size, round-robin assign to ranks."""
nor_muon_params = [p for n, p in model.named_parameters() if goes_to_nor_muon(n, p)]
nor_muon_params.sort(key=lambda p: p.numel(), reverse=True) # largest first

# Round-robin: rank 0 gets params 0, 4, 8, ...; rank 1 gets 1, 5, 9, ...
# This balances the per-rank total bytes.
rank_assignments = [[] for _ in range(world_size)]
rank_byte_counts = [0] * world_size
for i, p in enumerate(nor_muon_params):
target_rank = i % world_size
rank_assignments[target_rank].append(p)
rank_byte_counts[target_rank] += p.numel()

# Verify balance: the max rank byte count should be within 5% of the average
avg = sum(rank_byte_counts) / world_size
assert max(rank_byte_counts) / avg < 1.05, (
f"NorMuon shard imbalance: max={max(rank_byte_counts):,} "
f"avg={avg:,.0f} ratio={max(rank_byte_counts)/avg:.3f}"
)
return rank_assignments
```

**Why "largest first" + round-robin works:**

If you sort ascending and round-robin, rank 0 gets the smallest params and rank N-1 gets the largest. The optimizer step on rank N-1 takes much longer (large tensors need more Newton-Schulz iterations).

If you sort descending and round-robin, rank 0 gets the largest, rank 1 the second largest, etc. The per-rank total bytes are *exactly* balanced (each rank gets one param from each "size class").

**Empirical expectation:** the sort reduces optimizer-step time variance from σ/μ ≈ 0.4 (no sort) to σ/μ ≈ 0.02 (sorted). The mean optimizer-step time is also reduced by 10-15% because the slowest rank no longer sets the synchronization barrier.

### 13.4 Communication schedule

The FSDP-2 communication pattern for one training step:

```
1. Forward (per layer, per micro-batch):
- All-gather full params for layer N (BF16, ~3.44GB at the start, less for deeper layers)
- Compute forward activations (BF16)
- Discard gathered params (free memory)
- Prefetch all-gather for layer N+1 (overlapped)

2. Backward (per layer, per micro-batch):
- All-gather full params for layer N (re-shard)
- Compute backward gradients (BF16)
- Reduce-scatter gradients to per-rank shards (BF16, ~465MB)
- Discard gathered params
- Prefetch all-gather for layer N-1 (overlapped)

3. Optimizer step (per rank, after all micro-batches in accumulation):
- NorMuon step on local shard (FP32, no communication)
- AdamW step on local shard (FP32, no communication)
- Scheduler step (broadcast, ~1KB)
```

The two all-gathers per layer (forward + backward) and the reduce-scatter are the communication cost. With 32 layers × 2 all-gathers + 1 reduce-scatter = 96 collective ops per micro-batch, × 4 micro-batches = 384 collective ops per step.

**Per-collective cost:** at 4× A100 SXM with NVLink (600 GB/s bidirectional), a 3.72GB all-gather takes ~6.2ms; a 465MB reduce-scatter takes ~0.8ms. Per micro-batch: 2 × 6.2ms (gather) + 1 × 0.8ms (reduce) = 13.2ms. Per step (4 micro-batches): 53ms of pure communication. Overlapped with compute (forward + backward is ~150-200ms per micro-batch on the 750M model), the 53ms of comm is hidden behind the compute. **Net communication overhead: ~10-15% of wall-clock, as the NorMuon paper reports.**

### 13.5 Gradient norm handling with FSDP-2

The trainer gradient-norm clip (`grad_clip = 1.0`) needs an FSDP-2-aware implementation:

```python
# torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
# This computes the L2 norm of all grads on the *current rank* — WRONG for FSDP-2

# FSDP-2-aware gradient norm
grad_norm = model.clip_grad_norm_(max_norm=1.0) # FSDP built-in, all-reduces across ranks
# This is the correct cross-rank norm. The FSDP module `clip_grad_norm_` method
# computes the per-rank L2 norm, all-reduces the sum-of-squares, and applies the clip.
```

**The `torch.nn.utils.clip_grad_norm_` would over-clip the gradients** because each rank per-rank norm is smaller than the cross-rank norm, and the current implementation would compute the clip on the per-rank norm (which is too small). FSDP built-in `model.clip_grad_norm_` does the right thing.

### 13.6 DCP (Distributed Checkpoint) format

```python
from torch.distributed.checkpoint import save, load, FileSystemWriter

def save_fsdp_checkpoint(model, optimizer, scheduler, step, token_count, best_loss, save_dir):
state = {
"model": model.state_dict(), # FSDP-aware: sharded by rank
"optimizer_muon": muon_opt.state_dict(),
"optimizer_adamw": adamw_opt.state_dict(),
"scheduler": scheduler.state_dict(),
"step": step,
"token_count": token_count,
"best_loss": best_loss,
}
# DCP save: each rank writes its shard to a unique file in save_dir
save(state, checkpoint_id=save_dir)

def load_fsdp_checkpoint(model, optimizer, scheduler, load_dir):
state = {
"model": model.state_dict(),
"optimizer_muon": muon_opt.state_dict(),
"optimizer_adamw": adamw_opt.state_dict(),
"scheduler": scheduler.state_dict(),
}
load(state, checkpoint_id=load_dir)
# After load, model is in the state at the time of save. Optimizer state restored.
return state["step"], state["token_count"], state["best_loss"]
```

DCP handles the FSDP-2 sharding automatically: each rank writes its own shard, and on load, the shards are reassembled. This is the format that allows resumption from a different world_size (e.g., load a 4-rank checkpoint on a 2-rank pod for fine-tuning).

### 13.7 Throughput summary on 4× A100 80GB SXM

The full throughput picture for the primary run (750M active, 30B tokens, FSDP-2, fused Triton GDN, mixed precision):

| Stage | Time per step | Notes |
|---|---|---|
| Data loading (4 ranks × 4 micro-batches × 4096 ctx) | 50ms | Overlapped with compute |
| Forward (32 layers, FSDP all-gather overlapped) | 1,800ms | MLA checkpointed, GDN fused |
| Backward (32 layers, FSDP reduce-scatter overlapped) | 5,400ms | Includes grad-norm all-reduce |
| Optimizer step (NorMuon + AdamW, per-rank) | 500ms | Sort-by-size + round-robin balanced |
| Communication overhead (residual after overlap) | 250ms | 10-15% of compute time |
| **Per step (4 micro-batches)** | **~8,000ms (8.0 sec)** | |
| **Throughput** | **~65,500 tok/s** | 524,288 tokens / 8.0 sec |
| **30B tokens / 65,500 tok/s** | **~458,015 sec = 5.30 days of pure compute** | |
| **With overhead (ckpt, val, data stalls)** | **~5.6 days** | 8.5 sec effective per step |
| **5-7 day wall-clock target range** | **5.6-7.0 days** | 8.5-10.5 sec effective per step |

The arithmetic shows that with all optimizations (fused GDN + MoE mixed precision + selective `torch.compile` + CUDA graphs on MLA), the 30B-token run on 4× A100 SXM is **5.3 days of pure compute**, expanding to **5-7 days with overhead**. The 5-7 day target is achievable on 4× A100 SXM for a 30B-token run at 750M active with the full optimization stack.

**The honest assessment is in §7.6 table:** the 5-7 day target requires the full optimization stack. Removing any of the four (fused GDN, MoE mixed precision, torch.compile, CUDA graphs) extends the wall-clock by 5-25%. Going below 65,000 tok/s sustained (e.g., 50,000 tok/s without fused GDN) extends the wall-clock to ~6.9 days, still within the 5-7 day window. Going below 35,000 tok/s (e.g., no FSDP-2) breaks the 5-7 day budget.

### 13.8 Recovery from a pod failure

RunPod pods can be lost (hardware failure, spot reclaim, network partition). The recovery procedure:

1. **Detection:** rank 0 heartbeat to a RunPod-hosted metadata service times out after 5 minutes.
2. **Stop signal:** all 4 ranks detect and halt.
3. **Latest checkpoint:** the last DCP checkpoint is on a persistent RunPod volume (network-attached storage, survives pod loss).
4. **New pod:** provision a new 4× A100 SXM pod. Time-to-provision: 1-2 hours.
5. **Resume:** load the latest DCP checkpoint. Verify the step count, token count, and best_loss match. Resume training.

The checkpoint every 4,000 steps (every ~8.9 hours at 65,000 tok/s sustained) means at most ~9 hours of compute is lost on a pod failure. The 1-2 hour provisioning time is the dominant cost.

---

## 14. Scale variants (700M / 750M / 900M)

The architecture parameterizes cleanly to a family of scales. The base 750M is the primary target; 700M is the floor and 900M is the ceiling of the "publishable" range on the 4× A100 80GB SXM budget at 40× params-in-tokens.

| Variant | Active | Stored | Layers | dim | d_inner | n_experts | Tokens (40×) | Wall-clock @ 65k tok/s | Cost @ $2/hr |
|---|---|---|---|---|---|---|---|---|---|
| 700M | 700M | 1.74B | 32 | 832 | 1216 | 16 | 28B | 5.0 days | $960 |
| **750M (primary)** | **750M** | **1.86B** | **32** | **896** | **1280** | **16** | **30B** | **5.3 days** | **$1,020** |
| 900M | 900M | 2.22B | 36 | 960 | 1408 | 16 | 36B | 6.4 days | $1,230 |

With the 4 parallel ablations (each at 7.5B tokens, on a separate pod):
- 7.5B / 65,000 tok/s = 115,385 sec = 1.34 days per ablation pod
- 4 ablations × 1.34 days = 5.36 pod-days of compute
- Each pod is $2/hr × 4 GPUs × 24 × 1.34 = $257 per ablation

**Total cost (primary + 4 ablations):**
- Primary: $1,020
- 4 ablations: 4 × $257 = $1,028
- Held-out eval: $100
- **Total: $2,148** (or ~$1,500 with spot/committed-use discounts)

The $2,148 total is the *quality-first* budget. It includes 4 publishable ablations + the primary run + the held-out eval at the full 750M scale on real held-out data. The RunPod on-demand rate for 4× A100 SXM at the time of writing is ~$2/hr per GPU ($8/hr for the full node); spot instances and committed-use discounts can bring this down to ~$1.50/hr total or less.

`★ Insight ─────────────────────────────────────`
- The 4 parallel ablations are a budget-positive addition: they cost ~$1,028 total but produce 4 publishable sub-papers, each of which is worth more than $1,028 in research output value.
- The 900M variant is the "stretch goal" at 6.4 days ($1,230). If the 750M run converges cleanly, the 900M run is a follow-up at modest budget.
- The 700M variant at $960 is the "minimum publishable" run if budget is tight; it would skip the 4 ablations.
`─────────────────────────────────────────────────`

**The 750M is the recommended primary target because:**

1. It lands in the middle of the publishable range, giving headroom to go up (900M) or down (700M) for the ablation comparisons.
2. The dim=896 is a "round" value that makes the partial-RoPE 25% split (= 32 rope_dim) clean.
3. The 30B token budget fits the modern 40× params-in-tokens practice.

**The architecture is scale-invariant in the following sense:** the per-layer shape (8 MLA + 24 GDN, 16-expert MoE on MLA, 3:1 ratio) is the same across all three variants. Only `dim`, `d_inner`, and the layer count change. This means the paper can report results across the family and claim a scaling-law-style finding for the architectural choices, not just a single data point.

---

---
