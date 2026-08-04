# SKILLS.md — HyMo

> Companion to `AGENTS.md`. Day-to-day workflows for the flagship 3:1 GDN/MLA
> hybrid. Anchor metric: **750M active / 1.86B stored params**.

---

## Skill 1: Run the cool-by-design test suite

```bash
cd LLM/HyMo
uv run pytest tests/ -v              # ~1 min CPU; heavy tests skipped
uv run pytest tests/ --run-heavy -v    # full 1.86B model construction (GPU pod)
uv run mypy src/hymo
uv run ruff check src/hymo
```

Default tests use the ~760K-param tiny config from `tests/conftest.py`.
Never build `configs/hymo_750m.yaml` in a non-`heavy` test.

## Skill 2: Verify the Triton GDN kernel against the PyTorch reference

The fused kernel lives in `src/hymo/models/gdn_triton.py`. Regression tests
compare it to the naive selective scan:

```bash
uv run pytest tests/unit/test_triton_gdn_gpu.py -v
```

On non-Linux or without Triton, kernel tests skip via `HAS_TRITON`. Production
training on Linux must use the Triton path — no silent fallback during a real run.

## Skill 3: Derive an ablation config

All knobs are frozen dataclasses in `hymo.core.config`. Derive variants with
`dataclasses.replace` or `derive_config()` — never mutate YAML in place:

```python
from hymo.core.config import load_config, derive_config

base = load_config("configs/hymo_750m.yaml")
ablation = derive_config(base, moe_routed_experts=8)  # example; see ablations/
```

Ablation families are documented under `src/hymo/ablations/`.

## Skill 4: Launch FSDP-2 pretraining

Primary config: `configs/hymo_750m.yaml`. Target: 30B tokens on 4× A100 80GB.

```bash
cd LLM/HyMo
# single-node smoke (tiny config):
uv run python -m hymo.training.train --config configs/hymo_750m.yaml --dry-run

# multi-GPU (production):
torchrun --nproc_per_node=4 -m hymo.training.train --config configs/hymo_750m.yaml
```

Checkpoints use DCP (distributed checkpoint) with full RNG + optimizer state.
Resume from an arbitrary step via `--resume path/to/step_N`.

## Skill 5: Wire the shared data pipeline

HyMo uses its own BPE-64k + 256-byte tokenizer (vocab 64,256). Corpus prep can
reuse `LLM/shared_data/` for download, dedup, and shard format — then tokenize
with HyMo's shim in `data/prepare_data.py`. See `LLM/shared_data/README.md`
for mixture weights and manifest schema.

After changing the mixture or tokenizer, rebuild shards and bump the manifest
version before starting a new run.

## Skill 6: Debug NaN / OOM during training

1. Reproduce on tiny config first (`pytest` surrogate).
2. Disable optimizations one at a time: `torch.compile` → FA2 → BF16 FSDP.
3. Check MoE router logits (aux-loss-free gate) and MTP auxiliary heads
   (λ=[0.3, 0.1]).
4. Enable NaN-step skipping in `TrainingConfig`; inspect W&B for the first
   bad step and roll back to the prior DCP checkpoint.

Cross-reference: `.agents/skills/llm-architecture/SKILL.md` (GDN, MLA, MoE, MTP)
and `DeepSeek-v3-Lite/MLA.md` for MLA absorption details reused in HyMo.
