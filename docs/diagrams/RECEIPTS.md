# Archify delivery evidence — HyMo

The [interactive visual guide](hymo_interactive_guide.html) links four standalone showcase architecture, dataflow, and workflow diagrams.

All four: **9/9 showcase checks, 0 composition errors, 0 warnings; automated browser evidence passed.**

Chrome checked at 1440×900, 1600×1000, 1920×1080 and 2048×1320 in light/dark. All required viewport measurements passed horizontal/vertical containment, minimum projected text size and viewer-control clearance.

## Artifact bindings

### 32-Layer Hybrid Model Architecture (3:1 GDN:MLA)

- Diagram type: `architecture`
- Output: [hymo-architecture.html](hymo-architecture.html)
- Specification: `docs/diagrams/hymo-architecture.architecture.json`
- Specification SHA-256: `9e6d1ed0a4f922dcd64602e750fce79be6fbfc43a8ce0016aa88bdd9f351385b` (5,341 bytes)
- Artifact SHA-256: `512126a52a089e2c3b5d53972281ea96d85cfc7b26eddd3a66699b8608f83e92` (719,174 bytes)
- [Browser receipt](hymo-architecture.visual-check.json) · [Screenshot contact sheet](hymo-architecture.visual-check.html)
- `browser_evidence: passed` · `visual_review: passed` · `correction_rounds: 0`

### 30B-Token Multi-Source Data Pipeline

- Diagram type: `dataflow`
- Output: [hymo-dataflow.html](hymo-dataflow.html)
- Specification: `docs/diagrams/hymo-dataflow.dataflow.json`
- Specification SHA-256: `772158a0131e23f52114c1de89e85f5a781cab755c3b40dc15ce5374d81c3d02` (4,750 bytes)
- Artifact SHA-256: `a348b407f79eb2d56c463f175c2247f9458d2d2f447b910455f6bd0257f6a006` (714,119 bytes)
- [Browser receipt](hymo-dataflow.visual-check.json) · [Screenshot contact sheet](hymo-dataflow.visual-check.html)
- `browser_evidence: passed` · `visual_review: passed` · `correction_rounds: 0`

### Pretraining Pipeline & Multi-Token Supervision

- Diagram type: `workflow`
- Output: [hymo-training.html](hymo-training.html)
- Specification: `docs/diagrams/hymo-training.workflow.json`
- Specification SHA-256: `8867fa8df7479323ed36e05d8c1ec36f4d5d3c63512b1d47b4aaea5a5415d59e` (4,877 bytes)
- Artifact SHA-256: `fc79c8669943a784276aa97f0306d94fc9d1023ce7626ee824c8aca07b33b388` (715,121 bytes)
- [Browser receipt](hymo-training.visual-check.json) · [Screenshot contact sheet](hymo-training.visual-check.html)
- `browser_evidence: passed` · `visual_review: passed` · `correction_rounds: 0`

### Systems, Kernels & Optimization Stack

- Diagram type: `architecture`
- Output: [hymo-optimizations.html](hymo-optimizations.html)
- Specification: `docs/diagrams/hymo-optimizations.architecture.json`
- Specification SHA-256: `71f61b92a895b3390850658932260ed9a3c25cfa263a55ee13fdf5079f90411e` (4,700 bytes)
- Artifact SHA-256: `c8fa7c5ba29170ffcc4a9216da3d866bd1c6fab414ffaa37bda4cbc35076abd3` (716,277 bytes)
- [Browser receipt](hymo-optimizations.visual-check.json) · [Screenshot contact sheet](hymo-optimizations.visual-check.html)
- `browser_evidence: passed` · `visual_review: passed` · `correction_rounds: 0`

## Verification limits

Parameter count: 434M active parameters per token / 1.13B total stored parameters across 32 hybrid layers. Implemented with custom Triton GDN linear recurrence kernels, Multi-Head Latent Attention (MLA), Asymmetric DeepSeekMoE (1 shared + 8 routed experts, Top-2 active), NorMuon matrix-valued momentum + AdamW optimizers, and FSDP-2 distributed scaling. Pretrained on 30B tokens targeting held-out FineWeb-Edu perplexity ≤ 2.10.
