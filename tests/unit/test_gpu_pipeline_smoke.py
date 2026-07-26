"""GPU pipeline smoke (tiny HyMo, default suite).

Mirrors ``Trainer.train_step`` (forward + MTP + dual optimizer) so on GPU
the assertions reflect the wired training path, not ``model.forward`` alone.

Skipped when CUDA is not available.

ponytail: MTP+forward loop mirrors trainer.py:102-194 inline rather than
constructing a Trainer (avoids wandb init + scheduler + DCP wiring). If
the trainer internals move, this test must mirror them.
"""

from __future__ import annotations

import pytest
import torch
from torch.nn import functional as F

from hymo.core.config import HyMoConfig
from hymo.models import HyMo
from hymo.training.optimizer import build_optimizers

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="GPU pipeline smoke requires CUDA"
)


def test_gpu_forward_backward_optimizer_step(tiny_hymo_config: HyMoConfig) -> None:
    """Tiny HyMo forward + MTP + backward + dual optimizer step on CUDA stays finite."""
    device = torch.device("cuda")
    model = HyMo(tiny_hymo_config.model).to(device=device, dtype=torch.bfloat16)
    model.train()

    B, T = 1, min(64, tiny_hymo_config.model.max_seq_len)
    tokens = torch.randint(0, tiny_hymo_config.model.vocab_size, (B, T), device=device)
    targets = torch.randint(0, tiny_hymo_config.model.vocab_size, (B, T), device=device)

    optimizers = build_optimizers(model, tiny_hymo_config.optimizer)

    if model._mtp is not None:
        logits, mtp_outputs = model._mtp.forward(tokens)
    else:
        logits = model(tokens)
        mtp_outputs = []

    V = logits.size(-1)
    assert logits.device.type == "cuda"
    assert torch.isfinite(logits).all()

    main_loss = F.cross_entropy(logits[:, :-1, :].reshape(-1, V), targets[:, :-1].reshape(-1))
    total_loss = main_loss + sum(
        mtp_out.loss_weight * F.cross_entropy(mtp_out.logits.reshape(-1, V), mtp_out.targets.reshape(-1))
        for mtp_out in mtp_outputs
    )
    assert torch.isfinite(total_loss)
    total_loss.backward()

    trainable = [p for p in model.parameters() if p.requires_grad]
    for p in trainable:
        assert p.grad is not None
        assert torch.isfinite(p.grad).all()

    if optimizers.nor_muon is not None:
        optimizers.nor_muon.step()
    optimizers.adamw.step()
