"""Optimizer partition: NorMuon vs AdamW parameter groupings."""

from __future__ import annotations

from torch import nn

__all__ = [
    "goes_to_adamw",
    "partition_parameters",
    "ParameterPartition",
]


def goes_to_adamw(name: str, param: nn.Parameter) -> bool:
    """Classify a parameter for AdamW; dense 2-D leftovers go to NorMuon.

    Embeddings, gates, expert weights, norms, and learned scalar controls stay
    on AdamW because their geometry is not suited to matrix orthogonalization.
    """
    if param.ndim < 2:
        return True
    if param.ndim > 2:
        return True
    if name.endswith("embed.weight") or name.endswith("head.weight"):
        return True
    if name.endswith(".gate.weight") or name.endswith(".gate.bias"):
        return True

    if ".experts." in name and (
        name.endswith(".w1.weight")
        or name.endswith(".w2.weight")
        or name.endswith(".w3.weight")
    ):
        return True
    if (
        name.endswith(".shared_expert.w1.weight")
        or name.endswith(".shared_expert.w2.weight")
        or name.endswith(".shared_expert.w3.weight")
    ):
        return True

    if name.endswith(".A_log") or name.endswith(".dt_bias") or name.endswith(".D"):
        return True

    return name.endswith("norm.weight")


class ParameterPartition:
    """Disjoint parameter lists consumed by the two optimizer implementations."""

    __slots__ = ("adamw", "nor_muon")

    def __init__(self) -> None:
        self.adamw: list[nn.Parameter] = []
        self.nor_muon: list[nn.Parameter] = []

    def __len__(self) -> int:
        return len(self.adamw) + len(self.nor_muon)

    def __repr__(self) -> str:
        return (
            f"ParameterPartition(adamw={len(self.adamw)}, "
            f"nor_muon={len(self.nor_muon)})"
        )


def partition_parameters(model: nn.Module) -> ParameterPartition:
    """Deduplicate tied parameters and assign each parameter to one optimizer."""
    p = ParameterPartition()
    seen: set[int] = set()
    for name, param in model.named_parameters():
        if id(param) in seen:
            continue
        seen.add(id(param))
        if goes_to_adamw(name, param):
            p.adamw.append(param)
        elif param.ndim >= 2:
            p.nor_muon.append(param)
    return p
