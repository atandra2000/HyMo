"""Optimizer partition: NorMuon vs AdamW (the partition function).

Architecture doc §5.1. The partition is the *publishable claim 2*:
- NorMuon goes to: MLA attention weights (``wq_a, wq_b, wkv_a, wkv_b,
  wo``), GDN matrices (``in_proj, b_proj, c_proj, dt_proj, g_proj,
  out_proj``), and DenseFFN weights (``w1, w2, w3`` on GDN blocks).
- AdamW goes to: embedding, output head (tied), RMSNorm γ, MoE gate,
  MoE expert weights (``experts.*.w1/w2/w3``), shared expert, GDN
  scalars (``A_log, dt_bias, D``).

The function :func:`goes_to_adamw` is the canonical predicate. It is
called by:

- :func:`hymo.training.optimizer.build_optimizers` (Phase 3) to
  partition the parameters.
- Tests in :mod:`tests.unit.test_optimizer_partition` to verify the
  partition.
- The FSDP-aware checkpoint saver in :mod:`hymo.training.checkpoint`
  (Phase 3) to correctly restore optimizer state.
"""

from __future__ import annotations

from torch import nn

__all__ = [
    "goes_to_adamw",
    "goes_to_nor_muon",
    "partition_parameters",
    "ParameterPartition",
]


# Parameter-name substrings that go to AdamW.
# These are matched as substrings against ``state_dict`` keys.
_ADAMW_NAME_PATTERNS: tuple[str, ...] = (
    "embed.weight",
    "head.weight",
    "norm.weight",
    ".gate.weight",
    ".gate.bias",
    ".A_log",
    ".dt_bias",
    # GDN scalar "D" — matched as ".D" with word boundary, see predicate.
)

# Parameter-name patterns that go to NorMuon.
# NorMuon gets all 2D weights that aren't in the AdamW list.
_NORMUON_2D_NAME_PATTERNS: tuple[str, ...] = (
    ".in_proj.",
    ".b_proj.",
    ".c_proj.",
    ".dt_proj.",
    ".g_proj.",
    ".out_proj.",
    ".wq_a.",
    ".wq_b.",
    ".wkv_a.",
    ".wkv_b.",
    ".wo.",
    # DenseFFN on GDN blocks.
    "layers.",
)


def goes_to_adamw(name: str, param: nn.Parameter) -> bool:
    """Return True iff the parameter ``name`` should be on AdamW.

    The predicate is intentionally exact: a parameter is on AdamW iff
    its name matches one of the :data:`_ADAMW_NAME_PATTERNS` substrings
    *and* the parameter is non-2D OR it's a MoE expert weight OR it's
    the gate.

    The 1D / 2D rule is:
    - 1D (norm γ, biases) → AdamW.
    - 2D MoE expert weights (``experts.{i}.w{1,2,3}`` and
      ``shared_expert.w{1,2,3}``) → AdamW (claim 2).
    - 2D gate (16 × dim) → AdamW (1D-ish, also matches ``.gate.weight``).
    - 2D embed / head → AdamW (tied, sparse updates, large).
    - All other 2D → NorMuon.
    """
    # 1D → AdamW (norm γ, biases, scalars).
    if param.ndim < 2:
        return True

    # Tied embed / head.
    if name.endswith("embed.weight") or name.endswith("head.weight"):
        return True

    # MoE gate (2D but treated as 1D-ish).
    if name.endswith(".gate.weight") or name.endswith(".gate.bias"):
        return True

    # MoE expert weights — the novel claim (publishable claim 2).
    # PyTorch's ``nn.Linear`` registers its weight as ``<proj>.weight``,
    # so the full parameter name is ``...experts.<i>.<proj>.weight``.
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

    # GDN scalars.
    if name.endswith(".A_log") or name.endswith(".dt_bias") or name.endswith(".D"):
        return True

    # RMSNorm γ (1D, but check by name in case).
    return name.endswith("norm.weight")


def goes_to_nor_muon(name: str, param: nn.Parameter) -> bool:
    """Return True iff the parameter should be on NorMuon.

    NorMuon gets all 2D weights that aren't AdamW — i.e. attention
    weights, GDN matrices, and DenseFFN weights.
    """
    return not goes_to_adamw(name, param) and param.ndim >= 2


# ----------------------------------------------------------------------
# Partition container
# ----------------------------------------------------------------------


class ParameterPartition:
    """The result of partitioning a model's parameters.

    Attributes
    ----------
    adamw : list[nn.Parameter]
        Parameters that go on AdamW.
    nor_muon : list[nn.Parameter]
        Parameters that go on NorMuon.
    """

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
    """Partition the model's parameters into AdamW and NorMuon groups.

    Iterates ``model.named_parameters()`` and applies
    :func:`goes_to_adamw` to each. Returns a :class:`ParameterPartition`.
    """
    p = ParameterPartition()
    seen: set[int] = set()
    for name, param in model.named_parameters():
        if id(param) in seen:
            # Aliased parameter (e.g. tied head) — skip the duplicate.
            continue
        seen.add(id(param))
        if goes_to_adamw(name, param):
            p.adamw.append(param)
        elif param.ndim >= 2:
            p.nor_muon.append(param)
    return p
