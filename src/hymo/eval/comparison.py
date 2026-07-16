"""Markdown comparison table (architecture doc §15).

Produces a markdown table from a single HyMo result dict, with the
3 baseline columns.
"""

from __future__ import annotations

from hymo.eval.baselines import BASELINE_DISPLAY_NAMES, BASELINES

__all__ = ["format_comparison_table", "METRIC_ORDER"]


# The fixed metric order for the comparison table.
METRIC_ORDER: tuple[str, ...] = (
    "fineweb_edu_ppl",
    "hellaswag",
    "arc_challenge",
    "mmlu",
    "gsm8k",
    "humaneval",
)


def format_comparison_table(
    hymo_results: dict[str, float],
    *,
    baselines: tuple[str, ...] = ("pythia-1b", "mobile_moe_0.9b", "smollm2_1.7b"),
) -> str:
    """Format a markdown comparison table from HyMo + baseline numbers.

    Parameters
    ----------
    hymo_results : dict[str, float]
        The 6 HyMo metrics, keyed by metric name.
    baselines : tuple[str, ...]
        Which baselines to include (default: all three).

    Returns
    -------
    str
        A markdown table.
    """
    cols = ["Metric", "HyMo", *[BASELINE_DISPLAY_NAMES.get(b, b) for b in baselines]]
    lines = [
        "| " + " | ".join(cols) + " |",
        "|" + "|".join(["-" * 4] * (len(cols) + 1)) + "|",
    ]
    for metric in METRIC_ORDER:
        row = [metric, f"{hymo_results.get(metric, float('nan')):.3f}"]
        for b in baselines:
            v = BASELINES.get(b, {}).get(metric, float("nan"))
            row.append(f"{v:.3f}")
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)
