"""
Visualize the trial distributions behind a ComparisonResult.

A finding's numbers (mean, stdev, CI) are the rigorous version of the
argument; this plot is the fast version, useful for a reviewer or a
non-technical stakeholder who wants to see "yes, these two clouds of points
don't overlap" in about two seconds rather than parse a decision rule.

Requires matplotlib (see requirements-dev.txt); not a dependency of
harness.py itself.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless: write files, don't require a display
import matplotlib.pyplot as plt

from harness import ComparisonResult


def plot_trial_distributions(result: ComparisonResult, title: str, out_path: str) -> None:
    baseline_ms = [s * 1e3 for s in result.baseline.samples]
    optimized_ms = [s * 1e3 for s in result.optimized.samples]

    fig, ax = plt.subplots(figsize=(7, 4))
    bins = 20
    ax.hist(baseline_ms, bins=bins, alpha=0.6, label=f"baseline (n={result.baseline.n})")
    ax.hist(optimized_ms, bins=bins, alpha=0.6, label=f"optimized (n={result.optimized.n})")
    ax.set_xlabel("time per call (ms)")
    ax.set_ylabel("trial count")
    ax.set_title(f"{title}\nconfidence: {result.tier}, speedup: {result.speedup_pct:.1f}%")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
