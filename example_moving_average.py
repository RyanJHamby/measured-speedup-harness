"""
Worked example for harness.py: naive Python-loop moving average vs a
vectorized (cumulative-sum) implementation.

This is a deliberately unglamorous, universally recognizable optimization
(nothing hardware- or framework-specific) chosen so the harness itself -
the interleaved timing, the correctness gate, the decision rule - is the
thing on display, not the target op. Swap `baseline` / `optimized` below
for whatever real comparison you're running; the harness doesn't change.

Run: python3 example_moving_average.py
"""

import random

import numpy as np

from harness import compare, render_finding


def moving_average_naive(data: np.ndarray, window: int) -> np.ndarray:
    """Naive O(N*window) sliding-sum implementation."""
    out = np.empty(len(data) - window + 1)
    for i in range(len(out)):
        out[i] = sum(data[i : i + window]) / window
    return out


def moving_average_vectorized(data: np.ndarray, window: int) -> np.ndarray:
    """O(N) cumulative-sum implementation of the same sliding average.

    Known tradeoff, found by the Hypothesis suite in tests/test_equivalence.py:
    on high-dynamic-range data (large-magnitude values alongside much smaller
    differences between them), subtracting two nearly-equal large partial
    sums loses precision to catastrophic cancellation. The naive loop doesn't
    have this failure mode because it never carries a large running total.
    rtol=1e-6 below reflects that real tradeoff rather than papering over it.
    """
    cumsum = np.cumsum(np.insert(data, 0, 0.0))
    return (cumsum[window:] - cumsum[:-window]) / window


def check_equivalent(a: np.ndarray, b: np.ndarray) -> tuple[bool, str]:
    ok = np.allclose(a, b, rtol=1e-6, atol=1e-9)
    max_diff = float(np.max(np.abs(a - b)))
    return ok, f"max abs diff = {max_diff:.2e}, rtol=1e-6"


if __name__ == "__main__":
    random.seed(0)
    N = 4_000
    WINDOW = 50
    data = np.array([random.random() for _ in range(N)])

    result = compare(
        baseline_fn=lambda: moving_average_naive(data, WINDOW),
        optimized_fn=lambda: moving_average_vectorized(data, WINDOW),
        check_equivalent=check_equivalent,
        n_trials=25,
        warmup=3,
        min_speedup_pct=5.0,
        t_threshold=2.0,
        label="moving average: naive loop vs cumsum",
    )

    finding = render_finding(
        result,
        technique="Replace naive O(N*W) sliding-sum loop with O(N) cumsum-based moving average",
        target=f"moving average, N={N}, window={WINDOW}",
        source="example_moving_average.py, run locally, no external deps beyond numpy",
    )

    print(finding)
    with open("findings.md", "a") as f:
        f.write(finding + "\n\n")
