"""
Worked example for harness.py: naive Python-loop moving average vs a
vectorized (cumulative-sum) implementation.

This is a deliberately unglamorous, universally recognizable optimization
(nothing hardware- or framework-specific) chosen so the harness itself -
the interleaved timing, the correctness gate, the decision rule - is the
thing on display, not the target op. Swap `baseline` / `optimized` below
for whatever real comparison you're running; the harness doesn't change.

Run directly:      python3 example_moving_average.py
Run via the CLI:    python3 bench.py example_moving_average
"""

import random

import numpy as np


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

    Claimed correct (rtol=1e-6) for input magnitudes roughly within
    +/-1e3, per the domain the property tests actually check - not for
    arbitrary magnitudes. The error scales with the ratio of the running
    sum's magnitude to the output's magnitude, so no fixed tolerance bounds
    it for unbounded inputs. For a production use case that can't guarantee
    that input range, use compensated (Kahan) summation or a periodically
    reset rolling sum instead of one running cumsum.
    """
    cumsum = np.cumsum(np.insert(data, 0, 0.0))
    return (cumsum[window:] - cumsum[:-window]) / window


def _check_equivalent(a: np.ndarray, b: np.ndarray) -> tuple[bool, str]:
    ok = np.allclose(a, b, rtol=1e-6, atol=1e-9)
    max_diff = float(np.max(np.abs(a - b)))
    return ok, f"max abs diff = {max_diff:.2e}, rtol=1e-6"


# Scenario definition consumed by bench.py. N is kept small so the naive
# O(N*window) loop finishes in a reasonable time across N trials; shrink
# further if this is slow on your machine.
random.seed(0)
_N = 4_000
_WINDOW = 50
_DATA = np.array([random.random() for _ in range(_N)])

BASELINE_FN = lambda: moving_average_naive(_DATA, _WINDOW)
OPTIMIZED_FN = lambda: moving_average_vectorized(_DATA, _WINDOW)
CHECK_EQUIVALENT = _check_equivalent
TECHNIQUE = "Replace naive O(N*W) sliding-sum loop with O(N) cumsum-based moving average"
TARGET = f"moving average, N={_N}, window={_WINDOW}"
SOURCE = "example_moving_average.py, run locally, no external deps beyond numpy"


if __name__ == "__main__":
    import sys

    from bench import run_scenario

    result, finding = run_scenario(sys.modules[__name__], n_trials=25, warmup=3)
    print(finding)
    with open("findings.md", "a") as f:
        f.write(finding + "\n\n")
