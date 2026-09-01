"""
Worked example for harness.py: naive Python-loop moving average vs a
vectorized (cumulative-sum) implementation.

This is a deliberately unglamorous, universally recognizable optimization
(nothing hardware- or framework-specific) chosen so the harness itself -
the interleaved timing, the correctness gate, the decision rule - is the
thing on display, not the target op. Swap `baseline` / `optimized` below
for whatever real comparison you're running; the harness doesn't change.

Run directly:      python3 scenarios/moving_average.py
Run via the CLI:    python3 bench.py scenarios.moving_average
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


def moving_average_convolve(data: np.ndarray, window: int) -> np.ndarray:
    """np.convolve-based implementation of the same sliding average.

    Doesn't share the cumsum implementation's cancellation problem (no
    large running total is ever formed), at the cost of doing O(N*window)
    multiply-adds under the hood rather than cumsum's O(N) - convolution
    is the more general operation and doesn't get to exploit the sliding-
    window structure the way a running sum does. Whether that tradeoff is
    worth it is exactly the kind of question a leaderboard across several
    candidates answers better than eyeballing one comparison at a time -
    see leaderboard.py.
    """
    kernel = np.ones(window) / window
    return np.convolve(data, kernel, mode="valid")


def moving_average_kahan(data: np.ndarray, window: int) -> np.ndarray:
    """O(N) sliding-window moving average using Kahan (compensated) summation
    instead of a plain running cumsum.

    This is the fix `moving_average_vectorized`'s docstring promises but
    never built: a running sum that tracks a compensation term for the
    low-order bits lost on each addition/subtraction, so the running total
    doesn't accumulate error the way a plain cumsum does. Still one pass,
    still O(N) - the window slides by adding the incoming element and
    subtracting the outgoing one, both through the compensated-add step,
    rather than doing an O(window) sum per output position.

    Measured, not assumed (see tests/test_kahan_moving_average.py): on a
    long run of one repeated large value with a differing tail element
    (window=1, so the true answer is an exact copy of the input) at
    n=20000, magnitude=1e6, cumsum recovers the tail value wrong in its
    5th significant digit (relerr ~4.4e-6 - already outside the rtol=1e-6
    moving_average_vectorized is held to); Kahan recovers it wrong only in
    the 10th significant digit (relerr ~1.0e-10), roughly 44,000x more
    precise on this exact case. A genuine Hypothesis fuzzing campaign (not
    hand-picked) across random arrays and window sizes with elements up to
    +/-1e7 - three orders of magnitude past moving_average_vectorized's
    documented +/-1e3 domain limit - found zero cases where Kahan exceeded
    rtol=1e-9 in 300 examples. It is not immune, though: at magnitude 1e8,
    Kahan's error does exceed rtol=1e-9 (still ~100x smaller than cumsum's
    error on the same input). Kahan summation reduces accumulated rounding
    error, it does not make floating-point addition exact.
    """
    n = len(data)
    out = np.empty(n - window + 1)
    total = 0.0
    c = 0.0

    def kahan_add(total: float, c: float, x: float) -> tuple[float, float]:
        y = x - c
        t = total + y
        c = (t - total) - y
        return t, c

    for i in range(window):
        total, c = kahan_add(total, c, float(data[i]))
    out[0] = total / window

    for i in range(window, n):
        total, c = kahan_add(total, c, float(data[i]))
        total, c = kahan_add(total, c, -float(data[i - window]))
        out[i - window + 1] = total / window

    return out


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
SOURCE = "scenarios/moving_average.py, run locally, no external deps beyond numpy"


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from bench import run_scenario

    result, finding = run_scenario(sys.modules[__name__], n_trials=25, warmup=3)
    print(finding)
    with open("findings.md", "a") as f:
        f.write(finding + "\n\n")
