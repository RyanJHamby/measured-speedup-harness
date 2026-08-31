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

from harness import compare, render_lesson

N = 200_000
WINDOW = 50

random.seed(0)
DATA = np.array([random.random() for _ in range(N)])


def baseline() -> np.ndarray:
    """Naive O(N*WINDOW) sliding-sum implementation."""
    out = np.empty(N - WINDOW + 1)
    for i in range(len(out)):
        out[i] = sum(DATA[i : i + WINDOW]) / WINDOW
    return out


def optimized() -> np.ndarray:
    """O(N) cumulative-sum implementation of the same sliding average."""
    cumsum = np.cumsum(np.insert(DATA, 0, 0.0))
    return (cumsum[WINDOW:] - cumsum[:-WINDOW]) / WINDOW


def check_equivalent(a: np.ndarray, b: np.ndarray) -> tuple[bool, str]:
    ok = np.allclose(a, b, rtol=1e-9, atol=1e-9)
    max_diff = float(np.max(np.abs(a - b)))
    return ok, f"max abs diff = {max_diff:.2e}, rtol=1e-9"


if __name__ == "__main__":
    # N is intentionally small here so the naive baseline finishes in a
    # reasonable time under 30 trials; shrink further if this is slow on
    # your machine.
    small_n = 4_000
    DATA = DATA[:small_n]
    N = small_n

    result = compare(
        baseline_fn=baseline,
        optimized_fn=optimized,
        check_equivalent=check_equivalent,
        n_trials=25,
        warmup=3,
        min_speedup_pct=5.0,
        t_threshold=2.0,
        label="moving average: naive loop vs cumsum",
    )

    lesson = render_lesson(
        result,
        technique="Replace naive O(N*W) sliding-sum loop with O(N) cumsum-based moving average",
        target=f"moving average, N={N}, window={WINDOW}",
        source="example_moving_average.py, run locally, no external deps beyond numpy",
    )

    print(lesson)
    with open("lessons.md", "a") as f:
        f.write(lesson + "\n\n")
