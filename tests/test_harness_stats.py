"""
Sanity checks for the statistics in harness.py, independent of any of the
worked examples: synthetic timing data with a known, engineered speedup so
the interval and tier logic can be checked against ground truth rather than
real (noisy) wall-clock timings.
"""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness import TrialStats, _bootstrap_speedup_ci, _welch_t, compare


def test_bootstrap_ci_contains_true_speedup_for_synthetic_data() -> None:
    rng = random.Random(42)
    true_baseline_mean = 10.0
    true_optimized_mean = 6.0  # true speedup = 40%
    baseline = TrialStats([true_baseline_mean + rng.gauss(0, 0.3) for _ in range(50)])
    optimized = TrialStats([true_optimized_mean + rng.gauss(0, 0.3) for _ in range(50)])

    lo, hi = _bootstrap_speedup_ci(
        baseline, optimized, n_resamples=2000, confidence=0.95, rng=random.Random(1)
    )
    assert lo < 40.0 < hi


def test_welch_t_is_near_zero_for_identical_distributions() -> None:
    rng = random.Random(7)
    a = TrialStats([5.0 + rng.gauss(0, 1.0) for _ in range(200)])
    b = TrialStats([5.0 + rng.gauss(0, 1.0) for _ in range(200)])
    assert abs(_welch_t(a, b)) < 2.0


def test_compare_marks_no_real_difference_as_noise_or_marginal() -> None:
    """Two functions with identical expected cost shouldn't be called `confirmed`."""

    def same_cost() -> int:
        return sum(range(2000))

    result = compare(
        baseline_fn=same_cost,
        optimized_fn=same_cost,
        check_equivalent=lambda a, b: (a == b, "identical by construction"),
        n_trials=30,
        warmup=5,
    )
    assert result.tier in ("noise", "marginal")
