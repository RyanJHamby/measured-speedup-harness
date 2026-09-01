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

from harness import TrialStats, _bootstrap_speedup_ci, _percentile, _t_two_tailed_p_value, _welch_t, compare


def test_t_two_tailed_p_value_matches_known_critical_values():
    """Cross-check the from-scratch incomplete-beta-based p-value against
    standard t-table critical values (two-tailed, alpha=0.05) - these are
    textbook constants, not derived from this codebase."""
    known_critical_values = [(2.086, 20), (2.042, 30), (2.000, 60), (1.960, 1000)]
    for t_stat, df in known_critical_values:
        p = _t_two_tailed_p_value(t_stat, df)
        assert abs(p - 0.05) < 0.001


def test_p_value_decreases_as_t_stat_increases():
    p_small = _t_two_tailed_p_value(1.0, 30)
    p_large = _t_two_tailed_p_value(5.0, 30)
    assert p_large < p_small


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


def test_percentile_matches_known_values():
    data = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert _percentile(data, 50) == 3.0
    assert _percentile(data, 0) == 1.0
    assert _percentile(data, 100) == 5.0


def test_percentile_ordering_p50_le_p95_le_p99():
    rng = random.Random(3)
    stats = TrialStats([rng.expovariate(1.0) for _ in range(200)])
    assert stats.p50 <= stats.p95 <= stats.p99


def test_compare_randomizes_which_arm_runs_first():
    """Two different order_seed values shouldn't always produce the same
    baseline-first/optimized-first sequence - if they did, "randomizing"
    would be a no-op."""
    call_order_a = []
    call_order_b = []

    def make_tracker(log, label):
        def fn():
            log.append(label)
            return label

        return fn

    compare(
        baseline_fn=make_tracker(call_order_a, "base"),
        optimized_fn=make_tracker(call_order_a, "opt"),
        check_equivalent=lambda a, b: (True, "ok"),
        n_trials=20,
        warmup=0,
        order_seed=1,
    )
    compare(
        baseline_fn=make_tracker(call_order_b, "base"),
        optimized_fn=make_tracker(call_order_b, "opt"),
        check_equivalent=lambda a, b: (True, "ok"),
        n_trials=20,
        warmup=0,
        order_seed=2,
    )
    assert call_order_a != call_order_b


def test_compare_order_seed_is_reproducible():
    def make_tracker(log, label):
        def fn():
            log.append(label)
            return label

        return fn

    orders = []
    for _ in range(2):
        log = []
        compare(
            baseline_fn=make_tracker(log, "base"),
            optimized_fn=make_tracker(log, "opt"),
            check_equivalent=lambda a, b: (True, "ok"),
            n_trials=20,
            warmup=0,
            order_seed=42,
        )
        orders.append(tuple(log))
    assert orders[0] == orders[1]
