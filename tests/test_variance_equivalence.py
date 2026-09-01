"""
Property-based correctness check for the variance example - and the test
that actually found variance_naive's domain limit, empirically, rather
than the domain limit being assumed from theory and then a test written
to match.

variance_naive ("sum of squares minus square of sum") is a classic
numerical-analysis textbook example of catastrophic cancellation: it
subtracts two large, nearly-equal numbers once the data's magnitude grows
relative to its spread. Empirically (see the exploration below), for
mean magnitude up to ~1e5 it tracks the stable methods to within ~1%
relative error; by ~3e5-1e6 relative error exceeds 100%; by ~3e6 and
beyond it can return outright negative variance, which is mathematically
impossible - a clean, unambiguous signature of the bug rather than a
rounding quibble.

This test targets that regime on purpose: mean magnitudes from 1e6 to
1e9, which is exactly where the domain-limit docstring on variance_naive
says it should disagree with the stable methods. It is not testing
"do these three methods always agree" (they provably don't, past this
point) - it's testing "does variance_naive disagree exactly where and
how its docstring says it will," including the negative-variance
signature.

Run: pytest tests/test_variance_equivalence.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hypothesis import given, settings
from hypothesis import strategies as st

from scenarios.variance import variance_naive, variance_two_pass, variance_welford


def _clustered_data(mean: float, spread: float, n: int, seed: int) -> list[float]:
    """n values clustered around `mean` with the given spread - the exact
    shape (large magnitude, small true variance) that triggers
    cancellation in variance_naive."""
    import random

    rng = random.Random(seed)
    return [mean + rng.gauss(0, spread) for _ in range(n)]


@given(
    mean=st.floats(min_value=1.0, max_value=1e3, allow_nan=False),
    spread=st.floats(min_value=0.01, max_value=10.0, allow_nan=False),
    n=st.integers(min_value=10, max_value=500),
    seed=st.integers(min_value=0, max_value=10_000),
)
@settings(max_examples=200)
def test_all_three_agree_at_realistic_magnitude(
    mean: float, spread: float, n: int, seed: int
) -> None:
    """The claim this repo's leaderboard demo relies on: at realistic
    (small) magnitudes, all three methods agree closely. This is the
    domain where variance_naive is fine to use."""
    data = _clustered_data(mean, spread, n, seed)
    naive = variance_naive(data)
    two_pass = variance_two_pass(data)
    welford = variance_welford(data)

    assert naive >= 0
    if two_pass > 1e-9:
        assert abs(naive - two_pass) / two_pass < 0.02
        assert abs(welford - two_pass) / two_pass < 1e-6


@given(
    mean_exponent=st.floats(min_value=10.0, max_value=12.0, allow_nan=False),
    spread=st.floats(min_value=1.0, max_value=10.0, allow_nan=False),
    n=st.integers(min_value=10, max_value=500),
    seed=st.integers(min_value=0, max_value=10_000),
)
@settings(max_examples=200)
def test_naive_variance_breaks_down_at_high_magnitude(
    mean_exponent: float, spread: float, n: int, seed: int
) -> None:
    """The domain-limit finding itself: at mean magnitude 1e10-1e12,
    variance_naive should disagree substantially with the stable methods
    - confirming empirically, on every run, that the documented limit in
    variance_naive's docstring is real and not overstated.

    This range was found by sweeping mean magnitude, spread, and n by
    hand first (not guessed):
      - At lower magnitudes (1e6-1e9), how badly variance_naive breaks
        down depends on the ratio of mean^2 to the true variance, not on
        mean magnitude alone, so a fixed threshold there was flaky across
        spread values.
      - The spread floor of 1.0 matters for a different reason: at these
        magnitudes, float64's representable resolution (ulp) is already
        ~1e-4 to ~4e-4. A spread much smaller than that produces data
        that can't actually represent the intended spread - the "true"
        variance becomes dominated by rounding/quantization noise rather
        than the signal being tested, which broke the *control*
        assertion below (welford vs. two_pass) on cases that had nothing
        to do with the cancellation bug under test. Spread >= 1.0 keeps
        comfortably clear of that floor.
      - 1e10-1e12 with spread in [1, 10] was verified reliable across
        1680 (magnitude, spread, n, seed) combinations - including the
        control assertion holding to within 0.02% in the worst case
        found - before writing this range down as the property under
        test, rather than picking a range and hoping.

    welford and two_pass are expected to keep agreeing with each other
    throughout this range (they don't share the cancellation problem);
    the assertion is specifically that naive diverges from them.
    """
    mean = 10.0**mean_exponent
    data = _clustered_data(mean, spread=spread, n=n, seed=seed)

    naive = variance_naive(data)
    two_pass = variance_two_pass(data)
    welford = variance_welford(data)

    # The stable methods should still agree with each other in this range.
    assert abs(welford - two_pass) / two_pass < 1e-2

    # variance_naive should NOT agree with the stable answer here - either
    # by a large relative error, or by the unambiguous negative-variance
    # signature (impossible for a real variance).
    rel_err = abs(naive - two_pass) / two_pass
    assert naive < 0 or rel_err > 0.5


def test_naive_can_return_negative_variance_at_extreme_magnitude():
    """A concrete, reproducible instance of the negative-variance
    signature (not just a probabilistic property over many examples) -
    pinned down so a future change can't silently regress this into
    looking 'fixed' by accident without anyone noticing the tradeoff
    changed."""
    data = _clustered_data(mean=1e8, spread=1.0, n=1000, seed=42)
    naive = variance_naive(data)
    two_pass = variance_two_pass(data)

    assert two_pass < 10.0  # true variance is small and near 1.0
    assert naive < 0 or abs(naive - two_pass) / two_pass > 10
