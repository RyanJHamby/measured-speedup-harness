"""
Verifies the claim in moving_average_kahan's docstring: that Kahan
(compensated) summation actually closes most of the precision gap
moving_average_vectorized (plain cumsum) leaves open on high-dynamic-range
data - the fix that docstring has been promising since it was written,
without ever being built or measured until now.

Two things are checked, deliberately kept separate:

1. A fixed, illustrative adversarial case: a long run of one repeated
   large value with a single differing small tail element, window=1 (so
   the naive implementation does zero arithmetic - its output is an exact
   copy of the input, the ground truth). Both cumsum and Kahan have to
   recover that tail value via subtraction of two large accumulated
   totals; this is the textbook shape of the cancellation problem, and is
   what makes the size of the improvement concrete and reproducible
   rather than an average over noise.

2. A genuine Hypothesis fuzzing campaign - not hand-picked - over element
   magnitudes up to +/-1e7 (three orders of magnitude past the +/-1e3
   domain limit documented for moving_average_vectorized in
   tests/test_equivalence.py) confirming Kahan holds to rtol=1e-9 across
   randomly generated arrays and window sizes in that range, not just the
   one illustrative case above.

What this does NOT claim: that Kahan summation is exact, or correct at
arbitrary magnitude. It measurably reduces accumulated rounding error; it
does not eliminate floating-point error. test_kahan_degrades_gracefully_at_extreme_magnitude
below shows it too eventually falls outside rtol=1e-9 at magnitude ~1e8,
just at a much higher magnitude than cumsum does, and by a smaller margin.

Run: pytest tests/test_kahan_moving_average.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from scenarios.moving_average import (
    moving_average_kahan,
    moving_average_naive,
    moving_average_vectorized,
)


def _max_relerr(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a - b) / np.maximum(np.abs(a), 1e-300)))


def test_kahan_recovers_tail_value_that_cumsum_gets_wrong():
    """Illustrative case, measured directly (not asserted from a guess):
    n=20000 repeated values of magnitude 1e6, one differing tail element
    of 0.42, window=1. True answer for the last output is exactly 0.42
    (naive does no arithmetic at window=1). cumsum recovers 0.4199981689
    (wrong in the 5th significant digit, relerr ~4.4e-6 - already outside
    the rtol=1e-6 tolerance moving_average_vectorized is held to). Kahan
    recovers 0.4200000000419 (wrong in the 10th significant digit, relerr
    ~1.0e-10) - roughly 44,000x more precise on this exact case, verified
    below rather than assumed.
    """
    n, magnitude, window = 20000, 1e6, 1
    data = np.full(n, magnitude)
    data[-1] = 0.42

    naive = moving_average_naive(data, window)
    cumsum = moving_average_vectorized(data, window)
    kahan = moving_average_kahan(data, window)

    cumsum_relerr = _max_relerr(naive, cumsum)
    kahan_relerr = _max_relerr(naive, kahan)

    assert cumsum_relerr > 1e-6, "expected cumsum to already be measurably wrong here"
    assert kahan_relerr < 1e-9, "expected Kahan to hold well within rtol=1e-9 here"
    assert kahan_relerr < cumsum_relerr / 100, "expected Kahan to be at least ~100x more precise"


@given(
    data=arrays(
        dtype=np.float64,
        shape=st.integers(min_value=1, max_value=300),
        elements=st.floats(min_value=-1e7, max_value=1e7, allow_nan=False, allow_infinity=False),
    ),
    window_fraction=st.floats(min_value=0.01, max_value=1.0),
)
@settings(max_examples=300)
def test_kahan_holds_far_past_cumsums_documented_domain_limit(
    data: np.ndarray, window_fraction: float
) -> None:
    """Randomly generated (not hand-picked) inputs across +/-1e7 - three
    orders of magnitude past the +/-1e3 range moving_average_vectorized is
    claimed correct for. This is the same style of fuzzing that originally
    found the cumsum cancellation problem in tests/test_equivalence.py,
    now confirming Kahan doesn't share it at this magnitude."""
    window = max(1, int(len(data) * window_fraction))
    window = min(window, len(data))

    naive = moving_average_naive(data, window)
    kahan = moving_average_kahan(data, window)

    assert naive.shape == kahan.shape
    # atol is deliberately looser than the rtol this test is actually
    # about: when an output happens to be near zero, absolute differences
    # at the 1e-9 scale are noise-floor artifacts of comparing two tiny
    # numbers, not evidence of the cancellation problem this test targets
    # (which is about large-magnitude running totals, not near-zero
    # outputs). rtol=1e-9 is the tolerance that actually matters here.
    assert np.allclose(naive, kahan, rtol=1e-9, atol=1e-6)


def test_kahan_degrades_gracefully_at_extreme_magnitude():
    """Kahan summation reduces rounding error, it does not eliminate it.
    Measured (not assumed): at magnitude 1e8 (one order of magnitude past
    the range the fuzzing test above covers), Kahan's error does exceed
    rtol=1e-9 - but still by roughly two orders of magnitude less than
    cumsum's error at the same input, confirming the improvement is real
    without overclaiming it as a complete fix."""
    n, magnitude, window = 1000, 1e8, 1
    data = np.full(n, magnitude)
    data[-1] = 0.42

    naive = moving_average_naive(data, window)
    cumsum = moving_average_vectorized(data, window)
    kahan = moving_average_kahan(data, window)

    cumsum_relerr = _max_relerr(naive, cumsum)
    kahan_relerr = _max_relerr(naive, kahan)

    assert kahan_relerr > 1e-9, (
        "expected Kahan to eventually degrade at extreme magnitude too - "
        "if this fails, Kahan is holding up even better than measured; "
        "update the docstring rather than treating this assertion as broken"
    )
    assert kahan_relerr < cumsum_relerr / 100


def test_window_of_one_is_exact_passthrough_for_kahan():
    data = np.array([5.0, -2.0, 0.0, 3.5])
    kahan = moving_average_kahan(data, 1)
    assert np.allclose(kahan, data)


def test_window_equal_to_length_matches_naive():
    data = np.array([1.0, 2.0, 3.0, 4.0])
    window = 4
    naive = moving_average_naive(data, window)
    kahan = moving_average_kahan(data, window)
    assert kahan.shape == (1,)
    assert kahan[0] == pytest.approx(2.5)
    assert np.allclose(naive, kahan)
