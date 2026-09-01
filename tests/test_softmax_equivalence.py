"""
Property-based correctness check for the softmax example, plus pinned
regression tests for the two failure thresholds found by actually running
this (see example_softmax.py's module docstring and function docstrings):

  1. "Silent zero" regime: np.sum(exp(x)) overflows to inf before any
     individual exp(x) term does, for x roughly above log(float64_max / n)
     - every output silently becomes 0.0. No nan, no inf; only a violated
     sum-to-1 invariant reveals it.
  2. "nan" regime: individual exp(x) itself overflows to inf starting at
     x ~ 709.78 (confirmed by direct measurement - not the commonly-quoted
     "~709-710" approximation), then inf/inf = nan.

softmax_stable is claimed correct at any magnitude because subtracting
max(x) first bounds the largest exponent to exactly exp(0) = 1 - the
property test below fuzzes a wide magnitude range specifically to check
that claim rather than assume it.

Run: pytest tests/test_softmax_equivalence.py
"""

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from example_softmax import _is_valid_distribution, softmax_naive, softmax_stable

_safe_elements = st.floats(min_value=-50, max_value=50, allow_nan=False, allow_infinity=False)


@given(
    x=arrays(dtype=np.float64, shape=st.integers(min_value=1, max_value=200), elements=_safe_elements)
)
@settings(max_examples=200)
def test_naive_and_stable_agree_at_safe_magnitudes(x: np.ndarray) -> None:
    """Below the overflow thresholds, both implementations should compute
    the same distribution - the stabilization is a safety net, not a
    change in what's being computed."""
    naive_out = softmax_naive(x)
    stable_out = softmax_stable(x)
    assert np.allclose(naive_out, stable_out, rtol=1e-9, atol=1e-12)
    assert _is_valid_distribution(stable_out)


@given(
    x=arrays(
        dtype=np.float64,
        shape=st.integers(min_value=1, max_value=50),
        elements=st.floats(min_value=-1e4, max_value=1e4, allow_nan=False, allow_infinity=False),
    )
)
@settings(max_examples=200)
def test_stable_always_produces_a_valid_distribution(x: np.ndarray) -> None:
    """softmax_stable's claimed correctness isn't domain-limited the way
    moving_average_vectorized's is - fuzz a wide magnitude range
    (well past where softmax_naive is already broken, see the pinned
    threshold tests below) and confirm the claim holds rather than assume
    it from the "subtract max first" argument alone."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stable_out = softmax_stable(x)
    assert _is_valid_distribution(stable_out)


def test_naive_softmax_silently_zeros_before_it_nans() -> None:
    """Pinned regression test for the measured 'silent zero' threshold: for
    n=5 equal-valued elements, sum(exp(x)) overflows to inf (making every
    output 0.0, not nan) starting between x=708.0 and x=708.3 - well below
    the ~709.78 point where individual exp(x) terms overflow. 708.0 must
    still be correct; 708.3 must already be broken, and broken silently
    (finite zeros, not nan/inf) - that silence is exactly what
    _is_valid_distribution's sum-to-1 check exists to catch."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        still_correct = softmax_naive(np.array([708.0] * 5))
        already_broken = softmax_naive(np.array([708.3] * 5))

    assert _is_valid_distribution(still_correct)
    assert np.allclose(still_correct, 0.2, atol=1e-9)

    assert not np.isnan(already_broken).any()
    assert not np.isinf(already_broken).any()
    assert np.allclose(already_broken, 0.0)
    assert not _is_valid_distribution(already_broken)


def test_naive_softmax_nans_past_measured_exp_overflow_point() -> None:
    """Pinned regression test for the measured nan threshold: individual
    exp(x) overflows to inf starting between x=709.782 and x=709.783
    (float64), not the commonly-quoted approximate '~709-710'."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        just_below = np.exp(np.float64(709.782))
        just_above = np.exp(np.float64(709.783))
    assert np.isfinite(just_below)
    assert np.isinf(just_above)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        naive_out = softmax_naive(np.array([709.783] * 5))
        stable_out = softmax_stable(np.array([709.783] * 5))

    assert np.isnan(naive_out).all()
    assert not _is_valid_distribution(naive_out)
    assert _is_valid_distribution(stable_out)


def test_stable_handles_magnitude_naive_already_fails_at() -> None:
    """softmax_stable stays correct well past both of softmax_naive's
    measured failure thresholds (708.2ish and 709.78ish) - tested here at
    1e6, several orders of magnitude beyond either."""
    x = np.array([1e6, 1e6 + 1.0, 1e6 + 2.0])
    stable_out = softmax_stable(x)
    assert _is_valid_distribution(stable_out)
    # largest input gets the largest probability, as it should
    assert stable_out[-1] == max(stable_out)
