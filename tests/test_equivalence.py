"""
Property-based correctness check for the moving-average example.

The harness's correctness gate (harness.compare) only checks one fixed
input per run. That catches "the fast version is wrong" but not "the fast
version is wrong on inputs I didn't happen to pick" - the more common way a
vectorized rewrite actually breaks: off-by-one at the array boundary,
window sizes equal to or larger than the data, a single-element array, etc.

This uses Hypothesis to generate many random arrays and window sizes and
checks the naive and vectorized implementations agree on all of them,
including edge cases Hypothesis is specifically good at finding (empty-ish
inputs, boundary sizes) that a hand-picked example would likely miss.

Run: pytest tests/test_equivalence.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from example_moving_average import moving_average_naive, moving_average_vectorized


@given(
    data=arrays(
        dtype=np.float64,
        shape=st.integers(min_value=1, max_value=200),
        elements=st.floats(
            min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False
        ),
    ),
    window_fraction=st.floats(min_value=0.01, max_value=1.0),
)
@settings(max_examples=200)
def test_naive_and_vectorized_agree(data: np.ndarray, window_fraction: float) -> None:
    window = max(1, int(len(data) * window_fraction))
    window = min(window, len(data))

    naive_out = moving_average_naive(data, window)
    vectorized_out = moving_average_vectorized(data, window)

    assert naive_out.shape == vectorized_out.shape
    assert np.allclose(naive_out, vectorized_out, rtol=1e-9, atol=1e-9)


def test_window_equal_to_length_returns_single_value() -> None:
    data = np.array([1.0, 2.0, 3.0, 4.0])
    window = 4
    naive_out = moving_average_naive(data, window)
    vectorized_out = moving_average_vectorized(data, window)
    assert naive_out.shape == (1,)
    assert np.allclose(naive_out, vectorized_out)
    assert naive_out[0] == pytest.approx(2.5)


def test_window_of_one_returns_input_unchanged() -> None:
    data = np.array([5.0, -2.0, 0.0, 3.5])
    naive_out = moving_average_naive(data, 1)
    vectorized_out = moving_average_vectorized(data, 1)
    assert np.allclose(naive_out, data)
    assert np.allclose(vectorized_out, data)
