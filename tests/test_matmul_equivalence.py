import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from scenarios.matmul import matmul_naive, matmul_vectorized

_elements = st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False)


@given(
    a=arrays(dtype=np.float64, shape=(4, 3), elements=_elements),
    b=arrays(dtype=np.float64, shape=(3, 5), elements=_elements),
)
@settings(max_examples=50)
def test_naive_and_vectorized_matmul_agree(a: np.ndarray, b: np.ndarray) -> None:
    naive_out = matmul_naive(a, b)
    vectorized_out = matmul_vectorized(a, b)
    assert np.allclose(naive_out, vectorized_out, rtol=1e-8, atol=1e-8)
