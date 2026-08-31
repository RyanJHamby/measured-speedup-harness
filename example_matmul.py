"""
Second worked example for harness.py: naive triple-loop matrix
multiplication vs np.matmul (BLAS-backed).

The moving-average example is memory-bound - the win comes from doing
O(N) work instead of O(N*W) work, and the bottleneck is Python interpreter
overhead per element. This example is compute-bound in a different way:
both implementations do the same asymptotic amount of arithmetic
(O(n^3)), and the speedup comes entirely from using a vectorized,
cache-aware, BLAS-backed routine instead of scalar Python loops. Running
both examples through the same harness is meant to show the harness
doesn't care which kind of bottleneck it's measuring.

Run directly:      python3 example_matmul.py
Run via the CLI:    python3 bench.py example_matmul
"""

import random

import numpy as np


def matmul_naive(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Textbook triple-loop matrix multiply. O(n^3) scalar Python ops."""
    n, k = a.shape
    k2, m = b.shape
    assert k == k2
    out = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            total = 0.0
            for x in range(k):
                total += a[i, x] * b[x, j]
            out[i, j] = total
    return out


def matmul_vectorized(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Same computation via BLAS through np.matmul."""
    return np.matmul(a, b)


def _check_equivalent(a: np.ndarray, b: np.ndarray) -> tuple[bool, str]:
    ok = np.allclose(a, b, rtol=1e-8, atol=1e-8)
    max_diff = float(np.max(np.abs(a - b)))
    return ok, f"max abs diff = {max_diff:.2e}, rtol=1e-8"


# Scenario definition consumed by bench.py. N kept small so the naive
# O(n^3) loop finishes in a reasonable time across N trials.
random.seed(0)
_N = 40
_A = np.random.default_rng(0).random((_N, _N))
_B = np.random.default_rng(1).random((_N, _N))

BASELINE_FN = lambda: matmul_naive(_A, _B)
OPTIMIZED_FN = lambda: matmul_vectorized(_A, _B)
CHECK_EQUIVALENT = _check_equivalent
TECHNIQUE = "Replace naive triple-loop matmul with BLAS-backed np.matmul"
TARGET = f"matrix multiply, {_N}x{_N} @ {_N}x{_N}"
SOURCE = "example_matmul.py, run locally, no external deps beyond numpy"


if __name__ == "__main__":
    import sys

    from bench import run_scenario

    result, finding = run_scenario(sys.modules[__name__], n_trials=25, warmup=3)
    print(finding)
    with open("findings.md", "a") as f:
        f.write(finding + "\n\n")
