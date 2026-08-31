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

Run: python3 example_matmul.py
"""

import random

import numpy as np

from harness import compare, render_finding


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


def check_equivalent(a: np.ndarray, b: np.ndarray) -> tuple[bool, str]:
    ok = np.allclose(a, b, rtol=1e-8, atol=1e-8)
    max_diff = float(np.max(np.abs(a - b)))
    return ok, f"max abs diff = {max_diff:.2e}, rtol=1e-8"


if __name__ == "__main__":
    random.seed(0)
    N = 40  # kept small so the naive O(n^3) loop finishes in a reasonable
    # time across 25 trials - shrink further if this is slow on your machine
    A = np.random.default_rng(0).random((N, N))
    B = np.random.default_rng(1).random((N, N))

    result = compare(
        baseline_fn=lambda: matmul_naive(A, B),
        optimized_fn=lambda: matmul_vectorized(A, B),
        check_equivalent=check_equivalent,
        n_trials=25,
        warmup=3,
        min_speedup_pct=5.0,
        t_threshold=2.0,
        label="matmul: naive triple loop vs BLAS",
    )

    finding = render_finding(
        result,
        technique="Replace naive triple-loop matmul with BLAS-backed np.matmul",
        target=f"matrix multiply, {N}x{N} @ {N}x{N}",
        source="example_matmul.py, run locally, no external deps beyond numpy",
    )

    print(finding)
    with open("findings.md", "a") as f:
        f.write(finding + "\n\n")
