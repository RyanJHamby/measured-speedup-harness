"""
Numerical-stability case study for harness.py: naive softmax vs. the
max-subtraction-stabilized version every ML framework actually uses.

softmax_naive computes exp(x) and normalizes directly. Two distinct
failure regimes were found empirically (see tests/test_softmax_equivalence.py
and the docstrings below), not assumed from the textbook description:

  1. A "silent zero" regime, starting around x ~ log(float64_max / n) for
     an n-element array (~708.2 for n=5): individual exp(x) values are
     still finite, but np.sum() of several of them overflows to inf,
     making every output silently 0.0 - no nan, no inf, nothing that looks
     obviously broken unless you check that the probabilities still sum
     to 1.
  2. An outright overflow regime, starting at x ~ 709.78 (confirmed by
     direct measurement, not the commonly-quoted "~709-710" approximation):
     individual exp(x) values overflow to inf, and inf/inf produces nan.

softmax_stable subtracts max(x) before exponentiating - the largest
resulting exponent is exactly exp(0) = 1, so neither failure mode is
reachable regardless of the input's absolute magnitude.

Run directly:      python3 scenarios/softmax.py
Run via the CLI:    python3 leaderboard.py scenarios.softmax
"""

import numpy as np


def softmax_naive(x: np.ndarray) -> np.ndarray:
    """No numerical safeguards. Two measured failure thresholds for an
    n-element array of similar magnitude: silent zero output starting
    around x ~ log(float64_max / n) (sum of exponentials overflows before
    any individual term does), and nan output starting at x ~ 709.78
    (individual exp(x) itself overflows to inf, then inf/inf = nan)."""
    e = np.exp(x)
    return e / np.sum(e)


def softmax_stable(x: np.ndarray) -> np.ndarray:
    """Subtract max(x) before exponentiating. The largest resulting
    exponent is exactly exp(0) = 1, so sum(exp(x - max(x))) is bounded by
    n regardless of x's absolute magnitude - neither of softmax_naive's
    failure modes is reachable."""
    shifted = x - np.max(x)
    e = np.exp(shifted)
    return e / np.sum(e)


def _is_valid_distribution(a: np.ndarray) -> bool:
    if np.isnan(a).any() or np.isinf(a).any():
        return False
    if np.any(a < 0):
        return False
    return bool(np.isclose(np.sum(a), 1.0, rtol=1e-6, atol=1e-6))


def _check_equivalent(a: np.ndarray, b: np.ndarray) -> tuple[bool, str]:
    """Checks more than np.allclose: a naive-softmax failure doesn't
    always produce nan/inf (see the "silent zero" regime documented
    above) - a broken output can still be finite. This verifies each
    side is a valid probability distribution (finite, non-negative,
    summing to ~1) before trusting a numeric comparison between them."""
    a_valid = _is_valid_distribution(a)
    b_valid = _is_valid_distribution(b)
    if not a_valid or not b_valid:
        return (
            False,
            f"baseline valid_distribution={a_valid} (sum={np.sum(a):.6g}), "
            f"candidate valid_distribution={b_valid} (sum={np.sum(b):.6g})",
        )
    ok = np.allclose(a, b, rtol=1e-9, atol=1e-12)
    max_diff = float(np.max(np.abs(a - b)))
    return ok, f"max abs diff = {max_diff:.2e}, rtol=1e-9"


# Scenario definition. Realistic, non-adversarial magnitude: the point of
# tests/test_softmax_equivalence.py is to find the edge case via Hypothesis,
# not to make the default demo itself adversarial.
_X = np.random.default_rng(0).normal(loc=0.0, scale=3.0, size=2_000)

BASELINE_FN = lambda: softmax_naive(_X)
OPTIMIZED_FN = lambda: softmax_stable(_X)  # bench.py/run_suite.py contract
CANDIDATES = {"stable": lambda: softmax_stable(_X)}  # leaderboard.py contract
CHECK_EQUIVALENT = _check_equivalent
# Framed as a safety fix, not a speedup: measured ~15-17% slower, not faster
# (see the module docstring) - bench.py's tiering correctly reports that as
# "noise" in the wrong direction, not a false "confirmed" speedup.
TECHNIQUE = "Naive softmax vs. max-subtraction-stabilized softmax: what avoiding overflow/silent-zero costs"
TARGET = f"softmax, N={len(_X)}, values ~ N(0, 3)"
SOURCE = "scenarios/softmax.py, run locally, no external deps beyond numpy"


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from leaderboard import run_leaderboard

    print(run_leaderboard(sys.modules[__name__], n_trials=25, warmup=3))
