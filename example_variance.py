"""
Third worked example for harness.py: three ways to compute variance, one
of which is a textbook numerical-stability trap.

variance_naive is the single-pass "sum of squares minus square of sum"
formula (E[X^2] - E[X]^2). It looks like a reasonable, faster alternative
to a two-pass mean-then-deviations computation - fewer lines, one pass
over the data, and (in this pure-Python implementation) both of its sums
go through Python's C-optimized built-in sum() - and for realistic-
magnitude data it agrees with the stable methods below to many decimal
places. But it computes two large, nearly-equal numbers (a sum of squares
and a squared sum) and subtracts them, which is catastrophic cancellation
waiting to happen once the data's magnitude grows relative to its spread.
See variance_naive's docstring for the actual domain limit, found
empirically (not assumed) - see tests/test_variance_equivalence.py for
how.

variance_welford is the numerically stable single-pass alternative: a
running mean and running sum-of-squared-deviations, never forming the
large intermediate sums that cause the cancellation. Measured against
this specific pure-Python implementation, it is NOT faster than
variance_naive, and neither is variance_two_pass - both do meaningfully
more per-element work in an explicit Python loop or generator expression,
where variance_naive's two sum() calls stay in optimized C the whole way
through. Run the leaderboard below and both stable alternatives land at
a real, measured slowdown (not a rounding-error-sized one), correctly
tiered as "noise" by the harness rather than a false "confirmed" speedup.
This is the honest version of the tradeoff: the fix for the cancellation
bug costs real time here, it isn't a free lunch, and knowing the actual
cost (not assuming it's negligible) is the point of measuring it rather
than reasoning about it in the abstract.

Run directly:      python3 example_variance.py
Run via the CLI:    python3 bench.py example_variance
Leaderboard (all three): python3 leaderboard.py example_variance
"""

import random


def variance_naive(data: list[float]) -> float:
    """Single-pass 'sum of squares minus square of sum': E[X^2] - E[X]^2.

    Domain limit, found empirically via the Hypothesis suite in
    tests/test_variance_equivalence.py (not assumed from theory) - and the
    severity depends on the ratio of mean^2 to the true variance, not on
    mean magnitude by itself, so this is a range, not a single number:

    - At the realistic magnitudes this file's own demo scenario uses
      (mean ~100), the error against the stable methods below is
      negligible (~1e-14 relative, i.e. ordinary floating-point noise,
      not the cancellation bug).
    - By mean magnitude 1e10-1e12 (with the data's spread at least
      comparable to float64's resolution at that magnitude - see the
      property test for why that qualifier matters), this reliably
      disagrees with the stable methods by more than 50% relative error,
      or returns an outright *negative* variance - mathematically
      impossible, and a dead giveaway of the bug rather than a rounding
      quibble. Verified across 1680 (magnitude, spread, sample count,
      seed) combinations, not just one hand-picked case.
    - Earlier, smaller magnitudes (1e5-1e9) can also trigger this,
      depending on how large the mean is relative to the data's spread -
      just not reliably enough at a fixed threshold to write down as a
      guaranteed property, which is why the test above targets the range
      that IS reliable rather than the lowest magnitude that sometimes
      breaks.

    The cause: it subtracts two large, nearly-equal numbers (a sum of
    squares and a squared sum), which is catastrophic cancellation, and
    the cause doesn't have a fixed-tolerance fix - the error scales with
    how large the intermediate sums get relative to the true variance, so
    no rtol bounds it for unbounded input magnitudes. Use variance_welford
    instead if the input range can't be guaranteed to stay small.
    """
    n = len(data)
    total = sum(data)
    total_sq = sum(x * x for x in data)
    mean = total / n
    return total_sq / n - mean * mean


def variance_two_pass(data: list[float]) -> float:
    """Textbook two-pass approach: compute the mean first, then the mean
    of squared deviations from it. Numerically stable (no large
    intermediate sums), at the cost of iterating the data twice."""
    n = len(data)
    mean = sum(data) / n
    return sum((x - mean) ** 2 for x in data) / n


def variance_welford(data: list[float]) -> float:
    """Welford's online algorithm: one pass, running mean and running
    sum-of-squared-deviations (M2), no large intermediate sums - stable
    like the two-pass version, but single-pass like the naive one."""
    n = 0
    mean = 0.0
    m2 = 0.0
    for x in data:
        n += 1
        delta = x - mean
        mean += delta / n
        delta2 = x - mean
        m2 += delta * delta2
    return m2 / n


def _check_equivalent(a: float, b: float) -> tuple[bool, str]:
    rel_err = abs(a - b) / abs(b) if b else abs(a - b)
    ok = rel_err < 1e-6
    return ok, f"relative error = {rel_err:.2e}"


# Scenario definition consumed by bench.py/leaderboard.py. Deliberately
# realistic-magnitude data (like temperature or sensor readings) where all
# three implementations agree - this default scenario is not meant to
# trigger variance_naive's domain limit; the Hypothesis suite in
# tests/test_variance_equivalence.py is what's responsible for finding
# that edge, on purpose, rather than baking an adversarial case into the
# demo everyone runs by default.
random.seed(0)
_N = 5_000
_DATA = [100.0 + random.gauss(0, 5.0) for _ in range(_N)]

BASELINE_FN = lambda: variance_naive(_DATA)
OPTIMIZED_FN = lambda: variance_welford(_DATA)
CANDIDATES = {
    "two_pass": lambda: variance_two_pass(_DATA),
    "welford": lambda: variance_welford(_DATA),
}
CHECK_EQUIVALENT = _check_equivalent
# Framed as a check, not an optimization claim: BASELINE_FN is the fast-
# but-domain-limited naive formula; OPTIMIZED_FN is the numerically stable
# alternative. Measured here, the stable version is actually slower (see
# example_variance.py's module docstring) - bench.py's tiering correctly
# reports that as "noise", not a false "confirmed" speedup. The point of
# this scenario is quantifying the real cost of fixing the cancellation
# bug, not claiming Welford's algorithm is faster.
TECHNIQUE = "Naive single-pass variance vs. Welford's algorithm: what does fixing the cancellation bug actually cost"
TARGET = f"variance, N={_N}, realistic magnitude (mean~100)"
SOURCE = "example_variance.py, run locally, no external deps"


if __name__ == "__main__":
    import sys

    from bench import run_scenario

    result, finding = run_scenario(sys.modules[__name__], n_trials=25, warmup=3)
    print(finding)
    with open("findings.md", "a") as f:
        f.write(finding + "\n\n")
