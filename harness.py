"""
Noise-aware micro-benchmark harness.

The problem this solves: a single `time.time()` call before/after a change is
not evidence of a speedup. Timing noise (cache state, scheduler jitter, thermal
throttling, background load) routinely produces 10-30% swings between two runs
of *identical* code. Any optimization claim needs three things before it's
trustworthy:

  1. Correctness first  - the "optimized" version must produce the same
     output as the baseline, within a stated tolerance. A fast wrong answer
     is not a speedup.
  2. Enough trials to see the noise - one run tells you nothing about
     variance. This harness interleaves baseline/optimized calls (A/B/A/B...)
     rather than running all of one then all of the other, so slow drift
     (thermal, GC, background load) hits both arms equally instead of
     biasing whichever ran second.
  3. An explicit, pre-committed decision rule for calling a result "real" -
     not "it looked faster," but a statistical margin plus a minimum
     effect size, decided before looking at the numbers.

Usage: import `compare()` from this module and pass it a baseline callable,
an optimized callable, and a function that checks their outputs match.
"""

from __future__ import annotations

import math
import statistics
import time
from dataclasses import dataclass, field
from typing import Callable, Optional


@dataclass
class TrialStats:
    samples: list[float]

    @property
    def mean(self) -> float:
        return statistics.mean(self.samples)

    @property
    def stdev(self) -> float:
        return statistics.stdev(self.samples) if len(self.samples) > 1 else 0.0

    @property
    def n(self) -> int:
        return len(self.samples)


@dataclass
class ComparisonResult:
    baseline: TrialStats
    optimized: TrialStats
    correctness_passed: bool
    correctness_detail: str
    speedup_pct: float
    t_stat: float
    tier: str
    rule_applied: str


def _time_calls(fn: Callable[[], object], n_trials: int, warmup: int) -> list[float]:
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        samples.append(t1 - t0)
    return samples


def _welch_t(a: TrialStats, b: TrialStats) -> float:
    """Welch's t-statistic for two independent samples of unequal variance."""
    va, vb = a.stdev ** 2, b.stdev ** 2
    denom = math.sqrt(va / a.n + vb / b.n) if (va or vb) else 0.0
    if denom == 0:
        return math.inf if a.mean != b.mean else 0.0
    return (a.mean - b.mean) / denom


def compare(
    baseline_fn: Callable[[], object],
    optimized_fn: Callable[[], object],
    check_equivalent: Callable[[object, object], tuple[bool, str]],
    n_trials: int = 30,
    warmup: int = 5,
    min_speedup_pct: float = 5.0,
    t_threshold: float = 2.0,
    label: str = "unnamed comparison",
) -> ComparisonResult:
    """
    Run an interleaved A/B timing comparison with a correctness gate.

    check_equivalent(baseline_output, optimized_output) -> (passed, detail_str)

    Decision rule (declared here, applied mechanically, not adjusted after
    seeing results):
      - If correctness fails: tier = "fail" regardless of timing.
      - If correctness passes but speedup < min_speedup_pct: tier = "noise"
        (not enough measured effect to act on).
      - If speedup >= min_speedup_pct but the Welch t-statistic is below
        t_threshold: tier = "marginal" (effect is plausible but not clearly
        separated from trial-to-trial noise at this sample size).
      - If speedup >= min_speedup_pct and t_stat >= t_threshold: tier = "confirmed".
    """
    baseline_out = baseline_fn()
    optimized_out = optimized_fn()
    passed, detail = check_equivalent(baseline_out, optimized_out)

    # Interleave the timed calls so slow drift (thermal, GC, OS scheduling)
    # affects both arms rather than whichever one happens to run second.
    for _ in range(warmup):
        baseline_fn()
        optimized_fn()

    b_samples, o_samples = [], []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        baseline_fn()
        t1 = time.perf_counter()
        b_samples.append(t1 - t0)

        t0 = time.perf_counter()
        optimized_fn()
        t1 = time.perf_counter()
        o_samples.append(t1 - t0)

    baseline_stats = TrialStats(b_samples)
    optimized_stats = TrialStats(o_samples)

    speedup_pct = (
        (baseline_stats.mean - optimized_stats.mean) / baseline_stats.mean * 100.0
        if baseline_stats.mean
        else 0.0
    )
    t_stat = _welch_t(baseline_stats, optimized_stats)

    rule_applied = (
        f"correctness gate; min_speedup_pct={min_speedup_pct}; "
        f"t_threshold={t_threshold} (Welch's t, {n_trials} interleaved trials)"
    )

    if not passed:
        tier = "fail"
    elif speedup_pct < min_speedup_pct:
        tier = "noise"
    elif t_stat < t_threshold:
        tier = "marginal"
    else:
        tier = "confirmed"

    return ComparisonResult(
        baseline=baseline_stats,
        optimized=optimized_stats,
        correctness_passed=passed,
        correctness_detail=detail,
        speedup_pct=speedup_pct,
        t_stat=t_stat,
        tier=tier,
        rule_applied=rule_applied,
    )


def render_finding(result: ComparisonResult, technique: str, target: str, source: str) -> str:
    """Render a ComparisonResult as a finding: a technique claim tagged
    with a confidence tier and the measurement that backs it, rather than a
    bare assertion that something is "faster." Meant to be appended to a
    running findings log so later work can cite what was actually confirmed
    instead of re-litigating it."""
    b, o = result.baseline, result.optimized
    lines = [
        f"## {technique}",
        f"- target: {target}",
        f"- confidence: {result.tier}",
        f"- correctness: {'pass' if result.correctness_passed else 'FAIL'} "
        f"({result.correctness_detail})",
        f"- baseline: {b.mean * 1e3:.4f} ms +/- {b.stdev * 1e3:.4f} ms (n={b.n})",
        f"- optimized: {o.mean * 1e3:.4f} ms +/- {o.stdev * 1e3:.4f} ms (n={o.n})",
        f"- speedup: {result.speedup_pct:.1f}% (t={result.t_stat:.2f})",
        f"- decision_rule: {result.rule_applied}",
        f"- source: {source}",
    ]
    return "\n".join(lines)
