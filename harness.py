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
import random
import statistics
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
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
    speedup_ci_low: float
    speedup_ci_high: float
    ci_confidence: float
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


def _bootstrap_speedup_ci(
    baseline: TrialStats,
    optimized: TrialStats,
    n_resamples: int = 2000,
    confidence: float = 0.95,
    rng: random.Random | None = None,
) -> tuple[float, float]:
    """
    Bootstrap confidence interval for the speedup percentage.

    Welch's t-test assumes the underlying timing distributions are roughly
    normal, which trial timings often aren't (they're frequently right-skewed
    - most calls cluster near a floor, with occasional slow outliers from
    scheduling or GC). Bootstrapping resamples the observed trials with
    replacement and recomputes the speedup many times, giving a confidence
    interval that doesn't depend on that assumption. It's a second, largely
    independent check on the same question the t-test answers.
    """
    rng = rng or random.Random()
    b_samples, o_samples = baseline.samples, optimized.samples
    speedups = []
    for _ in range(n_resamples):
        b_resample = rng.choices(b_samples, k=len(b_samples))
        o_resample = rng.choices(o_samples, k=len(o_samples))
        b_mean = sum(b_resample) / len(b_resample)
        o_mean = sum(o_resample) / len(o_resample)
        speedups.append((b_mean - o_mean) / b_mean * 100.0 if b_mean else 0.0)

    speedups.sort()
    lo_idx = int((1 - confidence) / 2 * n_resamples)
    hi_idx = min(int((1 + confidence) / 2 * n_resamples), n_resamples - 1)
    return speedups[lo_idx], speedups[hi_idx]


def compare(
    baseline_fn: Callable[[], object],
    optimized_fn: Callable[[], object],
    check_equivalent: Callable[[object, object], tuple[bool, str]],
    n_trials: int = 30,
    warmup: int = 5,
    min_speedup_pct: float = 5.0,
    t_threshold: float = 2.0,
    label: str = "unnamed comparison",
    n_bootstrap_resamples: int = 2000,
    ci_confidence: float = 0.95,
    bootstrap_seed: int | None = None,
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
    ci_low, ci_high = _bootstrap_speedup_ci(
        baseline_stats,
        optimized_stats,
        n_resamples=n_bootstrap_resamples,
        confidence=ci_confidence,
        rng=random.Random(bootstrap_seed) if bootstrap_seed is not None else None,
    )

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
        speedup_ci_low=ci_low,
        speedup_ci_high=ci_high,
        ci_confidence=ci_confidence,
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
        f"- speedup: {result.speedup_pct:.1f}% (t={result.t_stat:.2f}), "
        f"{result.ci_confidence * 100:.0f}% CI [{result.speedup_ci_low:.1f}%, "
        f"{result.speedup_ci_high:.1f}%]",
        f"- decision_rule: {result.rule_applied}",
        f"- source: {source}",
    ]
    return "\n".join(lines)


def to_ledger_record(
    result: ComparisonResult, technique: str, target: str, source: str
) -> dict:
    """Machine-readable form of a finding, for a running JSONL ledger.

    The same technique/target pair gets re-run over time (different
    machine, after a dependency bump, on CI vs. locally); a ledger of these
    records is what lets a later run notice a `confirmed` result quietly
    became `noise` on some other occasion, instead of that being buried in
    a markdown log meant for humans. See regression_check.py.
    """
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "technique": technique,
        "target": target,
        "source": source,
        "tier": result.tier,
        "correctness_passed": result.correctness_passed,
        "correctness_detail": result.correctness_detail,
        "baseline_mean_ms": result.baseline.mean * 1e3,
        "optimized_mean_ms": result.optimized.mean * 1e3,
        "speedup_pct": result.speedup_pct,
        "t_stat": result.t_stat,
        "speedup_ci_low_pct": result.speedup_ci_low,
        "speedup_ci_high_pct": result.speedup_ci_high,
        "n_trials": result.baseline.n,
    }
