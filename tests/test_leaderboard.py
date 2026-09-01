import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness import compare_many, render_leaderboard
from leaderboard import run_leaderboard


def test_compare_many_runs_each_candidate_independently():
    def baseline():
        return sum(range(5000))

    candidates = {
        "fast": lambda: sum(range(5000)),
        "identity": lambda: sum(range(5000)),
    }

    results = compare_many(
        baseline, candidates, lambda a, b: (a == b, "identical by construction"), n_trials=10, warmup=2
    )
    assert set(results) == {"fast", "identity"}
    for result in results.values():
        assert result.correctness_passed


def test_render_leaderboard_sorts_correctness_failures_last():
    import harness

    good = harness.ComparisonResult(
        baseline=harness.TrialStats([1.0] * 5),
        optimized=harness.TrialStats([0.1] * 5),
        correctness_passed=True,
        correctness_detail="ok",
        speedup_pct=90.0,
        t_stat=50.0,
        df=8.0,
        p_value=1e-9,
        speedup_ci_low=85.0,
        speedup_ci_high=95.0,
        ci_confidence=0.95,
        tier="confirmed",
        rule_applied="rule",
    )
    broken_but_faster = harness.ComparisonResult(
        baseline=harness.TrialStats([1.0] * 5),
        optimized=harness.TrialStats([0.01] * 5),
        correctness_passed=False,
        correctness_detail="mismatch",
        speedup_pct=99.0,
        t_stat=50.0,
        df=8.0,
        p_value=1e-9,
        speedup_ci_low=95.0,
        speedup_ci_high=99.0,
        ci_confidence=0.95,
        tier="fail",
        rule_applied="rule",
    )
    table = render_leaderboard({"broken": broken_but_faster, "good": good})
    assert table.index("good") < table.index("broken")


def test_run_leaderboard_against_real_scenario():
    module = importlib.import_module("example_moving_average_variants")
    result_text = run_leaderboard(module, n_trials=10, warmup=2)
    assert "cumsum" in result_text
    assert "convolve" in result_text
