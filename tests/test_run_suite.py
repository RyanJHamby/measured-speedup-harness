import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from run_suite import render_summary, run_suite


def test_run_suite_runs_real_scenarios_and_tallies_tiers():
    results = run_suite(
        ["example_moving_average", "example_matmul"], n_trials=10, warmup=2
    )
    assert set(results) == {"example_moving_average", "example_matmul"}
    for outcome in results.values():
        assert outcome[0] == "ok"
        _, result, finding = outcome
        assert result.correctness_passed
        assert isinstance(finding, str)


def test_run_suite_isolates_a_bad_scenario_without_aborting_the_batch():
    results = run_suite(
        ["example_moving_average", "this_module_does_not_exist"], n_trials=10, warmup=2
    )
    assert results["example_moving_average"][0] == "ok"
    assert results["this_module_does_not_exist"][0] == "error"


def test_render_summary_counts_tiers_and_errors():
    results = {
        "a": ("error", "boom"),
        "b": ("ok", _FakeResult(tier="confirmed", speedup_pct=42.0, correctness_passed=True), ""),
        "c": ("ok", _FakeResult(tier="noise", speedup_pct=1.0, correctness_passed=True), ""),
    }
    summary = render_summary(results)
    assert "1 error" in summary or "error" in summary
    assert "confirmed" in summary
    assert "noise" in summary


class _FakeResult:
    def __init__(self, tier, speedup_pct, correctness_passed):
        self.tier = tier
        self.speedup_pct = speedup_pct
        self.correctness_passed = correctness_passed
