import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from regression_check import find_regressions


def _record(technique, target, tier, timestamp, speedup=50.0):
    return {
        "technique": technique,
        "target": target,
        "tier": tier,
        "timestamp": timestamp,
        "speedup_pct": speedup,
    }


def test_no_regression_when_tier_holds_or_improves():
    records = [
        _record("swap X for Y", "n=100", "marginal", "2026-01-01T00:00:00"),
        _record("swap X for Y", "n=100", "confirmed", "2026-01-02T00:00:00"),
    ]
    assert find_regressions(records) == []


def test_flags_a_confirmed_speedup_that_became_noise():
    records = [
        _record("swap X for Y", "n=100", "confirmed", "2026-01-01T00:00:00"),
        _record("swap X for Y", "n=100", "noise", "2026-01-02T00:00:00"),
    ]
    regressions = find_regressions(records)
    assert len(regressions) == 1
    key, prev, latest = regressions[0]
    assert prev["tier"] == "confirmed"
    assert latest["tier"] == "noise"


def test_different_techniques_are_tracked_independently():
    records = [
        _record("technique A", "n=100", "confirmed", "2026-01-01T00:00:00"),
        _record("technique B", "n=100", "noise", "2026-01-01T00:00:00"),
        _record("technique A", "n=100", "confirmed", "2026-01-02T00:00:00"),
        _record("technique B", "n=100", "confirmed", "2026-01-02T00:00:00"),
    ]
    assert find_regressions(records) == []


def test_only_the_most_recent_pair_matters():
    """An old regression that later recovered shouldn't still be flagged."""
    records = [
        _record("swap X for Y", "n=100", "confirmed", "2026-01-01T00:00:00"),
        _record("swap X for Y", "n=100", "noise", "2026-01-02T00:00:00"),
        _record("swap X for Y", "n=100", "confirmed", "2026-01-03T00:00:00"),
    ]
    assert find_regressions(records) == []
