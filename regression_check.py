"""
Detect when a comparison's confidence tier degrades between runs.

A `confirmed` speedup isn't permanent - a dependency upgrade, a platform
change, or an unrelated edit nearby can quietly erode it. Re-running the
same comparison over time and comparing tiers is how that gets caught,
the same way a test suite catches a correctness regression rather than
relying on someone noticing.

This reads a JSONL ledger (one record per run, written by
harness.to_ledger_record via bench.py's --ledger-file flag), groups
records by (technique, target), and flags any case where the most recent
run's tier ranks lower than the one before it for the same comparison.

Usage: python3 regression_check.py findings.jsonl
Exit code 1 if any regression is found, 0 otherwise.
"""

import json
import sys
from collections import defaultdict

TIER_RANK = {"fail": 0, "noise": 1, "marginal": 2, "confirmed": 3}


def load_ledger(path: str) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def find_regressions(records: list[dict]) -> list[tuple[str, dict, dict]]:
    by_key: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_key[f"{r['technique']} ({r['target']})"].append(r)

    regressions = []
    for key, entries in by_key.items():
        entries.sort(key=lambda r: r["timestamp"])
        if len(entries) < 2:
            continue
        prev, latest = entries[-2], entries[-1]
        if TIER_RANK[latest["tier"]] < TIER_RANK[prev["tier"]]:
            regressions.append((key, prev, latest))
    return regressions


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    if len(argv) != 1:
        print("usage: python3 regression_check.py <ledger.jsonl>", file=sys.stderr)
        return 2

    try:
        records = load_ledger(argv[0])
    except FileNotFoundError:
        print(f"no ledger at {argv[0]} yet - nothing to check")
        return 0

    regressions = find_regressions(records)
    if not regressions:
        print(f"no regressions across {len(records)} ledger entries")
        return 0

    print(f"{len(regressions)} regression(s) found:")
    for key, prev, latest in regressions:
        print(
            f"  {key}: {prev['tier']} -> {latest['tier']} "
            f"(speedup {prev['speedup_pct']:.1f}% -> {latest['speedup_pct']:.1f}%, "
            f"{prev['timestamp']} -> {latest['timestamp']})"
        )
    return 1


if __name__ == "__main__":
    sys.exit(main())
