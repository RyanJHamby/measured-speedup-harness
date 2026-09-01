"""
Run a batch of scenario comparisons and produce one consolidated summary.

A single comparison answers "is this one change faster." That's not the
shape of the problem a real migration poses: bumping a framework version,
moving to new hardware, or upgrading a compiler touches dozens to hundreds
of kernels or model subgraphs at once, and the question isn't "is op #37
faster" - it's "how many of these are still correct, how many actually got
faster, and which ones need a human to look at them." This runs the same
per-comparison discipline (correctness gate, noise-aware timing, a fixed
decision rule) across a whole batch and reports counts by tier instead of
requiring someone to read N individual reports.

Usage:
  python3 run_suite.py example_moving_average example_matmul
  python3 run_suite.py example_moving_average example_matmul --ledger-file findings.jsonl
"""

import argparse
import importlib
import json
import sys
from collections import Counter

from bench import run_scenario
from harness import to_ledger_record


def run_suite(scenario_names: list[str], **compare_kwargs) -> dict[str, tuple]:
    """Returns {scenario_name: ("ok", result, finding) | ("error", message)}.

    A single scenario failing to import or run (a bad module, a crashing
    candidate implementation) doesn't abort the batch - that would defeat
    the point of running many at once. It's recorded as its own outcome
    alongside the rest.
    """
    results = {}
    for name in scenario_names:
        try:
            module = importlib.import_module(name)
            result, finding = run_scenario(module, **compare_kwargs)
            results[name] = ("ok", result, finding)
        except Exception as e:
            results[name] = ("error", str(e))
    return results


def render_summary(results: dict[str, tuple]) -> str:
    tier_counts: Counter = Counter()
    rows = []
    for name, outcome in results.items():
        if outcome[0] == "error":
            tier_counts["error"] += 1
            rows.append((name, "error", "-", outcome[1]))
        else:
            _, result, _ = outcome
            tier_counts[result.tier] += 1
            rows.append(
                (
                    name,
                    result.tier,
                    f"{result.speedup_pct:.1f}%",
                    "pass" if result.correctness_passed else "FAIL",
                )
            )

    header = "Suite summary: " + ", ".join(
        f"{count} {tier}" for tier, count in tier_counts.most_common()
    )
    lines = [header, "", "| scenario | tier | speedup | correctness |", "|---|---|---|---|"]
    for name, tier, speedup, correctness in rows:
        lines.append(f"| {name} | {tier} | {speedup} | {correctness} |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a batch of scenario comparisons and summarize results by tier - "
        "the shape of validating many migrated kernels/models at once, not one at a time."
    )
    parser.add_argument("scenarios", nargs="+", help="Dotted module paths to run")
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--min-speedup", type=float, default=5.0, dest="min_speedup_pct")
    parser.add_argument("--t-threshold", type=float, default=2.0, dest="t_threshold")
    parser.add_argument("--ledger-file", default=None)
    parser.add_argument("--summary-file", default=None)
    args = parser.parse_args(argv)

    results = run_suite(
        args.scenarios,
        n_trials=args.n_trials,
        warmup=args.warmup,
        min_speedup_pct=args.min_speedup_pct,
        t_threshold=args.t_threshold,
    )

    if args.ledger_file:
        with open(args.ledger_file, "a") as f:
            for name, outcome in results.items():
                if outcome[0] == "error":
                    continue
                _, result, _ = outcome
                module = importlib.import_module(name)
                record = to_ledger_record(
                    result, module.TECHNIQUE, module.TARGET, module.SOURCE
                )
                f.write(json.dumps(record) + "\n")

    summary = render_summary(results)
    print(summary)
    if args.summary_file:
        with open(args.summary_file, "w") as f:
            f.write(summary + "\n")

    any_bad = any(
        outcome[0] == "error" or outcome[1].tier == "fail" for outcome in results.values()
    )
    return 1 if any_bad else 0


if __name__ == "__main__":
    sys.exit(main())
