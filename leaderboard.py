"""
CLI entry point for comparing several candidate implementations against
one baseline in a single run.

A scenario module must define:
  BASELINE_FN       () -> output
  CANDIDATES        dict[str, () -> output] - name -> candidate callable
  CHECK_EQUIVALENT  (baseline_out, candidate_out) -> (bool, str)
  TARGET            str, what was measured
  SOURCE            str, where this comparison lives / how to rerun it

Usage:
  python3 leaderboard.py example_moving_average_variants
"""

import argparse
import importlib
import sys
from types import ModuleType

from harness import compare_many, render_leaderboard

REQUIRED_ATTRS = ["BASELINE_FN", "CANDIDATES", "CHECK_EQUIVALENT", "TARGET", "SOURCE"]


def run_leaderboard(
    module: ModuleType,
    n_trials: int = 30,
    warmup: int = 5,
    min_speedup_pct: float = 5.0,
    t_threshold: float = 2.0,
) -> str:
    missing = [a for a in REQUIRED_ATTRS if not hasattr(module, a)]
    if missing:
        raise AttributeError(
            f"{module.__name__} is missing required attribute(s): {', '.join(missing)}"
        )

    results = compare_many(
        module.BASELINE_FN,
        module.CANDIDATES,
        module.CHECK_EQUIVALENT,
        n_trials=n_trials,
        warmup=warmup,
        min_speedup_pct=min_speedup_pct,
        t_threshold=t_threshold,
    )
    header = f"Leaderboard: {module.TARGET}\nsource: {module.SOURCE}\n"
    return header + "\n" + render_leaderboard(results)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare several candidate implementations against one baseline."
    )
    parser.add_argument(
        "scenario",
        help="Dotted module path exposing BASELINE_FN, CANDIDATES, "
        "CHECK_EQUIVALENT, TARGET, SOURCE",
    )
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--min-speedup", type=float, default=5.0, dest="min_speedup")
    parser.add_argument("--t-threshold", type=float, default=2.0, dest="t_threshold")
    args = parser.parse_args(argv)

    module = importlib.import_module(args.scenario)
    output = run_leaderboard(
        module,
        n_trials=args.n_trials,
        warmup=args.warmup,
        min_speedup_pct=args.min_speedup,
        t_threshold=args.t_threshold,
    )
    print(output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
