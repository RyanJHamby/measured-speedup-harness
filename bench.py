"""
CLI entry point for running a baseline-vs-candidate comparison defined in
any Python module, instead of only the hardcoded example scripts.

A scenario module must define:
  BASELINE_FN       () -> output
  OPTIMIZED_FN      () -> output
  CHECK_EQUIVALENT  (baseline_out, optimized_out) -> (bool, str)
  TECHNIQUE         str, one-line description of the change
  TARGET            str, what was measured (size, shape, etc.)
  SOURCE            str, where this comparison lives / how to rerun it

Usage:
  python bench.py example_moving_average
  python bench.py example_matmul --n-trials 50 --min-speedup 3
"""

import argparse
import importlib
import sys
from types import ModuleType

from harness import ComparisonResult, compare, render_finding

REQUIRED_ATTRS = [
    "BASELINE_FN",
    "OPTIMIZED_FN",
    "CHECK_EQUIVALENT",
    "TECHNIQUE",
    "TARGET",
    "SOURCE",
]


def run_scenario(
    module: ModuleType,
    n_trials: int = 30,
    warmup: int = 5,
    min_speedup_pct: float = 5.0,
    t_threshold: float = 2.0,
) -> tuple[ComparisonResult, str]:
    missing = [a for a in REQUIRED_ATTRS if not hasattr(module, a)]
    if missing:
        raise AttributeError(
            f"{module.__name__} is missing required attribute(s): {', '.join(missing)}"
        )

    result = compare(
        baseline_fn=module.BASELINE_FN,
        optimized_fn=module.OPTIMIZED_FN,
        check_equivalent=module.CHECK_EQUIVALENT,
        n_trials=n_trials,
        warmup=warmup,
        min_speedup_pct=min_speedup_pct,
        t_threshold=t_threshold,
    )
    finding = render_finding(
        result, technique=module.TECHNIQUE, target=module.TARGET, source=module.SOURCE
    )
    return result, finding


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a baseline-vs-candidate comparison defined in a Python module."
    )
    parser.add_argument(
        "scenario",
        help="Dotted module path exposing BASELINE_FN, OPTIMIZED_FN, "
        "CHECK_EQUIVALENT, TECHNIQUE, TARGET, SOURCE",
    )
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--min-speedup", type=float, default=5.0, dest="min_speedup")
    parser.add_argument("--t-threshold", type=float, default=2.0, dest="t_threshold")
    parser.add_argument("--findings-file", default="findings.md")
    parser.add_argument(
        "--plot",
        metavar="PNG_PATH",
        default=None,
        help="Save a histogram of the baseline/optimized trial distributions to this path",
    )
    args = parser.parse_args(argv)

    module = importlib.import_module(args.scenario)
    result, finding = run_scenario(
        module,
        n_trials=args.n_trials,
        warmup=args.warmup,
        min_speedup_pct=args.min_speedup,
        t_threshold=args.t_threshold,
    )

    print(finding)
    with open(args.findings_file, "a") as f:
        f.write(finding + "\n\n")

    if args.plot:
        from plot import plot_trial_distributions

        plot_trial_distributions(result, title=module.TECHNIQUE, out_path=args.plot)
        print(f"wrote {args.plot}")

    return 0 if result.tier != "fail" else 1


if __name__ == "__main__":
    sys.exit(main())
