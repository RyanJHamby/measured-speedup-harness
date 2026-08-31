"""
Runs a scenario's baseline or optimized callable in a tight loop for a
fixed duration. Meant to be launched under py-spy (see profile.py), not run
standalone - a single call to most of these functions is too fast for a
sampling profiler to get enough samples from, so this repeats it long
enough to profile.
"""

import argparse
import importlib
import time


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("scenario")
    parser.add_argument("--which", choices=["baseline", "optimized"], default="baseline")
    parser.add_argument("--seconds", type=float, default=3.0)
    args = parser.parse_args()

    module = importlib.import_module(args.scenario)
    fn = module.BASELINE_FN if args.which == "baseline" else module.OPTIMIZED_FN

    end = time.time() + args.seconds
    while time.time() < end:
        fn()


if __name__ == "__main__":
    main()
