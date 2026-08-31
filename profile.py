"""
Capture a flamegraph of a scenario's baseline (or optimized) path with
py-spy, before assuming where the time is going.

The harness answers "is it faster and by how much." It deliberately
doesn't answer "why," because guessing why from a timing number alone is
how optimization effort gets spent in the wrong place. This is the
profile-first half of the discipline: point a sampling profiler at the
actual baseline before deciding what to change.

Usage:
  python3 profile.py example_moving_average --which baseline --out baseline.svg
  python3 profile.py example_moving_average --which optimized --out optimized.svg

Requires py-spy (see requirements-dev.txt). On macOS, py-spy needs to run
as root to attach to another process, even one it launches itself:
  sudo python3 profile.py example_moving_average
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Capture a flamegraph of a scenario's baseline or optimized path."
    )
    parser.add_argument("scenario", help="Dotted module path, e.g. example_moving_average")
    parser.add_argument("--which", choices=["baseline", "optimized"], default="baseline")
    parser.add_argument("--seconds", type=float, default=3.0)
    parser.add_argument("--out", default=None, help="Output SVG path (default: <which>.svg)")
    args = parser.parse_args(argv)

    if shutil.which("py-spy") is None:
        print(
            "py-spy not found on PATH. Install it with: pip install py-spy "
            "(already listed in requirements-dev.txt)",
            file=sys.stderr,
        )
        return 1

    out_path = args.out or f"{args.which}.svg"
    target = Path(__file__).with_name("profile_target.py")

    cmd = [
        "py-spy",
        "record",
        "-o",
        out_path,
        "--",
        sys.executable,
        str(target),
        args.scenario,
        "--which",
        args.which,
        "--seconds",
        str(args.seconds),
    ]
    return subprocess.call(cmd)


if __name__ == "__main__":
    sys.exit(main())
