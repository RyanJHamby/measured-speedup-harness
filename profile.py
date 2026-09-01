"""
Capture a profile of a scenario's baseline (or optimized) path with py-spy,
before assuming where the time is going.

The harness answers "is it faster and by how much." It deliberately
doesn't answer "why," because guessing why from a timing number alone is
how optimization effort gets spent in the wrong place. This is the
profile-first half of the discipline: point a sampling profiler at the
actual baseline before deciding what to change.

Default output format is speedscope, not the raw inferno flamegraph SVG:
speedscope.app is a purpose-built viewer (drag-to-zoom, search, a
"sandwich" view grouping by function regardless of call path) and, unlike
an SVG rendered inline on GitHub, doesn't lose its interactivity to script
sanitization. See README.md for how these get published.

Usage:
  python3 profile.py scenarios.moving_average --which baseline --out baseline.speedscope.json
  python3 profile.py scenarios.moving_average --which optimized --out optimized.speedscope.json

Requires py-spy (see requirements-dev.txt). On macOS, py-spy needs to run
as root to attach to another process, even one it launches itself:
  sudo python3 profile.py scenarios.moving_average
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Capture a profile of a scenario's baseline or optimized path."
    )
    parser.add_argument("scenario", help="Dotted module path, e.g. scenarios.moving_average")
    parser.add_argument("--which", choices=["baseline", "optimized"], default="baseline")
    parser.add_argument("--seconds", type=float, default=3.0)
    parser.add_argument(
        "--format",
        choices=["speedscope", "flamegraph", "raw", "chrometrace"],
        default="speedscope",
    )
    parser.add_argument("--out", default=None, help="Output path (default: <which>.<format ext>)")
    args = parser.parse_args(argv)

    if shutil.which("py-spy") is None:
        print(
            "py-spy not found on PATH. Install it with: pip install py-spy "
            "(already listed in requirements-dev.txt)",
            file=sys.stderr,
        )
        return 1

    default_ext = {
        "speedscope": "speedscope.json",
        "flamegraph": "svg",
        "raw": "txt",
        "chrometrace": "chrometrace.json",
    }[args.format]
    out_path = args.out or f"{args.which}.{default_ext}"
    target = Path(__file__).with_name("profile_target.py")

    cmd = [
        "py-spy",
        "record",
        "-f",
        args.format,
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
