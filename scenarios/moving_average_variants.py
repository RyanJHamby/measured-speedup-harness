"""
Leaderboard scenario: three implementations of the same moving average,
compared against the naive loop in one run instead of one at a time.

Once there's more than one plausible replacement for something, the
question stops being "is X faster than baseline" and becomes "which of
these is actually the best choice, and is the fastest one even the right
one to ship" - moving_average_vectorized is faster in raw terms but has a
documented precision domain limit (see its docstring in
scenarios/moving_average.py).

moving_average_kahan was added expecting it to be the "best of both"
option (cumsum's speed, without cumsum's domain limit). Measuring it
(see tests/test_kahan_moving_average.py and the leaderboard output below)
found something less convenient but more honest: Kahan does extend the
safe-precision domain by roughly three orders of magnitude over cumsum,
but because the running compensation term is inherently sequential (each
step depends on the last), it can't be vectorized the way cumsum or
convolve can - a pure-Python-loop implementation of it runs roughly 40x
slower than cumsum here, not comparably fast. moving_average_convolve
turns out to dominate it outright for this specific operation: convolve
is both faster than Kahan *and* exact (not just "safer") on the same
adversarial input, since it never forms a large running total either.
Kahan's real value is as a general technique for problems that don't have
a convolution-shaped alternative available - not as the right choice for
this particular moving-average example. That's the point of measuring
several candidates instead of picking the one that sounds best on paper.

Run: python3 leaderboard.py scenarios.moving_average_variants
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from scenarios.moving_average import (
    _DATA,
    _WINDOW,
    _check_equivalent,
    moving_average_convolve,
    moving_average_kahan,
    moving_average_naive,
    moving_average_vectorized,
)

BASELINE_FN = lambda: moving_average_naive(_DATA, _WINDOW)
CANDIDATES = {
    "cumsum": lambda: moving_average_vectorized(_DATA, _WINDOW),
    "convolve": lambda: moving_average_convolve(_DATA, _WINDOW),
    "kahan": lambda: moving_average_kahan(_DATA, _WINDOW),
}
CHECK_EQUIVALENT = _check_equivalent
TARGET = f"moving average, N={len(_DATA)}, window={_WINDOW}"
SOURCE = "scenarios/moving_average_variants.py, run locally, no external deps beyond numpy"


if __name__ == "__main__":
    from leaderboard import run_leaderboard

    print(run_leaderboard(sys.modules[__name__], n_trials=25, warmup=3))
