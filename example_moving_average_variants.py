"""
Leaderboard scenario: three implementations of the same moving average,
compared against the naive loop in one run instead of one at a time.

Once there's more than one plausible replacement for something, the
question stops being "is X faster than baseline" and becomes "which of
these is actually the best choice, and is the fastest one even the right
one to ship" - moving_average_vectorized is faster in raw terms but has a
documented precision domain limit (see its docstring in
example_moving_average.py); moving_average_convolve is slower but doesn't
share that limitation. A leaderboard makes that tradeoff visible in one
table instead of requiring three separate write-ups.

Run: python3 leaderboard.py example_moving_average_variants
"""

import numpy as np

from example_moving_average import (
    _DATA,
    _WINDOW,
    _check_equivalent,
    moving_average_convolve,
    moving_average_naive,
    moving_average_vectorized,
)

BASELINE_FN = lambda: moving_average_naive(_DATA, _WINDOW)
CANDIDATES = {
    "cumsum": lambda: moving_average_vectorized(_DATA, _WINDOW),
    "convolve": lambda: moving_average_convolve(_DATA, _WINDOW),
}
CHECK_EQUIVALENT = _check_equivalent
TARGET = f"moving average, N={len(_DATA)}, window={_WINDOW}"
SOURCE = "example_moving_average_variants.py, run locally, no external deps beyond numpy"


if __name__ == "__main__":
    import sys

    from leaderboard import run_leaderboard

    print(run_leaderboard(sys.modules[__name__], n_trials=25, warmup=3))
