# measured-speedup-harness

A harness for verifying that a code change is actually faster, not just
faster on one lucky run.

## Why this exists

A single before/after timing is not evidence. Two runs of identical,
unchanged code can differ by 20-30% from cache state, GC pauses, scheduler
jitter, and thermal throttling alone. Most incorrect "this is faster"
claims trace back to one of two causes: nobody checked the output was still
correct, or nobody accounted for run-to-run variance before calling the
result real. Both are cheap to catch up front and expensive to find out
about later — either as a correctness regression that shipped because the
fast path skipped an edge case, or a benchmark result that quietly stops
holding up the next time someone reruns it.

This harness enforces three things, in order, before a speedup is allowed
to be called real:

1. **Correctness first.** Baseline and candidate outputs are compared
   before any timing is trusted. A fast, incorrect implementation is
   rejected outright — its timing is not evaluated.
2. **Interleaved trials, not two separate blocks.** Baseline and candidate
   calls alternate (A, B, A, B, ...) instead of timing all of one and then
   all of the other. If the machine slows down over the course of the run
   — thermal throttling, memory pressure — that drift affects both arms
   roughly equally instead of penalizing whichever one ran second.
3. **A decision rule fixed in advance.** A result is only marked
   `confirmed` if it clears both a minimum effect size (5% by default) and
   a statistical margin (a Welch's t-test against the variance in both
   samples). A lower mean by itself is not sufficient.

## What it produces

Each comparison renders as a **finding**: what was compared, whether it was
correct, the measured numbers, and a confidence tier — not a bare
percentage with no basis behind it.

```
## Replace naive O(N*W) sliding-sum loop with O(N) cumsum-based moving average
- target: moving average, N=4000, window=50
- confidence: confirmed
- correctness: pass (max abs diff = 2.89e-14, rtol=1e-9)
- baseline: 16.4050 ms +/- 1.3765 ms (n=25)
- optimized: 0.1092 ms +/- 0.0254 ms (n=25)
- speedup: 99.3% (t=59.18)
- decision_rule: correctness gate; min_speedup_pct=5.0; t_threshold=2.0 (Welch's t, 25 interleaved trials)
- source: example_moving_average.py, run locally, no external deps beyond numpy
```

| Tier | Meaning |
|---|---|
| `fail` | Outputs didn't match. Timing is not evaluated. |
| `noise` | Outputs matched, but the speedup is below the minimum threshold. |
| `marginal` | Speedup clears the threshold but isn't statistically distinguishable from sample noise yet. |
| `confirmed` | Clears both the effect-size threshold and the statistical margin. |

Findings accumulate in `findings.md` as a running log. The tiering is what
keeps that log honest over time: if a change marked `confirmed` stops
holding up under different conditions, that shows up as a new, lower-tier
entry rather than as a silently stale claim.

## Where this applies

The harness is deliberately generic — anything with a baseline
implementation, a candidate implementation, and a way to check they agree.
Some concrete cases it's suited for:

- **Serialization format changes** — e.g. switching a hot path from JSON to
  a binary format. Correctness check confirms round-trip equality; the
  harness confirms the encode/decode time actually improved and by how
  much, before the format change goes into a PR description.
- **Data structure swaps** — e.g. replacing a list-based lookup with a set
  or dict for membership checks. Easy to assume it's faster; the harness
  catches the case where the collection is small enough that the overhead
  of the new structure erases the algorithmic win.
- **Caching or memoization additions** — confirms the cached path returns
  identical results to the uncached one (a common source of silent
  staleness bugs) and quantifies the actual hit-path speedup, since cache
  effects are exactly the kind of thing that looks great on a warm run and
  disappears under real trial-to-trial variance.
- **Library or dependency upgrades** — e.g. a new major version of a
  parsing or numerical library that claims to be faster. Running both
  versions through the same harness turns a changelog claim into a
  measured one, on your actual workload rather than the library's own
  benchmark suite.
- **Vectorizing a loop** (the worked example in this repo) — replacing a
  Python-level loop with a NumPy/vectorized equivalent. The speedup is
  usually large and easy to confirm, but the correctness gate still
  matters: vectorized rewrites are a common place to introduce off-by-one
  or edge-case bugs at array boundaries.
- **Regression protection over time** — the same comparison can be re-run
  on every change to a hot path, so a `confirmed` speedup that later
  degrades to `noise` (a dependency update, a platform change, an
  unrelated code change nearby) is caught the same way a test suite catches
  a correctness regression.

## Using it

```python
from harness import compare, render_finding

result = compare(
    baseline_fn=my_baseline,       # () -> output
    optimized_fn=my_candidate,     # () -> output
    check_equivalent=my_check,     # (baseline_out, optimized_out) -> (bool, str)
    n_trials=30,
    warmup=5,
    min_speedup_pct=5.0,
    t_threshold=2.0,
)
print(render_finding(result, technique="...", target="...", source="..."))
```

`example_moving_average.py` (memory-bound) and `example_matmul.py`
(compute-bound) are fully worked examples. Both run standalone:

```
python3 example_moving_average.py
python3 example_matmul.py
```

Or run either through the CLI, which works against any module exposing
`BASELINE_FN`, `OPTIMIZED_FN`, `CHECK_EQUIVALENT`, `TECHNIQUE`, `TARGET`,
and `SOURCE`:

```
python3 bench.py example_moving_average --n-trials 50 --min-speedup 3
python3 bench.py example_matmul
```

Both examples double as the reference for wiring up a new comparison —
write a module with the six attributes above and `bench.py` runs it the
same way, without touching `harness.py`.

The one-off correctness check inside `compare()` only checks the specific
input each example happens to use. `tests/` runs the two implementations
against hundreds of generated inputs (via Hypothesis) to catch the boundary
cases a hand-picked example would miss — this is how the precision limit
documented in `moving_average_vectorized` was actually found, not
guessed at in advance:

```
pip install -r requirements-dev.txt
pytest tests/
```

## Design intent

The three-step discipline here — verify correctness, measure past noise,
decide the bar in advance — is the part meant to generalize. The moving
average example is incidental; the same approach applies to comparing two
implementations of anything, at any scale, where "is this actually faster"
needs to be answered with evidence rather than a single timestamp diff.
