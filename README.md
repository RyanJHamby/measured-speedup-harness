# measured-speedup-harness

I got tired of "I benchmarked it, it's faster" claims that fall apart the
second someone reruns the benchmark. This is the harness I actually use to
decide whether a change is worth keeping.

**The short version, if you don't write code:** a stopwatch lies if you only
click it once. Your laptop is doing a hundred other things while your code
runs, so two runs of the *exact same, unchanged* program can differ by
20-30% just from noise. If you only time something once before and once
after a change, you can't tell "it's faster" apart from "I got lucky." This
tool runs both versions back-to-back a bunch of times, checks that they
still give the same answer, and only calls something a real win if the
difference is bigger than the noise floor. That's it.

**The longer version, if you do:**

Every "we made this faster" claim I've seen go wrong has failed for one of
two boring reasons — nobody checked the output still matched, or nobody
looked at run-to-run variance before declaring victory. Both are cheap to
catch and annoying to debug after the fact, usually as either a silent
correctness regression that ships because the fast path skips an edge case,
or a benchmark result that quietly evaporates the next time someone reruns
it in CI.

So this harness enforces an order of operations:

1. **Correctness before speed, always.** Baseline and candidate outputs get
   compared before any timing number is trusted. A fast wrong answer gets
   rejected outright — timing a broken implementation is not interesting.
2. **Interleaved trials, not two separate blocks.** It alternates
   baseline/candidate calls (A, B, A, B, ...) instead of timing all of one
   then all of the other. If the machine gets slower over the course of the
   run — thermal throttling, memory pressure, whatever — that drift hits
   both arms about equally instead of unfairly penalizing whichever one ran
   second.
3. **A decision rule you commit to before looking at the numbers.** A
   result only gets called `confirmed` if it clears both a minimum effect
   size (5% by default — under that, who cares) *and* a statistical margin
   (a Welch's t-test against the noise in both samples). Otherwise you're
   just eyeballing a mean and hoping.

## What comes out the other end

Each comparison produces a **finding** — a plain-text record of what was
tried, whether it was actually correct, what the numbers were, and how
confident you should be, instead of a bare "20% faster!" claim with nothing
behind it:

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

Four tiers, in order of how much I'd trust them:

| Tier | What it means |
|---|---|
| `fail` | Outputs didn't match. Don't even look at the timing. |
| `noise` | Outputs matched, but the speedup is smaller than the threshold — could easily be nothing. |
| `marginal` | Speedup clears the threshold but isn't statistically distinct from the noise in the samples yet. Worth more trials before you tell anyone. |
| `confirmed` | Clears both bars. This is one I'd actually put in a PR description. |

The point of keeping these in a running `findings.md` is honesty over time
— if something marked `confirmed` six months ago turns out to be `noise`
on a different machine or after some other change, that's visible instead
of buried. A stale claim and a re-verified one shouldn't look the same.

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

`example_moving_average.py` is the worked example — naive O(N·W) sliding
sum vs. an O(N) cumsum-based version. I picked something boring on purpose;
the point is the harness, not the optimization. Swap in whatever you're
actually comparing and nothing about `harness.py` needs to change.

```
python3 example_moving_average.py
```

## Why bother with all this for a toy example

Because the discipline is the thing that transfers, not the moving
average. The same three steps — verify correctness first, measure enough
times to see past noise, and decide in advance what counts as a real
win — apply whether you're comparing two Python functions on a laptop or
two kernels on a cluster. Most of the benchmarking mistakes I've seen
weren't about the code being optimized; they were about skipping one of
these three steps under time pressure. This repo is small on purpose —
it's meant to be a habit you can point to, not a framework you have to
adopt.
