# measured-speedup-harness

A small harness for making "this is faster" claims that survive scrutiny.

## The problem

A single before/after timing is not evidence. Trial-to-trial noise from
cache state, GC pauses, scheduler jitter, and thermal throttling routinely
swings 10-30% between two runs of the *same* code. Two failure modes follow
from trusting a single run:

- **False positive:** a lucky run makes an unchanged (or even slower)
  version look like a win.
- **Silent correctness regression:** a change that's genuinely faster
  because it does less work — skips an edge case, truncates precision,
  changes behavior at a boundary — ships as a "speedup" because nobody
  checked the output matched.

This harness forces three things to happen, in order, before a speedup is
allowed to be called real:

1. **Correctness first.** Baseline and candidate outputs are compared
   before any timing is trusted. A fast wrong answer is rejected outright,
   regardless of how good the numbers look.
2. **Interleaved trials, not sequential blocks.** The harness alternates
   baseline/optimized calls (A/B/A/B/...) rather than timing all of one
   arm and then all of the other. Slow drift over the run (thermal
   throttling, memory pressure, background load) then affects both arms
   roughly equally instead of biasing whichever ran second.
3. **A decision rule fixed before looking at the numbers.** A result only
   counts as "confirmed" if it clears both a minimum effect-size threshold
   (default 5%) *and* a statistical margin (Welch's t-statistic vs. a
   threshold) — not just "the mean was lower."

## What it produces

Every comparison renders as a **finding**: a one-line technique
claim tagged with a confidence tier and the measurement that backs it,
rather than a bare assertion that something is faster.

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

Confidence tiers:

| Tier | Meaning |
|---|---|
| `fail` | Correctness check failed. Timing is irrelevant. |
| `noise` | Correct, but effect size is below the minimum threshold — noise-level. |
| `marginal` | Effect size clears the threshold, but isn't statistically separated from trial noise at this sample size — worth a second look with more trials, not yet worth acting on. |
| `confirmed` | Clears both the effect-size threshold and the statistical margin. |

The tiering exists so a growing `findings.md` log stays honest: a claim
that later turns out to be noise is visibly downgraded rather than quietly
forgotten, and a claim that hasn't been re-checked in a while can be told
apart from one that was actually confirmed.

## Usage

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

`example_moving_average.py` is a fully worked, dependency-light example
(naive O(N·W) sliding-sum loop vs. an O(N) cumsum-based implementation) —
deliberately generic so the harness itself, not the target op, is what's
on display. Swap in whatever real comparison you're running; the harness
doesn't change.

```
python3 example_moving_average.py
```

## Why this shape

Most "we made X faster" claims in practice fail for one of two reasons:
nobody checked correctness, or nobody looked at variance before declaring
victory. Both are cheap to prevent and expensive to discover after the
fact (a regression that ships because a benchmark was gamed, or a
"speedup" that evaporates the next time someone reruns it). This harness
is intentionally small — the point isn't the implementation, it's making
the judgment call ("is this real, and how sure am I") explicit and
repeatable instead of ad hoc.
