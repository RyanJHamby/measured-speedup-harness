# profiles/

Captured output from running this repo's own tooling against itself, kept
as concrete evidence rather than a description of what the tools produce.

- `baseline.speedscope.json`, `optimized.speedscope.json` — captured with:
  ```
  sudo python3 profile.py example_moving_average --which baseline --out profiles/baseline.speedscope.json
  sudo python3 profile.py example_moving_average --which optimized --out profiles/optimized.speedscope.json
  ```
  `py-spy` requires root on macOS to attach to a process, even one it
  launches itself, so these can't be regenerated from an unattended script
  - see the `sudo` invocations above. View them at
  [speedscope.app](https://www.speedscope.app/) (drag the file in, or use
  the `#profileURL=` links in the main README to load them straight from
  GitHub with full interactivity).

  What they show: 99.7% of baseline samples land in `moving_average_naive`
  itself. In the optimized profile, only 13.2% of samples are in
  `moving_average_vectorized` - the majority is numpy's internal
  argument-dispatch overhead for `np.insert`, not the `cumsum` arithmetic.
  See the main README's profiling section for the full breakdown and what
  it implies about where a further optimization would need to go.
