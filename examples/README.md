# examples/

Captured output from running this repo's own tooling against itself, kept
as concrete evidence rather than a description of what the tools produce.

- `baseline_flamegraph.svg` — captured with:
  ```
  sudo python3 profile.py example_moving_average --which baseline --seconds 2 --out baseline_flamegraph.svg
  ```
  Open it in a browser (it's interactive - click to zoom, `/` to search).
  208 samples, ~100% inside `moving_average_naive`, split between the loop
  header and the `sum()` call on each window slice - confirms the naive
  implementation's cost is exactly where you'd expect before optimizing it,
  rather than assuming.

  `py-spy` requires root on macOS to attach to a process, even one it
  launches itself, so this one can't be regenerated from an unattended
  script - see the `sudo` invocation above.
