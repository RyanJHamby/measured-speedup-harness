## Replace naive O(N*W) sliding-sum loop with O(N) cumsum-based moving average
- target: moving average, N=4000, window=50
- confidence: confirmed
- correctness: pass (max abs diff = 2.89e-14, rtol=1e-9)
- baseline: 14.9577 ms +/- 0.2682 ms (n=25)
- optimized: 0.0354 ms +/- 0.0047 ms (n=25)
- speedup: 99.8% (t=278.11)
- decision_rule: correctness gate; min_speedup_pct=5.0; t_threshold=2.0 (Welch's t, 25 interleaved trials)
- source: example_moving_average.py, run locally, no external deps beyond numpy
