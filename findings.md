## Replace naive O(N*W) sliding-sum loop with O(N) cumsum-based moving average
- target: moving average, N=4000, window=50
- confidence: confirmed
- correctness: pass (max abs diff = 2.89e-14, rtol=1e-9)
- baseline: 15.5153 ms +/- 0.7876 ms (n=25)
- optimized: 0.1118 ms +/- 0.0299 ms (n=25)
- speedup: 99.3% (t=97.72)
- decision_rule: correctness gate; min_speedup_pct=5.0; t_threshold=2.0 (Welch's t, 25 interleaved trials)
- source: example_moving_average.py, run locally, no external deps beyond numpy
