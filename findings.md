## Replace naive O(N*W) sliding-sum loop with O(N) cumsum-based moving average
- target: moving average, N=4000, window=50
- confidence: confirmed
- correctness: pass (max abs diff = 2.89e-14, rtol=1e-6)
- baseline: 15.0532 ms +/- 0.2631 ms (n=25)
- optimized: 0.0344 ms +/- 0.0034 ms (n=25)
- speedup: 99.8% (t=285.45), 95% CI [99.8%, 99.8%]
- decision_rule: correctness gate; min_speedup_pct=5.0; t_threshold=2.0 (Welch's t, 25 interleaved trials)
- source: example_moving_average.py, run locally, no external deps beyond numpy

## Replace naive triple-loop matmul with BLAS-backed np.matmul
- target: matrix multiply, 40x40 @ 40x40
- confidence: confirmed
- correctness: pass (max abs diff = 5.33e-15, rtol=1e-8)
- baseline: 16.2833 ms +/- 0.2554 ms (n=25)
- optimized: 0.0043 ms +/- 0.0017 ms (n=25)
- speedup: 100.0% (t=318.68), 95% CI [100.0%, 100.0%]
- decision_rule: correctness gate; min_speedup_pct=5.0; t_threshold=2.0 (Welch's t, 25 interleaved trials)
- source: example_matmul.py, run locally, no external deps beyond numpy

