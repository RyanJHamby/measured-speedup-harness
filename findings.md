## Replace naive O(N*W) sliding-sum loop with O(N) cumsum-based moving average
- target: moving average, N=4000, window=50
- confidence: confirmed
- correctness: pass (max abs diff = 2.89e-14, rtol=1e-6)
- baseline: 14.7356 ms +/- 0.3006 ms (n=25)
- optimized: 0.0393 ms +/- 0.0138 ms (n=25)
- speedup: 99.7% (t=244.17), 95% CI [99.7%, 99.8%]
- decision_rule: correctness gate; min_speedup_pct=5.0; t_threshold=2.0 (Welch's t, 25 interleaved trials)
- source: example_moving_average.py, run locally, no external deps beyond numpy

## Replace naive triple-loop matmul with BLAS-backed np.matmul
- target: matrix multiply, 40x40 @ 40x40
- confidence: confirmed
- correctness: pass (max abs diff = 5.33e-15, rtol=1e-8)
- baseline: 15.9139 ms +/- 0.3027 ms (n=25)
- optimized: 0.0052 ms +/- 0.0048 ms (n=25)
- speedup: 100.0% (t=262.78), 95% CI [100.0%, 100.0%]
- decision_rule: correctness gate; min_speedup_pct=5.0; t_threshold=2.0 (Welch's t, 25 interleaved trials)
- source: example_matmul.py, run locally, no external deps beyond numpy

