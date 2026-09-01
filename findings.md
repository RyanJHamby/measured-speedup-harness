Suite summary: 2 confirmed, 2 noise
4/4 significant after Benjamini-Hochberg FDR correction (alpha=0.05) across 4 comparisons

| scenario | tier | speedup | correctness | sig. after FDR correction |
|---|---|---|---|---|
| example_moving_average | confirmed | 99.7% | pass | yes |
| example_matmul | confirmed | 99.9% | pass | yes |
| example_variance | noise | -209.5% | pass | yes |
| example_softmax | noise | -16.8% | pass | yes |
