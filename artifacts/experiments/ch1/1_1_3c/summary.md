## 1.1.3(c) Warmup Sensitivity

Configuration:

- context length: `128`
- batch size: `4`
- vocabulary size: `10,000`
- precision: `fp32`
- compared warmup steps: `0`, `2`, `5`
- measurement steps: `10`

Warmup = `0`:

| Model size | Forward mean (ms) | Forward std (ms) | Backward mean (ms) | Backward std (ms) | Total mean (ms) | Total std (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| small | 61.952 | 127.593 | 36.451 | 49.355 | 98.403 | 176.948 |
| medium | 85.303 | 134.941 | 65.436 | 44.869 | 150.739 | 179.809 |
| large | 107.603 | 142.390 | 130.120 | 40.883 | 237.724 | 183.273 |
| xl | 147.018 | 144.945 | 226.578 | 41.748 | 373.596 | 186.694 |
| 2.7b | 196.757 | 121.461 | 327.541 | 35.854 | 524.298 | 157.315 |

Warmup = `2`:

| Model size | Forward mean (ms) | Forward std (ms) | Backward mean (ms) | Backward std (ms) | Total mean (ms) | Total std (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| small | 21.530 | 0.050 | 20.920 | 0.117 | 42.450 | 0.109 |
| medium | 42.929 | 1.211 | 50.888 | 0.076 | 93.817 | 1.262 |
| large | 63.079 | 0.172 | 117.294 | 0.337 | 180.373 | 0.266 |
| xl | 102.325 | 2.679 | 213.560 | 0.186 | 315.885 | 2.657 |
| 2.7b | 158.240 | 0.158 | 316.118 | 0.090 | 474.358 | 0.176 |

Warmup = `5` baseline:

- See [`1_1_3b/summary.md`](/Users/linzihan/Github/assignment2-systems/artifacts/experiments/ch1/1_1_3b/summary.md).

Takeaway:

- Without warmup, the first measured iteration absorbs one-time startup costs, which inflates both the means and the standard deviations.
- Two warmup steps already bring the measurements much closer to the `warmup = 5` baseline, but small differences remain because some lazy initialization and normal run-to-run noise are still present.
