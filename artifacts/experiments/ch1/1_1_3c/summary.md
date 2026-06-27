# 1.1.3(c) Warmup Sensitivity

Configuration:

- hardware: `single NVIDIA H100 80GB HBM3`
- context length: `128`
- batch size: `4`
- vocabulary size: `10,000`
- precision: `fp32`
- compared warmup steps: `0`, `2`, `5`
- measurement steps: `10`

Warmup = `0`:

Context length: 128
Batch size: 4
Precision: fp32
Warmup steps: 0
Measurement steps: 10

| Model size | Forward mean (ms) | Forward std (ms) | Backward mean (ms) | Backward std (ms) | Total mean (ms) | Total std (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| small | 70.824 | 138.522 | 28.828 | 35.125 | 99.652 | 173.232 |
| medium | 92.084 | 171.811 | 46.389 | 35.481 | 138.473 | 207.290 |
| large | 111.305 | 174.342 | 67.285 | 41.970 | 178.591 | 216.301 |
| xl | 132.240 | 179.263 | 86.227 | 44.695 | 218.467 | 223.941 |
| 2.7b | 106.683 | 174.661 | 94.627 | 30.276 | 201.310 | 204.937 |

Warmup = `2`:

Context length: 128
Batch size: 4
Precision: fp32
Warmup steps: 2
Measurement steps: 10

| Model size | Forward mean (ms) | Forward std (ms) | Backward mean (ms) | Backward std (ms) | Total mean (ms) | Total std (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| small | 28.363 | 30.559 | 17.775 | 0.708 | 46.139 | 31.237 |
| medium | 37.254 | 0.673 | 35.601 | 1.503 | 72.854 | 1.912 |
| large | 57.941 | 1.911 | 57.496 | 3.606 | 115.437 | 5.261 |
| xl | 76.337 | 1.308 | 74.071 | 2.026 | 150.408 | 3.075 |
| 2.7b | 51.515 | 1.342 | 85.308 | 0.146 | 136.822 | 1.409 |

Warmup = `5` baseline:

- See [`1_1_3b/summary.md`](../1_1_3b/summary.md).

Takeaway:

- Without warmup, the first measured iteration absorbs one-time startup costs and inflates variance.
- Two warmup steps already brings measurements close to the five-warmup baseline, but small lazy-initialization and run-to-run effects remain.
