## 1.1.3(b) Forward / Backward Benchmarking

Configuration:

- context length: `128`
- batch size: `4`
- vocabulary size: `10,000`
- precision: `fp32`
- warmup steps: `5`
- measurement steps: `10`

| Model size | Forward mean (ms) | Forward std (ms) | Backward mean (ms) | Backward std (ms) | Total mean (ms) | Total std (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| small | 21.837 | 0.057 | 21.164 | 0.039 | 43.001 | 0.076 |
| medium | 42.146 | 0.455 | 50.833 | 0.047 | 92.979 | 0.472 |
| large | 62.412 | 1.055 | 117.320 | 0.288 | 179.732 | 1.074 |
| xl | 101.257 | 0.124 | 213.640 | 0.122 | 314.898 | 0.168 |
| 2.7b | 158.532 | 0.209 | 316.248 | 0.187 | 474.780 | 0.315 |

Takeaway:

- Forward and backward latency both grow monotonically with model size.
- The standard deviations remain small relative to the means after warmup, so the benchmark is stable enough to use as a baseline for later Nsight comparisons.
