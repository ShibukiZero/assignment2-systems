# 1.1.3(b) Forward / Backward Benchmarking

Configuration:

- hardware: `single NVIDIA H100 80GB HBM3`
- context length: `128`
- batch size: `4`
- vocabulary size: `10,000`
- precision: `fp32`
- warmup steps: `5`
- measurement steps: `10`

Context length: 128
Batch size: 4
Precision: fp32
Warmup steps: 5
Measurement steps: 10

| Model size | Forward mean (ms) | Forward std (ms) | Backward mean (ms) | Backward std (ms) | Total mean (ms) | Total std (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| small | 18.470 | 0.433 | 16.676 | 0.165 | 35.147 | 0.409 |
| medium | 38.455 | 1.708 | 35.298 | 0.805 | 73.753 | 2.197 |
| large | 56.086 | 3.453 | 56.432 | 7.487 | 112.517 | 10.566 |
| xl | 74.703 | 1.190 | 73.215 | 1.053 | 147.919 | 2.049 |
| 2.7b | 51.443 | 2.963 | 85.604 | 0.288 | 137.047 | 3.102 |

Takeaway:

- Forward and backward latency both grow with model size, and the standard deviations remain small after warmup.
