# Distributed Communication on a Single Node

Source:
- Raw benchmark payload: `.agents/logs/distributed_communication_single_node.json`

Setup:
- single-node multi-process `all_reduce`
- float32 tensors
- backends: `Gloo + CPU`, `NCCL + GPU`
- process counts: `2`, `4`, `6`
- tensor sizes: `1 MB`, `10 MB`, `100 MB`, `1 GB`
- warmup: `5`
- measured iterations: `20`
- aggregation: per-iteration timings collected across ranks

## Mean Latency by Backend

### Gloo + CPU

| Processes | 1 MB (ms) | 10 MB (ms) | 100 MB (ms) | 1 GB (ms) |
| --- | ---: | ---: | ---: | ---: |
| 2 | 0.442 | 3.942 | 73.296 | 909.977 |
| 4 | 0.784 | 8.017 | 149.262 | 1272.314 |
| 6 | 1.120 | 11.554 | 182.700 | 1626.451 |

### NCCL + GPU

| Processes | 1 MB (ms) | 10 MB (ms) | 100 MB (ms) | 1 GB (ms) |
| --- | ---: | ---: | ---: | ---: |
| 2 | 0.073 | 0.165 | 0.931 | 8.110 |
| 4 | 0.196 | 0.280 | 1.280 | 10.972 |
| 6 | 0.199 | 0.359 | 1.688 | 11.960 |

## Key Takeaways

- `NCCL + GPU` is consistently much faster than `Gloo + CPU`, and the advantage becomes especially large for larger messages.
- Communication latency rises with tensor size for both backends.
- Increasing the number of worker processes generally increases latency, especially for the CPU/Gloo runs.
- Cross-rank means are tightly clustered in nearly all configurations, so there is no sign of a systematic rank skew or synchronization bug.
- The noisiest configuration is `NCCL`, `6` processes, `100 MB`, where the aggregate standard deviation is elevated because of a small number of slow outlier iterations; however, the per-rank mean timings still remain close to `1.69 ms`.
