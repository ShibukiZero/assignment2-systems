# Section 2.3.3(a): Bucketed DDP Benchmark

Configuration:

- `1 node x 2 GPUs`
- `XL` model size
- context length `128`
- global batch size `8`
- `NCCL`
- `FP32`

Measured step times:

| Bucket size (MB) | Total step time (ms) | Communication tail (ms) |
| --- | ---: | ---: |
| 1 | 423.133 | 6.181 |
| 10 | 422.614 | 6.299 |
| 100 | 433.551 | 11.167 |
| 1000 | 429.040 | 11.608 |

Comparison to previous baselines:

| Implementation | Total step time (ms) |
| --- | ---: |
| Naive per-parameter DDP | 445.923 |
| Single flattened all-reduce | 446.389 |
| Overlapped per-parameter DDP | 420.660 |
| Bucketed DDP, 1 MB | 423.133 |
| Bucketed DDP, 10 MB | 422.614 |
| Bucketed DDP, 100 MB | 433.551 |
| Bucketed DDP, 1000 MB | 429.040 |

Interpretation:

Bucketed DDP clearly improved over the naive and flat baselines, but it did not surpass the overlapped per-parameter implementation. The best result came from a medium-sized bucket (`10 MB`), which suggests a tradeoff between communication-call overhead for very small buckets and reduced overlap for very large buckets. In this simple implementation, the reduction in collective-call overhead is not enough to overcome the extra gradient packing and device-copy overhead introduced by bucketing, so the overall gain is limited. I would expect the bucketing story to look stronger for a larger or more communication-bound setup, or with a more optimized bucket implementation that avoids most of the pack/unpack overhead.

Profiling note:

The accompanying screenshots (`1mb.png`, `10mb.png`, `100mb.png`, `1000mb.png`) are consistent with this explanation. Smaller buckets show more frequent but earlier communication activity, while larger buckets show fewer but later communication regions and therefore less effective overlap. Since this implementation still performs explicit packing and copying around bucket communication, the reduction in collective-call overhead does not translate into a proportional end-to-end speedup.
