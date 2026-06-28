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
| 1 | 347.945 | 9.45 |
| 10 | 337.303 | 7.13 |
| 100 | 352.237 | 13.99 |
| 1000 | 377.051 | 21.26 |

Comparison to previous baselines:

| Implementation | Total step time (ms) |
| --- | ---: |
| Naive per-parameter DDP | 368.152 |
| Single flattened all-reduce | 377.222 |
| Overlapped per-parameter DDP | 371.354 |
| Bucketed DDP, 1 MB | 347.945 |
| Bucketed DDP, 10 MB | 337.303 |
| Bucketed DDP, 100 MB | 352.237 |
| Bucketed DDP, 1000 MB | 377.051 |

Interpretation:

The best result is the `10 MB` bucket (`337.3 ms`). The post-backward communication tail grows with bucket size (about `9.45`, `7.13`, `13.99`, and `21.26 ms` for `1`, `10`, `100`, and `1000 MB` respectively), so larger buckets become ready later and overlap less. The `10 MB` bucket is faster than the naive (`368.2 ms`), flattened (`377.2 ms`), and overlapped (`371.4 ms`) baselines, consistent with bucketing both overlapping communication and reducing collective-call overhead; very small (`1 MB`) and very large (`1000 MB`) buckets are worse, matching the expected tradeoff. I would expect the bucketing story to look stronger for a larger or more communication-bound setup, or with a more optimized bucket implementation that avoids most of the pack/unpack overhead.

Profiling note:

The accompanying screenshots (`1mb.png`, `10mb.png`, `100mb.png`, `1000mb.png`) are consistent with this explanation. Smaller buckets show more frequent but earlier communication activity, while larger buckets show fewer but later communication regions and therefore less effective overlap. Since this implementation still performs explicit packing and copying around bucket communication, the reduction in collective-call overhead does not translate into a proportional end-to-end speedup.

# Section 2.3.3(b): Idealized Bucketed-Communication Model

Under the handout's simplifying assumption, each bucket contains `s / n_b` bytes and the payload communication time per bucket is `s / (n_b * w)`. In the ideal overlapped pipeline, the only payload communication that remains visible after backward is the final bucket, while each of the `n_b` communication calls still pays a fixed launch overhead `o`. This gives the idealized post-backward overhead model:

```text
T_overhead(n_b) = s / (n_b * w) + n_b * o
```

Setting the derivative to zero gives the optimal number of buckets:

```text
n_b* = sqrt(s / (w * o))
```

and therefore the corresponding optimal bucket size:

```text
b* = s / n_b* = sqrt(s * w * o)
```
