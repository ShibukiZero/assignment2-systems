# Distributed Communication on a Single Node

Source:
- Raw benchmark payload: `artifacts/experiments/ch2/2_1_1/results.json`

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
| 2 | 2.849 | 16.692 | 170.475 | 1440.573 |
| 4 | 1.199 | 10.931 | 149.470 | 1483.266 |
| 6 | 2.130 | 17.344 | 170.018 | 1602.531 |

### NCCL + GPU

| Processes | 1 MB (ms) | 10 MB (ms) | 100 MB (ms) | 1 GB (ms) |
| --- | ---: | ---: | ---: | ---: |
| 2 | 0.124 | 0.128 | 0.440 | 3.251 |
| 4 | 0.128 | 0.133 | 0.554 | 4.560 |
| 6 | 0.164 | 0.207 | 0.545 | 4.262 |

Measured on a single-node 6x H100 80GB SXM (NVLink) instance.

## Key Takeaways

- `NCCL + GPU` is consistently much faster than `Gloo + CPU`, and the advantage becomes especially large for larger messages (about `440x` at `1 GB` / `2` processes: `1441 ms` vs `3.25 ms`). NCCL runs the collective directly over NVLink, while Gloo is host-bound.
- Communication latency rises with tensor size for both backends. For NCCL, small messages (`1-10 MB`) are latency-bound and nearly flat at `0.1-0.2 ms`, while `1 GB` is bandwidth-bound at `3-4.5 ms`.
- Increasing the NCCL world size raises latency only modestly (e.g. `1 GB`: `3.25 / 4.56 / 4.26 ms` for `2/4/6`), consistent with ring all-reduce moving a per-rank data volume roughly independent of world size.
- Cross-rank means are tightly clustered in every configuration, so there is no sign of a systematic rank skew or synchronization bug.
- The `Gloo + CPU` small-message numbers (`1-10 MB`) are noisy and not cleanly monotonic in process count, reflecting per-call launch overhead and CPU scheduling jitter rather than a bandwidth trend; the large-message `Gloo` numbers are well-behaved.
