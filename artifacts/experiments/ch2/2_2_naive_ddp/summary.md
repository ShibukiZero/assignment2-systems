# Naive DDP Benchmarking

Source:
- Raw benchmark payload: `artifacts/experiments/ch2/2_2_naive_ddp/timer_xl_ctx128_nccl_w2_gbs8_fp32.json`

Setup:
- single-node `2`-GPU run
- backend: `NCCL`
- model size: `XL`
- context length: `128`
- global batch size: `8`
- precision: `fp32`
- warmup: `5`
- measured iterations: `20`

## Aggregate Timing

| Metric | Mean |
| --- | ---: |
| Forward + backward | 313.364 ms |
| Gradient communication | 40.526 ms |
| Optimizer step | 91.821 ms |
| Total training step | 445.714 ms |
| Communication fraction | 9.091% |

## Per-rank Consistency

| Metric | Rank 0 | Rank 1 |
| --- | ---: | ---: |
| Mean total step time | 445.430 ms | 445.998 ms |
| Mean communication time | 40.311 ms | 40.741 ms |
| Mean communication fraction | 9.048% | 9.135% |

## Key Takeaways

- In this `1`-node, `2`-GPU, `XL` configuration, naive DDP spends about `9.1%` of each training step in explicit post-backward gradient synchronization.
- The largest portion of the step is still local model computation: `forward + backward` accounts for about `70.3%` of the total time, while `optimizer.step()` contributes about `20.6%`.
- The two ranks are closely matched, so the result does not show evidence of a rank skew or synchronization bug.
- The reported communication time measures the explicit gradient synchronization phase after `loss.backward()` and before `optimizer.step()`, not one-time setup costs such as the initial parameter broadcast.
