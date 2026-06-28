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
- warmup: `20`
- measured iterations: `100`

## Aggregate Timing

| Metric | Mean |
| --- | ---: |
| Forward + backward | 235.239 ms |
| Gradient communication | 45.578 ms |
| Optimizer step | 105.564 ms |
| Total training step | 386.385 ms |
| Communication fraction | 11.786% |

## Per-rank Consistency

| Metric | Rank 0 | Rank 1 |
| --- | ---: | ---: |
| Mean total step time | 386.499 ms | 386.271 ms |
| Mean communication time | 46.051 ms | 45.104 ms |
| Mean communication fraction | 11.877% | 11.696% |

## Key Takeaways

- In this `1`-node, `2`-GPU, `XL` configuration, naive DDP spends about `11.8%` of each training step in explicit post-backward gradient synchronization.
- The largest portion of the step is still local model computation: `forward + backward` accounts for about `60.9%` of the total time, while `optimizer.step()` contributes about `27.3%`.
- The two ranks are closely matched, so the result does not show evidence of a rank skew or synchronization bug.
- The reported communication time measures the explicit gradient synchronization phase after `loss.backward()` and before `optimizer.step()`, not one-time setup costs such as the initial parameter broadcast.
