# Optimizer State Sharding

Source files:
- `artifacts/experiments/ch3/3_1_optimizer_state_sharding/memory_full_xl_ctx128_nccl_w2_gbs8_fp32.json`
- `artifacts/experiments/ch3/3_1_optimizer_state_sharding/memory_sharded_xl_ctx128_nccl_w2_gbs8_fp32.json`
- `artifacts/experiments/ch3/3_1_optimizer_state_sharding/timer_full_xl_ctx128_nccl_w2_gbs8_fp32.json`
- `artifacts/experiments/ch3/3_1_optimizer_state_sharding/timer_sharded_xl_ctx128_nccl_w2_gbs8_fp32.json`
- `artifacts/experiments/ch3/3_1_optimizer_state_sharding/formula_xl_w2.json`

Setup:
- single-node `2`-GPU run
- backend: `NCCL`
- model size: `XL`
- context length: `128`
- global batch size: `8`
- precision: `fp32`
- timing warmup: `10`
- timing measured iterations: `50`

## Peak Memory

| Checkpoint | Full optimizer | Sharded optimizer |
| --- | ---: | ---: |
| After model initialization | `7.62 GiB` | `7.62 GiB` |
| Before optimizer step | `15.26 GiB` | `15.26 GiB` |
| After optimizer step | `30.49 GiB` | `22.75-22.99 GiB` |

The measured values are close to the simple formulas in `formula_xl_w2.json` for `P = 1,998,235,200` parameters: `4P = 7.44 GiB`, `8P = 14.89 GiB`, `16P = 29.78 GiB`, and `(8 + 8 / N)P = 22.33 GiB` for `N = 2`.

## Timing

| Metric | Full optimizer | Sharded optimizer |
| --- | ---: | ---: |
| Forward + backward | `355.36 ms` | `356.01 ms` |
| Optimizer step | `92.31 ms` | `79.14 ms` |
| Total step | `447.67 ms` | `435.15 ms` |

Interpretation:

- Optimizer state sharding leaves forward and backward time essentially unchanged in this configuration.
- The main runtime difference is a cheaper optimizer step, because each rank updates and stores only its local optimizer shard.
- End-to-end step time improves modestly (`447.67 -> 435.15 ms`, about `2.8%`), while post-step memory drops by roughly `7.5 GiB` per GPU.
