# Flattened-Gradient DDP Benchmarking

Source:
- Individual-gradient baseline: `artifacts/experiments/ch2/2_3_1_flat_ddp/individual_baseline_xl_ctx128_nccl_w2_gbs8_fp32.json`
- Flattened-gradient run: `artifacts/experiments/ch2/2_3_1_flat_ddp/flat_xl_ctx128_nccl_w2_gbs8_fp32.json`
- Naive CUDA HW trace screenshot: `artifacts/experiments/ch2/2_3_1_flat_ddp/naive profiling.png`
- Flattened CUDA HW trace screenshot: `artifacts/experiments/ch2/2_3_1_flat_ddp/flat profiling.png`

Setup:
- single-node `2`-GPU run
- backend: `NCCL`
- model size: `XL`
- context length: `128`
- global batch size: `8`
- precision: `fp32`
- warmup: `20`
- measured iterations: `100`
- same benchmark script and timing configuration for both runs

## Aggregate Timing

| Metric | Individual all-reduce | Flattened all-reduce |
| --- | ---: | ---: |
| Forward + backward | 214.478 ms | 211.457 ms |
| Gradient communication | 41.531 ms | 40.515 ms |
| Optimizer step | 105.486 ms | 105.376 ms |
| Total training step | 361.499 ms | 357.351 ms |
| Communication fraction | 11.489% | 11.342% |

## Key Takeaways

- Flattening reduced measured communication time from `41.531 ms` to `40.515 ms`, a drop of about `1.0 ms` (`2.4%` relative).
- The communication fraction also fell from `11.49%` to `11.34%`.
- End-to-end step time was nearly unchanged (`361.499 ms` vs `357.351 ms`), which suggests that this configuration remains dominated by local computation rather than communication overhead.
- A plausible explanation is that reducing the number of NCCL calls helped the communication phase itself, but the flattened implementation still incurred extra gradient packing and unpacking work, limiting the net end-to-end gain.
- The Nsight Systems CUDA HW screenshots are consistent with that explanation: the flattened run has a shorter all-reduce region, but it is followed by extra memory-copy activity that is not present to the same extent in the naive baseline.
