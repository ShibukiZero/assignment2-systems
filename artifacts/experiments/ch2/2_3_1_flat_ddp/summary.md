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
- warmup: `5`
- measured iterations: `20`
- same benchmark script and timing configuration for both runs

## Aggregate Timing

| Metric | Individual all-reduce | Flattened all-reduce |
| --- | ---: | ---: |
| Forward + backward | 313.317 ms | 314.240 ms |
| Gradient communication | 40.545 ms | 39.197 ms |
| Optimizer step | 92.059 ms | 92.949 ms |
| Total training step | 445.923 ms | 446.389 ms |
| Communication fraction | 9.092% | 8.781% |

## Key Takeaways

- Flattening reduced measured communication time from `40.545 ms` to `39.197 ms`, a drop of about `1.35 ms` (`3.3%` relative).
- The communication fraction also fell from `9.09%` to `8.78%`.
- End-to-end step time was nearly unchanged (`445.923 ms` vs `446.389 ms`), which suggests that this configuration remains dominated by local computation rather than communication overhead.
- A plausible explanation is that reducing the number of NCCL calls helped the communication phase itself, but the flattened implementation still incurred extra gradient packing and unpacking work, limiting the net end-to-end gain.
- The Nsight Systems CUDA HW screenshots are consistent with that explanation: the flattened run has a shorter all-reduce region, but it is followed by extra memory-copy activity that is not present to the same extent in the naive baseline.
