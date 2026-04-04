# Overlapping Communication with Individual Parameter Gradients

Source:
- Naive individual-gradient baseline: `artifacts/experiments/ch2/2_3_2_overlap_individual/individual_baseline_xl_ctx128_nccl_w2_gbs8_fp32.json`
- Flattened-gradient baseline: `artifacts/experiments/ch2/2_3_2_overlap_individual/flat_baseline_xl_ctx128_nccl_w2_gbs8_fp32.json`
- Overlap-individual benchmark: `artifacts/experiments/ch2/2_3_2_overlap_individual/overlap_individual_xl_ctx128_nccl_w2_gbs8_fp32.json`

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

| Metric | Naive individual | Flattened | Overlap individual |
| --- | ---: | ---: | ---: |
| Forward + backward | 313.317 ms | 314.240 ms | 322.575 ms |
| Communication tail | 40.545 ms | 39.197 ms | 6.027 ms |
| Optimizer step | 92.059 ms | 92.949 ms | 92.055 ms |
| Total training step | 445.923 ms | 446.389 ms | 420.660 ms |
| Communication fraction | 9.092% | 8.781% | 1.433% |

## Key Takeaways

- The overlap-individual implementation reduced total step time to `420.660 ms`, compared with `445.923 ms` for the naive baseline and `446.389 ms` for the flattened baseline.
- Relative to the naive baseline, this is an improvement of about `25.26 ms` (`5.7%`).
- The measured post-backward communication tail shrank sharply from `40.545 ms` to `6.027 ms`, indicating that most communication was hidden under the backward pass rather than paid entirely at the end of the step.
- The `forward + backward` segment became slightly longer in the overlap run because communication launch and hook overhead moved into that time window, but the large reduction in the final synchronization tail still produced a clear end-to-end speedup.
