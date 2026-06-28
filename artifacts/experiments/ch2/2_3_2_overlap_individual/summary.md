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
- warmup: `20`
- measured iterations: `100`

## Aggregate Timing

| Metric | Naive individual | Flattened | Overlap individual |
| --- | ---: | ---: | ---: |
| Forward + backward | 219.553 ms | 224.650 ms | 254.608 ms |
| Communication tail | 43.112 ms | 47.132 ms | 11.104 ms |
| Optimizer step | 105.484 ms | 105.437 ms | 105.638 ms |
| Total training step | 368.152 ms | 377.222 ms | 371.354 ms |
| Communication fraction | 11.682% | 12.415% | 2.950% |

## Key Takeaways

- The overlap-individual implementation reduced the post-backward communication tail dramatically, from about `43.1 ms` to about `11.1 ms` (communication fraction `11.68%` -> `2.95%`), confirming that most gradient communication was successfully hidden under the backward pass.
- However, the per-parameter asynchronous all-reduce hooks added overhead to the backward pass itself (`forward + backward` grew from about `219.6 ms` to about `254.6 ms`), so the end-to-end step time was roughly unchanged (`368.152 ms` -> `371.354 ms`).
- On H100 with fast NVLink the communication is already cheap, so the latency hidden by overlap is offset by the hook overhead -- unlike slower-interconnect settings where overlap would yield a clear end-to-end speedup.
