## 1.1.4(a) Forward Timing Comparison at Context Length 128

This table compares the Python timing emitted by the profiling sweep against the Nsight Systems NVTX-projected forward mean, using only the `measure_*` iterations from the NVTX export.

| Model size | Context length | Benchmark forward (ms) | nsys NVTX forward (ms) | Abs diff (ms) | Rel diff (%) |
| --- | ---: | ---: | ---: | ---: | ---: |
| small | 128 | 25.596 | 25.245 | 0.351 | 1.370 |
| medium | 128 | 50.119 | 49.488 | 0.631 | 1.259 |
| large | 128 | 76.889 | 75.931 | 0.958 | 1.246 |
| xl | 128 | 102.430 | 101.033 | 1.398 | 1.365 |
| 2.7b | 128 | 159.746 | 158.597 | 1.149 | 0.719 |

Conclusion:

At context length 128, the profiling sweep's Python timing and the Nsight Systems NVTX-projected timing agree closely. However, this is not identical to the earlier `1.1.3(b)` benchmark baseline: compared with that baseline, the `xl` and `2.7b` numbers remain very close, while the `small`, `medium`, and `large` runs show a more noticeable gap, so the matching claim should be stated more cautiously in the writeup.

Source files:

- Benchmark JSON logs: `.agents/logs/1_1_4_forward_attention/*_ctx128_forward_attention.json`
- Nsight stats export: `.agents/logs/1_1_4_forward_attention/stats/*_ctx128_forward_attention/nvtx_gpu_proj_sum_nvtx_gpu_proj_sum.csv`
- Comparison table generator: `scripts/compare_1_1_4a_forward_times.py`
