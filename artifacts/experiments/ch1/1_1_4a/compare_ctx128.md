## 1.1.4(a) Forward Timing Comparison at Context Length 128

This table compares the Python benchmark forward mean against the Nsight Systems NVTX-projected forward mean, using only the `measure_*` iterations so the comparison matches the benchmarking methodology from `1.1.3(b)`.

| Model size | Context length | Benchmark forward (ms) | nsys NVTX forward (ms) | Abs diff (ms) | Rel diff (%) |
| --- | ---: | ---: | ---: | ---: | ---: |
| small | 128 | 25.596 | 25.245 | 0.351 | 1.370 |
| medium | 128 | 50.119 | 49.488 | 0.631 | 1.259 |
| large | 128 | 76.889 | 75.931 | 0.958 | 1.246 |
| xl | 128 | 102.430 | 101.033 | 1.398 | 1.365 |
| 2.7b | 128 | 159.746 | 158.597 | 1.149 | 0.719 |

Conclusion:

The total forward-pass time reported by Nsight Systems closely matches the Python benchmark results. At context length 128, the relative difference stays within about 0.7%-1.4% across all five model sizes, so the profiler confirms the earlier benchmark timings up to small profiling and measurement overheads.

Source files:

- Benchmark JSON logs: `.agents/logs/1_1_4_forward_attention/*_ctx128_forward_attention.json`
- Nsight stats export: `.agents/logs/1_1_4_forward_attention/stats/*_ctx128_forward_attention/nvtx_gpu_proj_sum_nvtx_gpu_proj_sum.csv`
- Comparison table generator: `scripts/compare_1_1_4a_forward_times.py`
