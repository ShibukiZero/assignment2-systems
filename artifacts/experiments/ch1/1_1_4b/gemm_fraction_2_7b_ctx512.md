## 1.1.4(d) GEMM Fraction Summary

| Trace | GEMM share (%) | Non-GEMM share (%) |
| --- | ---: | ---: |
| Forward-only | 92.20 | 7.80 |
| Train-step | 83.90 | 15.90 |

Classification rule:

- A kernel is counted as GEMM if its `Name` contains `gemm`, `xmma`, or `cutlass`.

Source files:

- Forward summary: `artifacts/experiments/ch1/1_1_4b/2.7b_ctx512_forward_cuda_gpu_kern_sum.txt`
- Train-step summary: `artifacts/experiments/ch1/1_1_4b/2.7b_ctx512_train_step_cuda_gpu_kern_sum.txt`
