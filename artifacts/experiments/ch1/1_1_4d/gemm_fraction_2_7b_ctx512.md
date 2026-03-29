## 1.1.4(d) GEMM Fraction Summary (`2.7b`, `ctx=512`)

This table is generated from saved `CUDA GPU Kernel Summary` text files and classifies a kernel as GEMM if its `Name` contains `gemm`, `xmma`, or `cutlass`.

| Trace | GEMM share (%) | Non-GEMM share (%) |
| --- | ---: | ---: |
| Forward-only | 92.20 | 7.80 |
| Train-step | 83.90 | 15.90 |

Conclusion:

For this representative configuration, matrix multiplication still dominates the full training step, but its share drops from about `92.20%` in forward-only inference to about `83.90%` once backward and AdamW are included. The remaining share is largely taken by non-GEMM ATen elementwise, vectorized-elementwise, and reduction kernels that become more visible during gradient computation and optimizer updates.

Source files:

- `artifacts/experiments/ch1/1_1_4b/2.7b_ctx512_forward_cuda_gpu_kern_sum.txt`
- `artifacts/experiments/ch1/1_1_4b/2.7b_ctx512_train_step_cuda_gpu_kern_sum.txt`
- `scripts/summarize_1_1_4d_gemm_fraction.py`
