## 1.1.4(c) Non-Matmul Forward Kernels

Representative configuration:

- model size: `2.7b`
- batch size: `4`
- context length: `512`
- profile: `forward-only`

Evidence source:

- `CUDA GPU Kernel Summary` for the representative forward trace
- raw summaries archived under [`1_1_4b/`](/Users/linzihan/Github/assignment2-systems/artifacts/experiments/ch1/1_1_4b)

Observed non-matmul kernel families with visible runtime:

- ATen `elementwise_kernel`
- ATen `vectorized_elementwise_kernel`
- ATen `reduce_kernel`

Takeaway:

- The forward pass is overwhelmingly dominated by GEMM kernels.
- Still, several non-matmul kernels remain visible in the CUDA GPU Kernel Summary, with individual contributions on the order of roughly `0.5%-1.5%` in the representative `2.7b`, context-length-512 forward trace.
