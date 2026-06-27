## 1.1.4(b) Representative Kernel-Summary Readout (`2.7b`, `ctx=512`)

This note records the representative `nsys stats` observations currently used to answer `1.1.4(b)`.

Forward-only trace:

- Source profile: `.agents/logs/1_1_4_forward_attention/2.7b_ctx512_forward_attention.nsys-rep`
- Report used: `cuda_gpu_kern_sum` and `cuda_gpu_kern_sum:nvtx-name:base`
- Top cumulative kernel:
  - `sm90_xmma_gemm_f32f32_tf32f32_f32_tn_n_tilesize128x128x32_...`
- Total instances in the trace:
  - `3375`
- Since the trace contains `5` warmup passes and `10` measured passes, this corresponds to about:
  - `3375 / 15 = 225` invocations per forward pass

Training-step trace:

- Source profile: `.agents/logs/1_1_4_train_step_attention/2.7b_ctx512_train_step_attention.nsys-rep`
- Report used: `cuda_gpu_kern_sum` and `cuda_gpu_kern_sum:nvtx-name:base`
- Top cumulative kernel:
  - `optimizer_step/vectorized_elementwise_kernel`

Conclusion:

For this representative H100 configuration, the forward-only CUDA GPU Kernel Summary is dominated by the Tensor Core GEMM kernel `sm90_xmma_gemm_f32f32_tf32f32_f32_tn_n_tilesize128x128x32_...`, which appears 3375 times across 15 profiled forward passes, or about 225 times per forward pass. In the full training-step CUDA GPU Kernel Summary, the largest cumulative entry is optimizer-related `vectorized_elementwise_kernel` time, so the top kernel is no longer the same once backward and AdamW are included.

Evidence source:

- `2.7b_ctx512_forward_cuda_gpu_kern_sum.txt`
- `2.7b_ctx512_train_step_cuda_gpu_kern_sum_nvtx.txt`
