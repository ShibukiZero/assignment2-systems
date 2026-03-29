## 1.1.4(b) Representative Kernel-Summary Readout (`2.7b`, `ctx=512`)

This note records the representative `nsys stats` observations currently used to answer `1.1.4(b)`.

Forward-only trace:

- Source profile: `.agents/logs/1_1_4_forward_attention/2.7b_ctx512_forward_attention.nsys-rep`
- Report used: `cuda_gpu_kern_sum` and `cuda_gpu_kern_sum:nvtx-name:base`
- Top cumulative kernel:
  - `sm80_xmma_gemm_f32f32_f32f32_f32_tn_n_tilesize64x64x8_...`
- Total instances in the trace:
  - `975`
- Since the trace contains `5` warmup passes and `10` measured passes, this corresponds to about:
  - `975 / 15 = 65` invocations per forward pass

Training-step trace:

- Source profile: `.agents/logs/1_1_4_train_step_attention/2.7b_ctx512_train_step_attention.nsys-rep`
- Report used: `cuda_gpu_kern_sum` and `cuda_gpu_kern_sum:nvtx-name:base`
- Top cumulative kernel:
  - `sm80_xmma_gemm_f32f32_f32f32_f32_tn_n_tilesize128x64x8_...`

Conclusion:

For this representative configuration, the forward-only CUDA GPU Kernel Summary is dominated by the Tensor Core GEMM kernel `sm80_xmma_gemm_f32f32_f32f32_f32_tn_n_tilesize64x64x8_...`, which appears 975 times across 15 profiled forward passes, or about 65 times per forward pass. The full training-step CUDA GPU Kernel Summary is still dominated by a GEMM kernel, but the top entry changes to `sm80_xmma_gemm_f32f32_f32f32_f32_tn_n_tilesize128x64x8_...`, so it is not exactly the same kernel as in forward-only mode.

Evidence source:

- Remote `nsys stats` output copied into `.agents/logs/terminal.log`
