# 1.2.1 Benchmarking PyTorch Attention

Configuration:

- implementation: `eager`
- precision: `fp32`
- batch size: `8`
- embedding dimensions: `16, 32, 64, 128`
- sequence lengths: `256, 1024, 4096, 8192, 16384`

Results:

| d_model | Sequence length | Forward timing (ms) | Backward timing (ms) | Saved for backward (GiB) | Status |
| --- | --- | ---: | ---: | ---: | --- |
| 16 | 256 | 0.262 | 0.605 | 0.004 | success |
| 16 | 1024 | 0.323 | 0.849 | 0.063 | success |
| 16 | 4096 | 3.298 | 7.581 | 1.002 | success |
| 16 | 8192 | 13.383 | 31.941 | 4.005 | success |
| 16 | 16384 | 51.331 | 121.836 | 16.009 | success |
| 32 | 256 | 0.244 | 0.562 | 0.004 | success |
| 32 | 1024 | 0.339 | 0.866 | 0.064 | success |
| 32 | 4096 | 3.469 | 7.728 | 1.004 | success |
| 32 | 8192 | 14.281 | 29.686 | 4.009 | success |
| 32 | 16384 | 55.450 | 115.465 | 16.017 | success |
| 64 | 256 | 0.246 | 0.552 | 0.004 | success |
| 64 | 1024 | 0.366 | 0.918 | 0.065 | success |
| 64 | 4096 | 4.009 | 8.757 | 1.008 | success |
| 64 | 8192 | 16.264 | 33.791 | 4.016 | success |
| 64 | 16384 | 61.564 | 128.291 | 16.033 | success |
| 128 | 256 | 0.269 | 0.659 | 0.005 | success |
| 128 | 1024 | 0.449 | 1.117 | 0.066 | success |
| 128 | 4096 | 5.298 | 11.448 | 1.016 | success |
| 128 | 8192 | 20.726 | 42.572 | 4.032 | success |
| 128 | 16384 | 79.873 | 166.429 | 16.064 | success |

Key takeaways:

- No OOM was observed within the tested range.
- The dominant saved-for-backward term scales approximately as `T^2` and is nearly independent of `d_model`.
- The measured saved-for-backward memory is consistent with storing about two FP32 tensors of shape `(B, T, T)`.
- This motivates tiled attention with online softmax and recomputation rather than explicitly materializing the full attention matrix.
