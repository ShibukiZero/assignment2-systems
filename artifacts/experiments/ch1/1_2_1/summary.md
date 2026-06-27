# 1.2.1 Benchmarking PyTorch Attention

Configuration:

- hardware: `single NVIDIA H100 80GB HBM3`
- implementation: `eager`
- precision: `fp32`
- batch size: `8`
- embedding dimensions: `16, 32, 64, 128`
- sequence lengths: `256, 1024, 4096, 8192, 16384`
- warmup steps: `5`
- measurement steps: `100`

Results:

| d_model | Sequence length | Forward timing (ms) | Backward timing (ms) | Saved for backward (GiB) | Status |
| --- | --- | ---: | ---: | ---: | --- |
| 16 | 256 | 0.213 | 0.552 | 0.004 | success |
| 16 | 1024 | 0.250 | 0.738 | 0.063 | success |
| 16 | 4096 | 2.444 | 6.065 | 1.002 | success |
| 16 | 8192 | 9.527 | 22.917 | 4.005 | success |
| 16 | 16384 | 37.306 | 90.627 | 16.009 | success |
| 32 | 256 | 0.234 | 0.731 | 0.004 | success |
| 32 | 1024 | 0.272 | 0.778 | 0.064 | success |
| 32 | 4096 | 2.519 | 6.228 | 1.004 | success |
| 32 | 8192 | 9.574 | 22.905 | 4.009 | success |
| 32 | 16384 | 37.377 | 90.560 | 16.017 | success |
| 64 | 256 | 0.217 | 0.556 | 0.004 | success |
| 64 | 1024 | 0.249 | 0.731 | 0.065 | success |
| 64 | 4096 | 2.479 | 6.085 | 1.008 | success |
| 64 | 8192 | 9.568 | 23.046 | 4.016 | success |
| 64 | 16384 | 37.364 | 90.499 | 16.033 | success |
| 128 | 256 | 0.238 | 0.582 | 0.005 | success |
| 128 | 1024 | 0.276 | 0.794 | 0.066 | success |
| 128 | 4096 | 2.521 | 6.229 | 1.016 | success |
| 128 | 8192 | 9.763 | 23.332 | 4.032 | success |
| 128 | 16384 | 38.153 | 91.563 | 16.064 | success |

Key takeaways:

- No OOM was observed within the tested range on the H100 80GB run.
- The saved-for-backward memory scales approximately as `T^2` and is nearly independent of `d_model`.
- This motivates tiled attention with online softmax and recomputation rather than explicitly materializing the full attention matrix.
