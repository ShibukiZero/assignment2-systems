## 1.1.4(e) Runtime vs FLOPs for Attention Core

Representative configuration:

- model size: `2.7b`
- batch size: `4`
- context length: `512`
- profile: `forward-only`

Runtime evidence from `nvtx_gpu_proj_sum`:

| Operation | Range instances | Proj Avg (ns) | Total Proj Time (ns) |
| --- | ---: | ---: | ---: |
| attention_scores_matmul | 480 | 380,086.8 | 182,441,686 |
| attention_softmax | 480 | 626,460.6 | 300,701,099 |
| attention_value_matmul | 480 | 329,561.3 | 158,189,405 |

FLOP evidence from `scripts/calc_1_1_4e_attention_flops.py`:

| Operation | Per-layer FLOPs | Ratio vs softmax |
| --- | ---: | ---: |
| attention_scores_matmul | 5,368,709,120 | 22.87x |
| attention_softmax | 234,749,952 | 1.00x |
| attention_value_matmul | 5,368,709,120 | 22.87x |
| two matmuls combined | 10,737,418,240 | 45.74x |

Takeaway:

- `attention_softmax` is slower than either individual matmul, and remains on the same order as the two matmuls together.
- The FLOP gap is much larger than the runtime gap: each matmul has about `22.87x` the FLOPs of softmax, while softmax still takes comparable runtime.
- This supports the standard interpretation that softmax is more memory- and reduction-bound, whereas the matrix multiplications benefit more from highly optimized GEMM kernels.
