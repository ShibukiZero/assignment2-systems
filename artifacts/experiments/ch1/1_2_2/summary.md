# 1.2.2 torch.compile Attention Comparison

| d_model | Sequence length | Eager forward (ms) | Compiled forward (ms) | Forward speedup | Eager backward (ms) | Compiled backward (ms) | Backward speedup |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 256 | 0.257 | 0.212 | 1.21x | 0.605 | 0.507 | 1.19x |
| 16 | 1024 | 0.315 | 0.226 | 1.39x | 0.826 | 0.592 | 1.40x |
| 16 | 4096 | 3.291 | 1.568 | 2.10x | 7.520 | 3.486 | 2.16x |
| 16 | 8192 | 13.348 | 5.455 | 2.45x | 31.717 | 15.373 | 2.06x |
| 16 | 16384 | 51.325 | 21.015 | 2.44x | 121.785 | 57.350 | 2.12x |
| 32 | 256 | 0.266 | 0.216 | 1.23x | 0.604 | 0.495 | 1.22x |
| 32 | 1024 | 0.330 | 0.260 | 1.27x | 0.838 | 0.656 | 1.28x |
| 32 | 4096 | 3.458 | 1.712 | 2.02x | 7.692 | 3.592 | 2.14x |
| 32 | 8192 | 14.236 | 6.364 | 2.24x | 29.488 | 13.126 | 2.25x |
| 32 | 16384 | 55.150 | 25.263 | 2.18x | 115.017 | 51.274 | 2.24x |
| 64 | 256 | 0.245 | 0.261 | 0.94x | 0.564 | 0.604 | 0.93x |
| 64 | 1024 | 0.370 | 0.285 | 1.30x | 0.922 | 0.698 | 1.32x |
| 64 | 4096 | 3.986 | 2.255 | 1.77x | 8.716 | 4.672 | 1.87x |
| 64 | 8192 | 16.223 | 8.599 | 1.89x | 33.600 | 18.100 | 1.86x |
| 64 | 16384 | 61.485 | 31.538 | 1.95x | 128.266 | 64.758 | 1.98x |
| 128 | 256 | 0.261 | 0.248 | 1.05x | 0.639 | 0.615 | 1.04x |
| 128 | 1024 | 0.449 | 0.366 | 1.23x | 1.100 | 0.889 | 1.24x |
| 128 | 4096 | 5.293 | 3.581 | 1.48x | 11.422 | 7.451 | 1.53x |
| 128 | 8192 | 20.750 | 12.863 | 1.61x | 42.715 | 26.367 | 1.62x |
| 128 | 16384 | 79.879 | 50.131 | 1.59x | 166.506 | 103.913 | 1.60x |

Key takeaways:

- All eager and compiled runs succeeded on the tested grid.
- `torch.compile` provides only modest gains at short sequence length, but approaches about `2x` speedup on both forward and backward once `T` is large.
- The speedup is smaller at larger `d_model`, suggesting that compilation helps most when non-GEMM overhead is a larger fraction of the total work.
- Very small workloads may see little benefit or a slight slowdown, as in the `d_model=64`, `T=256` case.
