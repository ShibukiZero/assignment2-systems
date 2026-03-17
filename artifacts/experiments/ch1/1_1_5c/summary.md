## 1.1.5(c) BF16 Precision Sweep

Configuration:

- mode: `forward-backward`
- warmup steps: `5`
- measurement steps: `10`
- batch size: `4`
- vocabulary size: `10,000`

Key takeaways:

- At context length `128`, BF16 provides little benefit for the smallest model: `small` changes from `42.48 ms` to `46.74 ms` total (`0.91x`).
- At the same context length, the benefit is already large for `2.7b`: total time drops from `475.01 ms` to `202.29 ms` (`2.35x`).
- At context length `1024`, BF16 helps across the board, but the speedup still grows with scale: `small` improves from `195.26 ms` to `98.84 ms` (`1.98x`), whereas `2.7b` improves from `3766.59 ms` to `1181.84 ms` (`3.19x`).

### Context length = 128

| Model size | FP32 forward (ms) | BF16 forward (ms) | Forward speedup | FP32 backward (ms) | BF16 backward (ms) | Backward speedup | FP32 total (ms) | BF16 total (ms) | Total speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| small | 21.528 | 22.572 | 0.95x | 20.949 | 24.168 | 0.87x | 42.478 | 46.740 | 0.91x |
| medium | 41.968 | 46.359 | 0.91x | 51.163 | 47.871 | 1.07x | 93.131 | 94.230 | 0.99x |
| large | 62.025 | 69.081 | 0.90x | 117.834 | 71.964 | 1.64x | 179.859 | 141.045 | 1.28x |
| xl | 101.720 | 92.155 | 1.10x | 214.356 | 99.234 | 2.16x | 316.076 | 191.389 | 1.65x |
| 2.7b | 158.517 | 62.757 | 2.53x | 316.490 | 139.529 | 2.27x | 475.007 | 202.286 | 2.35x |

### Context length = 256

| Model size | FP32 forward (ms) | BF16 forward (ms) | Forward speedup | FP32 backward (ms) | BF16 backward (ms) | Backward speedup | FP32 total (ms) | BF16 total (ms) | Total speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| small | 22.098 | 24.155 | 0.91x | 30.164 | 24.245 | 1.24x | 52.262 | 48.400 | 1.08x |
| medium | 47.393 | 46.913 | 1.01x | 91.278 | 48.920 | 1.87x | 138.671 | 95.833 | 1.45x |
| large | 107.637 | 71.325 | 1.51x | 201.475 | 89.237 | 2.26x | 309.112 | 160.562 | 1.93x |
| xl | 177.498 | 94.785 | 1.87x | 379.733 | 155.652 | 2.44x | 557.231 | 250.437 | 2.23x |
| 2.7b | 300.634 | 91.777 | 3.28x | 577.257 | 202.651 | 2.85x | 877.891 | 294.428 | 2.98x |

### Context length = 512

| Model size | FP32 forward (ms) | BF16 forward (ms) | Forward speedup | FP32 backward (ms) | BF16 backward (ms) | Backward speedup | FP32 total (ms) | BF16 total (ms) | Total speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| small | 36.107 | 24.237 | 1.49x | 61.284 | 27.990 | 2.19x | 97.392 | 52.227 | 1.86x |
| medium | 94.228 | 47.858 | 1.97x | 180.134 | 77.747 | 2.32x | 274.362 | 125.605 | 2.18x |
| large | 202.201 | 83.501 | 2.42x | 387.231 | 163.741 | 2.36x | 589.432 | 247.242 | 2.38x |
| xl | 383.733 | 140.536 | 2.73x | 762.382 | 282.506 | 2.70x | 1146.115 | 423.042 | 2.71x |
| 2.7b | 641.927 | 176.662 | 3.63x | 1162.342 | 367.970 | 3.16x | 1804.269 | 544.632 | 3.31x |

### Context length = 1024

| Model size | FP32 forward (ms) | BF16 forward (ms) | Forward speedup | FP32 backward (ms) | BF16 backward (ms) | Backward speedup | FP32 total (ms) | BF16 total (ms) | Total speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| small | 66.007 | 35.502 | 1.86x | 129.248 | 63.334 | 2.04x | 195.255 | 98.836 | 1.98x |
| medium | 201.085 | 95.429 | 2.11x | 392.607 | 180.768 | 2.17x | 593.692 | 276.197 | 2.15x |
| large | 417.500 | 186.517 | 2.24x | 815.231 | 361.438 | 2.26x | 1232.731 | 547.955 | 2.25x |
| xl | 827.702 | 330.884 | 2.50x | 1683.083 | 648.727 | 2.59x | 2510.784 | 979.611 | 2.56x |
| 2.7b | 1304.775 | 395.704 | 3.30x | 2461.816 | 786.132 | 3.13x | 3766.591 | 1181.836 | 3.19x |
