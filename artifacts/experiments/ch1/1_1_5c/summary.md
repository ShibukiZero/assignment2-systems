## 1.1.5(c) BF16 Precision Sweep

Configuration:

- mode: `forward-backward`
- warmup steps: `5`
- measurement steps: `10`
- batch size: `4`
- vocabulary size: `10,000`

Key takeaways:

- At context length `128`, BF16 provides little benefit for the smallest model: `small` changes from `37.23 ms` to `39.25 ms` total (`0.95x`).
- At the same context length, the benefit is already large for `2.7b`: total time drops from `136.03 ms` to `127.41 ms` (`1.07x`).

### Context length = 128

| Model size | FP32 forward (ms) | BF16 forward (ms) | Forward speedup | FP32 backward (ms) | BF16 backward (ms) | Backward speedup | FP32 total (ms) | BF16 total (ms) | Total speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| small | 19.441 | 19.699 | 0.99x | 17.784 | 19.546 | 0.91x | 37.225 | 39.245 | 0.95x |
| medium | 36.519 | 38.777 | 0.94x | 34.146 | 38.641 | 0.88x | 70.665 | 77.418 | 0.91x |
| large | 55.736 | 58.114 | 0.96x | 57.099 | 58.987 | 0.97x | 112.835 | 117.101 | 0.96x |
| xl | 79.326 | 79.974 | 0.99x | 80.928 | 81.368 | 0.99x | 160.253 | 161.342 | 0.99x |
| 2.7b | 50.289 | 53.977 | 0.93x | 85.741 | 73.431 | 1.17x | 136.030 | 127.407 | 1.07x |

### Context length = 256

| Model size | FP32 forward (ms) | BF16 forward (ms) | Forward speedup | FP32 backward (ms) | BF16 backward (ms) | Backward speedup | FP32 total (ms) | BF16 total (ms) | Total speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| small | 18.856 | 19.974 | 0.94x | 18.758 | 20.769 | 0.90x | 37.614 | 40.744 | 0.92x |
| medium | 37.218 | 38.905 | 0.96x | 35.291 | 39.088 | 0.90x | 72.509 | 77.993 | 0.93x |
| large | 57.536 | 59.482 | 0.97x | 60.622 | 60.523 | 1.00x | 118.158 | 120.005 | 0.98x |
| xl | 80.625 | 86.251 | 0.93x | 104.776 | 82.779 | 1.27x | 185.401 | 169.030 | 1.10x |
| 2.7b | 55.368 | 55.122 | 1.00x | 129.996 | 97.668 | 1.33x | 185.364 | 152.789 | 1.21x |

### Context length = 512

| Model size | FP32 forward (ms) | BF16 forward (ms) | Forward speedup | FP32 backward (ms) | BF16 backward (ms) | Backward speedup | FP32 total (ms) | BF16 total (ms) | Total speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| small | 19.092 | 20.506 | 0.93x | 22.596 | 20.485 | 1.10x | 41.688 | 40.991 | 1.02x |
| medium | 38.758 | 40.452 | 0.96x | 55.840 | 45.746 | 1.22x | 94.598 | 86.198 | 1.10x |
| large | 60.083 | 61.643 | 0.97x | 114.843 | 88.814 | 1.29x | 174.926 | 150.458 | 1.16x |
| xl | 81.466 | 82.714 | 0.98x | 200.294 | 152.507 | 1.31x | 281.760 | 235.221 | 1.20x |
| 2.7b | 87.952 | 75.869 | 1.16x | 232.815 | 167.438 | 1.39x | 320.768 | 243.307 | 1.32x |

### Context length = 1024

| Model size | FP32 forward (ms) | BF16 forward (ms) | Forward speedup | FP32 backward (ms) | BF16 backward (ms) | Backward speedup | FP32 total (ms) | BF16 total (ms) | Total speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| small | 24.214 | 23.178 | 1.04x | 52.797 | 42.869 | 1.23x | 77.011 | 66.047 | 1.17x |
| medium | 61.298 | 58.940 | 1.04x | 137.023 | 112.826 | 1.21x | 198.321 | 171.766 | 1.15x |
| large | 117.209 | 109.163 | 1.07x | 273.648 | 215.160 | 1.27x | 390.857 | 324.323 | 1.21x |
