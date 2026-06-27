# 1.2.2(b) Full-Model torch.compile Comparison

| Model size | Vanilla forward (ms) | Compiled forward (ms) | Forward speedup | Vanilla train step (ms) | Compiled train step (ms) | Train-step speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| small | 17.165 | 9.019 | 1.90x | 51.871 | 38.758 | 1.34x |
| medium | 34.243 | 18.402 | 1.86x | 107.191 | 75.511 | 1.42x |
| large | 49.533 | 27.216 | 1.82x | 165.443 | 127.681 | 1.30x |
| xl | 65.820 | 35.898 | 1.83x | 258.230 | 220.877 | 1.17x |
| 2.7b | 43.312 | 24.548 | 1.76x | 303.817 | 279.127 | 1.09x |
