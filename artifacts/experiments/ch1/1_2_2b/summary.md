# 1.2.2(b) Full-Model torch.compile Comparison

| Model size | Vanilla forward (ms) | Compiled forward (ms) | Forward speedup | Vanilla train step (ms) | Compiled train step (ms) | Train-step speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| small | 18.753 | 6.983 | 2.69x | 56.712 | 32.093 | 1.77x |
| medium | 36.779 | 21.954 | 1.68x | 115.305 | 88.104 | 1.31x |
| large | 55.654 | 48.720 | 1.14x | 226.891 | 201.570 | 1.13x |
| xl | 100.988 | 90.707 | 1.11x | 406.375 | 381.541 | 1.07x |
| 2.7b | 158.315 | 149.824 | 1.06x | 618.726 | 596.189 | 1.04x |
