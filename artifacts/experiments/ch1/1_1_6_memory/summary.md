# 1.1.6 Memory Sweep Summary

Configuration:

- model size: `2.7b`
- context lengths: `128`, `256`, `512`
- modes: `forward`, `train-step`
- precisions: `fp32`, `bf16`

Key takeaways:

- In FP32, forward-only peak memory grows moderately from 12.93 GiB at context length 128 to 13.45 GiB at context length 512.
- In FP32, full training-step peak memory is much higher and rises more sharply, from 51.44 GiB at context length 128 to 65.52 GiB at context length 512.
- In this benchmark setup, BF16 reduces train-step peak memory at context length 512 (65.52 GiB to 62.69 GiB), but forward-only peak memory is actually higher for BF16 at the same context length (13.45 GiB vs 19.41 GiB).

## 1.1.6(b) FP32 Peak Memory

| Context length | Forward peak memory | Full training step peak memory |
| --- | ---: | ---: |
| 128 | 12.93 GiB | 51.44 GiB |
| 256 | 13.02 GiB | 51.44 GiB |
| 512 | 13.45 GiB | 65.52 GiB |

## 1.1.6(c) FP32 vs BF16 Peak Memory

| Context length | Mode | FP32 peak memory | BF16 peak memory | FP32/BF16 ratio |
| --- | --- | ---: | ---: | ---: |
| 128 | forward | 12.93 GiB | 19.16 GiB | 0.67x |
| 128 | train-step | 51.44 GiB | 51.44 GiB | 1.00x |
| 256 | forward | 13.02 GiB | 19.18 GiB | 0.68x |
| 256 | train-step | 51.44 GiB | 52.11 GiB | 0.99x |
| 512 | forward | 13.45 GiB | 19.41 GiB | 0.69x |
| 512 | train-step | 65.52 GiB | 62.69 GiB | 1.05x |
