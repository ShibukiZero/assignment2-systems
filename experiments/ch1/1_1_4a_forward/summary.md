Context lengths: 128, 256, 512, 1024
Batch size: 4
Precision: fp32
Warmup steps: 5
Measurement steps: 10
Mode: forward

All 20 `(model size, context length)` configurations produced both:

- `<tag>.json`
- `<tag>.nsys-rep`

Python benchmark forward mean (ms):

| Model size | ctx128 | ctx256 | ctx512 | ctx1024 |
| --- | ---: | ---: | ---: | ---: |
| small | 25.136 | 25.052 | 30.414 | 66.668 |
| medium | 49.840 | 48.411 | 95.402 | 203.335 |
| large | 74.541 | 108.404 | 203.913 | 421.812 |
| xl | 102.345 | 179.057 | 386.445 | 833.741 |
| 2.7b | 159.338 | 301.952 | 644.996 | 1309.863 |

Suggested manual readout columns for `1.1.4(a)` after opening the GUI:

| Model size | Context length | Python benchmark forward mean (ms) | Nsight forward time (ms) | Match? | Notes |
| --- | ---: | ---: | ---: | --- | --- |
| small | 128 | 25.136 | TODO | TODO | TODO |
| medium | 128 | 49.840 | TODO | TODO | TODO |
| large | 128 | 74.541 | TODO | TODO | TODO |
| xl | 128 | 102.345 | TODO | TODO | TODO |
| 2.7b | 128 | 159.338 | TODO | TODO | TODO |
