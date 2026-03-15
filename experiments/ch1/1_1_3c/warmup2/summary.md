Context length: 128
Batch size: 4
Precision: fp32
Warmup steps: 2
Measurement steps: 10

| Model size | Forward mean (ms) | Forward std (ms) | Backward mean (ms) | Backward std (ms) | Total mean (ms) | Total std (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| small | 21.530 | 0.050 | 20.920 | 0.117 | 42.450 | 0.109 |
| medium | 42.929 | 1.211 | 50.888 | 0.076 | 93.817 | 1.262 |
| large | 63.079 | 0.172 | 117.294 | 0.337 | 180.373 | 0.266 |
| xl | 102.325 | 2.679 | 213.560 | 0.186 | 315.885 | 2.657 |
| 2.7b | 158.240 | 0.158 | 316.118 | 0.090 | 474.358 | 0.176 |
