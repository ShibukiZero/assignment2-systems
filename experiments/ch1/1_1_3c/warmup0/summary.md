Context length: 128
Batch size: 4
Precision: fp32
Warmup steps: 0
Measurement steps: 10

| Model size | Forward mean (ms) | Forward std (ms) | Backward mean (ms) | Backward std (ms) | Total mean (ms) | Total std (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| small | 61.952 | 127.593 | 36.451 | 49.355 | 98.403 | 176.948 |
| medium | 85.303 | 134.941 | 65.436 | 44.869 | 150.739 | 179.809 |
| large | 107.603 | 142.390 | 130.120 | 40.883 | 237.724 | 183.273 |
| xl | 147.018 | 144.945 | 226.578 | 41.748 | 373.596 | 186.694 |
| 2.7b | 196.757 | 121.461 | 327.541 | 35.854 | 524.298 | 157.315 |
