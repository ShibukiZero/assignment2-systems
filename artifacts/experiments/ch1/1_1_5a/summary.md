## 1.1.5(a) Dtypes Under FP16 Autocast

Experiment:

- Script: [`scripts/inspect_1_1_5a_autocast_dtypes.py`](/Users/linzihan/Github/assignment2-systems/scripts/inspect_1_1_5a_autocast_dtypes.py)
- Device: CUDA
- Precision mode: `torch.autocast(device_type="cuda", dtype=torch.float16)`

Observed dtypes:

| Component | Observed dtype |
| --- | --- |
| model parameters | `float32` |
| `fc1` output | `float16` |
| `layer norm` output | `float32` |
| logits | `float16` |
| loss | `float32` |
| gradients | `float32` |

Takeaway:

- Autocast does not permanently convert parameters to FP16.
- Linear layers run in lower precision where possible, while numerically sensitive normalization, loss, and stored gradient state remain in FP32.
