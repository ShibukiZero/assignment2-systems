## Mixed Precision Accumulation

Code snippet from the handout:

```python
s = torch.tensor(0, dtype=torch.float32)
for i in range(1000):
    s += torch.tensor(0.01, dtype=torch.float32)
print(s)

s = torch.tensor(0, dtype=torch.float16)
for i in range(1000):
    s += torch.tensor(0.01, dtype=torch.float16)
print(s)

s = torch.tensor(0, dtype=torch.float32)
for i in range(1000):
    s += torch.tensor(0.01, dtype=torch.float16)
print(s)

s = torch.tensor(0, dtype=torch.float32)
for i in range(1000):
    x = torch.tensor(0.01, dtype=torch.float16)
    s += x.type(torch.float32)
print(s)
```

Observed outputs:

| Case | Output |
| --- | ---: |
| FP32 accumulator, FP32 input | `10.0001` |
| FP16 accumulator, FP16 input | `9.9531` |
| FP32 accumulator, FP16 input | `10.0021` |
| FP32 accumulator, FP16 input cast back to FP32 before add | `10.0021` |

Takeaway:

- The dominant source of error is low-precision accumulation, not just low-precision inputs.
- Keeping the accumulator in FP32 recovers most of the accuracy even when the added value is first quantized to FP16.
