## 1.1.5(b) LayerNorm and Mixed Precision

Question focus:

- Why LayerNorm is treated more carefully than feed-forward layers under FP16 autocast.
- Whether BF16 still requires special care.

Key observations:

- The sensitive parts of LayerNorm are the mean/variance reductions, accumulation of squared values, and the final normalization step.
- These operations are more vulnerable to rounding error, overflow, and underflow than GEMM-heavy feed-forward layers.
- BF16 is more stable than FP16 because it has the same exponent range as FP32, but it still has reduced mantissa precision, so higher-precision handling can still improve robustness.

Connection to 1.1.5(a):

- In the toy-model autocast check, `fc1` output and logits are `float16`, but the LayerNorm output remains `float32`, matching the expected mixed-precision behavior.
