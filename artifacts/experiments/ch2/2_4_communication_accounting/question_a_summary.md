# Section 2.4(a): XXL Memory Accounting

Assumptions:

- `d_model = 16384`
- `d_ff = 53248`
- `num_blocks = 126`
- Only the FFN weights are counted.
- Each block contains two linear layers: `d_model -> d_ff` and `d_ff -> d_model`.
- Master weights, accumulated gradients, and optimizer state are stored in FP32.
- Saved activations for backward are stored in BF16.

Parameter count:

- Per block: `2 * d_model * d_ff = 2 * 16384 * 53248 = 1,744,830,464`
- Total: `126 * 1,744,830,464 = 219,848,638,464`

FP32 memory:

- Master weights: `219,848,638,464 * 4 = 879,394,553,856 bytes = 819.0 GiB`
- Accumulated gradients: `219,848,638,464 * 4 = 879,394,553,856 bytes = 819.0 GiB`
- Optimizer states (Adam): `219,848,638,464 * 8 = 1,758,789,107,712 bytes = 1638.0 GiB`
- Total FP32 model state: `3,517,578,215,424 bytes = 3276.0 GiB`

Equivalent H100 80GB count:

- Lower bound without activations: `3,517,578,215,424 / (80 * 10^9) = 43.97`

Saved activations for backward:

- In the simplified FFN-only model, the backward pass still needs FFN activations of shape `[B, T, d_model]` and `[B, T, d_ff]` per block.
- Therefore the BF16 saved-activation memory is:

```text
num_blocks * B * T * (d_model + d_ff) * 2 bytes
= 126 * B * T * (16384 + 53248) * 2 bytes
= 17,547,264 * B * T bytes
```

Total training memory including saved activations:

```text
3,517,578,215,424 + 17,547,264 * B * T bytes
```

Required H100 80GB count including activations:

```text
ceil((3,517,578,215,424 + 17,547,264 * B * T) / (80 * 10^9))
```
