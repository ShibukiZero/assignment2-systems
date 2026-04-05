# Section 2.4(b): FSDP-Sharded Memory Accounting

Let:

- `W` = master weights
- `G` = accumulated gradients
- `O` = optimizer states
- `A` = total saved activations for backward

From part (a):

- `W + G + O = 3,517,578,215,424 bytes`
- `A = 17,547,264 * B * T bytes`

The problem states that master weights, optimizer state, gradients, and **half** of the activations are sharded across `N_FSDP` devices. The other half of the activations remains unsharded. Therefore the per-device memory is:

```text
M(N_FSDP) = (W + G + O + 0.5 * A) / N_FSDP + 0.5 * A
```

Substituting the values from part (a):

```text
M(N_FSDP) =
(3,517,578,215,424 + 0.5 * (17,547,264 * B * T)) / N_FSDP
+ 0.5 * (17,547,264 * B * T)
```

To fit within one v5p TPU device, require `M(N_FSDP) < 95 * 10^9`. Solving for `N_FSDP` gives:

```text
N_FSDP >
(3,517,578,215,424 + 0.5 * (17,547,264 * B * T))
/
(95 * 10^9 - 0.5 * (17,547,264 * B * T))
```

So the minimum valid value is:

```text
N_FSDP = ceil(
  (3,517,578,215,424 + 0.5 * (17,547,264 * B * T))
  /
  (95 * 10^9 - 0.5 * (17,547,264 * B * T))
)
```

This is still a function of `B` and `T`, because the activation term in part (a) is itself a function of token count.

Typical examples using the assignment-wide default `B = 4`:

| Batch size | Context length | Saved activations (GiB) | Minimum `N_FSDP` for `< 95 GB` |
| --- | ---: | ---: | ---: |
| 4 | 128 | 8.367 | 39 |
| 4 | 256 | 16.734 | 41 |
| 4 | 512 | 33.469 | 46 |
