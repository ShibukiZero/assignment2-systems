# Section 2.4(c): Compute-Bound Batch Threshold

For the mixed FSDP + TP forward-pass model from the Scaling Book, we use:

- `T_math = 4 * B * D * F / (N * C)`
- `T_FSDP = 4 * D * F / (Y * W_ici * M_X)`
- `T_TP = 4 * B * D / (X * W_ici * M_Y)`

and require

- `T_math >= max(T_FSDP, T_TP)`

Let `b = B / N` denote the per-device token batch size. Then:

- `T_math = 4 * b * D * F / C`
- `T_FSDP = 4 * D * F / (Y * W_ici * M_X)`
- `T_TP = 4 * Y * b * D / (W_ici * M_Y)`

Comparing `T_math` with `T_FSDP` gives the batch threshold:

- `b >= C / (Y * W_ici * M_X) = alpha / (Y * M_X)`

where

- `alpha = C / W_ici`

For the TPU v5p constants in the problem:

- `C = 4.6 * 10^14 FLOP/s`
- `W_ici = 2 * 9 * 10^10 bytes/s`
- `alpha = 2555.555555555556`
- `X = 16`
- `Y = 4`
- `M_X = 2`
- `M_Y = 1`
- `N = X * Y = 64`

This yields:

- per-device token batch threshold: `2555.555555555556 / (4 * 2) = 319.444444444444`
- minimum integer per-device token batch: `320`
- overall token batch threshold: `319.444444444444 * 64 = 20444.444444444445`
- minimum integer overall token batch: `320 * 64 = 20480`

The TP-side condition is:

- `F >= alpha * Y / M_Y`

Numerically:

- `53248 >= 2555.555555555556 * 4 / 1 = 10222.222222222223`

so the TP communication constraint is already satisfied, and the compute-bound threshold is set by the FSDP communication term.
