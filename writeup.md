## Problem `benchmarking_script`: Benchmarking Script (4 points)

### (b)
**Question:** Time the forward and backward passes for the model sizes described in Section 1.1.2. Use 5 warmup steps and compute the average and standard deviation over 10 measurement steps. How long does a forward pass take? How about a backward pass? Do you see high variability across measurements, or is the standard deviation small?

**Deliverable:** A 1-2 sentence response with your timings.

**Answer:** The table below reports the measured forward and backward timings for the five model sizes from Section 1.1.2 using 5 warmup steps and 10 measured steps at context length 128, batch size 4, vocabulary size 10,000, and FP32 precision. Forward and backward latency both increase with model size, while the standard deviations remain small relative to the means, indicating that the measurements are stable after warmup.

| Model size | Forward mean (ms) | Forward std (ms) | Backward mean (ms) | Backward std (ms) | Total mean (ms) | Total std (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| small | 21.837 | 0.057 | 21.164 | 0.039 | 43.001 | 0.076 |
| medium | 42.146 | 0.455 | 50.833 | 0.047 | 92.979 | 0.472 |
| large | 62.412 | 1.055 | 117.320 | 0.288 | 179.732 | 1.074 |
| xl | 101.257 | 0.124 | 213.640 | 0.122 | 314.898 | 0.168 |
| 2.7b | 158.532 | 0.209 | 316.248 | 0.187 | 474.780 | 0.315 |

### (c)
**Question:** Repeat the analysis without warmup steps. How does this affect your results? Why do you think this happens? Also try 1 or 2 warmup steps. Why might the result still be different?

**Deliverable:** A 2-3 sentence response.

**Answer:** Removing warmup makes the measurements much noisier and substantially inflates both the means and the standard deviations, because the first measured iteration absorbs one-time startup costs such as CUDA runtime initialization, kernel loading, memory allocation, and library autotuning. For example, with `warmup=0`, the first measured `small` step took about 602 ms and the first measured `xl` step took about 905 ms, even though subsequent steps were near 42 ms and 315 ms respectively. Using 2 warmup steps already brings the results much closer to the 5-warmup baseline, but small differences can remain because not all lazy initialization and caching effects are exhausted immediately, and ordinary run-to-run system noise is still present.

---

## Problem `nsys_profile`: Nsight Systems Profiler (5 points)

### (a)
**Question:** What is the total time spent on the forward pass? Does it match what was measured before with the Python standard library?

**Deliverable:** A 1-2 sentence response.

**Answer:** At context length 128, Nsight Systems reports forward-pass times of 25.245 ms (`small`), 49.488 ms (`medium`), 75.931 ms (`large`), 101.033 ms (`xl`), and 158.597 ms (`2.7b`). Compared with the earlier Python benchmark from `1.1.3(b)` (21.837 ms, 42.146 ms, 62.412 ms, 101.257 ms, and 158.532 ms respectively), the agreement is very close for `xl` and `2.7b`, but noticeably looser for the smaller three models, so the overall trend matches while the exact values do not align equally well across all sizes.

### (b)
**Question:** What CUDA kernel takes the most cumulative GPU time during the forward pass? How many times is this kernel invoked during a single forward pass of the model? Is it the same kernel that takes the most runtime when you do both forward and backward passes?

**Deliverable:** A 1-2 sentence response.

**Answer:** In a representative `2.7b`, context-length-512 forward trace, the CUDA GPU Kernel Summary is dominated by the Tensor Core GEMM kernel `sm80_xmma_gemm_f32f32_f32f32_f32_tn_n_tilesize64x64x8_...`, which appears 975 times across the 15 profiled forward passes, i.e. about 65 times per forward pass. In the corresponding full training-step trace, the top kernel is still a Tensor Core GEMM, but it changes to `sm80_xmma_gemm_f32f32_f32f32_f32_tn_n_tilesize128x64x8_...` rather than remaining exactly the same kernel.

### (c)
**Question:** Besides matrix multiplies, what other kernels account for non-trivial CUDA runtime in the forward pass?

**Deliverable:** A 1-2 sentence response.

**Answer:** Besides GEMM kernels, the CUDA GPU Kernel Summary still shows visible time in ATen `elementwise_kernel`, `vectorized_elementwise_kernel`, and `reduce_kernel` launches. In the representative `2.7b`, context-length-512 forward trace, these non-matmul kernels are much smaller than the dominant GEMMs, but several still contribute on the order of about 0.5%-1.5% each to total GPU time.

### (d)
**Question:** Profile one complete training step with AdamW. How does the fraction of time spent on matrix multiplication change compared to inference-only? How about other kernels?

**Deliverable:** A 1-2 sentence response.

**Answer:** In the representative `2.7b`, context-length-512 traces, kernels whose names contain `gemm`, `xmma`, or `cutlass` account for about `92.20%` of forward-only GPU time but about `83.90%` of full training-step GPU time, so matrix multiplication still dominates training but by a smaller margin than in inference. The missing share is taken up by backward- and optimizer-related ATen `elementwise_kernel`, `vectorized_elementwise_kernel`, and `reduce_kernel` launches, which become much more prominent once gradient computation and AdamW updates are included.

### (e)
**Question:** Compare the runtime of the softmax operation versus the matrix multiplication operations within the self-attention layer during a forward pass. How does the runtime difference compare to the FLOP difference?

**Deliverable:** A 1-2 sentence response.

**Answer:** Interpreting the "self-attention layer" here as the core scaled-dot-product attention, the representative `2.7b`, context-length-512 forward trace reports `attention_softmax` at about `626,461 ns` on average, versus about `380,087 ns` for `attention_scores_matmul` and `329,561 ns` for `attention_value_matmul`, so softmax is slower than either individual matmul and is still on the same order as the two matmuls together. However, the FLOP gap is much larger: the scripted estimate gives each matmul about `5.37e9` FLOPs per layer versus only about `2.35e8` for softmax (about `22.87x` larger for each matmul, or `45.74x` for the two matmuls combined), which suggests that softmax is much more memory- and reduction-bound than the GEMM kernels.

---

## Problem `mixed_precision_accumulation`: Mixed Precision (1 point)

### Accumulation experiment
**Question:** Run the accumulation example from the handout and comment on the accuracy of the results.

```python
s = torch.tensor(0, dtype=torch.float32)
for i in range(1000):
    s += torch.tensor(0.01, dtype=torch.float32)

s = torch.tensor(0, dtype=torch.float16)
for i in range(1000):
    s += torch.tensor(0.01, dtype=torch.float16)

s = torch.tensor(0, dtype=torch.float32)
for i in range(1000):
    s += torch.tensor(0.01, dtype=torch.float16)

s = torch.tensor(0, dtype=torch.float32)
for i in range(1000):
    x = torch.tensor(0.01, dtype=torch.float16)
    s += x.type(torch.float32)
```

**Deliverable:** A 2-3 sentence response.

**Answer:** Accumulating `0.01` in FP32 stays very close to the expected value of `10`, while accumulating in FP16 underestimates much more noticeably (`9.9531` in our run), because both the input value and the running sum are repeatedly rounded at FP16 precision. Using FP16 inputs with an FP32 accumulator is much more accurate (`10.0021` here), even though it is still slightly worse than pure FP32 because the value `0.01` has already been quantized once when it is first represented in FP16. This illustrates why mixed-precision training usually keeps reductions and accumulations in higher precision even when some inputs or matmuls use lower precision.

### 1.1.5(a) Dtypes Under Autocast
**Question:** For the toy model under FP16 autocast, what are the data types of:

```python
class ToyModel(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False)
        self.ln = nn.LayerNorm(10)
        self.fc2 = nn.Linear(10, out_features, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.ln(x)
        x = self.fc2(x)
        return x
```

- the model parameters within the autocast context,
- the output of the first feed-forward layer,
- the output of layer norm,
- the model's predicted logits,
- the loss,
- and the model's gradients?

**Deliverable:** The data types for each of the listed components.

**Answer:** In our CUDA autocast check, the model parameters remain `float32`, the output of the first feed-forward layer (`fc1`) is `float16`, the output of layer norm is `float32`, the model logits are `float16`, the loss is `float32`, and the gradients are `float32`. This matches the intended mixed-precision pattern: linear layers run in lower precision where possible, while numerically sensitive normalization, loss computation, and stored parameter/gradient state stay in FP32.

### 1.1.5(b) LayerNorm and Mixed Precision
**Question:** You should have seen that FP16 mixed precision autocasting treats the layer normalization layer differently than the feed-forward layers. What parts of layer normalization are sensitive to mixed precision? If we use BF16 instead of FP16, do we still need to treat layer normalization differently? Why or why not?

**Deliverable:** A 2-3 sentence response.

**Answer:** The numerically sensitive parts of layer normalization are the mean/variance reductions, the accumulation of squared values, and the normalization step itself (subtracting the mean and dividing by the standard deviation), because these operations can amplify rounding error and are more vulnerable to overflow or underflow in low precision. With FP16, this makes it important to keep LayerNorm in higher precision. BF16 is much more stable because it has the same exponent range as FP32, so the overflow/underflow problem is much less severe, but its mantissa is still shorter than FP32, so treating LayerNorm more carefully can still improve numerical robustness.

### 1.1.5(c) BF16 Benchmarking
**Question:** Modify the benchmarking script to optionally run with BF16 mixed precision. Time the forward and backward passes with and without mixed precision for each language model size in Section 1.1.2. Compare the results and comment on any trends as model size changes.

**Deliverable:** A 2-3 sentence response with timings and commentary.

**Answer:** BF16 mixed precision provides little or no benefit on the smallest workloads, but it becomes increasingly effective as model size and context length grow. For example, at context length `128`, the total step time changes from `42.48 ms` to `46.74 ms` for `small` (`0.91x`) but from `475.01 ms` to `202.29 ms` for `2.7b` (`2.35x`), while at context length `1024` the same comparison is `195.26 ms` to `98.84 ms` for `small` (`1.98x`) and `3766.59 ms` to `1181.84 ms` for `2.7b` (`3.19x`). This pattern is consistent with larger workloads benefiting much more from Tensor Core acceleration once matrix multiplications dominate the runtime. The full timing tables are archived in `artifacts/experiments/ch1/1_1_5c/summary.md`.

---

## Problem `memory_profiling`: Memory Profiling (4 points)

### (a)
**Question:** Run the memory profiler on the 2.7B model for inference-only and for a full training step. How do the active memory timelines look? Can you tell which stage is running based on the peaks?

**Deliverable:** Two images of the active memory timeline of a 2.7B model, one for the forward pass and one for a full training step, plus a 2-3 sentence response.

**Answer:**

Forward-pass timeline:

![Forward memory timeline](/Users/linzihan/Github/assignment2-systems/artifacts/experiments/ch1/1_1_6a/fp32_2.7b_ctx512_forward_active_memory_timeline.png)

Training-step timeline:

![Training memory timeline](/Users/linzihan/Github/assignment2-systems/artifacts/experiments/ch1/1_1_6a/fp32_2.7b_ctx512_train_step_active_memory_timeline.png)

Response:

The forward-only active-memory timeline is not completely flat: it shows a roughly periodic sequence of about `32` spikes, which lines up well with the `32` Transformer blocks in the `2.7b` model. The spike size is on the order of about `128 MiB`, which is consistent with transient attention score/probability tensors of shape `batch x heads x seq x seq` being materialized and then released within each block during inference. The full training-step timeline has a clearer multi-stage structure: memory first drops sharply, then rises relatively quickly, then decreases more gradually, and finally rises again. This is consistent with the forward pass building up saved activations, the backward pass releasing part of that activation memory while traversing the graph, and the final optimizer step plus allocator/cache effects changing the live-memory footprint again. So yes, the timeline shape is informative enough that the broad stages of the training step can be inferred from the peaks and valleys.

### (b)
**Question:** What is the peak memory usage of each context length when doing a forward pass? What about when doing a full training step?

**Deliverable:** A table with two numbers per context length.

**Answer:**

| Context length | Forward peak memory | Full training step peak memory |
| --- | --- | --- |
| 128 | 12.93 GiB | 51.44 GiB |
| 256 | 13.02 GiB | 51.44 GiB |
| 512 | 13.45 GiB | 65.52 GiB |

Forward-only peak memory grows with context length but only moderately, whereas the full training step uses much more memory overall and shows a much larger increase by context length 512. This is consistent with training needing to retain saved activations, gradients, and optimizer-related state in addition to the forward-pass allocations.

### (c)
**Question:** Find the peak memory usage of the 2.7B model when using mixed precision, for both a forward pass and a full optimizer step. Does mixed precision significantly affect memory usage?

**Deliverable:** A 2-3 sentence response.

**Answer:** In this setup, BF16 does not significantly reduce measured peak memory overall. For forward-only runs, the measured peak memory is actually higher under BF16 at all three tested context lengths (`12.93 -> 19.16 GiB`, `13.02 -> 19.18 GiB`, and `13.45 -> 19.41 GiB` for context lengths `128`, `256`, and `512` respectively), while for full training steps it is nearly unchanged at shorter contexts (`51.44 -> 51.44 GiB` at `128`, `51.44 -> 52.11 GiB` at `256`) and only modestly lower at `512` (`65.52 -> 62.69 GiB`). A plausible explanation is that BF16 autocast changes the execution path rather than simply shrinking every tensor: parameters and optimizer state still remain in FP32, while extra cast/workspace buffers can be introduced during lower-precision execution, so the net peak-memory effect is small and can even be negative for forward-only runs.

### (d)
**Question:** Consider the 2.7B model. At our reference hyperparameters, what is the size of a tensor of activations in the Transformer residual stream, in single precision? Give this size in MB.

**Deliverable:** A 1-2 sentence response with your derivation.

**Answer:** For the 2.7B model, the residual-stream activation tensor at the reference hyperparameters has shape `(batch_size, context_length, d_model) = (4, 128, 2560)`, so it contains `4 * 128 * 2560 = 1,310,720` elements. In single precision this is `1,310,720 * 4 = 5,242,880` bytes, which is exactly `5.00 MiB` after dividing by `1024^2`.

### (e)
**Question:** Now look closely at the “Active Memory Timeline” from pytorch.org/memory_viz of a memory snapshot of the 2.7B model doing a forward pass. When you reduce the “Detail” level, the tool hides the smallest allocations to the corresponding level. What is the size of the largest allocations shown? Looking through the stack trace, can you tell where those allocations come from?

**Deliverable:** A 1-2 sentence response.

**Answer:** The largest allocations visible in the forward-pass memory snapshot are about `128 MiB` each. Their stack traces point to the `softmax` call inside `scaled_dot_product_attention`, which matches the size of an explicitly materialized attention score/weight tensor of shape `(batch, heads, seq_len, seq_len)` for the `2.7b` model at `batch=4`, `heads=32`, and `seq_len=512` (`4 * 32 * 512 * 512 * 4 bytes = 128 MiB`), so these allocations come from the naive self-attention implementation rather than the residual-stream activations.

---

## Problem `pytorch_attention`: Benchmarking PyTorch Attention (2 points)

### (a)
**Question:** Report the timings or out-of-memory errors for the requested attention configurations. At what size do you get out-of-memory errors? Do the memory accounting for one of the smallest configurations that runs out of memory. How does the memory saved for backward change with sequence length? What would you do to eliminate this memory cost?

**Deliverable:** A table with timings, your memory-usage working, and a 1-2 paragraph response.

**Answer:**

| d_model | Sequence length | Forward timing (ms) | Backward timing (ms) | Saved for backward (GiB) | Status |
| --- | --- | ---: | ---: | ---: | --- |
| 16 | 256 | 0.262 | 0.605 | 0.004 | success |
| 16 | 1024 | 0.323 | 0.849 | 0.063 | success |
| 16 | 4096 | 3.298 | 7.581 | 1.002 | success |
| 16 | 8192 | 13.383 | 31.941 | 4.005 | success |
| 16 | 16384 | 51.331 | 121.836 | 16.009 | success |
| 32 | 256 | 0.244 | 0.562 | 0.004 | success |
| 32 | 1024 | 0.339 | 0.866 | 0.064 | success |
| 32 | 4096 | 3.469 | 7.728 | 1.004 | success |
| 32 | 8192 | 14.281 | 29.686 | 4.009 | success |
| 32 | 16384 | 55.450 | 115.465 | 16.017 | success |
| 64 | 256 | 0.246 | 0.552 | 0.004 | success |
| 64 | 1024 | 0.366 | 0.918 | 0.065 | success |
| 64 | 4096 | 4.009 | 8.757 | 1.008 | success |
| 64 | 8192 | 16.264 | 33.791 | 4.016 | success |
| 64 | 16384 | 61.564 | 128.291 | 16.033 | success |
| 128 | 256 | 0.269 | 0.659 | 0.005 | success |
| 128 | 1024 | 0.449 | 1.117 | 0.066 | success |
| 128 | 4096 | 5.298 | 11.448 | 1.016 | success |
| 128 | 8192 | 20.726 | 42.572 | 4.032 | success |
| 128 | 16384 | 79.873 | 166.429 | 16.064 | success |

Memory accounting:

For a single FP32 attention intermediate of shape `(B, T, T)`, the memory cost is `B * T * T * 4` bytes. However, the measured memory saved for backward is closer to about twice that amount, because by the end of the forward pass the naive implementation appears to keep roughly two large `B x T x T` FP32 intermediates alive, corresponding naturally to the pre-softmax attention scores and the post-softmax attention weights. For example, with `B=8` and `T=16384`, one such tensor would require `8 * 16384 * 16384 * 4 = 8 GiB`, while the measured saved-for-backward memory is about `16 GiB`, consistent with storing about two such tensors rather than just one.

Response:

In the tested range, none of the requested configurations ran out of memory, so we did not observe an OOM threshold within this sweep. Even without an actual OOM, however, the quadratic memory trend is very clear: the memory saved for backward depends overwhelmingly on sequence length rather than embedding dimension, and it grows by about `4x` whenever the sequence length doubles. For example, at `d_model=128`, the saved-for-backward memory increases from about `1.02 GiB` at `T=4096` to `4.03 GiB` at `T=8192` and then to `16.06 GiB` at `T=16384`.

The timing results show the same high-level pattern. Both forward and backward latency rise sharply with sequence length, while increasing `d_model` has a much smaller effect than increasing `T`. This indicates that the dominant bottleneck in naive attention at long sequence length is the `T x T` attention matrix and its associated memory traffic, not the `T x d_model` inputs themselves. To eliminate this memory cost, we would avoid explicitly materializing the full attention matrix and instead use a tiled attention algorithm with online softmax and recomputation, as in FlashAttention.

---

## Problem `torch_compile`: Benchmarking JIT-Compiled Attention (2 points)

### (a)
**Question:** Compare a compiled version of the PyTorch attention implementation against the uncompiled version under the same configuration as the previous attention benchmark.

**Deliverable:** A table comparing forward and backward timings for the compiled attention module with the uncompiled version.

**Answer:**

| d_model | Sequence length | Uncompiled forward (ms) | Compiled forward (ms) | Forward speedup | Uncompiled backward (ms) | Compiled backward (ms) | Backward speedup |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 256 | 0.257 | 0.212 | 1.21x | 0.605 | 0.507 | 1.19x |
| 16 | 1024 | 0.315 | 0.226 | 1.39x | 0.826 | 0.592 | 1.40x |
| 16 | 4096 | 3.291 | 1.568 | 2.10x | 7.520 | 3.486 | 2.16x |
| 16 | 8192 | 13.348 | 5.455 | 2.45x | 31.717 | 15.373 | 2.06x |
| 16 | 16384 | 51.325 | 21.015 | 2.44x | 121.785 | 57.350 | 2.12x |
| 32 | 256 | 0.266 | 0.216 | 1.23x | 0.604 | 0.495 | 1.22x |
| 32 | 1024 | 0.330 | 0.260 | 1.27x | 0.838 | 0.656 | 1.28x |
| 32 | 4096 | 3.458 | 1.712 | 2.02x | 7.692 | 3.592 | 2.14x |
| 32 | 8192 | 14.236 | 6.364 | 2.24x | 29.488 | 13.126 | 2.25x |
| 32 | 16384 | 55.150 | 25.263 | 2.18x | 115.017 | 51.274 | 2.24x |
| 64 | 256 | 0.245 | 0.261 | 0.94x | 0.564 | 0.604 | 0.93x |
| 64 | 1024 | 0.370 | 0.285 | 1.30x | 0.922 | 0.698 | 1.32x |
| 64 | 4096 | 3.986 | 2.255 | 1.77x | 8.716 | 4.672 | 1.87x |
| 64 | 8192 | 16.223 | 8.599 | 1.89x | 33.600 | 18.100 | 1.86x |
| 64 | 16384 | 61.485 | 31.538 | 1.95x | 128.266 | 64.758 | 1.98x |
| 128 | 256 | 0.261 | 0.248 | 1.05x | 0.639 | 0.615 | 1.04x |
| 128 | 1024 | 0.449 | 0.366 | 1.23x | 1.100 | 0.889 | 1.24x |
| 128 | 4096 | 5.293 | 3.581 | 1.48x | 11.422 | 7.451 | 1.53x |
| 128 | 8192 | 20.750 | 12.863 | 1.61x | 42.715 | 26.367 | 1.62x |
| 128 | 16384 | 79.879 | 50.131 | 1.59x | 166.506 | 103.913 | 1.60x |

`torch.compile` helps only modestly on the shortest sequences, but becomes much more effective as sequence length grows. Averaging across `d_model`, the forward/backward speedups are only about `1.11x/1.10x` at `T=256`, rise to about `1.30x/1.31x` at `T=1024`, and then reach roughly `1.84x/1.93x` at `T=4096` and about `2x` by `T=8192` and `T=16384`. This suggests that compilation is most helpful when long-sequence attention creates enough graph structure for PyTorch to fuse away substantial dispatch, elementwise, and reduction overhead.

The gains are smaller at larger `d_model`. Averaging across sequence lengths, the forward speedup decreases from about `1.92x` at `d_model=16` to about `1.39x` at `d_model=128`, and the backward speedup decreases similarly from about `1.79x` to about `1.41x`. A plausible explanation is that as `d_model` increases, matrix multiplications take a larger fraction of the runtime, leaving less relative overhead for `torch.compile` to eliminate. This also explains why very small workloads can see little benefit or even a slight slowdown, such as the `d_model=64`, `T=256` case.

### (b)
**Question:** Compile the entire Transformer model in the end-to-end benchmarking script. How does the performance of the forward pass change? What about the combined forward and backward passes and optimizer steps?

**Deliverable:** A table comparing the vanilla and compiled Transformer model.

**Answer:**

| Model size | Vanilla forward (ms) | Compiled forward (ms) | Forward speedup | Vanilla train step (ms) | Compiled train step (ms) | Train-step speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| small | 18.753 | 6.983 | 2.69x | 56.712 | 32.093 | 1.77x |
| medium | 36.779 | 21.954 | 1.68x | 115.305 | 88.104 | 1.31x |
| large | 55.654 | 48.720 | 1.14x | 226.891 | 201.570 | 1.13x |
| xl | 100.988 | 90.707 | 1.11x | 406.375 | 381.541 | 1.07x |
| 2.7b | 158.315 | 149.824 | 1.06x | 618.726 | 596.189 | 1.04x |

Compiling the full Transformer model improves forward-only performance at every tested model size, but the benefit decreases rapidly as the model grows. The forward speedup is very large for `small` (`2.69x`) and still noticeable for `medium` (`1.68x`), but it shrinks to only `1.06x` by `2.7b`. This suggests that `torch.compile` is especially helpful when a larger fraction of the forward pass is spent on Python overhead, dispatch overhead, and fusible non-GEMM operations, whereas for larger models the runtime is increasingly dominated by already highly optimized large matrix multiplications.

The train-step speedup is smaller across the board and becomes quite limited for the largest models. For example, `small` improves from `56.712 ms` to `32.093 ms` (`1.77x`), while `2.7b` improves only from `618.726 ms` to `596.189 ms` (`1.04x`). Looking at the train-step breakdown, compilation reduces forward time and helps backward somewhat, but the optimizer step is almost unchanged, so as model size increases and backward plus optimizer work take a larger share of the total iteration time, the overall end-to-end gain from compiling the model is substantially diluted.

---

## Problem `flash_benchmarking`: FlashAttention-2 Benchmarking (5 points)

### (a)
**Question:** Compare your FlashAttention-2 implementation against the PyTorch implementation using the requested settings, and report forward, backward, and end-to-end latencies.

**Deliverable:** A table of results comparing your FlashAttention-2 implementation with the PyTorch implementation, reporting forward, backward, and end-to-end latencies.

**Answer:** We benchmarked the requested sweep with batch size `1`, causal masking enabled, and fixed `q_tile_size = k_tile_size = 16`. The results show that the current FlashAttention implementation is consistently strong in FP32, especially at longer sequence lengths, and it remains runnable in the `seq_len = 65536`, FP32 cases where the regular PyTorch attention implementation runs out of memory. In BF16, however, the forward pass is still much faster but the recomputation-based PyTorch backward pass often eats away most of that gain, so the end-to-end speedups are smaller and can even reverse for larger `d` and longer sequences. The full archived table is in `artifacts/experiments/ch1/1_3_2/summary.md`.

| Configuration | PyTorch status | PyTorch forward (ms) | FlashAttention forward (ms) | PyTorch backward (ms) | FlashAttention backward (ms) | PyTorch end-to-end (ms) | FlashAttention end-to-end (ms) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| seq=128, d=16, fp32, q_tile=16, k_tile=16 | ok | 0.056 | 0.009 | 0.256 | 0.094 | 0.490 | 0.191 |
| seq=128, d=16, bf16, q_tile=16, k_tile=16 | ok | 0.043 | 0.009 | 0.233 | 0.139 | 0.447 | 0.228 |
| seq=128, d=32, fp32, q_tile=16, k_tile=16 | ok | 0.062 | 0.011 | 0.247 | 0.129 | 0.470 | 0.218 |
| seq=128, d=32, bf16, q_tile=16, k_tile=16 | ok | 0.044 | 0.010 | 0.244 | 0.179 | 0.445 | 0.276 |
| seq=128, d=64, fp32, q_tile=16, k_tile=16 | ok | 0.051 | 0.012 | 0.243 | 0.128 | 0.505 | 0.216 |
| seq=128, d=64, bf16, q_tile=16, k_tile=16 | ok | 0.043 | 0.010 | 0.230 | 0.202 | 0.430 | 0.287 |
| seq=128, d=128, fp32, q_tile=16, k_tile=16 | ok | 0.053 | 0.016 | 0.252 | 0.131 | 0.480 | 0.228 |
| seq=128, d=128, bf16, q_tile=16, k_tile=16 | ok | 0.047 | 0.012 | 0.241 | 0.194 | 0.455 | 0.377 |
| seq=256, d=16, fp32, q_tile=16, k_tile=16 | ok | 0.064 | 0.012 | 0.259 | 0.152 | 0.493 | 0.259 |
| seq=256, d=16, bf16, q_tile=16, k_tile=16 | ok | 0.045 | 0.011 | 0.240 | 0.217 | 0.453 | 0.319 |
| seq=256, d=32, fp32, q_tile=16, k_tile=16 | ok | 0.057 | 0.017 | 0.257 | 0.148 | 0.477 | 0.246 |
| seq=256, d=32, bf16, q_tile=16, k_tile=16 | ok | 0.045 | 0.014 | 0.236 | 0.186 | 0.479 | 0.282 |
| seq=256, d=64, fp32, q_tile=16, k_tile=16 | ok | 0.051 | 0.020 | 0.256 | 0.156 | 0.484 | 0.260 |
| seq=256, d=64, bf16, q_tile=16, k_tile=16 | ok | 0.045 | 0.016 | 0.236 | 0.194 | 0.449 | 0.293 |
| seq=256, d=128, fp32, q_tile=16, k_tile=16 | ok | 0.053 | 0.026 | 0.248 | 0.151 | 0.479 | 0.244 |
| seq=256, d=128, bf16, q_tile=16, k_tile=16 | ok | 0.046 | 0.018 | 0.242 | 0.190 | 0.455 | 0.280 |
| seq=512, d=16, fp32, q_tile=16, k_tile=16 | ok | 0.051 | 0.018 | 0.254 | 0.163 | 0.488 | 0.260 |
| seq=512, d=16, bf16, q_tile=16, k_tile=16 | ok | 0.049 | 0.017 | 0.259 | 0.195 | 0.450 | 0.310 |
| seq=512, d=32, fp32, q_tile=16, k_tile=16 | ok | 0.068 | 0.028 | 0.253 | 0.170 | 0.484 | 0.250 |
| seq=512, d=32, bf16, q_tile=16, k_tile=16 | ok | 0.049 | 0.024 | 0.264 | 0.212 | 0.459 | 0.297 |
| seq=512, d=64, fp32, q_tile=16, k_tile=16 | ok | 0.055 | 0.034 | 0.272 | 0.155 | 0.480 | 0.249 |
| seq=512, d=64, bf16, q_tile=16, k_tile=16 | ok | 0.049 | 0.026 | 0.246 | 0.190 | 0.444 | 0.286 |
| seq=512, d=128, fp32, q_tile=16, k_tile=16 | ok | 0.064 | 0.046 | 0.252 | 0.151 | 0.493 | 0.244 |
| seq=512, d=128, bf16, q_tile=16, k_tile=16 | ok | 0.050 | 0.031 | 0.237 | 0.201 | 0.445 | 0.302 |
| seq=1024, d=16, fp32, q_tile=16, k_tile=16 | ok | 0.071 | 0.031 | 0.255 | 0.157 | 0.490 | 0.252 |
| seq=1024, d=16, bf16, q_tile=16, k_tile=16 | ok | 0.064 | 0.029 | 0.246 | 0.211 | 0.463 | 0.303 |
| seq=1024, d=32, fp32, q_tile=16, k_tile=16 | ok | 0.072 | 0.049 | 0.255 | 0.160 | 0.491 | 0.267 |
| seq=1024, d=32, bf16, q_tile=16, k_tile=16 | ok | 0.066 | 0.042 | 0.256 | 0.200 | 0.505 | 0.324 |
| seq=1024, d=64, fp32, q_tile=16, k_tile=16 | ok | 0.075 | 0.061 | 0.267 | 0.155 | 0.487 | 0.252 |
| seq=1024, d=64, bf16, q_tile=16, k_tile=16 | ok | 0.067 | 0.046 | 0.251 | 0.195 | 0.497 | 0.287 |
| seq=1024, d=128, fp32, q_tile=16, k_tile=16 | ok | 0.088 | 0.084 | 0.261 | 0.205 | 0.492 | 0.282 |
| seq=1024, d=128, bf16, q_tile=16, k_tile=16 | ok | 0.067 | 0.055 | 0.234 | 0.215 | 0.450 | 0.309 |
| seq=2048, d=16, fp32, q_tile=16, k_tile=16 | ok | 0.148 | 0.065 | 0.347 | 0.349 | 0.495 | 0.412 |
| seq=2048, d=16, bf16, q_tile=16, k_tile=16 | ok | 0.120 | 0.057 | 0.251 | 0.352 | 0.468 | 0.406 |
| seq=2048, d=32, fp32, q_tile=16, k_tile=16 | ok | 0.153 | 0.106 | 0.352 | 0.367 | 0.506 | 0.472 |
| seq=2048, d=32, bf16, q_tile=16, k_tile=16 | ok | 0.123 | 0.084 | 0.242 | 0.360 | 0.463 | 0.444 |
| seq=2048, d=64, fp32, q_tile=16, k_tile=16 | ok | 0.168 | 0.139 | 0.378 | 0.422 | 0.545 | 0.557 |
| seq=2048, d=64, bf16, q_tile=16, k_tile=16 | ok | 0.122 | 0.097 | 0.247 | 0.395 | 0.470 | 0.490 |
| seq=2048, d=128, fp32, q_tile=16, k_tile=16 | ok | 0.211 | 0.202 | 0.449 | 0.539 | 0.658 | 0.743 |
| seq=2048, d=128, bf16, q_tile=16, k_tile=16 | ok | 0.120 | 0.126 | 0.275 | 0.479 | 0.512 | 0.601 |
| seq=4096, d=16, fp32, q_tile=16, k_tile=16 | ok | 0.611 | 0.175 | 1.379 | 1.179 | 1.971 | 1.354 |
| seq=4096, d=16, bf16, q_tile=16, k_tile=16 | ok | 0.379 | 0.140 | 0.800 | 1.155 | 1.167 | 1.293 |
| seq=4096, d=32, fp32, q_tile=16, k_tile=16 | ok | 0.632 | 0.239 | 1.401 | 1.232 | 2.016 | 1.472 |
| seq=4096, d=32, bf16, q_tile=16, k_tile=16 | ok | 0.379 | 0.195 | 0.795 | 1.164 | 1.162 | 1.356 |
| seq=4096, d=64, fp32, q_tile=16, k_tile=16 | ok | 0.677 | 0.324 | 1.472 | 1.383 | 2.148 | 1.710 |
| seq=4096, d=64, bf16, q_tile=16, k_tile=16 | ok | 0.379 | 0.220 | 0.794 | 1.246 | 1.157 | 1.465 |
| seq=4096, d=128, fp32, q_tile=16, k_tile=16 | ok | 0.807 | 0.498 | 1.702 | 1.774 | 2.513 | 2.276 |
| seq=4096, d=128, bf16, q_tile=16, k_tile=16 | ok | 0.382 | 0.302 | 0.807 | 1.514 | 1.179 | 1.814 |
| seq=8192, d=16, fp32, q_tile=16, k_tile=16 | ok | 2.391 | 0.528 | 5.145 | 4.495 | 7.521 | 5.026 |
| seq=8192, d=16, bf16, q_tile=16, k_tile=16 | ok | 1.439 | 0.423 | 2.965 | 4.347 | 4.388 | 4.770 |
| seq=8192, d=32, fp32, q_tile=16, k_tile=16 | ok | 2.359 | 0.665 | 5.087 | 4.572 | 7.433 | 5.239 |
| seq=8192, d=32, bf16, q_tile=16, k_tile=16 | ok | 1.441 | 0.525 | 2.981 | 4.283 | 4.410 | 4.811 |
| seq=8192, d=64, fp32, q_tile=16, k_tile=16 | ok | 2.534 | 0.943 | 5.432 | 5.120 | 7.968 | 6.065 |
| seq=8192, d=64, bf16, q_tile=16, k_tile=16 | ok | 1.443 | 0.608 | 2.972 | 4.579 | 4.409 | 5.187 |
| seq=8192, d=128, fp32, q_tile=16, k_tile=16 | ok | 3.176 | 1.951 | 6.448 | 6.475 | 9.608 | 8.416 |
| seq=8192, d=128, bf16, q_tile=16, k_tile=16 | ok | 1.443 | 0.886 | 2.973 | 5.435 | 4.413 | 6.316 |
| seq=16384, d=16, fp32, q_tile=16, k_tile=16 | ok | 8.876 | 1.813 | 19.551 | 17.216 | 28.419 | 19.026 |
| seq=16384, d=16, bf16, q_tile=16, k_tile=16 | ok | 5.343 | 1.427 | 11.361 | 16.627 | 16.690 | 18.053 |
| seq=16384, d=32, fp32, q_tile=16, k_tile=16 | ok | 9.160 | 2.365 | 19.850 | 17.817 | 29.011 | 20.184 |
| seq=16384, d=32, bf16, q_tile=16, k_tile=16 | ok | 5.354 | 1.813 | 11.368 | 16.716 | 16.707 | 18.533 |
| seq=16384, d=64, fp32, q_tile=16, k_tile=16 | ok | 10.733 | 3.546 | 22.113 | 20.816 | 32.862 | 24.351 |
| seq=16384, d=64, bf16, q_tile=16, k_tile=16 | ok | 5.362 | 2.152 | 11.378 | 18.681 | 16.728 | 20.834 |
| seq=16384, d=128, fp32, q_tile=16, k_tile=16 | ok | 12.116 | 5.699 | 24.897 | 24.752 | 36.985 | 30.594 |
| seq=16384, d=128, bf16, q_tile=16, k_tile=16 | ok | 5.415 | 3.205 | 11.440 | 20.599 | 16.860 | 23.798 |
| seq=32768, d=16, fp32, q_tile=16, k_tile=16 | ok | 35.058 | 7.023 | 77.385 | 68.163 | 112.499 | 75.203 |
| seq=32768, d=16, bf16, q_tile=16, k_tile=16 | ok | 20.836 | 5.294 | 44.669 | 65.122 | 65.468 | 69.949 |
| seq=32768, d=32, fp32, q_tile=16, k_tile=16 | ok | 36.265 | 8.930 | 78.505 | 70.648 | 114.825 | 79.672 |
| seq=32768, d=32, bf16, q_tile=16, k_tile=16 | ok | 21.058 | 6.919 | 44.927 | 65.500 | 65.792 | 72.368 |
| seq=32768, d=64, fp32, q_tile=16, k_tile=16 | ok | 44.648 | 12.788 | 89.900 | 85.315 | 134.556 | 99.193 |
| seq=32768, d=64, bf16, q_tile=16, k_tile=16 | ok | 21.176 | 8.147 | 45.071 | 75.868 | 66.255 | 84.324 |
| seq=32768, d=128, fp32, q_tile=16, k_tile=16 | ok | 50.127 | 22.720 | 101.028 | 101.933 | 151.025 | 124.032 |
| seq=32768, d=128, bf16, q_tile=16, k_tile=16 | ok | 21.383 | 12.148 | 45.321 | 83.164 | 66.673 | 94.959 |
| seq=65536, d=16, fp32, q_tile=16, k_tile=16 | oom | OOM | 27.822 | OOM | 174.916 | OOM | 202.090 |
| seq=65536, d=16, bf16, q_tile=16, k_tile=16 | ok | 83.389 | 20.848 | 178.910 | 160.770 | 262.257 | 180.953 |
| seq=65536, d=32, fp32, q_tile=16, k_tile=16 | oom | OOM | 35.145 | OOM | 185.920 | OOM | 220.718 |
| seq=65536, d=32, bf16, q_tile=16, k_tile=16 | ok | 83.757 | 27.270 | 179.186 | 163.407 | 263.054 | 190.309 |
| seq=65536, d=64, fp32, q_tile=16, k_tile=16 | oom | OOM | 51.491 | OOM | 232.515 | OOM | 282.778 |
| seq=65536, d=64, bf16, q_tile=16, k_tile=16 | ok | 84.618 | 32.177 | 180.245 | 191.020 | 265.125 | 223.817 |
| seq=65536, d=128, fp32, q_tile=16, k_tile=16 | oom | OOM | 91.235 | OOM | 299.592 | OOM | 390.194 |
| seq=65536, d=128, bf16, q_tile=16, k_tile=16 | ok | 86.052 | 48.535 | 182.035 | 217.778 | 268.231 | 259.729 |

---

## Problem `distributed_communication_single_node`: Distributed Communication on a Single Node (5 points)

### (a)
**Question:** Benchmark all-reduce runtime in the single-node multi-process setup while varying backend and device type, data size, and number of processes.

**Deliverable:** Plots and/or tables comparing the various settings, with 2-3 sentences of commentary about the results and how the factors interact.

**Answer:**

Results:

TODO

Commentary:

TODO

---

## Problem `naive_ddp_benchmarking`: Naive DDP Benchmarking (3 points)

### (a)
**Question:** Benchmark the language model trained with the naive DDP implementation. Measure total time per training step and the proportion of time spent communicating gradients in the single-node, 2-GPU, XL setup.

**Deliverable:** A description of the benchmarking setup, along with the measured time per training step and the proportion of time spent communicating gradients.

**Answer:**

Setup:

TODO

Results:

TODO

---

## Problem `minimal_ddp_flat_benchmarking`: Reducing the Number of Communication Calls (2 points)

### (a)
**Question:** Communicate a single flattened gradient tensor instead of issuing one all-reduce per parameter tensor. Compare the performance to the minimal DDP implementation that individually communicates gradients.

**Deliverable:** The measured time per training iteration and time spent communicating gradients, plus 1-2 sentences comparing batching versus individual communication.

**Answer:**

Results:

TODO

Comparison:

TODO

---

## Problem `ddp_overlap_individual_parameters_benchmarking`: Overlapping Computation with Communication of Individual Parameter Gradients (1 point)

### (a)
**Question:** Benchmark the DDP implementation that overlaps backward computation with communication of individual parameter gradients. Compare it against the earlier DDP baselines in the single-node, 2-GPU, XL setup.

**Deliverable:** The measured time per training iteration, with 1-2 sentences comparing the results.

**Answer:**

Results:

TODO

Comparison:

TODO

### (b)
**Question:** Use Nsight to compare the initial DDP implementation with the overlapped implementation, and visually demonstrate whether communication overlaps with the backward pass.

**Deliverable:** Two screenshots, one from each implementation, that visually show whether communication overlaps with the backward pass.

**Answer:**

Initial DDP trace:

![Initial DDP trace](TODO)

Overlapped DDP trace:

![Overlapped DDP trace](TODO)

---

## Problem `ddp_bucketed_benchmarking`: Overlapping Computation with Communication of Bucketed Parameter Gradients (3 points)

### (a)
**Question:** Benchmark bucketed DDP for bucket sizes of 1, 10, 100, and 1000 MB in the single-node, 2-GPU, XL setup. Compare against the non-bucketed baselines. Do the results align with your expectations? If not, why not? What changes in the experimental setup would you expect to make the results align better with your expectations?

**Deliverable:** Measured time per training iteration for various bucket sizes, plus 3-4 sentences of commentary.

**Answer:**

| Bucket size (MB) | Time per training iteration |
| --- | --- |
| 1 | TODO |
| 10 | TODO |
| 100 | TODO |
| 1000 | TODO |

Commentary:

TODO

### (b)
**Question:** Assume that the time to compute gradients for a bucket equals the time to communicate that bucket. Write an equation for DDP communication overhead as a function of total model size `s`, all-reduce bandwidth `w`, per-call overhead `o`, and number of buckets `n_b`. Then write the equation for the optimal bucket size.

**Deliverable:** An equation that models DDP overhead, and an equation for the optimal bucket size.

**Answer:** TODO

---

## Problem `communication_accounting`: 4D Parallelism (10 points)

### (a)
**Question:** For the XXL model, how much memory is required to store the master weights, accumulated gradients, and optimizer states in FP32 on a single device? How much memory is saved for backward in BF16? How many H100 80GB GPUs worth of memory is this?

**Deliverable:** Your calculations and a one-sentence response.

**Answer:** TODO

### (b)
**Question:** Assume master weights, optimizer state, gradients, and half of the activations are sharded across `N_FSDP` devices. Write an expression for the memory per device. What value of `N_FSDP` is needed for the total memory cost to be less than one v5p TPU device (95 GB per device)?

**Deliverable:** Your calculations and a one-sentence response.

**Answer:** TODO

### (c)
**Question:** Consider only the forward pass. Using the provided TPU v5p bandwidth and FLOP rate, with `M_X = 2`, `M_Y = 1`, `X = 16`, and `Y = 4`, at what per-device batch size is the model compute bound? What is the overall batch size in this setting?

**Deliverable:** Your calculations and a one-sentence response.

**Answer:** TODO

### (d)
**Question:** In practice, we want the overall batch size to be as small as possible while staying compute efficient instead of communication bound. What tricks can we use to reduce the batch size while retaining high throughput?

**Deliverable:** A one-paragraph response backed up with references and/or equations.

**Answer:** TODO

---

## Problem `optimizer_state_sharding_accounting`: Optimizer State Sharding (5 points)

### (a)
**Question:** Using the standard configuration (1 node, 2 GPUs, XL model size), report the peak memory usage after model initialization, directly before the optimizer step, and directly after the optimizer step, both with and without optimizer state sharding. Do the results align with your expectations? Break down the memory usage in each setting.

**Deliverable:** A 2-3 sentence response with peak memory usage results and a breakdown of the memory division across model and optimizer components.

**Answer:** TODO

### (b)
**Question:** How does optimizer state sharding affect training speed? Measure the time per iteration with and without optimizer state sharding in the standard configuration.

**Deliverable:** A 2-3 sentence response with your timings.

**Answer:** TODO

### (c)
**Question:** How does this optimizer-state-sharding approach differ from ZeRO stage 1 as described in Rajbhandari et al. (2020), especially with respect to memory and communication volume?

**Deliverable:** A 2-3 sentence summary of the differences.

**Answer:** TODO
