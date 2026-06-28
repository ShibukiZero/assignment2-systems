## Problem `benchmarking_script`: Benchmarking Script (4 points)

### (b)
**Question:** Time the forward and backward passes for the model sizes described in §1.1.2. Use 5 warmup steps and compute the average and standard deviation of timings over 10 measurement steps. How long does a forward pass take? How about a backward pass? Do you see high variability across measurements, or is the standard deviation small?

**Deliverable:** A 1-2 sentence response with your timings.

**Answer:** The table below reports forward and backward latency for the five model sizes from Section 1.1.2 using 5 warmup steps and 10 measured steps at context length 128, batch size 4, vocabulary size 10,000, and FP32 precision. Forward and backward latency both increase with model size, while the standard deviations remain small relative to the means after warmup.

| Model size | Forward mean (ms) | Forward std (ms) | Backward mean (ms) | Backward std (ms) | Total mean (ms) | Total std (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| small | 18.470 | 0.433 | 16.676 | 0.165 | 35.147 | 0.409 |
| medium | 38.455 | 1.708 | 35.298 | 0.805 | 73.753 | 2.197 |
| large | 56.086 | 3.453 | 56.432 | 7.487 | 112.517 | 10.566 |
| xl | 74.703 | 1.190 | 73.215 | 1.053 | 147.919 | 2.049 |
| 2.7b | 51.443 | 2.963 | 85.604 | 0.288 | 137.047 | 3.102 |

### (c)
**Question:** One caveat of benchmarking is not performing the warm-up steps. Repeat your analysis without the warm-up steps. How does this affect your results? Why do you think this happens? Also try to run the script with 1 or 2 warm-up steps. Why might the result still be different?

**Deliverable:** A 2-3 sentence response.

**Answer:** Removing warmup makes the H100 measurements much noisier because the first measured iterations absorb CUDA runtime initialization, kernel loading, allocation, and library autotuning costs. With `warmup=0`, total latency has very large variance (`small`: `99.652 +/- 173.232 ms`, `2.7b`: `201.310 +/- 204.937 ms`), while `warmup=2` already brings the `2.7b` total close to the 5-warmup baseline (`136.822 ms` vs. `137.047 ms`). Small differences can remain because some lazy initialization and cache effects are not exhausted immediately.

---

## Problem `nsys_profile`: Nsight Systems Profiler (5 points)

### (a)
**Question:** What is the total time spent on your forward pass? Does it match what we had measured before with the Python standard library?

**Deliverable:** A 1-2 sentence response.

**Answer:** At context length 128, Nsight Systems reports forward-pass times of `24.623 ms` (`small`), `45.611 ms` (`medium`), `70.640 ms` (`large`), `91.040 ms` (`xl`), and `63.095 ms` (`2.7b`). These match the corresponding Python timing run closely: the absolute differences are `0.437-1.187 ms`, or about `1.25%-1.74%` relative error.

### (b)
**Question:** What CUDA kernel takes the most cumulative GPU time during the forward pass? How many times is this kernel invoked during a single forward pass of your model? Is it the same kernel that takes the most runtime when you do both forward and backward passes?

**Deliverable:** A 1-2 sentence response.

**Answer:** In the representative `2.7b`, context-length-512 forward trace, the top CUDA kernel is an H100 Tensor Core GEMM kernel named `sm90_xmma_gemm_f32f32_tf32f32_f32_tn_n_tilesize128x128x32_...`; it appears `3375` times across the profiled forward trace, or about `225` times per forward pass if divided over the 15 profiled passes. In the full training-step trace, the single largest cumulative kernel group is instead optimizer-related `vectorized_elementwise_kernel` time, so the top kernel is not the same once backward and AdamW are included.

### (c)
**Question:** Although the vast majority of FLOPs take place in matrix multiplications, you will notice that several other kernels still take a non-trivial amount of the overall runtime. What other kernels besides matrix multiplies do you see accounting for non-trivial CUDA runtime in the forward pass?

**Deliverable:** A 1-2 sentence response.

**Answer:** Besides GEMM kernels, the H100 forward trace shows non-trivial time in ATen `vectorized_elementwise_kernel`, `elementwise_kernel`, and `reduce_kernel` launches, including the NVTX-attributed `attention_softmax` kernels. In the representative trace these are no longer tiny: forward-scope vectorized elementwise and elementwise kernels account for about `12.3%` and `10.4%`, and the attention softmax elementwise/reduce/vectorized kernels together account for a visible share of total GPU time.

### (d)
**Question:** Profile running one complete training step with your implementation of AdamW (i.e., the forward pass, computing the loss and running a backward pass, and finally an optimizer step, as you'd do during training). How does the fraction of time spent on matrix multiplication change, compared to doing inference (forward pass only)? How about other kernels?

**Deliverable:** A 1-2 sentence response.

**Answer:** In the representative `2.7b`, context-length-512 H100 traces, kernels whose names contain `gemm`, `xmma`, or `cutlass` account for about `50.10%` of forward-only GPU time but only about `32.80%` of full training-step GPU time. The non-GEMM share therefore grows from `49.90%` to `67.10%`, mostly from backward- and optimizer-related elementwise/reduction work.

### (e)
**Question:** Compare the runtime of the softmax operation versus the matrix multiplication operations within the self-attention layer of your model during a forward pass. How does the difference in runtimes compare to the difference in FLOPs?

**Deliverable:** A 1-2 sentence response.

**Answer:** Interpreting the "self-attention layer" as the core scaled-dot-product attention, the representative `2.7b`, context-length-512 H100 NVTX trace attributes roughly `451 us` per layer to the softmax-related kernels, versus about `162 us` for the attention-score matmul and `92 us` for the attention-value matmul. The FLOP gap points the other way: each attention matmul is estimated at `5.37e9` FLOPs per layer, about `22.87x` the softmax estimate of `2.35e8` FLOPs, so softmax is much more memory- and reduction-bound than the GEMM kernels.

---

## Problem `mixed_precision_accumulation`: Mixed Precision (1 point)

### Accumulation experiment
**Question:** Run the following code and comment on the accuracy of the results.

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

**Deliverable:** A 2-3 sentence response.

**Answer:** Accumulating `0.01` in FP32 stays very close to the expected value of `10`, while accumulating in FP16 underestimates much more noticeably (`9.9531` in this run), because both the input value and the running sum are repeatedly rounded at FP16 precision. Using FP16 inputs with an FP32 accumulator is much more accurate (`10.0021` here), even though it is still slightly worse than pure FP32 because the value `0.01` has already been quantized once when it is first represented in FP16. This illustrates why mixed-precision training usually keeps reductions and accumulations in higher precision even when some inputs or matmuls use lower precision.

### 1.1.5(a) Dtypes Under Autocast
**Question:** Consider the following model. Suppose we are training the model on a GPU and that the model parameters are originally in FP32. We'd like to use autocasting mixed precision with FP16. What are the data types of:

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

**Deliverable:** The data types for each of the components listed above.

**Answer:** In the CUDA autocast check, the model parameters remain `float32`, the output of the first feed-forward layer (`fc1`) is `float16`, the output of layer norm is `float32`, the model logits are `float16`, the loss is `float32`, and the gradients are `float32`. This matches the intended mixed-precision pattern: linear layers run in lower precision where possible, while numerically sensitive normalization, loss computation, and stored parameter/gradient state stay in FP32.

### 1.1.5(b) LayerNorm and Mixed Precision
**Question:** You should have seen that FP16 mixed precision autocasting treats the layer normalization layer differently than the feed-forward layers. What parts of layer normalization are sensitive to mixed precision? If we use BF16 instead of FP16, do we still need to treat layer normalization differently? Why or why not?

**Deliverable:** A 2-3 sentence response.

**Answer:** The numerically sensitive parts of layer normalization are the mean/variance reductions, the accumulation of squared values, and the normalization step itself (subtracting the mean and dividing by the standard deviation), because these operations can amplify rounding error and are more vulnerable to overflow or underflow in low precision. With FP16, this makes it important to keep LayerNorm in higher precision. BF16 is much more stable because it has the same exponent range as FP32, so the overflow/underflow problem is much less severe, but its mantissa is still shorter than FP32, so treating LayerNorm more carefully can still improve numerical robustness.

### 1.1.5(c) BF16 Benchmarking
**Question:** Modify your benchmarking script to optionally run the model using mixed precision with BF16. Time the forward and backward passes with and without mixed-precision for each language model size described in §1.1.2. Compare the results of using full vs. mixed precision, and comment on any trends as model size changes. You may find the `nullcontext` no-op context manager to be useful.

**Deliverable:** A 2-3 sentence response with timings and commentary.

**Answer:** BF16 mixed precision provides little or no benefit on the smallest H100 workloads, but it becomes more useful as context length and model size grow. At context length `128`, the total step time changes from `37.225 ms` to `39.245 ms` for `small` (`0.95x`) and from `136.030 ms` to `127.407 ms` for `2.7b` (`1.07x`); at context length `512`, the `2.7b` total drops from `320.768 ms` to `243.307 ms` (`1.32x`). At context length `1024`, the completed `small`/`medium`/`large` runs show total speedups of `1.17x`, `1.15x`, and `1.21x`, consistent with larger matmul-heavy workloads benefiting more from lower-precision Tensor Core execution. The full timing tables are archived in `artifacts/experiments/ch1/1_1_5c/summary.md`.

---

## Problem `memory_profiling`: Memory Profiling (4 points)

### (a)
**Question:** Add an option to your profiling script to run your model through the memory profiler. It may be helpful to reuse some of your previous infrastructure (e.g., to activate mixed-precision, load specific model sizes, etc). Then, run your script to get a memory profile of the 2.7B model when either doing inference only (just forward pass) or a full training step. How do your memory timelines look like? Can you tell which stage is running based on the peaks you see?

**Deliverable:** Two images of the "Active memory timeline" of a 2.7B model, from the memory_viz tool: one for the forward pass, and one for running a full training step (forward and backward passes, then optimizer step), and a 2-3 sentence response.

**Answer:**

Forward-pass timeline:

![Forward memory timeline](artifacts/experiments/ch1/1_1_6a/fp32_2.7b_ctx512_forward_active_memory_timeline.png)

Training-step timeline:

![Training memory timeline](artifacts/experiments/ch1/1_1_6a/fp32_2.7b_ctx512_train_step_active_memory_timeline.png)

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
**Question:** Find the peak memory usage of the 2.7B model when using mixed-precision, for both a forward pass and a full optimizer step. Does mixed-precision significantly affect memory usage?

**Deliverable:** A 2-3 sentence response.

**Answer:** In this setup, BF16 does not significantly reduce measured peak memory overall. For forward-only runs, the measured peak memory is actually higher under BF16 at all three tested context lengths (`12.93 -> 19.16 GiB`, `13.02 -> 19.18 GiB`, and `13.45 -> 19.41 GiB` for context lengths `128`, `256`, and `512` respectively), while for full training steps it is nearly unchanged at shorter contexts (`51.44 -> 51.44 GiB` at `128`, `51.44 -> 52.11 GiB` at `256`) and only modestly lower at `512` (`65.52 -> 62.69 GiB`). A plausible explanation is that BF16 autocast changes the execution path rather than simply shrinking every tensor: parameters and optimizer state still remain in FP32, while extra cast/workspace buffers can be introduced during lower-precision execution, so the net peak-memory effect is small and can even be negative for forward-only runs.

### (d)
**Question:** Consider the 2.7B model. At our reference hyperparameters, what is the size of a tensor of activations in the Transformer residual stream, in single-precision? Give this size in MB (i.e., divide the number of bytes by 1024^2).

**Deliverable:** A 1-2 sentence response with your derivation.

**Answer:** For the 2.7B model, the residual-stream activation tensor at the reference hyperparameters has shape

$$
(\mathrm{batch\ size}, \mathrm{context\ length}, d_{\mathrm{model}}) = (4, 128, 2560),
$$

so it contains

$$
4 \cdot 128 \cdot 2560 = 1{,}310{,}720
$$

elements. In single precision this is

$$
1{,}310{,}720 \cdot 4 = 5{,}242{,}880 \text{ bytes} = 5.00 \text{ MiB},
$$

after dividing by $1024^2$.

### (e)
**Question:** Now look closely at the "Active Memory Timeline" from pytorch.org/memory_viz of a memory snapshot of the 2.7B model doing a forward pass. When you reduce the "Detail" level, the tool hides the smallest allocations to the corresponding level (e.g., putting "Detail" at 10% only shows the 10% largest allocations). What is the size of the largest allocations shown? Looking through the stack trace, can you tell where those allocations come from?

**Deliverable:** A 1-2 sentence response.

**Answer:** The largest allocations visible in the forward-pass memory snapshot are about `128 MiB` each. Their stack traces point to the `softmax` call inside `scaled_dot_product_attention`, which matches the size of an explicitly materialized attention score/weight tensor of shape $(\mathrm{batch}, \mathrm{heads}, \mathrm{seq\ len}, \mathrm{seq\ len})$ for the `2.7b` model at `batch=4`, `heads=32`, and `seq_len=512`:

$$
4 \cdot 32 \cdot 512 \cdot 512 \cdot 4 \text{ bytes} = 128 \text{ MiB}.
$$

So these allocations come from the naive self-attention implementation rather than the residual-stream activations.

---

## Problem `pytorch_attention`: Benchmarking PyTorch Attention (2 points)

### (a)
**Question:** Benchmark your attention implementation at different scales. Write a script that will:

- Fix the batch size to 8 and don't use multihead attention (i.e. remove the head dimension).
- Iterate through the cartesian product of `[16, 32, 64, 128]` for the head embedding dimension `d_model`, and `[256, 1024, 4096, 8192, 16384]` for the sequence length.
- Create random inputs `Q`, `K`, `V` for the appropriate size.
- Time 100 forward passes through attention using the inputs.
- Measure how much memory is in use before the backward pass starts, and time 100 backward passes.
- Make sure to warm up, and to call `torch.cuda.synchronize()` after each forward/backward pass.

Report the timings (or out-of-memory errors) you get for these configurations. At what size do you get out-of-memory errors? Do the accounting for the memory usage of attention in one of the smallest configurations you find that runs out of memory (you can use the equations for memory usage of Transformers from Assignment 1). How does the memory saved for backward change with the sequence length? What would you do to eliminate this memory cost?

**Deliverable:** A table with your timings, your working out for the memory usage, and a 1-2 paragraph response.

**Answer:**

| d_model | Sequence length | Forward timing (ms) | Backward timing (ms) | Saved for backward (GiB) | Status |
| --- | --- | ---: | ---: | ---: | --- |
| 16 | 256 | 0.213 | 0.552 | 0.004 | success |
| 16 | 1024 | 0.250 | 0.738 | 0.063 | success |
| 16 | 4096 | 2.444 | 6.065 | 1.002 | success |
| 16 | 8192 | 9.527 | 22.917 | 4.005 | success |
| 16 | 16384 | 37.306 | 90.627 | 16.009 | success |
| 32 | 256 | 0.234 | 0.731 | 0.004 | success |
| 32 | 1024 | 0.272 | 0.778 | 0.064 | success |
| 32 | 4096 | 2.519 | 6.228 | 1.004 | success |
| 32 | 8192 | 9.574 | 22.905 | 4.009 | success |
| 32 | 16384 | 37.377 | 90.560 | 16.017 | success |
| 64 | 256 | 0.217 | 0.556 | 0.004 | success |
| 64 | 1024 | 0.249 | 0.731 | 0.065 | success |
| 64 | 4096 | 2.479 | 6.085 | 1.008 | success |
| 64 | 8192 | 9.568 | 23.046 | 4.016 | success |
| 64 | 16384 | 37.364 | 90.499 | 16.033 | success |
| 128 | 256 | 0.238 | 0.582 | 0.005 | success |
| 128 | 1024 | 0.276 | 0.794 | 0.066 | success |
| 128 | 4096 | 2.521 | 6.229 | 1.016 | success |
| 128 | 8192 | 9.763 | 23.332 | 4.032 | success |
| 128 | 16384 | 38.153 | 91.563 | 16.064 | success |

No OOM was observed within the requested sweep on the H100 80GB run. The saved-for-backward memory scales approximately as `T^2` and is nearly independent of `d_model`, which is visible from the table: at `T=16384`, the saved memory is already about `16 GiB` for every tested embedding dimension. For one concrete accounting point, the `batch=8`, `T=16384`, `d_model=16` run saves about `16.009 GiB`, dominated by tensors proportional to the materialized attention matrix rather than by the small embedding dimension.

This quadratic saved-activation cost is the memory pressure that FlashAttention-style kernels avoid. The fix is to compute attention in tiles with online softmax statistics and to recompute local probabilities during backward, rather than saving the full `T x T` attention matrix for the backward pass.

---

## Problem `torch_compile`: Benchmarking JIT-Compiled Attention (2 points)

### (a)
**Question:** Extend your attention benchmarking script to include a compiled version of your PyTorch implementation of attention, and compare its performance to the uncompiled version with the same configuration as the `pytorch_attention` problem above.

**Deliverable:** A table comparing your forward and backward pass timings for your compiled attention module with the uncompiled version from the `pytorch_attention` problem above.

**Answer:**

| d_model | Sequence length | Eager forward (ms) | Compiled forward (ms) | Forward speedup | Eager backward (ms) | Compiled backward (ms) | Backward speedup |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 256 | 0.209 | 0.238 | 0.88x | 0.533 | 0.480 | 1.11x |
| 16 | 1024 | 0.248 | 0.267 | 0.93x | 0.732 | 0.521 | 1.41x |
| 16 | 4096 | 2.442 | 0.889 | 2.75x | 6.072 | 2.542 | 2.39x |
| 16 | 8192 | 9.518 | 3.205 | 2.97x | 22.943 | 9.424 | 2.43x |
| 16 | 16384 | 37.309 | 12.091 | 3.09x | 90.610 | 35.598 | 2.55x |
| 32 | 256 | 0.236 | 0.239 | 0.99x | 0.641 | 0.484 | 1.33x |
| 32 | 1024 | 0.248 | 0.269 | 0.92x | 0.731 | 0.543 | 1.35x |
| 32 | 4096 | 2.478 | 0.926 | 2.67x | 6.062 | 2.585 | 2.35x |
| 32 | 8192 | 9.588 | 3.089 | 3.10x | 22.937 | 8.908 | 2.57x |
| 32 | 16384 | 37.380 | 11.814 | 3.16x | 90.546 | 34.261 | 2.64x |
| 64 | 256 | 0.220 | 0.246 | 0.89x | 0.568 | 0.529 | 1.07x |
| 64 | 1024 | 0.247 | 0.262 | 0.94x | 0.735 | 0.530 | 1.39x |
| 64 | 4096 | 2.474 | 0.927 | 2.67x | 6.106 | 2.573 | 2.37x |
| 64 | 8192 | 9.551 | 3.083 | 3.10x | 22.984 | 8.891 | 2.59x |
| 64 | 16384 | 37.377 | 11.852 | 3.15x | 90.489 | 35.377 | 2.56x |
| 128 | 256 | 0.222 | 0.239 | 0.93x | 0.551 | 0.500 | 1.10x |
| 128 | 1024 | 0.271 | 0.263 | 1.03x | 0.770 | 0.550 | 1.40x |
| 128 | 4096 | 2.511 | 0.949 | 2.65x | 6.206 | 2.648 | 2.34x |
| 128 | 8192 | 9.755 | 3.276 | 2.98x | 23.280 | 9.212 | 2.53x |
| 128 | 16384 | 38.184 | 12.611 | 3.03x | 91.600 | 35.337 | 2.59x |

Compiling attention is most helpful for long sequences. At `T=256` the forward pass is roughly flat or slightly slower in several cases, but by `T=4096` and above the compiled forward pass is about `2.65x-3.16x` faster and the compiled backward pass is about `2.34x-2.64x` faster. The pattern is consistent with `torch.compile` reducing Python/dispatch/fusible elementwise overhead once the attention workload is large enough to amortize compilation and launch overhead.

### (b)
**Question:** Now, compile your entire Transformer model in your end-to-end benchmarking script. How does the performance of the forward pass change? What about the combined forward and backward passes and optimizer steps?

**Deliverable:** A table comparing the vanilla and compiled Transformer model.

**Answer:**

| Model size | Vanilla forward (ms) | Compiled forward (ms) | Forward speedup | Vanilla train step (ms) | Compiled train step (ms) | Train-step speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| small | 17.165 | 9.019 | 1.90x | 51.871 | 38.758 | 1.34x |
| medium | 34.243 | 18.402 | 1.86x | 107.191 | 75.511 | 1.42x |
| large | 49.533 | 27.216 | 1.82x | 165.443 | 127.681 | 1.30x |
| xl | 65.820 | 35.898 | 1.83x | 258.230 | 220.877 | 1.17x |
| 2.7b | 43.312 | 24.548 | 1.76x | 303.817 | 279.127 | 1.09x |

Compiling the full Transformer model improves forward-only performance for every tested H100 model size, with forward speedups between `1.76x` and `1.90x`. The train-step speedup is smaller, ranging from `1.34x` on `small` down to `1.09x` on `2.7b`, because backward and optimizer work take a larger share of the end-to-end iteration and are not accelerated as much as the forward pass.

---

## Problem `flash_benchmarking`: FlashAttention-2 Benchmarking (5 points)

### (a)
**Question:** Write a benchmarking script using `triton.testing.do_bench` that compares the performance of your (partially) Triton implementation of FlashAttention-2 forward and backward passes with a regular PyTorch implementation (i.e., not using FlashAttention).

Specifically, you will report a table that includes latencies for forward, backward, and the end-to-end forward-backward pass, for both your Triton and PyTorch implementations. Randomly generate any necessary inputs before you start benchmarking, and run the benchmark on a single H100. Always use batch size 1 and causal masking. Sweep over the cartesian product of sequence lengths of various powers of 2 from 128 up to 65536, embedding dimension sizes of various powers of 2 from 16 up to size 128, and precisions of `torch.bfloat16` and `torch.float32`. You will likely need to adjust tile sizes depending on the input sizes.

**Deliverable:** A table of results comparing your implementation of FlashAttention-2 with the PyTorch implementation, using the settings above and reporting forward, backward, and end-to-end latencies.

**Answer:** The requested sweep was benchmarked on a single NVIDIA H100 80GB HBM3 with batch size `1`, causal masking enabled, and fixed `q_tile_size = k_tile_size = 16`. The results show that the Triton FlashAttention implementation is faster end-to-end for the successful PyTorch comparisons, and it remains runnable at `seq_len = 65536` where the regular PyTorch implementation runs out of memory for every tested precision and head dimension. The full archived table is in `artifacts/experiments/ch1/1_3_2/summary.md`.

| Seq | D | Precision | Q tile | K tile | PT status | PT fwd (ms) | PT bwd (ms) | PT e2e (ms) | Flash status | Flash fwd (ms) | Flash bwd (ms) | Flash e2e (ms) | E2E speedup |
| ---: | ---: | --- | ---: | ---: | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 128 | 16 | fp32 | 16 | 16 | ok | 0.096 | 0.297 | 0.583 | ok | 0.009 | 0.143 | 0.224 | 2.60x |
| 128 | 16 | bf16 | 16 | 16 | ok | 0.095 | 0.298 | 0.569 | ok | 0.008 | 0.120 | 0.218 | 2.61x |
| 128 | 32 | fp32 | 16 | 16 | ok | 0.086 | 0.292 | 0.556 | ok | 0.010 | 0.117 | 0.268 | 2.08x |
| 128 | 32 | bf16 | 16 | 16 | ok | 0.126 | 0.350 | 0.586 | ok | 0.009 | 0.125 | 0.238 | 2.47x |
| 128 | 64 | fp32 | 16 | 16 | ok | 0.085 | 0.306 | 0.606 | ok | 0.011 | 0.117 | 0.240 | 2.53x |
| 128 | 64 | bf16 | 16 | 16 | ok | 0.115 | 0.283 | 0.545 | ok | 0.010 | 0.124 | 0.217 | 2.51x |
| 128 | 128 | fp32 | 16 | 16 | ok | 0.077 | 0.312 | 0.570 | ok | 0.013 | 0.113 | 0.219 | 2.61x |
| 128 | 128 | bf16 | 16 | 16 | ok | 0.094 | 0.293 | 0.573 | ok | 0.011 | 0.113 | 0.219 | 2.62x |
| 256 | 16 | fp32 | 16 | 16 | ok | 0.085 | 0.294 | 0.584 | ok | 0.011 | 0.117 | 0.232 | 2.51x |
| 256 | 16 | bf16 | 16 | 16 | ok | 0.089 | 0.287 | 0.558 | ok | 0.012 | 0.116 | 0.216 | 2.58x |
| 256 | 32 | fp32 | 16 | 16 | ok | 0.085 | 0.297 | 0.562 | ok | 0.013 | 0.118 | 0.217 | 2.59x |
| 256 | 32 | bf16 | 16 | 16 | ok | 0.093 | 0.287 | 0.567 | ok | 0.013 | 0.122 | 0.237 | 2.39x |
| 256 | 64 | fp32 | 16 | 16 | ok | 0.079 | 0.292 | 0.526 | ok | 0.017 | 0.113 | 0.215 | 2.44x |
| 256 | 64 | bf16 | 16 | 16 | ok | 0.099 | 0.280 | 0.570 | ok | 0.013 | 0.121 | 0.217 | 2.62x |
| 256 | 128 | fp32 | 16 | 16 | ok | 0.081 | 0.291 | 0.554 | ok | 0.019 | 0.113 | 0.218 | 2.55x |
| 256 | 128 | bf16 | 16 | 16 | ok | 0.087 | 0.285 | 0.579 | ok | 0.015 | 0.113 | 0.212 | 2.73x |
| 512 | 16 | fp32 | 16 | 16 | ok | 0.080 | 0.287 | 0.614 | ok | 0.017 | 0.118 | 0.218 | 2.81x |
| 512 | 16 | bf16 | 16 | 16 | ok | 0.090 | 0.292 | 0.556 | ok | 0.018 | 0.116 | 0.217 | 2.56x |
| 512 | 32 | fp32 | 16 | 16 | ok | 0.098 | 0.331 | 0.580 | ok | 0.022 | 0.125 | 0.218 | 2.66x |
| 512 | 32 | bf16 | 16 | 16 | ok | 0.087 | 0.290 | 0.562 | ok | 0.021 | 0.128 | 0.237 | 2.37x |
| 512 | 64 | fp32 | 16 | 16 | ok | 0.082 | 0.332 | 0.646 | ok | 0.027 | 0.141 | 0.214 | 3.02x |
| 512 | 64 | bf16 | 16 | 16 | ok | 0.090 | 0.281 | 0.546 | ok | 0.022 | 0.114 | 0.219 | 2.50x |
| 512 | 128 | fp32 | 16 | 16 | ok | 0.081 | 0.287 | 0.546 | ok | 0.032 | 0.115 | 0.215 | 2.53x |
| 512 | 128 | bf16 | 16 | 16 | ok | 0.097 | 0.295 | 0.537 | ok | 0.025 | 0.115 | 0.230 | 2.34x |
| 1024 | 16 | fp32 | 16 | 16 | ok | 0.079 | 0.290 | 0.547 | ok | 0.028 | 0.115 | 0.222 | 2.46x |
| 1024 | 16 | bf16 | 16 | 16 | ok | 0.094 | 0.298 | 0.580 | ok | 0.032 | 0.117 | 0.217 | 2.67x |
| 1024 | 32 | fp32 | 16 | 16 | ok | 0.085 | 0.311 | 0.548 | ok | 0.038 | 0.114 | 0.219 | 2.50x |
| 1024 | 32 | bf16 | 16 | 16 | ok | 0.093 | 0.294 | 0.618 | ok | 0.036 | 0.114 | 0.221 | 2.80x |
| 1024 | 64 | fp32 | 16 | 16 | ok | 0.088 | 0.299 | 0.599 | ok | 0.048 | 0.146 | 0.239 | 2.51x |
| 1024 | 64 | bf16 | 16 | 16 | ok | 0.097 | 0.296 | 0.590 | ok | 0.039 | 0.119 | 0.222 | 2.65x |
| 1024 | 128 | fp32 | 16 | 16 | ok | 0.086 | 0.293 | 0.559 | ok | 0.060 | 0.144 | 0.229 | 2.44x |
| 1024 | 128 | bf16 | 16 | 16 | ok | 0.093 | 0.294 | 0.580 | ok | 0.044 | 0.127 | 0.245 | 2.36x |
| 2048 | 16 | fp32 | 16 | 16 | ok | 0.110 | 0.311 | 0.564 | ok | 0.053 | 0.150 | 0.247 | 2.28x |
| 2048 | 16 | bf16 | 16 | 16 | ok | 0.098 | 0.295 | 0.601 | ok | 0.059 | 0.121 | 0.231 | 2.60x |
| 2048 | 32 | fp32 | 16 | 16 | ok | 0.115 | 0.300 | 0.587 | ok | 0.069 | 0.145 | 0.236 | 2.48x |
| 2048 | 32 | bf16 | 16 | 16 | ok | 0.103 | 0.323 | 0.567 | ok | 0.066 | 0.130 | 0.235 | 2.41x |
| 2048 | 64 | fp32 | 16 | 16 | ok | 0.113 | 0.307 | 0.558 | ok | 0.091 | 0.196 | 0.279 | 2.00x |
| 2048 | 64 | bf16 | 16 | 16 | ok | 0.103 | 0.323 | 0.649 | ok | 0.071 | 0.126 | 0.226 | 2.87x |
| 2048 | 128 | fp32 | 16 | 16 | ok | 0.116 | 0.292 | 0.549 | ok | 0.113 | 0.276 | 0.378 | 1.45x |
| 2048 | 128 | bf16 | 16 | 16 | ok | 0.094 | 0.297 | 0.565 | ok | 0.083 | 0.132 | 0.263 | 2.15x |
| 4096 | 16 | fp32 | 16 | 16 | ok | 0.410 | 0.892 | 1.293 | ok | 0.103 | 0.309 | 0.410 | 3.16x |
| 4096 | 16 | bf16 | 16 | 16 | ok | 0.275 | 0.568 | 0.838 | ok | 0.116 | 0.179 | 0.290 | 2.89x |
| 4096 | 32 | fp32 | 16 | 16 | ok | 0.411 | 0.898 | 1.301 | ok | 0.138 | 0.310 | 0.447 | 2.91x |
| 4096 | 32 | bf16 | 16 | 16 | ok | 0.274 | 0.566 | 0.828 | ok | 0.125 | 0.155 | 0.282 | 2.94x |
| 4096 | 64 | fp32 | 16 | 16 | ok | 0.417 | 0.905 | 1.316 | ok | 0.190 | 0.444 | 0.632 | 2.08x |
| 4096 | 64 | bf16 | 16 | 16 | ok | 0.274 | 0.570 | 0.843 | ok | 0.141 | 0.197 | 0.335 | 2.52x |
| 4096 | 128 | fp32 | 16 | 16 | ok | 0.424 | 0.914 | 1.335 | ok | 0.245 | 0.684 | 0.942 | 1.42x |
| 4096 | 128 | bf16 | 16 | 16 | ok | 0.276 | 0.573 | 0.841 | ok | 0.173 | 0.306 | 0.487 | 1.73x |
| 8192 | 16 | fp32 | 16 | 16 | ok | 1.504 | 3.208 | 4.707 | ok | 0.270 | 0.760 | 1.035 | 4.55x |
| 8192 | 16 | bf16 | 16 | 16 | ok | 1.022 | 2.038 | 3.049 | ok | 0.266 | 0.472 | 0.747 | 4.08x |
| 8192 | 32 | fp32 | 16 | 16 | ok | 1.505 | 3.212 | 4.708 | ok | 0.332 | 0.816 | 1.159 | 4.06x |
| 8192 | 32 | bf16 | 16 | 16 | ok | 1.024 | 2.038 | 3.056 | ok | 0.286 | 0.404 | 0.706 | 4.33x |
| 8192 | 64 | fp32 | 16 | 16 | ok | 1.517 | 3.226 | 4.741 | ok | 0.464 | 1.246 | 1.735 | 2.73x |
| 8192 | 64 | bf16 | 16 | 16 | ok | 1.021 | 2.034 | 3.051 | ok | 0.326 | 0.565 | 0.900 | 3.39x |
| 8192 | 128 | fp32 | 16 | 16 | ok | 1.523 | 3.253 | 4.774 | ok | 0.644 | 2.097 | 2.801 | 1.70x |
| 8192 | 128 | bf16 | 16 | 16 | ok | 1.031 | 2.046 | 3.076 | ok | 0.421 | 0.924 | 1.359 | 2.26x |
| 16384 | 16 | fp32 | 16 | 16 | ok | 5.753 | 12.450 | 18.195 | ok | 0.877 | 2.137 | 3.044 | 5.98x |
| 16384 | 16 | bf16 | 16 | 16 | ok | 3.773 | 7.746 | 11.508 | ok | 0.761 | 1.415 | 2.200 | 5.23x |
| 16384 | 32 | fp32 | 16 | 16 | ok | 5.761 | 12.457 | 18.211 | ok | 0.995 | 2.509 | 3.508 | 5.19x |
| 16384 | 32 | bf16 | 16 | 16 | ok | 3.778 | 7.751 | 11.525 | ok | 0.810 | 1.357 | 2.193 | 5.26x |
| 16384 | 64 | fp32 | 16 | 16 | ok | 5.789 | 12.503 | 18.290 | ok | 1.376 | 3.606 | 4.977 | 3.67x |
| 16384 | 64 | bf16 | 16 | 16 | ok | 3.789 | 7.763 | 11.546 | ok | 0.932 | 1.693 | 2.672 | 4.32x |
| 16384 | 128 | fp32 | 16 | 16 | ok | 5.808 | 12.508 | 18.326 | ok | 1.994 | 6.796 | 8.777 | 2.09x |
| 16384 | 128 | bf16 | 16 | 16 | ok | 3.806 | 7.788 | 11.600 | ok | 1.296 | 2.757 | 4.056 | 2.86x |
| 32768 | 16 | fp32 | 16 | 16 | ok | 22.702 | 49.307 | 72.001 | ok | 2.853 | 7.006 | 9.841 | 7.32x |
| 32768 | 16 | bf16 | 16 | 16 | ok | 14.890 | 30.757 | 45.635 | ok | 2.386 | 4.379 | 6.765 | 6.75x |
| 32768 | 32 | fp32 | 16 | 16 | ok | 22.699 | 49.328 | 72.018 | ok | 3.535 | 8.004 | 11.537 | 6.24x |
| 32768 | 32 | bf16 | 16 | 16 | ok | 14.874 | 30.711 | 45.583 | ok | 2.778 | 4.244 | 7.020 | 6.49x |
| 32768 | 64 | fp32 | 16 | 16 | ok | 23.205 | 49.936 | 73.082 | ok | 4.743 | 13.377 | 18.118 | 4.03x |
| 32768 | 64 | bf16 | 16 | 16 | ok | 14.877 | 30.722 | 45.607 | ok | 3.244 | 6.087 | 9.331 | 4.89x |
| 32768 | 128 | fp32 | 16 | 16 | ok | 23.427 | 50.028 | 73.469 | ok | 7.057 | 24.211 | 31.266 | 2.35x |
| 32768 | 128 | bf16 | 16 | 16 | ok | 14.933 | 30.830 | 45.796 | ok | 4.495 | 10.097 | 14.599 | 3.14x |
| 65536 | 16 | fp32 | 16 | 16 | oom | OOM | OOM | OOM | ok | 10.494 | 26.864 | 37.330 |  |
| 65536 | 16 | bf16 | 16 | 16 | oom | OOM | OOM | OOM | ok | 8.570 | 16.697 | 25.271 |  |
| 65536 | 32 | fp32 | 16 | 16 | oom | OOM | OOM | OOM | ok | 12.567 | 30.830 | 43.383 |  |
| 65536 | 32 | bf16 | 16 | 16 | oom | OOM | OOM | OOM | ok | 9.844 | 15.957 | 25.803 |  |
| 65536 | 64 | fp32 | 16 | 16 | oom | OOM | OOM | OOM | ok | 17.689 | 51.654 | 69.315 |  |
| 65536 | 64 | bf16 | 16 | 16 | oom | OOM | OOM | OOM | ok | 11.496 | 22.966 | 34.476 |  |
| 65536 | 128 | fp32 | 16 | 16 | oom | OOM | OOM | OOM | ok | 26.938 | 94.150 | 120.605 |  |
| 65536 | 128 | bf16 | 16 | 16 | oom | OOM | OOM | OOM | ok | 16.759 | 39.189 | 55.959 |  |

---

## Problem `flash_leaderboard`: FlashAttention-2 Leaderboard

**Answer:** The leaderboard configuration was benchmarked with BF16 causal attention, batch size `1`, sequence length `16384`, `16` heads, and head dimension `64`. Because the current FlashAttention interface accepts `(batch, seq, d_head)`, the benchmark flattens `batch_size * num_heads` into an effective batch of `16`.

On a single NVIDIA H100 80GB HBM3, the handout-style benchmark window (`warmup=1000 ms`, `rep=10000 ms`) measured `8.864 ms` for the combined forward-backward pass with `torch.compile` enabled. The raw benchmark payload is archived in `artifacts/experiments/ch1/flash_leaderboard/flash_leaderboard_h100_handout.json`.

---

## Problem `distributed_communication_single_node`: Distributed Communication on a Single Node (5 points)

### (a)
**Question:** Write a script to benchmark the runtime of the all-reduce operation in the single-node multi-process setup. The example code above may provide a reasonable starting point. Experiment with varying the following settings:

- Backend + device type: Gloo + CPU, NCCL + GPU.
- All-reduce data size: float32 data tensors ranging over 1MB, 10MB, 100MB, 1GB.
- Number of processes: 2, 4, or 6 processes.

Resource requirements: Up to 6 GPUs. Each benchmarking run should take less than 5 minutes.

**Deliverable:** Plot(s) and/or table(s) comparing the various settings, with 2-3 sentences of commentary about your results and thoughts about how the various factors interact.

**Answer:**

Single-node `all_reduce` was benchmarked with `5` warmup iterations and `20` measured iterations per configuration, aggregating per-iteration timings across ranks. `Gloo + CPU` and `NCCL + GPU` were compared on float32 tensors of size `1 MB`, `10 MB`, `100 MB`, and `1 GB`, while varying the number of worker processes over `2`, `4`, and `6`. The full archived summary is in `artifacts/experiments/ch2/2_1_1/summary.md`, and the raw benchmark payload is in `artifacts/experiments/ch2/2_1_1/results.json`.

Gloo + CPU:

| Processes | 1 MB (ms) | 10 MB (ms) | 100 MB (ms) | 1 GB (ms) |
| --- | ---: | ---: | ---: | ---: |
| 2 | 2.849 | 16.692 | 170.475 | 1440.573 |
| 4 | 1.199 | 10.931 | 149.470 | 1483.266 |
| 6 | 2.130 | 17.344 | 170.018 | 1602.531 |

NCCL + GPU:

| Processes | 1 MB (ms) | 10 MB (ms) | 100 MB (ms) | 1 GB (ms) |
| --- | ---: | ---: | ---: | ---: |
| 2 | 0.124 | 0.128 | 0.440 | 3.251 |
| 4 | 0.128 | 0.133 | 0.554 | 4.560 |
| 6 | 0.164 | 0.207 | 0.545 | 4.262 |

The dominant trend is that `NCCL + GPU` is consistently much faster than `Gloo + CPU`, and the gap widens sharply with message size. At `1 GB` the mean all-reduce latency is about `1441 ms` vs `3.25 ms` for `2` processes (a roughly `440x` gap) and about `1603 ms` vs `4.26 ms` for `6` processes (about `375x`); the `Gloo` numbers are dominated by host-side CPU work and PCIe/host-memory bandwidth, while `NCCL` runs the collective over NVLink directly between the H100s. Within each backend, larger tensors lead to higher latency. For `NCCL`, small messages (`1-10 MB`) are latency-bound and nearly flat at `0.1-0.2 ms`, whereas the `1 GB` case is bandwidth-bound at `3-4.5 ms`. Increasing the process count raises `NCCL` latency only modestly (e.g. `1 GB`: `3.25 -> 4.56 -> 4.26 ms` for `2/4/6` processes), which is consistent with the ring all-reduce moving an amount of data per rank that is roughly independent of the world size while incurring more per-step hops. The `Gloo + CPU` small-message numbers are noisy and not cleanly monotonic in the process count (e.g. `1 MB`: `2.85 / 1.20 / 2.13 ms` for `2/4/6`), reflecting per-call launch overhead and CPU scheduling jitter rather than a real bandwidth trend; the large-message `Gloo` numbers, where the data transfer dominates, are well-behaved.

---

## Problem `naive_ddp_benchmarking`: Naive DDP Benchmarking (3 points)

### (a)
**Question:** In this naive DDP implementation, parameters are individually all-reduced across ranks after each backward pass. To better understand the overhead of data parallel training, create a script to benchmark your previously-implemented language model when trained with this naive implementation of DDP. Measure the total time per training step and the proportion of time spent on communicating gradients. Collect measurements in the single-node setting (1 node x 2 GPUs) for the XL model size described in §1.1.2.

**Deliverable:** A description of your benchmarking setup, along with the measured time per training step and the proportion of time spent communicating gradients.

**Answer:**

Setup:

The naive DDP training loop was benchmarked in a single-node `2`-GPU configuration using the `XL` language model, `NCCL`, context length `128`, global batch size `8`, and `fp32` precision. Each run used `20` warmup iterations followed by `100` measured iterations, with timing statistics aggregated across both ranks. The archived summary is in `artifacts/experiments/ch2/2_2_naive_ddp/summary.md`, and the raw benchmark payload is in `artifacts/experiments/ch2/2_2_naive_ddp/timer_xl_ctx128_nccl_w2_gbs8_fp32.json`.

Results:

| Metric | Mean |
| --- | ---: |
| Forward + backward | 235.239 ms |
| Gradient communication | 45.578 ms |
| Optimizer step | 105.564 ms |
| Total training step | 386.385 ms |
| Communication fraction | 11.786% |

The naive DDP baseline spends about `45.6 ms` per step in explicit gradient synchronization, which corresponds to roughly `11.8%` of the total training-step time in this setting. Most of the runtime is still local computation: `forward + backward` accounts for about `60.9%` of the step, while `optimizer.step()` contributes about `27.3%`. The two ranks were also closely matched (`386.50 ms` vs `386.27 ms` mean step time), so the benchmark does not show evidence of a rank imbalance or a synchronization bug.

The Nsight Systems trace is also consistent with this timing breakdown: in the measured step, the communication phase appears as a distinct post-backward region rather than overlapping with the backward pass.

![Naive DDP Nsight Systems trace](artifacts/experiments/ch2/2_2_naive_ddp/naive%20ddp%20nsys.png)

---

## Problem `minimal_ddp_flat_benchmarking`: Reducing the Number of Communication Calls (2 points)

### (a)
**Question:** Modify your minimal DDP implementation to communicate a tensor with flattened gradients from all parameters. Compare its performance with the minimal DDP implementation that issues an all-reduce for each parameter tensor under the previously-used conditions (1 node x 2 GPUs, XL model size as described in §1.1.2).

**Deliverable:** The measured time per training iteration and time spent communicating gradients under distributed data parallel training with a single batched all-reduce call. 1-2 sentences comparing the results when batching vs. individually communicating gradients.

**Answer:**

Results:

Both the individual-gradient baseline and the flattened-gradient variant were benchmarked with the same script and the same setup: `1` node, `2` GPUs, `XL` model size, context length `128`, global batch size `8`, `fp32`, `20` warmup iterations, and `100` measured iterations. The archived comparison summary is in `artifacts/experiments/ch2/2_3_1_flat_ddp/summary.md`, and the two raw benchmark payloads are in `artifacts/experiments/ch2/2_3_1_flat_ddp/individual_baseline_xl_ctx128_nccl_w2_gbs8_fp32.json` and `artifacts/experiments/ch2/2_3_1_flat_ddp/flat_xl_ctx128_nccl_w2_gbs8_fp32.json`.

| Metric | Individual all-reduce | Flattened all-reduce |
| --- | ---: | ---: |
| Forward + backward | 214.478 ms | 211.457 ms |
| Gradient communication | 41.531 ms | 40.515 ms |
| Optimizer step | 105.486 ms | 105.376 ms |
| Total training step | 361.499 ms | 357.351 ms |
| Communication fraction | 11.489% | 11.342% |

Comparison:

Flattening all gradients into a single communication buffer reduced the measured communication time by about `1.0 ms` (`41.531 -> 40.515 ms`), which lowered the communication fraction from `11.49%` to `11.34%`. However, the end-to-end training-step time was nearly unchanged (`361.50 ms` vs `357.35 ms`), which suggests that this workload is still dominated by local computation rather than communication overhead; in addition, the flattened implementation still performs extra gradient packing and unpacking work, which likely offsets much of the communication-side gain.

The Nsight Systems CUDA HW traces support this interpretation. In the flattened implementation, the NCCL all-reduce region itself is visibly shorter than in the per-parameter baseline, but it is followed by additional memory-copy activity associated with packing and unpacking the flattened gradient buffer. This helps explain why the communication phase becomes cheaper without producing a meaningful end-to-end step-time speedup.

Naive per-parameter all-reduce CUDA HW trace:

![Naive per-parameter all-reduce CUDA HW trace](artifacts/experiments/ch2/2_3_1_flat_ddp/naive%20profiling.png)

Flattened all-reduce CUDA HW trace:

![Flattened all-reduce CUDA HW trace](artifacts/experiments/ch2/2_3_1_flat_ddp/flat%20profiling.png)

---

## Problem `ddp_overlap_individual_parameters_benchmarking`: Overlapping Computation with Communication of Individual Parameter Gradients (1 point)

### (a)
**Question:** Benchmark the performance of your DDP implementation when overlapping backward pass computation with communication of individual parameter gradients. Compare its performance with our previously-studied settings (the minimal DDP implementation that either issues an all-reduce for each parameter tensor, or a single all-reduce on the concatenation of all parameter tensors) with the same setup: 1 node, 2 GPUs, and the XL model size described in §1.1.2.

**Deliverable:** The measured time per training iteration when overlapping the backward pass with communication of individual parameter gradients, with 1-2 sentences comparing the results.

**Answer:**

Results:

The overlap-individual DDP implementation was benchmarked in the same setting as the previous experiments: `1` node, `2` GPUs, `XL` model size, context length `128`, global batch size `8`, `fp32`, `20` warmup iterations, and `100` measured iterations. The archived comparison summary is in `artifacts/experiments/ch2/2_3_2_overlap_individual/summary.md`, and the raw benchmark payloads are in `artifacts/experiments/ch2/2_3_2_overlap_individual/individual_baseline_xl_ctx128_nccl_w2_gbs8_fp32.json`, `artifacts/experiments/ch2/2_3_2_overlap_individual/flat_baseline_xl_ctx128_nccl_w2_gbs8_fp32.json`, and `artifacts/experiments/ch2/2_3_2_overlap_individual/overlap_individual_xl_ctx128_nccl_w2_gbs8_fp32.json`.

| Metric | Naive individual | Flattened | Overlap individual |
| --- | ---: | ---: | ---: |
| Forward + backward | 219.553 ms | 224.650 ms | 254.608 ms |
| Communication tail | 43.112 ms | 47.132 ms | 11.104 ms |
| Optimizer step | 105.484 ms | 105.437 ms | 105.638 ms |
| Total training step | 368.152 ms | 377.222 ms | 371.354 ms |
| Communication fraction | 11.682% | 12.415% | 2.950% |

Comparison:

Overlapping per-parameter gradient communication with backward computation reduced the post-backward communication tail dramatically, from about `43.1 ms` to about `11.1 ms` (communication fraction `11.68%` -> `2.95%`), confirming that most gradient communication was successfully hidden under the backward pass. However, the per-parameter asynchronous all-reduce hooks added overhead to the backward pass itself (`forward + backward` grew from about `219.6 ms` to about `254.6 ms`), so the end-to-end step time was roughly unchanged (`368.2 ms` -> `371.4 ms`). On H100 with fast NVLink the communication is already cheap, so the latency hidden by overlap is offset by the hook overhead -- unlike slower-interconnect settings where overlap yields a clear end-to-end speedup.

### (b)
**Question:** Instrument your benchmarking code (using the 1 node, 2 GPUs, XL model size setup) with the Nsight profiler, comparing between the initial DDP implementation and this DDP implementation that overlaps backward computation and communication. Visually compare the two traces, and provide a profiler screenshot demonstrating that one implementation overlaps compute with communication while the other doesn't.

**Deliverable:** 2 screenshots (one from the initial DDP implementation, and another from this DDP implementation that overlaps compute with communication) that visually show that communication is or isn't overlapped with the backward pass.

**Answer:**

The Nsight traces show the expected qualitative difference between the two implementations. In the naive implementation, communication is concentrated in a separate region after the backward pass has finished. In the overlap implementation, communication activity appears during the backward pass itself, which is consistent with the much smaller post-backward communication tail measured in part (a).

Initial DDP trace:

![Initial DDP trace](artifacts/experiments/ch2/2_3_2_overlap_individual/naive%20profiling.png)

Overlapped DDP trace:

![Overlapped DDP trace](artifacts/experiments/ch2/2_3_2_overlap_individual/overlap%20profiling.png)

---

## Problem `ddp_bucketed_benchmarking`: Overlapping Computation with Communication of Bucketed Parameter Gradients (3 points)

### (a)
**Question:** Benchmark your bucketed DDP implementation using the same config as the previous experiments (1 node, 2 GPUs, XL model size), varying the maximum bucket size (1, 10, 100, 1000 MB). Compare your results to the previous experiments without bucketing--do the results align with your expectations? If they don't align, why not? You may have to use the PyTorch profiler as necessary to better understand how communication calls are ordered and/or executed. What changes in the experimental setup would you expect to yield results that are aligned with your expectations?

**Deliverable:** Measured time per training iteration for various bucket sizes. 3-4 sentence commentary about the results, your expectations, and potential reasons for any mismatch.

**Answer:**

| Bucket size (MB) | Time per training iteration (ms) |
| --- | ---: |
| 1 | 347.945 |
| 10 | 337.303 |
| 100 | 352.237 |
| 1000 | 377.051 |

Commentary:

The best result is the `10 MB` bucket (`337.3 ms`). The post-backward communication tail grows with bucket size (about `9.45`, `7.13`, `13.99`, and `21.26 ms` for `1`, `10`, `100`, and `1000 MB` respectively), so larger buckets become ready later and overlap less. The `10 MB` bucket (`337.3 ms`) is faster than the naive (`368.2 ms`), flattened (`377.2 ms`), and overlapped (`371.4 ms`) baselines, consistent with bucketing both overlapping communication and reducing collective-call overhead; very small (`1 MB`) and very large (`1000 MB`) buckets are worse, matching the expected tradeoff.

Profiling note:

The profiler traces support the same interpretation. For small buckets (`1 MB` and `10 MB`), communication is broken into many shorter collectives that can start earlier during backward, while for larger buckets (`100 MB` and `1000 MB`) the communication regions are coarser and start later, which leaves a longer post-backward tail. At the same time, the bucketed implementation still performs extra packing and copying work, so reducing the number of collective calls does not fully translate into end-to-end speedup.

1 MB trace:

![1 MB bucket trace](artifacts/experiments/ch2/2_3_3_bucketed_ddp/1mb.png)

10 MB trace:

![10 MB bucket trace](artifacts/experiments/ch2/2_3_3_bucketed_ddp/10mb.png)

100 MB trace:

![100 MB bucket trace](artifacts/experiments/ch2/2_3_3_bucketed_ddp/100mb.png)

1000 MB trace:

![1000 MB bucket trace](artifacts/experiments/ch2/2_3_3_bucketed_ddp/1000mb.png)

### (b)
**Question:** Assume that the time it takes to compute the gradients for a bucket is identical to the time it takes to communicate the gradient buckets. Write an equation that models the communication overhead of DDP (i.e., the amount of additional time spent after the backward pass) as a function of the total size (bytes) of the model parameters (`s`), the all-reduce algorithm bandwidth (`w`, computed as the size of each rank's data divided by the time it takes to finish the all-reduce), the overhead (seconds) associated with each communication call (`o`), and the number of buckets (`n_b`). From this equation, write an equation for the optimal bucket size that minimizes DDP overhead.

**Deliverable:** Equation that models DDP overhead, and an equation for the optimal bucket size.

**Answer:** Let each bucket contain `s / n_b` bytes of gradients. Under the stated assumption, the time to compute one bucket of gradients equals the payload communication time for one bucket, so the payload communication time per bucket is `s / (n_b * w)`. In the ideal overlapped pipeline, the payload portion of most communication calls is hidden under the computation of later buckets, but two kinds of overhead remain visible after backward: the payload communication time of the final bucket, which has no later computation to hide behind, and the fixed launch overhead `o` for each of the `n_b` communication calls. Therefore a simple model for the post-backward DDP overhead is:

$$
T_{\text{overhead}}(n_b) = \frac{s}{n_b w} + n_b o.
$$

To minimize this expression, differentiate with respect to `n_b` and set the derivative to zero:

$$
\frac{d T_{\text{overhead}}}{d n_b} = -\frac{s}{w n_b^2} + o = 0.
$$

which gives:

$$
n_b^* = \sqrt{\frac{s}{w o}}.
$$

Since the bucket size is `b = s / n_b`, the corresponding optimal bucket size is:

$$
b^* = \frac{s}{n_b^*} = \sqrt{s w o}.
$$

---

## Problem `communication_accounting`: 4D Parallelism (10 points)

### (a)
**Question:** Consider a new model config, XXL, with `d_model=16384`, `d_ff=53248`, and `num_blocks=126`. Because for very large models, the vast majority of FLOPs are in the feedforward networks, we make some simplifying assumptions. First, we omit attention, input embeddings, and output linear layers. Then, we assume that each FFN is simply two linear layers (ignoring the activation function), where the first has input size `d_model` and output size `d_ff`, and the second has input size `d_ff` and output size `d_model`. Your model consists of `num_blocks` blocks of these two linear layers. Don't do any activation checkpointing, and keep your activations and gradient communications in BF16, while your accumulated gradients, master weights and optimizer state should be in FP32.

How much memory would it take to store the master model weights, accumulated gradients and optimizer states in FP32 on a single device? How much memory is saved for backward (these will be in BF16)? How many H100 80GB GPUs worth of memory is this?

**Deliverable:** Your calculations and a one-sentence response.

**Answer:** In the simplified FFN-only XXL model, each block has

$$
2 \cdot d_{\text{model}} \cdot d_{\text{ff}} = 1{,}744{,}830{,}464
$$

parameters, so the full model has `219,848,638,464` parameters. Storing the master weights and accumulated gradients in FP32 requires `819.0 GiB` each, while Adam optimizer state requires `1638.0 GiB`, for a total of `3276.0 GiB` of FP32 model state; this is a lower bound of about `43.97` H100 80GB GPUs even before counting saved activations. Since attention is omitted, the saved-for-backward activations are no longer quadratic in sequence length, but they still scale with token count:

$$
\mathrm{num\ blocks} \cdot B \cdot T \cdot (d_{\mathrm{model}} + d_{\mathrm{ff}}) \cdot 2
= 17{,}547{,}264 \cdot B \cdot T \text{ bytes}.
$$

So the total training-memory requirement is

$$
3{,}517{,}578{,}215{,}424 + 17{,}547{,}264 \cdot B \cdot T \text{ bytes},
$$

corresponding to

$$
\left\lceil \frac{3{,}517{,}578{,}215{,}424 + 17{,}547{,}264 \cdot B \cdot T}{80 \cdot 10^9} \right\rceil
$$

H100 80GB GPUs. For three typical values using the assignment-wide default `B = 4`, the required H100 counts are about `44.08` (`T = 128`), `44.19` (`T = 256`), and `44.42` (`T = 512`). The full derivation is archived in `artifacts/experiments/ch2/2_4_communication_accounting/question_a_summary.md`.

### (b)
**Question:** Now assume your master weights, optimizer state, gradients and half of your activations (in practice every second layer) are sharded across $N_{\mathrm{FSDP}}$ devices. Write an expression for how much memory this would take per device. What value does $N_{\mathrm{FSDP}}$ need to be for the total memory cost to be less than 1 v5p TPU (95GB per device)?

**Deliverable:** Your calculations and a one-sentence response.

**Answer:** Let $W$, $G$, $O$, and $A$ denote the memory for master weights, accumulated gradients, optimizer states, and saved activations respectively. Since the problem states that $W$, $G$, $O$, and half of the activations are sharded across $N_{\mathrm{FSDP}}$ devices, the per-device memory is

$$
M(N_{\text{FSDP}}) = \frac{W + G + O + 0.5A}{N_{\text{FSDP}}} + 0.5A.
$$

Using part (a), this becomes

$$
\begin{aligned}
\frac{3{,}517{,}578{,}215{,}424 + 0.5 \cdot (17{,}547{,}264 \cdot B \cdot T)}{N_{\text{FSDP}}}
&+ 0.5 \cdot (17{,}547{,}264 \cdot B \cdot T)
\end{aligned}
$$

bytes per device. Requiring this to be below $95 \cdot 10^9$ bytes gives

$$
N_{\text{FSDP}} >
\frac{3{,}517{,}578{,}215{,}424 + 0.5 \cdot (17{,}547{,}264 \cdot B \cdot T)}
{95 \cdot 10^9 - 0.5 \cdot (17{,}547{,}264 \cdot B \cdot T)},
$$

so the minimum valid choice is the ceiling of that expression. For three typical values using $B = 4$, the minimum values are `39` ($T = 128$), `41` ($T = 256$), and `46` ($T = 512$). The full derivation is archived in `artifacts/experiments/ch2/2_4_communication_accounting/question_b_summary.md`.

### (c)
**Question:** Consider only the forward pass. Use the communication bandwidth of $W_{\mathrm{ici}} = 2 \cdot 9 \cdot 10^{10}$ and FLOPS/s of $C = 4.6 \cdot 10^{14}$ for TPU v5p as given in the TPU Scaling Book. Following the notation of the Scaling Book, use $M_X = 2$, $M_Y = 1$ (a 3D mesh), with $X = 16$ being your FSDP dimension, and $Y = 4$ being your TP dimension. At what per-device batch size is this model compute bound? What is the overall batch size in this setting?

**Deliverable:** Your calculations and a one-sentence response.

**Answer:** Following the mixed FSDP + TP forward-pass model from the Scaling Book, the comparison is between

$$
T_{\text{math}} = \frac{4 B D F}{N C}
$$

against

$$
T_{\text{comms}} = \max(T_{\text{FSDP}}, T_{\text{TP}}),
$$

where

$$
T_{\text{FSDP}} = \frac{4 D F}{Y W_{\text{ici}} M_X}
\quad \text{and} \quad
T_{\text{TP}} = \frac{4 B D}{X W_{\text{ici}} M_Y}.
$$

Writing $b = B / N$ for the per-device token batch size and using $C = 4.6 \cdot 10^{14}$, $W_{\mathrm{ici}} = 2 \cdot 9 \cdot 10^{10}$, $Y = 4$, and $M_X = 2$, the FSDP-side compute-bound threshold is

$$
b \geq \frac{C}{Y W_{\text{ici}} M_X}
= \frac{2555.56}{4 \cdot 2}
= 319.44,
$$

tokens per device, so the minimum integer per-device batch is `320` tokens. With $N = X \cdot Y = 64$, the corresponding overall token batch threshold is $319.44 \cdot 64 = 20{,}444.44$, so the minimum integer overall batch is `20,480` tokens. The TP-side condition

$$
F \geq \left(\frac{C}{W_{\text{ici}}}\right)\left(\frac{Y}{M_Y}\right)
$$

is also satisfied because $53{,}248 > 10{,}222.22$, so the FSDP communication term is the limiting factor. The full derivation is archived in `artifacts/experiments/ch2/2_4_communication_accounting/question_c_summary.md`.

### (d)
**Question:** In practice, we want the overall batch size to be as small as possible, and we also always use our compute effectively (in other words we want to never be communication bound). What other tricks can we employ to reduce the batch size of our model but retain high throughput?

**Deliverable:** A one-paragraph response. Back up your claims with references and/or equations.

**Answer:** Part (c) already assumes an idealized overlap model, so the cleanest ways to reduce the batch size needed for high throughput are the ones that directly improve the communication terms rather than simply "adding more overlap." One option is to increase the effective communication bandwidth, i.e. to improve $M_X$ and $M_Y$ through a better topology / placement so that the collective terms shrink. A second option is to rebalance the hybrid parallelism by changing $X$ and $Y$, so that neither the FSDP nor the TP communication term dominates the other. A third option is to reduce the communication volume itself, for example with lower-precision communication or more communication-efficient collectives, which shifts the compute/communication crossover to a smaller batch. Finally, gradient accumulation is a practical engineering workaround: it does change training dynamics by increasing the effective batch per optimizer step, but it allows the instantaneous microbatch to stay small enough to fit memory while amortizing synchronization overhead across more local work. A longer summary of these tradeoffs is archived in `artifacts/experiments/ch2/2_4_communication_accounting/question_d_summary.md`.

---

## Problem `optimizer_state_sharding_accounting`: Optimizer State Sharding (5 points)

### (a)
**Question:** Create a script to profile the peak memory usage when training language models with and without optimizer state sharding. Using the standard configuration (1 node, 2 GPUs, XL model size), report the peak memory usage after model initialization, directly before the optimizer step, and directly after the optimizer step. Do the results align with your expectations? Break down the memory usage in each setting (e.g., how much memory for parameters, how much for optimizer states, etc.).

**Deliverable:** 2-3 sentence response with peak memory usage results and a breakdown of how the memory is divided between different model and optimizer components.

**Answer:** On the standard 1-node, 2-GPU, XL setup, both the full and sharded optimizers peak at about `7.62 GiB` per GPU after model initialization and about `15.26 GiB` per GPU immediately before `optimizer.step()`, which matches the theoretical `4P` and `8P` scaling for `P = 1,998,235,200` FP32 parameters. After the first optimizer step, the full optimizer reaches about `30.49 GiB` per GPU, while the sharded optimizer reaches about `22.75 GiB` on rank 1 and `22.99 GiB` on rank 0, close to the theoretical `16P = 29.78 GiB` and `(8 + 8 / N)P = 22.33 GiB` expectations for `N = 2`. The breakdown is consistent with the implementation: parameters contribute about `7.44 GiB`, gradients another `7.44 GiB`, and Adam state drops from about `14.89 GiB` in the full optimizer to about `7.33-7.56 GiB` per rank in the sharded version, with the remaining gap explained by activations, temporary buffers, and allocator overhead.

### (b)
**Question:** How does our implementation of optimizer state sharding affect training speed? Measure the time taken per iteration with and without optimizer state sharding for the standard configuration (1 node, 2 GPUs, XL model size).

**Deliverable:** 2-3 sentence response with your timings.

**Answer:** Optimizer state sharding leaves the forward-plus-backward portion essentially unchanged in this setup (`254.253 ms` without sharding versus `265.566 ms` with sharding), but it reduces optimizer-step time from `105.446 ms` to `86.341 ms`. As a result, the mean iteration time drops from `359.699 ms` to `351.907 ms`, which is a modest `1.02x` speedup (about `2.2%`). The most likely reason is that each rank now updates and maintains only about half of the Adam state, so the local optimizer step becomes cheaper and the extra post-step parameter broadcasts do not outweigh that savings at `world_size = 2`.

### (c)
**Question:** How does our approach to optimizer state sharding differ from ZeRO stage 1 (described as ZeRO-DP `P_os` in Rajbhandari et al., 2020)?

**Deliverable:** 2-3 sentence summary of any differences, especially those related to memory and communication volume.

**Answer:** This implementation matches the core idea of ZeRO stage 1 (`P_os`): optimizer states are partitioned across data-parallel ranks, while parameters and gradients remain replicated, so each rank stores only about `1 / N` of the optimizer state. The main difference is that this version is a simplified teaching implementation: it keeps full gradients resident until `step()` and then explicitly broadcasts the updated parameter shards back to all ranks, whereas ZeRO stage 1 is described as part of a broader partition-aware communication schedule designed to keep communication volume close to standard data parallel training. In addition, Rajbhandari et al. analyze mixed-precision Adam, where the optimizer state also includes FP32 master parameters, while the experiments here use the course FP32 AdamW implementation, so the exact memory formulas differ even though the source of the savings is the same.
