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

**Deliverable:** A 2-3 sentence response.

**Answer:** TODO

### 1.1.5(a) Dtypes Under Autocast
**Question:** For the toy model under FP16 autocast, what are the data types of:

- the model parameters within the autocast context,
- the output of the first feed-forward layer,
- the output of layer norm,
- the model's predicted logits,
- the loss,
- and the model's gradients?

**Deliverable:** The data types for each of the listed components.

**Answer:** TODO

### 1.1.5(b) LayerNorm and Mixed Precision
**Question:** What parts of layer normalization are sensitive to mixed precision? If we use BF16 instead of FP16, do we still need to treat layer normalization differently? Why or why not?

**Deliverable:** A 2-3 sentence response.

**Answer:** TODO

### 1.1.5(c) BF16 Benchmarking
**Question:** Modify the benchmarking script to optionally run with BF16 mixed precision. Time the forward and backward passes with and without mixed precision for each language model size in Section 1.1.2. Compare the results and comment on any trends as model size changes.

**Deliverable:** A 2-3 sentence response with timings and commentary.

**Answer:** TODO

---

## Problem `memory_profiling`: Memory Profiling (4 points)

### (a)
**Question:** Run the memory profiler on the 2.7B model for inference-only and for a full training step. How do the active memory timelines look? Can you tell which stage is running based on the peaks?

**Deliverable:** Two images of the active memory timeline of a 2.7B model, one for the forward pass and one for a full training step, plus a 2-3 sentence response.

**Answer:**

Forward-pass timeline:

![Forward memory timeline](TODO)

Training-step timeline:

![Training memory timeline](TODO)

Response:

TODO

### (b)
**Question:** What is the peak memory usage of each context length when doing a forward pass? What about when doing a full training step?

**Deliverable:** A table with two numbers per context length.

**Answer:**

| Context length | Forward peak memory | Full training step peak memory |
| --- | --- | --- |
| 128 | TODO | TODO |
| 256 | TODO | TODO |
| 512 | TODO | TODO |

### (c)
**Question:** Find the peak memory usage of the 2.7B model when using mixed precision, for both a forward pass and a full optimizer step. Does mixed precision significantly affect memory usage?

**Deliverable:** A 2-3 sentence response.

**Answer:** TODO

### (d)
**Question:** At the reference hyperparameters for the 2.7B model, what is the size of one tensor of activations in the Transformer residual stream, in single precision? Give the size in MB.

**Deliverable:** A 1-2 sentence response with your derivation.

**Answer:** TODO

### (e)
**Question:** At lower detail in the active memory timeline, what is the size of the largest allocations shown? Looking through the stack trace, where do those allocations come from?

**Deliverable:** A 1-2 sentence response.

**Answer:** TODO

---

## Problem `pytorch_attention`: Benchmarking PyTorch Attention (2 points)

### (a)
**Question:** Report the timings or out-of-memory errors for the requested attention configurations. At what size do you get out-of-memory errors? Do the memory accounting for one of the smallest configurations that runs out of memory. How does the memory saved for backward change with sequence length? What would you do to eliminate this memory cost?

**Deliverable:** A table with timings, your memory-usage working, and a 1-2 paragraph response.

**Answer:**

| d_model | Sequence length | Forward timing | Backward timing | Memory before backward | Status |
| --- | --- | --- | --- | --- | --- |
| 16 | 256 | TODO | TODO | TODO | TODO |
| 16 | 1024 | TODO | TODO | TODO | TODO |
| 16 | 4096 | TODO | TODO | TODO | TODO |
| 16 | 8192 | TODO | TODO | TODO | TODO |
| 16 | 16384 | TODO | TODO | TODO | TODO |
| 32 | 256 | TODO | TODO | TODO | TODO |
| 32 | 1024 | TODO | TODO | TODO | TODO |
| 32 | 4096 | TODO | TODO | TODO | TODO |
| 32 | 8192 | TODO | TODO | TODO | TODO |
| 32 | 16384 | TODO | TODO | TODO | TODO |
| 64 | 256 | TODO | TODO | TODO | TODO |
| 64 | 1024 | TODO | TODO | TODO | TODO |
| 64 | 4096 | TODO | TODO | TODO | TODO |
| 64 | 8192 | TODO | TODO | TODO | TODO |
| 64 | 16384 | TODO | TODO | TODO | TODO |
| 128 | 256 | TODO | TODO | TODO | TODO |
| 128 | 1024 | TODO | TODO | TODO | TODO |
| 128 | 4096 | TODO | TODO | TODO | TODO |
| 128 | 8192 | TODO | TODO | TODO | TODO |
| 128 | 16384 | TODO | TODO | TODO | TODO |

Memory accounting:

TODO

Response:

TODO

---

## Problem `torch_compile`: Benchmarking JIT-Compiled Attention (2 points)

### (a)
**Question:** Compare a compiled version of the PyTorch attention implementation against the uncompiled version under the same configuration as the previous attention benchmark.

**Deliverable:** A table comparing forward and backward timings for the compiled attention module with the uncompiled version.

**Answer:**

| Configuration | Uncompiled forward | Compiled forward | Uncompiled backward | Compiled backward |
| --- | --- | --- | --- | --- |
| TODO | TODO | TODO | TODO | TODO |

### (b)
**Question:** Compile the entire Transformer model in the end-to-end benchmarking script. How does the performance of the forward pass change? What about the combined forward and backward passes and optimizer steps?

**Deliverable:** A table comparing the vanilla and compiled Transformer model.

**Answer:**

| Configuration | Vanilla forward | Compiled forward | Vanilla train step | Compiled train step |
| --- | --- | --- | --- | --- |
| TODO | TODO | TODO | TODO | TODO |

---

## Problem `flash_benchmarking`: FlashAttention-2 Benchmarking (5 points)

### (a)
**Question:** Compare your FlashAttention-2 implementation against the PyTorch implementation using the requested settings, and report forward, backward, and end-to-end latencies.

**Deliverable:** A table of results comparing your FlashAttention-2 implementation with the PyTorch implementation, reporting forward, backward, and end-to-end latencies.

**Answer:**

| Configuration | PyTorch forward | FlashAttention forward | PyTorch backward | FlashAttention backward | PyTorch end-to-end | FlashAttention end-to-end |
| --- | --- | --- | --- | --- | --- | --- |
| TODO | TODO | TODO | TODO | TODO | TODO | TODO |

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
