# Experiments

This directory stores experiment artifacts that support writeup claims and are worth keeping beyond temporary log review.

## Organization

- `ch1/`: profiling and benchmarking artifacts for Chapter 1

## Chapter 1

- `1_1_3b/`: forward/backward benchmark JSON logs and summary table
- `1_1_3c/`: warmup sensitivity comparisons

Temporary, ad hoc terminal output should remain in `.agents/logs/`.

Note:

- Large profiler binaries such as `.nsys-rep`, `.qdrep`, and `.qdstrm` are kept locally for inspection but ignored by git to avoid bloating the repository history.
- Until an experiment's conclusions are actually written into `writeup.md`, keep its artifacts out of `experiments/` and in temporary locations such as `.agents/logs/`.
