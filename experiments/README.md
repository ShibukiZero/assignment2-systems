# Experiments

This directory stores experiment artifacts that support writeup claims and are worth keeping beyond temporary log review.

## Organization

- `ch1/`: profiling and benchmarking artifacts for Chapter 1

## Chapter 1

- `1_1_3b/`: forward/backward benchmark JSON logs and summary table
- `1_1_3c/`: warmup sensitivity comparisons
- `1_1_4a_forward/`: forward-only Nsight collection, including benchmark JSON logs and `.nsys-rep` files

Temporary, ad hoc terminal output should remain in `.agents/logs/`.

Note:

- Large profiler binaries such as `.nsys-rep`, `.qdrep`, and `.qdstrm` are kept locally for inspection but ignored by git to avoid bloating the repository history.
