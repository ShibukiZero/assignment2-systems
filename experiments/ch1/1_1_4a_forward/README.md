# 1.1.4(a) Forward Profiling Artifacts

Each profiled configuration should produce:

- `<tag>.json`: benchmark script output from `cs336_systems.benchmark`
- `<tag>.nsys-rep`: Nsight Systems report for later inspection in the GUI

Recommended tag format:

- `<model_size>_ctx<context_length>_forward`

Examples:

- `small_ctx128_forward.json`
- `small_ctx128_forward.nsys-rep`

Suggested comparison table columns for the writeup:

- model size
- context length
- Python benchmark forward mean (ms)
- Nsight forward time (ms)
- note
