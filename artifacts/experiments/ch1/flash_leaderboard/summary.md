# FlashAttention Leaderboard Benchmark

- hardware: `single NVIDIA H100 80GB HBM3`
- benchmark window: `warmup=1000 ms`, `rep=10000 ms`
- batch size: `1`
- heads: `16`
- sequence length: `16384`
- head dimension: `64`
- precision: `bf16`
- causal: `true`
- `torch.compile`: `enabled`

| Metric | Value |
| --- | ---: |
| Forward + backward latency | 8.864 ms |

Raw payload: [`flash_leaderboard_h100_handout.json`](flash_leaderboard_h100_handout.json)
