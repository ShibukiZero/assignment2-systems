## 1.1.4(e) Attention FLOP Estimate

Configuration:

- model size: `2.7b`
- batch size: `4`
- context length: `512`
- d_model: `2560`
- num_heads: `32`
- head_dim: `80`
- num_layers: `32`

Per-layer FLOP estimate:

| Operation | FLOPs | Ratio vs softmax |
| --- | ---: | ---: |
| attention_scores_matmul | 5,368,709,120 | 22.87x |
| attention_softmax | 234,749,952 | 1.00x |
| attention_value_matmul | 5,368,709,120 | 22.87x |
| two matmuls combined | 10,737,418,240 | 45.74x |

Whole-model FLOP estimate (all attention layers):

| Operation | FLOPs |
| --- | ---: |
| attention_scores_matmul | 171,798,691,840 |
| attention_softmax | 7,511,998,464 |
| attention_value_matmul | 171,798,691,840 |

Notes:

- GEMM FLOPs count one multiply and one add as two FLOPs.
- Softmax FLOPs use a simple scalar-op approximation of `7T - 2` per row, covering max, subtract, exp, sum, and divide.
- The comparison is most meaningful at the per-layer level, because the NVTX timing also measures one attention layer invocation at a time.
