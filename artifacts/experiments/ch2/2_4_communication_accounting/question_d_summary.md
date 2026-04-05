# Section 2.4(d): Keeping Batch Size Small While Staying Efficient

Part (c) gives an idealized compute-bound threshold under the assumption that
communication is already perfectly overlapped with computation. To reduce the
batch size needed for high throughput, the most principled options are
therefore the ones that change the communication terms or the compute /
communication ratio itself.

First, we can improve the effective network bandwidth by increasing `M_X` and
`M_Y`, or more generally by using a topology / placement that gives the
collectives access to more parallel links. This directly reduces the FSDP and
TP communication terms in the model.

Second, we can rebalance the hybrid parallelism by changing `X` and `Y`. In
the simplified model, the FSDP communication term scales with `1 / Y`, while
the TP communication term depends on the TP-side split and the activation
traffic. Choosing a mesh shape that better balances these two terms reduces
the larger of the two communication bottlenecks, which lowers the batch size
needed to stay compute efficient.

Third, we can reduce the communication volume itself, for example by using
lower-precision communication, more aggressive compression, or communication
avoiding variants of the training stack. Since the threshold in part (c) is
set by comparing `T_math` against communication time, shrinking the
communication payload directly shifts the crossover point to a smaller batch.

Fourth, we can use gradient accumulation. This does change the effective
training dynamics because it increases the effective batch size per optimizer
step, but it is still a practical way to keep the instantaneous microbatch
small enough to fit memory while amortizing synchronization overhead across
more local work. In practice, this often allows training to remain efficient
at a smaller per-device microbatch than a single-step analysis would suggest.
