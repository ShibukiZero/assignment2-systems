from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

import torch
import torch.distributed as dist
import torch.nn as nn
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors


def _iter_unique_parameters(module: nn.Module) -> Iterator[nn.Parameter]:
    """Yield each parameter object once, even when weights are tied."""
    seen: set[int] = set()
    for parameter in module.parameters():
        parameter_id = id(parameter)
        if parameter_id in seen:
            continue
        seen.add(parameter_id)
        yield parameter


@dataclass
class _Bucket:
    """
    Minimal bookkeeping for one bucket of parameters.

    This is intentionally lightweight: for the first bucketed implementation,
    it is enough to know which parameters belong to the bucket, when all of
    them are ready, and which async communication handle corresponds to the
    in-flight all-reduce.
    """

    parameters: list[nn.Parameter]
    size_bytes: int
    ready_parameter_ids: set[int] = field(default_factory=set)
    pending_handle: dist.Work | None = None
    flat_grad_buffer: torch.Tensor | None = None
    grad_views: list[torch.Tensor] = field(default_factory=list)


class _BaseDDP(nn.Module):
    """
    Shared utilities for the assignment's DDP wrappers.

    All variants in Chapter 2 wrap an ordinary nn.Module, broadcast rank-0
    parameters before training, and then differ only in how they synchronize
    gradients after backward.
    """

    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self._broadcast_parameters_from_rank0()

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def finish_gradient_synchronization(self) -> None:
        raise NotImplementedError

    def _broadcast_parameters_from_rank0(self) -> None:
        if not dist.is_initialized() or self.world_size == 1:
            return

        for parameter in _iter_unique_parameters(self.module):
            dist.broadcast(parameter.data, src=0)

    def _iter_trainable_parameters(self) -> Iterator[nn.Parameter]:
        for parameter in _iter_unique_parameters(self.module):
            if parameter.requires_grad:
                yield parameter

    def _iter_trainable_parameters_with_grad(self) -> Iterator[nn.Parameter]:
        for parameter in self._iter_trainable_parameters():
            if parameter.grad is not None:
                yield parameter


class NaiveDDP(_BaseDDP):
    """
    Minimal DDP wrapper for Assignment 2 Section 2.2.

    This implementation is intentionally simple:
    1. keep a normal nn.Module as the underlying model,
    2. broadcast rank-0 weights before training,
    3. run local forward/backward on each rank's data shard,
    4. average gradients across ranks after backward,
    5. let the caller run optimizer.step().
    """

    def finish_gradient_synchronization(self) -> None:
        """
        Section 2.2 baseline: communicate each parameter gradient separately
        after the whole backward pass has finished.
        """
        if not dist.is_initialized() or self.world_size == 1:
            return

        for parameter in self._iter_trainable_parameters_with_grad():
            dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
            parameter.grad /= self.world_size


class FlatGradDDP(_BaseDDP):
    """
    Section 2.3.1 variant: flatten all gradients into one tensor, issue a
    single all-reduce, then scatter the synchronized values back into the
    original per-parameter gradient buffers.
    """

    def finish_gradient_synchronization(self) -> None:
        if not dist.is_initialized() or self.world_size == 1:
            return

        gradients = [parameter.grad for parameter in self._iter_trainable_parameters_with_grad()]
        if not gradients:
            return

        # Shape checkpoint: the flattened tensor is only a communication buffer.
        # We still restore the original gradient shapes before optimizer.step().
        flat_gradients = _flatten_dense_tensors(gradients)
        dist.all_reduce(flat_gradients, op=dist.ReduceOp.SUM)
        flat_gradients /= self.world_size

        synchronized_gradients = _unflatten_dense_tensors(flat_gradients, gradients)
        for gradient, synchronized_gradient in zip(gradients, synchronized_gradients):
            gradient.copy_(synchronized_gradient)


class OverlapIndividualGradDDP(_BaseDDP):
    """
    Section 2.3.2 variant: asynchronously all-reduce each parameter gradient
    as soon as that gradient has been accumulated in backward.

    The key ideas are:
    1. register a post-accumulate-grad hook on each unique trainable parameter,
    2. launch dist.all_reduce(..., async_op=True) inside the hook,
    3. wait on all pending handles right before optimizer.step().
    """

    def __init__(self, module: nn.Module):
        super().__init__(module)
        self._pending_gradient_syncs: list[tuple[nn.Parameter, dist.Work]] = []
        self._hook_handles: list[torch.utils.hooks.RemovableHandle] = []
        self._register_gradient_hooks()

    def finish_gradient_synchronization(self) -> None:
        if not dist.is_initialized() or self.world_size == 1:
            return

        for parameter, handle in self._pending_gradient_syncs:
            handle.wait()
            if parameter.grad is not None:
                parameter.grad /= self.world_size
        self._pending_gradient_syncs.clear()

    def _register_gradient_hooks(self) -> None:
        for parameter in self._iter_trainable_parameters():
            self._hook_handles.append(parameter.register_post_accumulate_grad_hook(self._make_overlap_hook(parameter)))

    def _make_overlap_hook(self, parameter: nn.Parameter):
        def _hook(_unused_param: nn.Parameter) -> None:
            if not dist.is_initialized() or self.world_size == 1:
                return
            if parameter.grad is None:
                return

            # Launch communication immediately when this gradient becomes ready.
            # We wait only once, right before optimizer.step().
            handle = dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM, async_op=True)
            self._pending_gradient_syncs.append((parameter, handle))

        return _hook


class BucketedGradDDP(_BaseDDP):
    """
    Section 2.3.3 implementation: bucket parameters, launch one async
    all-reduce when a whole bucket becomes ready, and wait before optimizer.step().

    This is the most direct bucketed version:
    1. build buckets in reverse parameter order,
    2. mark parameters as ready through post-accumulate-grad hooks,
    3. flatten a ready bucket into one temporary communication buffer,
    4. asynchronously all-reduce that buffer,
    5. wait and scatter the synchronized gradients back before optimizer.step().
    """

    def __init__(self, module: nn.Module, bucket_size_mb: float):
        self.bucket_size_mb = bucket_size_mb
        super().__init__(module)
        self.buckets = self._build_buckets(bucket_size_mb)
        self.parameter_to_bucket = {
            id(parameter): bucket
            for bucket in self.buckets
            for parameter in bucket.parameters
        }
        self._hook_handles: list[torch.utils.hooks.RemovableHandle] = []
        self._register_bucket_hooks()

    def on_train_batch_start(self) -> None:
        """
        Reset per-bucket bookkeeping at the start of each training step.

        tests/test_ddp.py calls this hook explicitly before zero_grad().
        """
        for bucket in self.buckets:
            self._reset_bucket_runtime_state(bucket)

    def finish_gradient_synchronization(self) -> None:
        if not dist.is_initialized() or self.world_size == 1:
            return

        for bucket in self.buckets:
            if bucket.pending_handle is not None:
                bucket.pending_handle.wait()
                if bucket.flat_grad_buffer is not None:
                    bucket.flat_grad_buffer /= self.world_size
                    synchronized_gradients = _unflatten_dense_tensors(bucket.flat_grad_buffer, bucket.grad_views)
                    for grad_view, synchronized_gradient in zip(bucket.grad_views, synchronized_gradients):
                        grad_view.copy_(synchronized_gradient)
                self._reset_bucket_runtime_state(bucket)

    def _build_buckets(self, bucket_size_mb: float) -> list[_Bucket]:
        """
        Build buckets using reverse parameter order, as suggested by the handout.

        This part is intentionally fully implemented because it is mostly
        bookkeeping rather than the core communication logic.
        """
        max_bucket_bytes = int(bucket_size_mb * 1024 * 1024)
        if max_bucket_bytes <= 0:
            raise ValueError(f"bucket_size_mb must be positive, got {bucket_size_mb}.")

        buckets: list[_Bucket] = []
        current_bucket_parameters: list[nn.Parameter] = []
        current_bucket_bytes = 0

        for parameter in reversed(list(self._iter_trainable_parameters())):
            parameter_bytes = parameter.numel() * parameter.element_size()

            if current_bucket_parameters and current_bucket_bytes + parameter_bytes > max_bucket_bytes:
                buckets.append(_Bucket(parameters=current_bucket_parameters, size_bytes=current_bucket_bytes))
                current_bucket_parameters = []
                current_bucket_bytes = 0

            current_bucket_parameters.append(parameter)
            current_bucket_bytes += parameter_bytes

        if current_bucket_parameters:
            buckets.append(_Bucket(parameters=current_bucket_parameters, size_bytes=current_bucket_bytes))

        return buckets

    def _register_bucket_hooks(self) -> None:
        for parameter in self._iter_trainable_parameters():
            self._hook_handles.append(parameter.register_post_accumulate_grad_hook(self._make_bucket_hook(parameter)))

    def _make_bucket_hook(self, parameter: nn.Parameter):
        def _hook(_unused_param: nn.Parameter) -> None:
            if not dist.is_initialized() or self.world_size == 1:
                return
            if parameter.grad is None:
                return

            bucket = self.parameter_to_bucket[id(parameter)]
            bucket.ready_parameter_ids.add(id(parameter))
            if len(bucket.ready_parameter_ids) == len(bucket.parameters) and bucket.pending_handle is None:
                self._launch_bucket_async_allreduce(bucket)

        return _hook

    def _launch_bucket_async_allreduce(self, bucket: _Bucket) -> None:
        """
        Start one async all-reduce for a ready bucket.

        This implementation uses the most direct pack-and-launch path:
        1. gather the ready per-parameter gradients,
        2. flatten them into one temporary communication buffer,
        3. launch a single async all-reduce for that buffer,
        4. store the runtime state on the bucket for finish_gradient_synchronization().
        """
        gradients = [parameter.grad for parameter in bucket.parameters if parameter.grad is not None]
        bucket.grad_views = gradients
        bucket.flat_grad_buffer = _flatten_dense_tensors(gradients)
        bucket.pending_handle = dist.all_reduce(bucket.flat_grad_buffer, op=dist.ReduceOp.SUM, async_op=True)

    def _reset_bucket_runtime_state(self, bucket: _Bucket) -> None:
        bucket.ready_parameter_ids.clear()
        bucket.pending_handle = None
        bucket.flat_grad_buffer = None
        bucket.grad_views.clear()




def naive_ddp_train_step(
    *,
    ddp_model: _BaseDDP,
    optimizer: torch.optim.Optimizer,
    loss_fn,
    inputs: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """
    One training-step skeleton showing where Section 2.2 communication fits.

    The important ordering is:
    forward -> loss -> backward -> all-reduce grads -> optimizer.step()
    """

    optimizer.zero_grad()

    outputs = ddp_model(inputs)
    loss = loss_fn(outputs, targets)
    loss.backward()

    ddp_model.finish_gradient_synchronization()

    optimizer.step()
    return loss
