from __future__ import annotations

from collections.abc import Iterator

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
