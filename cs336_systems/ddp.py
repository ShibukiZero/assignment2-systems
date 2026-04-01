from __future__ import annotations

from collections.abc import Iterator

import torch
import torch.distributed as dist
import torch.nn as nn


def _iter_unique_parameters(module: nn.Module) -> Iterator[nn.Parameter]:
    """Yield each parameter object once, even when weights are tied."""
    seen: set[int] = set()
    for parameter in module.parameters():
        parameter_id = id(parameter)
        if parameter_id in seen:
            continue
        seen.add(parameter_id)
        yield parameter


class NaiveDDP(nn.Module):
    """
    Minimal DDP skeleton for Assignment 2 Section 2.2.

    This wrapper is intentionally simple:
    1. keep a normal nn.Module as the underlying model,
    2. broadcast rank-0 weights before training,
    3. run local forward/backward on each rank's data shard,
    4. average gradients across ranks after backward,
    5. let the caller run optimizer.step().

    The TODO blocks mark the distributed pieces you are expected to fill in.
    """

    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.rank = dist.get_rank() if dist.is_initialized() else 0

        # TODO(student): synchronize parameters from rank 0 to every other rank.
        # Hint: iterate over parameters (and optionally buffers) and call
        # dist.broadcast(..., src=0).
        #
        # Questions to sanity-check yourself:
        # - Why must this happen before the first optimizer step?
        # - Which tensors should *not* be skipped, even if they do not require grad?
        self._broadcast_parameters_from_rank0()

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def finish_gradient_synchronization(self) -> None:
        """
        Naive Section 2.2 version: gradients are communicated only after the
        entire backward pass finishes, so this method can stay synchronous.
        """

        # TODO(student): average gradients across ranks.
        # Expected shape of the logic:
        #   for parameter in self._iter_trainable_parameters():
        #       if parameter.grad is None:
        #           continue
        #       dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
        #       parameter.grad /= self.world_size
        #
        # Questions to sanity-check yourself:
        # - Why do we average instead of leaving the summed gradients as-is?
        # - What happens for parameters with requires_grad=False?
        # - What happens for tied weights if you do not deduplicate by identity?
        # raise NotImplementedError("TODO: average parameter gradients across ranks after backward.")
        for parameter in self._iter_trainable_parameters():
            if parameter.grad is None:
                continue
            dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
            parameter.grad /= self.world_size

    def _broadcast_parameters_from_rank0(self) -> None:
        # TODO(student): implement the initial rank-0 parameter broadcast.
        # raise NotImplementedError("TODO: broadcast initial parameters from rank 0.")
        for parameter in _iter_unique_parameters(self.module):
            dist.broadcast(parameter.data, src=0)


    def _iter_trainable_parameters(self) -> Iterator[nn.Parameter]:
        for parameter in _iter_unique_parameters(self.module):
            if parameter.requires_grad:
                yield parameter


def naive_ddp_train_step(
    *,
    ddp_model: NaiveDDP,
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

    # TODO(student): in Section 2.2 this is where the naive gradient averaging happens.
    ddp_model.finish_gradient_synchronization()

    optimizer.step()
    return loss
