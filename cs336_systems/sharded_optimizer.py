from __future__ import annotations

from typing import Any, Type

import torch
import torch.distributed as dist


def _normalize_param_group(param_group: dict[str, Any]) -> dict[str, Any]:
    """
    Convert a user-supplied param group into a shallow copy whose "params"
    entry is always a list. This makes later ownership bookkeeping simpler.
    """
    normalized_group = dict(param_group)
    normalized_group["params"] = list(normalized_group["params"])
    return normalized_group


class ShardedOptimizer(torch.optim.Optimizer):
    """
    Teaching skeleton for Chapter 3 optimizer-state sharding.

    High-level design:
    1. keep the base Optimizer param_groups so zero_grad() still sees the full model,
    2. assign each parameter tensor to one owner rank,
    3. build a wrapped local optimizer over only the owned parameters,
    4. after local step(), synchronize the updated parameters back to all ranks.
    """

    def __init__(
        self,
        params,
        optimizer_cls: Type[torch.optim.Optimizer],
        **optimizer_kwargs: Any,
    ):
        if not issubclass(optimizer_cls, torch.optim.Optimizer):
            raise TypeError(f"optimizer_cls must be a torch.optim.Optimizer subclass, got {optimizer_cls}.")

        self.optimizer_cls = optimizer_cls
        self.optimizer_kwargs = dict(optimizer_kwargs)
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.rank = dist.get_rank() if dist.is_initialized() else 0

        self._parameter_to_owner_rank: dict[int, int] = {}
        self._parameter_order: list[torch.nn.Parameter] = []
        self._local_param_groups: list[dict[str, Any]] = []
        self._local_optimizer: torch.optim.Optimizer | None = None
        self._is_initializing = True

        # We intentionally keep defaults empty here: the wrapped optimizer is
        # the one that owns hyperparameters such as lr / betas / eps.
        super().__init__(params, defaults={})
        self._is_initializing = False

        self._initialize_local_optimizer()

    def add_param_group(self, param_group: dict[str, Any]) -> None:
        """
        Register a full-model param group on this wrapper and the owned subset
        on the wrapped local optimizer.
        """
        normalized_group = _normalize_param_group(param_group)
        torch.optim.Optimizer.add_param_group(self, normalized_group)

        full_group = self.param_groups[-1]
        local_group = self._build_local_param_group(full_group)
        if local_group is None:
            return

        if self._is_initializing:
            self._local_param_groups.append(local_group)
            return

        if self._local_optimizer is None:
            self._local_param_groups.append(local_group)
            self._initialize_local_optimizer()
        else:
            self._local_optimizer.add_param_group(local_group)

    def step(self, closure=None, **kwargs):
        """
        Run the wrapped optimizer on the local shard, then make the updated
        parameters visible on every rank.
        """
        loss = self._step_local_optimizer(closure=closure, **kwargs)
        self._synchronize_updated_parameters()
        return loss

    def _build_local_param_group(self, full_group: dict[str, Any]) -> dict[str, Any] | None:
        owned_parameters = [
            parameter
            for parameter in full_group["params"]
            if self._get_owner_rank(parameter) == self.rank
        ]

        if not owned_parameters:
            return None

        local_group = {key: value for key, value in full_group.items() if key != "params"}
        local_group["params"] = owned_parameters
        return local_group

    def _initialize_local_optimizer(self) -> None:
        if not self._local_param_groups:
            return

        self._local_optimizer = self.optimizer_cls(self._local_param_groups, **self.optimizer_kwargs)

    def _step_local_optimizer(self, closure=None, **kwargs):
        if self._local_optimizer is None:
            return None

        if closure is not None:
            return self._local_optimizer.step(closure=closure, **kwargs)
        return self._local_optimizer.step(**kwargs)

    def _get_owner_rank(self, parameter: torch.nn.Parameter) -> int:
        parameter_id = id(parameter)
        owner_rank = self._parameter_to_owner_rank.get(parameter_id)
        if owner_rank is not None:
            return owner_rank

        owner_rank = len(self._parameter_order) % self.world_size
        self._parameter_to_owner_rank[parameter_id] = owner_rank
        self._parameter_order.append(parameter)
        return owner_rank

    def _synchronize_updated_parameters(self) -> None:
        """
        Broadcast each owner's post-step parameter values to every rank.
        """
        if not dist.is_initialized() or self.world_size == 1:
            return

        for parameter in self._iter_unique_parameters():
            owner_rank = self._get_owner_rank(parameter)
            dist.broadcast(parameter.data, src=owner_rank)

    def _iter_unique_parameters(self):
        seen_parameter_ids: set[int] = set()
        for param_group in self.param_groups:
            for parameter in param_group["params"]:
                parameter_id = id(parameter)
                if parameter_id in seen_parameter_ids:
                    continue
                seen_parameter_ids.add(parameter_id)
                yield parameter
