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

    def state_dict(self) -> dict[str, Any]:
        """
        Serialize the wrapped local optimizer state instead of the wrapper's
        empty `self.state`, while also preserving the wrapper param-group
        metadata for round-trip loading.
        """
        wrapper_state_dict = super().state_dict()
        if self._local_optimizer is None:
            return {
                "state": {},
                "param_groups": [],
                "wrapper_param_groups": wrapper_state_dict["param_groups"],
            }

        local_state_dict = self._local_optimizer.state_dict()
        return {
            "state": local_state_dict["state"],
            "param_groups": local_state_dict["param_groups"],
            "wrapper_param_groups": wrapper_state_dict["param_groups"],
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """
        Restore both the wrapped local optimizer state and the wrapper's
        full-model param-group metadata.
        """
        self._load_wrapper_param_groups(state_dict.get("wrapper_param_groups"))

        local_optimizer_state_dict = {
            "state": state_dict.get("state", {}),
            "param_groups": state_dict.get("param_groups", []),
        }
        if self._local_optimizer is None:
            if local_optimizer_state_dict["state"] or local_optimizer_state_dict["param_groups"]:
                raise ValueError("Cannot load non-empty local optimizer state on a rank with no owned parameters.")
            return

        self._local_optimizer.load_state_dict(local_optimizer_state_dict)

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

    def _load_wrapper_param_groups(self, serialized_param_groups: list[dict[str, Any]] | None) -> None:
        if serialized_param_groups is None:
            return
        if len(serialized_param_groups) != len(self.param_groups):
            raise ValueError("Loaded state_dict has a different number of wrapper parameter groups.")

        for serialized_group, current_group in zip(serialized_param_groups, self.param_groups):
            if len(serialized_group["params"]) != len(current_group["params"]):
                raise ValueError("Loaded state_dict is incompatible with the current wrapper parameter groups.")

            current_parameters = current_group["params"]
            current_group.clear()
            current_group.update({key: value for key, value in serialized_group.items() if key != "params"})
            current_group["params"] = current_parameters

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
