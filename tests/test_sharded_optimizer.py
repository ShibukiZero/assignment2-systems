from copy import deepcopy
from typing import Type

import numpy
import pytest
import torch
import torch.multiprocessing as mp

from .adapters import get_sharded_optimizer
from .common import (
    ToyModel,
    ToyModelWithTiedWeights,
    _cleanup_process_group,
    _setup_process_group,
)


@pytest.mark.parametrize("model_class", [ToyModel, ToyModelWithTiedWeights])
def test_sharded_optimizer(model_class):
    world_size = 2
    mp.spawn(
        _test_sharded_optimizer,
        args=(world_size, model_class),
        nprocs=world_size,
        join=True,
    )


@pytest.mark.parametrize("model_class", [ToyModel, ToyModelWithTiedWeights])
def test_sharded_optimizer_state_dict_round_trip(model_class):
    world_size = 2
    mp.spawn(
        _test_sharded_optimizer_state_dict_round_trip,
        args=(world_size, model_class),
        nprocs=world_size,
        join=True,
    )


def _test_sharded_optimizer(rank: int, world_size: int, model_class: Type[torch.nn.Module]):
    # Use gloo backend for CPU
    device = _setup_process_group(rank=rank, world_size=world_size, backend="gloo")
    torch.manual_seed(42)
    optimizer_cls = torch.optim.AdamW
    # Since we've seeded, model states should be the same across ranks without having to broadcast.
    non_sharded_model = model_class().to(device)

    non_sharded_optimizer = optimizer_cls(
        non_sharded_model.parameters(),
        lr=0.1,
        weight_decay=0.1,
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    sharded_model = deepcopy(non_sharded_model)
    sharded_optimizer = get_sharded_optimizer(
        sharded_model.parameters(),
        optimizer_cls,
        lr=0.1,
        weight_decay=0.1,
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    for _ in range(10):
        non_sharded_optimizer.zero_grad()
        sharded_optimizer.zero_grad()

        # batch size 32, 10 input features, 5 output features
        input_ = torch.rand((32, 10)).to(device)
        labels = torch.rand((32, 5)).to(device)
        non_sharded_input = deepcopy(input_)
        sharded_input = deepcopy(input_)
        non_sharded_labels = deepcopy(labels)
        sharded_labels = deepcopy(labels)

        non_sharded_model_logits = non_sharded_model(non_sharded_input)
        sharded_model_logits = sharded_model(sharded_input)

        non_sharded_model_loss = ((non_sharded_labels - non_sharded_model_logits) ** 2).sum()
        sharded_model_loss = ((sharded_labels - sharded_model_logits) ** 2).sum()

        non_sharded_model_loss.backward()
        sharded_model_loss.backward()

        non_sharded_optimizer.step()
        sharded_optimizer.step()

    # Check that the final model weights are the same regardless of if we're using
    # the sharded or non-sharded optimizer.
    for non_sharded_parameters, sharded_parameters in zip(non_sharded_model.parameters(), sharded_model.parameters()):
        numpy.testing.assert_allclose(
            non_sharded_parameters.detach().cpu().numpy(),
            sharded_parameters.detach().cpu().numpy(),
        )
    _cleanup_process_group()


def _take_optimizer_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    input_: torch.Tensor,
    labels: torch.Tensor,
) -> None:
    optimizer.zero_grad()
    logits = model(input_)
    loss = ((labels - logits) ** 2).sum()
    loss.backward()
    optimizer.step()


def _test_sharded_optimizer_state_dict_round_trip(rank: int, world_size: int, model_class: Type[torch.nn.Module]):
    device = _setup_process_group(rank=rank, world_size=world_size, backend="gloo")
    torch.manual_seed(42)
    optimizer_cls = torch.optim.AdamW

    model = model_class().to(device)
    optimizer = get_sharded_optimizer(
        model.parameters(),
        optimizer_cls,
        lr=0.1,
        weight_decay=0.1,
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    for _ in range(3):
        input_ = torch.rand((32, 10)).to(device)
        labels = torch.rand((32, 5)).to(device)
        _take_optimizer_step(model, optimizer, input_, labels)

    resumed_model = deepcopy(model)
    resumed_optimizer = get_sharded_optimizer(
        resumed_model.parameters(),
        optimizer_cls,
        lr=0.1,
        weight_decay=0.1,
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    resumed_optimizer.load_state_dict(deepcopy(optimizer.state_dict()))

    next_input = torch.rand((32, 10)).to(device)
    next_labels = torch.rand((32, 5)).to(device)

    _take_optimizer_step(model, optimizer, deepcopy(next_input), deepcopy(next_labels))
    _take_optimizer_step(resumed_model, resumed_optimizer, deepcopy(next_input), deepcopy(next_labels))

    for original_parameter, resumed_parameter in zip(model.parameters(), resumed_model.parameters()):
        numpy.testing.assert_allclose(
            original_parameter.detach().cpu().numpy(),
            resumed_parameter.detach().cpu().numpy(),
        )

    _cleanup_process_group()
