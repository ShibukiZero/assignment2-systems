from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.optimizer import AdamW
from cs336_systems.sharded_optimizer import ShardedOptimizer


@dataclass(frozen=True)
class ModelPreset:
    d_model: int
    d_ff: int
    num_layers: int
    num_heads: int


MODEL_PRESETS: dict[str, ModelPreset] = {
    "small": ModelPreset(d_model=768, d_ff=3072, num_layers=12, num_heads=12),
    "medium": ModelPreset(d_model=1024, d_ff=4096, num_layers=24, num_heads=16),
    "large": ModelPreset(d_model=1280, d_ff=5120, num_layers=36, num_heads=20),
    "xl": ModelPreset(d_model=1600, d_ff=6400, num_layers=48, num_heads=25),
    "2.7b": ModelPreset(d_model=2560, d_ff=10240, num_layers=32, num_heads=32),
}


@dataclass(frozen=True)
class BenchmarkConfig:
    mode: str
    optimizer_mode: str
    model_size: str
    context_length: int
    global_batch_size: int
    vocab_size: int
    rope_theta: float
    precision: str
    backend: str
    world_size: int
    device: str
    warmup_steps: int
    measure_steps: int
    learning_rate: float
    weight_decay: float
    seed: int
    master_addr: str
    master_port: int
    output_path: str | None


def parse_args() -> BenchmarkConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark Chapter 3 optimizer state sharding with a replicated model "
            "and explicit gradient averaging across ranks."
        )
    )
    parser.add_argument("--mode", choices=("memory", "timer"), default="memory")
    parser.add_argument(
        "--optimizer-mode",
        choices=("full", "sharded"),
        default="full",
        help="Compare a standard replicated AdamW optimizer against the sharded wrapper.",
    )
    parser.add_argument("--model-size", choices=tuple(MODEL_PRESETS), default="xl")
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument("--global-batch-size", type=int, default=8)
    parser.add_argument("--vocab-size", type=int, default=10_000)
    parser.add_argument("--rope-theta", type=float, default=10_000.0)
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--backend", choices=("gloo", "nccl"), default="nccl")
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--measure-steps", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--master-addr", type=str, default="127.0.0.1")
    parser.add_argument("--master-port", type=int, default=29510)
    parser.add_argument("--output-path", type=str, default=None)
    args = parser.parse_args()
    return BenchmarkConfig(
        mode=args.mode,
        optimizer_mode=args.optimizer_mode,
        model_size=args.model_size,
        context_length=args.context_length,
        global_batch_size=args.global_batch_size,
        vocab_size=args.vocab_size,
        rope_theta=args.rope_theta,
        precision=args.precision,
        backend=args.backend,
        world_size=args.world_size,
        device=args.device,
        warmup_steps=args.warmup_steps,
        measure_steps=args.measure_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seed=args.seed,
        master_addr=args.master_addr,
        master_port=args.master_port,
        output_path=args.output_path,
    )


def resolve_device(config: BenchmarkConfig, rank: int) -> torch.device:
    visible_device_count = torch.cuda.device_count()

    if config.device == "cpu":
        return torch.device("cpu")
    if config.device == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA was requested, but no CUDA device is available.")
        if visible_device_count <= rank:
            raise ValueError(
                f"rank {rank} requested CUDA device, but only {visible_device_count} visible devices exist."
            )
        torch.cuda.set_device(rank)
        return torch.device(f"cuda:{rank}")

    if config.backend == "nccl":
        if not torch.cuda.is_available():
            raise ValueError("NCCL benchmarking requires CUDA.")
        if visible_device_count <= rank:
            raise ValueError(
                f"rank {rank} requested CUDA device, but only {visible_device_count} visible devices exist."
            )
        torch.cuda.set_device(rank)
        return torch.device(f"cuda:{rank}")
    return torch.device("cpu")


def make_precision_context(config: BenchmarkConfig, device: torch.device):
    if config.precision == "fp32":
        return nullcontext()
    if device.type != "cuda":
        raise ValueError("BF16 autocast is only supported on CUDA in this script.")
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)


def maybe_synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def summarize_series(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean_ms": 0.0, "stdev_ms": 0.0, "min_ms": 0.0, "max_ms": 0.0}
    stdev = statistics.stdev(values) if len(values) > 1 else 0.0
    return {
        "mean_ms": statistics.mean(values),
        "stdev_ms": stdev,
        "min_ms": min(values),
        "max_ms": max(values),
    }


def iter_unique_parameters(module: torch.nn.Module):
    seen_parameter_ids: set[int] = set()
    for parameter in module.parameters():
        parameter_id = id(parameter)
        if parameter_id in seen_parameter_ids:
            continue
        seen_parameter_ids.add(parameter_id)
        yield parameter


def make_model(config: BenchmarkConfig, device: torch.device) -> BasicsTransformerLM:
    preset = MODEL_PRESETS[config.model_size]
    model = BasicsTransformerLM(
        vocab_size=config.vocab_size,
        context_length=config.context_length,
        d_model=preset.d_model,
        num_layers=preset.num_layers,
        num_heads=preset.num_heads,
        d_ff=preset.d_ff,
        rope_theta=config.rope_theta,
    )
    return model.to(device)


def broadcast_parameters_from_rank0(model: torch.nn.Module) -> None:
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return
    for parameter in iter_unique_parameters(model):
        dist.broadcast(parameter.data, src=0)


def make_local_batch(config: BenchmarkConfig, device: torch.device, rank: int) -> tuple[torch.Tensor, torch.Tensor]:
    if config.global_batch_size % config.world_size != 0:
        raise ValueError(
            f"global_batch_size={config.global_batch_size} must divide world_size={config.world_size}."
        )
    local_batch_size = config.global_batch_size // config.world_size
    generator = torch.Generator(device="cpu")
    generator.manual_seed(config.seed + rank)
    input_ids = torch.randint(
        low=0,
        high=config.vocab_size,
        size=(local_batch_size, config.context_length),
        generator=generator,
    ).to(device)
    targets = torch.randint(
        low=0,
        high=config.vocab_size,
        size=(local_batch_size, config.context_length),
        generator=generator,
    ).to(device)
    return input_ids, targets


def compute_loss(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    targets: torch.Tensor,
    precision_context,
) -> torch.Tensor:
    with precision_context:
        logits = model(input_ids)
        return F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))


def average_gradients(model: torch.nn.Module) -> None:
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return

    world_size = dist.get_world_size()
    for parameter in iter_unique_parameters(model):
        if not parameter.requires_grad or parameter.grad is None:
            continue
        dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
        parameter.grad /= world_size


def make_optimizer(config: BenchmarkConfig, model: torch.nn.Module) -> torch.optim.Optimizer:
    if config.optimizer_mode == "sharded":
        return ShardedOptimizer(
            model.parameters(),
            AdamW,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

    return AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )


def tensor_tree_num_bytes(value) -> int:
    if torch.is_tensor(value):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(tensor_tree_num_bytes(nested_value) for nested_value in value.values())
    if isinstance(value, (list, tuple)):
        return sum(tensor_tree_num_bytes(nested_value) for nested_value in value)
    return 0


def parameter_bytes(model: torch.nn.Module) -> int:
    return sum(parameter.numel() * parameter.element_size() for parameter in iter_unique_parameters(model))


def gradient_bytes(model: torch.nn.Module) -> int:
    return sum(
        parameter.grad.numel() * parameter.grad.element_size()
        for parameter in iter_unique_parameters(model)
        if parameter.grad is not None
    )


def optimizer_state_bytes(optimizer: torch.optim.Optimizer) -> int:
    if isinstance(optimizer, ShardedOptimizer):
        if optimizer._local_optimizer is None:
            return 0
        return sum(tensor_tree_num_bytes(state) for state in optimizer._local_optimizer.state.values())

    return sum(tensor_tree_num_bytes(state) for state in optimizer.state.values())


def collect_cuda_memory_stats(device: torch.device) -> dict[str, int]:
    stats = torch.cuda.memory_stats(device)
    return {
        "current_allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "current_reserved_bytes": int(torch.cuda.memory_reserved(device)),
        "max_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "max_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "active_peak_bytes": int(stats.get("active_bytes.all.peak", 0)),
        "requested_peak_bytes": int(stats.get("requested_bytes.all.peak", 0)),
    }


def collect_memory_checkpoint(
    *,
    label: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict[str, int | str]:
    if device.type != "cuda":
        raise ValueError("Memory checkpoint collection in this script currently expects CUDA.")

    memory_stats = collect_cuda_memory_stats(device)
    parameter_tensor_bytes = parameter_bytes(model)
    gradient_tensor_bytes = gradient_bytes(model)
    optimizer_state_tensor_bytes = optimizer_state_bytes(optimizer)
    accounted_bytes = parameter_tensor_bytes + gradient_tensor_bytes + optimizer_state_tensor_bytes
    residual_current_allocated_bytes = max(
        0,
        memory_stats["current_allocated_bytes"] - accounted_bytes,
    )

    return {
        "label": label,
        **memory_stats,
        "parameter_tensor_bytes": parameter_tensor_bytes,
        "gradient_tensor_bytes": gradient_tensor_bytes,
        "optimizer_state_tensor_bytes": optimizer_state_tensor_bytes,
        "residual_current_allocated_bytes": residual_current_allocated_bytes,
    }


def maybe_reset_peak_memory_stats(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def run_training_step(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    input_ids: torch.Tensor,
    targets: torch.Tensor,
    config: BenchmarkConfig,
    device: torch.device,
) -> None:
    optimizer.zero_grad(set_to_none=True)
    precision_context = make_precision_context(config, device)
    loss = compute_loss(model, input_ids, targets, precision_context)
    loss.backward()
    average_gradients(model)
    optimizer.step()


def run_memory_benchmark(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    input_ids: torch.Tensor,
    targets: torch.Tensor,
    config: BenchmarkConfig,
    device: torch.device,
) -> dict[str, object]:
    if device.type != "cuda":
        raise ValueError("The Chapter 3 memory benchmark is intended for CUDA devices.")

    maybe_synchronize(device)
    maybe_reset_peak_memory_stats(device)
    after_model_init = collect_memory_checkpoint(
        label="after_model_initialization",
        model=model,
        optimizer=optimizer,
        device=device,
    )

    maybe_reset_peak_memory_stats(device)
    optimizer.zero_grad(set_to_none=True)
    precision_context = make_precision_context(config, device)
    loss = compute_loss(model, input_ids, targets, precision_context)
    loss.backward()
    average_gradients(model)
    maybe_synchronize(device)
    before_optimizer_step = collect_memory_checkpoint(
        label="before_optimizer_step",
        model=model,
        optimizer=optimizer,
        device=device,
    )

    maybe_reset_peak_memory_stats(device)
    optimizer.step()
    maybe_synchronize(device)
    after_optimizer_step = collect_memory_checkpoint(
        label="after_optimizer_step",
        model=model,
        optimizer=optimizer,
        device=device,
    )

    return {
        "checkpoints": [
            after_model_init,
            before_optimizer_step,
            after_optimizer_step,
        ]
    }


def run_timed_step(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    input_ids: torch.Tensor,
    targets: torch.Tensor,
    config: BenchmarkConfig,
    device: torch.device,
) -> dict[str, float]:
    if device.type == "cuda":
        maybe_synchronize(device)
        total_start = torch.cuda.Event(enable_timing=True)
        forward_backward_end = torch.cuda.Event(enable_timing=True)
        optimizer_end = torch.cuda.Event(enable_timing=True)

        total_start.record()
        optimizer.zero_grad(set_to_none=True)
        precision_context = make_precision_context(config, device)
        loss = compute_loss(model, input_ids, targets, precision_context)
        loss.backward()
        average_gradients(model)
        forward_backward_end.record()

        optimizer.step()
        optimizer_end.record()
        maybe_synchronize(device)

        return {
            "forward_backward_ms": total_start.elapsed_time(forward_backward_end),
            "optimizer_step_ms": forward_backward_end.elapsed_time(optimizer_end),
            "total_step_ms": total_start.elapsed_time(optimizer_end),
        }

    total_start_time = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    precision_context = make_precision_context(config, device)
    loss = compute_loss(model, input_ids, targets, precision_context)
    loss.backward()
    average_gradients(model)
    forward_backward_end_time = time.perf_counter()
    optimizer.step()
    optimizer_end_time = time.perf_counter()
    return {
        "forward_backward_ms": (forward_backward_end_time - total_start_time) * 1000.0,
        "optimizer_step_ms": (optimizer_end_time - forward_backward_end_time) * 1000.0,
        "total_step_ms": (optimizer_end_time - total_start_time) * 1000.0,
    }


def run_timer_benchmark(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    input_ids: torch.Tensor,
    targets: torch.Tensor,
    config: BenchmarkConfig,
    device: torch.device,
) -> dict[str, object]:
    for _ in range(config.warmup_steps):
        run_training_step(
            model=model,
            optimizer=optimizer,
            input_ids=input_ids,
            targets=targets,
            config=config,
            device=device,
        )

    local_step_metrics: list[dict[str, float]] = []
    for _ in range(config.measure_steps):
        local_step_metrics.append(
            run_timed_step(
                model=model,
                optimizer=optimizer,
                input_ids=input_ids,
                targets=targets,
                config=config,
                device=device,
            )
        )

    gathered_step_metrics: list[list[dict[str, float]] | None] = [None for _ in range(config.world_size)]
    dist.all_gather_object(gathered_step_metrics, local_step_metrics)
    resolved_rank_metrics = [metrics if metrics is not None else [] for metrics in gathered_step_metrics]

    if dist.get_rank() != 0:
        return {}

    flattened_step_metrics = [step for rank_metrics in resolved_rank_metrics for step in rank_metrics]
    return {
        "aggregate": {
            metric_name: summarize_series([step[metric_name] for step in flattened_step_metrics])
            for metric_name in ("forward_backward_ms", "optimizer_step_ms", "total_step_ms")
        },
        "per_rank_total_step_mean_ms": [
            statistics.mean([step["total_step_ms"] for step in rank_metrics]) if rank_metrics else 0.0
            for rank_metrics in resolved_rank_metrics
        ],
        "per_rank_step_metrics": resolved_rank_metrics,
    }


def maybe_write_output(output_path: str | None, payload: dict[str, object]) -> None:
    path = resolve_output_path(output_path, payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def resolve_output_path(output_path: str | None, payload: dict[str, object]) -> Path:
    if output_path is not None:
        return Path(output_path)

    config = payload["config"]
    log_dir = ".agents/logs/3_1_optimizer_state_sharding"
    filename = (
        f"{config['mode']}_{config['optimizer_mode']}_{config['model_size']}_"
        f"ctx{config['context_length']}_{config['backend']}_w{config['world_size']}_"
        f"gbs{config['global_batch_size']}_{config['precision']}.json"
    )
    return Path(f"{log_dir}/{filename}")


def worker_main(rank: int, config: BenchmarkConfig) -> None:
    os.environ["MASTER_ADDR"] = config.master_addr
    os.environ["MASTER_PORT"] = str(config.master_port)

    device = resolve_device(config, rank)
    dist.init_process_group(config.backend, rank=rank, world_size=config.world_size)

    torch.manual_seed(config.seed)
    model = make_model(config, device)
    broadcast_parameters_from_rank0(model)
    optimizer = make_optimizer(config, model)
    input_ids, targets = make_local_batch(config, device, rank)

    if config.mode == "memory":
        local_result = run_memory_benchmark(
            model=model,
            optimizer=optimizer,
            input_ids=input_ids,
            targets=targets,
            config=config,
            device=device,
        )
        gathered_results: list[dict[str, object] | None] = [None for _ in range(config.world_size)]
        dist.all_gather_object(gathered_results, local_result)
        if rank == 0:
            output = {
                "benchmark": "optimizer_state_sharding_accounting_a",
                "config": asdict(config),
                "per_rank": gathered_results,
            }
            maybe_write_output(config.output_path, output)
            print(json.dumps(output, indent=2))
    else:
        result = run_timer_benchmark(
            model=model,
            optimizer=optimizer,
            input_ids=input_ids,
            targets=targets,
            config=config,
            device=device,
        )
        if rank == 0:
            output = {
                "benchmark": "optimizer_state_sharding_accounting_b",
                "config": asdict(config),
                **result,
            }
            maybe_write_output(config.output_path, output)
            print(json.dumps(output, indent=2))

    dist.barrier()
    dist.destroy_process_group()


def main() -> None:
    config = parse_args()
    mp.spawn(worker_main, args=(config,), nprocs=config.world_size, join=True)


if __name__ == "__main__":
    main()
