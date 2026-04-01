from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.optimizer import AdamW
from cs336_systems.ddp import NaiveDDP


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
    model_size: str
    context_length: int
    global_batch_size: int
    vocab_size: int
    rope_theta: float
    precision: str
    backend: str
    device: str
    world_size: int
    nvtx: bool
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
            "Benchmark naive DDP training for Assignment 2 Chapter 2. "
            "This script reports per-step runtime and communication share, "
            "and can optionally add NVTX ranges for Nsight Systems."
        )
    )
    parser.add_argument("--model-size", choices=tuple(MODEL_PRESETS), default="xl")
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument(
        "--global-batch-size",
        type=int,
        default=8,
        help="Global batch size across all ranks. Must divide world_size.",
    )
    parser.add_argument("--vocab-size", type=int, default=10_000)
    parser.add_argument("--rope-theta", type=float, default=10_000.0)
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--backend", choices=("gloo", "nccl"), default="nccl")
    parser.add_argument(
        "--world-size",
        type=int,
        default=2,
        help="Number of worker processes / ranks to launch. The handout benchmark uses 2 GPUs.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Auto picks CUDA for NCCL and CPU for Gloo.",
    )
    parser.add_argument(
        "--nvtx",
        action="store_true",
        help="Annotate warmup, measured steps, and training sub-phases with NVTX ranges for Nsight Systems.",
    )
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--measure-steps", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--master-addr", type=str, default="127.0.0.1")
    parser.add_argument("--master-port", type=int, default=29500)
    parser.add_argument("--output-path", type=str, default=None)
    args = parser.parse_args()
    return BenchmarkConfig(
        model_size=args.model_size,
        context_length=args.context_length,
        global_batch_size=args.global_batch_size,
        vocab_size=args.vocab_size,
        rope_theta=args.rope_theta,
        precision=args.precision,
        backend=args.backend,
        device=args.device,
        world_size=args.world_size,
        nvtx=args.nvtx,
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
    if config.backend == "nccl":
        if not torch.cuda.is_available():
            raise ValueError("NCCL benchmarking requires CUDA.")
        if torch.cuda.device_count() <= rank:
            raise ValueError(
                f"rank {rank} requested CUDA device, but only {torch.cuda.device_count()} visible devices exist."
            )
        torch.cuda.set_device(rank)
        return torch.device(f"cuda:{rank}")

    if config.device == "cuda":
        raise ValueError("Gloo benchmarking in this script is intended for CPU tensors.")
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


@contextmanager
def maybe_nvtx_range(label: str, enabled: bool, device: torch.device):
    if not enabled or device.type != "cuda":
        yield
        return

    torch.cuda.nvtx.range_push(label)
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()


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


def summarize_step_metrics(step_metrics: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    metric_names = ("forward_backward_ms", "communication_ms", "optimizer_step_ms", "total_step_ms")
    return {
        metric_name: summarize_series([step[metric_name] for step in step_metrics])
        for metric_name in metric_names
    }


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


def make_local_batch(config: BenchmarkConfig, device: torch.device, world_size: int, rank: int) -> tuple[torch.Tensor, torch.Tensor]:
    if config.global_batch_size % world_size != 0:
        raise ValueError(
            f"global_batch_size={config.global_batch_size} must divide world_size={world_size} for data parallelism."
        )
    local_batch_size = config.global_batch_size // world_size

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


def run_timed_step(
    ddp_model: NaiveDDP,
    optimizer: AdamW,
    input_ids: torch.Tensor,
    targets: torch.Tensor,
    config: BenchmarkConfig,
    device: torch.device,
) -> dict[str, float]:
    optimizer.zero_grad(set_to_none=True)
    precision_context = make_precision_context(config, device)

    if device.type == "cuda":
        maybe_synchronize(device)
        total_start = torch.cuda.Event(enable_timing=True)
        forward_backward_end = torch.cuda.Event(enable_timing=True)
        communication_start = torch.cuda.Event(enable_timing=True)
        communication_end = torch.cuda.Event(enable_timing=True)
        optimizer_end = torch.cuda.Event(enable_timing=True)

        total_start.record()
        with maybe_nvtx_range("forward_and_loss", config.nvtx, device):
            loss = compute_loss(ddp_model, input_ids, targets, precision_context)
        with maybe_nvtx_range("backward", config.nvtx, device):
            loss.backward()
        forward_backward_end.record()

        communication_start.record()
        with maybe_nvtx_range("gradient_all_reduce", config.nvtx, device):
            ddp_model.finish_gradient_synchronization()
        communication_end.record()

        with maybe_nvtx_range("optimizer_step", config.nvtx, device):
            optimizer.step()
        optimizer_end.record()

        maybe_synchronize(device)
        forward_backward_ms = total_start.elapsed_time(forward_backward_end)
        communication_ms = communication_start.elapsed_time(communication_end)
        total_step_ms = total_start.elapsed_time(optimizer_end)
        optimizer_step_ms = communication_end.elapsed_time(optimizer_end)
    else:
        total_start_time = time.perf_counter()
        with maybe_nvtx_range("forward_and_loss", config.nvtx, device):
            loss = compute_loss(ddp_model, input_ids, targets, precision_context)
        with maybe_nvtx_range("backward", config.nvtx, device):
            loss.backward()
        forward_backward_end_time = time.perf_counter()

        communication_start_time = time.perf_counter()
        with maybe_nvtx_range("gradient_all_reduce", config.nvtx, device):
            ddp_model.finish_gradient_synchronization()
        communication_end_time = time.perf_counter()

        with maybe_nvtx_range("optimizer_step", config.nvtx, device):
            optimizer.step()
        optimizer_end_time = time.perf_counter()

        forward_backward_ms = (forward_backward_end_time - total_start_time) * 1000.0
        communication_ms = (communication_end_time - communication_start_time) * 1000.0
        optimizer_step_ms = (optimizer_end_time - communication_end_time) * 1000.0
        total_step_ms = (optimizer_end_time - total_start_time) * 1000.0

    communication_fraction = communication_ms / total_step_ms if total_step_ms > 0.0 else 0.0
    return {
        "forward_backward_ms": forward_backward_ms,
        "communication_ms": communication_ms,
        "optimizer_step_ms": optimizer_step_ms,
        "total_step_ms": total_step_ms,
        "communication_fraction": communication_fraction,
    }

def gather_step_metrics(local_step_metrics: list[dict[str, float]], world_size: int) -> list[list[dict[str, float]]]:
    gathered_metrics: list[list[dict[str, float]] | None] = [None for _ in range(world_size)]
    dist.all_gather_object(gathered_metrics, local_step_metrics)
    return [metrics if metrics is not None else [] for metrics in gathered_metrics]


def run_timer_benchmark(
    ddp_model: NaiveDDP,
    optimizer: AdamW,
    input_ids: torch.Tensor,
    targets: torch.Tensor,
    config: BenchmarkConfig,
    device: torch.device,
) -> dict[str, object]:
    for step_idx in range(config.warmup_steps):
        with maybe_nvtx_range(f"warmup_{step_idx}", config.nvtx, device):
            _ = run_timed_step(ddp_model, optimizer, input_ids, targets, config, device)

    local_step_metrics = []
    for step_idx in range(config.measure_steps):
        with maybe_nvtx_range(f"measure_{step_idx}", config.nvtx, device):
            local_step_metrics.append(run_timed_step(ddp_model, optimizer, input_ids, targets, config, device))
    gathered_step_metrics = gather_step_metrics(local_step_metrics, dist.get_world_size())

    if dist.get_rank() != 0:
        return {}

    per_rank_summary = [summarize_step_metrics(rank_metrics) for rank_metrics in gathered_step_metrics]
    flattened_step_metrics = [step for rank_metrics in gathered_step_metrics for step in rank_metrics]
    aggregate_summary = summarize_step_metrics(flattened_step_metrics)
    aggregate_summary["communication_fraction"] = summarize_series(
        [step["communication_fraction"] * 100.0 for step in flattened_step_metrics]
    )

    return {
        "aggregate": aggregate_summary,
        "per_rank_total_step_mean_ms": [
            rank_summary["total_step_ms"]["mean_ms"] for rank_summary in per_rank_summary
        ],
        "per_rank_communication_mean_ms": [
            rank_summary["communication_ms"]["mean_ms"] for rank_summary in per_rank_summary
        ],
        "per_rank_communication_fraction_percent_mean": [
            statistics.mean([step["communication_fraction"] * 100.0 for step in rank_metrics])
            for rank_metrics in gathered_step_metrics
        ],
    }


def setup_process_group(config: BenchmarkConfig, rank: int, world_size: int) -> torch.device:
    os.environ["MASTER_ADDR"] = config.master_addr
    os.environ["MASTER_PORT"] = str(config.master_port)
    device = resolve_device(config, rank)
    dist.init_process_group(config.backend, rank=rank, world_size=world_size)
    return device


def cleanup_process_group() -> None:
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def validate_world_size_for_config(config: BenchmarkConfig, world_size: int) -> None:
    if config.backend == "nccl" and torch.cuda.device_count() < world_size:
        raise ValueError(
            f"Requested world_size={world_size} with NCCL, but only {torch.cuda.device_count()} CUDA devices are visible."
        )


def worker_main(rank: int, world_size: int, config: BenchmarkConfig, result_queue: mp.SimpleQueue) -> None:
    validate_world_size_for_config(config, world_size)
    device = setup_process_group(config, rank, world_size)
    torch.manual_seed(config.seed + rank)

    try:
        model = make_model(config, device)
        ddp_model = NaiveDDP(model)
        optimizer = AdamW(
            ddp_model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        input_ids, targets = make_local_batch(config, device, world_size, rank)
        result = run_timer_benchmark(ddp_model, optimizer, input_ids, targets, config, device)

        if rank == 0:
            result_queue.put(result)
    finally:
        cleanup_process_group()


def maybe_write_output(output_path: str | None, payload: dict[str, object]) -> None:
    path = resolve_output_path(output_path, payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def resolve_output_path(output_path: str | None, payload: dict[str, object]) -> Path:
    if output_path is not None:
        return Path(output_path)

    config = payload["config"]
    return Path(
        ".agents/logs/2_2_naive_ddp/"
        f"timer_{config['model_size']}_ctx{config['context_length']}_"
        f"{config['backend']}_w{config['world_size']}_gbs{config['global_batch_size']}_"
        f"{config['precision']}.json"
    )


def main() -> None:
    config = parse_args()

    spawn_context = mp.get_context("spawn")
    result_queue = spawn_context.SimpleQueue()
    mp.spawn(
        worker_main,
        args=(config.world_size, config, result_queue),
        nprocs=config.world_size,
        join=True,
    )

    result = result_queue.get()
    payload = {
        "benchmark": "naive_ddp_benchmarking",
        "config": asdict(config),
        "world_size": config.world_size,
        "result": result,
    }
    maybe_write_output(config.output_path, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
