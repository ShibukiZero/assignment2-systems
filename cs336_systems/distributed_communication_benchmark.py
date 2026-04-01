from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


BYTES_PER_MB = 1024 * 1024
FLOAT32_BYTES = 4


@dataclass(frozen=True)
class BenchmarkConfig:
    backends: tuple[str, ...]
    sizes_mb: tuple[int, ...]
    process_counts: tuple[int, ...]
    warmup_steps: int
    measure_steps: int
    master_addr: str
    master_port_base: int
    output_path: str | None


def parse_args() -> BenchmarkConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark single-node all-reduce communication for Assignment 2 Chapter 2. "
            "This script follows the handout setup: Gloo on CPU, NCCL on GPU, "
            "multiple warmup iterations, and timing aggregation across ranks."
        )
    )
    parser.add_argument(
        "--backends",
        nargs="+",
        choices=("gloo", "nccl"),
        default=("gloo", "nccl"),
        help="Communication backends to benchmark.",
    )
    parser.add_argument(
        "--sizes-mb",
        nargs="+",
        type=int,
        default=(1, 10, 100, 1024),
        help="All-reduce tensor sizes in MiB. The assignment asks for 1, 10, 100, and 1GB.",
    )
    parser.add_argument(
        "--process-counts",
        nargs="+",
        type=int,
        default=(2, 4, 6),
        help="Number of worker processes to launch for each run.",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=5,
        help="Warmup iterations before timing. The handout suggests 5 as a good default.",
    )
    parser.add_argument(
        "--measure-steps",
        type=int,
        default=20,
        help="Measured iterations per configuration.",
    )
    parser.add_argument(
        "--master-addr",
        type=str,
        default="127.0.0.1",
        help="Address for the single-node process group master.",
    )
    parser.add_argument(
        "--master-port-base",
        type=int,
        default=29500,
        help="Base TCP port. Each spawned run increments this to avoid collisions.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Optional path to write the benchmark results as JSON.",
    )
    args = parser.parse_args()
    return BenchmarkConfig(
        backends=tuple(args.backends),
        sizes_mb=tuple(args.sizes_mb),
        process_counts=tuple(args.process_counts),
        warmup_steps=args.warmup_steps,
        measure_steps=args.measure_steps,
        master_addr=args.master_addr,
        master_port_base=args.master_port_base,
        output_path=args.output_path,
    )


def maybe_synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def summarize_samples(samples: list[float]) -> dict[str, float]:
    if not samples:
        return {"mean_ms": 0.0, "stdev_ms": 0.0, "min_ms": 0.0, "max_ms": 0.0}

    stdev = statistics.stdev(samples) if len(samples) > 1 else 0.0
    return {
        "mean_ms": statistics.mean(samples),
        "stdev_ms": stdev,
        "min_ms": min(samples),
        "max_ms": max(samples),
    }


def setup_process_group(rank: int, world_size: int, backend: str, master_addr: str, master_port: int) -> torch.device:
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)

    if backend == "nccl":
        if not torch.cuda.is_available():
            raise RuntimeError("NCCL was requested, but CUDA is not available.")
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")
    else:
        device = torch.device("cpu")

    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    return device


def cleanup_process_group() -> None:
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def benchmark_all_reduce_for_size(
    *,
    size_mb: int,
    warmup_steps: int,
    measure_steps: int,
    world_size: int,
    device: torch.device,
    rank: int,
) -> dict[str, object]:
    num_bytes = size_mb * BYTES_PER_MB
    numel = num_bytes // FLOAT32_BYTES
    if numel <= 0:
        raise ValueError(f"Tensor size must be positive, got {size_mb} MB.")

    # A zero tensor keeps repeated all-reduce calls numerically stable without
    # introducing extra refill work inside the timed loop.
    tensor = torch.zeros(numel, dtype=torch.float32, device=device)

    for _ in range(warmup_steps):
        dist.all_reduce(tensor, async_op=False)
    maybe_synchronize(device)

    local_timings_ms: list[float] = []
    for _ in range(measure_steps):
        # Synchronize before and after each measured iteration so the timing
        # reflects the actual communication cost rather than queued GPU work.
        maybe_synchronize(device)
        start_time = time.perf_counter()
        dist.all_reduce(tensor, async_op=False)
        maybe_synchronize(device)
        end_time = time.perf_counter()
        local_timings_ms.append((end_time - start_time) * 1000.0)

    local_timings_tensor = torch.tensor(local_timings_ms, dtype=torch.float64, device=device)
    gathered_timings = [torch.empty_like(local_timings_tensor) for _ in range(world_size)]
    dist.all_gather(gathered_timings, local_timings_tensor)

    if rank != 0:
        return {}

    per_rank_timings_ms = [timings.cpu().tolist() for timings in gathered_timings]
    per_rank_summary = [summarize_samples(rank_timings) for rank_timings in per_rank_timings_ms]
    flattened_timings_ms = [timing for rank_timings in per_rank_timings_ms for timing in rank_timings]
    aggregate_summary = summarize_samples(flattened_timings_ms)

    return {
        "size_mb": size_mb,
        "size_bytes": num_bytes,
        "numel_float32": numel,
        "measure_steps": measure_steps,
        "aggregate": aggregate_summary,
        "per_rank_mean_ms": [rank_summary["mean_ms"] for rank_summary in per_rank_summary],
        "slowest_rank_mean_ms": max(rank_summary["mean_ms"] for rank_summary in per_rank_summary),
        "fastest_rank_mean_ms": min(rank_summary["mean_ms"] for rank_summary in per_rank_summary),
    }


def worker_main(
    rank: int,
    backend: str,
    world_size: int,
    config: BenchmarkConfig,
    master_port: int,
    result_queue: mp.SimpleQueue,
) -> None:
    device = setup_process_group(rank, world_size, backend, config.master_addr, master_port)

    try:
        size_results: list[dict[str, object]] = []
        for size_mb in config.sizes_mb:
            result = benchmark_all_reduce_for_size(
                size_mb=size_mb,
                warmup_steps=config.warmup_steps,
                measure_steps=config.measure_steps,
                world_size=world_size,
                device=device,
                rank=rank,
            )
            if rank == 0:
                size_results.append(result)

        if rank == 0:
            result_queue.put(
                {
                    "backend": backend,
                    "device_type": device.type,
                    "world_size": world_size,
                    "warmup_steps": config.warmup_steps,
                    "measure_steps": config.measure_steps,
                    "sizes": size_results,
                }
            )
    finally:
        cleanup_process_group()


def validate_run_request(backend: str, world_size: int) -> str | None:
    if world_size <= 0:
        return "world_size must be positive"

    if backend == "nccl":
        if not torch.cuda.is_available():
            return "CUDA is not available for NCCL benchmarking"
        if torch.cuda.device_count() < world_size:
            return (
                f"requested {world_size} NCCL workers but only "
                f"{torch.cuda.device_count()} CUDA devices are visible"
            )

    return None


def run_single_configuration(
    *,
    backend: str,
    world_size: int,
    config: BenchmarkConfig,
    master_port: int,
) -> dict[str, object]:
    skip_reason = validate_run_request(backend, world_size)
    if skip_reason is not None:
        return {
            "backend": backend,
            "world_size": world_size,
            "status": "skipped",
            "reason": skip_reason,
        }

    spawn_context = mp.get_context("spawn")
    result_queue = spawn_context.SimpleQueue()
    mp.spawn(
        worker_main,
        args=(backend, world_size, config, master_port, result_queue),
        nprocs=world_size,
        join=True,
    )
    result = result_queue.get()
    result["status"] = "ok"
    return result


def maybe_write_output(output_path: str | None, payload: dict[str, object]) -> None:
    if output_path is None:
        return

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    config = parse_args()

    all_results: list[dict[str, object]] = []
    run_index = 0
    for backend in config.backends:
        for world_size in config.process_counts:
            master_port = config.master_port_base + run_index
            result = run_single_configuration(
                backend=backend,
                world_size=world_size,
                config=config,
                master_port=master_port,
            )
            all_results.append(result)
            run_index += 1

            status = result["status"]
            if status == "ok":
                print(
                    f"[ok] backend={backend} world_size={world_size} "
                    f"sizes={len(result['sizes'])} measured",
                    flush=True,
                )
            else:
                print(
                    f"[skipped] backend={backend} world_size={world_size}: {result['reason']}",
                    flush=True,
                )

    payload = {
        "benchmark": "distributed_communication_single_node",
        "config": asdict(config),
        "results": all_results,
    }
    maybe_write_output(config.output_path, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
