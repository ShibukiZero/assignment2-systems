from __future__ import annotations

import argparse
import json
import statistics
import timeit
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import torch

from cs336_basics.model import scaled_dot_product_attention


Implementation = Literal["eager", "compiled"]
Precision = Literal["fp32", "bf16"]


@dataclass(frozen=True)
class AttentionBenchmarkConfig:
    batch_size: int
    sequence_length: int
    embedding_dim: int
    implementation: Implementation
    precision: Precision
    warmup_steps: int
    measure_steps: int
    seed: int
    device: str
    is_causal: bool
    output_path: str | None


def parse_args() -> AttentionBenchmarkConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark standalone scaled-dot-product attention on random Q/K/V tensors. "
            "This is intended for the Assignment 2 attention-only sweeps rather than "
            "the full Transformer model benchmark."
        )
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument(
        "--embedding-dim",
        "--d-model",
        dest="embedding_dim",
        type=int,
        default=64,
        help="Embedding dimension for the standalone Q/K/V tensors.",
    )
    parser.add_argument(
        "--implementation",
        choices=("eager", "compiled"),
        default="eager",
        help="Use eager PyTorch attention or torch.compile on the same function.",
    )
    parser.add_argument(
        "--precision",
        choices=("fp32", "bf16"),
        default="fp32",
        help="Use BF16 autocast on CUDA, or run attention in FP32.",
    )
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument(
        "--measure-steps",
        type=int,
        default=100,
        help="Number of timed forward/backward iterations.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Use auto to prefer CUDA when available.",
    )
    parser.add_argument(
        "--causal",
        action="store_true",
        help="Apply a standard lower-triangular causal mask.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Optional path to write the benchmark JSON result.",
    )
    args = parser.parse_args()
    return AttentionBenchmarkConfig(
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        embedding_dim=args.embedding_dim,
        implementation=args.implementation,
        precision=args.precision,
        warmup_steps=args.warmup_steps,
        measure_steps=args.measure_steps,
        seed=args.seed,
        device=args.device,
        is_causal=args.causal,
        output_path=args.output_path,
    )


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested, but no CUDA device is available.")
    return torch.device(device_arg)


def maybe_synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def make_precision_context(config: AttentionBenchmarkConfig, device: torch.device):
    if config.precision == "fp32":
        return nullcontext()
    if device.type != "cuda":
        raise ValueError("BF16 autocast is only supported on CUDA in this script.")
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)


def make_attention_inputs(
    config: AttentionBenchmarkConfig,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    shape = (config.batch_size, config.sequence_length, config.embedding_dim)
    q = torch.randn(*shape, device=device, requires_grad=True)
    k = torch.randn(*shape, device=device, requires_grad=True)
    v = torch.randn(*shape, device=device, requires_grad=True)
    return q, k, v


def make_attention_mask(
    config: AttentionBenchmarkConfig,
    device: torch.device,
) -> torch.Tensor | None:
    if not config.is_causal:
        return None

    query_positions = torch.arange(config.sequence_length, device=device)
    key_positions = torch.arange(config.sequence_length, device=device)
    return query_positions[:, None] >= key_positions[None, :]


def make_attention_impl(config: AttentionBenchmarkConfig):
    def attention_impl(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        return scaled_dot_product_attention(Q=q, K=k, V=v, mask=mask)

    if config.implementation == "compiled":
        return torch.compile(attention_impl)
    return attention_impl


def summarize_series(series: list[float | int]) -> dict[str, float]:
    numeric_series = [float(value) for value in series]
    stdev = statistics.stdev(numeric_series) if len(numeric_series) > 1 else 0.0
    return {
        "mean": statistics.mean(numeric_series),
        "stdev": stdev,
        "min": min(numeric_series),
        "max": max(numeric_series),
    }


def summarize_timings(step_metrics: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    if not step_metrics:
        return {}

    metric_names = ("forward_seconds", "backward_seconds", "total_seconds")
    return {
        metric_name: summarize_series([step[metric_name] for step in step_metrics])
        for metric_name in metric_names
    }


def summarize_memory(step_metrics: list[dict[str, float]], device: torch.device) -> dict[str, dict[str, float]]:
    if device.type != "cuda" or not step_metrics:
        return {}

    metric_names = (
        "allocated_before_forward_bytes",
        "allocated_before_backward_bytes",
        "reserved_before_forward_bytes",
        "reserved_before_backward_bytes",
        "saved_for_backward_bytes",
        "peak_allocated_bytes",
        "peak_reserved_bytes",
    )
    return {
        metric_name: summarize_series([step[metric_name] for step in step_metrics])
        for metric_name in metric_names
    }


def run_step(
    config: AttentionBenchmarkConfig,
    attention_impl,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None,
    device: torch.device,
):
    q.grad = None
    k.grad = None
    v.grad = None

    precision_context = make_precision_context(config, device)
    step_metrics: dict[str, float] = {}

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        step_metrics["allocated_before_forward_bytes"] = float(torch.cuda.memory_allocated(device))
        step_metrics["reserved_before_forward_bytes"] = float(torch.cuda.memory_reserved(device))

    start = timeit.default_timer()
    with precision_context:
        output = attention_impl(q, k, v, mask)
    maybe_synchronize(device)
    forward_end = timeit.default_timer()

    if device.type == "cuda":
        allocated_before_backward = float(torch.cuda.memory_allocated(device))
        reserved_before_backward = float(torch.cuda.memory_reserved(device))
        step_metrics["allocated_before_backward_bytes"] = allocated_before_backward
        step_metrics["reserved_before_backward_bytes"] = reserved_before_backward
        step_metrics["saved_for_backward_bytes"] = (
            allocated_before_backward - step_metrics["allocated_before_forward_bytes"]
        )

    loss = output.sum()
    loss.backward()
    maybe_synchronize(device)
    backward_end = timeit.default_timer()

    step_metrics["forward_seconds"] = forward_end - start
    step_metrics["backward_seconds"] = backward_end - forward_end
    step_metrics["total_seconds"] = backward_end - start

    if device.type == "cuda":
        step_metrics["peak_allocated_bytes"] = float(torch.cuda.max_memory_allocated(device))
        step_metrics["peak_reserved_bytes"] = float(torch.cuda.max_memory_reserved(device))

    del output
    del loss
    return step_metrics


def main() -> None:
    config = parse_args()
    device = resolve_device(config.device)

    torch.manual_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)

    q, k, v = make_attention_inputs(config, device)
    mask = make_attention_mask(config, device)
    attention_impl = make_attention_impl(config)

    for _ in range(config.warmup_steps):
        run_step(config, attention_impl, q, k, v, mask, device)

    step_metrics: list[dict[str, float]] = []
    for _ in range(config.measure_steps):
        step_metrics.append(run_step(config, attention_impl, q, k, v, mask, device))

    result = {
        "config": asdict(config),
        "device": str(device),
        "step_metrics": step_metrics,
        "timings": summarize_timings(step_metrics),
        "memory_bytes": summarize_memory(step_metrics, device),
    }
    result_json = json.dumps(result, indent=2)

    if config.output_path is not None:
        output_path = Path(config.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(result_json + "\n", encoding="utf-8")

    print(result_json)


if __name__ == "__main__":
    main()
