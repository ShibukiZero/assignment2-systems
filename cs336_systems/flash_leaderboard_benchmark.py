from __future__ import annotations

import argparse
import json
import sys
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import torch

from cs336_systems.flash_attention import (
    FlashAttention2TritonFunction,
    flash_attention_tile_size_override,
)

try:
    import triton.testing
except ImportError as exc:  # pragma: no cover - benchmark only runs where Triton is installed.
    raise ImportError("flash_leaderboard_benchmark.py requires Triton to be installed.") from exc


Precision = Literal["bf16", "fp32"]


@dataclass(frozen=True)
class FlashLeaderboardBenchmarkConfig:
    batch_size: int
    num_heads: int
    d_head: int
    sequence_length: int
    precision: Precision
    warmup_ms: int
    rep_ms: int
    seed: int
    device: str
    is_causal: bool
    compile_flash: bool
    q_tile_size: int | None
    k_tile_size: int | None
    output_path: str | None


def parse_args() -> FlashLeaderboardBenchmarkConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark the single leaderboard configuration for FlashAttention-2. "
            "This mirrors the handout's timing setup, but defaults to 10x shorter "
            "warmup and measurement windows for faster iteration."
        )
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--d-head", type=int, default=64)
    parser.add_argument("--sequence-length", type=int, default=16_384)
    parser.add_argument(
        "--precision",
        choices=("bf16", "fp32"),
        default="bf16",
        help="The leaderboard uses BF16. FP32 is kept here as a debugging fallback.",
    )
    parser.add_argument(
        "--warmup-ms",
        type=int,
        default=20,
        help="Warmup time in milliseconds passed to triton.testing.do_bench.",
    )
    parser.add_argument(
        "--rep-ms",
        type=int,
        default=200,
        help="Measurement time in milliseconds passed to triton.testing.do_bench.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device",
        choices=("auto", "cuda"),
        default="auto",
        help="Use auto to prefer CUDA when available. Triton benchmarks require CUDA.",
    )
    parser.add_argument(
        "--no-compile-flash",
        action="store_true",
        help="Disable torch.compile around FlashAttention2TritonFunction.apply.",
    )
    parser.add_argument(
        "--q-tile-size",
        type=int,
        default=None,
        help="Optional query tile size override for leaderboard tuning.",
    )
    parser.add_argument(
        "--k-tile-size",
        type=int,
        default=None,
        help="Optional key tile size override for leaderboard tuning.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Optional path to write the benchmark JSON payload.",
    )
    args = parser.parse_args()
    return FlashLeaderboardBenchmarkConfig(
        batch_size=args.batch_size,
        num_heads=args.num_heads,
        d_head=args.d_head,
        sequence_length=args.sequence_length,
        precision=args.precision,
        warmup_ms=args.warmup_ms,
        rep_ms=args.rep_ms,
        seed=args.seed,
        device=args.device,
        is_causal=True,
        compile_flash=not args.no_compile_flash,
        q_tile_size=args.q_tile_size,
        k_tile_size=args.k_tile_size,
        output_path=args.output_path,
    )


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_arg)
    if device.type != "cuda":
        raise ValueError(
            "flash_leaderboard_benchmark.py requires CUDA because it benchmarks Triton kernels."
        )
    if not torch.cuda.is_available():
        raise ValueError("CUDA was requested, but no CUDA device is available.")
    return device


def reset_leaf_grads(*tensors: torch.Tensor) -> None:
    for tensor in tensors:
        tensor.grad = None


def make_inputs(
    config: FlashLeaderboardBenchmarkConfig,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dtype = torch.bfloat16 if config.precision == "bf16" else torch.float32
    effective_batch = config.batch_size * config.num_heads
    shape = (effective_batch, config.sequence_length, config.d_head)
    q = torch.randn(*shape, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(*shape, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(*shape, device=device, dtype=dtype, requires_grad=True)
    return q, k, v


def make_flash_runner(config: FlashLeaderboardBenchmarkConfig):
    flash_impl = FlashAttention2TritonFunction.apply
    if config.compile_flash:
        flash_impl = torch.compile(flash_impl)

    def flash_forward_backward(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None:
        override_context = (
            flash_attention_tile_size_override(
                q_tile_size=config.q_tile_size,
                k_tile_size=config.k_tile_size,
            )
            if config.q_tile_size is not None or config.k_tile_size is not None
            else nullcontext()
        )
        with override_context:
            output = flash_impl(q, k, v, config.is_causal)
        loss = output.sum()
        loss.backward()

    return flash_forward_backward


def benchmark_leaderboard_case(
    config: FlashLeaderboardBenchmarkConfig,
    device: torch.device,
) -> float:
    q, k, v = make_inputs(config, device)
    flash_forward_backward = make_flash_runner(config)

    def closure() -> None:
        reset_leaf_grads(q, k, v)
        flash_forward_backward(q, k, v)

    return float(
        triton.testing.do_bench(
            closure,
            warmup=config.warmup_ms,
            rep=config.rep_ms,
        )
    )


def build_payload(
    config: FlashLeaderboardBenchmarkConfig,
    device: torch.device,
    latency_ms: float,
) -> dict[str, object]:
    effective_batch = config.batch_size * config.num_heads
    return {
        "benchmark": "flash_leaderboard",
        "config": asdict(config),
        "derived": {
            "device": str(device),
            "effective_batch_for_flash_attention": effective_batch,
            "qkv_shape": [effective_batch, config.sequence_length, config.d_head],
            "d_model": config.num_heads * config.d_head,
            "note": (
                "The current FlashAttention interface accepts (batch, seq, d_head), "
                "so the leaderboard benchmark flattens batch_size * num_heads into "
                "the batch axis."
            ),
        },
        "results": {
            "forward_backward_ms": latency_ms,
        },
    }


def write_payload(payload: dict[str, object], output_path: str | None) -> None:
    serialized = json.dumps(payload, indent=2)
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)


def format_config_label(config: FlashLeaderboardBenchmarkConfig) -> str:
    return (
        f"batch={config.batch_size}, heads={config.num_heads}, seq={config.sequence_length}, "
        f"d_head={config.d_head}, precision={config.precision}, causal={config.is_causal}, "
        f"compile={config.compile_flash}, q_tile={config.q_tile_size}, "
        f"k_tile={config.k_tile_size}"
    )


def main() -> None:
    config = parse_args()
    torch.random.manual_seed(config.seed)
    device = resolve_device(config.device)

    print(
        (
            f"[flash_leaderboard] Running leaderboard benchmark "
            f"({format_config_label(config)}, warmup={config.warmup_ms} ms, "
            f"rep={config.rep_ms} ms)"
        ),
        file=sys.stderr,
    )
    latency_ms = benchmark_leaderboard_case(config, device)
    print(
        f"[flash_leaderboard] Completed benchmark: forward+backward={latency_ms:.3f} ms",
        file=sys.stderr,
    )

    payload = build_payload(config, device, latency_ms)
    write_payload(payload, config.output_path)


if __name__ == "__main__":
    main()
