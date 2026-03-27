from __future__ import annotations

import argparse
import gc
import itertools
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import torch

from cs336_basics.model import scaled_dot_product_attention
from cs336_systems.flash_attention import (
    FlashAttention2TritonFunction,
    flash_attention_tile_size_override,
)

try:
    import triton.testing
except ImportError as exc:  # pragma: no cover - benchmark only runs where Triton is installed.
    raise ImportError("flash_benchmark.py requires Triton to be installed.") from exc


Precision = Literal["fp32", "bf16"]


DEFAULT_SEQUENCE_LENGTHS = [2 ** exponent for exponent in range(7, 17)]
DEFAULT_EMBEDDING_DIMS = [16, 32, 64, 128]
DEFAULT_PRECISIONS: list[Precision] = ["fp32", "bf16"]
DEFAULT_TILE_SIZES = [16]


@dataclass(frozen=True)
class FlashBenchmarkConfig:
    batch_size: int
    sequence_lengths: list[int]
    embedding_dims: list[int]
    precisions: list[Precision]
    q_tile_sizes: list[int]
    k_tile_sizes: list[int]
    warmup_ms: int
    rep_ms: int
    seed: int
    device: str
    is_causal: bool
    run_benchmarks: bool
    output_path: str | None


@dataclass(frozen=True)
class FlashBenchmarkCase:
    batch_size: int
    sequence_length: int
    embedding_dim: int
    precision: Precision
    q_tile_size: int
    k_tile_size: int
    is_causal: bool


class BenchmarkStageError(RuntimeError):
    def __init__(self, stage: str, cause: BaseException):
        super().__init__(str(cause))
        self.stage = stage
        self.cause = cause


def parse_args() -> FlashBenchmarkConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark FlashAttention against regular PyTorch attention using "
            "triton.testing.do_bench, while sweeping the cartesian product of "
            "sequence length, embedding dimension, precision, and tile sizes."
        )
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--sequence-lengths",
        type=int,
        nargs="+",
        default=DEFAULT_SEQUENCE_LENGTHS,
        help="Sequence lengths to sweep. Handout default is powers of 2 from 128 to 65536.",
    )
    parser.add_argument(
        "--embedding-dims",
        type=int,
        nargs="+",
        default=DEFAULT_EMBEDDING_DIMS,
        help="Embedding dimensions to sweep. Handout default is 16, 32, 64, and 128.",
    )
    parser.add_argument(
        "--precisions",
        nargs="+",
        choices=("fp32", "bf16"),
        default=DEFAULT_PRECISIONS,
        help="Precisions to sweep.",
    )
    parser.add_argument(
        "--q-tile-sizes",
        type=int,
        nargs="+",
        default=DEFAULT_TILE_SIZES,
        help="One or more query tile sizes to include in the sweep grid.",
    )
    parser.add_argument(
        "--k-tile-sizes",
        type=int,
        nargs="+",
        default=DEFAULT_TILE_SIZES,
        help="One or more key tile sizes to include in the sweep grid.",
    )
    parser.add_argument(
        "--warmup-ms",
        type=int,
        default=25,
        help="Warmup time in milliseconds passed to triton.testing.do_bench.",
    )
    parser.add_argument(
        "--rep-ms",
        type=int,
        default=100,
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
        "--grid-only",
        action="store_true",
        help="Only emit the cartesian-product sweep grid JSON without running benchmarks.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Optional path to write the JSON payload.",
    )
    args = parser.parse_args()
    return FlashBenchmarkConfig(
        batch_size=args.batch_size,
        sequence_lengths=args.sequence_lengths,
        embedding_dims=args.embedding_dims,
        precisions=list(args.precisions),
        q_tile_sizes=args.q_tile_sizes,
        k_tile_sizes=args.k_tile_sizes,
        warmup_ms=args.warmup_ms,
        rep_ms=args.rep_ms,
        seed=args.seed,
        device=args.device,
        is_causal=True,
        run_benchmarks=not args.grid_only,
        output_path=args.output_path,
    )


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_arg)
    if device.type != "cuda":
        raise ValueError("flash_benchmark.py requires CUDA because it benchmarks Triton kernels.")
    if not torch.cuda.is_available():
        raise ValueError("CUDA was requested, but no CUDA device is available.")
    return device


def _validate_sweep_values(name: str, values: list[int]) -> list[int]:
    unique_values = sorted(set(values))
    if not unique_values:
        raise ValueError(f"Expected at least one value for {name}.")
    return unique_values


def build_sweep_grid(config: FlashBenchmarkConfig) -> list[FlashBenchmarkCase]:
    sequence_lengths = _validate_sweep_values("sequence_lengths", config.sequence_lengths)
    embedding_dims = _validate_sweep_values("embedding_dims", config.embedding_dims)
    q_tile_sizes = _validate_sweep_values("q_tile_sizes", config.q_tile_sizes)
    k_tile_sizes = _validate_sweep_values("k_tile_sizes", config.k_tile_sizes)

    sweep_grid: list[FlashBenchmarkCase] = []
    for sequence_length, embedding_dim, precision, q_tile_size, k_tile_size in itertools.product(
        sequence_lengths,
        embedding_dims,
        config.precisions,
        q_tile_sizes,
        k_tile_sizes,
    ):
        sweep_grid.append(
            FlashBenchmarkCase(
                batch_size=config.batch_size,
                sequence_length=sequence_length,
                embedding_dim=embedding_dim,
                precision=precision,
                q_tile_size=q_tile_size,
                k_tile_size=k_tile_size,
                is_causal=config.is_causal,
            )
        )
    return sweep_grid


def make_inputs(
    case: FlashBenchmarkCase,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    dtype = torch.float32 if case.precision == "fp32" else torch.bfloat16
    shape = (case.batch_size, case.sequence_length, case.embedding_dim)
    q = torch.randn(*shape, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(*shape, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(*shape, device=device, dtype=dtype, requires_grad=True)
    grad_o = torch.randn(*shape, device=device, dtype=dtype)
    return q, k, v, grad_o


def make_causal_mask(case: FlashBenchmarkCase, device: torch.device) -> torch.Tensor:
    query_positions = torch.arange(case.sequence_length, device=device)
    key_positions = torch.arange(case.sequence_length, device=device)
    return query_positions[:, None] >= key_positions[None, :]


def reset_leaf_grads(*tensors: torch.Tensor) -> None:
    for tensor in tensors:
        tensor.grad = None


def _is_oom_error(error: BaseException) -> bool:
    if isinstance(error, torch.cuda.OutOfMemoryError):
        return True
    if isinstance(error, RuntimeError):
        return "out of memory" in str(error).lower()
    return False


def _cleanup_after_failure(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)


def _empty_metrics() -> dict[str, float | None]:
    return {
        "forward_ms": None,
        "backward_ms": None,
        "end_to_end_ms": None,
    }


def _raise_with_stage(stage: str, error: BaseException) -> None:
    raise BenchmarkStageError(stage, error) from error


def benchmark_pytorch_attention_case(
    case: FlashBenchmarkCase,
    device: torch.device,
    *,
    warmup_ms: int,
    rep_ms: int,
) -> dict[str, float]:
    try:
        q_forward, k_forward, v_forward, _ = make_inputs(case, device)
        q_backward, k_backward, v_backward, grad_backward = make_inputs(case, device)
        q_total, k_total, v_total, grad_total = make_inputs(case, device)
        mask = make_causal_mask(case, device)
    except Exception as error:  # pragma: no cover - exercised in remote benchmark runs.
        _raise_with_stage("input_setup", error)

    def regular_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return scaled_dot_product_attention(Q=q, K=k, V=v, mask=mask)

    def forward_closure() -> None:
        regular_attention(q_forward, k_forward, v_forward)

    try:
        output_backward = regular_attention(q_backward, k_backward, v_backward)
    except Exception as error:  # pragma: no cover - exercised in remote benchmark runs.
        _raise_with_stage("backward_graph_setup", error)

    def backward_closure() -> None:
        reset_leaf_grads(q_backward, k_backward, v_backward)
        torch.autograd.grad(
            outputs=output_backward,
            inputs=(q_backward, k_backward, v_backward),
            grad_outputs=grad_backward,
            retain_graph=True,
        )

    def total_closure() -> None:
        reset_leaf_grads(q_total, k_total, v_total)
        output = regular_attention(q_total, k_total, v_total)
        output.backward(grad_total)

    try:
        forward_ms = float(triton.testing.do_bench(forward_closure, warmup=warmup_ms, rep=rep_ms))
    except Exception as error:  # pragma: no cover - exercised in remote benchmark runs.
        _raise_with_stage("forward", error)
    try:
        backward_ms = float(triton.testing.do_bench(backward_closure, warmup=warmup_ms, rep=rep_ms))
    except Exception as error:  # pragma: no cover - exercised in remote benchmark runs.
        _raise_with_stage("backward", error)
    try:
        end_to_end_ms = float(triton.testing.do_bench(total_closure, warmup=warmup_ms, rep=rep_ms))
    except Exception as error:  # pragma: no cover - exercised in remote benchmark runs.
        _raise_with_stage("end_to_end", error)

    return {
        "forward_ms": forward_ms,
        "backward_ms": backward_ms,
        "end_to_end_ms": end_to_end_ms,
    }


def benchmark_flash_attention_case(
    case: FlashBenchmarkCase,
    device: torch.device,
    *,
    warmup_ms: int,
    rep_ms: int,
) -> dict[str, float]:
    try:
        q_forward, k_forward, v_forward, _ = make_inputs(case, device)
        q_backward, k_backward, v_backward, grad_backward = make_inputs(case, device)
        q_total, k_total, v_total, grad_total = make_inputs(case, device)
    except Exception as error:  # pragma: no cover - exercised in remote benchmark runs.
        _raise_with_stage("input_setup", error)

    def flash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        with flash_attention_tile_size_override(
            q_tile_size=case.q_tile_size,
            k_tile_size=case.k_tile_size,
        ):
            return FlashAttention2TritonFunction.apply(q, k, v, case.is_causal)

    def forward_closure() -> None:
        flash_attention(q_forward, k_forward, v_forward)

    try:
        output_backward = flash_attention(q_backward, k_backward, v_backward)
    except Exception as error:  # pragma: no cover - exercised in remote benchmark runs.
        _raise_with_stage("backward_graph_setup", error)

    def backward_closure() -> None:
        reset_leaf_grads(q_backward, k_backward, v_backward)
        torch.autograd.grad(
            outputs=output_backward,
            inputs=(q_backward, k_backward, v_backward),
            grad_outputs=grad_backward,
            retain_graph=True,
        )

    def total_closure() -> None:
        reset_leaf_grads(q_total, k_total, v_total)
        output = flash_attention(q_total, k_total, v_total)
        output.backward(grad_total)

    try:
        forward_ms = float(triton.testing.do_bench(forward_closure, warmup=warmup_ms, rep=rep_ms))
    except Exception as error:  # pragma: no cover - exercised in remote benchmark runs.
        _raise_with_stage("forward", error)
    try:
        backward_ms = float(triton.testing.do_bench(backward_closure, warmup=warmup_ms, rep=rep_ms))
    except Exception as error:  # pragma: no cover - exercised in remote benchmark runs.
        _raise_with_stage("backward", error)
    try:
        end_to_end_ms = float(triton.testing.do_bench(total_closure, warmup=warmup_ms, rep=rep_ms))
    except Exception as error:  # pragma: no cover - exercised in remote benchmark runs.
        _raise_with_stage("end_to_end", error)

    return {
        "forward_ms": forward_ms,
        "backward_ms": backward_ms,
        "end_to_end_ms": end_to_end_ms,
    }


def run_benchmark_with_oom_tolerance(
    implementation_name: str,
    benchmark_fn,
    case: FlashBenchmarkCase,
    device: torch.device,
    config: FlashBenchmarkConfig,
) -> dict[str, object]:
    try:
        metrics = benchmark_fn(
            case,
            device,
            warmup_ms=config.warmup_ms,
            rep_ms=config.rep_ms,
        )
        return {
            "status": "ok",
            "failure_kind": None,
            "failure_stage": None,
            "failure_message": None,
            **metrics,
        }
    except Exception as error:  # pragma: no cover - exercised in remote benchmark runs.
        _cleanup_after_failure(device)
        original_error = error.cause if isinstance(error, BenchmarkStageError) else error
        failure_kind = "oom" if _is_oom_error(original_error) else "error"
        failure_stage = error.stage if isinstance(error, BenchmarkStageError) else "case"
        print(
            (
                f"[flash_benchmark] {implementation_name} failed for "
                f"{format_case_label(case)} at stage={failure_stage} "
                f"with {failure_kind}: {original_error}"
            ),
            file=sys.stderr,
        )
        return {
            "status": failure_kind,
            "failure_kind": failure_kind,
            "failure_stage": failure_stage,
            "failure_message": str(original_error),
            **_empty_metrics(),
        }


def benchmark_case(
    case: FlashBenchmarkCase,
    device: torch.device,
    config: FlashBenchmarkConfig,
) -> dict[str, object]:
    pytorch_metrics = run_benchmark_with_oom_tolerance(
        "pytorch",
        benchmark_pytorch_attention_case,
        case,
        device,
        config,
    )
    flash_metrics = run_benchmark_with_oom_tolerance(
        "flash",
        benchmark_flash_attention_case,
        case,
        device,
        config,
    )
    return {
        **asdict(case),
        "pytorch": pytorch_metrics,
        "flash": flash_metrics,
    }


def build_payload(
    config: FlashBenchmarkConfig,
    sweep_grid: list[FlashBenchmarkCase],
    results: list[dict[str, object]],
) -> dict[str, object]:
    return {
        "benchmark": "flash_benchmarking",
        "config": asdict(config),
        "sweep_grid": [asdict(case) for case in sweep_grid],
        "results": results,
    }


def write_payload(payload: dict[str, object], output_path: str | None) -> None:
    serialized = json.dumps(payload, indent=2)
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)


def format_case_label(case: FlashBenchmarkCase) -> str:
    return (
        f"batch={case.batch_size}, seq={case.sequence_length}, d={case.embedding_dim}, "
        f"precision={case.precision}, q_tile={case.q_tile_size}, "
        f"k_tile={case.k_tile_size}, causal={case.is_causal}"
    )


def main() -> None:
    config = parse_args()
    torch.random.manual_seed(config.seed)
    device = resolve_device(config.device)
    sweep_grid = build_sweep_grid(config)

    results: list[dict[str, object]] = []
    if config.run_benchmarks:
        total_cases = len(sweep_grid)
        print(
            (
                f"[flash_benchmark] Running {total_cases} cases "
                f"(warmup={config.warmup_ms} ms, rep={config.rep_ms} ms)"
            ),
            file=sys.stderr,
        )
        for case_index, case in enumerate(sweep_grid, start=1):
            print(
                f"[flash_benchmark] [{case_index}/{total_cases}] {format_case_label(case)}",
                file=sys.stderr,
            )
            results.append(benchmark_case(case, device, config))
        print("[flash_benchmark] Completed all benchmark cases.", file=sys.stderr)

    payload = build_payload(config, sweep_grid, results)
    write_payload(payload, config.output_path)


if __name__ == "__main__":
    main()
