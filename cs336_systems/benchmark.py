from __future__ import annotations

import argparse
import json
import statistics
import timeit
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import torch
import torch.nn.functional as F

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.optimizer import AdamW


Mode = Literal["forward", "forward-backward", "train-step"]
Precision = Literal["fp32", "bf16"]


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
    batch_size: int
    vocab_size: int
    rope_theta: float
    mode: Mode
    precision: Precision
    warmup_steps: int
    measure_steps: int
    learning_rate: float
    weight_decay: float
    seed: int
    device: str
    nvtx: bool
    output_path: str | None


def parse_args() -> BenchmarkConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark BasicsTransformerLM with fixed model presets and a small set "
            "of experiment controls."
        )
    )
    parser.add_argument(
        "--model-size",
        choices=tuple(MODEL_PRESETS),
        default="small",
        help="Model preset from the assignment handout.",
    )
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--vocab-size", type=int, default=10_000)
    parser.add_argument("--rope-theta", type=float, default=10_000.0)
    parser.add_argument(
        "--mode",
        choices=("forward", "forward-backward", "train-step"),
        default="forward",
        help="What work to include in each measured step.",
    )
    parser.add_argument(
        "--precision",
        choices=("fp32", "bf16"),
        default="fp32",
        help="Use BF16 autocast on CUDA, or run everything in FP32.",
    )
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--measure-steps", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Use auto to prefer CUDA when available.",
    )
    parser.add_argument(
        "--nvtx",
        action="store_true",
        help="Annotate warmup, measured iterations, and step phases with NVTX ranges.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Optional path to write the benchmark JSON result.",
    )
    args = parser.parse_args()
    return BenchmarkConfig(
        model_size=args.model_size,
        context_length=args.context_length,
        batch_size=args.batch_size,
        vocab_size=args.vocab_size,
        rope_theta=args.rope_theta,
        mode=args.mode,
        precision=args.precision,
        warmup_steps=args.warmup_steps,
        measure_steps=args.measure_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seed=args.seed,
        device=args.device,
        nvtx=args.nvtx,
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


def make_batch(config: BenchmarkConfig, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    input_ids = torch.randint(
        low=0,
        high=config.vocab_size,
        size=(config.batch_size, config.context_length),
        device=device,
    )
    targets = torch.randint(
        low=0,
        high=config.vocab_size,
        size=(config.batch_size, config.context_length),
        device=device,
    )
    return input_ids, targets


def make_precision_context(config: BenchmarkConfig, device: torch.device):
    if config.precision == "fp32":
        return nullcontext()
    if device.type != "cuda":
        raise ValueError("BF16 autocast is only supported on CUDA in this script.")
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)


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


def run_forward_only(
    model: BasicsTransformerLM,
    input_ids: torch.Tensor,
    precision_context,
) -> None:
    model.eval()
    with torch.no_grad():
        with precision_context:
            _ = model(input_ids)


def forward_and_loss(
    model: BasicsTransformerLM,
    input_ids: torch.Tensor,
    targets: torch.Tensor,
    precision_context,
) -> torch.Tensor:
    model.train()
    with precision_context:
        logits = model(input_ids)
        # Random targets are enough here because we care about runtime, not convergence.
        return F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))


def run_step(
    config: BenchmarkConfig,
    model: BasicsTransformerLM,
    optimizer: AdamW | None,
    input_ids: torch.Tensor,
    targets: torch.Tensor,
    device: torch.device,
) -> dict[str, float]:
    # Keep model architecture choices in presets, and expose only experiment controls via CLI.
    precision_context = make_precision_context(config, device)
    timings: dict[str, float] = {}

    if config.mode == "forward":
        start = timeit.default_timer()
        with maybe_nvtx_range("forward", config.nvtx, device):
            run_forward_only(model, input_ids, precision_context)
        maybe_synchronize(device)
        forward_end = timeit.default_timer()
        timings["forward_seconds"] = forward_end - start
        timings["total_seconds"] = timings["forward_seconds"]
    else:
        model.zero_grad(set_to_none=True)
        start = timeit.default_timer()
        with maybe_nvtx_range("forward", config.nvtx, device):
            loss = forward_and_loss(model, input_ids, targets, precision_context)
        maybe_synchronize(device)
        forward_end = timeit.default_timer()

        with maybe_nvtx_range("backward", config.nvtx, device):
            loss.backward()
        maybe_synchronize(device)
        backward_end = timeit.default_timer()

        timings["forward_seconds"] = forward_end - start
        timings["backward_seconds"] = backward_end - forward_end

        if config.mode == "train-step":
            if optimizer is None:
                raise ValueError("train-step mode requires an optimizer.")
            with maybe_nvtx_range("optimizer_step", config.nvtx, device):
                optimizer.step()
            maybe_synchronize(device)
            optimizer_end = timeit.default_timer()
            timings["optimizer_step_seconds"] = optimizer_end - backward_end
            timings["total_seconds"] = optimizer_end - start
        elif config.mode == "forward-backward":
            timings["total_seconds"] = backward_end - start
        else:
            raise ValueError(f"Unsupported mode: {config.mode}")

    return timings


def summarize_series(series: list[float]) -> dict[str, float]:
    stdev = statistics.stdev(series) if len(series) > 1 else 0.0
    return {
        "mean_seconds": statistics.mean(series),
        "stdev_seconds": stdev,
        "min_seconds": min(series),
        "max_seconds": max(series),
    }


def summarize_timings(step_timings: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    if not step_timings:
        raise ValueError("step_timings must be non-empty.")

    metric_names = step_timings[0].keys()
    summary: dict[str, dict[str, float]] = {}
    for metric_name in metric_names:
        summary[metric_name] = summarize_series([timing[metric_name] for timing in step_timings])
    return summary


def main() -> None:
    config = parse_args()
    device = resolve_device(config.device)
    torch.manual_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)

    model = make_model(config, device)
    optimizer = None
    if config.mode == "train-step":
        optimizer = AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

    input_ids, targets = make_batch(config, device)

    for step_idx in range(config.warmup_steps):
        with maybe_nvtx_range(f"warmup_{step_idx}", config.nvtx, device):
            run_step(config, model, optimizer, input_ids, targets, device)

    step_timings: list[dict[str, float]] = []
    for step_idx in range(config.measure_steps):
        with maybe_nvtx_range(f"measure_{step_idx}", config.nvtx, device):
            step_timings.append(run_step(config, model, optimizer, input_ids, targets, device))

    summary = summarize_timings(step_timings)
    result = {
        "config": asdict(config),
        "device": str(device),
        "model_preset": asdict(MODEL_PRESETS[config.model_size]),
        "num_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "timings": summary,
        "step_timings_seconds": step_timings,
    }
    result_json = json.dumps(result, indent=2)

    if config.output_path is not None:
        output_path = Path(config.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(result_json + "\n", encoding="utf-8")

    print(result_json)


if __name__ == "__main__":
    main()
