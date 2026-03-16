from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path


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
class AttentionFlops:
    scores_matmul_flops: int
    softmax_flops: int
    value_matmul_flops: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate self-attention FLOPs for the score matmul, softmax, and value matmul "
            "for a representative Chapter 1 profiling configuration."
        )
    )
    parser.add_argument("--model-size", choices=tuple(MODEL_PRESETS), default="2.7b")
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Optional path to write the markdown summary.",
    )
    return parser.parse_args()


def estimate_attention_flops(
    batch_size: int,
    num_heads: int,
    sequence_length: int,
    head_dim: int,
) -> AttentionFlops:
    # Standard GEMM FLOP accounting counts one multiply and one add as two FLOPs.
    scores_matmul_flops = 2 * batch_size * num_heads * sequence_length * sequence_length * head_dim
    value_matmul_flops = 2 * batch_size * num_heads * sequence_length * sequence_length * head_dim

    # Approximate softmax FLOPs per row of length T as:
    #   max reduction: (T - 1)
    #   subtract max: T
    #   exp: T
    #   sum reduction: (T - 1)
    #   divide by sum: T
    # Total ~= 5T + 2(T - 1) = 7T - 2 scalar ops per row.
    softmax_flops = batch_size * num_heads * sequence_length * (7 * sequence_length - 2)

    return AttentionFlops(
        scores_matmul_flops=scores_matmul_flops,
        softmax_flops=softmax_flops,
        value_matmul_flops=value_matmul_flops,
    )


def format_int(value: int) -> str:
    return f"{value:,}"


def format_ratio(numerator: int, denominator: int) -> str:
    return f"{numerator / denominator:.2f}x"


def build_markdown(
    model_size: str,
    batch_size: int,
    context_length: int,
    preset: ModelPreset,
    per_layer: AttentionFlops,
) -> str:
    head_dim = preset.d_model // preset.num_heads
    total_scores = per_layer.scores_matmul_flops * preset.num_layers
    total_softmax = per_layer.softmax_flops * preset.num_layers
    total_value = per_layer.value_matmul_flops * preset.num_layers

    return "\n".join(
        [
            "## 1.1.4(e) Attention FLOP Estimate",
            "",
            "Configuration:",
            "",
            f"- model size: `{model_size}`",
            f"- batch size: `{batch_size}`",
            f"- context length: `{context_length}`",
            f"- d_model: `{preset.d_model}`",
            f"- num_heads: `{preset.num_heads}`",
            f"- head_dim: `{head_dim}`",
            f"- num_layers: `{preset.num_layers}`",
            "",
            "Per-layer FLOP estimate:",
            "",
            "| Operation | FLOPs | Ratio vs softmax |",
            "| --- | ---: | ---: |",
            f"| attention_scores_matmul | {format_int(per_layer.scores_matmul_flops)} | {format_ratio(per_layer.scores_matmul_flops, per_layer.softmax_flops)} |",
            f"| attention_softmax | {format_int(per_layer.softmax_flops)} | 1.00x |",
            f"| attention_value_matmul | {format_int(per_layer.value_matmul_flops)} | {format_ratio(per_layer.value_matmul_flops, per_layer.softmax_flops)} |",
            f"| two matmuls combined | {format_int(per_layer.scores_matmul_flops + per_layer.value_matmul_flops)} | {format_ratio(per_layer.scores_matmul_flops + per_layer.value_matmul_flops, per_layer.softmax_flops)} |",
            "",
            "Whole-model FLOP estimate (all attention layers):",
            "",
            "| Operation | FLOPs |",
            "| --- | ---: |",
            f"| attention_scores_matmul | {format_int(total_scores)} |",
            f"| attention_softmax | {format_int(total_softmax)} |",
            f"| attention_value_matmul | {format_int(total_value)} |",
            "",
            "Notes:",
            "",
            "- GEMM FLOPs count one multiply and one add as two FLOPs.",
            "- Softmax FLOPs use a simple scalar-op approximation of `7T - 2` per row, covering max, subtract, exp, sum, and divide.",
            "- The comparison is most meaningful at the per-layer level, because the NVTX timing also measures one attention layer invocation at a time.",
        ]
    )


def main() -> None:
    args = parse_args()
    preset = MODEL_PRESETS[args.model_size]
    head_dim = preset.d_model // preset.num_heads
    per_layer = estimate_attention_flops(
        batch_size=args.batch_size,
        num_heads=preset.num_heads,
        sequence_length=args.context_length,
        head_dim=head_dim,
    )
    markdown = build_markdown(
        model_size=args.model_size,
        batch_size=args.batch_size,
        context_length=args.context_length,
        preset=preset,
        per_layer=per_layer,
    )

    if args.output_path is not None:
        args.output_path.parent.mkdir(parents=True, exist_ok=True)
        args.output_path.write_text(markdown + "\n", encoding="utf-8")

    print(markdown)


if __name__ == "__main__":
    main()
