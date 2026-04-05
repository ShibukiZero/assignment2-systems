from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass


FP32_BYTES = 4
BF16_BYTES = 2


@dataclass(frozen=True)
class XXLConfig:
    d_model: int = 16_384
    d_ff: int = 53_248
    num_blocks: int = 126
    h100_capacity_gb: float = 80.0


def ffn_parameters_per_block(config: XXLConfig) -> int:
    """
    Count only the FFN weights described in the handout:
    - first linear: d_model -> d_ff
    - second linear: d_ff -> d_model

    We ignore biases because the problem statement reduces each FFN to two
    linear layers and the bias terms are negligible relative to the weights.
    """
    return 2 * config.d_model * config.d_ff


def total_parameter_count(config: XXLConfig) -> int:
    return config.num_blocks * ffn_parameters_per_block(config)


def master_weights_bytes(config: XXLConfig) -> int:
    return total_parameter_count(config) * FP32_BYTES


def accumulated_gradients_bytes(config: XXLConfig) -> int:
    return total_parameter_count(config) * FP32_BYTES


def optimizer_state_bytes(config: XXLConfig) -> int:
    """
    Assume Adam-style optimizer state:
    - first moment (FP32)
    - second moment (FP32)
    """
    return total_parameter_count(config) * 2 * FP32_BYTES


def total_fp32_model_state_bytes(config: XXLConfig) -> int:
    return (
        master_weights_bytes(config)
        + accumulated_gradients_bytes(config)
        + optimizer_state_bytes(config)
    )


def saved_activations_bf16_formula(config: XXLConfig) -> str:
    """
    A minimal FFN-only activation model for backward.

    For each block, backward needs the inputs to the two linear layers:
    - x for the first linear, shape [B, T, d_model]
    - h for the second linear, shape [B, T, d_ff]

    Since the handout's simplified model removes attention, there is no O(T^2)
    attention-state term, but FFN activations still scale with the token count
    B * T. We therefore keep this part symbolic instead of hard-coding a batch
    size or context length into the answer for part (a).

    The resulting BF16 activation formula is:

        num_blocks * B * T * (d_model + d_ff) * BF16_BYTES
    """
    coefficient = config.num_blocks * (config.d_model + config.d_ff) * BF16_BYTES
    return f"{coefficient} * B * T bytes"


def bytes_to_gib(num_bytes: int) -> float:
    return num_bytes / (1024**3)


def h100_80gb_equivalent(num_bytes: int, h100_capacity_gb: float) -> float:
    return num_bytes / (h100_capacity_gb * 10**9)


def build_question_a_report(config: XXLConfig) -> dict[str, float | int | dict[str, float | int]]:
    master = master_weights_bytes(config)
    gradients = accumulated_gradients_bytes(config)
    optimizer = optimizer_state_bytes(config)
    fp32_total = total_fp32_model_state_bytes(config)

    return {
        "config": asdict(config),
        "parameter_count": total_parameter_count(config),
        "master_weights": {
            "bytes": master,
            "gib": bytes_to_gib(master),
        },
        "accumulated_gradients": {
            "bytes": gradients,
            "gib": bytes_to_gib(gradients),
        },
        "optimizer_state": {
            "bytes": optimizer,
            "gib": bytes_to_gib(optimizer),
        },
        "fp32_model_state_total": {
            "bytes": fp32_total,
            "gib": bytes_to_gib(fp32_total),
            "h100_80gb_equivalent": h100_80gb_equivalent(fp32_total, config.h100_capacity_gb),
        },
        "saved_activations_bf16": {
            "formula": "num_blocks * B * T * (d_model + d_ff) * 2 bytes",
            "instantiated_formula": saved_activations_bf16_formula(config),
        },
    }


def parse_args() -> XXLConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Compute the symbolic/numeric memory accounting terms needed for "
            "Assignment 2 Section 2.4(a)."
        )
    )
    parser.add_argument("--d-model", type=int, default=16_384)
    parser.add_argument("--d-ff", type=int, default=53_248)
    parser.add_argument("--num-blocks", type=int, default=126)
    parser.add_argument("--h100-capacity-gb", type=float, default=80.0)
    args = parser.parse_args()
    return XXLConfig(
        d_model=args.d_model,
        d_ff=args.d_ff,
        num_blocks=args.num_blocks,
        h100_capacity_gb=args.h100_capacity_gb,
    )


def main() -> None:
    config = parse_args()
    report = build_question_a_report(config)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
