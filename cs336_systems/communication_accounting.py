from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass


FP32_BYTES = 4
BF16_BYTES = 2


@dataclass(frozen=True)
class XXLConfig:
    d_model: int = 16_384
    d_ff: int = 53_248
    num_blocks: int = 126
    h100_capacity_gb: float = 80.0
    tpu_v5p_capacity_gb: float = 95.0
    tpu_v5p_ici_bandwidth: float = 2 * 9 * 10**10
    tpu_v5p_flops: float = 4.6 * 10**14
    mesh_x: int = 16
    mesh_y: int = 4
    mesh_axes_x: int = 2
    mesh_axes_y: int = 1
    batch_size: int | None = None
    context_length: int | None = None


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


def saved_activations_bf16_bytes(config: XXLConfig) -> int | None:
    if config.batch_size is None or config.context_length is None:
        return None
    coefficient = config.num_blocks * (config.d_model + config.d_ff) * BF16_BYTES
    return coefficient * config.batch_size * config.context_length


def total_training_memory_formula(config: XXLConfig) -> str:
    fp32_total = total_fp32_model_state_bytes(config)
    coefficient = config.num_blocks * (config.d_model + config.d_ff) * BF16_BYTES
    return f"{fp32_total} + {coefficient} * B * T bytes"


def required_h100_formula(config: XXLConfig) -> str:
    fp32_total = total_fp32_model_state_bytes(config)
    coefficient = config.num_blocks * (config.d_model + config.d_ff) * BF16_BYTES
    denominator = config.h100_capacity_gb * 10**9
    return f"ceil(({fp32_total} + {coefficient} * B * T) / {denominator})"


def fsdp_memory_per_device_formula(config: XXLConfig) -> str:
    fp32_total = total_fp32_model_state_bytes(config)
    activation_formula = saved_activations_bf16_formula(config)
    return (
        f"({fp32_total} + 0.5 * ({activation_formula})) / N_FSDP + "
        f"0.5 * ({activation_formula})"
    )


def minimum_fsdp_devices_formula(config: XXLConfig) -> str:
    fp32_total = total_fp32_model_state_bytes(config)
    activation_formula = saved_activations_bf16_formula(config)
    capacity_bytes = int(config.tpu_v5p_capacity_gb * 10**9)
    return (
        "N_FSDP > "
        f"({fp32_total} + 0.5 * ({activation_formula})) / "
        f"({capacity_bytes} - 0.5 * ({activation_formula}))"
    )


def minimum_fsdp_devices_numeric(config: XXLConfig) -> int | None:
    activation_bytes = saved_activations_bf16_bytes(config)
    if activation_bytes is None:
        return None

    fp32_total = total_fp32_model_state_bytes(config)
    capacity_bytes = config.tpu_v5p_capacity_gb * 10**9
    replicated_activation = 0.5 * activation_bytes
    sharded_numerator = fp32_total + replicated_activation
    denominator = capacity_bytes - replicated_activation
    if denominator <= 0:
        return None

    threshold = sharded_numerator / denominator
    return int(threshold) + 1


def build_typical_examples(config: XXLConfig) -> list[dict[str, float | int]]:
    fp32_total = total_fp32_model_state_bytes(config)
    coefficient = config.num_blocks * (config.d_model + config.d_ff) * BF16_BYTES
    examples: list[dict[str, float | int]] = []

    batch_size = 4
    for context_length in (128, 256, 512):
        activation_bytes = coefficient * batch_size * context_length
        total_h100 = (fp32_total + activation_bytes) / (config.h100_capacity_gb * 10**9)
        fsdp_threshold = (fp32_total + 0.5 * activation_bytes) / (
            config.tpu_v5p_capacity_gb * 10**9 - 0.5 * activation_bytes
        )
        examples.append(
            {
                "batch_size": batch_size,
                "context_length": context_length,
                "saved_activations_bytes": activation_bytes,
                "saved_activations_gib": bytes_to_gib(activation_bytes),
                "required_h100_80gb": total_h100,
                "minimum_fsdp_devices_for_less_than_95gb": int(fsdp_threshold) + 1,
            }
        )
    return examples


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
        "total_training_memory": {
            "formula": "fp32_model_state_total + saved_activations_bf16",
            "instantiated_formula": total_training_memory_formula(config),
        },
        "required_h100_80gb": {
            "lower_bound_without_activations": h100_80gb_equivalent(fp32_total, config.h100_capacity_gb),
            "formula_including_activations": required_h100_formula(config),
        },
        "typical_examples_batch4": build_typical_examples(config),
    }


def build_question_b_report(config: XXLConfig) -> dict[str, object]:
    report: dict[str, object] = {
        "memory_per_device": {
            "formula": (
                "(master_weights + optimizer_state + gradients + 0.5 * activations) / N_FSDP "
                "+ 0.5 * activations"
            ),
            "instantiated_formula": fsdp_memory_per_device_formula(config),
        },
        "minimum_fsdp_devices_for_less_than_95gb": {
            "formula": minimum_fsdp_devices_formula(config),
        },
    }

    activation_bytes = saved_activations_bf16_bytes(config)
    minimum_devices = minimum_fsdp_devices_numeric(config)
    if activation_bytes is not None and minimum_devices is not None:
        report["activation_instantiation"] = {
            "batch_size": config.batch_size,
            "context_length": config.context_length,
            "saved_activations_bytes": activation_bytes,
            "saved_activations_gib": bytes_to_gib(activation_bytes),
        }
        report["minimum_fsdp_devices_for_less_than_95gb"]["instantiated_value"] = minimum_devices

    report["typical_examples_batch4"] = build_typical_examples(config)

    return report


def total_mesh_devices(config: XXLConfig) -> int:
    return config.mesh_x * config.mesh_y


def arithmetic_intensity_alpha(config: XXLConfig) -> float:
    return config.tpu_v5p_flops / config.tpu_v5p_ici_bandwidth


def fsdp_tp_forward_math_formula() -> str:
    return "T_math = 4 * B * D * F / (N * C)"


def fsdp_tp_forward_fsdp_comms_formula() -> str:
    return "T_FSDP = 4 * D * F / (Y * W_ici * M_X)"


def fsdp_tp_forward_tp_comms_formula() -> str:
    return "T_TP = 4 * B * D / (X * W_ici * M_Y)"


def per_device_token_batch_threshold_formula() -> str:
    return "b >= C / (Y * W_ici * M_X) = alpha / (Y * M_X)"


def tp_feasibility_formula() -> str:
    return "F >= (C / W_ici) * (Y / M_Y) = alpha * Y / M_Y"


def per_device_token_batch_threshold(config: XXLConfig) -> float:
    return config.tpu_v5p_flops / (
        config.mesh_y * config.tpu_v5p_ici_bandwidth * config.mesh_axes_x
    )


def tp_compute_bound_condition_satisfied(config: XXLConfig) -> bool:
    rhs = arithmetic_intensity_alpha(config) * config.mesh_y / config.mesh_axes_y
    return config.d_ff >= rhs


def build_question_c_report(config: XXLConfig) -> dict[str, object]:
    alpha = arithmetic_intensity_alpha(config)
    threshold_tokens = per_device_token_batch_threshold(config)
    minimum_integer_tokens = math.ceil(threshold_tokens)
    total_devices = total_mesh_devices(config)
    overall_threshold_tokens = threshold_tokens * total_devices
    minimum_integer_overall_tokens = minimum_integer_tokens * total_devices
    tp_rhs = alpha * config.mesh_y / config.mesh_axes_y

    report: dict[str, object] = {
        "forward_time_formulas": {
            "math": fsdp_tp_forward_math_formula(),
            "fsdp_communication": fsdp_tp_forward_fsdp_comms_formula(),
            "tp_communication": fsdp_tp_forward_tp_comms_formula(),
        },
        "compute_bound_condition": "T_math >= max(T_FSDP, T_TP)",
        "per_device_token_batch_threshold": {
            "formula": per_device_token_batch_threshold_formula(),
            "value": threshold_tokens,
            "minimum_integer_value": minimum_integer_tokens,
        },
        "overall_token_batch_threshold": {
            "formula": "B = b * N",
            "value": overall_threshold_tokens,
            "minimum_integer_value": minimum_integer_overall_tokens,
        },
        "tp_communication_check": {
            "formula": tp_feasibility_formula(),
            "lhs_d_ff": config.d_ff,
            "rhs": tp_rhs,
            "condition_satisfied": tp_compute_bound_condition_satisfied(config),
        },
        "hardware_constants": {
            "C_flops_per_device": config.tpu_v5p_flops,
            "W_ici_bytes_per_second": config.tpu_v5p_ici_bandwidth,
            "alpha": alpha,
            "X": config.mesh_x,
            "Y": config.mesh_y,
            "M_X": config.mesh_axes_x,
            "M_Y": config.mesh_axes_y,
            "N": total_devices,
        },
    }

    if config.context_length is not None and config.context_length > 0:
        report["equivalent_sequence_batch_if_context_length_fixed"] = {
            "context_length": config.context_length,
            "per_device_sequences": threshold_tokens / config.context_length,
            "minimum_integer_per_device_sequences": math.ceil(
                minimum_integer_tokens / config.context_length
            ),
            "overall_sequences": overall_threshold_tokens / config.context_length,
            "minimum_integer_overall_sequences": math.ceil(
                minimum_integer_overall_tokens / config.context_length
            ),
        }

    return report


def parse_args() -> XXLConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Compute the symbolic/numeric memory accounting terms needed for "
            "Assignment 2 Section 2.4(a) and 2.4(b)."
        )
    )
    parser.add_argument("--d-model", type=int, default=16_384)
    parser.add_argument("--d-ff", type=int, default=53_248)
    parser.add_argument("--num-blocks", type=int, default=126)
    parser.add_argument("--h100-capacity-gb", type=float, default=80.0)
    parser.add_argument("--tpu-v5p-capacity-gb", type=float, default=95.0)
    parser.add_argument("--tpu-v5p-ici-bandwidth", type=float, default=2 * 9 * 10**10)
    parser.add_argument("--tpu-v5p-flops", type=float, default=4.6 * 10**14)
    parser.add_argument("--mesh-x", type=int, default=16)
    parser.add_argument("--mesh-y", type=int, default=4)
    parser.add_argument("--mesh-axes-x", type=int, default=2)
    parser.add_argument("--mesh-axes-y", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--context-length", type=int, default=None)
    args = parser.parse_args()
    return XXLConfig(
        d_model=args.d_model,
        d_ff=args.d_ff,
        num_blocks=args.num_blocks,
        h100_capacity_gb=args.h100_capacity_gb,
        tpu_v5p_capacity_gb=args.tpu_v5p_capacity_gb,
        tpu_v5p_ici_bandwidth=args.tpu_v5p_ici_bandwidth,
        tpu_v5p_flops=args.tpu_v5p_flops,
        mesh_x=args.mesh_x,
        mesh_y=args.mesh_y,
        mesh_axes_x=args.mesh_axes_x,
        mesh_axes_y=args.mesh_axes_y,
        batch_size=args.batch_size,
        context_length=args.context_length,
    )


def main() -> None:
    config = parse_args()
    report = {
        "question_a": build_question_a_report(config),
        "question_b": build_question_b_report(config),
        "question_c": build_question_c_report(config),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
