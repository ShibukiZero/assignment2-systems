from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path


FP32_BYTES = 4


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
class FormulaConfig:
    model_size: str
    vocab_size: int
    world_size: int
    output_path: str | None


def parse_args() -> FormulaConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Compute Chapter 3 optimizer-state-sharding memory formulas without "
            "instantiating the model."
        )
    )
    parser.add_argument("--model-size", choices=tuple(MODEL_PRESETS), default="xl")
    parser.add_argument("--vocab-size", type=int, default=10_000)
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--output-path", type=str, default=None)
    args = parser.parse_args()
    return FormulaConfig(
        model_size=args.model_size,
        vocab_size=args.vocab_size,
        world_size=args.world_size,
        output_path=args.output_path,
    )


def bytes_to_gib(num_bytes: int) -> float:
    return num_bytes / (1024**3)


def tensor_numels_in_parameter_order(config: FormulaConfig) -> list[int]:
    preset = MODEL_PRESETS[config.model_size]
    tensor_numels: list[int] = []

    # token_embeddings.weight
    tensor_numels.append(config.vocab_size * preset.d_model)

    for _ in range(preset.num_layers):
        # Attention: q, k, v, output projections
        tensor_numels.extend(
            [
                preset.d_model * preset.d_model,
                preset.d_model * preset.d_model,
                preset.d_model * preset.d_model,
                preset.d_model * preset.d_model,
            ]
        )
        # SwiGLU: w1, w2, w3
        tensor_numels.extend(
            [
                preset.d_model * preset.d_ff,
                preset.d_ff * preset.d_model,
                preset.d_model * preset.d_ff,
            ]
        )
        # RMSNorm: ln1, ln2
        tensor_numels.extend([preset.d_model, preset.d_model])

    # ln_final.weight, lm_head.weight
    tensor_numels.extend([preset.d_model, config.vocab_size * preset.d_model])
    return tensor_numels


def parameter_count_breakdown(config: FormulaConfig) -> dict[str, int]:
    preset = MODEL_PRESETS[config.model_size]
    attention_per_layer = 4 * preset.d_model * preset.d_model
    ffn_per_layer = 3 * preset.d_model * preset.d_ff
    norms_per_layer = 2 * preset.d_model
    embedding = config.vocab_size * preset.d_model
    final_norm = preset.d_model
    lm_head = config.vocab_size * preset.d_model
    total = embedding + preset.num_layers * (attention_per_layer + ffn_per_layer + norms_per_layer) + final_norm + lm_head
    return {
        "embedding": embedding,
        "attention_per_layer": attention_per_layer,
        "ffn_per_layer": ffn_per_layer,
        "norms_per_layer": norms_per_layer,
        "final_norm": final_norm,
        "lm_head": lm_head,
        "total": total,
    }


def sharded_parameter_bytes_per_rank(config: FormulaConfig) -> list[int]:
    per_rank_bytes = [0 for _ in range(config.world_size)]
    for tensor_index, tensor_numel in enumerate(tensor_numels_in_parameter_order(config)):
        owner_rank = tensor_index % config.world_size
        per_rank_bytes[owner_rank] += tensor_numel * FP32_BYTES
    return per_rank_bytes


def checkpoint_bytes(parameter_bytes: int, optimizer_state_bytes: int) -> dict[str, int]:
    gradient_bytes = parameter_bytes
    return {
        "after_model_initialization_bytes": parameter_bytes,
        "before_optimizer_step_bytes": parameter_bytes + gradient_bytes,
        "after_optimizer_step_bytes": parameter_bytes + gradient_bytes + optimizer_state_bytes,
    }


def add_gib_fields(obj):
    if isinstance(obj, dict):
        enriched = {}
        for key, value in obj.items():
            enriched[key] = add_gib_fields(value)
            if key.endswith("_bytes") and isinstance(value, int):
                enriched[key.replace("_bytes", "_gib")] = bytes_to_gib(value)
        return enriched
    if isinstance(obj, list):
        return [add_gib_fields(value) for value in obj]
    return obj


def main() -> None:
    config = parse_args()
    breakdown = parameter_count_breakdown(config)
    total_parameter_count = breakdown["total"]
    parameter_bytes = total_parameter_count * FP32_BYTES
    gradient_bytes = total_parameter_count * FP32_BYTES
    optimizer_state_full_bytes = total_parameter_count * 2 * FP32_BYTES

    idealized_sharded_optimizer_state_bytes = optimizer_state_full_bytes // config.world_size
    implementation_sharded_parameter_bytes_per_rank = sharded_parameter_bytes_per_rank(config)
    implementation_sharded_optimizer_state_bytes_per_rank = [
        2 * parameter_bytes_for_rank
        for parameter_bytes_for_rank in implementation_sharded_parameter_bytes_per_rank
    ]

    report = {
        "config": asdict(config),
        "parameter_count_breakdown": breakdown,
        "formula_summary": {
            "parameter_bytes_formula": "4 * P",
            "gradient_bytes_formula": "4 * P",
            "optimizer_state_full_bytes_formula": "8 * P",
            "optimizer_state_sharded_idealized_bytes_formula": "8 * P / N",
            "full_after_init_formula": "4 * P",
            "full_before_step_formula": "8 * P",
            "full_after_step_formula": "16 * P",
            "sharded_after_init_formula": "4 * P",
            "sharded_before_step_formula": "8 * P",
            "sharded_after_step_idealized_formula": "(8 + 8 / N) * P",
        },
        "numeric_summary": {
            "parameter_bytes": parameter_bytes,
            "gradient_bytes": gradient_bytes,
            "optimizer_state_full_bytes": optimizer_state_full_bytes,
            "optimizer_state_sharded_idealized_bytes": idealized_sharded_optimizer_state_bytes,
        },
        "checkpoints": {
            "full": checkpoint_bytes(parameter_bytes, optimizer_state_full_bytes),
            "sharded_idealized": checkpoint_bytes(parameter_bytes, idealized_sharded_optimizer_state_bytes),
            "sharded_implementation_per_rank": [
                {
                    "rank": rank,
                    **checkpoint_bytes(parameter_bytes, optimizer_state_bytes),
                    "owned_parameter_bytes": owned_parameter_bytes,
                    "owned_optimizer_state_bytes": optimizer_state_bytes,
                }
                for rank, (owned_parameter_bytes, optimizer_state_bytes) in enumerate(
                    zip(
                        implementation_sharded_parameter_bytes_per_rank,
                        implementation_sharded_optimizer_state_bytes_per_rank,
                    )
                )
            ],
        },
    }

    report = add_gib_fields(report)
    result_json = json.dumps(report, indent=2)
    if config.output_path is not None:
        output_path = Path(config.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(result_json + "\n", encoding="utf-8")
    print(result_json)


if __name__ == "__main__":
    main()
