from __future__ import annotations

import argparse
import dataclasses
import importlib
import inspect
import json
import sys
from itertools import product
from pathlib import Path

import triton.testing


DEFAULT_KERNEL_NAMES = ("forward", "backward_dq", "backward_dkdv")
DEFAULT_NUM_WARPS = (2, 4, 8)
DEFAULT_NUM_STAGES = (2, 3, 4)
DEFAULT_CONFIG_PATH = Path(__file__).with_name("flash_attention_autotune_configs.json")
DEFAULT_SEARCH_SPACE_ARCHIVE_PATH = Path(".agents/logs/flash_leaderboard/autotune_search_space.json")


def _is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def _build_power_of_two_values(min_value: int, max_value: int) -> list[int]:
    if min_value > max_value:
        raise ValueError("Expected min_value <= max_value.")
    if not _is_power_of_two(min_value) or not _is_power_of_two(max_value):
        raise ValueError("Expected min_value and max_value to both be powers of two.")

    values: list[int] = []
    current = min_value
    while current <= max_value:
        values.append(current)
        current *= 2
    return values


def _dedupe_positive(values: list[int], *, field_name: str) -> list[int]:
    if not values:
        raise ValueError(f"Expected at least one value for {field_name}.")
    if any(value <= 0 for value in values):
        raise ValueError(f"Expected all values for {field_name} to be positive.")
    return sorted(set(values))


def _resolve_values(
    *,
    explicit_values: list[int] | None,
    min_value: int,
    max_value: int,
) -> list[int]:
    if explicit_values is not None:
        return _dedupe_positive(explicit_values, field_name="explicit values")
    return _build_power_of_two_values(min_value, max_value)


def _build_kernel_configs(
    q_tile_sizes: list[int],
    k_tile_sizes: list[int],
    *,
    num_warps_values: list[int],
    num_stages_values: list[int],
) -> list[dict[str, int]]:
    configs: list[dict[str, int]] = []
    for q_tile_size, k_tile_size, num_warps, num_stages in product(
        q_tile_sizes,
        k_tile_sizes,
        num_warps_values,
        num_stages_values,
    ):
        configs.append(
            {
                "Q_TILE_SIZE": q_tile_size,
                "K_TILE_SIZE": k_tile_size,
                "num_warps": num_warps,
                "num_stages": num_stages,
            }
        )
    return configs


def _prune_search_space(
    payload: dict[str, list[dict[str, int]]],
    *,
    sequence_length: int,
    d_head: int,
) -> dict[str, list[dict[str, int]]]:
    pruned_payload: dict[str, list[dict[str, int]]] = {}
    for kernel_name, configs in payload.items():
        pruned_configs: list[dict[str, int]] = []
        for config in configs:
            q_tile_size = int(config["Q_TILE_SIZE"])
            k_tile_size = int(config["K_TILE_SIZE"])
            num_warps = int(config["num_warps"])
            num_stages = int(config["num_stages"])

            # Keep the first-round search broad, but stay within the range that has
            # been stable for the current FlashAttention kernels on the remote stack.
            if q_tile_size > 128 or k_tile_size > 128:
                continue

            # The dK/dV kernel is currently the only one still tripping Triton
            # backend assertions on the remote stack, so search it more conservatively.
            if kernel_name == "backward_dkdv":
                if q_tile_size > 64 or k_tile_size > 64:
                    continue
                if num_warps > 4 or num_stages > 3:
                    continue

            if q_tile_size > max(16, 2 * sequence_length):
                continue
            if k_tile_size > max(16, 2 * sequence_length):
                continue

            score_tile_elements = q_tile_size * k_tile_size
            if score_tile_elements > 16_384:
                continue

            approximate_working_set = (q_tile_size + 2 * k_tile_size) * d_head + score_tile_elements
            if approximate_working_set > 131_072:
                continue

            if q_tile_size * k_tile_size < 1_024 and num_warps > 4:
                continue
            if q_tile_size * k_tile_size < 4_096 and num_stages > 3:
                continue

            if max(q_tile_size, k_tile_size) >= 128 and num_warps > 4:
                continue
            if max(q_tile_size, k_tile_size) >= 128 and num_stages > 3:
                continue
            if min(q_tile_size, k_tile_size) <= 16 and max(q_tile_size, k_tile_size) >= 128:
                continue

            pruned_configs.append(config)

        pruned_payload[kernel_name] = pruned_configs if pruned_configs else configs
    return pruned_payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Search a large FlashAttention autotune config space, then export the "
            "best configs back to flash_attention_autotune_configs.json."
        )
    )
    parser.add_argument("--min-tile-size", type=int, default=16)
    parser.add_argument("--max-tile-size", type=int, default=256)
    parser.add_argument("--q-tile-sizes", nargs="+", type=int, default=None)
    parser.add_argument("--k-tile-sizes", nargs="+", type=int, default=None)
    parser.add_argument("--num-warps", nargs="+", type=int, default=list(DEFAULT_NUM_WARPS))
    parser.add_argument("--num-stages", nargs="+", type=int, default=list(DEFAULT_NUM_STAGES))
    parser.add_argument(
        "--kernels",
        nargs="+",
        default=list(DEFAULT_KERNEL_NAMES),
        choices=DEFAULT_KERNEL_NAMES,
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--d-head", type=int, default=64)
    parser.add_argument("--sequence-length", type=int, default=16_384)
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--warmup-ms", type=int, default=200)
    parser.add_argument("--rep-ms", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cuda"), default="auto")
    parser.add_argument(
        "--skip-final-benchmark",
        action="store_true",
        help="Only run the autotune search and export the best-config JSON.",
    )
    parser.add_argument(
        "--archive-search-space-path",
        type=Path,
        default=DEFAULT_SEARCH_SPACE_ARCHIVE_PATH,
        help="Optional archive path for the generated large search-space JSON.",
    )
    parser.add_argument(
        "--benchmark-output-path",
        type=str,
        default=".agents/logs/flash_leaderboard/autotune_search.json",
    )
    parser.add_argument(
        "--config-path",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="The default config JSON consumed by flash_attention.py.",
    )
    return parser.parse_args()


def _build_search_payload(
    *,
    config: object,
    device: object,
    latency_ms: float,
    best_payload: dict[str, list[dict[str, int]]],
    search_space_path: Path,
    per_kernel_counts: dict[str, int],
) -> dict[str, object]:
    config_dict = dataclasses.asdict(config) if dataclasses.is_dataclass(config) else dict(vars(config))
    batch_size = int(config_dict["batch_size"])
    num_heads = int(config_dict["num_heads"])
    sequence_length = int(config_dict["sequence_length"])
    d_head = int(config_dict["d_head"])
    effective_batch = batch_size * num_heads
    return {
        "benchmark": "flash_autotune_search",
        "config": config_dict,
        "derived": {
            "device": str(device),
            "effective_batch_for_flash_attention": effective_batch,
            "qkv_shape": [effective_batch, sequence_length, d_head],
            "d_model": num_heads * d_head,
            "search_space_path": str(search_space_path),
            "autotune_candidate_counts": per_kernel_counts,
        },
        "results": {
            "forward_backward_ms": latency_ms,
            "best_configs": best_payload,
        },
    }


def main() -> None:
    args = parse_args()
    q_tile_sizes = _resolve_values(
        explicit_values=args.q_tile_sizes,
        min_value=args.min_tile_size,
        max_value=args.max_tile_size,
    )
    k_tile_sizes = _resolve_values(
        explicit_values=args.k_tile_sizes,
        min_value=args.min_tile_size,
        max_value=args.max_tile_size,
    )
    num_warps_values = _dedupe_positive(args.num_warps, field_name="num_warps")
    num_stages_values = _dedupe_positive(args.num_stages, field_name="num_stages")

    payload = {
        kernel_name: _build_kernel_configs(
            q_tile_sizes,
            k_tile_sizes,
            num_warps_values=num_warps_values,
            num_stages_values=num_stages_values,
        )
        for kernel_name in args.kernels
    }
    pruned_payload = _prune_search_space(
        payload,
        sequence_length=args.sequence_length,
        d_head=args.d_head,
    )
    total_configs = sum(len(configs) for configs in pruned_payload.values())
    per_kernel_counts = {kernel_name: len(configs) for kernel_name, configs in pruned_payload.items()}

    args.archive_search_space_path.parent.mkdir(parents=True, exist_ok=True)
    args.archive_search_space_path.write_text(json.dumps(pruned_payload, indent=2) + "\n", encoding="utf-8")
    print(
        f"[flash_autotune_search] Archived {total_configs} configs to {args.archive_search_space_path} "
        f"(per-kernel={per_kernel_counts})",
        file=sys.stderr,
    )

    original_config_text = args.config_path.read_text(encoding="utf-8") if args.config_path.exists() else None
    args.config_path.write_text(json.dumps(pruned_payload, indent=2) + "\n", encoding="utf-8")

    try:
        flash_attention_module = importlib.import_module("cs336_systems.flash_attention")
        flash_attention_module = importlib.reload(flash_attention_module)

        benchmark_module = importlib.import_module("cs336_systems.flash_leaderboard_benchmark")
        benchmark_module = importlib.reload(benchmark_module)

        config_kwargs = dict(
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
            compile_flash=False,
            output_path=args.benchmark_output_path,
        )
        config_field_names = {
            field.name
            for field in dataclasses.fields(benchmark_module.FlashLeaderboardBenchmarkConfig)
        }
        if "autotune_config_path" in config_field_names:
            config_kwargs["autotune_config_path"] = None
        config = benchmark_module.FlashLeaderboardBenchmarkConfig(**config_kwargs)

        benchmark_module.torch.random.manual_seed(config.seed)
        device = benchmark_module.resolve_device(config.device)
        q, k, v = benchmark_module.make_inputs(config, device)
        make_flash_runner_signature = inspect.signature(benchmark_module.make_flash_runner)
        if len(make_flash_runner_signature.parameters) == 1:
            flash_forward_backward = benchmark_module.make_flash_runner(config)
        else:
            flash_forward_backward = benchmark_module.make_flash_runner(
                config,
                flash_attention_module,
            )

        progress_counter = 0
        original_do_bench = triton.testing.do_bench

        def progress_do_bench(fn, *bench_args, **bench_kwargs):
            nonlocal progress_counter
            progress_counter += 1
            if (
                progress_counter <= 5
                or progress_counter == total_configs
                or progress_counter % 10 == 0
            ):
                print(
                    f"[flash_autotune_search] Autotune bench {progress_counter}/{total_configs}",
                    file=sys.stderr,
                )
            return original_do_bench(fn, *bench_args, **bench_kwargs)

        triton.testing.do_bench = progress_do_bench
        benchmark_module.triton.testing.do_bench = progress_do_bench
        try:
            benchmark_module.reset_leaf_grads(q, k, v)
            flash_forward_backward(q, k, v)
        finally:
            triton.testing.do_bench = original_do_bench
            benchmark_module.triton.testing.do_bench = original_do_bench

        best_configs = flash_attention_module.get_flash_attention_best_configs()
        best_payload = {
            kernel_name: [kernel_config]
            for kernel_name, kernel_config in best_configs.items()
            if kernel_config is not None
        }
        if not best_payload:
            raise RuntimeError("Autotune search did not produce any best configs.")

        args.config_path.write_text(json.dumps(best_payload, indent=2) + "\n", encoding="utf-8")
        print(
            f"[flash_autotune_search] Exported best configs to {args.config_path}: "
            f"{json.dumps(best_payload, sort_keys=True)}",
            file=sys.stderr,
        )

        if args.skip_final_benchmark:
            return

        def closure() -> None:
            benchmark_module.reset_leaf_grads(q, k, v)
            flash_forward_backward(q, k, v)

        latency_ms = float(
            triton.testing.do_bench(
                closure,
                warmup=config.warmup_ms,
                rep=config.rep_ms,
            )
        )
        payload_out = _build_search_payload(
            config=config,
            device=device,
            latency_ms=latency_ms,
            best_payload=best_payload,
            search_space_path=args.archive_search_space_path,
            per_kernel_counts=per_kernel_counts,
        )
        benchmark_module.write_payload(payload_out, config.output_path)
    except Exception:
        if original_config_text is not None:
            args.config_path.write_text(original_config_text, encoding="utf-8")
        else:
            try:
                args.config_path.unlink()
            except FileNotFoundError:
                pass
        raise


if __name__ == "__main__":
    main()
