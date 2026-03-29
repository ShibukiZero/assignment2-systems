from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render flash_benchmark.py JSON output into a Markdown table, "
            "with one row per benchmark grid case."
        )
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        required=True,
        help="Path to the flash benchmark JSON results file.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Optional path to write the Markdown output.",
    )
    return parser.parse_args()


def format_metric(value: float | None, status: str) -> str:
    if value is None:
        if status == "oom":
            return "OOM"
        if status == "error":
            return "ERROR"
        return ""
    return f"{value:.3f}"


def format_speedup(
    pytorch_value: float | None,
    flash_value: float | None,
    pytorch_status: str,
    flash_status: str,
) -> str:
    if pytorch_value is None or flash_value is None:
        return ""
    if pytorch_status != "ok" or flash_status != "ok":
        return ""
    return f"{pytorch_value / flash_value:.2f}x"


def load_results(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sort_results(results: list[dict]) -> list[dict]:
    precision_order = {"fp32": 0, "bf16": 1}
    return sorted(
        results,
        key=lambda row: (
            row["sequence_length"],
            row["embedding_dim"],
            precision_order.get(row["precision"], 99),
            row["q_tile_size"],
            row["k_tile_size"],
        ),
    )


def render_markdown(data: dict) -> str:
    config = data["config"]
    results = sort_results(data["results"])

    lines = [
        "# Flash Benchmark Table",
        "",
        f"- Source: `{config['output_path']}`" if config.get("output_path") else "- Source: input JSON",
        f"- Batch size: `{config['batch_size']}`",
        f"- Causal: `{config['is_causal']}`",
        f"- Warmup: `{config['warmup_ms']}` ms",
        f"- Rep: `{config['rep_ms']}` ms",
        "",
        "| Seq | D | Precision | Q tile | K tile | PT status | PT fwd (ms) | PT bwd (ms) | PT e2e (ms) | Flash status | Flash fwd (ms) | Flash bwd (ms) | Flash e2e (ms) | E2E speedup |",
        "| ---: | ---: | --- | ---: | ---: | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |",
    ]

    for row in results:
        pytorch = row["pytorch"]
        flash = row["flash"]
        lines.append(
            "| "
            f"{row['sequence_length']} | "
            f"{row['embedding_dim']} | "
            f"{row['precision']} | "
            f"{row['q_tile_size']} | "
            f"{row['k_tile_size']} | "
            f"{pytorch['status']} | "
            f"{format_metric(pytorch['forward_ms'], pytorch['status'])} | "
            f"{format_metric(pytorch['backward_ms'], pytorch['status'])} | "
            f"{format_metric(pytorch['end_to_end_ms'], pytorch['status'])} | "
            f"{flash['status']} | "
            f"{format_metric(flash['forward_ms'], flash['status'])} | "
            f"{format_metric(flash['backward_ms'], flash['status'])} | "
            f"{format_metric(flash['end_to_end_ms'], flash['status'])} | "
            f"{format_speedup(pytorch['end_to_end_ms'], flash['end_to_end_ms'], pytorch['status'], flash['status'])} |"
        )

    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    data = load_results(args.input_path)
    markdown = render_markdown(data)

    if args.output_path is not None:
        args.output_path.parent.mkdir(parents=True, exist_ok=True)
        args.output_path.write_text(markdown, encoding="utf-8")

    print(markdown, end="")


if __name__ == "__main__":
    main()
