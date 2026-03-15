from __future__ import annotations

import argparse
import json
from pathlib import Path


MODEL_ORDER = {
    "small": 0,
    "medium": 1,
    "large": 2,
    "xl": 3,
    "2.7b": 4,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert benchmark JSON logs into a markdown table."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing benchmark JSON logs.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Optional path to write the markdown output.",
    )
    return parser.parse_args()


def format_ms(seconds: float) -> str:
    return f"{seconds * 1000:.3f}"


def load_benchmark_rows(input_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(input_dir.glob("*.json"), key=lambda p: MODEL_ORDER.get(p.stem, 10_000)):
        data = json.loads(path.read_text(encoding="utf-8"))
        timings = data["timings"]
        rows.append(
            {
                "model_size": data["config"]["model_size"],
                "context_length": data["config"]["context_length"],
                "batch_size": data["config"]["batch_size"],
                "precision": data["config"]["precision"],
                "measure_steps": data["config"]["measure_steps"],
                "warmup_steps": data["config"]["warmup_steps"],
                "forward_mean_ms": format_ms(timings["forward_seconds"]["mean_seconds"]),
                "forward_std_ms": format_ms(timings["forward_seconds"]["stdev_seconds"]),
                "backward_mean_ms": format_ms(timings["backward_seconds"]["mean_seconds"]),
                "backward_std_ms": format_ms(timings["backward_seconds"]["stdev_seconds"]),
                "total_mean_ms": format_ms(timings["total_seconds"]["mean_seconds"]),
                "total_std_ms": format_ms(timings["total_seconds"]["stdev_seconds"]),
            }
        )
    return rows


def render_markdown(rows: list[dict]) -> str:
    if not rows:
        raise ValueError("No benchmark JSON files were found in the input directory.")

    first = rows[0]
    lines = [
        f"Context length: {first['context_length']}",
        f"Batch size: {first['batch_size']}",
        f"Precision: {first['precision']}",
        f"Warmup steps: {first['warmup_steps']}",
        f"Measurement steps: {first['measure_steps']}",
        "",
        "| Model size | Forward mean (ms) | Forward std (ms) | Backward mean (ms) | Backward std (ms) | Total mean (ms) | Total std (ms) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for row in rows:
        lines.append(
            "| "
            f"{row['model_size']} | "
            f"{row['forward_mean_ms']} | "
            f"{row['forward_std_ms']} | "
            f"{row['backward_mean_ms']} | "
            f"{row['backward_std_ms']} | "
            f"{row['total_mean_ms']} | "
            f"{row['total_std_ms']} |"
        )

    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    rows = load_benchmark_rows(args.input_dir)
    markdown = render_markdown(rows)

    if args.output_path is not None:
        args.output_path.parent.mkdir(parents=True, exist_ok=True)
        args.output_path.write_text(markdown, encoding="utf-8")

    print(markdown, end="")


if __name__ == "__main__":
    main()
