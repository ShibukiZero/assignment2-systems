from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path


MODEL_ORDER = ["small", "medium", "large", "xl", "2.7b"]


@dataclass(frozen=True)
class BenchmarkEntry:
    model_size: str
    mode: str
    compiled: bool
    total_ms: float
    forward_ms: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize full-model vanilla vs compiled benchmark JSON files for "
            "Problem 1.2.2(b)."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path(".agents/logs/1_2_2b_model_compile"),
        help="Directory containing full-model benchmark JSON files.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Optional markdown output path.",
    )
    return parser.parse_args()


def load_entry(json_path: Path) -> BenchmarkEntry:
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    config = payload["config"]
    timings = payload["timings"]
    return BenchmarkEntry(
        model_size=str(config["model_size"]),
        mode=str(config["mode"]),
        compiled=bool(config.get("compile_model", False)),
        total_ms=float(timings["total_seconds"]["mean_seconds"]) * 1e3,
        forward_ms=float(timings["forward_seconds"]["mean_seconds"]) * 1e3,
    )


def format_ms(value: float) -> str:
    return f"{value:.3f}"


def format_speedup(vanilla_ms: float, compiled_ms: float) -> str:
    if compiled_ms == 0:
        return "n/a"
    return f"{vanilla_ms / compiled_ms:.2f}x"


def build_markdown(entries: list[BenchmarkEntry]) -> str:
    table: dict[tuple[str, str, bool], BenchmarkEntry] = {}
    for entry in entries:
        table[(entry.model_size, entry.mode, entry.compiled)] = entry

    lines = [
        "# 1.2.2(b) Full-Model torch.compile Comparison",
        "",
        "| Model size | Vanilla forward (ms) | Compiled forward (ms) | Forward speedup | Vanilla train step (ms) | Compiled train step (ms) | Train-step speedup |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for model_size in MODEL_ORDER:
        vanilla_forward = table.get((model_size, "forward", False))
        compiled_forward = table.get((model_size, "forward", True))
        vanilla_train = table.get((model_size, "train-step", False))
        compiled_train = table.get((model_size, "train-step", True))

        if None in (vanilla_forward, compiled_forward, vanilla_train, compiled_train):
            continue

        assert vanilla_forward is not None
        assert compiled_forward is not None
        assert vanilla_train is not None
        assert compiled_train is not None

        lines.append(
            f"| {model_size} | "
            f"{format_ms(vanilla_forward.forward_ms)} | {format_ms(compiled_forward.forward_ms)} | {format_speedup(vanilla_forward.forward_ms, compiled_forward.forward_ms)} | "
            f"{format_ms(vanilla_train.total_ms)} | {format_ms(compiled_train.total_ms)} | {format_speedup(vanilla_train.total_ms, compiled_train.total_ms)} |"
        )

    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    entries = [load_entry(path) for path in sorted(args.input_dir.glob("*.json"))]
    markdown = build_markdown(entries)

    if args.output_path is not None:
        args.output_path.parent.mkdir(parents=True, exist_ok=True)
        args.output_path.write_text(markdown + "\n", encoding="utf-8")

    print(markdown)


if __name__ == "__main__":
    main()
