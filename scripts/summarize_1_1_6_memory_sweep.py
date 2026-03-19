from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path


CONTEXT_ORDER = [128, 256, 512]
MODE_ORDER = ["forward", "train-step"]
PRECISION_ORDER = ["fp32", "bf16"]


@dataclass(frozen=True)
class MemoryEntry:
    context_length: int
    mode: str
    precision: str
    max_allocated_bytes: int
    max_reserved_bytes: int
    active_peak_bytes: int
    requested_peak_bytes: int
    snapshot_path: str
    json_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize 1.1.6 memory sweep JSON results into Markdown tables."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path(".agents/logs/1_1_6_memory_sweep"),
        help="Directory containing memory sweep JSON files.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Optional path to write the Markdown summary.",
    )
    return parser.parse_args()


def parse_entry(json_path: Path) -> MemoryEntry:
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    config = payload["config"]
    memory_after = payload["memory_profile"]["memory_after_profile_bytes"]
    return MemoryEntry(
        context_length=int(config["context_length"]),
        mode=str(config["mode"]),
        precision=str(config["precision"]),
        max_allocated_bytes=int(memory_after["max_allocated_bytes"]),
        max_reserved_bytes=int(memory_after["max_reserved_bytes"]),
        active_peak_bytes=int(memory_after["active_peak_bytes"]),
        requested_peak_bytes=int(memory_after["requested_peak_bytes"]),
        snapshot_path=str(payload["memory_profile"]["snapshot_path"]),
        json_path=json_path,
    )


def load_entries(input_dir: Path) -> dict[tuple[int, str, str], MemoryEntry]:
    entries: dict[tuple[int, str, str], MemoryEntry] = {}
    for json_path in sorted(input_dir.glob("*.json")):
        entry = parse_entry(json_path)
        entries[(entry.context_length, entry.mode, entry.precision)] = entry
    return entries


def format_gib(value: int) -> str:
    return f"{value / 1024 / 1024 / 1024:.2f} GiB"


def format_ratio(baseline: int, compare: int) -> str:
    if compare == 0:
        return "n/a"
    return f"{baseline / compare:.2f}x"


def build_part_b_table(entries: dict[tuple[int, str, str], MemoryEntry]) -> str:
    lines = [
        "## 1.1.6(b) FP32 Peak Memory",
        "",
        "| Context length | Forward peak memory | Full training step peak memory |",
        "| --- | ---: | ---: |",
    ]
    for context_length in CONTEXT_ORDER:
        forward = entries[(context_length, "forward", "fp32")]
        train = entries[(context_length, "train-step", "fp32")]
        lines.append(
            f"| {context_length} | {format_gib(forward.max_allocated_bytes)} | {format_gib(train.max_allocated_bytes)} |"
        )
    return "\n".join(lines)


def build_part_c_table(entries: dict[tuple[int, str, str], MemoryEntry]) -> str:
    lines = [
        "## 1.1.6(c) FP32 vs BF16 Peak Memory",
        "",
        "| Context length | Mode | FP32 peak memory | BF16 peak memory | FP32/BF16 ratio |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for context_length in CONTEXT_ORDER:
        for mode in MODE_ORDER:
            fp32_entry = entries[(context_length, mode, "fp32")]
            bf16_entry = entries[(context_length, mode, "bf16")]
            lines.append(
                f"| {context_length} | {mode} | {format_gib(fp32_entry.max_allocated_bytes)} | {format_gib(bf16_entry.max_allocated_bytes)} | {format_ratio(fp32_entry.max_allocated_bytes, bf16_entry.max_allocated_bytes)} |"
            )
    return "\n".join(lines)


def build_takeaways(entries: dict[tuple[int, str, str], MemoryEntry]) -> list[str]:
    forward_128 = entries[(128, "forward", "fp32")]
    forward_512 = entries[(512, "forward", "fp32")]
    train_128 = entries[(128, "train-step", "fp32")]
    train_512 = entries[(512, "train-step", "fp32")]
    bf16_forward_512 = entries[(512, "forward", "bf16")]
    bf16_train_512 = entries[(512, "train-step", "bf16")]

    return [
        f"- In FP32, forward-only peak memory grows moderately from {format_gib(forward_128.max_allocated_bytes)} at context length 128 to {format_gib(forward_512.max_allocated_bytes)} at context length 512.",
        f"- In FP32, full training-step peak memory is much higher and rises more sharply, from {format_gib(train_128.max_allocated_bytes)} at context length 128 to {format_gib(train_512.max_allocated_bytes)} at context length 512.",
        f"- In this benchmark setup, BF16 reduces train-step peak memory at context length 512 ({format_gib(train_512.max_allocated_bytes)} to {format_gib(bf16_train_512.max_allocated_bytes)}), but forward-only peak memory is actually higher for BF16 at the same context length ({format_gib(forward_512.max_allocated_bytes)} vs {format_gib(bf16_forward_512.max_allocated_bytes)}).",
    ]


def build_markdown(entries: dict[tuple[int, str, str], MemoryEntry]) -> str:
    sections = [
        "# 1.1.6 Memory Sweep Summary",
        "",
        "Configuration:",
        "",
        "- model size: `2.7b`",
        "- context lengths: `128`, `256`, `512`",
        "- modes: `forward`, `train-step`",
        "- precisions: `fp32`, `bf16`",
        "",
        "Key takeaways:",
        "",
        *build_takeaways(entries),
        "",
        build_part_b_table(entries),
        "",
        build_part_c_table(entries),
    ]
    return "\n".join(sections).rstrip() + "\n"


def main() -> None:
    args = parse_args()
    entries = load_entries(args.input_dir)
    markdown = build_markdown(entries)

    if args.output_path is not None:
        args.output_path.parent.mkdir(parents=True, exist_ok=True)
        args.output_path.write_text(markdown, encoding="utf-8")

    print(markdown, end="")


if __name__ == "__main__":
    main()
