from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path


MODEL_ORDER = ["small", "medium", "large", "xl", "2.7b"]


@dataclass(frozen=True)
class TimingSummary:
    model_size: str
    context_length: int
    precision: str
    forward_ms: float
    backward_ms: float
    total_ms: float
    json_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize 1.1.5(c) FP32 vs BF16 benchmark JSON results into Markdown tables."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path(".agents/logs/1_1_5c_precision_sweep"),
        help="Directory containing fp32/bf16 benchmark JSON files.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Optional path to write the Markdown summary.",
    )
    return parser.parse_args()


def parse_entry(json_path: Path) -> TimingSummary:
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    config = payload["config"]
    timings = payload["timings"]
    return TimingSummary(
        model_size=str(config["model_size"]),
        context_length=int(config["context_length"]),
        precision=str(config["precision"]),
        forward_ms=float(timings["forward_seconds"]["mean_seconds"]) * 1e3,
        backward_ms=float(timings["backward_seconds"]["mean_seconds"]) * 1e3,
        total_ms=float(timings["total_seconds"]["mean_seconds"]) * 1e3,
        json_path=json_path,
    )


def load_entries(input_dir: Path) -> dict[tuple[int, str, str], TimingSummary]:
    entries: dict[tuple[int, str, str], TimingSummary] = {}
    for json_path in sorted(input_dir.glob("*.json")):
        entry = parse_entry(json_path)
        key = (entry.context_length, entry.model_size, entry.precision)
        entries[key] = entry
    return entries


def format_ms(value: float) -> str:
    return f"{value:.3f}"


def format_speedup(fp32_ms: float, bf16_ms: float) -> str:
    if bf16_ms == 0:
        return "n/a"
    return f"{fp32_ms / bf16_ms:.2f}x"


def build_context_table(
    context_length: int,
    entries: dict[tuple[int, str, str], TimingSummary],
) -> str:
    header = "\n".join(
        [
            f"### Context length = {context_length}",
            "",
            "| Model size | FP32 forward (ms) | BF16 forward (ms) | Forward speedup | FP32 backward (ms) | BF16 backward (ms) | Backward speedup | FP32 total (ms) | BF16 total (ms) | Total speedup |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )

    rows: list[str] = []
    for model_size in MODEL_ORDER:
        fp32_entry = entries.get((context_length, model_size, "fp32"))
        bf16_entry = entries.get((context_length, model_size, "bf16"))
        if fp32_entry is None or bf16_entry is None:
            continue

        rows.append(
            "| "
            + " | ".join(
                [
                    model_size,
                    format_ms(fp32_entry.forward_ms),
                    format_ms(bf16_entry.forward_ms),
                    format_speedup(fp32_entry.forward_ms, bf16_entry.forward_ms),
                    format_ms(fp32_entry.backward_ms),
                    format_ms(bf16_entry.backward_ms),
                    format_speedup(fp32_entry.backward_ms, bf16_entry.backward_ms),
                    format_ms(fp32_entry.total_ms),
                    format_ms(bf16_entry.total_ms),
                    format_speedup(fp32_entry.total_ms, bf16_entry.total_ms),
                ]
            )
            + " |"
        )

    return header + ("\n" + "\n".join(rows) if rows else "")


def build_takeaways(entries: dict[tuple[int, str, str], TimingSummary]) -> list[str]:
    bullets: list[str] = []

    small_128_fp32 = entries.get((128, "small", "fp32"))
    small_128_bf16 = entries.get((128, "small", "bf16"))
    large_128_fp32 = entries.get((128, "2.7b", "fp32"))
    large_128_bf16 = entries.get((128, "2.7b", "bf16"))
    small_1024_fp32 = entries.get((1024, "small", "fp32"))
    small_1024_bf16 = entries.get((1024, "small", "bf16"))
    large_1024_fp32 = entries.get((1024, "2.7b", "fp32"))
    large_1024_bf16 = entries.get((1024, "2.7b", "bf16"))

    if small_128_fp32 and small_128_bf16:
        bullets.append(
            f"- At context length `128`, BF16 provides little benefit for the smallest model: `small` changes from `{small_128_fp32.total_ms:.2f} ms` to `{small_128_bf16.total_ms:.2f} ms` total (`{small_128_fp32.total_ms / small_128_bf16.total_ms:.2f}x`)."
        )
    if large_128_fp32 and large_128_bf16:
        bullets.append(
            f"- At the same context length, the benefit is already large for `2.7b`: total time drops from `{large_128_fp32.total_ms:.2f} ms` to `{large_128_bf16.total_ms:.2f} ms` (`{large_128_fp32.total_ms / large_128_bf16.total_ms:.2f}x`)."
        )
    if small_1024_fp32 and small_1024_bf16 and large_1024_fp32 and large_1024_bf16:
        bullets.append(
            f"- At context length `1024`, BF16 helps across the board, but the speedup still grows with scale: `small` improves from `{small_1024_fp32.total_ms:.2f} ms` to `{small_1024_bf16.total_ms:.2f} ms` (`{small_1024_fp32.total_ms / small_1024_bf16.total_ms:.2f}x`), whereas `2.7b` improves from `{large_1024_fp32.total_ms:.2f} ms` to `{large_1024_bf16.total_ms:.2f} ms` (`{large_1024_fp32.total_ms / large_1024_bf16.total_ms:.2f}x`)."
        )

    return bullets


def build_markdown(entries: dict[tuple[int, str, str], TimingSummary]) -> str:
    context_lengths = sorted({context_length for context_length, _, _ in entries})
    sections: list[str] = [
        "## 1.1.5(c) BF16 Precision Sweep",
        "",
        "Configuration:",
        "",
        "- mode: `forward-backward`",
        "- warmup steps: `5`",
        "- measurement steps: `10`",
        "- batch size: `4`",
        "- vocabulary size: `10,000`",
        "",
        "Key takeaways:",
        "",
    ]
    sections.extend(build_takeaways(entries))

    for context_length in context_lengths:
        sections.extend(["", build_context_table(context_length, entries)])

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
