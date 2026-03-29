from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TimingEntry:
    embedding_dim: int
    sequence_length: int
    implementation: str
    forward_ms: float
    backward_ms: float
    total_ms: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize 1.2.2 eager vs compiled attention benchmark JSON files into a "
            "writeup-ready markdown table."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path(".agents/logs/1_2_2_torch_compile"),
        help="Directory containing eager/compiled attention benchmark JSON files.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Optional markdown output path.",
    )
    return parser.parse_args()


def load_entry(json_path: Path) -> TimingEntry:
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    config = payload["config"]
    timings = payload["timings"]
    return TimingEntry(
        embedding_dim=int(config["embedding_dim"]),
        sequence_length=int(config["sequence_length"]),
        implementation=str(config["implementation"]),
        forward_ms=float(timings["forward_seconds"]["mean"]) * 1e3,
        backward_ms=float(timings["backward_seconds"]["mean"]) * 1e3,
        total_ms=float(timings["total_seconds"]["mean"]) * 1e3,
    )


def format_ms(value: float) -> str:
    return f"{value:.3f}"


def format_speedup(eager_ms: float, compiled_ms: float) -> str:
    if compiled_ms == 0:
        return "n/a"
    return f"{eager_ms / compiled_ms:.2f}x"


def build_markdown(entries: list[TimingEntry]) -> str:
    paired: dict[tuple[int, int], dict[str, TimingEntry]] = {}
    for entry in entries:
        paired.setdefault((entry.embedding_dim, entry.sequence_length), {})[entry.implementation] = entry

    header = "\n".join(
        [
            "# 1.2.2 torch.compile Attention Comparison",
            "",
            "| d_model | Sequence length | Eager forward (ms) | Compiled forward (ms) | Forward speedup | Eager backward (ms) | Compiled backward (ms) | Backward speedup |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )

    rows: list[str] = []
    for key in sorted(paired):
        eager = paired[key].get("eager")
        compiled = paired[key].get("compiled")
        if eager is None or compiled is None:
            continue

        rows.append(
            f"| {eager.embedding_dim} | {eager.sequence_length} | "
            f"{format_ms(eager.forward_ms)} | {format_ms(compiled.forward_ms)} | {format_speedup(eager.forward_ms, compiled.forward_ms)} | "
            f"{format_ms(eager.backward_ms)} | {format_ms(compiled.backward_ms)} | {format_speedup(eager.backward_ms, compiled.backward_ms)} |"
        )

    return header + ("\n" + "\n".join(rows) if rows else "")


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
