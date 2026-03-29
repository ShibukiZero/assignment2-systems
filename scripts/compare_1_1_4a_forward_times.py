from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path


MODEL_ORDER = ["small", "medium", "large", "xl", "2.7b"]


@dataclass(frozen=True)
class BenchmarkEntry:
    model_size: str
    context_length: int
    benchmark_forward_seconds: float
    json_path: Path


@dataclass(frozen=True)
class NsysEntry:
    model_size: str
    context_length: int
    nsys_forward_seconds: float
    instances: int | None
    csv_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare Python benchmark forward timing against nsys stats NVTX forward timing "
            "for the same model/context configuration."
        )
    )
    parser.add_argument(
        "--benchmark-dir",
        type=Path,
        default=Path(".agents/logs/1_1_4_forward_attention"),
        help="Directory containing benchmark JSON files from the forward sweep.",
    )
    parser.add_argument(
        "--stats-dir",
        type=Path,
        default=Path(".agents/logs/1_1_4_forward_attention/stats"),
        help="Directory containing exported nsys stats CSV folders.",
    )
    parser.add_argument(
        "--context-length",
        type=int,
        default=128,
        help="Filter to one context length. 1.1.4(a) is most directly comparable at 128.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Optional path to write the markdown comparison table.",
    )
    return parser.parse_args()


def parse_benchmark_entries(benchmark_dir: Path, context_length: int) -> dict[str, BenchmarkEntry]:
    entries: dict[str, BenchmarkEntry] = {}
    for json_path in sorted(benchmark_dir.glob("*_forward_attention.json")):
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        config = payload["config"]
        if int(config["context_length"]) != context_length:
            continue
        model_size = str(config["model_size"])
        benchmark_forward_seconds = float(payload["timings"]["forward_seconds"]["mean_seconds"])
        entries[model_size] = BenchmarkEntry(
            model_size=model_size,
            context_length=context_length,
            benchmark_forward_seconds=benchmark_forward_seconds,
            json_path=json_path,
        )
    return entries


def normalize_range_name(raw_name: str) -> str:
    return raw_name.strip().lstrip(":").strip()


def pick_column(fieldnames: list[str], patterns: list[str]) -> str | None:
    lowered = {fieldname.lower(): fieldname for fieldname in fieldnames}
    for pattern in patterns:
        for lowered_name, original_name in lowered.items():
            if pattern in lowered_name:
                return original_name
    return None


def detect_delimiter(line: str) -> str:
    if line.count("\t") >= max(line.count(","), line.count(";")):
        return "\t"
    if line.count(";") > line.count(","):
        return ";"
    return ","


def looks_like_header(line: str) -> bool:
    lowered = line.lower()
    return (
        ("range" in lowered or "name" in lowered)
        and ("avg" in lowered or "average" in lowered or "total" in lowered)
        and any(delimiter in line for delimiter in [",", ";", "\t"])
    )


def load_csv_rows(csv_path: Path) -> tuple[list[str], list[dict[str, str]]]:
    lines = csv_path.read_text(encoding="utf-8").splitlines()

    header_index: int | None = None
    for idx, line in enumerate(lines):
        if looks_like_header(line):
            header_index = idx
            break

    if header_index is None:
        raise ValueError(f"Could not find CSV header row in {csv_path}")

    delimiter = detect_delimiter(lines[header_index])
    reader = csv.DictReader(lines[header_index:], delimiter=delimiter)
    fieldnames = reader.fieldnames or []
    rows = list(reader)
    return fieldnames, rows


def infer_seconds(value: str, header: str) -> float:
    numeric = float(value)
    lowered = header.lower()
    if "(ns" in lowered or lowered.endswith(" ns") or "[ns]" in lowered:
        return numeric / 1e9
    if "(us" in lowered or lowered.endswith(" us") or "[us]" in lowered:
        return numeric / 1e6
    if "(ms" in lowered or lowered.endswith(" ms") or "[ms]" in lowered:
        return numeric / 1e3
    if "(s" in lowered or lowered.endswith(" s") or "[s]" in lowered:
        return numeric
    # nsys stats time columns are typically emitted in ns if unit text is absent.
    return numeric / 1e9


def parse_instances(value: str | None) -> int | None:
    if value is None or value == "":
        return None
    return int(float(value))


def parse_nsys_entry(csv_path: Path, context_length: int) -> NsysEntry:
    stem = csv_path.parent.name
    match = re.match(r"(?P<model>.+)_ctx(?P<context>\d+)_forward_attention", stem)
    if match is None:
        raise ValueError(f"Could not parse model/context from {stem}")

    model_size = match.group("model")
    parsed_context = int(match.group("context"))
    if parsed_context != context_length:
        raise ValueError(f"Unexpected context length in {csv_path}: {parsed_context}")

    fieldnames, rows = load_csv_rows(csv_path)
    name_column = pick_column(fieldnames, ["range name", "range", "name"])
    if name_column is None:
        raise ValueError(f"Could not find NVTX range name column in {csv_path}")

    avg_column = pick_column(fieldnames, ["avg", "average", "mean"])
    total_column = pick_column(fieldnames, ["total", "sum"])
    instances_column = pick_column(fieldnames, ["instances", "count", "calls"])

    measure_rows = [
        row
        for row in rows
        if re.fullmatch(r"measure_\d+", normalize_range_name(row[name_column]))
    ]
    if not measure_rows:
        raise ValueError(f"No measure_* NVTX rows found in {csv_path}")

    durations_seconds: list[float] = []
    for row in measure_rows:
        instances = parse_instances(row.get(instances_column) if instances_column else None)
        if avg_column and row.get(avg_column):
            durations_seconds.append(infer_seconds(row[avg_column], avg_column))
            continue
        if total_column and row.get(total_column):
            total_seconds = infer_seconds(row[total_column], total_column)
            divisor = instances if instances and instances > 0 else 1
            durations_seconds.append(total_seconds / divisor)
            continue
        raise ValueError(
            f"Could not infer measure duration from {csv_path}; "
            "expected avg column or total column."
        )

    nsys_forward_seconds = sum(durations_seconds) / len(durations_seconds)
    return NsysEntry(
        model_size=model_size,
        context_length=parsed_context,
        nsys_forward_seconds=nsys_forward_seconds,
        instances=len(measure_rows),
        csv_path=csv_path,
    )


def parse_nsys_entries(stats_dir: Path, context_length: int) -> dict[str, NsysEntry]:
    entries: dict[str, NsysEntry] = {}
    for csv_path in sorted(stats_dir.glob("*/nvtx_gpu_proj_sum*.csv")):
        stem = csv_path.parent.name
        if f"_ctx{context_length}_" not in stem:
            continue
        entry = parse_nsys_entry(csv_path, context_length)
        entries[entry.model_size] = entry
    return entries


def format_markdown_table(rows: list[dict[str, str]]) -> str:
    header = (
        "| Model size | Context length | Benchmark forward (ms) | "
        "nsys NVTX forward (ms) | Abs diff (ms) | Rel diff (%) |\n"
        "| --- | ---: | ---: | ---: | ---: | ---: |"
    )
    body = "\n".join(
        (
            f"| {row['model_size']} | {row['context_length']} | {row['benchmark_ms']} | "
            f"{row['nsys_ms']} | {row['abs_diff_ms']} | {row['rel_diff_pct']} |"
        )
        for row in rows
    )
    return header + ("\n" + body if body else "")


def main() -> None:
    args = parse_args()

    benchmark_entries = parse_benchmark_entries(args.benchmark_dir, args.context_length)
    nsys_entries = parse_nsys_entries(args.stats_dir, args.context_length)

    rows: list[dict[str, str]] = []
    missing_models: list[str] = []

    for model_size in MODEL_ORDER:
        benchmark_entry = benchmark_entries.get(model_size)
        nsys_entry = nsys_entries.get(model_size)
        if benchmark_entry is None or nsys_entry is None:
            missing_models.append(model_size)
            continue

        benchmark_ms = benchmark_entry.benchmark_forward_seconds * 1e3
        nsys_ms = nsys_entry.nsys_forward_seconds * 1e3
        abs_diff_ms = abs(nsys_ms - benchmark_ms)
        rel_diff_pct = abs_diff_ms / benchmark_ms * 100 if benchmark_ms else 0.0

        rows.append(
            {
                "model_size": model_size,
                "context_length": str(args.context_length),
                "benchmark_ms": f"{benchmark_ms:.3f}",
                "nsys_ms": f"{nsys_ms:.3f}",
                "abs_diff_ms": f"{abs_diff_ms:.3f}",
                "rel_diff_pct": f"{rel_diff_pct:.3f}",
            }
        )

    output = format_markdown_table(rows)
    if missing_models:
        output += (
            "\n\nMissing models: " + ", ".join(missing_models) +
            "\nCheck that benchmark JSON files and `nvtx_gpu_proj_sum` CSV files both exist for these models."
        )

    if args.output_path is not None:
        args.output_path.write_text(output + "\n", encoding="utf-8")

    print(output)


if __name__ == "__main__":
    main()
