from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class KernelRow:
    time_pct: float
    name: str


TABLE_ROW_RE = re.compile(
    r"^\s*"
    r"(?P<time_pct>\d+(?:\.\d+)?)"
    r"\s+"
    r"(?P<total_time>\d+)"
    r"\s+"
    r"(?P<instances>\d+)"
    r"\s+"
    r"(?P<avg>\S+)"
    r"\s+"
    r"(?P<med>\S+)"
    r"\s+"
    r"(?P<min>\S+)"
    r"\s+"
    r"(?P<max>\S+)"
    r"\s+"
    r"(?P<stddev>\S+)"
    r"\s+"
    r"(?P<name>.+?)"
    r"\s*$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize GEMM-vs-non-GEMM time fractions from two saved "
            "CUDA GPU Kernel Summary text files."
        )
    )
    parser.add_argument(
        "--forward-summary",
        type=Path,
        default=Path("artifacts/experiments/ch1/1_1_4b/2.7b_ctx512_forward_cuda_gpu_kern_sum.txt"),
        help="Saved `nsys stats --report cuda_gpu_kern_sum` text for the forward-only trace.",
    )
    parser.add_argument(
        "--train-step-summary",
        type=Path,
        default=Path("artifacts/experiments/ch1/1_1_4b/2.7b_ctx512_train_step_cuda_gpu_kern_sum.txt"),
        help="Saved `nsys stats --report cuda_gpu_kern_sum` text for the full train-step trace.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Optional path to write the markdown summary.",
    )
    return parser.parse_args()


def parse_kernel_summary(summary_path: Path) -> list[KernelRow]:
    rows: list[KernelRow] = []
    for line in summary_path.read_text(encoding="utf-8").splitlines():
        match = TABLE_ROW_RE.match(line)
        if match is None:
            continue
        rows.append(
            KernelRow(
                time_pct=float(match.group("time_pct")),
                name=match.group("name").strip(),
            )
        )
    if not rows:
        raise ValueError(f"No CUDA GPU Kernel Summary rows found in {summary_path}")
    return rows


def is_gemm_kernel(kernel_name: str) -> bool:
    lowered = kernel_name.lower()
    return any(token in lowered for token in ("gemm", "xmma", "cutlass"))


def summarize_fraction(rows: list[KernelRow]) -> tuple[float, float]:
    gemm_pct = sum(row.time_pct for row in rows if is_gemm_kernel(row.name))
    non_gemm_pct = sum(row.time_pct for row in rows if not is_gemm_kernel(row.name))
    return gemm_pct, non_gemm_pct


def build_markdown(
    forward_summary_path: Path,
    train_step_summary_path: Path,
    forward_gemm_pct: float,
    forward_non_gemm_pct: float,
    train_gemm_pct: float,
    train_non_gemm_pct: float,
) -> str:
    return "\n".join(
        [
            "## 1.1.4(d) GEMM Fraction Summary",
            "",
            "| Trace | GEMM share (%) | Non-GEMM share (%) |",
            "| --- | ---: | ---: |",
            f"| Forward-only | {forward_gemm_pct:.2f} | {forward_non_gemm_pct:.2f} |",
            f"| Train-step | {train_gemm_pct:.2f} | {train_non_gemm_pct:.2f} |",
            "",
            "Classification rule:",
            "",
            "- A kernel is counted as GEMM if its `Name` contains `gemm`, `xmma`, or `cutlass`.",
            "",
            "Source files:",
            "",
            f"- Forward summary: `{forward_summary_path}`",
            f"- Train-step summary: `{train_step_summary_path}`",
        ]
    )


def main() -> None:
    args = parse_args()

    forward_rows = parse_kernel_summary(args.forward_summary)
    train_rows = parse_kernel_summary(args.train_step_summary)

    forward_gemm_pct, forward_non_gemm_pct = summarize_fraction(forward_rows)
    train_gemm_pct, train_non_gemm_pct = summarize_fraction(train_rows)

    markdown = build_markdown(
        args.forward_summary,
        args.train_step_summary,
        forward_gemm_pct,
        forward_non_gemm_pct,
        train_gemm_pct,
        train_non_gemm_pct,
    )

    if args.output_path is not None:
        args.output_path.write_text(markdown + "\n", encoding="utf-8")

    print(markdown)


if __name__ == "__main__":
    main()
