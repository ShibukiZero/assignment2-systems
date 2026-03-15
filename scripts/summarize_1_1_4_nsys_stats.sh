#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

INPUT_DIR="${1:-.agents/logs/1_1_4_forward_attention}"
OUTPUT_DIR="${2:-${INPUT_DIR}/stats}"

mkdir -p "${OUTPUT_DIR}"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/summarize_1_1_4_nsys_stats.sh [input_dir] [output_dir]

Examples:
  bash scripts/summarize_1_1_4_nsys_stats.sh .agents/logs/1_1_4_forward_attention
  bash scripts/summarize_1_1_4_nsys_stats.sh .agents/logs/1_1_4_train_step_attention .agents/logs/1_1_4_train_step_attention/stats

This script exports a small set of nsys stats reports for every .nsys-rep file:
  - nvtx_gpu_proj_sum: GPU-projected duration by NVTX range
  - nvtx_gpu_proj_trace: per-instance GPU-projected NVTX trace
  - nvtx_kern_sum: kernels grouped by containing NVTX range
  - cuda_gpu_kern_sum: global CUDA kernel summary
  - cuda_gpu_kern_sum:nvtx-name:base: kernel summary tagged with innermost NVTX range
EOF
}

if [[ "${INPUT_DIR}" == "--help" || "${INPUT_DIR}" == "-h" ]]; then
  usage
  exit 0
fi

if ! command -v nsys >/dev/null 2>&1; then
  echo "nsys is not installed on this machine." >&2
  exit 1
fi

summary_file="${OUTPUT_DIR}/report_inventory.tsv"
printf "rep_path\treport\toutput_csv\n" > "${summary_file}"

reports=(
  "nvtx_gpu_proj_sum"
  "nvtx_gpu_proj_trace"
  "nvtx_kern_sum:base"
  "cuda_gpu_kern_sum:base"
  "cuda_gpu_kern_sum:nvtx-name:base"
)

found_any=0

for rep_path in "${INPUT_DIR}"/*.nsys-rep; do
  if [[ ! -e "${rep_path}" ]]; then
    continue
  fi

  found_any=1
  base_name="$(basename "${rep_path}" .nsys-rep)"
  rep_output_dir="${OUTPUT_DIR}/${base_name}"
  mkdir -p "${rep_output_dir}"

  echo "Exporting stats for ${base_name}"

  for report in "${reports[@]}"; do
    safe_report_name="${report//:/__}"
    output_prefix="${rep_output_dir}/${safe_report_name}"

    nsys stats \
      --report "${report}" \
      --format csv \
      --output "${output_prefix}" \
      "${rep_path}"

    printf "%s\t%s\t%s.csv\n" "${rep_path}" "${report}" "${output_prefix}" >> "${summary_file}"
  done
done

if [[ "${found_any}" -eq 0 ]]; then
  echo "No .nsys-rep files found under ${INPUT_DIR}" >&2
  exit 1
fi
