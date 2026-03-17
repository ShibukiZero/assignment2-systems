#!/usr/bin/env bash

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

MODE="forward-backward"
OUTPUT_DIR=".agents/logs/1_1_5c_precision_sweep"
WARMUP_STEPS=5
MEASURE_STEPS=10
BATCH_SIZE=4
VOCAB_SIZE=10000
DEVICE="cuda"

PRECISIONS=(fp32 bf16)
SIZES=(small medium large xl 2.7b)
CONTEXT_LENGTHS=(128 256 512 1024)

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_1_1_5c_precision_sweep.sh [options]

Options:
  --mode <forward|forward-backward|train-step>
  --output-dir <path>
  --warmup-steps <int>
  --measure-steps <int>
  --batch-size <int>
  --vocab-size <int>
  --device <auto|cpu|cuda>
  --precisions "<space-separated precisions>"
  --sizes "<space-separated sizes>"
  --contexts "<space-separated context lengths>"

Examples:
  bash scripts/run_1_1_5c_precision_sweep.sh
  bash scripts/run_1_1_5c_precision_sweep.sh --precisions "fp32 bf16" --sizes "xl 2.7b" --contexts "128 512"
EOF
}

while (($# > 0)); do
  case "$1" in
    --mode)
      MODE="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --warmup-steps)
      WARMUP_STEPS="$2"
      shift 2
      ;;
    --measure-steps)
      MEASURE_STEPS="$2"
      shift 2
      ;;
    --batch-size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --vocab-size)
      VOCAB_SIZE="$2"
      shift 2
      ;;
    --device)
      DEVICE="$2"
      shift 2
      ;;
    --precisions)
      read -r -a PRECISIONS <<<"$2"
      shift 2
      ;;
    --sizes)
      read -r -a SIZES <<<"$2"
      shift 2
      ;;
    --contexts)
      read -r -a CONTEXT_LENGTHS <<<"$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

case "${MODE}" in
  forward|forward-backward|train-step)
    ;;
  *)
    echo "Unsupported mode: ${MODE}" >&2
    exit 1
    ;;
esac

mkdir -p "${OUTPUT_DIR}"

SUMMARY_FILE="${OUTPUT_DIR}/run_summary.tsv"
printf "tag\tprecision\tmodel_size\tcontext_length\tmode\tstatus\tjson_path\tlog_path\n" > "${SUMMARY_FILE}"

for precision in "${PRECISIONS[@]}"; do
  for size in "${SIZES[@]}"; do
    for context_length in "${CONTEXT_LENGTHS[@]}"; do
      tag="${precision}_${size}_ctx${context_length}_${MODE//-/_}"
      echo "Benchmarking ${tag}"

      log_path="${OUTPUT_DIR}/${tag}.log"
      json_path="${OUTPUT_DIR}/${tag}.json"

      benchmark_args=(
        --model-size "${size}"
        --context-length "${context_length}"
        --batch-size "${BATCH_SIZE}"
        --vocab-size "${VOCAB_SIZE}"
        --mode "${MODE}"
        --precision "${precision}"
        --warmup-steps "${WARMUP_STEPS}"
        --measure-steps "${MEASURE_STEPS}"
        --device "${DEVICE}"
        --output-path "${json_path}"
      )

      if uv run python -m cs336_systems.benchmark "${benchmark_args[@]}" >"${log_path}" 2>&1; then
        printf "%s\t%s\t%s\t%s\t%s\tsuccess\t%s\t%s\n" \
          "${tag}" "${precision}" "${size}" "${context_length}" "${MODE}" "${json_path}" "${log_path}" \
          >> "${SUMMARY_FILE}"
        continue
      fi

      status="failed"
      if grep -Eiq "out of memory|oom|cuda error: out of memory|cuda out of memory" "${log_path}"; then
        status="oom"
      fi

      echo "Skipping ${tag} after ${status}; see ${log_path}" >&2
      printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "${tag}" "${precision}" "${size}" "${context_length}" "${MODE}" "${status}" "${json_path}" "${log_path}" \
        >> "${SUMMARY_FILE}"
    done
  done
done
