#!/usr/bin/env bash

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

MODEL_SIZE="2.7b"
OUTPUT_DIR=".agents/logs/1_1_6_memory_sweep"
WARMUP_STEPS=5
MEASURE_STEPS=0
BATCH_SIZE=4
VOCAB_SIZE=10000
DEVICE="cuda"

PRECISIONS=(fp32 bf16)
MODES=(forward train-step)
CONTEXT_LENGTHS=(128 256 512)

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_1_1_6_memory_sweep.sh [options]

Options:
  --model-size <size>
  --output-dir <path>
  --warmup-steps <int>
  --measure-steps <int>
  --batch-size <int>
  --vocab-size <int>
  --device <auto|cpu|cuda>
  --precisions "<space-separated precisions>"
  --modes "<space-separated modes>"
  --contexts "<space-separated context lengths>"

Examples:
  bash scripts/run_1_1_6_memory_sweep.sh
  bash scripts/run_1_1_6_memory_sweep.sh --precisions "fp32" --modes "forward train-step"
EOF
}

while (($# > 0)); do
  case "$1" in
    --model-size)
      MODEL_SIZE="$2"
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
    --modes)
      read -r -a MODES <<<"$2"
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

mkdir -p "${OUTPUT_DIR}"

SUMMARY_FILE="${OUTPUT_DIR}/run_summary.tsv"
printf "tag\tprecision\tmode\tcontext_length\tstatus\tjson_path\tsnapshot_path\tlog_path\n" > "${SUMMARY_FILE}"

for precision in "${PRECISIONS[@]}"; do
  for mode in "${MODES[@]}"; do
    for context_length in "${CONTEXT_LENGTHS[@]}"; do
      tag="${precision}_${MODEL_SIZE}_ctx${context_length}_${mode//-/_}_memory"
      echo "Profiling memory for ${tag}"

      log_path="${OUTPUT_DIR}/${tag}.log"
      json_path="${OUTPUT_DIR}/${tag}.json"
      snapshot_path="${OUTPUT_DIR}/${tag}.pickle"

      benchmark_args=(
        --model-size "${MODEL_SIZE}"
        --context-length "${context_length}"
        --batch-size "${BATCH_SIZE}"
        --vocab-size "${VOCAB_SIZE}"
        --mode "${mode}"
        --precision "${precision}"
        --warmup-steps "${WARMUP_STEPS}"
        --measure-steps "${MEASURE_STEPS}"
        --device "${DEVICE}"
        --memory-profile
        --memory-snapshot-path "${snapshot_path}"
        --output-path "${json_path}"
      )

      if uv run python -m cs336_systems.benchmark "${benchmark_args[@]}" >"${log_path}" 2>&1; then
        printf "%s\t%s\t%s\t%s\tsuccess\t%s\t%s\t%s\n" \
          "${tag}" "${precision}" "${mode}" "${context_length}" "${json_path}" "${snapshot_path}" "${log_path}" \
          >> "${SUMMARY_FILE}"
        continue
      fi

      status="failed"
      if grep -Eiq "out of memory|oom|cuda error: out of memory|cuda out of memory" "${log_path}"; then
        status="oom"
      fi

      echo "Skipping ${tag} after ${status}; see ${log_path}" >&2
      printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "${tag}" "${precision}" "${mode}" "${context_length}" "${status}" "${json_path}" "${snapshot_path}" "${log_path}" \
        >> "${SUMMARY_FILE}"
    done
  done
done
