#!/usr/bin/env bash

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

MODE="forward"
OUTPUT_DIR=""
WARMUP_STEPS=5
MEASURE_STEPS=10
ENABLE_ATTENTION_NVTX=0
BATCH_SIZE=4
VOCAB_SIZE=10000
DEVICE="cuda"

SIZES=(small medium large xl 2.7b)
CONTEXT_LENGTHS=(128 256 512 1024)

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_1_1_4_sweep.sh [options]

Options:
  --mode <forward|forward-backward|train-step>
  --output-dir <path>
  --warmup-steps <int>
  --measure-steps <int>
  --attention-nvtx
  --batch-size <int>
  --vocab-size <int>
  --device <auto|cpu|cuda>
  --sizes "<space-separated sizes>"
  --contexts "<space-separated context lengths>"

Examples:
  bash scripts/run_1_1_4_sweep.sh --mode forward
  bash scripts/run_1_1_4_sweep.sh --mode train-step
  bash scripts/run_1_1_4_sweep.sh --mode forward --attention-nvtx --sizes "xl 2.7b" --contexts "512 1024"
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
    --attention-nvtx)
      ENABLE_ATTENTION_NVTX=1
      shift
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

MODE_TAG="${MODE//-/_}"
if [[ -z "${OUTPUT_DIR}" ]]; then
  OUTPUT_DIR=".agents/logs/1_1_4_${MODE_TAG}"
  if [[ "${ENABLE_ATTENTION_NVTX}" -eq 1 ]]; then
    OUTPUT_DIR="${OUTPUT_DIR}_attention"
  fi
fi

mkdir -p "${OUTPUT_DIR}"

SUMMARY_FILE="${OUTPUT_DIR}/run_summary.tsv"
printf "tag\tstatus\trep_path\tjson_path\tlog_path\n" > "${SUMMARY_FILE}"

for size in "${SIZES[@]}"; do
  for context_length in "${CONTEXT_LENGTHS[@]}"; do
    tag="${size}_ctx${context_length}_${MODE_TAG}"
    if [[ "${ENABLE_ATTENTION_NVTX}" -eq 1 ]]; then
      tag="${tag}_attention"
    fi

    echo "Profiling ${tag}"

    log_path="${OUTPUT_DIR}/${tag}.log"
    rep_path="${OUTPUT_DIR}/${tag}.nsys-rep"
    json_path="${OUTPUT_DIR}/${tag}.json"

    benchmark_args=(
      --model-size "${size}"
      --context-length "${context_length}"
      --batch-size "${BATCH_SIZE}"
      --vocab-size "${VOCAB_SIZE}"
      --mode "${MODE}"
      --warmup-steps "${WARMUP_STEPS}"
      --measure-steps "${MEASURE_STEPS}"
      --device "${DEVICE}"
      --nvtx
      --output-path "${json_path}"
    )

    if [[ "${ENABLE_ATTENTION_NVTX}" -eq 1 ]]; then
      benchmark_args+=(--attention-nvtx)
    fi

    if nsys profile \
      --force-overwrite=true \
      -t cuda,nvtx,osrt \
      -o "${OUTPUT_DIR}/${tag}" \
      uv run python -m cs336_systems.benchmark "${benchmark_args[@]}" \
      >"${log_path}" 2>&1; then
      printf "%s\tsuccess\t%s\t%s\t%s\n" "${tag}" "${rep_path}" "${json_path}" "${log_path}" >> "${SUMMARY_FILE}"
      continue
    fi

    status="failed"
    if grep -Eiq "out of memory|oom|cuda error: out of memory|cuda out of memory" "${log_path}"; then
      status="oom"
    fi

    echo "Skipping ${tag} after ${status}; see ${log_path}" >&2
    printf "%s\t%s\t%s\t%s\t%s\n" "${tag}" "${status}" "${rep_path}" "${json_path}" "${log_path}" >> "${SUMMARY_FILE}"
  done
done
