#!/usr/bin/env bash

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

OUTPUT_DIR=".agents/logs/1_2_2b_model_compile"
WARMUP_STEPS=5
MEASURE_STEPS=10
BATCH_SIZE=4
VOCAB_SIZE=10000
CONTEXT_LENGTH=128
DEVICE="cuda"
PRECISION="fp32"

MODEL_SIZES=(small medium large xl 2.7b)
MODES=(forward train-step)
MODEL_IMPLS=(vanilla compiled)

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_1_2_2b_model_compile_sweep.sh [options]

Options:
  --output-dir <path>
  --warmup-steps <int>
  --measure-steps <int>
  --batch-size <int>
  --vocab-size <int>
  --context-length <int>
  --device <auto|cpu|cuda>
  --precision <fp32|bf16>
  --sizes "<space-separated sizes>"
  --modes "<space-separated modes>"
  --model-impls "<space-separated model implementations>"

Examples:
  bash scripts/run_1_2_2b_model_compile_sweep.sh
  bash scripts/run_1_2_2b_model_compile_sweep.sh --sizes "xl 2.7b" --modes "forward train-step"
EOF
}

while (($# > 0)); do
  case "$1" in
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
    --context-length)
      CONTEXT_LENGTH="$2"
      shift 2
      ;;
    --device)
      DEVICE="$2"
      shift 2
      ;;
    --precision)
      PRECISION="$2"
      shift 2
      ;;
    --sizes)
      read -r -a MODEL_SIZES <<<"$2"
      shift 2
      ;;
    --modes)
      read -r -a MODES <<<"$2"
      shift 2
      ;;
    --model-impls)
      read -r -a MODEL_IMPLS <<<"$2"
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

case "${PRECISION}" in
  fp32|bf16)
    ;;
  *)
    echo "Unsupported precision: ${PRECISION}" >&2
    exit 1
    ;;
esac

mkdir -p "${OUTPUT_DIR}"

SUMMARY_FILE="${OUTPUT_DIR}/run_summary.tsv"
printf "tag\tmodel_impl\tprecision\tmodel_size\tcontext_length\tbatch_size\tmode\tstatus\tjson_path\tlog_path\n" > "${SUMMARY_FILE}"

for model_impl in "${MODEL_IMPLS[@]}"; do
  for model_size in "${MODEL_SIZES[@]}"; do
    for mode in "${MODES[@]}"; do
      tag="${model_impl}_${PRECISION}_${model_size}_ctx${CONTEXT_LENGTH}_${mode//-/_}"
      echo "Benchmarking ${tag}"

      log_path="${OUTPUT_DIR}/${tag}.log"
      json_path="${OUTPUT_DIR}/${tag}.json"

      benchmark_args=(
        --model-size "${model_size}"
        --context-length "${CONTEXT_LENGTH}"
        --batch-size "${BATCH_SIZE}"
        --vocab-size "${VOCAB_SIZE}"
        --mode "${mode}"
        --precision "${PRECISION}"
        --warmup-steps "${WARMUP_STEPS}"
        --measure-steps "${MEASURE_STEPS}"
        --device "${DEVICE}"
        --output-path "${json_path}"
      )

      if [[ "${model_impl}" == "compiled" ]]; then
        benchmark_args+=(--compile-model)
      fi

      if uv run python -m cs336_systems.benchmark "${benchmark_args[@]}" >"${log_path}" 2>&1; then
        printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\tsuccess\t%s\t%s\n" \
          "${tag}" "${model_impl}" "${PRECISION}" "${model_size}" "${CONTEXT_LENGTH}" "${BATCH_SIZE}" "${mode}" "${json_path}" "${log_path}" \
          >> "${SUMMARY_FILE}"
        continue
      fi

      status="failed"
      if grep -Eiq "out of memory|oom|cuda error: out of memory|cuda out of memory" "${log_path}"; then
        status="oom"
      fi

      echo "Skipping ${tag} after ${status}; see ${log_path}" >&2
      printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "${tag}" "${model_impl}" "${PRECISION}" "${model_size}" "${CONTEXT_LENGTH}" "${BATCH_SIZE}" "${mode}" "${status}" "${json_path}" "${log_path}" \
        >> "${SUMMARY_FILE}"
    done
  done
done
