#!/usr/bin/env bash

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

OUTPUT_DIR=".agents/logs/1_2_1_attention_sweep"
WARMUP_STEPS=5
MEASURE_STEPS=100
BATCH_SIZE=8
DEVICE="cuda"
PRECISION="fp32"
CAUSAL=0

IMPLEMENTATIONS=(eager)
EMBEDDING_DIMS=(16 32 64 128)
SEQUENCE_LENGTHS=(256 1024 4096 8192 16384)

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_1_2_1_attention_sweep.sh [options]

Options:
  --output-dir <path>
  --warmup-steps <int>
  --measure-steps <int>
  --batch-size <int>
  --device <auto|cpu|cuda>
  --precision <fp32|bf16>
  --causal
  --implementations "<space-separated implementations>"
  --dims "<space-separated embedding dimensions>"
  --sequence-lengths "<space-separated sequence lengths>"

Examples:
  bash scripts/run_1_2_1_attention_sweep.sh
  bash scripts/run_1_2_1_attention_sweep.sh --dims "64 128" --sequence-lengths "4096 8192 16384"
  bash scripts/run_1_2_1_attention_sweep.sh --implementations "eager compiled" --causal
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
    --device)
      DEVICE="$2"
      shift 2
      ;;
    --precision)
      PRECISION="$2"
      shift 2
      ;;
    --causal)
      CAUSAL=1
      shift 1
      ;;
    --implementations)
      read -r -a IMPLEMENTATIONS <<<"$2"
      shift 2
      ;;
    --dims)
      read -r -a EMBEDDING_DIMS <<<"$2"
      shift 2
      ;;
    --sequence-lengths)
      read -r -a SEQUENCE_LENGTHS <<<"$2"
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
printf "tag\timplementation\tprecision\tbatch_size\tembedding_dim\tsequence_length\tcausal\tstatus\tjson_path\tlog_path\n" > "${SUMMARY_FILE}"

for implementation in "${IMPLEMENTATIONS[@]}"; do
  for embedding_dim in "${EMBEDDING_DIMS[@]}"; do
    for sequence_length in "${SEQUENCE_LENGTHS[@]}"; do
      tag="${implementation}_${PRECISION}_b${BATCH_SIZE}_d${embedding_dim}_t${sequence_length}"
      if [[ "${CAUSAL}" == "1" ]]; then
        tag="${tag}_causal"
      fi

      echo "Benchmarking ${tag}"

      log_path="${OUTPUT_DIR}/${tag}.log"
      json_path="${OUTPUT_DIR}/${tag}.json"

      benchmark_args=(
        --batch-size "${BATCH_SIZE}"
        --sequence-length "${sequence_length}"
        --embedding-dim "${embedding_dim}"
        --implementation "${implementation}"
        --precision "${PRECISION}"
        --warmup-steps "${WARMUP_STEPS}"
        --measure-steps "${MEASURE_STEPS}"
        --device "${DEVICE}"
        --output-path "${json_path}"
      )

      if [[ "${CAUSAL}" == "1" ]]; then
        benchmark_args+=(--causal)
      fi

      if uv run python -m cs336_systems.attention_benchmark "${benchmark_args[@]}" >"${log_path}" 2>&1; then
        printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\tsuccess\t%s\t%s\n" \
          "${tag}" "${implementation}" "${PRECISION}" "${BATCH_SIZE}" "${embedding_dim}" "${sequence_length}" "${CAUSAL}" "${json_path}" "${log_path}" \
          >> "${SUMMARY_FILE}"
        continue
      fi

      status="failed"
      if grep -Eiq "out of memory|oom|cuda error: out of memory|cuda out of memory" "${log_path}"; then
        status="oom"
      fi

      echo "Skipping ${tag} after ${status}; see ${log_path}" >&2
      printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "${tag}" "${implementation}" "${PRECISION}" "${BATCH_SIZE}" "${embedding_dim}" "${sequence_length}" "${CAUSAL}" "${status}" "${json_path}" "${log_path}" \
        >> "${SUMMARY_FILE}"
    done
  done
done
