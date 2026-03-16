#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

MODEL_SIZE="${1:-2.7b}"
CONTEXT_LENGTH="${2:-512}"
OUTPUT_DIR="${3:-artifacts/experiments/ch1/1_1_4b}"

FORWARD_REP=".agents/logs/1_1_4_forward_attention/${MODEL_SIZE}_ctx${CONTEXT_LENGTH}_forward_attention.nsys-rep"
TRAIN_STEP_REP=".agents/logs/1_1_4_train_step_attention/${MODEL_SIZE}_ctx${CONTEXT_LENGTH}_train_step_attention.nsys-rep"

mkdir -p "${OUTPUT_DIR}"

if ! command -v nsys >/dev/null 2>&1; then
  echo "nsys is not installed on this machine." >&2
  exit 1
fi

if [[ ! -f "${FORWARD_REP}" ]]; then
  echo "Missing forward profile: ${FORWARD_REP}" >&2
  exit 1
fi

if [[ ! -f "${TRAIN_STEP_REP}" ]]; then
  echo "Missing train-step profile: ${TRAIN_STEP_REP}" >&2
  exit 1
fi

echo "Capturing CUDA GPU Kernel Summary for ${MODEL_SIZE}, ctx=${CONTEXT_LENGTH}"

nsys stats \
  --report cuda_gpu_kern_sum \
  "${FORWARD_REP}" \
  | tee "${OUTPUT_DIR}/${MODEL_SIZE}_ctx${CONTEXT_LENGTH}_forward_cuda_gpu_kern_sum.txt"

nsys stats \
  --report cuda_gpu_kern_sum \
  "${TRAIN_STEP_REP}" \
  | tee "${OUTPUT_DIR}/${MODEL_SIZE}_ctx${CONTEXT_LENGTH}_train_step_cuda_gpu_kern_sum.txt"

nsys stats \
  --report cuda_gpu_kern_sum:nvtx-name:base \
  "${FORWARD_REP}" \
  | tee "${OUTPUT_DIR}/${MODEL_SIZE}_ctx${CONTEXT_LENGTH}_forward_cuda_gpu_kern_sum_nvtx.txt"

nsys stats \
  --report cuda_gpu_kern_sum:nvtx-name:base \
  "${TRAIN_STEP_REP}" \
  | tee "${OUTPUT_DIR}/${MODEL_SIZE}_ctx${CONTEXT_LENGTH}_train_step_cuda_gpu_kern_sum_nvtx.txt"
