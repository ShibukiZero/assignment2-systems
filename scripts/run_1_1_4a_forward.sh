#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

OUTPUT_DIR="${1:-experiments/ch1/1_1_4a_forward}"
mkdir -p "${OUTPUT_DIR}"

SIZES=(small medium large xl 2.7b)
CONTEXT_LENGTHS=(128 256 512 1024)

for size in "${SIZES[@]}"; do
  for context_length in "${CONTEXT_LENGTHS[@]}"; do
    tag="${size}_ctx${context_length}_forward"
    echo "Profiling ${tag}"
    nsys profile \
      --force-overwrite=true \
      -t cuda,nvtx,osrt \
      -o "${OUTPUT_DIR}/${tag}" \
      uv run python -m cs336_systems.benchmark \
        --model-size "${size}" \
        --context-length "${context_length}" \
        --mode forward \
        --warmup-steps 5 \
        --measure-steps 10 \
        --nvtx \
        --output-path "${OUTPUT_DIR}/${tag}.json"
  done
done
