#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATASET="${1:-ncet}"
TIMESTAMP="$(date +"%Y-%m-%d_%H-%M-%S")"

LOG_DIR="${SCRIPT_DIR}/logs"
OUTPUT_DIR="${SCRIPT_DIR}/outputs"
DATA_DIR="${DATA_DIR:-/root/autodl-tmp}"
mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}"

LOG_FILE="${LOG_DIR}/transformer_${DATASET}_${TIMESTAMP}.log"
echo "=== Running dataset=${DATASET}, fixed_context=local, data_dir=${DATA_DIR} ===" | tee "${LOG_FILE}"
"${PYTHON_BIN}" "${SCRIPT_DIR}/main.py" \
  --dataset "${DATASET}" \
  --data_dir "${DATA_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  2>&1 | tee -a "${LOG_FILE}"
