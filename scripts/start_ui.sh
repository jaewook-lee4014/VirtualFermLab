#!/bin/bash
# Start the VirtualFermLab Flask web UI on a compute node
#
# Usage:
#   srun --jobid=<JOB_ID> --overlap bash scripts/start_ui.sh
#
# Environment variables:
#   UI_PORT        (default: 8080)
#   VLLM_BASE_URL  (default: http://localhost:8000/v1)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${SCRIPT_DIR}/../.venv/bin/python"
PORT="${UI_PORT:-8080}"
export VLLM_BASE_URL="${VLLM_BASE_URL:-http://localhost:8000/v1}"

echo "Starting VirtualFermLab UI on port ${PORT}"
echo "VLLM_BASE_URL: ${VLLM_BASE_URL}"
echo "Python: ${PYTHON}"

"${PYTHON}" -m virtualfermlab.web.app --host 0.0.0.0 --port "${PORT}"
