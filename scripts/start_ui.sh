#!/bin/bash
# Start the VirtualFermLab Flask web UI on a compute node
#
# Usage:
#   srun --jobid=<JOB_ID> --overlap bash scripts/start_ui.sh
#
# Environment variables:
#   UI_PORT        (default: 51665)
#   VLLM_BASE_URL  (default: http://localhost:8000/v1)

set -euo pipefail

PYTHON="/scratch/users/k23070952/vllm_env310/bin/python"
PORT="${UI_PORT:-51665}"
export VLLM_BASE_URL="${VLLM_BASE_URL:-http://localhost:8000/v1}"

echo "Starting VirtualFermLab UI on port ${PORT}"
echo "VLLM_BASE_URL: ${VLLM_BASE_URL}"
echo "Python: ${PYTHON}"

"${PYTHON}" -m virtualfermlab.web.app --host 0.0.0.0 --port "${PORT}"
