#!/bin/bash
# Start vLLM server on a GPU node (H200 recommended for 32B model)
#
# Usage:
#   srun --jobid=<JOB_ID> --overlap bash scripts/start_vllm.sh
#
# Environment variables:
#   VLLM_PORT    (default: 8000)
#   VLLM_MODEL   (default: Qwen/Qwen2.5-32B-Instruct)
#   VLLM_TP_SIZE (default: 2) — tensor parallel size (number of GPUs)
#   HF_HOME      (default: /scratch/users/k23070952/hf_cache)

set -euo pipefail

export HF_HOME="${HF_HOME:-/dev/shm/hf_cache}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${SCRIPT_DIR}/../.venv/bin/python"
PORT="${VLLM_PORT:-8000}"
MODEL="${VLLM_MODEL:-Qwen/Qwen3.5-27B}"
TP_SIZE="${VLLM_TP_SIZE:-2}"

echo "Starting vLLM server on port ${PORT} with model ${MODEL}"
echo "Tensor parallel size: ${TP_SIZE}"
echo "HF_HOME: ${HF_HOME}"
echo "Python: ${PYTHON}"

"${PYTHON}" -m vllm.entrypoints.openai.api_server \
    --model "${MODEL}" \
    --host 0.0.0.0 \
    --port "${PORT}" \
    --tensor-parallel-size "${TP_SIZE}" \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.90
