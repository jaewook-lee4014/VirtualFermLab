#!/bin/bash
# Start vLLM server on a GPU node (H200 recommended for 32B model)
#
# Usage:
#   srun --jobid=<JOB_ID> --overlap bash scripts/start_vllm.sh
#
# Environment variables:
#   VLLM_PORT  (default: 8000)
#   VLLM_MODEL (default: Qwen/Qwen2.5-32B-Instruct)
#   HF_HOME    (default: /scratch/users/k23070952/hf_cache)

set -euo pipefail

export HF_HOME="${HF_HOME:-/scratch/users/k23070952/hf_cache}"

PYTHON="/scratch/users/k23070952/vllm_env310/bin/python"
PORT="${VLLM_PORT:-8000}"
MODEL="${VLLM_MODEL:-Qwen/Qwen2.5-32B-Instruct}"

echo "Starting vLLM server on port ${PORT} with model ${MODEL}"
echo "HF_HOME: ${HF_HOME}"
echo "Python: ${PYTHON}"

"${PYTHON}" -m vllm.entrypoints.openai.api_server \
    --model "${MODEL}" \
    --host 0.0.0.0 \
    --port "${PORT}" \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.90
