#!/bin/bash
# Start the LLM Parameter Extraction REST API server.
#
# Environment variables:
#   API_PORT        (default: 5001)
#   VLLM_BASE_URL   (default: http://localhost:8000/v1)
#   VLLM_MODEL      (default: Qwen/Qwen2.5-32B-Instruct)
#   VLLM_MAX_TOKENS (default: 4096)

set -euo pipefail

export API_PORT="${API_PORT:-5001}"
export VLLM_BASE_URL="${VLLM_BASE_URL:-http://localhost:8000/v1}"
export VLLM_MODEL="${VLLM_MODEL:-Qwen/Qwen2.5-32B-Instruct}"
export VLLM_MAX_TOKENS="${VLLM_MAX_TOKENS:-4096}"

echo "Starting Extraction API on port ${API_PORT}"
echo "  VLLM_BASE_URL:   ${VLLM_BASE_URL}"
echo "  VLLM_MODEL:      ${VLLM_MODEL}"
echo "  VLLM_MAX_TOKENS: ${VLLM_MAX_TOKENS}"

python -m virtualfermlab.extraction_api.app --port "${API_PORT}"
