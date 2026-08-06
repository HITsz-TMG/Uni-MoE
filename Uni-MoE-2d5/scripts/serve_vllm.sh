#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

: "${MODEL_PATH:?Set MODEL_PATH to the UniMoE-2.5 checkpoint directory}"
HOST=${HOST:-0.0.0.0}
PORT=${PORT:-8000}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-unimoe2d5}
TP_SIZE=${TP_SIZE:-1}
DP_SIZE=${DP_SIZE:-1}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.90}
LIMIT_MM_PER_PROMPT=${LIMIT_MM_PER_PROMPT:-'{"image":1,"video":1,"audio":1}'}
ENFORCE_EAGER=${ENFORCE_EAGER:-1}
ENABLE_EXPERT_PARALLEL=${ENABLE_EXPERT_PARALLEL:-0}

export HCCL_BUFFSIZE=${HCCL_BUFFSIZE:-512}
export HCCL_OP_EXPANSION_MODE=${HCCL_OP_EXPANSION_MODE:-AIV}

if [[ "${ASCEND_LAUNCH_BLOCKING:-0}" == "1" ]]; then
    echo "[serve-vllm] unsetting ASCEND_LAUNCH_BLOCKING=1" >&2
    unset ASCEND_LAUNCH_BLOCKING
fi

args=(
    serve "${MODEL_PATH}"
    --host "${HOST}"
    --port "${PORT}"
    --served-model-name "${SERVED_MODEL_NAME}"
    --tensor-parallel-size "${TP_SIZE}"
    --data-parallel-size "${DP_SIZE}"
    --max-model-len "${MAX_MODEL_LEN}"
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
    --limit-mm-per-prompt "${LIMIT_MM_PER_PROMPT}"
    --chat-template "${REPO_ROOT}/assets/chat_template_vllm.jinja"
    --trust-remote-code
)

if [[ "${ENFORCE_EAGER}" == "1" ]]; then
    args+=(--enforce-eager)
fi
if [[ "${ENABLE_EXPERT_PARALLEL}" == "1" ]]; then
    args+=(--enable-expert-parallel)
fi

echo "[serve-vllm] served_model=${SERVED_MODEL_NAME} tp=${TP_SIZE} dp=${DP_SIZE} eager=${ENFORCE_EAGER}"
exec vllm "${args[@]}" "$@"
