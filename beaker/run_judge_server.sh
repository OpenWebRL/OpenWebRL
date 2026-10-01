#!/usr/bin/env bash
# In-container entry point for a standalone VLM judge (WebJudge-7B by default)
# served by sglang on one GPU, launched by beaker/launch_judge_server.py with
# --host-networking so other jobs on the cluster can reach it by node IP.
#
# Once /health_generate answers, the OpenAI-compatible base URL is written to
# JUDGE_ENDPOINT_FILE on Weka; eval/RL jobs read that file (run_eval.sh,
# run_rl_sweep.sh) instead of hardcoding a host. The file is removed on exit so a
# dead judge is never picked up. The job then blocks until it is killed.
set -euo pipefail

OPENWEBRL_ROOT="${OPENWEBRL_ROOT:-/weka/oe-training-default/new_peters/OpenWebRL}"
MODEL_PATH="${MODEL_PATH:-/weka/oe-training-default/new_peters/models/WebJudge-7B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-WebJudge-7B}"
JUDGE_PORT="${JUDGE_PORT:-8300}"
JUDGE_ENDPOINT_FILE="${JUDGE_ENDPOINT_FILE:-${OPENWEBRL_ROOT}/outputs/judge/${SERVED_MODEL_NAME}.endpoint}"
SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-0.85}"
SGLANG_CONTEXT_LENGTH="${SGLANG_CONTEXT_LENGTH:-32768}"
SGLANG_STARTUP_TIMEOUT_SECS="${SGLANG_STARTUP_TIMEOUT_SECS:-1200}"

[ -d "${MODEL_PATH}" ] || { echo "ERROR: judge model dir not found: ${MODEL_PATH}"; exit 2; }
mkdir -p "$(dirname "${JUDGE_ENDPOINT_FILE}")"

# Same B300 fixes as run_eval.sh / run_rl_full.sh.
if [ -x /usr/local/cuda/bin/ptxas ]; then
  export TRITON_PTXAS_PATH="${TRITON_PTXAS_PATH:-/usr/local/cuda/bin/ptxas}"
fi
export FLASHINFER_DISABLE_VERSION_CHECK=1

# Node IP: with --host-networking this is routable from other pods.
NODE_IP="$(hostname -I | awk '{print $1}')"
ENDPOINT="http://${NODE_IP}:${JUDGE_PORT}"
LOG_DIR="$(dirname "${JUDGE_ENDPOINT_FILE}")/logs"
mkdir -p "${LOG_DIR}"
SERVER_LOG="${LOG_DIR}/${SERVED_MODEL_NAME}_$(date -u +%Y%m%dT%H%M%SZ).log"

cleanup() {
  rm -f "${JUDGE_ENDPOINT_FILE}"
  [ -n "${SERVER_PID:-}" ] && kill "${SERVER_PID}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "serving ${MODEL_PATH} as ${SERVED_MODEL_NAME} at ${ENDPOINT}; log ${SERVER_LOG}"
python -m sglang.launch_server \
  --model-path "${MODEL_PATH}" --served-model-name "${SERVED_MODEL_NAME}" \
  --host 0.0.0.0 --port "${JUDGE_PORT}" \
  --trust-remote-code --dtype bfloat16 --tp 1 \
  --mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC}" \
  --context-length "${SGLANG_CONTEXT_LENGTH}" \
  --attention-backend flashinfer --mm-attention-backend triton_attn \
  > "${SERVER_LOG}" 2>&1 &
SERVER_PID=$!

deadline=$((SECONDS + SGLANG_STARTUP_TIMEOUT_SECS))
until curl -sf "http://127.0.0.1:${JUDGE_PORT}/health_generate" > /dev/null; do
  kill -0 "${SERVER_PID}" 2>/dev/null || { echo "sglang exited early"; tail -n 50 "${SERVER_LOG}"; exit 1; }
  [ "${SECONDS}" -lt "${deadline}" ] || { echo "sglang startup timed out"; tail -n 50 "${SERVER_LOG}"; exit 1; }
  sleep 5
done

# Smoke test through the OpenAI-compatible route before advertising.
curl -sf "http://127.0.0.1:${JUDGE_PORT}/v1/chat/completions" -H 'Content-Type: application/json' \
  -d "{\"model\":\"${SERVED_MODEL_NAME}\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with OK.\"}],\"max_tokens\":4}" \
  | tee /dev/stderr | grep -q '"content"' || { echo "chat completion smoke test failed"; exit 1; }

echo "${ENDPOINT}" > "${JUDGE_ENDPOINT_FILE}.tmp" && mv "${JUDGE_ENDPOINT_FILE}.tmp" "${JUDGE_ENDPOINT_FILE}"
echo "judge ready: ${ENDPOINT} (published to ${JUDGE_ENDPOINT_FILE})"

# Stay alive; drop the endpoint file the moment the server dies.
while kill -0 "${SERVER_PID}" 2>/dev/null; do sleep 30; done
echo "sglang server exited"; tail -n 50 "${SERVER_LOG}"; exit 1
