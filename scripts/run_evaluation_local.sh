#!/bin/bash
set -euo pipefail
set -x

export AZURE_RESOURCE_NAME="${AZURE_RESOURCE_NAME:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# Force browser generation to use the local subprocess-backed env_server path.
export SLIME_BROWSER_ENV_MODE="${SLIME_BROWSER_ENV_MODE:-local_process}"
export SLIME_BROWSER_LOCAL_PROCESS_HOST="${SLIME_BROWSER_LOCAL_PROCESS_HOST:-127.0.0.1}"
export SLIME_BROWSER_LOCAL_PROCESS_PORT_START="${SLIME_BROWSER_LOCAL_PROCESS_PORT_START:-18000}"
export SLIME_BROWSER_LOCAL_PROCESS_PORT_END="${SLIME_BROWSER_LOCAL_PROCESS_PORT_END:-18999}"
export SLIME_BROWSER_LOCAL_PROCESS_MAX_PROCESSES="${SLIME_BROWSER_LOCAL_PROCESS_MAX_PROCESSES:-8}"
export SLIME_BROWSER_LOCAL_PROCESS_ACQUIRE_TIMEOUT_SECS="${SLIME_BROWSER_LOCAL_PROCESS_ACQUIRE_TIMEOUT_SECS:-600}"
export SLIME_BROWSER_LOCAL_PROCESS_STARTUP_TIMEOUT_SECS="${SLIME_BROWSER_LOCAL_PROCESS_STARTUP_TIMEOUT_SECS:-60}"
export SLIME_BROWSER_LOCAL_PROCESS_REQUEST_TIMEOUT_SECS="${SLIME_BROWSER_LOCAL_PROCESS_REQUEST_TIMEOUT_SECS:-300}"
export SLIME_BROWSER_LOCAL_PROCESS_EXIT_TIMEOUT_SECS="${SLIME_BROWSER_LOCAL_PROCESS_EXIT_TIMEOUT_SECS:-10}"
export SLIME_BROWSER_LOCAL_PROCESS_KILL_TIMEOUT_SECS="${SLIME_BROWSER_LOCAL_PROCESS_KILL_TIMEOUT_SECS:-5}"
export SLIME_BROWSER_LOCAL_PROCESS_LOG_DIR="${SLIME_BROWSER_LOCAL_PROCESS_LOG_DIR:-/tmp/slime_browser_local_process_env_logs_eval}"
export SLIME_BROWSER_LOCAL_PROCESS_PORT_LOCK_DIR="${SLIME_BROWSER_LOCAL_PROCESS_PORT_LOCK_DIR:-/tmp/slime_browser_local_process_ports_eval}"
export SLIME_BROWSER_LOCAL_PROCESS_PYTHON="${SLIME_BROWSER_LOCAL_PROCESS_PYTHON:-$(command -v python)}"
export PLAYWRIGHT_CLI="${PLAYWRIGHT_CLI:-/usr/local/bin/playwright}"

if [ -z "${PLAYWRIGHT_BROWSERS_PATH:-}" ]; then
    if find /root/.cache/ms-playwright -path '*/chromium_headless_shell-*/*/chrome-headless-shell' -perm -111 -print -quit 2>/dev/null | grep -q .; then
        export PLAYWRIGHT_BROWSERS_PATH="/root/.cache/ms-playwright"
    else
        export PLAYWRIGHT_BROWSERS_PATH="${REPO_ROOT}/.cache/ms-playwright"
    fi
else
    export PLAYWRIGHT_BROWSERS_PATH
fi

MODEL_PATH="${MODEL_PATH:-}"
if [ -z "${MODEL_PATH}" ]; then
    echo "ERROR: set MODEL_PATH to a converted HF checkpoint before running evaluation."
    exit 1
fi
TASK_FILE="${TASK_FILE:-openwebrl/data/online-mind2web.jsonl}"
OUTPUT_ROOT="${OUTPUT_ROOT:-eval_onlinemind2web/local_process_smoke}"
TASK_INDICES="${TASK_INDICES:-}"
N_PARALLEL="${N_PARALLEL:-8}"

BROWSER_RESPONSE_FORMAT_MODE="${BROWSER_RESPONSE_FORMAT_MODE:-browser_env}"
BROWSER_INCLUDE_TOOL_RESPONSE="${BROWSER_INCLUDE_TOOL_RESPONSE:-1}"
export SLIME_BROWSER_CHAT_TEMPLATE_ENABLE_THINKING="${SLIME_BROWSER_CHAT_TEMPLATE_ENABLE_THINKING:-1}"
export SLIME_BROWSER_THINKING_OPEN_TAG="${SLIME_BROWSER_THINKING_OPEN_TAG:-<think>}"
export SLIME_BROWSER_THINKING_CLOSE_TAG="${SLIME_BROWSER_THINKING_CLOSE_TAG:-</think>}"
export SLIME_BROWSER_APPEND_THINKING_PREFILL="${SLIME_BROWSER_APPEND_THINKING_PREFILL:-0}"
TURN_HISTORY_REASONING_MODE="${TURN_HISTORY_REASONING_MODE:-full}"
CONTEXT_NUM_SCREENSHOTS="${CONTEXT_NUM_SCREENSHOTS:-1}"
JUDGE_MAX_ATTACHED_IMGS="${JUDGE_MAX_ATTACHED_IMGS:-3}"
BROWSER_MAX_STEPS="${BROWSER_MAX_STEPS:-30}"
TEMPERATURE="${TEMPERATURE:-0.0}"
INFERENCE_STEP_TIMEOUT_SECS="${INFERENCE_STEP_TIMEOUT_SECS:-120}"
TASK_TIMEOUT_SECS="${TASK_TIMEOUT_SECS:-600}"

SGLANG_AUTO_SERVE="${SGLANG_AUTO_SERVE:-1}"
SGLANG_SERVER_HOST="${SGLANG_SERVER_HOST:-0.0.0.0}"
SGLANG_CONNECT_IP="${SGLANG_CONNECT_IP:-127.0.0.1}"
SGLANG_PORT="${SGLANG_PORT:-9000}"
SGLANG_DTYPE="${SGLANG_DTYPE:-bfloat16}"
SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-0.9}"
SGLANG_CONTEXT_LENGTH="${SGLANG_CONTEXT_LENGTH:-32768}"
SGLANG_TP="${SGLANG_TP:-2}"
SGLANG_DP="${SGLANG_DP:-4}"
SGLANG_STARTUP_TIMEOUT_SECS="${SGLANG_STARTUP_TIMEOUT_SECS:-600}"
SGLANG_VENV_ACTIVATE="${SGLANG_VENV_ACTIVATE:-}"

JUDGE_MODEL="${JUDGE_MODEL:-gpt-4.1}"
JUDGE_API_MODE="${JUDGE_API_MODE:-token}"
JUDGE_API_BASE="${JUDGE_API_BASE:-}"
JUDGE_API_HOST="${JUDGE_API_HOST:-}"
JUDGE_API_PORT="${JUDGE_API_PORT:-}"
JUDGE_PROMPT_VARIANT="${JUDGE_PROMPT_VARIANT:-action_history}"
export JUDGE_API_BASE JUDGE_API_HOST JUDGE_API_PORT

mkdir -p "${OUTPUT_ROOT}" "${SLIME_BROWSER_LOCAL_PROCESS_LOG_DIR}" "${SLIME_BROWSER_LOCAL_PROCESS_PORT_LOCK_DIR}"

ensure_playwright_browser() {
    local preflight_log="${OUTPUT_ROOT}/playwright_preflight.log"

    mkdir -p "${PLAYWRIGHT_BROWSERS_PATH}"
    if "${SLIME_BROWSER_LOCAL_PROCESS_PYTHON}" - <<'PY' > "${preflight_log}" 2>&1
import asyncio
from playwright.async_api import async_playwright

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True, args=["--no-sandbox", "--disable-dev-shm-usage"])
        page = await browser.new_page()
        await page.goto("about:blank")
        await browser.close()

asyncio.run(main())
PY
    then
        echo "Playwright Chromium preflight OK. PLAYWRIGHT_BROWSERS_PATH=${PLAYWRIGHT_BROWSERS_PATH}"
        return 0
    fi

    if grep -q "playwright install" "${preflight_log}" || grep -q "Executable doesn't exist" "${preflight_log}"; then
        echo "Playwright Chromium is missing; installing into PLAYWRIGHT_BROWSERS_PATH=${PLAYWRIGHT_BROWSERS_PATH}"
        if [ -x "${PLAYWRIGHT_CLI}" ]; then
            "${PLAYWRIGHT_CLI}" install chromium
        else
            "${SLIME_BROWSER_LOCAL_PROCESS_PYTHON}" -m playwright install chromium
        fi
        "${SLIME_BROWSER_LOCAL_PROCESS_PYTHON}" - <<'PY' > "${preflight_log}" 2>&1
import asyncio
from playwright.async_api import async_playwright

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True, args=["--no-sandbox", "--disable-dev-shm-usage"])
        page = await browser.new_page()
        await page.goto("about:blank")
        await browser.close()

asyncio.run(main())
PY
        echo "Playwright Chromium preflight OK after install."
        return 0
    fi

    echo "Playwright Chromium preflight failed. Log: ${preflight_log}" >&2
    tail -n 80 "${preflight_log}" >&2 || true
    return 1
}

echo "SLIME_BROWSER_ENV_MODE=${SLIME_BROWSER_ENV_MODE}"
echo "SLIME_BROWSER_LOCAL_PROCESS_MAX_PROCESSES=${SLIME_BROWSER_LOCAL_PROCESS_MAX_PROCESSES}"
echo "SLIME_BROWSER_LOCAL_PROCESS_LOG_DIR=${SLIME_BROWSER_LOCAL_PROCESS_LOG_DIR}"
echo "PLAYWRIGHT_BROWSERS_PATH=${PLAYWRIGHT_BROWSERS_PATH}"
echo "PLAYWRIGHT_CLI=${PLAYWRIGHT_CLI}"
echo "MODEL_PATH=${MODEL_PATH}"
echo "TASK_FILE=${TASK_FILE}"
echo "TASK_INDICES=${TASK_INDICES}"
echo "N_PARALLEL=${N_PARALLEL}"
echo "BROWSER_MAX_STEPS=${BROWSER_MAX_STEPS}"
echo "OUTPUT_ROOT=${OUTPUT_ROOT}"

SGLANG_SERVER_PID=""
SGLANG_SERVER_LOG=""

stop_sglang_server() {
    if [ -n "${SGLANG_SERVER_PID}" ]; then
        kill -TERM -- "-${SGLANG_SERVER_PID}" 2>/dev/null || true
        sleep 5
        kill -KILL -- "-${SGLANG_SERVER_PID}" 2>/dev/null || true
        wait "${SGLANG_SERVER_PID}" 2>/dev/null || true
        SGLANG_SERVER_PID=""
    fi
}

start_sglang_server() {
    local current_model_name="$1"
    local model_tag="$2"
    local log_dir="${OUTPUT_ROOT}/server_logs"
    local python_bin="python"
    local startup_deadline
    local launch_cmd

    if [ "${SGLANG_AUTO_SERVE}" != "1" ]; then
        return 0
    fi

    stop_sglang_server
    mkdir -p "${log_dir}"
    SGLANG_SERVER_LOG="${log_dir}/${model_tag}.log"

    if [ -n "${SGLANG_VENV_ACTIVATE}" ]; then
        launch_cmd="source \"${SGLANG_VENV_ACTIVATE}\" && exec ${python_bin} -m sglang.launch_server"
    else
        launch_cmd="exec ${python_bin} -m sglang.launch_server"
    fi
    launch_cmd="${launch_cmd} --model-path \"${current_model_name}\" --host \"${SGLANG_SERVER_HOST}\" --port \"${SGLANG_PORT}\" --trust-remote-code --dtype \"${SGLANG_DTYPE}\" --mem-fraction-static \"${SGLANG_MEM_FRACTION_STATIC}\" --context-length \"${SGLANG_CONTEXT_LENGTH}\" --tp \"${SGLANG_TP}\" --dp \"${SGLANG_DP}\""

    echo "Starting sglang server for model_name=${current_model_name}"
    setsid bash -lc "${launch_cmd}" > "${SGLANG_SERVER_LOG}" 2>&1 &
    SGLANG_SERVER_PID=$!
    startup_deadline=$((SECONDS + SGLANG_STARTUP_TIMEOUT_SECS))

    until curl -sf "http://${SGLANG_CONNECT_IP}:${SGLANG_PORT}/health_generate" > /dev/null; do
        if ! kill -0 "${SGLANG_SERVER_PID}" 2>/dev/null; then
            echo "sglang server exited early. Log: ${SGLANG_SERVER_LOG}" >&2
            tail -n 80 "${SGLANG_SERVER_LOG}" || true
            return 1
        fi
        if [ "${SECONDS}" -ge "${startup_deadline}" ]; then
            echo "Timed out waiting for sglang server. Log: ${SGLANG_SERVER_LOG}" >&2
            tail -n 80 "${SGLANG_SERVER_LOG}" || true
            return 1
        fi
        sleep 2
    done

    curl -sf "http://${SGLANG_CONNECT_IP}:${SGLANG_PORT}/get_model_info" > /dev/null || true
}

trap stop_sglang_server EXIT

model_tag="$(basename "${MODEL_PATH}")"
ensure_playwright_browser
start_sglang_server "${MODEL_PATH}" "${model_tag}"

python openwebrl/run_evaluate.py \
    --task-file "${TASK_FILE}" \
    --hf-checkpoint "${MODEL_PATH}" \
    --output "${OUTPUT_ROOT}/${model_tag}" \
    --n-parallel "${N_PARALLEL}" \
    --sglang-ip "${SGLANG_CONNECT_IP}" \
    --sglang-port "${SGLANG_PORT}" \
    --context-num-screenshots "${CONTEXT_NUM_SCREENSHOTS}" \
    --judge-max-attached-imgs "${JUDGE_MAX_ATTACHED_IMGS}" \
    --judge-model "${JUDGE_MODEL}" \
    --judge-api-mode "${JUDGE_API_MODE}" \
    --judge-prompt-variant "${JUDGE_PROMPT_VARIANT}" \
    --turn-history-reasoning-mode "${TURN_HISTORY_REASONING_MODE}" \
    --browser-response-format-mode "${BROWSER_RESPONSE_FORMAT_MODE}" \
    --browser-include-tool-response "${BROWSER_INCLUDE_TOOL_RESPONSE}" \
    --max-steps "${BROWSER_MAX_STEPS}" \
    --temperature "${TEMPERATURE}" \
    --task-indices "${TASK_INDICES}" \
    --inference-step-timeout-secs "${INFERENCE_STEP_TIMEOUT_SECS}" \
    --task-timeout-secs "${TASK_TIMEOUT_SECS}" \
    --turn-level

stop_sglang_server
