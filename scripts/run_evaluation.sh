#!/bin/bash
set -ex

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Load credentials/settings (JUDGE_*/AZURE_TOKEN_PATH/SANDBOX_*/BROWSER_USE_API_KEY)
# into this session so run_evaluate.py inherits them.
if [ -f "${REPO_ROOT}/.env" ]; then
    source "${REPO_ROOT}/.env"
fi

export SLIME_BROWSER_SANDBOX_MANIFEST_DIR=/tmp/slime_browser_sandboxes_eval
echo "SLIME_BROWSER_SANDBOX_MANIFEST_DIR=${SLIME_BROWSER_SANDBOX_MANIFEST_DIR}"


model_list_dir="${MODEL_LIST_DIR_DEFAULT:-}"


# OUTPUT_ROOT="${OUTPUT_ROOT:-eval_webvoyager_val/gpt4.1-judge-prompt-acthistory_SFT_1images_addenvfeedback_modebrowser_eval1image_evalstep30_fromsft_epoch1_v2}"
OUTPUT_ROOT="${OUTPUT_ROOT:-eval_onlinemind2web/gpt4.1-judge-prompt-acthistory_SFT_1images_addenvfeedback_modebrowser_eval1image_trainmaxstep15_evalstep30_fromsftepoch3_ppoepoch1}"
# OUTPUT_ROOT="${OUTPUT_ROOT:-eval_onlinemind2web/gpt4.1-judge-prompt-acthistory_SFT_1images_addenvfeedback_modebrowser_eval1image_trainmaxstep30_evalstep30_fromrl15stepsckpt84_sftepoch3}"
mkdir -p "${OUTPUT_ROOT}"
task_file="${TASK_FILE:-openwebrl/data/eval/online-mind2web.jsonl}"
# Eval judge protocol: browser | webvoyager | online_mind2web | deepshop
# (should match the benchmark in task_file).
EVAL_PROTOCOL="${EVAL_PROTOCOL:-online_mind2web}"

BROWSER_RESPONSE_FORMAT_MODE="${BROWSER_RESPONSE_FORMAT_MODE:-browser_env}" # slime
BROWSER_INCLUDE_TOOL_RESPONSE="${BROWSER_INCLUDE_TOOL_RESPONSE:-1}" ###################################tool response
SLIME_BROWSER_CHAT_TEMPLATE_ENABLE_THINKING="${SLIME_BROWSER_CHAT_TEMPLATE_ENABLE_THINKING:-1}"
SLIME_BROWSER_THINKING_OPEN_TAG="${SLIME_BROWSER_THINKING_OPEN_TAG:-<think>}"
SLIME_BROWSER_THINKING_CLOSE_TAG="${SLIME_BROWSER_THINKING_CLOSE_TAG:-</think>}"
SLIME_BROWSER_APPEND_THINKING_PREFILL="${SLIME_BROWSER_APPEND_THINKING_PREFILL:-0}"
TURN_HISTORY_REASONING_MODE="${TURN_HISTORY_REASONING_MODE:-full}" # full



MODEL_LIST_OVERRIDE="${MODEL_LIST_OVERRIDE:-/data/users/qianhuiwu/openwebrl_outputs/qwen3-vl-8b-thinking_browser_rl_turn_bs48_0329_gpt4.1reward_maxstep20_img1_fixbs256_ppoepoch2_fixmultimodalinput_20260705_061604_no-0.1_fixactions_temperature0.8_fromSFT912_epoch3_trainstep30fromckpt59_lr5e-7_hf/iter_0000054}"
MODEL_LIST_DIR="${MODEL_LIST_DIR:-${model_list_dir}}"
if [ -z "${MODEL_LIST_OVERRIDE}" ] && [ -z "${MODEL_LIST_DIR}" ]; then
    echo "ERROR: set MODEL_LIST_DIR or MODEL_LIST_OVERRIDE to the converted checkpoint(s) to evaluate."
    exit 1
fi

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
# SGLANG_PYTHON_BIN="${SGLANG_PYTHON_BIN:-/root/.venv_sglang/bin/python}"

# Leave JUDGE_MODEL empty to use each protocol's canonical judge
# (webvoyager/deepshop=gpt-4o, online_mind2web=o4-mini, browser=gpt-4.1);
# set it (e.g. JUDGE_MODEL=gpt-4.1) to force one judge model across all protocols.
JUDGE_MODEL="${JUDGE_MODEL:-}"
JUDGE_API_MODE="${JUDGE_API_MODE:-token}" 
# JUDGE_API_MODE="${JUDGE_API_MODE:-served}" # token, served, api_key
JUDGE_API_HOST="${JUDGE_API_HOST:-}"
JUDGE_API_PORT="${JUDGE_API_PORT:-}"
JUDGE_PROMPT_VARIANT="${JUDGE_PROMPT_VARIANT:-action_history}" # default
CONTEXT_NUM_SCREENSHOTS="${CONTEXT_NUM_SCREENSHOTS:-1}" #####################################################
JUDGE_MAX_ATTACHED_IMGS="${JUDGE_MAX_ATTACHED_IMGS:-3}"
BROWSER_MAX_STEPS="${BROWSER_MAX_STEPS:-30}"

export JUDGE_API_BASE
export JUDGE_API_HOST
export JUDGE_API_PORT

export SLIME_BROWSER_SANDBOX_MAX_SANDBOXES="${SLIME_BROWSER_SANDBOX_MAX_SANDBOXES:-20}"
export SLIME_BROWSER_SANDBOX_ACQUIRE_TIMEOUT_SECS="${SLIME_BROWSER_SANDBOX_ACQUIRE_TIMEOUT_SECS:-600}"

echo "SANDBOX_MAX_SANDBOXES=${SLIME_BROWSER_SANDBOX_MAX_SANDBOXES}"
echo "SANDBOX_ACQUIRE_TIMEOUT_SECS=${SLIME_BROWSER_SANDBOX_ACQUIRE_TIMEOUT_SECS}"
echo "EVAL_PROTOCOL=${EVAL_PROTOCOL}"
echo "JUDGE_MODEL=${JUDGE_MODEL:-<per-protocol default>}"
echo "JUDGE_API_MODE=${JUDGE_API_MODE}"
echo "JUDGE_API_BASE=${JUDGE_API_BASE}"
echo "JUDGE_API_HOST=${JUDGE_API_HOST}"
echo "JUDGE_API_PORT=${JUDGE_API_PORT}"
echo "JUDGE_PROMPT_VARIANT=${JUDGE_PROMPT_VARIANT}"
echo "CONTEXT_NUM_SCREENSHOTS=${CONTEXT_NUM_SCREENSHOTS}"
echo "JUDGE_MAX_ATTACHED_IMGS=${JUDGE_MAX_ATTACHED_IMGS}"
echo "TURN_HISTORY_REASONING_MODE=${TURN_HISTORY_REASONING_MODE}"
echo "BROWSER_RESPONSE_FORMAT_MODE=${BROWSER_RESPONSE_FORMAT_MODE}"
echo "BROWSER_INCLUDE_TOOL_RESPONSE=${BROWSER_INCLUDE_TOOL_RESPONSE}"
echo "BROWSER_MAX_STEPS=${BROWSER_MAX_STEPS}"
echo "MODEL_LIST_OVERRIDE=${MODEL_LIST_OVERRIDE}"
echo "MODEL_LIST_DIR=${MODEL_LIST_DIR}"
echo "SGLANG_AUTO_SERVE=${SGLANG_AUTO_SERVE}"
echo "SGLANG_SERVER_HOST=${SGLANG_SERVER_HOST}"
echo "SGLANG_CONNECT_IP=${SGLANG_CONNECT_IP}"
echo "SGLANG_PORT=${SGLANG_PORT}"
echo "SGLANG_DTYPE=${SGLANG_DTYPE}"
echo "SGLANG_MEM_FRACTION_STATIC=${SGLANG_MEM_FRACTION_STATIC}"
echo "SGLANG_CONTEXT_LENGTH=${SGLANG_CONTEXT_LENGTH}"
echo "SGLANG_TP=${SGLANG_TP}"
echo "SGLANG_DP=${SGLANG_DP}"
echo "SGLANG_STARTUP_TIMEOUT_SECS=${SGLANG_STARTUP_TIMEOUT_SECS}"


ip="${SGLANG_CONNECT_IP}"
port="${SGLANG_PORT}"



SGLANG_SERVER_PID=""
SGLANG_SERVER_LOG=""

# resolve_sglang_python() {
#     if [ -n "${SGLANG_PYTHON_BIN}" ]; then
#         if [ ! -x "${SGLANG_PYTHON_BIN}" ]; then
#             echo "Configured SGLANG_PYTHON_BIN is not executable: ${SGLANG_PYTHON_BIN}" >&2
#             return 1
#         fi
#         echo "${SGLANG_PYTHON_BIN}"
#         return 0
#     fi
#     if [ -n "${SGLANG_VENV_ACTIVATE}" ]; then
#         echo "python3"
#         return 0
#     fi
#     if [ -x "${REPO_ROOT}/.venv_sglang/bin/python" ]; then
#         echo "${REPO_ROOT}/.venv_sglang/bin/python"
#         return 0
#     fi
#         return 0
#     fi
#     echo "python3"
# }

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
    local python_bin
    local startup_deadline
    local launch_cmd

    if [ "${SGLANG_AUTO_SERVE}" != "1" ]; then
        return 0
    fi

    stop_sglang_server
    mkdir -p "${log_dir}"
    SGLANG_SERVER_LOG="${log_dir}/${model_tag}.log"
    # python_bin="$(resolve_sglang_python)"
    python_bin='python'

    if [ -n "${SGLANG_VENV_ACTIVATE}" ]; then
        launch_cmd="source \"${SGLANG_VENV_ACTIVATE}\" && exec ${python_bin} -m sglang.launch_server"
    else
        launch_cmd="exec ${python_bin} -m sglang.launch_server"
    fi

    launch_cmd="${launch_cmd} --model-path \"${current_model_name}\" --host \"${SGLANG_SERVER_HOST}\" --port \"${SGLANG_PORT}\" --trust-remote-code --dtype \"${SGLANG_DTYPE}\" --mem-fraction-static \"${SGLANG_MEM_FRACTION_STATIC}\" --context-length \"${SGLANG_CONTEXT_LENGTH}\" --tp \"${SGLANG_TP}\" --dp \"${SGLANG_DP}\" --attention-backend flashinfer --mm-attention-backend triton_attn "

    echo "Starting sglang server for model_name=${current_model_name}"
    setsid bash -lc "${launch_cmd}" > "${SGLANG_SERVER_LOG}" 2>&1 &
    SGLANG_SERVER_PID=$!
    startup_deadline=$((SECONDS + SGLANG_STARTUP_TIMEOUT_SECS))

    until curl -sf "http://${ip}:${port}/health_generate" > /dev/null; do
        if ! kill -0 "${SGLANG_SERVER_PID}" 2>/dev/null; then
            echo "sglang server exited early. Log: ${SGLANG_SERVER_LOG}"
            tail -n 50 "${SGLANG_SERVER_LOG}" || true
            return 1
        fi
        if [ "${SECONDS}" -ge "${startup_deadline}" ]; then
            echo "Timed out waiting for sglang server. Log: ${SGLANG_SERVER_LOG}"
            tail -n 50 "${SGLANG_SERVER_LOG}" || true
            return 1
        fi
        sleep 2
    done

    curl -sf "http://${ip}:${port}/get_model_info" > /dev/null || true
}

trap stop_sglang_server EXIT

run_eval_for_model() {
    local current_model_name="$1"
    local model_tag
    model_tag="$(basename "${current_model_name}")"

    echo "Evaluating model_name=${current_model_name}"
    start_sglang_server "${current_model_name}" "${model_tag}"

    # Only pass --judge-model when explicitly set, so the protocol's canonical
    # judge default applies otherwise.
    local judge_model_arg=()
    if [ -n "${JUDGE_MODEL}" ]; then
        judge_model_arg=(--judge-model "${JUDGE_MODEL}")
    fi

    python openwebrl/run_evaluate.py --task-file  ${task_file} \
            --eval-protocol "${EVAL_PROTOCOL}" \
            --hf-checkpoint "${current_model_name}" \
            --output "${OUTPUT_ROOT}/${model_tag}" \
            --n-parallel "${SLIME_BROWSER_SANDBOX_MAX_SANDBOXES}" \
            --sglang-ip "${ip}" --sglang-port "${port}" \
            --context-num-screenshots "${CONTEXT_NUM_SCREENSHOTS}" \
            --judge-max-attached-imgs "${JUDGE_MAX_ATTACHED_IMGS}" \
            "${judge_model_arg[@]}" \
            --judge-api-mode "${JUDGE_API_MODE}" \
            --judge-prompt-variant "${JUDGE_PROMPT_VARIANT}" \
            --turn-history-reasoning-mode "${TURN_HISTORY_REASONING_MODE}" \
            --browser-response-format-mode "${BROWSER_RESPONSE_FORMAT_MODE}" \
            --browser-include-tool-response "${BROWSER_INCLUDE_TOOL_RESPONSE}" \
            --max-steps "${BROWSER_MAX_STEPS}" \
            --turn-level
    stop_sglang_server
}

collect_models_from_dir() {
    local model_list_dir="$1"
    local collected_models=()

    if [ ! -d "${model_list_dir}" ]; then
        echo "MODEL_LIST_DIR does not exist: ${model_list_dir}" >&2
        return 1
    fi

    while IFS= read -r model_dir; do
        collected_models+=("${model_dir}")
    done < <(
        find "${model_list_dir}" -maxdepth 2 -mindepth 2 -type d -name '*_converted' | sort -V -r
    )

    if [ "${#collected_models[@]}" -eq 0 ]; then
        echo "No converted model directories found under MODEL_LIST_DIR=${model_list_dir}" >&2
        return 1
    fi

    printf '%s\n' "${collected_models[@]}"
}

MODELS_TO_EVAL=()
if [ -n "${MODEL_LIST_OVERRIDE}" ]; then
    read -r -a MODELS_TO_EVAL <<< "${MODEL_LIST_OVERRIDE}"
elif [ -n "${MODEL_LIST_DIR}" ]; then
    mapfile -t MODELS_TO_EVAL < <(collect_models_from_dir "${MODEL_LIST_DIR}")
elif [ "${#MODEL_LIST[@]}" -gt 0 ]; then
    MODELS_TO_EVAL=("${MODEL_LIST[@]}")
else
    MODELS_TO_EVAL=("${model_name}")
fi

printf 'Resolved MODELS_TO_EVAL:\n'
printf '  %s\n' "${MODELS_TO_EVAL[@]}"

for model_name in "${MODELS_TO_EVAL[@]}"; do
    run_eval_for_model "${model_name}"
done
         
        # --task-indices "0,1"

# bash scripts/clean_processes.sh

# sleep 5

#         --hf-checkpoint ${model_name} \
#         --output  eval \
#         --n-parallel 4 \
#         --sglang-ip 0.0.0.0 --sglang-port 30001  

# bash scripts/clean_processes.sh
