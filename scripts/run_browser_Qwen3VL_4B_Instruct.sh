#!/bin/bash
# ==============================================================================
# Train Qwen3-VL-4B-Instruct for Browser Agent tasks.
#
# Hardware: 1 node × 8 GPUs (180GB B200)
# Backend: Megatron (TP=2, SP)
# Method:  GRPO with format + LLM-as-a-Judge reward on WebVoyager tasks
#
# Model details:
#   - Dense 4B parameter VL model
#   - Megatron shards across 2 TP ranks for training
#   - 8 parallel SGLang engines for rollout (1 GPU each)
#
# Usage:
#   bash scripts/run_browser_Qwen3VL_4B_Instruct.sh
#
# Prerequisites:
#   - Model weights at ${MODEL_ROOT}/Qwen3-VL-4B-Instruct (or HF_TOKEN for download)
#   - WebVoyager data: openwebrl/data/webvoyager_{train,val}.parquet
#   - Browser env configured in openwebrl/env/config.yaml
#   - OPENAI_API_KEY and OPENAI_API_BASE set for LLM judge reward
#   - Megatron-LM installed at /root/Megatron-LM/
# ==============================================================================

set -ex

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ==============================================================================
# 1. Configuration
# ==============================================================================
export AZURE_RESOURCE_NAME="${AZURE_RESOURCE_NAME:-}"
export SLIME_BROWSER_SANDBOX_MANIFEST_DIR=/tmp/slime_browser_sandboxes_train
RUN_TIMESTAMP="$(date -u +%Y%m%d_%H%M%S)"



# MODEL_NAME="Qwen3-VL-4B-Thinking"
MODEL_NAME="OpenWebRL/OpenWebRL-4B-SFT"


MODEL_ROOT="${SLIME_MODEL_ROOT:-${REPO_ROOT}/models}"
SAVE_ROOT="${SLIME_SAVE_ROOT:-${REPO_ROOT}/outputs}"
# DEFAULT_SAVE_DIR="${SAVE_ROOT}/qwen3-vl-4b-thinking_browser_rl_turn_bs48_0329_gpt4.1reward_maxstep20_img1_fixbs256_ppoepoch2_fixmultimodalinput_${RUN_TIMESTAMP}_no-0.1_fixactions_temperature0.8_fromSFT912_epoch3_trainstep30fromckpt59"
DEFAULT_SAVE_DIR="${SAVE_ROOT}/qwen3-vl-4b-thinking_browser_rl_turn_bs48_0329_gpt4.1reward_maxstep20_img1_fixbs256_ppoepoch2_fixmultimodalinput_${RUN_TIMESTAMP}_no-0.1_fixactions_temperature0.8_fromSFT912_epoch3_trainstep30fromckpt59_lr5e-7"
# DEFAULT_SAVE_DIR="${SAVE_ROOT}/qwen3-vl-4b-thinking_browser_rl_turn_bs48_0329_gpt4.1reward_maxstep15_img1_fixbs256_ppoepoch2_fixmultimodalinput_${RUN_TIMESTAMP}_no-0.1_fixactions_temperature0.8_frombase_notoolresponse"
# Resume from:
#   ${SLIME_LOAD_CHECKPOINT}/iter_0000024
# Megatron expects --load to be the checkpoint root; --ckpt-step selects the iter.

SLIME_LOAD_CHECKPOINT="${SLIME_LOAD_CHECKPOINT:-}"
SLIME_CKPT_STEP="${SLIME_CKPT_STEP:-}"
# SLIME_CKPT_STEP="${SLIME_CKPT_STEP:-44}"
# SLIME_LOAD_CHECKPOINT=
# SLIME_CKPT_STEP=


BROWSER_TRAIN_LEVEL="${BROWSER_TRAIN_LEVEL:-turn}" # "turn" or "trajectory"
JUDGE_PROMPT_VARIANT="${JUDGE_PROMPT_VARIANT:-action_history}" # action_history
JUDGE_MODEL="${JUDGE_MODEL:-gpt-4.1}"
JUDGE_API_MODE="${JUDGE_API_MODE:-token}"
SLIME_BROWSER_SANDBOX_MAX_SANDBOXES=90
SLIME_BROWSER_ENV_MODE="${SLIME_BROWSER_ENV_MODE:-sandbox}" # sandbox, local_process
PLAYWRIGHT_BROWSERS_PATH="${PLAYWRIGHT_BROWSERS_PATH:-/root/.cache/ms-playwright}"
SLIME_BROWSER_LOCAL_PROCESS_HOST="${SLIME_BROWSER_LOCAL_PROCESS_HOST:-127.0.0.1}"
SLIME_BROWSER_LOCAL_PROCESS_PORT_START="${SLIME_BROWSER_LOCAL_PROCESS_PORT_START:-18000}"
SLIME_BROWSER_LOCAL_PROCESS_PORT_END="${SLIME_BROWSER_LOCAL_PROCESS_PORT_END:-18999}"
SLIME_BROWSER_LOCAL_PROCESS_MAX_PROCESSES="${SLIME_BROWSER_LOCAL_PROCESS_MAX_PROCESSES:-16}"
SLIME_BROWSER_LOCAL_PROCESS_ACQUIRE_TIMEOUT_SECS="${SLIME_BROWSER_LOCAL_PROCESS_ACQUIRE_TIMEOUT_SECS:-600}"
SLIME_BROWSER_LOCAL_PROCESS_STARTUP_TIMEOUT_SECS="${SLIME_BROWSER_LOCAL_PROCESS_STARTUP_TIMEOUT_SECS:-60}"
SLIME_BROWSER_LOCAL_PROCESS_REQUEST_TIMEOUT_SECS="${SLIME_BROWSER_LOCAL_PROCESS_REQUEST_TIMEOUT_SECS:-300}"
SLIME_BROWSER_LOCAL_PROCESS_EXIT_TIMEOUT_SECS="${SLIME_BROWSER_LOCAL_PROCESS_EXIT_TIMEOUT_SECS:-10}"
SLIME_BROWSER_LOCAL_PROCESS_KILL_TIMEOUT_SECS="${SLIME_BROWSER_LOCAL_PROCESS_KILL_TIMEOUT_SECS:-5}"
SLIME_BROWSER_LOCAL_PROCESS_LOG_DIR="${SLIME_BROWSER_LOCAL_PROCESS_LOG_DIR:-/tmp/slime_browser_local_process_env_logs_train}"
SLIME_BROWSER_LOCAL_PROCESS_PORT_LOCK_DIR="${SLIME_BROWSER_LOCAL_PROCESS_PORT_LOCK_DIR:-/tmp/slime_browser_local_process_ports_train}"
SLIME_BROWSER_LOCAL_PROCESS_PYTHON="${SLIME_BROWSER_LOCAL_PROCESS_PYTHON:-$(command -v python)}"
# JUDGE_API_MODE="${JUDGE_API_MODE:-served}" # token, served, api_key
JUDGE_API_BASE="${JUDGE_API_BASE:-}"
# JUDGE_MODEL="${JUDGE_MODEL:-gpt-4.1}"
# JUDGE_API_MODE="${JUDGE_API_MODE:-token}" 
# JUDGE_API_BASE="${JUDGE_API_BASE:-}"
JUDGE_API_HOST="${JUDGE_API_HOST:-}"
JUDGE_API_PORT="${JUDGE_API_PORT:-}"
# JUDGE_API_MODE="${JUDGE_API_MODE:-served}" # token, served, api_key
# JUDGE_API_BASE="${JUDGE_API_BASE:-http://127.0.0.1:30000/v1}"


CONTEXT_NUM_SCREENSHOTS="${CONTEXT_NUM_SCREENSHOTS:-1}"
JUDGE_MAX_ATTACHED_IMGS="${JUDGE_MAX_ATTACHED_IMGS:-3}"
TURN_HISTORY_REASONING_MODE="${TURN_HISTORY_REASONING_MODE:-full}" # hide_thinking， full
BROWSER_HISTORY_REASONING_MAX_TURNS="${BROWSER_HISTORY_REASONING_MAX_TURNS:-0}" # 0 disables reasoning-window compression
BROWSER_RESPONSE_FORMAT_MODE="${BROWSER_RESPONSE_FORMAT_MODE:-browser_env}" ############### slime
BROWSER_INCLUDE_TOOL_RESPONSE="${BROWSER_INCLUDE_TOOL_RESPONSE:-1}" ########### no tool response in LLM input

BROWSER_MAX_STEPS="${BROWSER_MAX_STEPS:-30}"
ROLLOUT_HEALTH_CHECK_INTERVAL="${ROLLOUT_HEALTH_CHECK_INTERVAL:-30}"
ROLLOUT_HEALTH_CHECK_TIMEOUT="${ROLLOUT_HEALTH_CHECK_TIMEOUT:-30}"
ROLLOUT_HEALTH_CHECK_FIRST_WAIT="${ROLLOUT_HEALTH_CHECK_FIRST_WAIT:-180}"
SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-0.6}"
SGLANG_VLM_CACHE_SIZE_MB="${SGLANG_VLM_CACHE_SIZE_MB:-4096}"
HF_VISION_ATTN_IMPL="${SLIME_HF_VISION_ATTN_IMPL:-flash_attention_2}"
PPO_EPOCHS="${PPO_EPOCHS:-2}" ##############################################################################2 
TURN_LEVEL_LOSS_WEIGHT_BY_NUM_TURNS="${TURN_LEVEL_LOSS_WEIGHT_BY_NUM_TURNS:-0}"
ADAPTIVE_QUERY_SAMPLING="${ADAPTIVE_QUERY_SAMPLING:-0}"
ADAPTIVE_QUERY_WARMUP_SAMPLES="${ADAPTIVE_QUERY_WARMUP_SAMPLES:-3}"
ADAPTIVE_QUERY_WEIGHT_ALL_FAIL="${ADAPTIVE_QUERY_WEIGHT_ALL_FAIL:-0.5}"
ADAPTIVE_QUERY_WEIGHT_LOW="${ADAPTIVE_QUERY_WEIGHT_LOW:-1.0}"
ADAPTIVE_QUERY_WEIGHT_MID="${ADAPTIVE_QUERY_WEIGHT_MID:-1.5}"
ADAPTIVE_QUERY_WEIGHT_HIGH="${ADAPTIVE_QUERY_WEIGHT_HIGH:-1.0}"
ADAPTIVE_QUERY_WEIGHT_ALL_SUCCESS="${ADAPTIVE_QUERY_WEIGHT_ALL_SUCCESS:-0.5}"
ADAPTIVE_QUERY_STALE_CAP="${ADAPTIVE_QUERY_STALE_CAP:-10}"
ADAPTIVE_QUERY_STALE_ALPHA="${ADAPTIVE_QUERY_STALE_ALPHA:-1.0}"
ADAPTIVE_QUERY_LENGTH_ALPHA="${ADAPTIVE_QUERY_LENGTH_ALPHA:-0.0}" # not used in current experiments, as we keep response length fixed; set >0 to ramp up length over time
ADAPTIVE_QUERY_LENGTH_RAMP_ROLLOUTS="${ADAPTIVE_QUERY_LENGTH_RAMP_ROLLOUTS:-30}"
ADAPTIVE_QUERY_LENGTH_CONFIDENCE_K="${ADAPTIVE_QUERY_LENGTH_CONFIDENCE_K:-3.0}"
ADAPTIVE_QUERY_LENGTH_CAP="${ADAPTIVE_QUERY_LENGTH_CAP:-2.0}"
SLIME_BROWSER_DEBUG_TRACE_PROB="${SLIME_BROWSER_DEBUG_TRACE_PROB:-0.05}"
SLIME_BROWSER_LOG_LLM_OUTPUT="${SLIME_BROWSER_LOG_LLM_OUTPUT:-1.0}"
SLIME_BROWSER_LOG_LLM_OUTPUT_PROB="${SLIME_BROWSER_LOG_LLM_OUTPUT_PROB:-0.05}"
SLIME_BROWSER_LOG_JUDGE_OUTPUT="${SLIME_BROWSER_LOG_JUDGE_OUTPUT:-1.0}"
SLIME_BROWSER_LOG_JUDGE_OUTPUT_PROB="${SLIME_BROWSER_LOG_JUDGE_OUTPUT_PROB:-0.5}"
SLIME_BROWSER_CHAT_TEMPLATE_ENABLE_THINKING="${SLIME_BROWSER_CHAT_TEMPLATE_ENABLE_THINKING:-1}"
# SLIME_BROWSER_THINKING_OPEN_TAG="${SLIME_BROWSER_THINKING_OPEN_TAG:-<thinking>}"
# SLIME_BROWSER_THINKING_CLOSE_TAG="${SLIME_BROWSER_THINKING_CLOSE_TAG:-</thinking>}"
SLIME_BROWSER_THINKING_OPEN_TAG="${SLIME_BROWSER_THINKING_OPEN_TAG:-<think>}"
SLIME_BROWSER_THINKING_CLOSE_TAG="${SLIME_BROWSER_THINKING_CLOSE_TAG:-</think>}"
SLIME_BROWSER_APPEND_THINKING_PREFILL="${SLIME_BROWSER_APPEND_THINKING_PREFILL:-0}"

if [ "${BROWSER_TRAIN_LEVEL}" = "turn" ]; then
  GENERATE_FN="openwebrl.generate_browser.generate_turn_sample"
else
  GENERATE_FN="openwebrl.generate_browser.generate_trajectory_sample"
fi


NUM_GPUS=8
HF_CHECKPOINT="${MODEL_ROOT}/${MODEL_NAME}"
HF_REPO="${MODEL_NAME}"
if [[ "${MODEL_NAME}" != */* ]]; then
  HF_REPO="Qwen/${MODEL_NAME}"
fi
if [ -n "${SLIME_LOAD_CHECKPOINT}" ]; then
  LOAD_CHECKPOINT="${SLIME_LOAD_CHECKPOINT}"
else
  LOAD_CHECKPOINT="${HF_CHECKPOINT}"
fi
EXTERNAL_RAY="${SLIME_SCRIPT_EXTERNAL_RAY:-0}"

TRAIN_DATA="${TRAIN_DATA:-openwebrl/data/webgym_filtered_popular_2102_cleaned.parquet}"
# TRAIN_DATA="${TRAIN_DATA:-openwebrl/data/webvoyager_train_clean.parquet}"
EVAL_DATA="${EVAL_DATA:-openwebrl/data/webvoyager_val.parquet}"

echo "MODEL_NAME:    ${MODEL_NAME}"
echo "NUM_GPUS:      ${NUM_GPUS}"
echo "HF_CHECKPOINT: ${HF_CHECKPOINT}"
echo "LOAD_CHECKPOINT: ${LOAD_CHECKPOINT}"
echo "SLIME_CKPT_STEP: ${SLIME_CKPT_STEP}"
echo "JUDGE_MODEL: ${JUDGE_MODEL}"
echo "JUDGE_API_MODE: ${JUDGE_API_MODE}"
echo "JUDGE_API_BASE: ${JUDGE_API_BASE}"
echo "JUDGE_API_HOST: ${JUDGE_API_HOST}"
echo "JUDGE_API_PORT: ${JUDGE_API_PORT}"
echo "JUDGE_PROMPT_VARIANT: ${JUDGE_PROMPT_VARIANT}"
echo "SLIME_BROWSER_ENV_MODE: ${SLIME_BROWSER_ENV_MODE}"
echo "PLAYWRIGHT_BROWSERS_PATH: ${PLAYWRIGHT_BROWSERS_PATH}"
echo "SLIME_BROWSER_LOCAL_PROCESS_MAX_PROCESSES: ${SLIME_BROWSER_LOCAL_PROCESS_MAX_PROCESSES}"
echo "SLIME_BROWSER_LOCAL_PROCESS_PORT_RANGE: ${SLIME_BROWSER_LOCAL_PROCESS_PORT_START}-${SLIME_BROWSER_LOCAL_PROCESS_PORT_END}"
echo "SLIME_BROWSER_LOCAL_PROCESS_LOG_DIR: ${SLIME_BROWSER_LOCAL_PROCESS_LOG_DIR}"
echo "CONTEXT_NUM_SCREENSHOTS: ${CONTEXT_NUM_SCREENSHOTS}"
echo "JUDGE_MAX_ATTACHED_IMGS: ${JUDGE_MAX_ATTACHED_IMGS}"
echo "TURN_HISTORY_REASONING_MODE: ${TURN_HISTORY_REASONING_MODE}"
echo "BROWSER_HISTORY_REASONING_MAX_TURNS: ${BROWSER_HISTORY_REASONING_MAX_TURNS}"
echo "BROWSER_RESPONSE_FORMAT_MODE: ${BROWSER_RESPONSE_FORMAT_MODE}"
echo "BROWSER_INCLUDE_TOOL_RESPONSE: ${BROWSER_INCLUDE_TOOL_RESPONSE}"
echo "BROWSER_MAX_STEPS: ${BROWSER_MAX_STEPS}"
echo "TURN_LEVEL_LOSS_WEIGHT_BY_NUM_TURNS: ${TURN_LEVEL_LOSS_WEIGHT_BY_NUM_TURNS}"
echo "ROLLOUT_HEALTH_CHECK_INTERVAL: ${ROLLOUT_HEALTH_CHECK_INTERVAL}"
echo "ROLLOUT_HEALTH_CHECK_TIMEOUT: ${ROLLOUT_HEALTH_CHECK_TIMEOUT}"
echo "ROLLOUT_HEALTH_CHECK_FIRST_WAIT: ${ROLLOUT_HEALTH_CHECK_FIRST_WAIT}"
echo "SGLANG_MEM_FRACTION_STATIC: ${SGLANG_MEM_FRACTION_STATIC}"
echo "SGLANG_VLM_CACHE_SIZE_MB: ${SGLANG_VLM_CACHE_SIZE_MB}"
echo "SLIME_HF_VISION_ATTN_IMPL: ${HF_VISION_ATTN_IMPL}"
echo "PPO_EPOCHS: ${PPO_EPOCHS}"
echo "ADAPTIVE_QUERY_SAMPLING: ${ADAPTIVE_QUERY_SAMPLING}"
echo "SLIME_BROWSER_DEBUG_TRACE_PROB: ${SLIME_BROWSER_DEBUG_TRACE_PROB}"
echo "SLIME_BROWSER_LOG_LLM_OUTPUT: ${SLIME_BROWSER_LOG_LLM_OUTPUT}"
echo "SLIME_BROWSER_LOG_LLM_OUTPUT_PROB: ${SLIME_BROWSER_LOG_LLM_OUTPUT_PROB}"
echo "SLIME_BROWSER_LOG_JUDGE_OUTPUT: ${SLIME_BROWSER_LOG_JUDGE_OUTPUT}"
echo "SLIME_BROWSER_LOG_JUDGE_OUTPUT_PROB: ${SLIME_BROWSER_LOG_JUDGE_OUTPUT_PROB}"
echo "SLIME_BROWSER_CHAT_TEMPLATE_ENABLE_THINKING: ${SLIME_BROWSER_CHAT_TEMPLATE_ENABLE_THINKING}"
echo "SLIME_BROWSER_THINKING_OPEN_TAG: ${SLIME_BROWSER_THINKING_OPEN_TAG}"
echo "SLIME_BROWSER_THINKING_CLOSE_TAG: ${SLIME_BROWSER_THINKING_CLOSE_TAG}"
echo "SLIME_BROWSER_APPEND_THINKING_PREFILL: ${SLIME_BROWSER_APPEND_THINKING_PREFILL}"
export BROWSER_ENV_SERVER_ACCESS_LOG=0
export SLIME_ROUTER_ACCESS_LOG=0
# ==============================================================================
# 2. Prepare: download model & verify data
# ==============================================================================

if [ ! -d "${HF_CHECKPOINT}" ]; then
  mkdir -p "${MODEL_ROOT}"
  huggingface-cli download "${HF_REPO}" --local-dir "${HF_CHECKPOINT}" || true
fi

if [ "${LOAD_CHECKPOINT}" != "${HF_CHECKPOINT}" ] && [ ! -f "${LOAD_CHECKPOINT}/latest_checkpointed_iteration.txt" ]; then
  echo "ERROR: ${LOAD_CHECKPOINT} is not a valid Megatron checkpoint root (missing latest_checkpointed_iteration.txt)"
  exit 1
fi

if [ -n "${SLIME_CKPT_STEP}" ]; then
  CKPT_ITER_DIR=$(printf "iter_%07d" "${SLIME_CKPT_STEP}")
  if [ ! -d "${LOAD_CHECKPOINT}/${CKPT_ITER_DIR}" ]; then
    echo "ERROR: checkpoint step ${SLIME_CKPT_STEP} not found at ${LOAD_CHECKPOINT}/${CKPT_ITER_DIR}"
    exit 1
  fi
fi

[ -f "${TRAIN_DATA}" ] || { echo "ERROR: ${TRAIN_DATA} not found"; exit 1; }
[ -f "${EVAL_DATA}" ] || { echo "ERROR: ${EVAL_DATA} not found"; exit 1; }

# ==============================================================================
# 3. Environment setup
# ==============================================================================

export SGLANG_VLM_CACHE_SIZE_MB         # browser screenshots can overflow small default caches
export PYTHONBUFFERED=16
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export SLIME_HF_VISION_ATTN_IMPL="${HF_VISION_ATTN_IMPL}"
export JUDGE_API_BASE
export JUDGE_API_HOST
export JUDGE_API_PORT
export SLIME_BROWSER_ENV_MODE
export PLAYWRIGHT_BROWSERS_PATH
export SLIME_BROWSER_LOCAL_PROCESS_HOST
export SLIME_BROWSER_LOCAL_PROCESS_PORT_START
export SLIME_BROWSER_LOCAL_PROCESS_PORT_END
export SLIME_BROWSER_LOCAL_PROCESS_MAX_PROCESSES
export SLIME_BROWSER_LOCAL_PROCESS_ACQUIRE_TIMEOUT_SECS
export SLIME_BROWSER_LOCAL_PROCESS_STARTUP_TIMEOUT_SECS
export SLIME_BROWSER_LOCAL_PROCESS_REQUEST_TIMEOUT_SECS
export SLIME_BROWSER_LOCAL_PROCESS_EXIT_TIMEOUT_SECS
export SLIME_BROWSER_LOCAL_PROCESS_KILL_TIMEOUT_SECS
export SLIME_BROWSER_LOCAL_PROCESS_LOG_DIR
export SLIME_BROWSER_LOCAL_PROCESS_PORT_LOCK_DIR
export SLIME_BROWSER_LOCAL_PROCESS_PYTHON
export SLIME_BROWSER_SANDBOX_MAX_SANDBOXES="${SLIME_BROWSER_SANDBOX_MAX_SANDBOXES}"
export SLIME_BROWSER_SANDBOX_ACQUIRE_TIMEOUT_SECS="${SLIME_BROWSER_SANDBOX_ACQUIRE_TIMEOUT_SECS:-600}"
export SLIME_BROWSER_SANDBOX_CLIENT_CONN_LIMIT="${SLIME_BROWSER_SANDBOX_MAX_SANDBOXES}"
export SLIME_BROWSER_SANDBOX_CLIENT_CONN_LIMIT_PER_HOST="${SLIME_BROWSER_SANDBOX_MAX_SANDBOXES}"
export SLIME_BROWSER_SANDBOX_CLIENT_KEEPALIVE_TIMEOUT="${SLIME_BROWSER_SANDBOX_CLIENT_KEEPALIVE_TIMEOUT:-15}"
export SLIME_BROWSER_STEP_CURL_TIMEOUT="${SLIME_BROWSER_STEP_CURL_TIMEOUT:-45}"
export SLIME_BROWSER_STEP_MAX_RETRIES="${SLIME_BROWSER_STEP_MAX_RETRIES:-2}"
export SLIME_BROWSER_STEP_RETRY_DELAY_SECS="${SLIME_BROWSER_STEP_RETRY_DELAY_SECS:-2}"
export SLIME_BROWSER_ENV_EXIT_TIMEOUT_SECS="${SLIME_BROWSER_ENV_EXIT_TIMEOUT_SECS:-15}"
export SLIME_ROLLOUT_ABORT_WAIT_TIMEOUT_SECS="${SLIME_ROLLOUT_ABORT_WAIT_TIMEOUT_SECS:-30}"
export SLIME_BROWSER_ROLLOUT_TASK_TIMEOUT_SECS="${SLIME_BROWSER_ROLLOUT_TASK_TIMEOUT_SECS:-600}"
export SLIME_BROWSER_DEBUG_TRACE_PROB
export SLIME_BROWSER_LOG_LLM_OUTPUT
export SLIME_BROWSER_LOG_LLM_OUTPUT_PROB
export SLIME_BROWSER_LOG_JUDGE_OUTPUT
export SLIME_BROWSER_LOG_JUDGE_OUTPUT_PROB
export SLIME_BROWSER_CHAT_TEMPLATE_ENABLE_THINKING
export SLIME_BROWSER_THINKING_OPEN_TAG
export SLIME_BROWSER_THINKING_CLOSE_TAG
export SLIME_BROWSER_APPEND_THINKING_PREFILL

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
HAS_NVLINK=$([ "$NVLINK_COUNT" -gt 0 ] && echo 1 || echo 0)

echo "SANDBOX_MAX_SANDBOXES: ${SLIME_BROWSER_SANDBOX_MAX_SANDBOXES}"
echo "SANDBOX_ACQUIRE_TIMEOUT_SECS: ${SLIME_BROWSER_SANDBOX_ACQUIRE_TIMEOUT_SECS}"
echo "SANDBOX_CLIENT_CONN_LIMIT: ${SLIME_BROWSER_SANDBOX_CLIENT_CONN_LIMIT}"
echo "SANDBOX_CLIENT_CONN_LIMIT_PER_HOST: ${SLIME_BROWSER_SANDBOX_CLIENT_CONN_LIMIT_PER_HOST}"
echo "SANDBOX_CLIENT_KEEPALIVE_TIMEOUT: ${SLIME_BROWSER_SANDBOX_CLIENT_KEEPALIVE_TIMEOUT}"
echo "STEP_CURL_TIMEOUT: ${SLIME_BROWSER_STEP_CURL_TIMEOUT}"
echo "STEP_MAX_RETRIES: ${SLIME_BROWSER_STEP_MAX_RETRIES}"
echo "STEP_RETRY_DELAY_SECS: ${SLIME_BROWSER_STEP_RETRY_DELAY_SECS}"
echo "ENV_EXIT_TIMEOUT_SECS: ${SLIME_BROWSER_ENV_EXIT_TIMEOUT_SECS}"
echo "ROLLOUT_ABORT_WAIT_TIMEOUT_SECS: ${SLIME_ROLLOUT_ABORT_WAIT_TIMEOUT_SECS}"
echo "ROLLOUT_TASK_TIMEOUT_SECS: ${SLIME_BROWSER_ROLLOUT_TASK_TIMEOUT_SECS}"


SAVE_DIR="${SLIME_SAVE_DIR:-${DEFAULT_SAVE_DIR}}"
echo "SAVE_DIR: ${SAVE_DIR}"
mkdir -p "${SAVE_DIR}"
WANDB_LOCAL_DIR="${WANDB_DIR:-${SAVE_DIR}/wandb}"
mkdir -p "${WANDB_LOCAL_DIR}"
echo "WANDB_LOCAL_DIR: ${WANDB_LOCAL_DIR}"
WANDB_RUN_NAME="${WANDB_NAME:-${SAVE_DIR#${SAVE_ROOT}/}}"
echo "WANDB_RUN_NAME: ${WANDB_RUN_NAME}"
WANDB_GROUP="${WANDB_GROUP:-${WANDB_RUN_NAME}}"
if [ "${#WANDB_GROUP}" -gt 128 ]; then
    WANDB_GROUP="${WANDB_GROUP:0:128}"
fi
echo "WANDB_GROUP: ${WANDB_GROUP}"

# ==============================================================================
# 4. Kill stale processes & start ray
# ==============================================================================

KILL_STALE_SGLANG="${KILL_STALE_SGLANG:-0}"  # Set to 1 when no external judge server is running.
if [ "${KILL_STALE_SGLANG}" = "1" ]; then
    pkill -9 sglang || true
    sleep 3
else
    echo "Skipping pkill -9 sglang because KILL_STALE_SGLANG=${KILL_STALE_SGLANG}; this preserves served judge processes."
fi
if [ "${EXTERNAL_RAY}" != "1" ]; then
    ray stop --force || true; pkill -9 ray || true
fi
pkill -9 slime || true; sleep 3
if [ "${EXTERNAL_RAY}" != "1" ]; then pkill -9 ray || true; fi
pkill -9 slime || true; pkill -9 redis || true

if [ "${EXTERNAL_RAY}" != "1" ]; then
    ray start --head \
        --node-ip-address "${MASTER_ADDR}" \
        --num-gpus ${NUM_GPUS} \
        --disable-usage-stats
fi

# ==============================================================================
# 5. Training arguments
# ==============================================================================

# ── Checkpoint ───────────────────────────────────────────────────────────────
CKPT_ARGS=(
    --hf-checkpoint "${HF_CHECKPOINT}"
    --save "${SAVE_DIR}"
    --save-interval 5
    --async-save
)

# ── Rollout (browser-specific) ───────────────────────────────────────────────
# No --label-key: reward from LLM judge, not ground-truth.
# No --apply-chat-template: prompt is a messages list; generate_browser.py
# rebuilds the full chat from task metadata at rollout time.
ROLLOUT_ARGS=(
    --data-source-path "${REPO_ROOT}/slime/rollout/data_source.py:RolloutDataSourceWithBuffer"
    --prompt-data "${TRAIN_DATA}"
    --input-key prompt
    --rollout-shuffle
    --use-fault-tolerance
    # Custom browser generate / reward / config
    --custom-generate-function-path ${GENERATE_FN}
    --custom-rm-path openwebrl.reward_browser.reward_func
    --custom-config-path openwebrl/browser_training_config.yaml
    --max-steps "${BROWSER_MAX_STEPS}"
    --context-num-screenshots "${CONTEXT_NUM_SCREENSHOTS}"
    --judge-max-attached-imgs "${JUDGE_MAX_ATTACHED_IMGS}"
    --judge-api-model "${JUDGE_MODEL}"
    --judge-api-mode "${JUDGE_API_MODE}"
    --judge-prompt-variant "${JUDGE_PROMPT_VARIANT}"
    --turn-history-reasoning-mode "${TURN_HISTORY_REASONING_MODE}"
    --browser-history-reasoning-max-turns "${BROWSER_HISTORY_REASONING_MAX_TURNS}"
    --browser-response-format-mode "${BROWSER_RESPONSE_FORMAT_MODE}"
    --browser-include-tool-response "${BROWSER_INCLUDE_TOOL_RESPONSE}"
    --dynamic-sampling-filter-path slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonempty_nonzero_std
    --rollout-health-check-interval "${ROLLOUT_HEALTH_CHECK_INTERVAL}"
    --rollout-health-check-timeout "${ROLLOUT_HEALTH_CHECK_TIMEOUT}"
    --rollout-health-check-first-wait "${ROLLOUT_HEALTH_CHECK_FIRST_WAIT}"
    # Hyper-params
    --num-rollout 100                       # 100 × batch_size32 = 3200 prompts 
    --rollout-batch-size 48
    --n-samples-per-prompt 5
    --rollout-max-response-len 1024
    --rollout-max-context-len 32768         # headroom for long browser trajectories
    --rollout-temperature 0.8 # 1.0
    # --use-dynamic-global-batch-size # 1520 turn samples,  (1520 // 8) * 8 
    --global-batch-size 256       # (1520 // 256) * 256   
)

if [ "${ADAPTIVE_QUERY_SAMPLING}" = "1" ]; then
    ROLLOUT_ARGS+=(
        --enable-adaptive-query-sampling
        --adaptive-query-warmup-samples "${ADAPTIVE_QUERY_WARMUP_SAMPLES}"
        --adaptive-query-weight-all-fail "${ADAPTIVE_QUERY_WEIGHT_ALL_FAIL}"
        --adaptive-query-weight-low "${ADAPTIVE_QUERY_WEIGHT_LOW}"
        --adaptive-query-weight-mid "${ADAPTIVE_QUERY_WEIGHT_MID}"
        --adaptive-query-weight-high "${ADAPTIVE_QUERY_WEIGHT_HIGH}"
        --adaptive-query-weight-all-success "${ADAPTIVE_QUERY_WEIGHT_ALL_SUCCESS}"
        --adaptive-query-stale-cap "${ADAPTIVE_QUERY_STALE_CAP}"
        --adaptive-query-stale-alpha "${ADAPTIVE_QUERY_STALE_ALPHA}"
        --adaptive-query-length-alpha "${ADAPTIVE_QUERY_LENGTH_ALPHA}"
        --adaptive-query-length-ramp-rollouts "${ADAPTIVE_QUERY_LENGTH_RAMP_ROLLOUTS}"
        --adaptive-query-length-confidence-k "${ADAPTIVE_QUERY_LENGTH_CONFIDENCE_K}"
        --adaptive-query-length-cap "${ADAPTIVE_QUERY_LENGTH_CAP}"
    )
fi

# ── Eval ─────────────────────────────────────────────────────────────────────
EVAL_ARGS=(
    --eval-interval 5
    --eval-prompt-data webvoyager "${EVAL_DATA}"
    --n-samples-per-eval-prompt 1
    --eval-temperature 0.0
    --eval-top-p 1.0
    --eval-top-k 1
    --eval-max-response-len 1024
)

# ── GRPO ─────────────────────────────────────────────────────────────────────
GRPO_ARGS=(
    --advantage-estimator grpo
    --ppo-epochs "${PPO_EPOCHS}"
    --kl-loss-coef 0.00
    --kl-loss-type low_var_kl
    --kl-coef 0.00
    --entropy-coef 0.00
    --eps-clip 0.2
    --eps-clip-high 0.28
)

if [ "${TURN_LEVEL_LOSS_WEIGHT_BY_NUM_TURNS}" = "1" ]; then
    GRPO_ARGS+=(
        --turn-level-loss-weight-by-num-turns
    )
fi

if [ "${PPO_EPOCHS}" -gt 1 ]; then
    GRPO_ARGS+=(
        --use-rollout-logprobs
    )
fi

# ── Optimizer ────────────────────────────────────────────────────────────────
OPTIMIZER_ARGS=(
    --optimizer adam
    --lr 5e-7  ###########################1e-6
    --lr-decay-style constant
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.98
    --optimizer-cpu-offload
    --use-precision-aware-optimizer
# 4B is small enough, no CPU offload needed on 180GB B200 with Megatron TP=2
)

# ── SGLang rollout engine ────────────────────────────────────────────────────
# 4B dense fits on 1 GPU comfortably, keep 1 GPU per engine for throughput
# SGLANG_ARGS=(
#     --rollout-num-gpus ${NUM_GPUS}
    # --rollout-num-gpus-per-engine 1         # 4B fits on 1 GPU → up to 8 engines
    # --sglang-mem-fraction-static 0.55       # 180GB × 0.55 ≈ 99GB KV cache per GPU
    # --sglang-cuda-graph-bs 1 2 4 8 $(seq 16 8 256)
# )
SGLANG_ARGS=(
   --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC}"
#    --rollout-num-gpus 2
   --rollout-num-gpus-per-engine 1
#    --use-slime-router
   --sglang-log-level warning
   --sglang-log-level-http warning
   --sglang-server-concurrency 48
   --sglang-max-running-requests 48
#    --sglang-chunked-prefill-size 4096
   --sglang-chunked-prefill-size 8192
)

# ── Megatron backend ─────────────────────────────────────────────────────────
BACKEND_ARGS=(
    --train-backend megatron
    --load "${LOAD_CHECKPOINT}"
    --tensor-model-parallel-size 4
    --sequence-parallel
    --pipeline-model-parallel-size 1
    --context-parallel-size 1
    --expert-model-parallel-size 1
    --expert-tensor-parallel-size 1
    # --recompute-granularity full
    # --recompute-method uniform
    # --recompute-num-layers 1
    --micro-batch-size 1
    --attention-dropout 0.0
    --hidden-dropout 0.0
    # --accumulate-allreduce-grads-in-fp32
    # --attention-softmax-in-fp32
    --attention-backend flash
    --megatron-to-hf-mode bridge
    --moe-token-dispatcher-type alltoall
        # --use-dynamic-batch-size
    # --max-tokens-per-gpu 8192
)

if [ -n "${SLIME_CKPT_STEP}" ]; then
    BACKEND_ARGS+=(
        --ckpt-step "${SLIME_CKPT_STEP}"
    )
fi

# ── Wandb (optional) ────────────────────────────────────────────────────────
WANDB_ENTITY="${WANDB_ENTITY:-yangrui_rl}"
WANDB_ARGS=()
if [ -n "${WANDB_API_KEY}" ]; then
    WANDB_ARGS=(
        --use-wandb
        --wandb-project slime-dev
        --wandb-group "${WANDB_GROUP}"
        --wandb-dir "${WANDB_LOCAL_DIR}"
        --wandb-key "${WANDB_API_KEY}"
        --disable-wandb-random-suffix
    )
    [ -n "${WANDB_ENTITY}" ] && WANDB_ARGS+=(--wandb-team "${WANDB_ENTITY}")
fi

# ── Misc ─────────────────────────────────────────────────────────────────────
MISC_ARGS=(
    --actor-num-nodes 1
    --actor-num-gpus-per-node ${NUM_GPUS}
    --colocate
)

# ==============================================================================
# 6. Launch
# ==============================================================================

# Megatron model type: Qwen3-VL-4B (OpenWebRL-4B-SFT) → qwen3-4B
# (hidden 2560 / ffn 9728 / 36 layers / tied embeddings — matches qwen3-4B.sh)
MEGATRON_MODEL_TYPE="qwen3-4B"
export MODEL_ARGS_ROTARY_BASE=5000000

# Source the megatron model definition to get ${MODEL_ARGS[@]}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${REPO_ROOT}/scripts/model_configs/${MEGATRON_MODEL_TYPE}.sh"

# Persist adaptive query sampling knobs with each run for reproducibility.
mkdir -p "${SAVE_DIR}/run_config"
cp "${BASH_SOURCE[0]}" "${SAVE_DIR}/run_config/launch_script_snapshot.sh"
(
    set -o posix
    set
) > "${SAVE_DIR}/run_config/shell_vars.env"
cat > "${SAVE_DIR}/run_config/adaptive_query_sampling.env" <<EOF
ADAPTIVE_QUERY_SAMPLING=${ADAPTIVE_QUERY_SAMPLING}
ADAPTIVE_QUERY_WARMUP_SAMPLES=${ADAPTIVE_QUERY_WARMUP_SAMPLES}
ADAPTIVE_QUERY_WEIGHT_ALL_FAIL=${ADAPTIVE_QUERY_WEIGHT_ALL_FAIL}
ADAPTIVE_QUERY_WEIGHT_LOW=${ADAPTIVE_QUERY_WEIGHT_LOW}
ADAPTIVE_QUERY_WEIGHT_MID=${ADAPTIVE_QUERY_WEIGHT_MID}
ADAPTIVE_QUERY_WEIGHT_HIGH=${ADAPTIVE_QUERY_WEIGHT_HIGH}
ADAPTIVE_QUERY_WEIGHT_ALL_SUCCESS=${ADAPTIVE_QUERY_WEIGHT_ALL_SUCCESS}
ADAPTIVE_QUERY_STALE_CAP=${ADAPTIVE_QUERY_STALE_CAP}
ADAPTIVE_QUERY_STALE_ALPHA=${ADAPTIVE_QUERY_STALE_ALPHA}
ADAPTIVE_QUERY_LENGTH_ALPHA=${ADAPTIVE_QUERY_LENGTH_ALPHA}
ADAPTIVE_QUERY_LENGTH_RAMP_ROLLOUTS=${ADAPTIVE_QUERY_LENGTH_RAMP_ROLLOUTS}
ADAPTIVE_QUERY_LENGTH_CONFIDENCE_K=${ADAPTIVE_QUERY_LENGTH_CONFIDENCE_K}
ADAPTIVE_QUERY_LENGTH_CAP=${ADAPTIVE_QUERY_LENGTH_CAP}
EOF

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${REPO_ROOT}:/root/Megatron-LM/\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"no_proxy\": \"127.0.0.1,${MASTER_ADDR}\",
    \"MASTER_ADDR\": \"${MASTER_ADDR}\",
    \"JUDGE_API_BASE\": \"${JUDGE_API_BASE}\",
    \"JUDGE_API_HOST\": \"${JUDGE_API_HOST}\",
    \"JUDGE_API_PORT\": \"${JUDGE_API_PORT}\",
    \"JUDGE_API_KEY\": \"${JUDGE_API_KEY:-}\",
    \"SGLANG_VLM_CACHE_SIZE_MB\": \"${SGLANG_VLM_CACHE_SIZE_MB}\",
    \"SLIME_BROWSER_ENV_MODE\": \"${SLIME_BROWSER_ENV_MODE}\",
    \"PLAYWRIGHT_BROWSERS_PATH\": \"${PLAYWRIGHT_BROWSERS_PATH}\",
    \"SLIME_BROWSER_LOCAL_PROCESS_HOST\": \"${SLIME_BROWSER_LOCAL_PROCESS_HOST}\",
    \"SLIME_BROWSER_LOCAL_PROCESS_PORT_START\": \"${SLIME_BROWSER_LOCAL_PROCESS_PORT_START}\",
    \"SLIME_BROWSER_LOCAL_PROCESS_PORT_END\": \"${SLIME_BROWSER_LOCAL_PROCESS_PORT_END}\",
    \"SLIME_BROWSER_LOCAL_PROCESS_MAX_PROCESSES\": \"${SLIME_BROWSER_LOCAL_PROCESS_MAX_PROCESSES}\",
    \"SLIME_BROWSER_LOCAL_PROCESS_ACQUIRE_TIMEOUT_SECS\": \"${SLIME_BROWSER_LOCAL_PROCESS_ACQUIRE_TIMEOUT_SECS}\",
    \"SLIME_BROWSER_LOCAL_PROCESS_STARTUP_TIMEOUT_SECS\": \"${SLIME_BROWSER_LOCAL_PROCESS_STARTUP_TIMEOUT_SECS}\",
    \"SLIME_BROWSER_LOCAL_PROCESS_REQUEST_TIMEOUT_SECS\": \"${SLIME_BROWSER_LOCAL_PROCESS_REQUEST_TIMEOUT_SECS}\",
    \"SLIME_BROWSER_LOCAL_PROCESS_EXIT_TIMEOUT_SECS\": \"${SLIME_BROWSER_LOCAL_PROCESS_EXIT_TIMEOUT_SECS}\",
    \"SLIME_BROWSER_LOCAL_PROCESS_KILL_TIMEOUT_SECS\": \"${SLIME_BROWSER_LOCAL_PROCESS_KILL_TIMEOUT_SECS}\",
    \"SLIME_BROWSER_LOCAL_PROCESS_LOG_DIR\": \"${SLIME_BROWSER_LOCAL_PROCESS_LOG_DIR}\",
    \"SLIME_BROWSER_LOCAL_PROCESS_PORT_LOCK_DIR\": \"${SLIME_BROWSER_LOCAL_PROCESS_PORT_LOCK_DIR}\",
    \"SLIME_BROWSER_LOCAL_PROCESS_PYTHON\": \"${SLIME_BROWSER_LOCAL_PROCESS_PYTHON}\",
    \"SLIME_BROWSER_SANDBOX_MAX_SANDBOXES\": \"${SLIME_BROWSER_SANDBOX_MAX_SANDBOXES}\",
    \"SLIME_BROWSER_SANDBOX_ACQUIRE_TIMEOUT_SECS\": \"${SLIME_BROWSER_SANDBOX_ACQUIRE_TIMEOUT_SECS}\",
    \"SLIME_BROWSER_SANDBOX_CLIENT_CONN_LIMIT\": \"${SLIME_BROWSER_SANDBOX_CLIENT_CONN_LIMIT}\",
    \"SLIME_BROWSER_SANDBOX_CLIENT_CONN_LIMIT_PER_HOST\": \"${SLIME_BROWSER_SANDBOX_CLIENT_CONN_LIMIT_PER_HOST}\",
    \"SLIME_BROWSER_SANDBOX_CLIENT_KEEPALIVE_TIMEOUT\": \"${SLIME_BROWSER_SANDBOX_CLIENT_KEEPALIVE_TIMEOUT}\"
  }
}"

export no_proxy="127.0.0.1"

ray job submit --address="http://127.0.0.1:8265" \
    --runtime-env-json="${RUNTIME_ENV_JSON}" \
    -- python3 train.py \
    ${MODEL_ARGS[@]} \
    ${CKPT_ARGS[@]} \
    ${ROLLOUT_ARGS[@]} \
    ${EVAL_ARGS[@]} \
    ${GRPO_ARGS[@]} \
    ${OPTIMIZER_ARGS[@]} \
    ${SGLANG_ARGS[@]} \
    ${BACKEND_ARGS[@]} \
    ${MISC_ARGS[@]} \
    ${WANDB_ARGS[@]}

# run the cleanup script after training to kill any remaining processes
bash scripts/clean_processes.sh
