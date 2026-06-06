#!/usr/bin/env bash
# One-command OpenWebRL SFT warm-start workflow.
#
# Default data source:
#   https://huggingface.co/datasets/OpenWebRL/OpenWebRL-SFT-Trajectories
#
# Phases:
#   1. Download the released OpenWebRL SFT trajectories from Hugging Face
#      unless OPENWEBRL_SFT_RAW_DATA points to a local JSONL.
#   2. Convert the raw prompt/response trajectory JSONL into LLaMAFactory's
#      ShareGPT-style multimodal format and extract screenshots to PNG files.
#   3. Generate dataset_info.json and a LLaMAFactory train YAML.
#   4. Optionally launch LLaMAFactory SFT.
#   5. Optionally post-process the checkpoint for serving/OpenWebRL RL init.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPENWEBRL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

LLAMAFACTORY_ROOT="${LLAMAFACTORY_ROOT:-/root/LlamaFactory}"
UV="${UV:-uv}"

HF_DATASET_REPO="${HF_DATASET_REPO:-OpenWebRL/OpenWebRL-SFT-Trajectories}"
HF_DATASET_FILE="${HF_DATASET_FILE:-OpenWebRL_SFT_trajectories.jsonl}"
OPENWEBRL_SFT_RAW_DATA="${OPENWEBRL_SFT_RAW_DATA:-}"

WORK_DIR="${OPENWEBRL_SFT_WORK_DIR:-${OPENWEBRL_ROOT}/outputs/sft/llamafactory}"
DATA_DIR="${OPENWEBRL_SFT_DATA_DIR:-${WORK_DIR}/data}"
RAW_DATA_DIR="${OPENWEBRL_SFT_RAW_DATA_DIR:-${WORK_DIR}/raw}"
DATASET_NAME="${OPENWEBRL_SFT_DATASET_NAME:-openwebrl_sft_trajectories}"
# Model-agnostic canonical OpenAI-format episodes (Stage 1 output). Shared across
# every base model / tool format; built once and cached.
CANONICAL_DATA="${OPENWEBRL_SFT_CANONICAL_DATA:-${DATA_DIR}/openwebrl_sft_openai.jsonl}"

# Model family preset. Picks sensible defaults for the base model, chat
# template, and checkpoint naming. Every value below stays overridable via its
# own environment variable. The SFT data is built once into a model-agnostic
# canonical format (Stage 1), then rendered per model by Stage 2 using that
# model's OFFICIAL chat template (apply_chat_template) — the same call the
# OpenWebRL runtime uses at inference — so each family gets its own native
# tool-calling format automatically.
#   qwen3_vl -> multimodal Qwen3-VL warm start (Qwen3-VL/Hermes JSON tool calls).
#   qwen3_5  -> Qwen3.5 warm start (Qwen3.5 XML function tool calls).
MODEL_FAMILY="${MODEL_FAMILY:-qwen3_vl}"
case "${MODEL_FAMILY}" in
  qwen3_vl)
    DEFAULT_MODEL_NAME_OR_PATH="Qwen/Qwen3-VL-4B-Thinking"
    DEFAULT_TEMPLATE="qwen3_vl"
    DEFAULT_TRAIN_CONFIG_NAME="openwebrl_qwen3_vl_sft.yaml"
    ;;
  qwen3_5)
    DEFAULT_MODEL_NAME_OR_PATH="Qwen/Qwen3.5-9B"
    DEFAULT_TEMPLATE="qwen3_5"
    DEFAULT_TRAIN_CONFIG_NAME="openwebrl_qwen3_5_sft.yaml"
    ;;
  *)
    echo "ERROR: Unknown MODEL_FAMILY='${MODEL_FAMILY}'. Supported: qwen3_vl, qwen3_5." >&2
    exit 1
    ;;
esac

MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-${DEFAULT_MODEL_NAME_OR_PATH}}"
# Name the checkpoint dir after the base model, e.g. Qwen/Qwen3.5-27B -> qwen3.5-27b-openwebrl-sft.
DEFAULT_CKPT_SLUG="$(basename "${MODEL_NAME_OR_PATH}" | tr '[:upper:]' '[:lower:]')-openwebrl-sft"
OUTPUT_DIR="${OUTPUT_DIR:-${WORK_DIR}/checkpoints/${DEFAULT_CKPT_SLUG}}"
RUN_NAME="${RUN_NAME:-${DEFAULT_CKPT_SLUG}}"
# The train config lives inside the checkpoint dir so it travels with the checkpoint.
TRAIN_CONFIG="${OPENWEBRL_SFT_TRAIN_CONFIG:-${OUTPUT_DIR}/${DEFAULT_TRAIN_CONFIG_NAME}}"

TEMPLATE="${TEMPLATE:-${DEFAULT_TEMPLATE}}"
# Granularity: PER_TURN=1 expands each episode into per-turn examples (single
# current screenshot) and masks the history assistant responses (mask_history),
# reproducing the released recipe. PER_TURN=0 keeps full-episode trajectories
# (all screenshots) and supervises every assistant turn in one pass.
PER_TURN="${PER_TURN:-1}"
if [[ "${PER_TURN}" == "1" ]]; then
  GRANULARITY="perturn"
  MASK_HISTORY="${MASK_HISTORY:-true}"
else
  GRANULARITY="trajectory"
  MASK_HISTORY="${MASK_HISTORY:-false}"
fi
# Per-config workspace so different (model template, granularity) renders never
# clobber each other's JSONL / images / dataset_info.json.
PREP_DIR="${OPENWEBRL_SFT_PREP_DIR:-${DATA_DIR}/${TEMPLATE}_${GRANULARITY}}"
PREPARED_DATA="${OPENWEBRL_SFT_PREPARED_DATA:-${PREP_DIR}/openwebrl_sft_llamafactory.jsonl}"
DATASET_INFO_PATH="${PREP_DIR}/dataset_info.json"
CUTOFF_LEN="${CUTOFF_LEN:-36864}"
IMAGE_MAX_PIXELS="${IMAGE_MAX_PIXELS:-262144}"
VIDEO_MAX_PIXELS="${VIDEO_MAX_PIXELS:-16384}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-examples/deepspeed/ds_z2_config.json}"

FINETUNING_TYPE="${FINETUNING_TYPE:-full}"
FREEZE_VISION_TOWER="${FREEZE_VISION_TOWER:-true}"
FREEZE_MULTI_MODAL_PROJECTOR="${FREEZE_MULTI_MODAL_PROJECTOR:-true}"
FREEZE_LANGUAGE_MODEL="${FREEZE_LANGUAGE_MODEL:-false}"

NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-3.0}"
LEARNING_RATE="${LEARNING_RATE:-1.0e-5}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-16}"
LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE:-cosine}"
WARMUP_RATIO="${WARMUP_RATIO:-0.1}"
BF16="${BF16:-true}"
SAVE_STRATEGY="${SAVE_STRATEGY:-epoch}"
# Reporting backend. Defaults to Weights & Biases. When wandb is selected for a
# real training run but no WANDB_API_KEY is available, prompt for one in an
# interactive shell, or fail with instructions in a non-interactive one. Pass
# REPORT_TO=none to opt out.
REPORT_TO="${REPORT_TO:-wandb}"
if [[ "${REPORT_TO}" == "wandb" && -z "${WANDB_API_KEY:-}" && "${RUN_TRAIN:-1}" == "1" ]]; then
  if [[ -t 0 ]]; then
    read -rsp "WANDB_API_KEY is not set. Paste your W&B API key (leave empty to disable wandb): " WANDB_API_KEY || true
    echo
    if [[ -z "${WANDB_API_KEY}" ]]; then
      echo "No key entered; disabling W&B logging (REPORT_TO=none)." >&2
      REPORT_TO="none"
    else
      export WANDB_API_KEY
    fi
  else
    echo "ERROR: REPORT_TO=wandb but WANDB_API_KEY is not set." >&2
    echo "       Export WANDB_API_KEY (or run 'wandb login'), or pass REPORT_TO=none to disable W&B." >&2
    exit 1
  fi
fi
PREPROCESSING_NUM_WORKERS="${PREPROCESSING_NUM_WORKERS:-16}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-4}"
DDP_TIMEOUT="${DDP_TIMEOUT:-180000000}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-null}"

RUN_DATA_PREPARE="${RUN_DATA_PREPARE:-1}"
RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_POST_PROCESS="${RUN_POST_PROCESS:-1}"
POST_PROCESS_INCLUDE_SUBSTEPS="${POST_PROCESS_INCLUDE_SUBSTEPS:-1}"
FIX_MOE_EXPERTS="${FIX_MOE_EXPERTS:-0}"
MAX_ROWS="${MAX_ROWS:-}"

require_file() {
  local path="$1"
  local description="$2"
  if [[ ! -f "${path}" ]]; then
    echo "ERROR: ${description} not found: ${path}" >&2
    exit 1
  fi
}

require_dir() {
  local path="$1"
  local description="$2"
  if [[ ! -d "${path}" ]]; then
    echo "ERROR: ${description} not found: ${path}" >&2
    exit 1
  fi
}

run_llamafactory_python() {
  (cd "${LLAMAFACTORY_ROOT}" && "${UV}" run python "$@")
}

run_llamafactory_cli() {
  (cd "${LLAMAFACTORY_ROOT}" && "${UV}" run llamafactory-cli "$@")
}

mkdir -p "${DATA_DIR}" "${PREP_DIR}" "${RAW_DATA_DIR}" "${OUTPUT_DIR}"

require_dir "${LLAMAFACTORY_ROOT}" "LLaMAFactory root"
if ! command -v "${UV}" >/dev/null 2>&1; then
  echo "ERROR: uv is required. Install uv and set up the LLaMAFactory uv environment first." >&2
  exit 1
fi
require_file "${LLAMAFACTORY_ROOT}/pyproject.toml" "LLaMAFactory pyproject.toml"
require_file "${SCRIPT_DIR}/convert_to_openai_messages.py" "Stage 1 canonical converter"
require_file "${SCRIPT_DIR}/prepare_openai_for_llamafactory.py" "Stage 2 LLaMAFactory prep"
require_file "${SCRIPT_DIR}/post_process_llamafactory_ckpt.py" "checkpoint post-processor"

if [[ -z "${OPENWEBRL_SFT_RAW_DATA}" ]]; then
  echo "Downloading ${HF_DATASET_REPO}/${HF_DATASET_FILE} ..."
  export HF_DATASET_REPO HF_DATASET_FILE RAW_DATA_DIR
  OPENWEBRL_SFT_RAW_DATA="$(
    run_llamafactory_python -c 'import os
from huggingface_hub import hf_hub_download
print(hf_hub_download(
    repo_id=os.environ["HF_DATASET_REPO"],
    filename=os.environ["HF_DATASET_FILE"],
    repo_type="dataset",
    local_dir=os.environ["RAW_DATA_DIR"],
))
'
  )"
else
  echo "Using local SFT trajectory JSONL: ${OPENWEBRL_SFT_RAW_DATA}"
fi

require_file "${OPENWEBRL_SFT_RAW_DATA}" "OpenWebRL SFT trajectory JSONL"

# Stage 1: build the model-agnostic canonical OpenAI-format episodes (cached).
if [[ "${RUN_DATA_PREPARE}" == "1" || ! -f "${CANONICAL_DATA}" ]]; then
  echo "Stage 1: converting trajectories to canonical OpenAI format ..."
  CONVERT_ARGS=(
    "${SCRIPT_DIR}/convert_to_openai_messages.py"
    --input-path "${OPENWEBRL_SFT_RAW_DATA}"
    --output-path "${CANONICAL_DATA}"
  )
  if [[ -n "${MAX_ROWS}" ]]; then
    CONVERT_ARGS+=(--max-rows "${MAX_ROWS}")
  fi
  run_llamafactory_python "${CONVERT_ARGS[@]}"
else
  echo "Skipping Stage 1 because ${CANONICAL_DATA} already exists."
fi
require_file "${CANONICAL_DATA}" "canonical OpenAI-format data"

# Stage 2: render the canonical episodes into LLaMAFactory data for this model
# using the model's official chat template (apply_chat_template) and the chosen
# granularity. Also writes dataset_info.json (formatting: sharegpt) next to the output.
if [[ "${RUN_DATA_PREPARE}" == "1" || ! -f "${PREPARED_DATA}" ]]; then
  echo "Stage 2: rendering LLaMAFactory data via ${MODEL_NAME_OR_PATH} chat template (${GRANULARITY}) ..."
  PREP_ARGS=(
    "${SCRIPT_DIR}/prepare_openai_for_llamafactory.py"
    --input-path "${CANONICAL_DATA}"
    --output-path "${PREPARED_DATA}"
    --model-name-or-path "${MODEL_NAME_OR_PATH}"
    --dataset-name "${DATASET_NAME}"
    --image-path-mode relative
  )
  if [[ "${PER_TURN}" == "1" ]]; then
    PREP_ARGS+=(--per-turn)
  fi
  run_llamafactory_python "${PREP_ARGS[@]}"
else
  echo "Skipping Stage 2 because ${PREPARED_DATA} already exists."
fi

require_file "${PREPARED_DATA}" "prepared LLaMAFactory data"
require_file "${DATASET_INFO_PATH}" "dataset_info.json"

cat > "${TRAIN_CONFIG}" <<EOF
### model
model_name_or_path: ${MODEL_NAME_OR_PATH}
image_max_pixels: ${IMAGE_MAX_PIXELS}
video_max_pixels: ${VIDEO_MAX_PIXELS}
trust_remote_code: true

### method
stage: sft
do_train: true
finetuning_type: ${FINETUNING_TYPE}
freeze_vision_tower: ${FREEZE_VISION_TOWER}
freeze_multi_modal_projector: ${FREEZE_MULTI_MODAL_PROJECTOR}
freeze_language_model: ${FREEZE_LANGUAGE_MODEL}
deepspeed: ${DEEPSPEED_CONFIG}

### dataset
dataset_dir: ${PREP_DIR}
media_dir: ${PREP_DIR}
dataset: ${DATASET_NAME}
template: ${TEMPLATE}
cutoff_len: ${CUTOFF_LEN}
preprocessing_num_workers: ${PREPROCESSING_NUM_WORKERS}
dataloader_num_workers: ${DATALOADER_NUM_WORKERS}
mask_history: ${MASK_HISTORY}

### output
output_dir: ${OUTPUT_DIR}
logging_steps: 1
save_strategy: ${SAVE_STRATEGY}
plot_loss: true
overwrite_output_dir: true
save_only_model: false
report_to: ${REPORT_TO}
run_name: ${RUN_NAME}

### train
num_train_epochs: ${NUM_TRAIN_EPOCHS}
learning_rate: ${LEARNING_RATE}
per_device_train_batch_size: ${PER_DEVICE_TRAIN_BATCH_SIZE}
gradient_accumulation_steps: ${GRADIENT_ACCUMULATION_STEPS}
lr_scheduler_type: ${LR_SCHEDULER_TYPE}
warmup_ratio: ${WARMUP_RATIO}
bf16: ${BF16}
ddp_timeout: ${DDP_TIMEOUT}
resume_from_checkpoint: ${RESUME_FROM_CHECKPOINT}
EOF

echo "Model family: ${MODEL_FAMILY} | template: ${TEMPLATE} | granularity: ${GRANULARITY} (mask_history=${MASK_HISTORY}) | model: ${MODEL_NAME_OR_PATH}"
echo "Generated dataset info: ${DATASET_INFO_PATH}"
echo "Generated train config: ${TRAIN_CONFIG}"

if [[ "${RUN_TRAIN}" == "1" ]]; then
  echo "Launching LLaMAFactory SFT ..."
  run_llamafactory_cli train "${TRAIN_CONFIG}"
else
  echo "RUN_TRAIN=0, skipping LLaMAFactory training."
fi

if [[ "${RUN_POST_PROCESS}" == "1" ]]; then
  echo "Post-processing SFT checkpoint for serving/OpenWebRL RL initialization ..."
  POST_ARGS=(
    "${SCRIPT_DIR}/post_process_llamafactory_ckpt.py"
    --ckpt-path "${OUTPUT_DIR}"
    --original-model-path "${MODEL_NAME_OR_PATH}"
  )
  if [[ "${POST_PROCESS_INCLUDE_SUBSTEPS}" == "1" ]]; then
    POST_ARGS+=(--include-substeps)
  fi
  if [[ "${FIX_MOE_EXPERTS}" == "1" ]]; then
    POST_ARGS+=(--fix-moe-experts)
  fi
  run_llamafactory_python "${POST_ARGS[@]}"
fi

echo "Done."
