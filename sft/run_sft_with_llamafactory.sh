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
CONFIG_DIR="${OPENWEBRL_SFT_CONFIG_DIR:-${WORK_DIR}/configs}"
RAW_DATA_DIR="${OPENWEBRL_SFT_RAW_DATA_DIR:-${WORK_DIR}/raw}"
PREPARED_DATA="${OPENWEBRL_SFT_PREPARED_DATA:-${DATA_DIR}/openwebrl_sft_data.jsonl}"
DATASET_NAME="${OPENWEBRL_SFT_DATASET_NAME:-openwebrl_sft_trajectories}"
DATASET_INFO_PATH="${DATA_DIR}/dataset_info.json"
TRAIN_CONFIG="${OPENWEBRL_SFT_TRAIN_CONFIG:-${CONFIG_DIR}/openwebrl_qwen3_vl_sft.yaml}"

MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-Qwen/Qwen3-VL-4B-Thinking}"
OUTPUT_DIR="${OUTPUT_DIR:-${WORK_DIR}/checkpoints/qwen3-vl-openwebrl-sft}"
RUN_NAME="${RUN_NAME:-qwen3-vl-openwebrl-sft}"

TEMPLATE="${TEMPLATE:-qwen3_vl}"
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
REPORT_TO="${REPORT_TO:-none}"
PREPROCESSING_NUM_WORKERS="${PREPROCESSING_NUM_WORKERS:-16}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-4}"
DDP_TIMEOUT="${DDP_TIMEOUT:-180000000}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-null}"

RUN_PREPARE="${RUN_PREPARE:-1}"
RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_POST_PROCESS="${RUN_POST_PROCESS:-0}"
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

mkdir -p "${DATA_DIR}" "${CONFIG_DIR}" "${RAW_DATA_DIR}" "${OUTPUT_DIR}"

require_dir "${LLAMAFACTORY_ROOT}" "LLaMAFactory root"
if ! command -v "${UV}" >/dev/null 2>&1; then
  echo "ERROR: uv is required. Install uv and set up the LLaMAFactory uv environment first." >&2
  exit 1
fi
require_file "${LLAMAFACTORY_ROOT}/pyproject.toml" "LLaMAFactory pyproject.toml"
require_file "${SCRIPT_DIR}/prepare_llamafactory_sft_data.py" "SFT data converter"
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

if [[ "${RUN_PREPARE}" == "1" || ! -f "${PREPARED_DATA}" ]]; then
  echo "Preparing LLaMAFactory SFT data ..."
  PREPARE_ARGS=(
    "${SCRIPT_DIR}/prepare_llamafactory_sft_data.py"
    --input-path "${OPENWEBRL_SFT_RAW_DATA}"
    --output-path "${PREPARED_DATA}"
    --image-path-mode relative
  )
  if [[ -n "${MAX_ROWS}" ]]; then
    PREPARE_ARGS+=(--max-rows "${MAX_ROWS}")
  fi
  run_llamafactory_python "${PREPARE_ARGS[@]}"
else
  echo "Skipping data preparation because ${PREPARED_DATA} already exists."
fi

require_file "${PREPARED_DATA}" "prepared LLaMAFactory data"

export DATASET_INFO_PATH DATASET_NAME PREPARED_DATA
run_llamafactory_python -c 'import json, os
from pathlib import Path
info = {
    os.environ["DATASET_NAME"]: {
        "file_name": os.environ["PREPARED_DATA"],
        "formatting": "sharegpt",
        "columns": {
            "messages": "messages",
            "images": "images"
        },
        "tags": {
            "role_tag": "role",
            "content_tag": "content",
            "user_tag": "user",
            "assistant_tag": "assistant",
            "system_tag": "system"
        }
    }
}
Path(os.environ["DATASET_INFO_PATH"]).write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
'

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
dataset_dir: ${DATA_DIR}
media_dir: ${DATA_DIR}
dataset: ${DATASET_NAME}
template: ${TEMPLATE}
cutoff_len: ${CUTOFF_LEN}
preprocessing_num_workers: ${PREPROCESSING_NUM_WORKERS}
dataloader_num_workers: ${DATALOADER_NUM_WORKERS}
mask_history: true

### output
output_dir: ${OUTPUT_DIR}
logging_steps: 10
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
