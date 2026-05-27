#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

VAL_DATA="${VAL_DATA:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRIPT_DIR}}"

# Supports both Hugging Face repo ids and local paths.
# Examples:
#   Qwen/Qwen3-VL-32B-Instruct
#   /path/to/local/Qwen3-VL-4B-Thinking
MODEL_LIST="${MODEL_LIST:-Qwen/Qwen3-VL-32B-Instruct,Qwen/Qwen3-VL-8B-Thinking,Qwen/Qwen3-VL-32B-Thinking}"

NUM_WORKERS="${NUM_WORKERS:-8}"
GPU_IDS="${GPU_IDS:-${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}}"
BATCH_SIZE="${BATCH_SIZE:-4}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
SAMPLE_SEED="${SAMPLE_SEED:-0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"
TEMPERATURE="${TEMPERATURE:-0.0}"
TOP_P="${TOP_P:-1.0}"
DEVICE_MAP="${DEVICE_MAP:-auto}"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
PRINT_OUTPUT="${PRINT_OUTPUT:-0}"

# If your server cannot access the internet but the model already exists in HF cache,
# run with:
#   HF_OFFLINE=1 bash this_script.sh
HF_OFFLINE="${HF_OFFLINE:-0}"

[ -f "${VAL_DATA}" ] || { echo "ERROR: validation data not found: ${VAL_DATA}"; exit 1; }

if [ "${HF_OFFLINE}" = "1" ]; then
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
fi

IFS=',' read -r -a MODELS <<< "${MODEL_LIST}"
IFS=',' read -r -a GPU_ID_ARRAY <<< "${GPU_IDS}"
AVAILABLE_GPUS="${#GPU_ID_ARRAY[@]}"

if [ "${NUM_WORKERS}" -gt "${AVAILABLE_GPUS}" ]; then
  echo "ERROR: NUM_WORKERS (${NUM_WORKERS}) is larger than visible GPU ids (${AVAILABLE_GPUS}): ${GPU_IDS}"
  exit 1
fi

trim() {
  local s="$*"
  s="${s#"${s%%[![:space:]]*}"}"
  s="${s%"${s##*[![:space:]]}"}"
  printf '%s' "$s"
}

is_local_model_path() {
  local path="$1"
  [[ "${path}" == /* || "${path}" == ./* || "${path}" == ../* || "${path}" == ~/* ]]
}

expand_local_path() {
  local path="$1"
  if [[ "${path}" == ~/* ]]; then
    printf '%s/%s' "${HOME}" "${path#~/}"
  else
    printf '%s' "${path}"
  fi
}

safe_model_name() {
  local model="$1"

  if is_local_model_path "${model}"; then
    model="$(expand_local_path "${model}")"
    model="$(basename "${model}")"
  else
    # Keep organization info for HF repo ids and avoid directory conflicts.
    # Example: Qwen/Qwen3-VL-32B-Instruct -> Qwen__Qwen3-VL-32B-Instruct
    model="${model//\//__}"
  fi

  model="${model// /_}"
  printf '%s' "${model}"
}

merge_worker_outputs() {
  local model_out="$1"
  local model_path="$2"
  local samplewise="${model_out}/samplewise.jsonl"
  local summary="${model_out}/summary.json"

  python3 - "${model_out}" "${model_path}" "${VAL_DATA}" "${samplewise}" "${summary}" <<'PY'
import glob
import json
import os
import sys
import time

model_out, model_path, val_data, samplewise_path, summary_path = sys.argv[1:]

rows = []
for path in sorted(glob.glob(os.path.join(model_out, "worker_*", "samplewise.jsonl"))):
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

rows.sort(key=lambda row: (row.get("source") or "", row.get("line_number") or 0, row.get("index") or 0))

with open(samplewise_path, "w", encoding="utf-8") as f:
    for idx, row in enumerate(rows, start=1):
        row["index"] = idx
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

total = len(rows)
parsed = sum(row.get("verdict") is not None for row in rows)
oracle_rows = [row for row in rows if row.get("oracle_verdict") is not None]
matches = sum(row.get("matches_oracle") is True for row in rows)

tp = sum(row.get("verdict") == "SUCCESS" and row.get("oracle_verdict") == "SUCCESS" for row in oracle_rows)
fp = sum(row.get("verdict") == "SUCCESS" and row.get("oracle_verdict") == "NOT SUCCESS" for row in oracle_rows)
fn = sum(row.get("verdict") == "NOT SUCCESS" and row.get("oracle_verdict") == "SUCCESS" for row in oracle_rows)
tn = sum(row.get("verdict") == "NOT SUCCESS" and row.get("oracle_verdict") == "NOT SUCCESS" for row in oracle_rows)

precision = 0.0 if tp + fp == 0 else tp / (tp + fp)
recall = 0.0 if tp + fn == 0 else tp / (tp + fn)
f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)

summary = {
    "model": model_path,
    "input": val_data,
    "num_samples": total,
    "parsed_count": parsed,
    "unparsed_count": total - parsed,
    "success_count": sum(row.get("verdict") == "SUCCESS" for row in rows),
    "not_success_count": sum(row.get("verdict") == "NOT SUCCESS" for row in rows),
    "oracle_count": len(oracle_rows),
    "oracle_accuracy": 0.0 if not oracle_rows else matches / len(oracle_rows),
    "positive_label": "SUCCESS",
    "true_positive": tp,
    "false_positive": fp,
    "false_negative": fn,
    "true_negative": tn,
    "precision": precision,
    "recall": recall,
    "f1": f1,
    "shard_id": -1,
    "num_shards": len(glob.glob(os.path.join(model_out, "worker_*", "summary.json"))),
    "merged_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
}

with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print(json.dumps(summary, ensure_ascii=False, indent=2))
PY
}

write_summary_tsv() {
  local output_root="$1"
  local summary_tsv="${output_root}/judge_eval_test_models_summary.tsv"

  python3 - "${output_root}" > "${summary_tsv}" <<'PY'
import glob
import json
import os
import sys

output_root = sys.argv[1]

print("model\taccuracy\tprecision\trecall\tf1\tparsed\tnum_samples\tunparsed\tsuccess\tnot_success")

summary_paths = sorted(glob.glob(os.path.join(output_root, "judge_eval_test_*", "summary.json")))

for path in summary_paths:
    with open(path, encoding="utf-8") as f:
        summary = json.load(f)

    dirname = os.path.basename(os.path.dirname(path))
    model_name = dirname.removeprefix("judge_eval_test_").removesuffix("_fix")

    print(
        f"{model_name}\t"
        f"{summary.get('oracle_accuracy')}\t"
        f"{summary.get('precision')}\t"
        f"{summary.get('recall')}\t"
        f"{summary.get('f1')}\t"
        f"{summary.get('parsed_count')}\t"
        f"{summary.get('num_samples')}\t"
        f"{summary.get('unparsed_count')}\t"
        f"{summary.get('success_count')}\t"
        f"{summary.get('not_success_count')}"
    )
PY

  echo "Summary TSV: ${summary_tsv}"
  cat "${summary_tsv}"
}

echo "VAL_DATA: ${VAL_DATA}"
echo "OUTPUT_ROOT: ${OUTPUT_ROOT}"
echo "MODEL_LIST: ${MODEL_LIST}"
echo "NUM_WORKERS: ${NUM_WORKERS}"
echo "GPU_IDS: ${GPU_IDS}"
echo "BATCH_SIZE: ${BATCH_SIZE}"
echo "MAX_SAMPLES: ${MAX_SAMPLES}"
echo "MAX_NEW_TOKENS: ${MAX_NEW_TOKENS}"
echo "TEMPERATURE: ${TEMPERATURE}"
echo "TORCH_DTYPE: ${TORCH_DTYPE}"
echo "HF_OFFLINE: ${HF_OFFLINE}"

for RAW_MODEL_PATH in "${MODELS[@]}"; do
  MODEL_PATH="$(trim "${RAW_MODEL_PATH}")"

  if [ -z "${MODEL_PATH}" ]; then
    continue
  fi

  # Only validate paths that explicitly look like local paths.
  # Hugging Face repo ids such as Qwen/Qwen3-VL-32B-Instruct are passed directly
  # to single_file_judge_inference.py and resolved by Transformers.
  if is_local_model_path "${MODEL_PATH}"; then
    MODEL_PATH="$(expand_local_path "${MODEL_PATH}")"
    [ -d "${MODEL_PATH}" ] || { echo "ERROR: local model path not found: ${MODEL_PATH}"; exit 1; }
  fi

  MODEL_NAME="$(safe_model_name "${MODEL_PATH}")"
  MODEL_OUT="${OUTPUT_ROOT}/judge_eval_test_${MODEL_NAME}_fix"
  mkdir -p "${MODEL_OUT}"

  echo
  echo "============================================================"
  echo "Evaluating ${MODEL_NAME}"
  echo "MODEL_PATH: ${MODEL_PATH}"
  echo "MODEL_OUT: ${MODEL_OUT}"
  echo "============================================================"

  ARGS=(
    --model "${MODEL_PATH}"
    --input "${VAL_DATA}"
    --output-jsonl "${MODEL_OUT}/samplewise.jsonl"
    --summary-json "${MODEL_OUT}/summary.json"
    --max-samples "${MAX_SAMPLES}"
    --sample-seed "${SAMPLE_SEED}"
    --batch-size "${BATCH_SIZE}"
    --max-new-tokens "${MAX_NEW_TOKENS}"
    --temperature "${TEMPERATURE}"
    --top-p "${TOP_P}"
    --device-map "${DEVICE_MAP}"
    --torch-dtype "${TORCH_DTYPE}"
  )

  if [ "${PRINT_OUTPUT}" = "1" ]; then
    ARGS+=(--print-output)
  fi

  pids=()

  for (( worker_id=0; worker_id<NUM_WORKERS; worker_id++ )); do
    worker_gpu="${GPU_ID_ARRAY[$worker_id]}"
    worker_dir="${MODEL_OUT}/worker_${worker_id}"
    mkdir -p "${worker_dir}"

    (
      export CUDA_VISIBLE_DEVICES="${worker_gpu}"

      python3 "${SCRIPT_DIR}/single_file_judge_inference.py" \
        "${ARGS[@]}" \
        --output-jsonl "${worker_dir}/samplewise.jsonl" \
        --summary-json "${worker_dir}/summary.json" \
        --shard-id "${worker_id}" \
        --num-shards "${NUM_WORKERS}" \
        > "${worker_dir}/run.log" 2>&1
    ) &

    pids+=($!)
    echo "Started ${MODEL_NAME} worker ${worker_id}/${NUM_WORKERS} on GPU ${worker_gpu}, output=${worker_dir}"
  done

  failed=0
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      failed=1
    fi
  done

  if [ "${failed}" -ne 0 ]; then
    echo "ERROR: one or more workers failed for ${MODEL_NAME}."
    echo "Check logs under: ${MODEL_OUT}/worker_*/run.log"
    exit 1
  fi

  merge_worker_outputs "${MODEL_OUT}" "${MODEL_PATH}" | tee "${MODEL_OUT}/run.log"
done

echo
write_summary_tsv "${OUTPUT_ROOT}"
