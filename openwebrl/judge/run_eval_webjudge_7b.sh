#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

# Default WebJudge-7B HF repo. Override MODEL_PATH with a local downloaded path
# if the machine has no internet access.
MODEL_PATH="${MODEL_PATH:-osunlp/WebJudge-7B}"
VAL_DATA="${VAL_DATA:-}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/judge_eval_webjudge_7b_eval500}"

NUM_WORKERS="${NUM_WORKERS:-8}"
GPU_IDS="${GPU_IDS:-${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}}"
GPU_ID="${GPU_ID:-0}"
BATCH_SIZE="${BATCH_SIZE:-4}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
SAMPLE_SEED="${SAMPLE_SEED:-0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"
TEMPERATURE="${TEMPERATURE:-0.0}"
TOP_P="${TOP_P:-1.0}"
DEVICE_MAP="${DEVICE_MAP:-auto}"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
PROCESSOR_PATH="${PROCESSOR_PATH:-}"
WEBJUDGE_KEY_POINTS_FIELD="${WEBJUDGE_KEY_POINTS_FIELD:-key_points}"
WEBJUDGE_STATUS_RETRY="${WEBJUDGE_STATUS_RETRY:-1}"
WEBJUDGE_STATUS_RETRY_MAX_NEW_TOKENS="${WEBJUDGE_STATUS_RETRY_MAX_NEW_TOKENS:-64}"
PRINT_OUTPUT="${PRINT_OUTPUT:-0}"
HF_OFFLINE="${HF_OFFLINE:-0}"

[ -f "${VAL_DATA}" ] || { echo "ERROR: validation data not found: ${VAL_DATA}"; exit 1; }

if [ "${HF_OFFLINE}" = "1" ]; then
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
fi

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

if is_local_model_path "${MODEL_PATH}"; then
  MODEL_PATH="$(expand_local_path "${MODEL_PATH}")"
  [ -d "${MODEL_PATH}" ] || { echo "ERROR: local model path not found: ${MODEL_PATH}"; exit 1; }
fi

mkdir -p "${OUTPUT_DIR}"

IFS=',' read -r -a GPU_ID_ARRAY <<< "${GPU_IDS}"
AVAILABLE_GPUS="${#GPU_ID_ARRAY[@]}"

if [ "${NUM_WORKERS}" -gt "${AVAILABLE_GPUS}" ]; then
  echo "ERROR: NUM_WORKERS (${NUM_WORKERS}) is larger than visible GPU ids (${AVAILABLE_GPUS}): ${GPU_IDS}"
  exit 1
fi

merge_worker_outputs() {
  local out_dir="$1"
  local samplewise="${out_dir}/samplewise.jsonl"
  local summary="${out_dir}/summary.json"

  python3 - "${out_dir}" "${MODEL_PATH}" "${VAL_DATA}" "${samplewise}" "${summary}" <<'PY'
import glob
import json
import os
import sys
import time

out_dir, model_path, val_data, samplewise_path, summary_path = sys.argv[1:]

rows = []
for path in sorted(glob.glob(os.path.join(out_dir, "worker_*", "samplewise.jsonl"))):
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
    "num_shards": len(glob.glob(os.path.join(out_dir, "worker_*", "summary.json"))),
    "merged_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
}

with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print(json.dumps(summary, ensure_ascii=False, indent=2))
PY
}

ARGS=(
  --model "${MODEL_PATH}"
  --input "${VAL_DATA}"
  --output-jsonl "${OUTPUT_DIR}/samplewise.jsonl"
  --summary-json "${OUTPUT_DIR}/summary.json"
  --max-samples "${MAX_SAMPLES}"
  --sample-seed "${SAMPLE_SEED}"
  --batch-size "${BATCH_SIZE}"
  --max-new-tokens "${MAX_NEW_TOKENS}"
  --temperature "${TEMPERATURE}"
  --top-p "${TOP_P}"
  --device-map "${DEVICE_MAP}"
  --torch-dtype "${TORCH_DTYPE}"
  --prompt-template webjudge
  --webjudge-key-points-field "${WEBJUDGE_KEY_POINTS_FIELD}"
  --webjudge-status-retry "${WEBJUDGE_STATUS_RETRY}"
  --webjudge-status-retry-max-new-tokens "${WEBJUDGE_STATUS_RETRY_MAX_NEW_TOKENS}"
)

if [ -n "${PROCESSOR_PATH}" ]; then
  ARGS+=(--processor-path "${PROCESSOR_PATH}")
fi

if [ "${PRINT_OUTPUT}" = "1" ]; then
  ARGS+=(--print-output)
fi

echo "MODEL_PATH: ${MODEL_PATH}"
echo "VAL_DATA: ${VAL_DATA}"
echo "OUTPUT_DIR: ${OUTPUT_DIR}"
echo "NUM_WORKERS: ${NUM_WORKERS}"
echo "GPU_IDS: ${GPU_IDS}"
echo "BATCH_SIZE: ${BATCH_SIZE}"
echo "MAX_SAMPLES: ${MAX_SAMPLES}"
echo "MAX_NEW_TOKENS: ${MAX_NEW_TOKENS}"
echo "TEMPERATURE: ${TEMPERATURE}"
echo "TORCH_DTYPE: ${TORCH_DTYPE}"
echo "WEBJUDGE_KEY_POINTS_FIELD: ${WEBJUDGE_KEY_POINTS_FIELD}"
echo "WEBJUDGE_STATUS_RETRY: ${WEBJUDGE_STATUS_RETRY}"
echo "WEBJUDGE_STATUS_RETRY_MAX_NEW_TOKENS: ${WEBJUDGE_STATUS_RETRY_MAX_NEW_TOKENS}"
echo "HF_OFFLINE: ${HF_OFFLINE}"

if [ "${NUM_WORKERS}" -le 1 ]; then
  CUDA_VISIBLE_DEVICES="${GPU_ID}" python3 "${SCRIPT_DIR}/single_file_judge_inference.py" \
    "${ARGS[@]}" 2>&1 | tee "${OUTPUT_DIR}/run.log"
else
  pids=()

  for (( worker_id=0; worker_id<NUM_WORKERS; worker_id++ )); do
    worker_gpu="${GPU_ID_ARRAY[$worker_id]}"
    worker_dir="${OUTPUT_DIR}/worker_${worker_id}"
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
    echo "Started WebJudge-7B worker ${worker_id}/${NUM_WORKERS} on GPU ${worker_gpu}, output=${worker_dir}"
  done

  failed=0
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      failed=1
    fi
  done

  if [ "${failed}" -ne 0 ]; then
    echo "ERROR: one or more workers failed."
    echo "Check logs under: ${OUTPUT_DIR}/worker_*/run.log"
    exit 1
  fi

  merge_worker_outputs "${OUTPUT_DIR}" | tee "${OUTPUT_DIR}/run.log"
fi

echo
echo "Done. Results saved under: ${OUTPUT_DIR}"
echo "Samplewise: ${OUTPUT_DIR}/samplewise.jsonl"
echo "Summary: ${OUTPUT_DIR}/summary.json"
