#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

CKPT_ROOT="${CKPT_ROOT:-}"
VAL_DATA="${VAL_DATA:-}"
OUTPUT_DIR="${OUTPUT_DIR:-judge_eval_eval500_8B_v2}"

GPU_ID="${GPU_ID:-0}"
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
PROCESSOR_PATH="${PROCESSOR_PATH:-}"
PRINT_OUTPUT="${PRINT_OUTPUT:-0}"

[ -d "${CKPT_ROOT}" ] || { echo "ERROR: checkpoint root not found: ${CKPT_ROOT}"; exit 1; }
[ -f "${VAL_DATA}" ] || { echo "ERROR: validation data not found: ${VAL_DATA}"; exit 1; }

mkdir -p "${OUTPUT_DIR}"

echo "CKPT_ROOT: ${CKPT_ROOT}"
echo "VAL_DATA: ${VAL_DATA}"
echo "OUTPUT_DIR: ${OUTPUT_DIR}"
echo "NUM_WORKERS: ${NUM_WORKERS}"
echo "GPU_IDS: ${GPU_IDS}"
echo "GPU_ID: ${GPU_ID} (used only when NUM_WORKERS=1)"
echo "BATCH_SIZE: ${BATCH_SIZE}"
echo "MAX_SAMPLES: ${MAX_SAMPLES}"
echo "MAX_NEW_TOKENS: ${MAX_NEW_TOKENS}"
echo "TEMPERATURE: ${TEMPERATURE}"
echo "TORCH_DTYPE: ${TORCH_DTYPE}"

shopt -s nullglob
MODELS=("${CKPT_ROOT}"/iter_*/*_converted)
shopt -u nullglob

if [ "${#MODELS[@]}" -eq 0 ]; then
  echo "ERROR: no converted checkpoints found under ${CKPT_ROOT}/iter_*/*_converted"
  exit 1
fi

IFS=',' read -r -a GPU_ID_ARRAY <<< "${GPU_IDS}"
AVAILABLE_GPUS="${#GPU_ID_ARRAY[@]}"

if [ "${NUM_WORKERS}" -gt "${AVAILABLE_GPUS}" ]; then
  echo "ERROR: NUM_WORKERS (${NUM_WORKERS}) is larger than visible GPU ids (${AVAILABLE_GPUS}): ${GPU_IDS}"
  exit 1
fi

merge_worker_outputs() {
  local step_out="$1"
  local model_path="$2"
  local samplewise="${step_out}/samplewise.jsonl"
  local summary="${step_out}/summary.json"

  python3 - "${step_out}" "${model_path}" "${VAL_DATA}" "${samplewise}" "${summary}" <<'PY'
import glob
import json
import os
import sys
import time

step_out, model_path, val_data, samplewise_path, summary_path = sys.argv[1:]
rows = []
for path in sorted(glob.glob(os.path.join(step_out, "worker_*", "samplewise.jsonl"))):
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
    "num_shards": len(glob.glob(os.path.join(step_out, "worker_*", "summary.json"))),
    "merged_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
}
with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY
}

for MODEL_PATH in "${MODELS[@]}"; do
  STEP="$(basename "$(dirname "${MODEL_PATH}")")"
  STEP_OUT="${OUTPUT_DIR}/${STEP}"
  mkdir -p "${STEP_OUT}"

  echo
  echo "============================================================"
  echo "Evaluating ${STEP}"
  echo "MODEL_PATH: ${MODEL_PATH}"
  echo "STEP_OUT: ${STEP_OUT}"
  echo "============================================================"

  ARGS=(
    --model "${MODEL_PATH}"
    --input "${VAL_DATA}"
    --output-jsonl "${STEP_OUT}/samplewise.jsonl"
    --summary-json "${STEP_OUT}/summary.json"
    --max-samples "${MAX_SAMPLES}"
    --sample-seed "${SAMPLE_SEED}"
    --batch-size "${BATCH_SIZE}"
    --max-new-tokens "${MAX_NEW_TOKENS}"
    --temperature "${TEMPERATURE}"
    --top-p "${TOP_P}"
    --device-map "${DEVICE_MAP}"
    --torch-dtype "${TORCH_DTYPE}"
  )

  if [ -n "${PROCESSOR_PATH}" ]; then
    ARGS+=(--processor-path "${PROCESSOR_PATH}")
  fi

  if [ "${PRINT_OUTPUT}" = "1" ]; then
    ARGS+=(--print-output)
  fi

  if [ "${NUM_WORKERS}" -le 1 ]; then
    CUDA_VISIBLE_DEVICES="${GPU_ID}" python3 "${SCRIPT_DIR}/single_file_judge_inference.py" \
      "${ARGS[@]}" 2>&1 | tee "${STEP_OUT}/run.log"
  else
    pids=()
    for (( worker_id=0; worker_id<NUM_WORKERS; worker_id++ )); do
      worker_gpu="${GPU_ID_ARRAY[$worker_id]}"
      worker_dir="${STEP_OUT}/worker_${worker_id}"
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
      echo "Started ${STEP} worker ${worker_id}/${NUM_WORKERS} on GPU ${worker_gpu}, output=${worker_dir}"
    done

    for pid in "${pids[@]}"; do
      wait "${pid}"
    done

    merge_worker_outputs "${STEP_OUT}" "${MODEL_PATH}" | tee "${STEP_OUT}/run.log"
  fi
done

SUMMARY_TSV="${OUTPUT_DIR}/summary.tsv"
python3 - "${OUTPUT_DIR}" > "${SUMMARY_TSV}" <<'PY'
import glob
import json
import os
import sys

output_dir = sys.argv[1]
print("step\taccuracy\tprecision\trecall\tf1\tparsed\tnum_samples\tunparsed\tsuccess\tnot_success")
for path in sorted(glob.glob(os.path.join(output_dir, "iter_*", "summary.json"))):
    with open(path, encoding="utf-8") as f:
        summary = json.load(f)
    step = os.path.basename(os.path.dirname(path))
    print(
        f"{step}\t{summary.get('oracle_accuracy')}\t"
        f"{summary.get('precision')}\t{summary.get('recall')}\t{summary.get('f1')}\t"
        f"{summary.get('parsed_count')}\t{summary.get('num_samples')}\t"
        f"{summary.get('unparsed_count')}\t{summary.get('success_count')}\t"
        f"{summary.get('not_success_count')}"
    )
PY

echo
echo "Done. Results saved under: ${OUTPUT_DIR}"
echo "Summary TSV: ${SUMMARY_TSV}"
cat "${SUMMARY_TSV}"
