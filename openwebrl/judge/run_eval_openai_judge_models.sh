#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

VAL_DATA="${VAL_DATA:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRIPT_DIR}}"

# Comma-separated API model/deployment names.
MODEL_LIST="${MODEL_LIST:-gpt-4o,o4-mini}"

NUM_WORKERS="${NUM_WORKERS:-1}"
CONCURRENCY="${CONCURRENCY:-8}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
SAMPLE_SEED="${SAMPLE_SEED:-0}"
MAX_COMPLETION_TOKENS="${MAX_COMPLETION_TOKENS:-2048}"
TIMEOUT_SECS="${TIMEOUT_SECS:-180}"
MAX_RETRIES="${MAX_RETRIES:-8}"
PRINT_OUTPUT="${PRINT_OUTPUT:-0}"
RESUME="${RESUME:-1}"

# openai: OPENAI_API_KEY with optional OPENAI_BASE_URL / OPENAI_API_BASE
# token / azure_token: AZURE_RESOURCE_NAME + AZURE_TOKEN_PATH
# api_key / azure_api_key: AZURE_OPENAI_ENDPOINT or OPENAI_API_BASE + OPENAI_API_KEY
# served: JUDGE_API_BASE / JUDGE_API_HOST / JUDGE_API_PORT
if [ -z "${API_MODE:-}" ]; then
  if [ -n "${AZURE_TOKEN_PATH:-}" ] && [ -f "${AZURE_TOKEN_PATH}" ]; then
    API_MODE="token"
  else
    API_MODE="openai"
  fi
fi

[ -f "${VAL_DATA}" ] || { echo "ERROR: validation data not found: ${VAL_DATA}"; exit 1; }

case "${API_MODE}" in
  openai)
    [ -n "${OPENAI_API_KEY:-}" ] || {
      echo "ERROR: API_MODE=openai requires OPENAI_API_KEY."
      echo "Set OPENAI_API_KEY=... or run with API_MODE=token if you use AZURE_TOKEN_PATH."
      exit 1
    }
    ;;
  token|azure_token)
    [ -n "${AZURE_RESOURCE_NAME:-}" ] || { echo "ERROR: API_MODE=${API_MODE} requires AZURE_RESOURCE_NAME."; exit 1; }
    [ -n "${AZURE_TOKEN_PATH:-}" ] && [ -f "${AZURE_TOKEN_PATH}" ] || {
      echo "ERROR: API_MODE=${API_MODE} requires AZURE_TOKEN_PATH to point to a token file."
      exit 1
    }
    ;;
  api_key|azure_api_key)
    [ -n "${OPENAI_API_KEY:-${AZURE_OPENAI_API_KEY:-}}" ] || {
      echo "ERROR: API_MODE=${API_MODE} requires OPENAI_API_KEY or AZURE_OPENAI_API_KEY."
      exit 1
    }
    [ -n "${AZURE_OPENAI_ENDPOINT:-${OPENAI_API_BASE:-}}" ] || {
      echo "ERROR: API_MODE=${API_MODE} requires AZURE_OPENAI_ENDPOINT or OPENAI_API_BASE."
      exit 1
    }
    ;;
  served)
    [ -n "${JUDGE_API_BASE:-${JUDGE_API_HOST:-}}" ] || {
      echo "ERROR: API_MODE=served requires JUDGE_API_BASE or JUDGE_API_HOST."
      exit 1
    }
    ;;
  *)
    echo "ERROR: unsupported API_MODE=${API_MODE}"
    exit 1
    ;;
esac

IFS=',' read -r -a MODELS <<< "${MODEL_LIST}"

trim() {
  local s="$*"
  s="${s#"${s%%[![:space:]]*}"}"
  s="${s%"${s##*[![:space:]]}"}"
  printf '%s' "$s"
}

safe_model_name() {
  local model="$1"
  model="${model//\//__}"
  model="${model// /_}"
  model="${model//:/_}"
  printf '%s' "${model}"
}

merge_worker_outputs() {
  local model_out="$1"
  local model_name="$2"
  local samplewise="${model_out}/samplewise.jsonl"
  local summary="${model_out}/summary.json"

  python3 - "${model_out}" "${model_name}" "${VAL_DATA}" "${samplewise}" "${summary}" "${API_MODE}" <<'PY'
import glob
import json
import os
import sys
import time

model_out, model_name, val_data, samplewise_path, summary_path, api_mode = sys.argv[1:]

def verdict_for_metrics(verdict):
    return "SUCCESS" if verdict == "SUCCESS" else "NOT SUCCESS"

rows = []
for path in sorted(glob.glob(os.path.join(model_out, "worker_*", "samplewise.jsonl"))):
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                row["verdict_for_metrics"] = row.get("verdict_for_metrics") or verdict_for_metrics(row.get("verdict"))
                rows.append(row)

total = len(rows)
parsed = sum(row.get("verdict") is not None for row in rows)
oracle_rows = [row for row in rows if row.get("oracle_verdict") is not None]

for row in rows:
    oracle = row.get("oracle_verdict")
    row["matches_oracle"] = (row.get("verdict_for_metrics") == oracle) if oracle is not None else None

rows.sort(key=lambda row: (row.get("source") or "", row.get("line_number") or 0, row.get("index") or 0))

with open(samplewise_path, "w", encoding="utf-8") as f:
    for idx, row in enumerate(rows, start=1):
        row["index"] = idx
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

matches = sum(row.get("matches_oracle") is True for row in oracle_rows)
tp = sum(row.get("verdict_for_metrics") == "SUCCESS" and row.get("oracle_verdict") == "SUCCESS" for row in oracle_rows)
fp = sum(row.get("verdict_for_metrics") == "SUCCESS" and row.get("oracle_verdict") == "NOT SUCCESS" for row in oracle_rows)
fn = sum(row.get("verdict_for_metrics") == "NOT SUCCESS" and row.get("oracle_verdict") == "SUCCESS" for row in oracle_rows)
tn = sum(row.get("verdict_for_metrics") == "NOT SUCCESS" and row.get("oracle_verdict") == "NOT SUCCESS" for row in oracle_rows)

precision = 0.0 if tp + fp == 0 else tp / (tp + fp)
recall = 0.0 if tp + fn == 0 else tp / (tp + fn)
f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)

summary = {
    "model": model_name,
    "input": val_data,
    "num_samples": total,
    "parsed_count": parsed,
    "unparsed_count": total - parsed,
    "unparsed_treated_as": "NOT SUCCESS",
    "success_count": sum(row.get("verdict_for_metrics") == "SUCCESS" for row in rows),
    "not_success_count": sum(row.get("verdict_for_metrics") == "NOT SUCCESS" for row in rows),
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
    "api_mode": api_mode,
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
  local summary_tsv="${output_root}/judge_eval_openai_models_summary.tsv"

  python3 - "${output_root}" > "${summary_tsv}" <<'PY'
import glob
import json
import os
import sys

output_root = sys.argv[1]
print("model\taccuracy\tprecision\trecall\tf1\tparsed\tnum_samples\tunparsed\tsuccess\tnot_success")

for path in sorted(glob.glob(os.path.join(output_root, "judge_eval_openai_*", "summary.json"))):
    with open(path, encoding="utf-8") as f:
        summary = json.load(f)
    dirname = os.path.basename(os.path.dirname(path))
    model_name = dirname.removeprefix("judge_eval_openai_")
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
echo "CONCURRENCY: ${CONCURRENCY}"
echo "MAX_SAMPLES: ${MAX_SAMPLES}"
echo "MAX_COMPLETION_TOKENS: ${MAX_COMPLETION_TOKENS}"
echo "TIMEOUT_SECS: ${TIMEOUT_SECS}"
echo "API_MODE: ${API_MODE}"
echo "RESUME: ${RESUME}"

for RAW_MODEL in "${MODELS[@]}"; do
  MODEL="$(trim "${RAW_MODEL}")"
  if [ -z "${MODEL}" ]; then
    continue
  fi

  MODEL_NAME="$(safe_model_name "${MODEL}")"
  MODEL_OUT="${OUTPUT_ROOT}/judge_eval_openai_${MODEL_NAME}"
  mkdir -p "${MODEL_OUT}"

  echo
  echo "============================================================"
  echo "Evaluating ${MODEL}"
  echo "MODEL_OUT: ${MODEL_OUT}"
  echo "============================================================"

  ARGS=(
    --model "${MODEL}"
    --input "${VAL_DATA}"
    --max-samples "${MAX_SAMPLES}"
    --sample-seed "${SAMPLE_SEED}"
    --concurrency "${CONCURRENCY}"
    --max-completion-tokens "${MAX_COMPLETION_TOKENS}"
    --timeout-secs "${TIMEOUT_SECS}"
    --max-retries "${MAX_RETRIES}"
    --api-mode "${API_MODE}"
  )

  if [ "${PRINT_OUTPUT}" = "1" ]; then
    ARGS+=(--print-output)
  fi
  if [ "${RESUME}" = "1" ]; then
    ARGS+=(--resume)
  fi

  pids=()
  for (( worker_id=0; worker_id<NUM_WORKERS; worker_id++ )); do
    worker_dir="${MODEL_OUT}/worker_${worker_id}"
    mkdir -p "${worker_dir}"

    (
      python3 "${SCRIPT_DIR}/single_file_openai_judge_inference.py" \
        "${ARGS[@]}" \
        --output-jsonl "${worker_dir}/samplewise.jsonl" \
        --summary-json "${worker_dir}/summary.json" \
        --shard-id "${worker_id}" \
        --num-shards "${NUM_WORKERS}" \
        > "${worker_dir}/run.log" 2>&1
    ) &

    pids+=($!)
    echo "Started ${MODEL_NAME} worker ${worker_id}/${NUM_WORKERS}, output=${worker_dir}"
  done

  failed=0
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      failed=1
    fi
  done

  if [ "${failed}" -ne 0 ]; then
    echo "ERROR: one or more workers failed for ${MODEL}."
    echo "Check logs under: ${MODEL_OUT}/worker_*/run.log"
    exit 1
  fi

  merge_worker_outputs "${MODEL_OUT}" "${MODEL}" | tee "${MODEL_OUT}/run.log"
done

echo
write_summary_tsv "${OUTPUT_ROOT}"
