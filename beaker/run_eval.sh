#!/usr/bin/env bash
# In-container entry point for one (checkpoint, benchmark) eval, launched by
# beaker/launch_eval.py on one 8-GPU node. run_evaluation.sh already carries the
# paper's protocol defaults, so only the checkpoint, judge endpoint, and browser
# env mode are overridden here.
#
# Usage: CKPT=<path|hf-repo> BENCH=om2w|webvoyager|deepshop bash beaker/run_eval.sh
set -euo pipefail
set -x

OPENWEBRL_ROOT="${OPENWEBRL_ROOT:-/weka/oe-training-default/new_peters/OpenWebRL}"
cd "${OPENWEBRL_ROOT}"

BENCH="${BENCH:?set BENCH=om2w|webvoyager|deepshop}"

# ---- Paths ----
export SLIME_REPO_ROOT="${OPENWEBRL_ROOT}"
export PYTHONPATH="${OPENWEBRL_ROOT}:/root/Megatron-LM"
export HF_HOME="${HF_HOME:-/weka/oe-training-default/new_peters/cache/hf}"

# ---- Checkpoint: a local path, a HF repo id, or a whitespace-separated list.
# A list runs sequentially (one sglang server at a time), which is the only safe
# way to sweep when the cloud-browser concurrency cap is account-wide. ----
CKPT="${CKPT:-/weka/oe-training-default/new_peters/models/OpenWebRL/OpenWebRL-4B-SFT}"
for _c in ${CKPT}; do
  if [ ! -d "${_c}" ] && [[ "${_c}" != */* ]]; then
    echo "ERROR: checkpoint is neither a local dir nor a HF repo id: ${_c}"; exit 2
  fi
done
export MODEL_LIST_OVERRIDE="${CKPT}"
export MODEL_LIST_DIR=""
# For a list, the per-model subdirectory separates results, so the tree is
# named by RUN_TAG instead.
if [ "$(echo ${CKPT} | wc -w)" -gt 1 ]; then
  CKPT_SLUG="multi"
else
  CKPT_SLUG="$(basename "${CKPT}")"
fi

# ---- Benchmark selection. Paper Table 8 sets 3 judge screenshots; run_eval.py's
# per-protocol default would be 30 for webvoyager/deepshop. Measured: no
# difference on WebVoyager (11 tasks up, 11 down) or om2w (+1.0, inside noise). ----
# TASK_FILE may be preset (e.g. a pilot subset) and then only the protocol comes from BENCH.
case "${BENCH}" in
  om2w)
    export TASK_FILE="${TASK_FILE:-openwebrl/data/eval/online-mind2web.jsonl}"
    export EVAL_PROTOCOL="online_mind2web"
    export JUDGE_MAX_ATTACHED_IMGS=3
    ;;
  webvoyager)
    export TASK_FILE="${TASK_FILE:-openwebrl/data/eval/webvoyager_fara.jsonl}"
    export EVAL_PROTOCOL="webvoyager"
    export JUDGE_MAX_ATTACHED_IMGS=3
    ;;
  deepshop)
    export TASK_FILE="${TASK_FILE:-openwebrl/data/eval/deepshop.jsonl}"
    export EVAL_PROTOCOL="deepshop"
    export JUDGE_MAX_ATTACHED_IMGS=3
    ;;
  *)
    echo "ERROR: BENCH must be om2w|webvoyager|deepshop, got '${BENCH}'"; exit 2 ;;
esac
# A subset run (TASK_INDICES set) must not land in the same tree as a full run,
# or compute_eval_success_rate.py will average the two together.
RUN_TAG="${RUN_TAG:-${BENCH}}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-${OPENWEBRL_ROOT}/outputs/eval/${CKPT_SLUG}/${RUN_TAG}}"
mkdir -p "${OUTPUT_ROOT}"

# ---- Judge. Default: public OpenAI API with JUDGE_MODEL empty so run_evaluate.py
# applies the protocol default; `served` is the only mode that talks to non-Azure
# OpenAI. Alternative: a self-hosted judge (beaker/run_judge_server.sh) that
# publishes its URL to JUDGE_ENDPOINT_FILE; then JUDGE_MODEL must be set. ----
export JUDGE_API_MODE="served"
if [ -n "${JUDGE_ENDPOINT_FILE:-}" ]; then
  for _i in $(seq 1 120); do   # the judge job may still be loading weights
    [ -s "${JUDGE_ENDPOINT_FILE}" ] && break
    echo "waiting for judge endpoint file ${JUDGE_ENDPOINT_FILE} (${_i}/120)"; sleep 30
  done
  JUDGE_API_BASE="$(cat "${JUDGE_ENDPOINT_FILE}")"
  [ -n "${JUDGE_API_BASE}" ] || { echo "ERROR: empty judge endpoint file"; exit 4; }
fi
export JUDGE_API_BASE="${JUDGE_API_BASE:-https://api.openai.com}"
export JUDGE_MODEL="${JUDGE_MODEL:-}"   # empty => protocol default (o4-mini om2w / gpt-4o wv)
if [ "${JUDGE_API_BASE}" = "https://api.openai.com" ]; then
  : "${OPENAI_API_KEY:?OPENAI_API_KEY must be injected as a gantry secret-env}"
else
  : "${JUDGE_MODEL:?a non-OpenAI JUDGE_API_BASE needs an explicit JUDGE_MODEL}"
  export JUDGE_API_KEY="${JUDGE_API_KEY:-EMPTY}"
  export JUDGE_MAX_TOKENS="${JUDGE_MAX_TOKENS:-2048}"   # sglang's default 128 truncates the verdict
fi

# ---- Browser env mode. local_process blocks on ~16% of tasks; browser-use is
# the paper's backend and is worth +20 points on om2w. ----
export SLIME_BROWSER_ENV_MODE="${SLIME_BROWSER_ENV_MODE:-local_process}"
if [ "${SLIME_BROWSER_ENV_MODE}" = "browser-use" ]; then
  : "${BROWSER_USE_API_KEY:?browser-use mode requires BROWSER_USE_API_KEY}"
  export BROWSER_USE_API_KEY
  python -c "import browser_use_sdk" 2>/dev/null \
    || pip install --no-cache-dir browser-use-sdk \
    || { echo "ERROR: could not install the browser-use SDK"; exit 7; }
  # One session per task, so stay strictly below the account cap (teardown lag).
  BU_MAX_CONCURRENCY="${BU_MAX_CONCURRENCY:-25}"
  if [ "${SLIME_BROWSER_SANDBOX_MAX_SANDBOXES:-16}" -ge "${BU_MAX_CONCURRENCY}" ]; then
    echo "ERROR: n_parallel=${SLIME_BROWSER_SANDBOX_MAX_SANDBOXES} would meet or exceed the"
    echo "       Browser-Use concurrency cap of ${BU_MAX_CONCURRENCY}. Lower it."
    exit 8
  fi
  # Table 8 viewport. Stealth otherwise randomises it per session
  # (~1280x744..788), costing 2-5 points. Set to "" to keep the remote default.
  export BROWSER_USE_FORCE_VIEWPORT="${BROWSER_USE_FORCE_VIEWPORT-1280x1000}"
  echo "browser-use mode: n_parallel=${SLIME_BROWSER_SANDBOX_MAX_SANDBOXES} cap=${BU_MAX_CONCURRENCY} viewport=${BROWSER_USE_FORCE_VIEWPORT:-<remote default>}"
  # Stop anything a previous aborted run left billing.
  python -m openwebrl.env.browser_use_env --cleanup || true
fi
if [ "${SLIME_BROWSER_ENV_MODE}" = "browserbase" ]; then
  # Bills per browser-minute, so keep n_parallel and the task set deliberate.
  # Standing rules: proxies OFF, advanced stealth ON, viewport 1280x720 (what
  # stealth renders anyway; asking for anything else mis-scales clicks).
  : "${BROWSERBASE_API_KEY:?browserbase mode requires BROWSERBASE_API_KEY}"
  : "${BROWSERBASE_PROJECT_ID:?browserbase mode requires BROWSERBASE_PROJECT_ID}"
  export BROWSERBASE_API_KEY BROWSERBASE_PROJECT_ID
  export BROWSERBASE_PROXIES="${BROWSERBASE_PROXIES:-false}"
  export BROWSERBASE_ADVANCED_STEALTH="${BROWSERBASE_ADVANCED_STEALTH:-true}"
  export BROWSERBASE_SOLVE_CAPTCHAS="${BROWSERBASE_SOLVE_CAPTCHAS:-true}"
  export SLIME_BROWSER_VIEWPORT="${SLIME_BROWSER_VIEWPORT:-1280x720}"
  # Let the step budget end the episode, not the project's default session cap.
  export BROWSERBASE_SESSION_TIMEOUT_S="${BROWSERBASE_SESSION_TIMEOUT_S:-1800}"
  python -c "import browserbase" 2>/dev/null \
    || pip install --no-cache-dir "browserbase==1.4.0" \
    || { echo "ERROR: could not install the browserbase SDK"; exit 6; }
  echo "browserbase mode: proxies=${BROWSERBASE_PROXIES} stealth=${BROWSERBASE_ADVANCED_STEALTH} viewport=${SLIME_BROWSER_VIEWPORT}"
  # Release anything a previous aborted run left live before opening new sessions.
  python -m openwebrl.env.browserbase_env --cleanup || true
fi
if [ "${SLIME_BROWSER_ENV_MODE}" = "sandbox" ]; then
  : "${SANDBOX_ORCHESTRATOR_URL:?sandbox mode requires SANDBOX_ORCHESTRATOR_URL}"
  : "${BROWSER_SANDBOX_IMAGE:?sandbox mode requires BROWSER_SANDBOX_IMAGE (registry the cluster can pull)}"
  export SANDBOX_ORCHESTRATOR_URL SANDBOX_API_KEY BROWSER_SANDBOX_IMAGE
  [ -e "${OPENWEBRL_ROOT}/sandbox/client/sandbox_client.py" ] \
    || { echo "ERROR: Orchard client missing at sandbox/client/sandbox_client.py"; exit 5; }
  echo "sandbox mode: orchestrator=${SANDBOX_ORCHESTRATOR_URL} image=${BROWSER_SANDBOX_IMAGE}"
fi
export PLAYWRIGHT_BROWSERS_PATH="${PLAYWRIGHT_BROWSERS_PATH:-/root/.cache/ms-playwright}"
# --n-parallel comes from SLIME_BROWSER_SANDBOX_MAX_SANDBOXES; the local process
# pool gets headroom above it.
export SLIME_BROWSER_SANDBOX_MAX_SANDBOXES="${SLIME_BROWSER_SANDBOX_MAX_SANDBOXES:-16}"   # paper Table 8
export SLIME_BROWSER_LOCAL_PROCESS_MAX_PROCESSES="${SLIME_BROWSER_LOCAL_PROCESS_MAX_PROCESSES:-32}"
export SLIME_BROWSER_LOCAL_PROCESS_LOG_DIR="${OUTPUT_ROOT}/env_server_logs"
mkdir -p "${SLIME_BROWSER_LOCAL_PROCESS_LOG_DIR}"

# ---- Triton's bundled ptxas is 12.8 and rejects --gpu-name sm_103, so every JIT
# compile fails on B300 with "PTXAS error: Internal Triton PTX codegen error".
# The system 12.9 ptxas accepts it. Same fix as beaker/run_rl_full.sh:128. ----
if [ -x /usr/local/cuda/bin/ptxas ]; then
  export TRITON_PTXAS_PATH="${TRITON_PTXAS_PATH:-/usr/local/cuda/bin/ptxas}"
  echo "TRITON_PTXAS_PATH=${TRITON_PTXAS_PATH}"
fi

# ---- Base image ships mismatched flashinfer (0.5.3) vs jit-cache (0.6.3). ----
export FLASHINFER_DISABLE_VERSION_CHECK=1

# ---- Fail fast rather than scoring a run that could not browse or judge. ----
echo "Probing outbound web access..."
curl -sS --connect-timeout 10 --max-time 20 -o /dev/null -w "example.com -> %{http_code}\n" https://example.com \
  || { echo "ERROR: no outbound web access; local_process rollouts cannot browse. Aborting."; exit 3; }
curl -sS --connect-timeout 10 --max-time 20 -o /dev/null -w "judge -> %{http_code}\n" \
  -H "Authorization: Bearer ${OPENAI_API_KEY:-${JUDGE_API_KEY:-EMPTY}}" "${JUDGE_API_BASE%/}/v1/models" \
  || { echo "ERROR: judge endpoint ${JUDGE_API_BASE} unreachable. Aborting."; exit 4; }

# ---- Chromium for Playwright (python package is in the image's requirements) ----
python -m playwright install chromium || playwright install chromium

# Paper Table 8 defines TWO decoding configs. EVAL_SCORE_MODE selects one:
#   official  -> temp 0.6 / top_p 0.95 / top_k 20   (the Table 2 headline number)
#   noaborted -> temp 0.0 / top_p 1.0  / top_k 1    ("success rate w/o aborted tasks")
# Explicit EVAL_TEMPERATURE/TOP_P/TOP_K still override.
EVAL_SCORE_MODE="${EVAL_SCORE_MODE:-noaborted}"
case "${EVAL_SCORE_MODE}" in
  official)  _T=0.6; _P=0.95; _K=20 ;;
  noaborted) _T=0.0; _P=1.0;  _K=1  ;;
  *) echo "ERROR: EVAL_SCORE_MODE must be official|noaborted, got '${EVAL_SCORE_MODE}'"; exit 9 ;;
esac
export EVAL_TEMPERATURE="${EVAL_TEMPERATURE:-${_T}}"
export EVAL_TOP_P="${EVAL_TOP_P:-${_P}}"
export EVAL_TOP_K="${EVAL_TOP_K:-${_K}}"
export JUDGE_TIMEOUT_SECS="${JUDGE_TIMEOUT_SECS:-120}"   # paper Table 8
echo "=== eval config ==="
echo "BENCH=${BENCH} PROTOCOL=${EVAL_PROTOCOL} TASK_FILE=${TASK_FILE} TASKS=$(wc -l < "${TASK_FILE}")"
echo "CKPT=${CKPT}"
echo "OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "n_parallel=${SLIME_BROWSER_SANDBOX_MAX_SANDBOXES} env_mode=${SLIME_BROWSER_ENV_MODE}"
echo "task_indices=${TASK_INDICES:-<all>}"
echo "score_mode=${EVAL_SCORE_MODE} temp=${EVAL_TEMPERATURE} top_p=${EVAL_TOP_P} top_k=${EVAL_TOP_K} judge_timeout=${JUDGE_TIMEOUT_SECS}"
echo "==================="

bash "${OPENWEBRL_ROOT}/scripts/run_evaluation.sh"

# ---- Score. Results nest one level deeper than compute_eval_success_rate.py
# globs, so point it at each leaf. ----
echo "=== success rate ==="
find "${OUTPUT_ROOT}" -name 'results_task_*.jsonl' -printf '%h\n' | sort -u | while read -r leaf; do
  python "${OPENWEBRL_ROOT}/openwebrl/compute_eval_success_rate.py" "${leaf}" || true
done
