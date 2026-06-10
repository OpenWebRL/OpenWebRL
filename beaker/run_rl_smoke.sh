#!/usr/bin/env bash
# ==============================================================================
# In-container entry script for a Tier-3 OpenWebRL GRPO *smoke* run on Beaker.
#
# Launched by beaker/launch_rl_smoke.py via `gantry run ... -- <this script>`.
# Runs on one 8-GPU node. The OpenWebRL repo is read from the Weka mount (not
# uploaded by gantry), and the slime/Megatron/SGLang stack comes from the
# Beaker image. This script:
#   1. wires paths + judge + browser env for local_process live-web browsing,
#   2. installs Chromium for Playwright (python pkg ships in the image deps),
#   3. patches the canonical launcher down to a 2-rollout smoke scale + fixes
#      the qwen3-8B->qwen3-4B model-config line, in a throwaway copy,
#   4. runs it.
# ==============================================================================
set -euo pipefail
set -x

OPENWEBRL_ROOT="${OPENWEBRL_ROOT:-/weka/oe-training-default/new_peters/OpenWebRL}"
cd "${OPENWEBRL_ROOT}"

# ---- Paths ----
export SLIME_REPO_ROOT="${OPENWEBRL_ROOT}"
export PYTHONPATH="${OPENWEBRL_ROOT}:/root/Megatron-LM"
export SLIME_MODEL_ROOT="${SLIME_MODEL_ROOT:-/weka/oe-training-default/new_peters/models}"
export SLIME_SAVE_ROOT="${SLIME_SAVE_ROOT:-/weka/oe-training-default/new_peters/OpenWebRL/outputs/rl_smoke}"
export SLIME_OUTPUT_ROOT="${SLIME_OUTPUT_ROOT:-${SLIME_SAVE_ROOT}/debug}"
mkdir -p "${SLIME_SAVE_ROOT}" "${SLIME_OUTPUT_ROOT}"

# ---- HuggingFace (needs online to fetch OpenWebRL/OpenWebRL-4B-SFT) ----
export HF_HUB_OFFLINE=0
export TRANSFORMERS_OFFLINE=0
export HF_HOME="${HF_HOME:-/weka/oe-training-default/new_peters/cache/hf}"
# HUGGINGFACE_HUB_TOKEN is injected as a secret-env by gantry.

# ---- Browser env: live-web via local subprocess Chromium ----
export SLIME_BROWSER_ENV_MODE=local_process
export PLAYWRIGHT_BROWSERS_PATH="${PLAYWRIGHT_BROWSERS_PATH:-/root/.cache/ms-playwright}"
export SLIME_BROWSER_LOCAL_PROCESS_MAX_PROCESSES="${SLIME_BROWSER_LOCAL_PROCESS_MAX_PROCESSES:-8}"
# Put per-env env_server logs on Weka so server-side reset/Chromium errors are
# readable after the run (default is a node-local /tmp dir we can't reach).
export SLIME_BROWSER_LOCAL_PROCESS_LOG_DIR="${SLIME_SAVE_ROOT}/env_server_logs"
mkdir -p "${SLIME_BROWSER_LOCAL_PROCESS_LOG_DIR}"

# ---- Judge: standard OpenAI key -> served mode (api_key mode is Azure-only) ----
export JUDGE_API_MODE="${JUDGE_API_MODE:-served}"
export JUDGE_API_BASE="${JUDGE_API_BASE:-https://api.openai.com}"
export JUDGE_MODEL="${JUDGE_MODEL:-gpt-4.1}"
# OPENAI_API_KEY injected as secret-env; served mode falls back to it for JUDGE_API_KEY.

# ---- Smoke-scale knobs that ARE env-overridable in the launcher ----
export BROWSER_MAX_STEPS="${BROWSER_MAX_STEPS:-4}"

# ---- Image packaging workaround: base image ships mismatched flashinfer
# (0.5.3) vs flashinfer-jit-cache (0.6.3). Bypass the strict version check so
# SGLang engines start; revisit by pinning matching versions if kernels fail.
export FLASHINFER_DISABLE_VERSION_CHECK=1

# ---- Disable W&B for the smoke test. The launcher hardcodes the authors'
# entity (yangrui_rl), which our key can't write to (permission denied). The
# launcher only enables W&B when WANDB_API_KEY is non-empty, so unset it.
# For a real run, set your own WANDB_ENTITY and re-enable.
unset WANDB_API_KEY

# ---- Connectivity probe: fail fast if live web is unreachable ----
echo "Probing outbound web access..."
curl -sS --connect-timeout 10 --max-time 20 -o /dev/null -w "example.com -> %{http_code}\n" https://example.com \
  || { echo "ERROR: no outbound web access; local_process rollouts cannot browse. Aborting."; exit 3; }

# ---- Chromium for Playwright (python package is in the image's requirements) ----
python -m playwright install chromium || playwright install chromium

# ---- Patch the canonical launcher to smoke scale ----
# Keep the copy inside scripts/ so the launcher's REPO_ROOT=$(dirname)/.. math
# still resolves to the repo root (model configs, data, PYTHONPATH depend on it).
SRC="${OPENWEBRL_ROOT}/scripts/run_browser_Qwen3VL_4B_Instruct.sh"
SMOKE="${OPENWEBRL_ROOT}/scripts/run_browser_Qwen3VL_4B_smoke.sh"
cp "${SRC}" "${SMOKE}"
sed -i \
  -e 's/--num-rollout 100/--num-rollout 2/' \
  -e 's/--rollout-batch-size 48/--rollout-batch-size 2/' \
  -e 's/--n-samples-per-prompt 5/--n-samples-per-prompt 8/' \
  -e 's/--global-batch-size 256/--global-batch-size 4/' \
  -e 's/--eval-interval 5/--eval-interval 100000/' \
  -e 's/^MEGATRON_MODEL_TYPE="qwen3-8B"/MEGATRON_MODEL_TYPE="qwen3-4B"/' \
  "${SMOKE}"

echo "=== smoke launcher diff vs canonical ==="
diff "${SRC}" "${SMOKE}" || true
echo "========================================"

bash "${SMOKE}"
