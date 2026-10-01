#!/usr/bin/env bash
# In-container entry point for the RL-step optimisation sweep, launched by
# beaker/launch_rl_sweep.py on one 8-GPU node.
#
# A self-contained launcher (no sed patching of scripts/run_browser_*.sh):
#   - paper stage-1 hyperparameters (OpenWebRL Table 7) as defaults,
#   - Browserbase rollouts: stealth on, proxies OFF (standing rule), 1280x720,
#     all 240 episodes of an iteration in flight at once behind the spawn
#     throttle in openwebrl/env/browserbase_env.py,
#   - a self-hosted WebJudge-7B judge (beaker/run_judge_server.sh) with the
#     single-call AgentTrek om2w prompt as the reward,
#   - train set == eval set: the fixed 48-task om2w subset, so
#     rollout/raw_reward_mean and eval/* measure the same thing.
# Every hyperparameter below is an env knob so one launcher covers the sweep.
set -euo pipefail

OPENWEBRL_ROOT="${OPENWEBRL_ROOT:-/weka/oe-training-default/new_peters/OpenWebRL}"
cd "${OPENWEBRL_ROOT}"
REPO_ROOT="${OPENWEBRL_ROOT}"

# ---- Run identity ----
RUN_NAME="${RUN_NAME:?RUN_NAME is required (used for save dir, W&B run, config copy)}"
SLIME_SAVE_ROOT="${SLIME_SAVE_ROOT:-${OPENWEBRL_ROOT}/outputs/rl_sweep}"
SAVE_DIR="${SLIME_SAVE_DIR:-${SLIME_SAVE_ROOT}/${RUN_NAME}}"
mkdir -p "${SAVE_DIR}"

# ---- Sweepable hyperparameters (defaults = paper stage 1) ----
NUM_ROLLOUT="${NUM_ROLLOUT:-40}"                      # absolute stop index (resume-aware)
BROWSER_MAX_STEPS="${BROWSER_MAX_STEPS:-15}"
RL_LR="${RL_LR:-1e-6}"
PPO_EPOCHS="${PPO_EPOCHS:-2}"
# Reuse the rollout logprobs as the old policy instead of a separate forward pass.
USE_ROLLOUT_LOGPROBS="${USE_ROLLOUT_LOGPROBS:-}"
# USE_TIS=1: truncated importance sampling (https://fengyao.notion.site/off-policy-rl) reweighs
# each token by min(p_learner/p_sampler, TIS_CLIP), correcting the gap between what SGLang
# sampled and what Megatron trains on. Needs the trainer's own old log-probs, so it cannot be
# combined with --use-rollout-logprobs (i.e. needs PPO_EPOCHS=1).
USE_TIS="${USE_TIS:-0}"
TIS_CLIP="${TIS_CLIP:-2.0}"
# SKIP_NONFINITE_GRAD=1: skip the optimizer step when the gradient norm is NaN/Inf instead of
# aborting the job (slime honours Megatron's --no-check-for-nan-in-loss-and-grad that way).
SKIP_NONFINITE_GRAD="${SKIP_NONFINITE_GRAD:-}"
# bshd micro-batches are budgeted on padded tokens (k * padded_len) rather than a
# fixed sample count; "" keeps the fixed --micro-batch-size. Browser turns are ~2k
# tokens, so micro-batch 1-2 left the GPU running 3-6k token forwards.
BSHD_TOKEN_BUDGET="${BSHD_TOKEN_BUDGET:-}"
# 1 to run the output layer only at response positions. The loss reads ~69 of ~1,950
# positions per browser turn, so this cuts the fp32 logits ~28x and is what lets
# BSHD_TOKEN_BUDGET go up.
RESPONSE_ONLY_LOGITS="${RESPONSE_ONLY_LOGITS:-}"
# "ROLLOUT:STEP" to record a kernel-level torch trace of that one optimizer step on rank 0,
# written under outputs/rl_sweep/<run>/prof. Use a rollout >= 1 so kernels are warm.
PROFILE_TRAIN_STEP="${PROFILE_TRAIN_STEP:-}"
# 1 to also record Python call stacks in that trace (names every host-sync call site).
PROFILE_STACKS="${PROFILE_STACKS:-}"
# NCCL collective timeout for the trainer. A rank running the stack profiler can lag the
# others by more than the 10-minute default.
DISTRIBUTED_TIMEOUT_MINUTES="${DISTRIBUTED_TIMEOUT_MINUTES:-}"
# With ATTENTION_BACKEND=auto: force TE's unfused attention only for heads at least this
# wide (129 for Qwen3.5's 256-wide LM heads, which cuDNN cannot do backward for), so the
# vision tower keeps fused/flash kernels.
TE_UNFUSED_MIN_HEAD_DIM="${TE_UNFUSED_MIN_HEAD_DIM:-}"
# 1 to veto TE's cuDNN fused attention at selection time (no backward plan in this image),
# so `auto` falls through to flash-attn where installed, else unfused.
TE_NO_FUSED="${TE_NO_FUSED:-}"
# Regex of parameter names to freeze (slime --freeze-params-name-list), e.g. "vision_model\\..*"
# to leave the vision tower out of training: no backward or activations through it.
FREEZE_PARAMS="${FREEZE_PARAMS:-}"
# 1 to run the vision tower's packed (thd, head_dim <= 128) attention through torch SDPA
# instead of TE's unfused path; same math, tower stays trainable.
VISION_SDPA="${VISION_SDPA:-}"
# 1 to stop handing TransformerEngine the all-ones padding mask megatron-bridge fabricates in
# bshd mode. That mask carries no information but routes every attention layer through TE's
# padding path, which reads per-sequence lengths on the host (~64k GPU->CPU syncs per step).
# Valid because slime right-pads and the loss never reads pad positions.
DROP_PADDING_MASK="${DROP_PADDING_MASK:-}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-48}"
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-5}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-256}"
ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-0.8}"
ROLLOUT_TOP_P="${ROLLOUT_TOP_P:-1.0}"
EVAL_TOP_P="${EVAL_TOP_P:-${ROLLOUT_TOP_P}}"
EVAL_TOP_K="${EVAL_TOP_K:--1}"
EVAL_LABEL="${EVAL_LABEL:-om2w48}"
# Stochastic eval, same temperature as training rollouts (greedy would hide sampling variance).
EVAL_TEMPERATURE="${EVAL_TEMPERATURE:-${ROLLOUT_TEMPERATURE}}"
ROLLOUT_MAX_RESPONSE_LEN="${ROLLOUT_MAX_RESPONSE_LEN:-1024}"
ROLLOUT_MAX_CONTEXT_LEN="${ROLLOUT_MAX_CONTEXT_LEN:-32768}"
CONTEXT_NUM_SCREENSHOTS="${CONTEXT_NUM_SCREENSHOTS:-1}"
RL_SEED="${RL_SEED:-1234}"
EVAL_INTERVAL="${EVAL_INTERVAL:-5}"
# 1 to skip the evaluation that otherwise runs before rollout 0. Set it together with a
# large EVAL_INTERVAL to time training without any eval in the way.
SKIP_EVAL_BEFORE_TRAIN="${SKIP_EVAL_BEFORE_TRAIN:-0}"
SAVE_INTERVAL="${SAVE_INTERVAL:-10}"
# Every SAVE_HF_INTERVAL completed steps (a multiple of SAVE_INTERVAL) also export
# HuggingFace weights to ${SAVE_DIR}/hf/step_<N> for offline evaluation. "" = never.
SAVE_HF_INTERVAL="${SAVE_HF_INTERVAL:-}"
# LOGPROB_DEBUG=1 logs trainer-vs-SGLang log-prob agreement per sample on the first micro-batches;
# IMAGE_TOKEN_DEBUG=1 turns on slime's image-pad-token count check.
LOGPROB_DEBUG="${LOGPROB_DEBUG:-}"
LOGPROB_DUMP="${LOGPROB_DUMP:-}"   # dir: per-token trainer/rollout log-probs of the debugged micro-batches
IMAGE_TOKEN_DEBUG="${IMAGE_TOKEN_DEBUG:-}"
ROLLOUT_TASK_TIMEOUT="${ROLLOUT_TASK_TIMEOUT:-600}"   # Table 7; Browserbase sessions are warm
DYNAMIC_SAMPLING_FILTER="${DYNAMIC_SAMPLING_FILTER:-1}" # drop groups with zero reward std
# Infra knobs
RL_MAX_TOKENS_PER_GPU="${RL_MAX_TOKENS_PER_GPU:-32768}"  # dynamic micro-batching; "" = static mbs 1
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
# Linear-attention layers cannot take packed sequences ("GDN does not support packed
# sequence for now"), and dynamic batching packs. Qwen3.5 therefore needs bshd, which
# slime only allows with static micro-batches.
QKV_FORMAT="${QKV_FORMAT:-thd}"
# TransformerEngine backend for the training forward/backward (flash/fused/unfused/auto).
# Qwen3.5's 256-wide heads have no cuDNN fused backward plan, and flash rejects these
# inputs outright ("No dot product attention backend is available"), so it needs unfused.
ATTENTION_BACKEND="${ATTENTION_BACKEND:-flash}"
# Activation recompute; unfused attention materialises the score matrix, so long samples
# need this to fit.
RL_RECOMPUTE="${RL_RECOMPUTE:-0}"
if [ "${QKV_FORMAT}" = "bshd" ]; then
  RL_MAX_TOKENS_PER_GPU=""
fi
ASYNC_STREAM="${ASYNC_STREAM:-0}"                       # see the block below
if [ "${ASYNC_STREAM}" = "1" ]; then
  # dedicated engines: no Megatron on these GPUs, take the memory; concurrency set below
  SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-0.85}"
else
  SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-0.6}"
  SGLANG_SERVER_CONCURRENCY="${SGLANG_SERVER_CONCURRENCY:-48}"  # per engine; 8 engines
fi
SGLANG_CHUNKED_PREFILL_SIZE="${SGLANG_CHUNKED_PREFILL_SIZE:-8192}"
SGLANG_ATTENTION_BACKEND="${SGLANG_ATTENTION_BACKEND:-}"
# Vision tower attention. Blackwell has no fa3 and fa4 needs the cutlass DSL that is
# missing from this image, so a VLM rollout wants triton_attn; empty = auto.
SGLANG_MM_ATTENTION_BACKEND="${SGLANG_MM_ATTENTION_BACKEND:-}"
SGLANG_DISABLE_CUDA_GRAPH="${SGLANG_DISABLE_CUDA_GRAPH:-0}"
# --enable-precise-embedding-interpolation (ViT position-grid resize with align_corners=True).
# Measured no effect for Qwen3.5 in sglang 0.5.15 (its vectorized path already uses linspace).
SGLANG_PRECISE_EMBED_INTERP="${SGLANG_PRECISE_EMBED_INTERP:-0}"
TP_SIZE="${TP_SIZE:-4}"
NUM_GPUS="${NUM_GPUS:-8}"                                # per node
NUM_NODES="${NUM_NODES:-1}"                              # Beaker replicas (launch_rl_sweep.py --nodes)
# ASYNC_STREAM=1: non-colocated streaming rollout (slime/rollout/streaming_rollout.py).
# ROLLOUT_NUM_GPUS GPUs (on the head node) serve SGLang permanently, the rest train;
# trajectories keep generating while the actor trains. Single node only.
ROLLOUT_NUM_GPUS="${ROLLOUT_NUM_GPUS:-2}"
STREAM_INFLIGHT_GROUPS="${STREAM_INFLIGHT_GROUPS:-}"     # default 1.5 x ROLLOUT_BATCH_SIZE
STREAM_MAX_BACKLOG_GROUPS="${STREAM_MAX_BACKLOG_GROUPS:-}" # default ROLLOUT_BATCH_SIZE
STREAM_MAX_VERSION_LAG="${STREAM_MAX_VERSION_LAG:-}"     # default -1 = keep all
STREAM_UPDATE_EVERY="${STREAM_UPDATE_EVERY:-1}"          # push weights every k steps
# Trainer-only benchmarking on saved rollouts (slime --debug-train-only):
#   SAVE_ROLLOUT_DATA=<path template with {rollout_id}>  save every rollout batch (torch.save)
#   ROLLOUT_ONLY=1                                        generate + save, never train
#   TRAIN_ONLY_ROLLOUT_DATA=<same template>              replay saved batches; no rollout
#     engines, no Browserbase, no eval; every GPU goes to the trainer.
SAVE_ROLLOUT_DATA="${SAVE_ROLLOUT_DATA:-}"
ROLLOUT_ONLY="${ROLLOUT_ONLY:-0}"
TRAIN_ONLY_ROLLOUT_DATA="${TRAIN_ONLY_ROLLOUT_DATA:-}"
if [ -n "${TRAIN_ONLY_ROLLOUT_DATA}" ]; then
  ACTOR_NUM_GPUS="${NUM_GPUS}"
  ROLLOUT_NUM_GPUS=0
  SGLANG_SERVER_CONCURRENCY="${SGLANG_SERVER_CONCURRENCY:-1}"
elif [ "${ASYNC_STREAM}" = "1" ]; then
  ACTOR_NUM_GPUS=$((NUM_GPUS - ROLLOUT_NUM_GPUS))
  # 240 sessions over ROLLOUT_NUM_GPUS engines, plus headroom for eval
  SGLANG_SERVER_CONCURRENCY="${SGLANG_SERVER_CONCURRENCY:-$(( (ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT + 48) / ROLLOUT_NUM_GPUS + 1 ))}"
else
  ACTOR_NUM_GPUS="${NUM_GPUS}"
fi
REPLICA_RANK="${BEAKER_REPLICA_RANK:-0}"
# Rank 0 runs the Ray head, Browserbase cleanup, judge check and train.py; other
# ranks only join the Ray cluster and idle until the head goes away.
IS_HEAD=$([ "${REPLICA_RANK}" = "0" ] && echo 1 || echo 0)
# Ray object store (bytes). Must fit in /dev/shm (gantry --shared-memory) or Ray
# falls back to disk; the 100 GiB default spilled 50-84 GB per iteration.
RAY_OBJECT_STORE_BYTES="${RAY_OBJECT_STORE_BYTES:-400000000000}"

# ---- Data: the fixed 48-task om2w subset, used for train and eval ----
TRAIN_DATA="${TRAIN_DATA:-openwebrl/data/sweep/om2w_train48.parquet}"
EVAL_DATA="${EVAL_DATA:-${TRAIN_DATA}}"
[ -f "${TRAIN_DATA}" ] || { echo "ERROR: ${TRAIN_DATA} not found"; exit 1; }
[ -f "${EVAL_DATA}" ] || { echo "ERROR: ${EVAL_DATA} not found"; exit 1; }

# ---- Model ----
export SLIME_MODEL_ROOT="${SLIME_MODEL_ROOT:-/weka/oe-training-default/new_peters/models}"
MODEL_NAME="${MODEL_NAME:-OpenWebRL/OpenWebRL-4B-SFT}"
# An absolute MODEL_NAME is a local checkpoint dir (our own SFT); anything else is an
# HF repo id cached under SLIME_MODEL_ROOT.
case "${MODEL_NAME}" in
  /*) HF_CHECKPOINT="${MODEL_NAME}" ;;
  *)  HF_CHECKPOINT="${SLIME_MODEL_ROOT}/${MODEL_NAME}" ;;
esac
export HF_HOME="${HF_HOME:-/weka/oe-training-default/new_peters/cache/hf}"
if [ ! -d "${HF_CHECKPOINT}" ]; then
  case "${MODEL_NAME}" in
    /*) echo "ERROR: MODEL_NAME=${MODEL_NAME} does not exist"; exit 1 ;;
  esac
  mkdir -p "${SLIME_MODEL_ROOT}"
  huggingface-cli download "${MODEL_NAME}" --local-dir "${HF_CHECKPOINT}"
fi
# Megatron model definition sourced at the bottom (scripts/model_configs/<name>.sh).
MEGATRON_MODEL_CONFIG="${MEGATRON_MODEL_CONFIG:-qwen3-4B}"
# Rollout prompt/action format: "browser_env" (OpenWebRL's own) or "molmoweb" (the
# computer_use tool-call schema our SFT checkpoints speak).
BROWSER_RESPONSE_FORMAT_MODE="${BROWSER_RESPONSE_FORMAT_MODE:-browser_env}"
# Who plays the episodes: "openwebrl" (openwebrl/generate_browser.py, this repo's agent loop
# and env) or "molmoweb" (openwebrl/generate_molmoweb.py: molmoweb's fara15_mw agent,
# BrowserbaseEnv and trajectory logging in a subprocess per episode, SGLang for sampling;
# each episode saved under ${SAVE_DIR}/rollouts/v<weight_version>/). Trainer and reward
# are identical in both.
ROLLOUT_ENGINE="${ROLLOUT_ENGINE:-openwebrl}"
export MOLMOWEB_ROOT="${MOLMOWEB_ROOT:-/weka/oe-training-default/new_peters/molmoweb-rl}"
export SLIME_MW_SAVE_PROB="${SLIME_MW_SAVE_PROB:-1.0}"        # fraction of episodes written to disk
export SLIME_MW_SYSTEM_MESSAGE="${SLIME_MW_SYSTEM_MESSAGE:-molmoweb}"
export SLIME_MW_ROBUST_NAV="${SLIME_MW_ROBUST_NAV:-1}"
case "${ROLLOUT_ENGINE}" in
  openwebrl) GENERATE_FN="openwebrl.generate_browser.generate_turn_sample" ;;
  molmoweb)
    GENERATE_FN="openwebrl.generate_molmoweb.generate_turn_sample"
    [ -d "${MOLMOWEB_ROOT}/agent" ] || { echo "ERROR: MOLMOWEB_ROOT=${MOLMOWEB_ROOT} is not a molmoweb checkout"; exit 1; }
    # the episode worker imports molmoweb: sync playwright, fasthtml (trajectory.html), tenacity, pydantic
    # fasthtml 0.14 dropped ft.Card, which molmoweb's trajectory.html uses; 0.13.4 is what its eval venv runs
    python -c "import fasthtml.common as ft, tenacity, jinja2, requests, pydantic, playwright.sync_api; ft.Card" 2>/dev/null \
      || pip install --no-cache-dir "python-fasthtml==0.13.4" tenacity jinja2 requests pydantic \
      || { echo "ERROR: could not install the molmoweb worker dependencies"; exit 6; }
    ;;
  *) echo "ERROR: ROLLOUT_ENGINE=${ROLLOUT_ENGINE} (openwebrl|molmoweb)"; exit 1 ;;
esac
# Completed steps kept in the prompt; must match the window the SFT data was built
# with (molmoweb: 10). 0 = unbounded.
export SLIME_BROWSER_MAX_PAST_STEPS="${SLIME_BROWSER_MAX_PAST_STEPS:-0}"
# Readiness-based step instead of fixed timers (openwebrl/env/web_env.py): 9 s -> 0.5 s
# standalone, and the screenshot is taken only once the page has actually rendered.
export SLIME_BROWSER_FAST_STEP="${SLIME_BROWSER_FAST_STEP:-0}"
SLIME_LOAD_CHECKPOINT="${SLIME_LOAD_CHECKPOINT:-}"
# Beaker auto-resume after a preemption re-runs this script unchanged; pick the run's own
# latest checkpoint up so the restart does not begin at rollout 0 (b3ft_rl3 lost 3 restarts).
if [ -z "${SLIME_LOAD_CHECKPOINT}" ] && [ -f "${SAVE_DIR}/latest_checkpointed_iteration.txt" ] && [ -z "${TRAIN_ONLY_ROLLOUT_DATA:-}" ]; then
  SLIME_LOAD_CHECKPOINT="${SAVE_DIR}"
  echo "auto-resume: found ${SAVE_DIR}/latest_checkpointed_iteration.txt ($(cat "${SAVE_DIR}/latest_checkpointed_iteration.txt"))"
fi
SLIME_CKPT_STEP="${SLIME_CKPT_STEP:-}"
LOAD_CHECKPOINT="${SLIME_LOAD_CHECKPOINT:-${HF_CHECKPOINT}}"
if [ -n "${SLIME_LOAD_CHECKPOINT}" ] && [ ! -f "${LOAD_CHECKPOINT}/latest_checkpointed_iteration.txt" ]; then
  echo "ERROR: ${LOAD_CHECKPOINT} is not a Megatron checkpoint root"; exit 1
fi

# ---- Browserbase rollouts ----
export SLIME_BROWSER_ENV_MODE="${SLIME_BROWSER_ENV_MODE:-browserbase}"
if [ "${SLIME_BROWSER_ENV_MODE}" = "browserbase" ]; then
  : "${BROWSERBASE_API_KEY:?browserbase mode requires BROWSERBASE_API_KEY}"
  : "${BROWSERBASE_PROJECT_ID:?browserbase mode requires BROWSERBASE_PROJECT_ID}"
  export BROWSERBASE_API_KEY BROWSERBASE_PROJECT_ID
  export BROWSERBASE_PROXIES="${BROWSERBASE_PROXIES:-false}"          # standing rule: off
  export BROWSERBASE_ADVANCED_STEALTH="${BROWSERBASE_ADVANCED_STEALTH:-true}"
  export BROWSERBASE_SOLVE_CAPTCHAS="${BROWSERBASE_SOLVE_CAPTCHAS:-true}"
  export SLIME_BROWSER_VIEWPORT="${SLIME_BROWSER_VIEWPORT:-1280x720}"
  export BROWSERBASE_SESSION_TIMEOUT_S="${BROWSERBASE_SESSION_TIMEOUT_S:-1800}"
  # Account limits: 250 concurrent, 100 spawns/min. Throttle is in browserbase_env.py.
  export BROWSERBASE_MAX_SESSIONS="${BROWSERBASE_MAX_SESSIONS:-240}"
  export BROWSERBASE_SPAWN_PER_MINUTE="${BROWSERBASE_SPAWN_PER_MINUTE:-90}"
  python -c "import browserbase" 2>/dev/null \
    || pip install --no-cache-dir "browserbase==1.4.0" \
    || { echo "ERROR: could not install the browserbase SDK"; exit 6; }
  # Head only: a worker replica starting late would release the head's live sessions.
  [ "${IS_HEAD}" = "1" ] && { python -m openwebrl.env.browserbase_env --cleanup || true; }
fi
# The whole iteration (batch x samples) in flight at once; the env slot pool caps it.
export SLIME_BROWSER_ROLLOUT_CONCURRENCY="${SLIME_BROWSER_ROLLOUT_CONCURRENCY:-$((ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT))}"
export SLIME_BROWSER_SANDBOX_MANIFEST_DIR="/tmp/slime_browser_sandboxes_${RUN_NAME}"
export SLIME_BROWSER_STEP_CURL_TIMEOUT="${SLIME_BROWSER_STEP_CURL_TIMEOUT:-45}"
export SLIME_BROWSER_STEP_MAX_RETRIES="${SLIME_BROWSER_STEP_MAX_RETRIES:-2}"
export SLIME_BROWSER_STEP_RETRY_DELAY_SECS="${SLIME_BROWSER_STEP_RETRY_DELAY_SECS:-2}"
export SLIME_BROWSER_ENV_EXIT_TIMEOUT_SECS="${SLIME_BROWSER_ENV_EXIT_TIMEOUT_SECS:-15}"
export SLIME_ROLLOUT_ABORT_WAIT_TIMEOUT_SECS="${SLIME_ROLLOUT_ABORT_WAIT_TIMEOUT_SECS:-30}"
export SLIME_BROWSER_ROLLOUT_TASK_TIMEOUT_SECS="${ROLLOUT_TASK_TIMEOUT}"
export SLIME_BROWSER_DEBUG_TRACE_PROB="${SLIME_BROWSER_DEBUG_TRACE_PROB:-0.05}"
export SLIME_BROWSER_LOG_LLM_OUTPUT="${SLIME_BROWSER_LOG_LLM_OUTPUT:-1.0}"
export SLIME_BROWSER_LOG_LLM_OUTPUT_PROB="${SLIME_BROWSER_LOG_LLM_OUTPUT_PROB:-0.05}"
export SLIME_BROWSER_LOG_JUDGE_OUTPUT="${SLIME_BROWSER_LOG_JUDGE_OUTPUT:-1.0}"
export SLIME_BROWSER_LOG_JUDGE_OUTPUT_PROB="${SLIME_BROWSER_LOG_JUDGE_OUTPUT_PROB:-0.5}"
export SLIME_BROWSER_CHAT_TEMPLATE_ENABLE_THINKING="${SLIME_BROWSER_CHAT_TEMPLATE_ENABLE_THINKING:-1}"
export SLIME_BROWSER_THINKING_OPEN_TAG="${SLIME_BROWSER_THINKING_OPEN_TAG:-<think>}"
export SLIME_BROWSER_THINKING_CLOSE_TAG="${SLIME_BROWSER_THINKING_CLOSE_TAG:-</think>}"
export SLIME_BROWSER_APPEND_THINKING_PREFILL="${SLIME_BROWSER_APPEND_THINKING_PREFILL:-0}"
export BROWSER_ENV_SERVER_ACCESS_LOG=0
export SLIME_ROUTER_ACCESS_LOG=0

# ---- Judge: self-hosted WebJudge-7B via its endpoint file, or (JUDGE_PROMPT_VARIANT=molmoweb_om2w)
# molmoweb's Online-Mind2Web WebJudge on o4-mini read from the saved episode dirs ----
export JUDGE_API_MODE="served"
JUDGE_PROMPT_VARIANT="${JUDGE_PROMPT_VARIANT:-agenttrek}"
JUDGE_MAX_ATTACHED_IMGS="${JUDGE_MAX_ATTACHED_IMGS:-3}"
export JUDGE_MAX_TOKENS="${JUDGE_MAX_TOKENS:-2048}"
export JUDGE_TEMPERATURE="${JUDGE_TEMPERATURE:-}"   # "" = endpoint default; 0 = greedy (released judge inference)
export JUDGE_API_KEY="${JUDGE_API_KEY:-EMPTY}"
export BROWSER_REWARD_CONCURRENCY="${BROWSER_REWARD_CONCURRENCY:-32}"
if [ "${JUDGE_PROMPT_VARIANT}" = "molmoweb_om2w" ]; then
  JUDGE_MODEL="${JUDGE_MODEL:-o4-mini}"
  : "${OPENAI_API_KEY:?JUDGE_PROMPT_VARIANT=molmoweb_om2w needs OPENAI_API_KEY (gantry secret PS_OPENAI_API_KEY)}"
  [ "${ROLLOUT_ENGINE}" = "molmoweb" ] || { echo "ERROR: molmoweb_om2w judge needs ROLLOUT_ENGINE=molmoweb (saved episode dirs)"; exit 1; }
  export SLIME_MW_SAVE_PROB=1.0
  JUDGE_ENDPOINT_FILE=""; JUDGE_API_BASE="${JUDGE_API_BASE:-unused}"
  python -c "import backoff, openai" 2>/dev/null || pip install --no-cache-dir backoff openai || { echo "ERROR: judge deps"; exit 6; }
  echo "judge: molmoweb om2w WebJudge on ${JUDGE_MODEL}"
else
  JUDGE_MODEL="${JUDGE_MODEL:-WebJudge-7B}"
fi
# JUDGE_API_BASE=https://api.openai.com (the paper's gpt-4.1 judge over the public API): the key is
# the OPENAI_API_KEY gantry secret and the endpoint file is ignored.
if [ "${JUDGE_API_BASE:-}" = "https://api.openai.com" ]; then
  : "${OPENAI_API_KEY:?JUDGE_API_BASE=https://api.openai.com needs OPENAI_API_KEY (gantry secret PS_OPENAI_API_KEY)}"
  JUDGE_API_KEY="${OPENAI_API_KEY}"; export JUDGE_API_KEY
  JUDGE_ENDPOINT_FILE=""
fi
JUDGE_WAIT_ITERS="${JUDGE_WAIT_ITERS:-120}"   # x30 s; the judge job may sit in the Beaker queue
if [ -n "${JUDGE_ENDPOINT_FILE:-}" ]; then
  for _i in $(seq 1 "${JUDGE_WAIT_ITERS}"); do
    [ -s "${JUDGE_ENDPOINT_FILE}" ] && break
    echo "waiting for judge endpoint file ${JUDGE_ENDPOINT_FILE} (${_i}/${JUDGE_WAIT_ITERS})"; sleep 30
  done
  [ -s "${JUDGE_ENDPOINT_FILE}" ] || { echo "ERROR: judge endpoint file never appeared"; exit 4; }
  JUDGE_API_BASE="$(cat "${JUDGE_ENDPOINT_FILE}")"
fi
: "${JUDGE_API_BASE:?JUDGE_API_BASE or JUDGE_ENDPOINT_FILE is required}"
export JUDGE_API_BASE
if [ "${IS_HEAD}" = "1" ] && [ "${JUDGE_PROMPT_VARIANT}" != "molmoweb_om2w" ]; then
  curl -sSf --connect-timeout 10 --max-time 20 -o /dev/null \
    -H "Authorization: Bearer ${JUDGE_API_KEY}" "${JUDGE_API_BASE%/}/v1/models" \
    || { echo "ERROR: judge ${JUDGE_API_BASE} unreachable. Aborting."; exit 4; }
  echo "judge ok: ${JUDGE_API_BASE} model=${JUDGE_MODEL} variant=${JUDGE_PROMPT_VARIANT}"
fi
if [ "${IS_HEAD}" = "1" ]; then

  # ---- Fail fast without outbound web ----
  curl -sS --connect-timeout 10 --max-time 20 -o /dev/null -w "example.com -> %{http_code}\n" https://example.com \
    || { echo "ERROR: no outbound web access"; exit 3; }
fi

# ---- Qwen3.5 support, installed at job start ----
# The sm_103 image predates Qwen3.5: transformers 4.57 has no qwen3_5 config,
# sglang 0.5.6 cannot serve Qwen3_5ForConditionalGeneration, and megatron-bridge
# 0.3.0rc0 has no Qwen3.5 bridge. These are the versions MSR-Orchard/slime's
# Dockerfile uses for its Qwen3.5-9B RL run.
if [ "${PIP_UPGRADE_QWEN35:-0}" = "1" ]; then
  pip install --no-cache-dir "transformers>=5.8" flash-linear-attention==0.4.1     || { echo "ERROR: transformers/fla upgrade failed"; exit 9; }
  pip install --no-cache-dir --no-deps --no-build-isolation     git+https://github.com/radixark/Megatron-Bridge.git@bridge     || { echo "ERROR: Megatron-Bridge install failed"; exit 9; }
  pip install --no-cache-dir "sglang==${SGLANG_UPGRADE_VERSION:-0.5.12.post1}"     || { echo "ERROR: sglang upgrade failed"; exit 9; }
  python -c "import transformers, sglang; print('qwen35 stack:', transformers.__version__, sglang.__version__)"
fi

# ---- SGLang Qwen3.5 mrope fix (beaker/patch_sglang_qwen35_mrope.py) ----
# The fused CUDA norm+rope kernel in sglang's qwen3_5.py reads mrope (3, T) positions as 1-D, so
# image tokens lose their 2-D rotary components: text log-probs match HF, click-coordinate digits
# are off by 10-35 nats (b3ft_rl2/3/4 all trained on that). Patched in the container before any
# engine starts; SGLANG_MROPE_PATCH=0 disables.
SGLANG_MROPE_PATCH="${SGLANG_MROPE_PATCH:-1}"
if [ "${SGLANG_MROPE_PATCH}" = "1" ]; then
  python beaker/patch_sglang_qwen35_mrope.py || { echo "ERROR: sglang mrope patch failed"; exit 7; }
fi

# ---- B300 / image fixes (same as run_rl_full.sh) ----
export FLASHINFER_DISABLE_VERSION_CHECK=1
if [ -x /usr/local/cuda/bin/ptxas ]; then
  export TRITON_PTXAS_PATH="${TRITON_PTXAS_PATH:-/usr/local/cuda/bin/ptxas}"
fi
export SLIME_REPO_ROOT="${OPENWEBRL_ROOT}"
export PYTHONPATH="${OPENWEBRL_ROOT}:/root/Megatron-LM"
export PYTHONBUFFERED=16
if [ "${NUM_NODES}" -gt 1 ]; then
  # Set by gantry --leader-selection (host networking). Ray wants an IP, not a hostname.
  : "${BEAKER_LEADER_REPLICA_HOSTNAME:?multi-node needs BEAKER_LEADER_REPLICA_HOSTNAME (gantry --leader-selection)}"
  MASTER_ADDR="$(getent hosts "${BEAKER_LEADER_REPLICA_HOSTNAME}" | awk '{print $1; exit}')"
  [ -n "${MASTER_ADDR}" ] || { echo "ERROR: cannot resolve ${BEAKER_LEADER_REPLICA_HOSTNAME}"; exit 5; }
  NODE_IP="$(hostname -I | awk '{print $1}')"
else
  MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
  NODE_IP="${MASTER_ADDR}"
fi
export MASTER_ADDR
export SGLANG_VLM_CACHE_SIZE_MB="${SGLANG_VLM_CACHE_SIZE_MB:-4096}"
export SLIME_HF_VISION_ATTN_IMPL="${SLIME_HF_VISION_ATTN_IMPL:-flash_attention_2}"

# ---- Per-run copy of the browser config (parallel sweep jobs must not share one) ----
CFG_SRC="${OPENWEBRL_ROOT}/openwebrl/browser_training_config.yaml"
CFG_RUN="${OPENWEBRL_ROOT}/openwebrl/browser_training_config_${RUN_NAME}.yaml"
cp "${CFG_SRC}" "${CFG_RUN}"
# 30 s default aborts a few trajectories per step when 240 prompts hit SGLang at once (14 in the baseline run)
INFERENCE_STEP_TIMEOUT="${INFERENCE_STEP_TIMEOUT:-120}"
sed -i -e "s|^rollout_task_timeout_secs:.*|rollout_task_timeout_secs: ${ROLLOUT_TASK_TIMEOUT}.0|" \
       -e "s|^inference_step_timeout_secs:.*|inference_step_timeout_secs: ${INFERENCE_STEP_TIMEOUT}.0|" \
       -e "s|^browser_rollout_concurrency:.*|browser_rollout_concurrency: ${SLIME_BROWSER_ROLLOUT_CONCURRENCY}|" "${CFG_RUN}"
echo "=== config diff ==="; diff "${CFG_SRC}" "${CFG_RUN}" || true

# ---- W&B ----
export WANDB_ENTITY="${WANDB_ENTITY:-ai2-llm}"
WANDB_PROJECT_NAME="${WANDB_PROJECT_NAME:-molmoweb}"
WANDB_LOCAL_DIR="${SAVE_DIR}/wandb"; mkdir -p "${WANDB_LOCAL_DIR}"

echo "=================================================================="
echo "run=${RUN_NAME} save_dir=${SAVE_DIR} resume=${SLIME_LOAD_CHECKPOINT:-<none>}"
echo "model=${MODEL_NAME} megatron_config=${MEGATRON_MODEL_CONFIG} format=${BROWSER_RESPONSE_FORMAT_MODE} max_past_steps=${SLIME_BROWSER_MAX_PAST_STEPS} train=${TRAIN_DATA} eval=${EVAL_LABEL}:${EVAL_DATA}"
echo "rollout_engine=${ROLLOUT_ENGINE} generate_fn=${GENERATE_FN} molmoweb_root=${MOLMOWEB_ROOT} save_prob=${SLIME_MW_SAVE_PROB} sglang_mrope_patch=${SGLANG_MROPE_PATCH}"
echo "num_rollout=${NUM_ROLLOUT} max_steps=${BROWSER_MAX_STEPS} lr=${RL_LR} ppo_epochs=${PPO_EPOCHS} bs=${ROLLOUT_BATCH_SIZE}x${N_SAMPLES_PER_PROMPT} gbs=${GLOBAL_BATCH_SIZE} temp=${ROLLOUT_TEMPERATURE} seed=${RL_SEED}"
echo "tis=${USE_TIS} tis_clip=${TIS_CLIP} use_rollout_logprobs=${USE_ROLLOUT_LOGPROBS:-0}"
echo "env=${SLIME_BROWSER_ENV_MODE} rollout_concurrency=${SLIME_BROWSER_ROLLOUT_CONCURRENCY} proxies=${BROWSERBASE_PROXIES:-n/a} stealth=${BROWSERBASE_ADVANCED_STEALTH:-n/a} viewport=${SLIME_BROWSER_VIEWPORT:-n/a} task_timeout=${ROLLOUT_TASK_TIMEOUT} networkidle_ms=${SLIME_BROWSER_NETWORKIDLE_TIMEOUT_MS:-5000} post_action_ms=${SLIME_BROWSER_POST_ACTION_WAIT_MS:-1000}"
echo "sglang: mem=${SGLANG_MEM_FRACTION_STATIC} conc=${SGLANG_SERVER_CONCURRENCY} prefill=${SGLANG_CHUNKED_PREFILL_SIZE} attn=${SGLANG_ATTENTION_BACKEND:-auto} cudagraph_off=${SGLANG_DISABLE_CUDA_GRAPH}  train: nodes=${NUM_NODES}x${NUM_GPUS}gpu rank=${REPLICA_RANK} tp=${TP_SIZE} qkv=${QKV_FORMAT} mbs=${MICRO_BATCH_SIZE} attn=${ATTENTION_BACKEND} recompute=${RL_RECOMPUTE}/${RECOMPUTE_METHOD:-uniform}/${RECOMPUTE_NUM_LAYERS:-1} max_tokens_per_gpu=${RL_MAX_TOKENS_PER_GPU:-static} bshd_token_budget=${BSHD_TOKEN_BUDGET:-off} response_only_logits=${RESPONSE_ONLY_LOGITS:-off} drop_padding_mask=${DROP_PADDING_MASK:-off}"
echo "shards=${STREAM_SHARDS:-0} async_stream=${ASYNC_STREAM} rollout_gpus=${ROLLOUT_NUM_GPUS} actor_gpus=${ACTOR_NUM_GPUS} inflight=${STREAM_INFLIGHT_GROUPS:-default} backlog=${STREAM_MAX_BACKLOG_GROUPS:-default} max_lag=${STREAM_MAX_VERSION_LAG:-off} update_every=${STREAM_UPDATE_EVERY}"
echo "=================================================================="

# ---- Ray ----
ray stop --force || true; pkill -9 ray || true; pkill -9 slime || true; pkill -9 redis || true; sleep 3
if [ "${IS_HEAD}" = "1" ]; then
  ray start --head --node-ip-address "${MASTER_ADDR}" --num-gpus "${NUM_GPUS}" --disable-usage-stats \
      --object-store-memory "${RAY_OBJECT_STORE_BYTES}"
  if [ "${NUM_NODES}" -gt 1 ]; then
    WANT_GPUS=$((NUM_NODES * NUM_GPUS))
    for _i in $(seq 1 120); do
      HAVE_GPUS=$(python -c "import ray; ray.init(address='auto', log_to_driver=False); print(int(ray.cluster_resources().get('GPU', 0)))" 2>/dev/null || echo 0)
      [ "${HAVE_GPUS}" -ge "${WANT_GPUS}" ] && break
      echo "waiting for Ray workers: ${HAVE_GPUS}/${WANT_GPUS} GPUs (${_i}/120)"; sleep 15
    done
    [ "${HAVE_GPUS}" -ge "${WANT_GPUS}" ] || { echo "ERROR: only ${HAVE_GPUS}/${WANT_GPUS} GPUs joined"; exit 5; }
    echo "Ray cluster ready: ${HAVE_GPUS} GPUs on ${NUM_NODES} nodes"
  fi
else
  # Worker replica: join the head, then idle until the head's GCS disappears.
  for _i in $(seq 1 120); do
    ray start --address "${MASTER_ADDR}:6379" --node-ip-address "${NODE_IP}" --num-gpus "${NUM_GPUS}" \
        --disable-usage-stats --object-store-memory "${RAY_OBJECT_STORE_BYTES}" && break
    echo "waiting for Ray head ${MASTER_ADDR}:6379 (${_i}/120)"; sleep 15
  done
  ray status --address "${MASTER_ADDR}:6379" >/dev/null 2>&1 || { echo "ERROR: could not join Ray head"; exit 5; }
  echo "joined Ray head ${MASTER_ADDR} as worker rank ${REPLICA_RANK}"
  while ray status --address "${MASTER_ADDR}:6379" >/dev/null 2>&1; do sleep 60; done
  echo "Ray head gone; worker rank ${REPLICA_RANK} exiting"
  ray stop --force || true
  exit 0
fi
NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
HAS_NVLINK=$([ "${NVLINK_COUNT}" -gt 0 ] && echo 1 || echo 0)

# ---- Training arguments (mirrors scripts/run_browser_Qwen3VL_4B_Instruct.sh) ----
CKPT_ARGS=(
    --hf-checkpoint "${HF_CHECKPOINT}"
    --save "${SAVE_DIR}"
    --save-interval "${SAVE_INTERVAL}"
)
if [ -n "${SAVE_HF_INTERVAL}" ]; then
  CKPT_ARGS+=(--save-hf "${SAVE_DIR}/hf/step_{step}")
fi
CKPT_ARGS+=(
    --async-save
)
ROLLOUT_ARGS=(
    --data-source-path "${REPO_ROOT}/slime/rollout/data_source.py:RolloutDataSourceWithBuffer"
    --prompt-data "${TRAIN_DATA}"
    --input-key prompt
    --rollout-shuffle
    --use-fault-tolerance
    --custom-generate-function-path "${GENERATE_FN}"
    --custom-rm-path openwebrl.reward_browser.reward_func
    --custom-config-path "openwebrl/browser_training_config_${RUN_NAME}.yaml"
    --max-steps "${BROWSER_MAX_STEPS}"
    --context-num-screenshots "${CONTEXT_NUM_SCREENSHOTS}"
    --judge-max-attached-imgs "${JUDGE_MAX_ATTACHED_IMGS}"
    --judge-api-model "${JUDGE_MODEL}"
    --judge-api-mode "${JUDGE_API_MODE}"
    --judge-prompt-variant "${JUDGE_PROMPT_VARIANT}"
    --judge-timeout-secs "${JUDGE_TIMEOUT_SECS:-120}"
    --turn-history-reasoning-mode full
    --browser-history-reasoning-max-turns 0
    --browser-response-format-mode "${BROWSER_RESPONSE_FORMAT_MODE}"
    --browser-include-tool-response 1
    --rollout-health-check-interval 30
    --rollout-health-check-timeout 30
    --rollout-health-check-first-wait 180
    --num-rollout "${NUM_ROLLOUT}"
    --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
    --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}"
    --rollout-max-response-len "${ROLLOUT_MAX_RESPONSE_LEN}"
    --rollout-max-context-len "${ROLLOUT_MAX_CONTEXT_LEN}"
    --rollout-temperature "${ROLLOUT_TEMPERATURE}"
    --rollout-top-p "${ROLLOUT_TOP_P}"
    --global-batch-size "${GLOBAL_BATCH_SIZE}"
    --seed "${RL_SEED}"
)
case "${DYNAMIC_SAMPLING_FILTER}" in
  1) ROLLOUT_ARGS+=(--dynamic-sampling-filter-path slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonempty_nonzero_std) ;;
  # strips unscored (timed-out/aborted) trajectories, keeps zero-std groups; with
  # --rollout-single-wave an all-unscored group is dropped instead of resubmitted
  nonempty) ROLLOUT_ARGS+=(--dynamic-sampling-filter-path slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonempty) ;;
esac
if [ "${ROLLOUT_SINGLE_WAVE:-0}" = "1" ]; then
  ROLLOUT_ARGS+=(--rollout-single-wave)
fi
EVAL_ARGS=(
    --eval-interval "${EVAL_INTERVAL}"
    --eval-prompt-data "${EVAL_LABEL}" "${EVAL_DATA}"
    --n-samples-per-eval-prompt 1
    --eval-temperature "${EVAL_TEMPERATURE}"
    --eval-top-p "${EVAL_TOP_P}"
    --eval-top-k "${EVAL_TOP_K}"
    --eval-max-response-len "${ROLLOUT_MAX_RESPONSE_LEN}"
)
if [ "${SKIP_EVAL_BEFORE_TRAIN}" = "1" ]; then
  EVAL_ARGS+=(--skip-eval-before-train)
fi
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
if [ "${PPO_EPOCHS}" -gt 1 ] || [ "${USE_ROLLOUT_LOGPROBS}" = "1" ]; then
  if [ "${USE_TIS}" = "1" ]; then
    echo "ERROR: USE_TIS=1 needs the trainer's old log-probs; set PPO_EPOCHS=1 and USE_ROLLOUT_LOGPROBS=0"
    exit 1
  fi
  GRPO_ARGS+=(--use-rollout-logprobs)
fi
if [ "${USE_TIS}" = "1" ]; then
  GRPO_ARGS+=(--use-tis --tis-clip "${TIS_CLIP}")
fi
OPTIMIZER_ARGS=(
    --optimizer adam
    --lr "${RL_LR}"
    --lr-decay-style constant
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.98
    --use-precision-aware-optimizer
)
# B300 has 288 GB; the CPU-offloaded Adam is only needed on 80 GB parts. RL_OPTIMIZER_CPU_OFFLOAD=0 keeps it on GPU.
[ "${RL_OPTIMIZER_CPU_OFFLOAD:-1}" = "1" ] && OPTIMIZER_ARGS+=(--optimizer-cpu-offload)
# Resuming with a different lr: Megatron asserts the scheduler matches the checkpoint.
[ -n "${SLIME_LOAD_CHECKPOINT}" ] && OPTIMIZER_ARGS+=(--override-opt_param-scheduler)
[ "${SKIP_NONFINITE_GRAD}" = "1" ] && OPTIMIZER_ARGS+=(--no-check-for-nan-in-loss-and-grad)
SGLANG_ARGS=(
    --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC}"
    --rollout-num-gpus-per-engine 1
    --sglang-log-level warning
    --sglang-log-level-http warning
    --sglang-server-concurrency "${SGLANG_SERVER_CONCURRENCY}"
    --sglang-max-running-requests "${SGLANG_SERVER_CONCURRENCY}"
    --sglang-chunked-prefill-size "${SGLANG_CHUNKED_PREFILL_SIZE}"
)
[ -n "${SGLANG_ATTENTION_BACKEND}" ] && SGLANG_ARGS+=(--sglang-attention-backend "${SGLANG_ATTENTION_BACKEND}")
[ -n "${SGLANG_MM_ATTENTION_BACKEND}" ] && SGLANG_ARGS+=(--sglang-mm-attention-backend "${SGLANG_MM_ATTENTION_BACKEND}")
[ "${SGLANG_DISABLE_CUDA_GRAPH}" = "1" ] && SGLANG_ARGS+=(--sglang-disable-cuda-graph)
[ "${SGLANG_PRECISE_EMBED_INTERP}" = "1" ] && SGLANG_ARGS+=(--sglang-enable-precise-embedding-interpolation)
BACKEND_ARGS=(
    --train-backend megatron
    --load "${LOAD_CHECKPOINT}"
    --tensor-model-parallel-size "${TP_SIZE}"
    --pipeline-model-parallel-size 1
    --context-parallel-size 1
    --expert-model-parallel-size 1
    --expert-tensor-parallel-size 1
    --micro-batch-size "${MICRO_BATCH_SIZE}"
    --qkv-format "${QKV_FORMAT}"
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --attention-backend "${ATTENTION_BACKEND}"
    --megatron-to-hf-mode bridge
    --moe-token-dispatcher-type alltoall
)
[ "${TP_SIZE}" -gt 1 ] && BACKEND_ARGS+=(--sequence-parallel)
[ -n "${RL_MAX_TOKENS_PER_GPU}" ] && BACKEND_ARGS+=(--use-dynamic-batch-size --max-tokens-per-gpu "${RL_MAX_TOKENS_PER_GPU}")
# RECOMPUTE_METHOD=block with RECOMPUTE_NUM_LAYERS=N checkpoints only the first N layers of
# the stage and stores full activations for the rest: a memory-for-compute trade against the
# default uniform/1 (every layer recomputed, +33% GPU work).
RECOMPUTE_METHOD="${RECOMPUTE_METHOD:-uniform}"
RECOMPUTE_NUM_LAYERS="${RECOMPUTE_NUM_LAYERS:-1}"
[ "${RL_RECOMPUTE}" = "1" ] && BACKEND_ARGS+=(--recompute-granularity full --recompute-method "${RECOMPUTE_METHOD}" --recompute-num-layers "${RECOMPUTE_NUM_LAYERS}")
[ -n "${DISTRIBUTED_TIMEOUT_MINUTES}" ] && BACKEND_ARGS+=(--distributed-timeout-minutes "${DISTRIBUTED_TIMEOUT_MINUTES}")
[ "${IMAGE_TOKEN_DEBUG}" = "1" ] && BACKEND_ARGS+=(--log-megatron-image-token-debug)
[ -n "${FREEZE_PARAMS}" ] && BACKEND_ARGS+=(--freeze-params-name-list "${FREEZE_PARAMS}")

[ -n "${SLIME_CKPT_STEP}" ] && BACKEND_ARGS+=(--ckpt-step "${SLIME_CKPT_STEP}")
WANDB_ARGS=()
if [ -n "${WANDB_API_KEY:-}" ]; then
  WANDB_ARGS=(
    --use-wandb
    --wandb-project "${WANDB_PROJECT_NAME}"
    --wandb-team "${WANDB_ENTITY}"
    --wandb-group "${RUN_NAME}"
    --wandb-dir "${WANDB_LOCAL_DIR}"
    --wandb-key "${WANDB_API_KEY}"
    --disable-wandb-random-suffix
  )
fi
if [ "${ASYNC_STREAM}" = "1" ]; then
  [ "${NUM_NODES}" = "1" ] || { echo "ERROR: ASYNC_STREAM=1 is single-node only"; exit 1; }
  MISC_ARGS=(
      --actor-num-nodes 1
      --actor-num-gpus-per-node "${ACTOR_NUM_GPUS}"
      --rollout-num-gpus "${ROLLOUT_NUM_GPUS}"
  )
  ROLLOUT_ARGS+=(--rollout-function-path slime.rollout.streaming_rollout.generate_rollout)
  [ -n "${SAVE_ROLLOUT_DATA}" ] && ROLLOUT_ARGS+=(--save-debug-rollout-data "${SAVE_ROLLOUT_DATA}")
  [ "${ROLLOUT_ONLY}" = "1" ] && MISC_ARGS+=(--debug-rollout-only)
  if [ -n "${TRAIN_ONLY_ROLLOUT_DATA}" ]; then
    MISC_ARGS+=(--debug-train-only --load-debug-rollout-data "${TRAIN_ONLY_ROLLOUT_DATA}")
  fi
  # Non-colocated trainer never loads TorchMemorySaver, so the allocator knob
  # is safe here. TP1 at 16k left 33 GiB reserved-but-unallocated and OOM'd on
  # an 18 GiB logits tensor (s3 step 2).
  TRAIN_ENV_JSON="{"
  if [ "${RL_EXPANDABLE_SEGMENTS:-0}" = "1" ]; then
    TRAIN_ENV_JSON="${TRAIN_ENV_JSON}\"PYTORCH_CUDA_ALLOC_CONF\":\"expandable_segments:True\","
  fi
  # TransformerEngine backend selection for the training forward/backward. cuDNN's
  # fused attention has no backward plan for every head size (Qwen3.5 uses 256), which
  # surfaces as "cuDNN Error: No valid execution plans built" in fused_attn_bwd;
  # NVTE_FUSED_ATTN=0 falls back to flash/unfused.
  if [ -n "${NVTE_FUSED_ATTN:-}" ]; then
    TRAIN_ENV_JSON="${TRAIN_ENV_JSON}\"NVTE_FUSED_ATTN\":\"${NVTE_FUSED_ATTN}\","
  fi
  if [ -n "${NVTE_FLASH_ATTN:-}" ]; then
    TRAIN_ENV_JSON="${TRAIN_ENV_JSON}\"NVTE_FLASH_ATTN\":\"${NVTE_FLASH_ATTN}\","
  fi
  # NVTE_DEBUG=1 NVTE_DEBUG_LEVEL=2 makes TE log why each attention backend was accepted or
  # rejected for every new input shape.
  if [ -n "${NVTE_DEBUG:-}" ]; then
    TRAIN_ENV_JSON="${TRAIN_ENV_JSON}\"NVTE_DEBUG\":\"${NVTE_DEBUG}\",\"NVTE_DEBUG_LEVEL\":\"${NVTE_DEBUG_LEVEL:-2}\","
  fi
  if [ -n "${BSHD_TOKEN_BUDGET}" ]; then
    TRAIN_ENV_JSON="${TRAIN_ENV_JSON}\"SLIME_BSHD_TOKEN_BUDGET\":\"${BSHD_TOKEN_BUDGET}\","
  fi
  if [ -n "${RESPONSE_ONLY_LOGITS}" ]; then
    TRAIN_ENV_JSON="${TRAIN_ENV_JSON}\"SLIME_RESPONSE_ONLY_LOGITS\":\"${RESPONSE_ONLY_LOGITS}\","
  fi
  if [ -n "${DROP_PADDING_MASK}" ]; then
    TRAIN_ENV_JSON="${TRAIN_ENV_JSON}\"SLIME_BSHD_DROP_PADDING_MASK\":\"${DROP_PADDING_MASK}\","
  fi
  if [ -n "${TE_UNFUSED_MIN_HEAD_DIM}" ]; then
    TRAIN_ENV_JSON="${TRAIN_ENV_JSON}\"SLIME_TE_UNFUSED_MIN_HEAD_DIM\":\"${TE_UNFUSED_MIN_HEAD_DIM}\","
  fi
  if [ -n "${TE_NO_FUSED}" ]; then
    TRAIN_ENV_JSON="${TRAIN_ENV_JSON}\"SLIME_TE_NO_FUSED\":\"${TE_NO_FUSED}\","
  fi
  if [ -n "${SAVE_HF_INTERVAL}" ]; then
    TRAIN_ENV_JSON="${TRAIN_ENV_JSON}\"SLIME_SAVE_HF_INTERVAL\":\"${SAVE_HF_INTERVAL}\","
  fi
  if [ -n "${LOGPROB_DEBUG}" ]; then
    TRAIN_ENV_JSON="${TRAIN_ENV_JSON}\"SLIME_LOGPROB_DEBUG\":\"${LOGPROB_DEBUG}\","
  fi
  if [ -n "${LOGPROB_DUMP}" ]; then
    TRAIN_ENV_JSON="${TRAIN_ENV_JSON}\"SLIME_LOGPROB_DUMP\":\"${LOGPROB_DUMP}\","
  fi
  if [ -n "${VISION_SDPA}" ]; then
    TRAIN_ENV_JSON="${TRAIN_ENV_JSON}\"SLIME_VISION_SDPA\":\"${VISION_SDPA}\","
  fi
  if [ -n "${PROFILE_TRAIN_STEP}" ]; then
    PROF_DIR="${OPENWEBRL_ROOT}/outputs/rl_sweep/${RUN_NAME}/prof"
    mkdir -p "${PROF_DIR}"
    TRAIN_ENV_JSON="${TRAIN_ENV_JSON}\"SLIME_PROFILE_TRAIN_STEP\":\"${PROFILE_TRAIN_STEP}\",\"SLIME_PROFILE_DIR\":\"${PROF_DIR}\","
    [ "${PROFILE_STACKS}" = "1" ] && TRAIN_ENV_JSON="${TRAIN_ENV_JSON}\"SLIME_PROFILE_STACKS\":\"1\","
  fi
  TRAIN_ENV_JSON="${TRAIN_ENV_JSON%,}}"
  if [ "${TRAIN_ENV_JSON}" != "{}" ]; then
    MISC_ARGS+=(--train-env-vars "${TRAIN_ENV_JSON}")
  fi
else
  MISC_ARGS=(
      --actor-num-nodes "${NUM_NODES}"
      --actor-num-gpus-per-node "${NUM_GPUS}"
      --colocate
  )
fi

# Megatron model definition. Qwen3-VL-4B (OpenWebRL-4B-SFT) needs the 5M rotary base;
# other configs carry their own --rotary-base.
if [ "${MEGATRON_MODEL_CONFIG}" = "qwen3-4B" ]; then
  export MODEL_ARGS_ROTARY_BASE="${MODEL_ARGS_ROTARY_BASE:-5000000}"
fi
source "${REPO_ROOT}/scripts/model_configs/${MEGATRON_MODEL_CONFIG}.sh"

mkdir -p "${SAVE_DIR}/run_config"
cp "${BASH_SOURCE[0]}" "${SAVE_DIR}/run_config/launch_script_snapshot.sh"
cp "${CFG_RUN}" "${SAVE_DIR}/run_config/browser_training_config.yaml"
( set -o posix; set ) | grep -v -i "key\|token\|secret" > "${SAVE_DIR}/run_config/shell_vars.env" || true

# Ray workers inherit the shell env of `ray start`; the JSON re-states what the
# reward/rollout code reads so a runtime-env reset cannot drop it.
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${REPO_ROOT}:/root/Megatron-LM/\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"no_proxy\": \"127.0.0.1,${MASTER_ADDR}\",
    \"MASTER_ADDR\": \"${MASTER_ADDR}\",
    \"JUDGE_API_BASE\": \"${JUDGE_API_BASE}\",
    \"JUDGE_ENDPOINT_FILE\": \"${JUDGE_ENDPOINT_FILE:-}\",
    \"JUDGE_API_KEY\": \"${JUDGE_API_KEY}\",
    \"OPENAI_API_KEY\": \"${OPENAI_API_KEY:-}\",
    \"JUDGE_MAX_TOKENS\": \"${JUDGE_MAX_TOKENS}\",
    \"JUDGE_TEMPERATURE\": \"${JUDGE_TEMPERATURE}\",
    \"BROWSER_REWARD_CONCURRENCY\": \"${BROWSER_REWARD_CONCURRENCY}\",
    \"SGLANG_VLM_CACHE_SIZE_MB\": \"${SGLANG_VLM_CACHE_SIZE_MB}\",
    \"SLIME_BROWSER_ENV_MODE\": \"${SLIME_BROWSER_ENV_MODE}\",
    \"SLIME_BROWSER_VIEWPORT\": \"${SLIME_BROWSER_VIEWPORT:-}\",
    \"SLIME_BROWSER_ROLLOUT_CONCURRENCY\": \"${SLIME_BROWSER_ROLLOUT_CONCURRENCY}\",
    \"SLIME_BROWSER_MAX_PAST_STEPS\": \"${SLIME_BROWSER_MAX_PAST_STEPS}\",
    \"SLIME_BROWSER_FAST_STEP\": \"${SLIME_BROWSER_FAST_STEP:-0}\",
    \"SLIME_BROWSER_ROLLOUT_TASK_TIMEOUT_SECS\": \"${SLIME_BROWSER_ROLLOUT_TASK_TIMEOUT_SECS}\",
    \"SLIME_BROWSER_NETWORKIDLE_TIMEOUT_MS\": \"${SLIME_BROWSER_NETWORKIDLE_TIMEOUT_MS:-5000}\",
    \"SLIME_BROWSER_POST_ACTION_WAIT_MS\": \"${SLIME_BROWSER_POST_ACTION_WAIT_MS:-1000}\",
    \"BROWSERBASE_API_KEY\": \"${BROWSERBASE_API_KEY:-}\",
    \"BROWSERBASE_PROJECT_ID\": \"${BROWSERBASE_PROJECT_ID:-}\",
    \"BROWSERBASE_PROXIES\": \"${BROWSERBASE_PROXIES:-false}\",
    \"BROWSERBASE_ADVANCED_STEALTH\": \"${BROWSERBASE_ADVANCED_STEALTH:-true}\",
    \"BROWSERBASE_SOLVE_CAPTCHAS\": \"${BROWSERBASE_SOLVE_CAPTCHAS:-true}\",
    \"BROWSERBASE_SESSION_TIMEOUT_S\": \"${BROWSERBASE_SESSION_TIMEOUT_S:-1800}\",
    \"BROWSERBASE_MAX_SESSIONS\": \"${BROWSERBASE_MAX_SESSIONS:-240}\",
    \"BROWSERBASE_SPAWN_PER_MINUTE\": \"${BROWSERBASE_SPAWN_PER_MINUTE:-90}\",
    \"SLIME_BROWSER_SANDBOX_MANIFEST_DIR\": \"${SLIME_BROWSER_SANDBOX_MANIFEST_DIR}\",
    \"SLIME_STREAM_INFLIGHT_GROUPS\": \"${STREAM_INFLIGHT_GROUPS}\",
    \"SLIME_STREAM_MAX_BACKLOG_GROUPS\": \"${STREAM_MAX_BACKLOG_GROUPS}\",
    \"SLIME_STREAM_MIN_INFLIGHT_GROUPS\": \"${STREAM_MIN_INFLIGHT_GROUPS:-0}\",
    \"SLIME_STREAM_SHARDS\": \"${STREAM_SHARDS:-0}\",
    \"SLIME_STREAM_MAX_VERSION_LAG\": \"${STREAM_MAX_VERSION_LAG}\",
    \"SLIME_STREAM_UPDATE_EVERY\": \"${STREAM_UPDATE_EVERY}\",
    \"SLIME_STREAM_DRAIN_BEFORE_EVAL\": \"${STREAM_DRAIN_BEFORE_EVAL:-1}\",
    \"SLIME_STREAM_EVAL_DRAIN_SECS\": \"${STREAM_EVAL_DRAIN_SECS:-900}\",
    \"SLIME_MM_PIXEL_DTYPE\": \"${MM_PIXEL_DTYPE:-bf16}\",
    \"SLIME_ENCODE_IN_THREAD\": \"${ENCODE_IN_THREAD:-1}\",
    \"SLIME_STREAM_PREFETCH\": \"${STREAM_PREFETCH:-0}\",
    \"SLIME_ENCODE_EXECUTOR\": \"${ENCODE_EXECUTOR:-thread}\",
    \"MOLMOWEB_ROOT\": \"${MOLMOWEB_ROOT}\",
    \"SLIME_MW_SAVE_PROB\": \"${SLIME_MW_SAVE_PROB}\",
    \"SLIME_MW_SYSTEM_MESSAGE\": \"${SLIME_MW_SYSTEM_MESSAGE}\",
    \"SLIME_MW_ROBUST_NAV\": \"${SLIME_MW_ROBUST_NAV}\"
  }
}"
export no_proxy="127.0.0.1"

# Free-form passthrough for one-off experiments, e.g. SLIME_EXTRA_ARGS="--dump-details /path".
[ -n "${SLIME_EXTRA_ARGS:-}" ] && MISC_ARGS+=(${SLIME_EXTRA_ARGS})
echo "train.py args:" "${MODEL_ARGS[@]}" "${CKPT_ARGS[@]}" "${ROLLOUT_ARGS[@]}" "${EVAL_ARGS[@]}" \
  "${GRPO_ARGS[@]}" "${OPTIMIZER_ARGS[@]}" "${SGLANG_ARGS[@]}" "${BACKEND_ARGS[@]}" "${MISC_ARGS[@]}"
ray job submit --address="http://127.0.0.1:8265" \
    --runtime-env-json="${RUNTIME_ENV_JSON}" \
    -- python3 train.py \
    "${MODEL_ARGS[@]}" \
    "${CKPT_ARGS[@]}" \
    "${ROLLOUT_ARGS[@]}" \
    "${EVAL_ARGS[@]}" \
    "${GRPO_ARGS[@]}" \
    "${OPTIMIZER_ARGS[@]}" \
    "${SGLANG_ARGS[@]}" \
    "${BACKEND_ARGS[@]}" \
    "${MISC_ARGS[@]}" \
    "${WANDB_ARGS[@]}"

bash scripts/clean_processes.sh || true
