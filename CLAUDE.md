# CLAUDE.md — OpenWebRL

Notes for Claude on how RL works in this repo and how to run it. Verified against the
code on 2026-06-08 (branch `main`). File:line refs may drift — re-check before relying.

## What this repo is

OpenWebRL does **online multi-turn RL for visual web agents** (Qwen3-VL). It is the
`slime` (Megatron + SGLang) RL stack with browser-specific rollout, reward, data, and
eval components bolted on under `openwebrl/`. The RL algorithm is **MM-GRPO** (GRPO with
multimodal samples): generate a group of trajectories per prompt, reward each, and do a
group-relative policy update. No critic by default.

The loop, end to end:

```
prompt (web task) ──► browser rollout (multi-turn, screenshots + tool calls)
                          │  openwebrl/generate_browser.py
                          ▼
                      trajectory samples (tokens, loss_mask, log_probs, images)
                          │
                          ▼
                      reward = format check + VLM-as-judge success
                          │  openwebrl/reward_browser.py
                          ▼
                      GRPO group-relative advantage ──► Megatron policy update
                          │  train.py (slime)
                          ▼
                      repeat for num_rollout steps
```

## The three custom plug-ins (this is the whole integration)

slime is generic; OpenWebRL injects three things via CLI args in the launcher:

| slime arg | points at | role |
|---|---|---|
| `--custom-generate-function-path` | `openwebrl.generate_browser.generate_turn_sample` **or** `…generate_trajectory_sample` | runs a browser episode, returns training samples |
| `--custom-rm-path` | `openwebrl.reward_browser.reward_func` | scores samples (returns one float per sample) |
| `--custom-config-path` | `openwebrl/browser_training_config.yaml` | extra knobs read via `getattr(args, …)` in the two above |

`train.py` is the slime driver loop: `create_placement_groups` → `create_rollout_manager`
→ `create_training_models`, then per step `rollout_manager.generate.remote(rollout_id)`
(rollout + reward happen here) → `actor_model.async_train(...)`. Reward is **not** a
separate phase in `train.py`; it runs inside generation via `--custom-rm-path`.

### Turn-level vs trajectory-level
Controlled by `BROWSER_TRAIN_LEVEL` (`turn` default, or `trajectory`) in the launcher:
- **turn**: `generate_turn_sample` returns a **list** of Samples, one per turn. Keeps only
  the last K screenshots per turn (`--context-num-screenshots`). This is the default.
- **trajectory**: `generate_trajectory_sample` returns **one** Sample with the whole
  episode concatenated.

## Rollout mechanics (`openwebrl/generate_browser.py`)

- Entry fns: `generate_turn_sample` (~L1947) / `generate_trajectory_sample` (~L1594).
  Impls: `_generate_turn_sample_impl` / `_generate_trajectory_sample_impl`.
- Loop: `env.reset()` → for up to `max_steps` (default 16, see browser_training_config.yaml):
  observation (screenshot data-URL + optional a11y tree) → model generates
  `<tool_call>…</tool_call>` → parse → `env.step(actions)` → tool feedback appended as
  `<tool_response>…</tool_response>` (toggle `browser_include_tool_response`).
- Exit conditions: env `terminated`, generation length limit, context-length overflow
  (`rollout_max_context_len`, default 32768), `max_consecutive_parse_failures` (default 3),
  or `max_steps` reached.
- Samples carry `tokens`, `loss_mask` (1 on assistant tokens, 0 on obs/tool), 
  `rollout_log_probs`, and multimodal image inputs. Loss-mask correctness is the thing to
  watch if you touch this file.

### Browser env: two modes (`openwebrl/env/config.yaml`, `mode:`)
- `local_process`: one local `python -m openwebrl.docker.env_server` subprocess per env,
  HTTP on ports 18000–18999. For debugging / small eval. Override: `SLIME_BROWSER_ENV_MODE=local_process`.
- `sandbox` (default): Orchard (K8s) pods, per-episode network isolation, scales to
  hundreds. Needs `SANDBOX_ORCHESTRATOR_URL`, `SANDBOX_API_KEY`, `BROWSER_SANDBOX_IMAGE`.
  Cuts website block rate (25.7%→17.7% on Online-Mind2Web) — matters because GRPO groups
  hammer the same site within one step.
- Env server HTTP API: `/reset`, `/step`, `/exit`, `/health` (`openwebrl/docker/env_server.py`).
- Leftover pods after a crash: `bash scripts/clean_processes.sh`.

## Reward (`openwebrl/reward_browser.py`)

- `reward_func(args, samples, **kwargs)` → list of floats. slime calls it via
  `slime/rollout/rm_hub/`.
- Combination is **hard-gated, not weighted** (despite a weighted-formula comment):
  - format invalid → `0.0`
  - format `format_error_failed` → `-1.0`
  - format valid → `1.0` iff judge says SUCCESS else `0.0`
- Format reward (`compute_format_reward`): valid thinking/action/tool-call tags + parseable.
- Judge = VLM-as-a-judge (`compute_judge_reward` / `…_actionhistory`): sends task + final
  answer + last few screenshots to an LLM, scores SUCCESS→1.0. Retries w/ backoff.
- `JUDGE_API_MODE`:
  - `token` — Azure AD token auth (`AZURE_RESOURCE_NAME`, `AZURE_TOKEN_PATH`)
  - `api_key` — Azure w/ key (`OPENAI_API_KEY`, `OPENAI_API_BASE`)
  - `served` — any OpenAI-compatible endpoint (`JUDGE_API_BASE` or `JUDGE_API_HOST`+`PORT`, optional `JUDGE_API_KEY`)
- Concurrency: `BROWSER_REWARD_CONCURRENCY` (default 8).

## Data

Training prompts are parquet; eval task files are JSONL. Schema (`prompt`, `metadata`):
```python
{"prompt": [{"role": "user", "content": <task>}],
 "metadata": {"task_id": "<prefix>/<id>", "start_url": <url>, "task": <task>}}
```
- Train default: `openwebrl/data/webgym_filtered_popular_2102_cleaned.parquet`
- Eval/val default: `openwebrl/data/webvoyager_val.parquet`, `openwebrl/data/online-mind2web.jsonl`
- Convert your own benchmark JSONL → parquet:
  `python openwebrl/data/convert_benchmark_jsonl_to_parquet.py --input X.jsonl --output X.parquet --task-prefix mybench`
  (JSONL rows need `task_name`, `website`, `task_id`.)

## How to actually run RL

1. **Setup**: `bash set_up.sh && pip install -e . && cp .env.example .env` then edit `.env`.
   Set `SLIME_REPO_ROOT`, `SLIME_MODEL_ROOT`, `SLIME_SAVE_ROOT`, `SLIME_OUTPUT_ROOT`,
   `PYTHONPATH` (this repo + your Megatron-LM checkout).
2. **Env mode**: edit `openwebrl/env/config.yaml` `mode:` (use `local_process` to debug first).
3. **Judge**: set `JUDGE_API_MODE`, `JUDGE_MODEL`, and the matching endpoint vars.
4. **Model**: `MODEL_NAME` (launcher default `OpenWebRL/OpenWebRL-4B-SFT`). RL can start from
   a base VLM, but an SFT warm start is **recommended** — see `sft/README.md`
   (`sft/run_sft_with_llamafactory.sh`), then point `MODEL_NAME` at that checkpoint.
5. **Launch**: `bash scripts/run_browser_Qwen3VL_4B_Instruct.sh` (or `…_8B_…`).
   Submits a Ray job: `ray job submit … -- python3 train.py <MODEL/CKPT/ROLLOUT args>`.

### Key knobs (in the launcher)
`MODEL_NAME`, `TRAIN_DATA`, `EVAL_DATA`, `NUM_GPUS` (=8), `BROWSER_TRAIN_LEVEL`,
`--num-rollout 100`, `--rollout-batch-size 48`, `--n-samples-per-prompt 5` (GRPO group size),
`--advantage-estimator grpo`, `--lr 5e-7`, `PPO_EPOCHS`, `SLIME_BROWSER_SANDBOX_MAX_SANDBOXES`,
`SGLANG_MEM_FRACTION_STATIC`. Episode/rollout limits live in `browser_training_config.yaml`
(`max_steps`, `rollout_max_context_len`, timeouts).

### Evaluate
`bash scripts/run_convert_hf.sh <iter-dir> <hf-model-dir>` (Megatron→HF), then
`MODEL_PATH=<hf> TASK_FILE=openwebrl/data/online-mind2web.jsonl bash scripts/run_evaluation.sh`
(`…_local.sh` for local-process smoke test). Score: `openwebrl/compute_eval_success_rate.py`.

## Gotchas

- **4B launcher currently sources the 8B Megatron config**: in
  `scripts/run_browser_Qwen3VL_4B_Instruct.sh` ~L519-520, `MEGATRON_MODEL_TYPE="qwen3-8B"`
  is active and `"qwen3-4B"` is commented out, even though `MODEL_NAME` is the 4B SFT model.
  If you actually train 4B, flip this to `qwen3-4B`. Confirm with the user — may be intentional.
- README TODO lists "Support RL with Qwen3.5" as **not yet done**; only SFT-with-Qwen3.5 is
  checked. The committed SFT scripts are Qwen3.5; the RL launchers target Qwen3-VL.
- Reward is all-or-nothing (judge-gated), not a weighted sum — don't trust the formula comment.
- Sandbox pods leak on crashes → run `scripts/clean_processes.sh`.
- Secrets (orchestrator URL, judge keys) go in `.env` / env vars, never committed.
