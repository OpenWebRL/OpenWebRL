"""slime custom generate function that runs rollouts with molmoweb's harness.

Selected with ``--custom-generate-function-path openwebrl.generate_molmoweb.generate_turn_sample``
(``ROLLOUT_ENGINE=molmoweb`` in beaker/run_rl_sweep.sh). Same contract as
``openwebrl.generate_browser.generate_turn_sample``: one trajectory in, one slime Sample
per agent turn out, so the trainer, the reward function and every downstream metric are
untouched.

What changes is who plays the episode. Each trajectory is a subprocess running
``openwebrl/molmoweb_rollout_worker.py``: molmoweb's ``Fara15VLLMAgent`` (the ``fara15_mw``
eval agent -- same prompt, history window, image budget, parser and action mapping the
molmoweb harness evaluates with), molmoweb's ``BrowserbaseEnv`` and molmoweb's
``log_episode`` (trajectory.json + trajectory.html + screenshots per episode under
``<save>/rollouts/v<weight_version>/<task>__s<index>/``). The agent's model call goes to
slime's SGLang server and records prompt text, images, token ids and log-probs per turn;
this module turns those records into Samples with the same encoder
(``_encode_with_processor``) the original path uses, so the token stream the trainer sees is
built exactly as before.

Env knobs (all optional):
  MOLMOWEB_ROOT            molmoweb checkout to import (default: a pinned worktree, see below)
  SLIME_MW_CHAT_TEMPLATE   jinja file rendered for prompts (default: MOLMOWEB_ROOT/beaker/
                           fara_instruct_chat_template.jinja, the template the eval serves)
  SLIME_MW_SAVE_PROB       fraction of episodes whose trajectory dir is written (default 1.0)
  SLIME_MW_SYSTEM_MESSAGE  system prompt (default "molmoweb", the SFT sentinel)
  SLIME_MW_ROBUST_NAV      1/0 molmoweb's robust start-URL navigation (default 1, as om2w eval)
  SLIME_MW_WORKER_PYTHON   interpreter for the worker (default: this one)
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from loguru import logger

from openwebrl import generate_browser as gb
from openwebrl.molmoweb_rollout_worker import _release_session
from slime.rollout.sglang_rollout import GenerateState
from slime.rollout.streaming_rollout import current_weight_version
from slime.utils.types import Sample

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
_WORKER_PATH = os.path.join(_THIS_DIR, "molmoweb_rollout_worker.py")
DEFAULT_MOLMOWEB_ROOT = "/weka/oe-training-default/new_peters/molmoweb-rl"

_POOL: ThreadPoolExecutor | None = None
_CHAT_TEMPLATE: str | None = None


def _molmoweb_root() -> str:
    root = os.environ.get("MOLMOWEB_ROOT") or DEFAULT_MOLMOWEB_ROOT
    if not os.path.isdir(os.path.join(root, "agent")):
        raise FileNotFoundError(f"MOLMOWEB_ROOT={root} is not a molmoweb checkout")
    return root


def _chat_template() -> str:
    global _CHAT_TEMPLATE
    if _CHAT_TEMPLATE is None:
        path = os.environ.get("SLIME_MW_CHAT_TEMPLATE") or os.path.join(
            _molmoweb_root(), "beaker", "fara_instruct_chat_template.jinja"
        )
        with open(path) as f:
            _CHAT_TEMPLATE = f.read()
        logger.info(f"molmoweb rollouts: chat template {path}")
    return _CHAT_TEMPLATE


def _pool(args: Any) -> ThreadPoolExecutor:
    global _POOL
    if _POOL is None:
        n = gb._get_browser_rollout_concurrency(args) or 64
        _POOL = ThreadPoolExecutor(max_workers=n, thread_name_prefix="mw_episode")
    return _POOL


def _save_root(args: Any) -> str:
    return os.path.join(getattr(args, "save", None) or os.environ.get("SLIME_SAVE_DIR", "."), "rollouts")


def _sanitize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)[:120]


def _hard_timeout_secs(args: Any) -> float:
    task = gb._get_rollout_task_timeout_secs(args) or 900.0
    setup = float(getattr(args, "rollout_setup_grace_secs", 300.0))
    return task + setup + 120.0  # worker enforces the task budget itself; this is the backstop


def _run_worker(work_dir: str, hard_timeout: float) -> dict[str, Any]:
    """Blocking: run one episode subprocess, return its result dict."""
    python = os.environ.get("SLIME_MW_WORKER_PYTHON") or sys.executable
    env = dict(os.environ)
    env.pop("RAY_ADDRESS", None)
    log_path = os.path.join(work_dir, "worker.log")
    with open(log_path, "w") as log_f:
        try:
            subprocess.run(
                # by file path, not -m: importing the openwebrl package pulls in torch and
                # slime, which the worker must not pay for on every episode
                [python, _WORKER_PATH, work_dir],
                stdout=log_f, stderr=subprocess.STDOUT, env=env, cwd=_REPO_ROOT,
                timeout=hard_timeout, check=False,
            )
        except subprocess.TimeoutExpired:
            sid_path = os.path.join(work_dir, "session_id")
            sid = open(sid_path).read().strip() if os.path.exists(sid_path) else None
            _release_session(sid)
            return {
                "status": "aborted",
                "terminate_reason": f"generation_error: rollout_task_timeout after {hard_timeout:.0f}s (worker killed)",
                "turns": [], "messages": [], "screenshots": [], "total_steps": 0,
                "final_answer": "", "rollout_dir": None, "bb_session_id": sid,
            }
    result_path = os.path.join(work_dir, "result.json")
    if not os.path.exists(result_path):
        tail = ""
        try:
            with open(log_path) as f:
                tail = f.read()[-2000:]
        except OSError:
            pass
        return {
            "status": "aborted", "terminate_reason": "generation_error: worker produced no result",
            "error": tail, "turns": [], "messages": [], "screenshots": [], "total_steps": 0,
            "final_answer": "", "rollout_dir": None, "bb_session_id": None,
        }
    with open(result_path) as f:
        return json.load(f)


_STATUS = {
    "completed": Sample.Status.COMPLETED,
    "truncated": Sample.Status.TRUNCATED,
    "aborted": Sample.Status.ABORTED,
    "failed": Sample.Status.FAILED,
}


async def _build_turn_samples(
    args: Any, sample: Sample, task_data: dict[str, Any], result: dict[str, Any], state: GenerateState
) -> list[Sample]:
    tokenizer, processor = state.tokenizer, state.processor
    gb._set_im_end_id(tokenizer)
    status = _STATUS[result["status"]]
    screenshots: list[str] = result["screenshots"]
    turn_samples: list[Sample] = []
    step_env_secs = result.get("step_env_secs") or []

    for step, turn in enumerate(result["turns"]):
        img_list = [screenshots[i] for i in turn["image_idx"]]
        t0 = time.monotonic()
        input_tokens, image_grid_thw, _pil, mm_train = await gb._encode_async(
            processor, tokenizer, turn["prompt_text"], img_list
        )
        encode_secs = time.monotonic() - t0

        new_tokens = list(turn["tokens"])
        new_logprobs = [float(x) for x in turn["logprobs"]]
        response_text = gb._ensure_im_end_w_new_line(turn["response_text"], new_tokens, new_logprobs)

        ts = Sample(
            index=gb._make_turn_sample_index(sample.index, step),
            group_index=sample.group_index,
            prompt=turn["prompt_text"],
            response_length=0,
            tokens=input_tokens,
            loss_mask=[],
            rollout_log_probs=[],
            metadata=dict(task_data),
            status=status,
        )
        ts.metadata.update({
            "trajectory_id": sample.index,
            "turn_sample_id": ts.index,
            "turn_index": step,
            "is_last_turn": False,
            "image_grid_thw": image_grid_thw,
            "step_encode_secs": encode_secs,
            "step_infer_secs": float(turn.get("infer_secs") or 0.0),
            "step_env_secs": float(step_env_secs[step]) if step < len(step_env_secs) else 0.0,
            "weight_version": current_weight_version(),
            "terminate_reason": result["terminate_reason"],
            "total_steps": result["total_steps"],
            "mw_rollout_dir": result.get("rollout_dir"),
        })
        if step == 0:
            ts.metadata["env_setup_secs"] = float(result.get("env_setup_secs") or 0.0)
            ts.metadata["step_env_secs"] += ts.metadata["env_setup_secs"]
        ts.multimodal_inputs = {"images": img_list}
        ts.multimodal_train_inputs = mm_train
        gb._append_to_sample(ts, new_tokens, new_logprobs, loss_mask_val=1)
        ts.response = response_text
        ts.response_length = len(new_tokens)
        gb._ensure_correct_sample(ts, tokenizer)
        turn_samples.append(ts)

    if turn_samples:
        last = turn_samples[-1]
        last.metadata["is_last_turn"] = True
        last.metadata["messages"] = result["messages"]
        # chronological, one per turn (the store is de-duplicated; the judge wants the sequence)
        last.metadata["full_image_list"] = [
            screenshots[t["image_idx"][-1]] for t in result["turns"] if t.get("image_idx")
        ]
        last.metadata["final_answer"] = result.get("final_answer", "")
        if result.get("bb_session_id"):
            last.metadata["bb_session_id"] = result["bb_session_id"]
    return turn_samples


async def _generate_impl(args: Any, sample: Sample, sampling_params: dict, evaluation: bool) -> list[Sample]:
    task_id = (sample.metadata or {}).get("task_id", "unknown")
    task_data = gb._build_fallback_task_data(task_id, sample.metadata)
    state = GenerateState(args)
    traj_start = time.monotonic()
    version = current_weight_version()
    eps_name = f"{_sanitize(task_id)}__s{sample.index}"
    save_prob = float(os.environ.get("SLIME_MW_SAVE_PROB", "1.0"))
    save = random.random() < save_prob
    sub = "eval" if evaluation else f"v{version:04d}"
    rollout_dir = os.path.join(_save_root(args), sub, eps_name) if save else None
    work_dir = os.path.join(rollout_dir, "_worker") if rollout_dir else os.path.join(
        "/tmp", "mw_rollouts", f"{eps_name}_{os.getpid()}_{time.time_ns()}"
    )
    os.makedirs(work_dir, exist_ok=True)

    payload = {
        "molmoweb_root": _molmoweb_root(),
        "chat_template": _chat_template(),
        "sglang_url": f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate",
        "sampling_params": dict(sampling_params),
        "weight_version": version,
        "rollout_dir": rollout_dir,
        "task": {
            "task_id": task_id,
            "eps_name": eps_name,
            "intent": task_data["intent"],
            "start_url": task_data["start_url"],
            "task_type": "eval" if evaluation else "rl",
        },
        "config": {
            "max_steps": int(getattr(args, "max_steps", 30)),
            "max_past_steps": gb._get_max_past_steps(args),
            "context_num_screenshots": int(getattr(args, "context_num_screenshots", 1) or 1),
            "system_message": os.environ.get("SLIME_MW_SYSTEM_MESSAGE", "molmoweb"),
            "max_consecutive_parse_failures": int(getattr(args, "max_consecutive_parse_failures", 3)),
            "inference_step_timeout_secs": getattr(args, "inference_step_timeout_secs", None),
            "rollout_max_context_len": getattr(args, "rollout_max_context_len", None),
            "rollout_task_timeout_secs": gb._get_rollout_task_timeout_secs(args),
            "inference_abort_retries": gb._INFERENCE_ABORT_RETRIES,
            "robust_navigation": os.environ.get("SLIME_MW_ROBUST_NAV", "1") == "1",
            "proxies": os.environ.get("BROWSERBASE_PROXIES", "false").lower() == "true",
            "viewport_width": 1280,
            "viewport_height": 720,
        },
    }
    vp = os.environ.get("SLIME_BROWSER_VIEWPORT", "")
    if "x" in vp:
        w, h = vp.lower().split("x", 1)
        payload["config"]["viewport_width"], payload["config"]["viewport_height"] = int(w), int(h)
    with open(os.path.join(work_dir, "payload.json"), "w") as f:
        json.dump(payload, f)

    # Browserbase account limits (concurrent sessions, creates per minute) are enforced
    # here, in the process that knows how many episodes are in flight; the worker just
    # creates one session.
    from openwebrl.env.browserbase_env import _get_throttle

    throttle = _get_throttle({})
    await throttle.acquire_slot()
    try:
        await throttle.wait_for_spawn()
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(_pool(args), _run_worker, work_dir, _hard_timeout_secs(args))
    finally:
        throttle.release_slot()

    if result.get("error"):
        logger.warning("Task {} ({}): worker error: {}", task_id, result.get("terminate_reason"), str(result["error"])[-1500:])
    if gb._should_log_task_start(args):
        logger.info(
            "Task {} done: status={} reason={} turns={} dir={}",
            task_id, result["status"], result["terminate_reason"], result.get("total_steps"), result.get("rollout_dir"),
        )

    turn_samples = await _build_turn_samples(args, sample, task_data, result, state)
    if not turn_samples:
        sample.status = _STATUS[result["status"]] if result["status"] != "completed" else Sample.Status.ABORTED
        sample.metadata = sample.metadata or {}
        sample.metadata.update({
            "is_last_turn": True, "turn_index": 0, "total_steps": 0,
            "terminate_reason": result["terminate_reason"], "mw_rollout_dir": result.get("rollout_dir"),
        })
        sample.metadata.setdefault("intent", task_data["intent"])
        turn_samples = [sample]

    if not rollout_dir:
        shutil.rmtree(work_dir, ignore_errors=True)
    else:
        for name in ("payload.json", "result.json", "result.json.tmp", "session_id"):
            try:
                os.remove(os.path.join(work_dir, name))
            except OSError:
                pass
        try:
            os.replace(os.path.join(work_dir, "worker.log"), os.path.join(rollout_dir, "worker.log"))
            os.rmdir(work_dir)
        except OSError:
            pass
        with open(os.path.join(rollout_dir, "episode.json"), "w") as f:
            json.dump({
                "task_id": task_id, "intent": task_data["intent"], "start_url": task_data["start_url"],
                "status": result["status"], "terminate_reason": result["terminate_reason"],
                "total_steps": result["total_steps"], "final_answer": result.get("final_answer", ""),
                "weight_version": version, "sample_index": sample.index, "group_index": sample.group_index,
                "bb_session_id": result.get("bb_session_id"), "traj_wall_secs": time.monotonic() - traj_start,
                "env_setup_secs": result.get("env_setup_secs"),
            }, f, indent=2)
    return gb._finalize_turn_samples(turn_samples, traj_start)


async def generate_turn_sample(
    args: Any, sample: Sample, sampling_params: dict, evaluation: bool = False
) -> list[Sample]:
    """molmoweb-harness rollout of one trajectory -> per-turn Samples."""
    task_id = (sample.metadata or {}).get("task_id", "unknown")
    async with gb._browser_rollout_submission_slot(args, task_id):
        traj_start_ts = time.time()
        try:
            turn_samples = await _generate_impl(args, sample, sampling_params, evaluation)
        except Exception as exc:  # noqa: BLE001
            logger.opt(exception=True).warning("Task {}: molmoweb rollout failed: {}: {}", task_id, type(exc).__name__, exc)
            sample.status = Sample.Status.ABORTED
            sample.metadata = sample.metadata or {}
            sample.metadata.update({
                "is_last_turn": True, "turn_index": 0, "total_steps": 0,
                "terminate_reason": f"generation_error: {exc}",
            })
            turn_samples = [gb._mark_remove_sample_if_needed(sample)]
        turn_samples[-1].metadata["traj_start_ts"] = traj_start_ts
        turn_samples[-1].metadata["traj_end_ts"] = time.time()
        return turn_samples
