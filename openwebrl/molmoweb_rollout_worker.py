"""One browser episode with molmoweb's agent, env and trajectory logging.

Runs as a plain subprocess (``python openwebrl/molmoweb_rollout_worker.py <work_dir>``) so that
molmoweb's synchronous Playwright env never touches the rollout process's asyncio
loop, and a hung episode can be killed without leaking into the trainer. The parent
(``openwebrl/generate_molmoweb.py``) writes ``payload.json`` into a work dir and reads
``result.json`` back; ``session_id`` is written as soon as the Browserbase session
exists so the parent can release it if it has to kill this process.

Everything about *how the agent sees the task* is molmoweb's code, imported from
``MOLMOWEB_ROOT``:
  - ``agent.fara15_vllm.Fara15VLLMAgent``  prompt construction, history window, image
    budget, output parsing, action mapping (the ``fara15_mw`` eval configuration)
  - ``utils.envs.BrowserbaseEnv``           the browser (sync Playwright over CDP)
  - ``benchmarks.traj_logging.log_episode``  trajectory.json / trajectory.html / images

The one substitution is the agent's ``predictor``: instead of an OpenAI-compatible vLLM
call it renders the chat template itself, posts to slime's SGLang ``/generate`` and
records the exact prompt text, images, sampled token ids and log-probs of every turn
-- that is what the trainer needs and what a chat-completions API cannot return.

This module must stay light: no slime, no torch, no transformers, and no ``openwebrl``
imports (the package __init__ imports the slime rollout stack); it is run by file path.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from typing import Any

import requests

logger = logging.getLogger("molmoweb_rollout_worker")

IMAGE_PLACEHOLDER = "<|vision_start|><|image_pad|><|vision_end|>"


# ---------------------------------------------------------------------------
# Chat template rendering (byte-identical to tokenizer.apply_chat_template)
# ---------------------------------------------------------------------------

def _jinja_env():
    """The environment transformers builds for chat templates."""
    import jinja2
    from jinja2.ext import loopcontrols
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    def raise_exception(message):
        raise jinja2.exceptions.TemplateError(message)

    def tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
        return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)

    def strftime_now(fmt):
        return datetime.now().strftime(fmt)

    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=[loopcontrols])
    env.filters["tojson"] = tojson
    env.globals["raise_exception"] = raise_exception
    env.globals["strftime_now"] = strftime_now
    return env


class PromptRenderer:
    def __init__(self, chat_template: str):
        self._template = _jinja_env().from_string(chat_template)

    def render(self, messages: list[dict]) -> str:
        return self._template.render(messages=messages, add_generation_prompt=True, tools=None)


def _strip_private_keys(messages: list[dict]) -> list[dict]:
    """molmoweb tags user turns with ``preserve_text``; the template must not see it."""
    return [{"role": m["role"], "content": m["content"]} for m in messages]


def _image_urls_in_order(messages: list[dict]) -> list[str]:
    urls = []
    for m in messages:
        content = m.get("content")
        if isinstance(content, list):
            for item in content:
                if item.get("type") == "image_url":
                    url = item["image_url"]
                    urls.append(url["url"] if isinstance(url, dict) else url)
    return urls


# ---------------------------------------------------------------------------
# The predictor: SGLang /generate with recorded tokens and log-probs
# ---------------------------------------------------------------------------

class ContextLengthExceeded(RuntimeError):
    pass


class SGLangRecordingPredictor:
    """Drop-in for ``Fara15VLLMPredictor``: ``predict(messages) -> str``.

    Every successful call appends a turn record to ``self.turns``. A failed call
    (timeout, HTTP error) raises and records nothing, so a retried call is never
    double-counted.
    """

    def __init__(
        self,
        url: str,
        sampling_params: dict,
        renderer: PromptRenderer,
        timeout_secs: float | None,
        max_context_len: int | None,
        abort_retries: int = 3,
        abort_retry_sleep_secs: float = 2.0,
    ):
        self.url = url
        self.sampling_params = dict(sampling_params)
        self.renderer = renderer
        self.timeout_secs = timeout_secs
        self.max_context_len = max_context_len
        self.abort_retries = abort_retries
        self.abort_retry_sleep_secs = abort_retry_sleep_secs
        self.turns: list[dict[str, Any]] = []
        self.session = requests.Session()

    def _guard_context(self, prompt_text: str, n_images: int) -> None:
        if not self.max_context_len:
            return
        # Conservative estimate: ~3 chars per text token, ~1,100 tokens per 1280x736
        # screenshot (40 x 23 patches of 32 px). Only meant to catch the runaway
        # trajectory before SGLang rejects it; normal prompts sit far below 32k.
        est = len(prompt_text) / 3.0 + 1100 * n_images
        reserved = int(self.sampling_params.get("max_new_tokens", 0) or 0)
        if est + reserved > self.max_context_len:
            raise ContextLengthExceeded(
                f"estimated prompt {est:.0f} + {reserved} reserved tokens exceeds {self.max_context_len}"
            )

    def predict(self, messages: list[dict]) -> str:
        clean = _strip_private_keys(messages)
        prompt_text = self.renderer.render(clean)
        image_data = _image_urls_in_order(clean)
        self._guard_context(prompt_text, len(image_data))

        payload: dict[str, Any] = {
            "text": prompt_text,
            "sampling_params": self.sampling_params,
            "return_logprob": True,
        }
        if image_data:
            payload["image_data"] = image_data

        t0 = time.monotonic()
        deadline = t0 + self.timeout_secs if self.timeout_secs else None
        output = None
        for attempt in range(self.abort_retries + 1):
            remaining = None
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"SGLang /generate timed out after {self.timeout_secs}s")
            try:
                resp = self.session.post(self.url, json=payload, timeout=remaining)
            except requests.Timeout as exc:
                raise TimeoutError(f"SGLang /generate timed out after {self.timeout_secs}s") from exc
            if resp.status_code != 200:
                raise RuntimeError(f"SGLang /generate HTTP {resp.status_code}: {resp.text[:500]}")
            output = resp.json()
            meta = output.get("meta_info", {})
            # A weight sync aborts in-flight requests with finish_reason "abort" and no
            # tokens; retry within the same budget (same as generate_browser.py).
            if meta.get("finish_reason", {}).get("type") == "abort" and attempt < self.abort_retries:
                time.sleep(self.abort_retry_sleep_secs)
                continue
            break

        meta = output.get("meta_info", {})
        if "output_token_logprobs" not in meta:
            raise ValueError("output_token_logprobs not found in SGLang meta_info")
        tokens = [item[1] for item in meta["output_token_logprobs"]]
        logprobs = [item[0] for item in meta["output_token_logprobs"]]
        finish_type = meta.get("finish_reason", {}).get("type", "stop")
        response_text = output["text"]

        self.turns.append({
            "prompt_text": prompt_text,
            "image_data": image_data,
            "response_text": response_text,
            "tokens": tokens,
            "logprobs": logprobs,
            "finish_type": finish_type,
            "infer_secs": time.monotonic() - t0,
            "prompt_tokens": meta.get("prompt_tokens"),
            "cached_tokens": meta.get("cached_tokens"),
        })
        return response_text


# ---------------------------------------------------------------------------
# Episode
# ---------------------------------------------------------------------------

def _messages_for_judge(chat_history: list[dict]) -> list[dict]:
    """molmoweb chat history -> the mm_messages shape openwebrl's judges read
    (assistant content = raw response text; user turns = text + image_url items)."""
    out = []
    for m in chat_history:
        content = m["content"]
        if isinstance(content, list):
            items = []
            for item in content:
                if item.get("type") == "image_url":
                    url = item["image_url"]
                    items.append({"type": "image_url", "image_url": url["url"] if isinstance(url, dict) else url})
                elif item.get("type") == "text":
                    items.append({"type": "text", "text": f"{IMAGE_PLACEHOLDER}{item['text']}"})
            out.append({"role": m["role"], "content": items})
        else:
            out.append({"role": m["role"], "content": content})
    return out


def _release_session(session_id: str | None) -> None:
    if not session_id:
        return
    try:
        from browserbase import Browserbase

        Browserbase(api_key=os.environ["BROWSERBASE_API_KEY"]).sessions.update(
            session_id, project_id=os.environ["BROWSERBASE_PROJECT_ID"], status="REQUEST_RELEASE"
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("could not release Browserbase session %s: %s", session_id, exc)


def run_episode(payload: dict[str, Any], work_dir: str) -> dict[str, Any]:
    sys.path.insert(0, payload["molmoweb_root"])
    from agent.actions import SendMsgToUser
    from agent.fara15_vllm import Fara15VLLMAgent
    from utils.envs import BrowserbaseEnv
    from utils.eval_utils.episode import Interaction

    task = payload["task"]
    cfg = payload["config"]
    result: dict[str, Any] = {
        "status": "aborted",
        "terminate_reason": "generation_error: worker did not finish",
        "total_steps": 0,
        "turns": [],
        "messages": [],
        "screenshots": [],
        "final_answer": "",
        "rollout_dir": None,
        "bb_session_id": None,
        "env_setup_secs": None,
        "error": None,
    }

    renderer = PromptRenderer(payload["chat_template"])
    predictor = SGLangRecordingPredictor(
        url=payload["sglang_url"],
        sampling_params=payload["sampling_params"],
        renderer=renderer,
        timeout_secs=cfg.get("inference_step_timeout_secs"),
        max_context_len=cfg.get("rollout_max_context_len"),
        abort_retries=int(cfg.get("inference_abort_retries", 3)),
    )
    agent = Fara15VLLMAgent(
        endpoint="http://unused.invalid/v1",
        max_n_images=int(cfg.get("context_num_screenshots", 1)),
        preserve_aged_text=True,
        max_past_steps=int(cfg.get("max_past_steps", 10)),
        image_text_newline=True,
        system_message=cfg.get("system_message") or "molmoweb",
        auto_user_reply=True,
    )
    agent.predictor = predictor  # the only molmoweb seam we touch

    env = BrowserbaseEnv(
        start_url=task["start_url"] or "about:blank",
        goal=task["intent"],
        viewport_width=int(cfg.get("viewport_width", 1280)),
        viewport_height=int(cfg.get("viewport_height", 720)),
        robust_navigation=bool(cfg.get("robust_navigation", True)),
        proxies=bool(cfg.get("proxies", False)),
    )

    interactions: list[Interaction] = []
    screenshots: list[str] = []  # data URLs, one per agent turn (the screenshot it saw)
    status = "failed"
    terminate_reason = "max_steps_exhausted"
    final_answer = ""
    consecutive_parse_failures = 0
    max_parse_failures = int(cfg.get("max_consecutive_parse_failures", 3))
    max_steps = int(cfg.get("max_steps", 30))
    task_timeout = cfg.get("rollout_task_timeout_secs")
    step_env_secs: list[float] = []

    try:
        t0 = time.monotonic()
        obs, info = env.reset()
        result["env_setup_secs"] = time.monotonic() - t0
        session_id = getattr(getattr(env, "bb_session", None), "id", None)
        result["bb_session_id"] = session_id
        with open(os.path.join(work_dir, "session_id"), "w") as f:
            f.write(session_id or "")
        agent.reset()
        deadline = time.monotonic() + float(task_timeout) if task_timeout else None

        state = {"obs": obs, "reward": 0, "terminated": False, "truncated": False, "info": info}
        for step in range(max_steps):
            if deadline is not None and time.monotonic() > deadline:
                status, terminate_reason = "aborted", f"generation_error: rollout_task_timeout after {task_timeout}s"
                break
            action_timestamp = datetime.now(timezone.utc).timestamp()

            n_turns_before = len(predictor.turns)
            try:
                raw_output, action = agent.predict_action(state["obs"])
            except ContextLengthExceeded:
                status, terminate_reason = "truncated", "generation_length_limit"
                break
            if len(predictor.turns) == n_turns_before:
                raise RuntimeError("agent.predict_action returned without a model call")
            turn = predictor.turns[-1]
            # the screenshot this turn was prompted with = the last image in the request
            screenshots.append(turn["image_data"][-1] if turn["image_data"] else None)
            state["agent_inputs"] = agent.get_last_model_inputs()

            if turn["finish_type"] == "length":
                interactions.append(Interaction(state=state, action=action, raw_output=raw_output,
                                                action_timestamp=action_timestamp))
                status, terminate_reason = "truncated", "generation_length_limit"
                break
            if turn["finish_type"] == "abort":
                interactions.append(Interaction(state=state, action=action, raw_output=raw_output,
                                                action_timestamp=action_timestamp))
                status, terminate_reason = "aborted", "generation_abort"
                break

            action_args = action.get("action") or {}
            action_name = action_args.get("action", "") if isinstance(action_args, dict) else ""
            parse_failed = not action_args
            if parse_failed:
                consecutive_parse_failures += 1
            else:
                consecutive_parse_failures = 0

            action_obj = action["action_output"].action
            t_env = time.monotonic()
            try:
                next_obs = env.step(action=action_obj)
            except Exception as exc:  # noqa: BLE001
                interactions.append(Interaction(state=state, action=action, raw_output=raw_output,
                                                action_timestamp=action_timestamp,
                                                error={"ENV_STEP_ERROR": str(exc)}))
                step_env_secs.append(time.monotonic() - t_env)
                status, terminate_reason = "aborted", "env_step_error"
                break
            step_env_secs.append(time.monotonic() - t_env)
            next_state = {"obs": next_obs, "reward": None, "terminated": None, "truncated": None, "info": None}
            interactions.append(Interaction(state=state, action=action, raw_output=raw_output,
                                            next_state=next_state, action_timestamp=action_timestamp))
            state = next_state

            if action_name == "terminate":
                # molmoweb reports terminate as [ANSWER] then a synthetic [EXIT] turn;
                # the trajectory is complete here, the [EXIT] step is not a model call.
                final_answer = str(action_args.get("answer", ""))
                status, terminate_reason = "completed", "task_completed"
                break
            if isinstance(action_obj, SendMsgToUser) and (action_obj.msg or "").startswith("[EXIT]"):
                status, terminate_reason = "completed", "task_completed"
                break
            if parse_failed and consecutive_parse_failures >= max_parse_failures:
                status, terminate_reason = "failed", "format_error_failed"
                break
        else:
            status, terminate_reason = "failed", "max_steps_exhausted"
    except Exception as exc:  # noqa: BLE001
        status, terminate_reason = "aborted", f"generation_error: {type(exc).__name__}: {exc}"
        result["error"] = traceback.format_exc()
    finally:
        try:
            env.close()
        except Exception as exc:  # noqa: BLE001
            logger.warning("env.close failed: %s", exc)
            _release_session(result.get("bb_session_id"))

    turns = predictor.turns
    result.update({
        "status": status,
        "terminate_reason": terminate_reason,
        "total_steps": len(turns),
        "turns": [{k: v for k, v in t.items() if k != "image_data"} for t in turns],
        "messages": _messages_for_judge(agent.chat_history),
        "final_answer": final_answer,
        "step_env_secs": step_env_secs,
    })
    # de-duplicate images: every screenshot once, turns reference them by index
    index: dict[str, int] = {}
    for url in screenshots:
        if url is not None and url not in index:
            index[url] = len(index)
    result["screenshots"] = list(index.keys())
    for t, rec in zip(turns, result["turns"], strict=True):
        rec["image_idx"] = []
        for url in t["image_data"]:
            if url not in index:
                index[url] = len(index)
                result["screenshots"].append(url)
            rec["image_idx"].append(index[url])

    # molmoweb's trajectory log (trajectory.json / .html, images/, user_message/)
    rollout_dir = payload.get("rollout_dir")
    if rollout_dir and interactions:
        try:
            from benchmarks.traj_logging import log_episode

            log_episode(
                interactions=interactions,
                metadata={"goal": task["intent"], "eps_name": task["eps_name"],
                          "task_id": task["task_id"], "status": status,
                          "terminate_reason": terminate_reason, "weight_version": payload.get("weight_version")},
                system_message=agent.system_message,
                outdir=rollout_dir,
                instruction=task["intent"],
                task_type=task.get("task_type", "rl"),
                bb_session_id=result.get("bb_session_id"),
            )
            result["rollout_dir"] = rollout_dir
        except Exception as exc:  # noqa: BLE001
            logger.warning("log_episode failed: %s", exc)
            result["log_error"] = traceback.format_exc()
    return result


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    work_dir = sys.argv[1]
    with open(os.path.join(work_dir, "payload.json")) as f:
        payload = json.load(f)
    try:
        result = run_episode(payload, work_dir)
    except Exception:  # noqa: BLE001
        result = {"status": "aborted", "terminate_reason": "generation_error: worker crashed",
                  "error": traceback.format_exc(), "turns": [], "messages": [], "screenshots": [],
                  "total_steps": 0, "final_answer": "", "rollout_dir": None, "bb_session_id": None}
    tmp = os.path.join(work_dir, "result.json.tmp")
    with open(tmp, "w") as f:
        json.dump(result, f)
    os.replace(tmp, os.path.join(work_dir, "result.json"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
