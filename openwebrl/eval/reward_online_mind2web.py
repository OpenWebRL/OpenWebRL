"""
Online-Mind2Web reward (FARA-style, AgentTrek single-call).

Single LLM call on (task, interleaved Thought_i/Action_i log, final screenshot)
returning Thoughts: ... / Status: "success"|"failure". Matches FARA's default
eval_method='AgentTrek_eval' for OM2W. Lenient relative to OSU's WebJudge
(see notes/eval_protocal_online_min2web.md), but cheap: 1 LLM call per task.

References:
    OSU-NLP-Group/Online-Mind2Web/src/methods/agenttrek_eval.py
    OSU-NLP-Group/Online-Mind2Web/src/utils.py            (extract_predication)
    microsoft/fara webeval/.../om2w/om2w.py               (defaults to this method)

Usage:
    python examples/browser/run_evaluate.py \
        --task-file examples/browser/env/tasks/online_mind2web_val.jsonl \
        --eval-protocol online_mind2web
"""

import asyncio
import json
import logging
import re
from typing import Any

from openwebrl.base.utils import ToolParser
from openwebrl.eval._shared import _TOOLS_INFO, resolve_parser_type
from openwebrl.reward_browser import (
    _get_openai_client,
    _reward_semaphore,
)
from slime.utils.types import Sample

logger = logging.getLogger(__name__)


# ---- Prompts (verbatim from OSU's agenttrek_eval.py) ----------------------

SYSTEM_PROMPT = """You are an expert in evaluating the performance of a web navigation agent. The agent is designed to help a human user navigate a website to complete a task. Given the user's task goal, the agent's trajectory, your goal is to decide whether the agent's execution is successful or not.

*Evaluation Criteria*
Whether the agent's trajectory is effective and corresponding to the goal

*Instructions* 1. Review the agent's actions and reasoning processes step by step.
2. if the agent is stuck in the very first login stage, which means it fails to log into target website at the beginning, that's a failure.
3. Determine if the agent has achieved the task goal based on the trajectory. A task can be considered successful if most trajectory is effective.
4. the agent sometimes can't stop after finishing a task and continue doing repeated actions. these actions may be some failed attempt after a series of correct actions. the task should be regarded as successful if the correct actions are effective and almost reach the goal.
5. if the agent is stuck in the loop at the early stage of the task, which means they don't even get close to the goal before they get stuck in the loop, that's a failure. for example, the agent begin to get stuck before third step.
6. when the task is to change the google account password, it can't be regarded as successful when agent finish at trying to click "manage your account".
7. if there are over 8 correct action in the trajectory, it can be regard as a successful agent.
8. final saving action is not a must. the task is successful if the agent does most things right and just forget to save the change at last.
9. if the original task has 2 subtasks, the agent only complete one of them, that's still a success. e.g. the task is to update name and birthday, but agent only update name, that's fine.
10. if the task is to post a review, the agent can be considered successful when it finish writing the review and reach the step to post it, don't have to click the post button.
11. Since we don't have a printer, some printing related task can be considered successful if the agent reach the step to click print button.
12. if the task is finished at the initial state and the agent do nothing because of it, it should also be regarded as successful.

*IMPORTANT*
1. in the trajectory, an action always follows a corresponding reasoning, which shows the observation and thought of the agent.
2. your response should be contain:
Thoughts: <your thoughts and reasoning process>
Status: "success" or "failure"
"""

USER_PROMPT = """The goal of the task: {task}

Trajectory:
{thoughts_and_actions}

The last snapshot of the web page is shown in the image."""


# ---- Helpers ---------------------------------------------------------------

# Thinking-model assistant turns start *inside* <think> (no opening tag) and
# end with </think>; non-thinking turns may carry an explicit <think>...</think>.
# Match either: capture everything up to the first </think>.
_THINK_PATTERN = re.compile(r"(?:<think>)?(.*?)</think>", re.DOTALL)


def _extract_thought(content: str) -> str:
    m = _THINK_PATTERN.search(content)
    return m.group(1).strip() if m else ""


def _format_action(name: str, params: Any) -> str:
    if isinstance(params, str):
        try:
            params = json.loads(params)
        except json.JSONDecodeError:
            return f"{name}({params})"
    if isinstance(params, dict):
        kv = ", ".join(f"{k}={v!r}" for k, v in params.items())
        return f"{name}({kv})"
    return f"{name}({params})"


def _extract_thoughts_and_actions(messages: list[dict], tool_parser: ToolParser) -> tuple[list[str], list[str]]:
    """One thought + one action string per assistant turn."""
    thoughts: list[str] = []
    actions: list[str] = []
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content", "") or ""
        thoughts.append(_extract_thought(content))
        parsed = tool_parser.parse(content)
        if not parsed.success or not parsed.calls:
            actions.append("(no parseable action)")
            continue
        actions.append("; ".join(_format_action(c.name, c.parameters) for c in parsed.calls))
    return thoughts, actions


def _format_trajectory(thoughts: list[str], actions: list[str]) -> str:
    """Mirror OSU's `Thought i: ...\\nAction i: ...\\n\\n` formatting."""
    parts: list[str] = []
    for i, (t, a) in enumerate(zip(thoughts, actions)):
        t = t.replace("\n\n", " ")
        a = a.replace("\n\n", " ")
        parts.append(f"Thought {i + 1}: {t}\nAction {i + 1}: {a}")
    return "\n\n".join(parts)


def _parse_verdict(text: str) -> float | None:
    """OSU extract_predication for AgentTrek_eval: 'success' (case-insensitive) after 'status:'.

    Returns ``None`` if the verdict format is missing (judge parse failure
    rather than a real failure verdict).
    """
    lower = text.lower()
    if "status:" not in lower:
        return None
    return 1.0 if "success" in lower.split("status:", 1)[1] else 0.0


# ---- Main scoring ----------------------------------------------------------

async def _judge(args: Any, sample: Sample, tool_parser: ToolParser) -> tuple[float | None, str, bool]:
    """One LLM call with retry.

    Returns (score, text, saw_timeout). ``score`` is 1.0 / 0.0 for
    success / failure, or ``None`` if the judge call/parse failed
    (uninformative about the agent).
    """
    task = sample.metadata["intent"]
    messages = sample.metadata.get("messages", [])
    full_image_list = sample.metadata.get("full_image_list", [])

    thoughts, actions = _extract_thoughts_and_actions(messages, tool_parser)
    if not actions:
        return 0.0, "No actions extracted from trajectory.", False
    if not full_image_list:
        return 0.0, "No screenshots in trajectory.", False

    last_b64 = full_image_list[-1]
    image_url = last_b64 if last_b64.startswith("data:") else f"data:image/png;base64,{last_b64}"

    user_text = USER_PROMPT.format(
        task=task,
        thoughts_and_actions=_format_trajectory(thoughts, actions),
    )
    if getattr(args, "log_judge_output", False):
        logger.info("Judge user prompt: %s", user_text)

    llm_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "text", "text": user_text},
            {"type": "image_url", "image_url": {"url": image_url, "detail": "high"}},
        ]},
    ]

    api_model = getattr(args, "judge_api_model")
    timeout_secs = getattr(args, "judge_timeout_secs", 120.0)
    client = _get_openai_client(method=getattr(args, "judge_api_mode", "token"))
    max_retries = 4
    saw_timeout = False
    for attempt in range(max_retries):
        try:
            coro = client.chat.completions.create(
                model=api_model, messages=llm_messages, seed=42,
            )
            resp = await (asyncio.wait_for(coro, timeout=timeout_secs) if timeout_secs else coro)
            text = resp.choices[0].message.content or ""
            logger.info("Judge response: %s", text)
            return _parse_verdict(text), text, False

        except asyncio.TimeoutError as e:
            saw_timeout = True
            logger.warning(
                "Judge API call timed out (attempt %d/%d, timeout=%ss): %s",
                attempt + 1, max_retries, timeout_secs, e,
            )
            if attempt < max_retries - 1:
                await asyncio.sleep(3 ** (attempt + 1))
            else:
                return None, f"Judge timeout exhausted after {max_retries} attempts.", True
        except Exception as e:
            logger.warning(
                "Judge API call failed (attempt %d/%d): %s", attempt + 1, max_retries, e
            )
            if attempt < max_retries - 1:
                await asyncio.sleep(3 ** (attempt + 1))
            else:
                return None, "All judge API retry attempts exhausted.", saw_timeout

    return None, "Unexpected error.", saw_timeout


async def _score_single(args: Any, sample: Sample) -> float | None:
    if not isinstance(sample, Sample):
        raise TypeError(f"Expected Sample, got {type(sample)}")
    if "intent" not in sample.metadata:
        raise ValueError("om2w reward requires sample.metadata['intent']")

    parser_type = resolve_parser_type(args.hf_checkpoint)
    tool_parser = ToolParser(_TOOLS_INFO, parser_type=parser_type)

    async with _reward_semaphore:
        judge_timeout = False
        if sample.status != Sample.Status.COMPLETED:
            score = 0.0
            judge_text = f"Judge not run for status={sample.status}"
        else:
            score, judge_text, judge_timeout = await _judge(args, sample, tool_parser)
            if score is None:
                sample.remove_sample = True
                if judge_timeout:
                    sample.metadata["judge_timeout"] = True

        logger.info(
            "OM2W reward: judge=%s (status=%s, task_id=%s)",
            score, sample.status, sample.metadata.get("task_id"),
        )

        sample.metadata["reward"] = {
            "judge": score,
            "combined": score,
            "judge_text": judge_text,
            "judge_timeout": judge_timeout,
            "protocol": "online_mind2web",
        }
        sample.reward = score
        return score


async def reward_func(args: Any, samples: Sample | list[Sample], **kwargs) -> float | None | list[float | None]:
    """Same signature as reward_browser.reward_func."""
    if isinstance(samples, Sample):
        return await _score_single(args, samples)

    is_turn_level = any("turn_index" in s.metadata for s in samples)
    if not is_turn_level:
        raise ValueError("Expected a list of turn-level samples with 'turn_index' in metadata.")

    last_turn_sample = next((s for s in samples if s.metadata.get("is_last_turn", False)), None)
    if last_turn_sample is None:
        raise ValueError("No turn sample with is_last_turn=True found.")

    trajectory_reward = await _score_single(args, last_turn_sample)
    reward_meta = last_turn_sample.metadata.get("reward", {})
    for s in samples:
        s.metadata["reward"] = reward_meta
        s.reward = trajectory_reward
        if last_turn_sample.remove_sample:
            s.remove_sample = True
    return [trajectory_reward] * len(samples)
