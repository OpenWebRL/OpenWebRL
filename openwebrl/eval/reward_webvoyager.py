"""
WebVoyager-protocol reward (FARA-style).

Pure LLM-as-a-judge on (task, final_answer, screenshots) -> SUCCESS / NOT SUCCESS,
using the original WebVoyager system/user prompts verbatim. No format check.

Reference:
    microsoft/fara webeval/src/webeval/benchmarks/webvoyager/webvoyager.py

Usage of this script:

    python examples/browser/run_evaluate.py \
    --task-file examples/browser/env/tasks/webvoyager_val.jsonl \
    --task-start 0 --task-end 2 --n-parallel 2 \
    --eval-protocol webvoyager
"""

import asyncio
import logging
from typing import Any

from openwebrl.base.utils import ToolParser
from openwebrl.eval._shared import _TOOLS_INFO, _extract_final_answer
from openwebrl.reward_browser import (
    _get_openai_client,
    _reward_semaphore,
)
from slime.utils.types import Sample

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """As an evaluator, you will be presented with three primary components to assist you in your role:

1. Web Task Instruction: This is a clear and specific directive provided in natural language, detailing the online activity to be carried out. These requirements may include conducting searches, verifying information, comparing prices, checking availability, or any other action relevant to the specified web service (such as Amazon, Apple, ArXiv, BBC News, Booking etc).

2. Result Screenshots: This is a visual representation of the screen showing the result or intermediate state of performing a web task. It serves as visual proof of the actions taken in response to the instruction.

3. Result Response: This is a textual response obtained after the execution of the web task. It serves as textual result in response to the instruction.

-- You DO NOT NEED to interact with web pages or perform actions such as booking flights or conducting searches on websites.
-- You SHOULD NOT make assumptions based on information not presented in the screenshot when comparing it to the instructions.
-- Your primary responsibility is to conduct a thorough assessment of the web task instruction against the outcome depicted in the screenshot and in the response, evaluating whether the actions taken align with the given instructions.
-- NOTE that the instruction may involve more than one task, for example, locating the garage and summarizing the review. Failing to complete either task, such as not providing a summary, should be considered unsuccessful.
-- NOTE that the screenshot is authentic, but the response provided by LLM is generated at the end of web browsing, and there may be discrepancies between the text and the screenshots.
-- Note the difference: 1) Result response may contradict the screenshot, then the content of the screenshot prevails, 2) The content in the Result response is not mentioned on the screenshot, choose to believe the content.

You should elaborate on how you arrived at your final evaluation and then provide a definitive verdict on whether the task has been successfully accomplished, either as 'SUCCESS' or 'NOT SUCCESS'."""

USER_PROMPT = "TASK: {task}\nResult Response: {answer}\n{num} screenshots at the end: "


async def _judge(args: Any, sample: Sample, tool_parser: ToolParser) -> tuple[float | None, str, bool]:
    """Single LLM-judge call with retry/timeout.

    Returns (score, text, saw_timeout). ``score`` is 1.0 / 0.0 for
    SUCCESS / NOT SUCCESS, or ``None`` if the judge call/parse failed
    (uninformative about the agent).
    """
    task = sample.metadata["intent"]
    answer = _extract_final_answer(tool_parser, sample.response or "")
    if not answer:
        logger.warning("No answer extracted from response; judge score = 0.0")
        return 0.0, "No answer extracted.", False

    n_imgs = getattr(args, "judge_max_attached_imgs", 30)
    screenshots = sample.metadata.get("full_image_list", [])[-n_imgs:]

    user_text = USER_PROMPT.format(task=task, answer=answer, num=len(screenshots))
    user_content: list[dict[str, Any]] = [{"type": "text", "text": user_text}]
    for b64 in screenshots:
        url = b64 if b64.startswith("data:") else f"data:image/png;base64,{b64}"
        user_content.append({"type": "image_url", "image_url": {"url": url}})

    if getattr(args, "log_judge_output", False):
        logger.info("Judge user prompt: %s", user_text)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_content},
        {"role": "user",   "content": "Your verdict:\n"},
    ]

    api_model = getattr(args, "judge_api_model")
    timeout_secs = getattr(args, "judge_timeout_secs", 120.0)
    client = _get_openai_client(method=getattr(args, "judge_api_mode", "token"))

    max_retries = 4
    saw_timeout = False
    for attempt in range(max_retries):
        try:
            coro = client.chat.completions.create(model=api_model, messages=messages, seed=42)
            resp = await (asyncio.wait_for(coro, timeout=timeout_secs) if timeout_secs else coro)
            text = resp.choices[0].message.content or ""
            logger.info("Judge response: %s", text)

            # Order matters: "NOT SUCCESS" contains "SUCCESS" as a substring.
            if "NOT SUCCESS" in text:
                return 0.0, text, False
            if "SUCCESS" in text:
                return 1.0, text, False
            logger.warning("Judge response missing SUCCESS/NOT SUCCESS: %s", text)
            return None, text, False

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

    return None, "Unexpected error occurred.", saw_timeout


async def _score_single(args: Any, sample: Sample) -> float | None:
    if not isinstance(sample, Sample):
        raise TypeError(f"Expected Sample, got {type(sample)}")
    if "intent" not in sample.metadata:
        raise ValueError("webvoyager reward requires sample.metadata['intent']")

    parser_type = "qwen3_coder" if "Qwen3.5" in args.hf_checkpoint else "qwen25"
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
            "WebVoyager reward: judge=%s (status=%s, task_id=%s)",
            score, sample.status, sample.metadata.get("task_id"),
        )

        sample.metadata["reward"] = {
            "judge": score,
            "combined": score,
            "judge_text": judge_text,
            "judge_timeout": judge_timeout,
            "protocol": "webvoyager",
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
