"""
DeepShop reward (Molmo-Web-style).

Single LLM call on (task, final_answer, screenshots) returning a structured
verdict {thought, verdict in {"SUCCESS", "NOT SUCCESS"}}. Matches AllenAI
Molmo Web's deepshop_judge.py byte-for-byte on prompt and message shape.

Reference:
    allenai/molmoweb benchmarks/judges/deepshop_judge.py

Usage:
    python examples/browser/run_evaluate.py \
        --task-file examples/browser/env/tasks/deepshop_val.jsonl \
        --eval-protocol deepshop
"""

import asyncio
import logging
from typing import Any, Literal

from pydantic import BaseModel, Field

from openwebrl.base.utils import ToolParser
from openwebrl.eval._shared import _TOOLS_INFO, _extract_final_answer, resolve_parser_type
from openwebrl.reward_browser import (
    _get_openai_client,
    _reward_semaphore,
)
from slime.utils.types import Sample

logger = logging.getLogger(__name__)


# ---- Prompt (verbatim from Molmo's deepshop_judge.py) ---------------------

SYSTEM_PROMPT = """As an evaluator, you will be presented with three primary components to assist you in your role:
1. Web Task Instruction: A clear and precise natural language directive that specifies an
online shopping activity to be executed. The instruction may involve locating products that
meet certain attribute requirements (e.g., color, size, brand), applying specific search filters
(e.g., price range, customer ratings, availability), or fulfilling user-defined sorting preferences
(e.g., lowest price, newest arrivals, best sellers). Tasks may also include verifying product
details, comparing offers, or checking for shipping and return policies, depending on the
scenario.
2. Result Screenshots: This is a visual representation of the screen showing the result or
intermediate state of performing a web task. It serves as visual proof of the actions taken in
response to the instruction.
3. Result Response: This is a textual response obtained after the execution of the web task.
It serves as textual result in response to the instruction.
-- You DO NOT NEED to interact with web pages or perform actions such as conducting
searches on websites.
-- You SHOULD NOT make assumptions based on information not presented in the
screenshot when comparing it to the instructions.
-- Your primary responsibility is to conduct a thorough assessment of the web task instruction
against the outcome depicted in the screenshot and in the response, evaluating whether the
actions taken align with the given instructions.
-- NOTE that the instruction may involve more than one task, for example, locating the
garage and summarizing the review. Failing to complete either task, such as not providing a
summary, should be considered unsuccessful.
-- NOTE that the screenshot is authentic, but the response provided by LLM is generated at
the end of web browsing, and there may be discrepancies between the text and the
screenshots.
-- Note the difference: 1) Result response may contradict the screenshot, then the content of
the screenshot prevails, 2) The content in the Result response is not mentioned on the
screenshot, choose to believe the content.
You should elaborate on how you arrived at your final evaluation and then provide a
definitive verdict on whether the task has been successfully accomplished, either as
'SUCCESS' or 'NOT SUCCESS'."""


# ---- Structured-output schema (matches Molmo's DeepShopVerdict) -----------

class DeepShopVerdict(BaseModel):
    thought: str = Field(description="Reasoning behind the verdict")
    verdict: Literal["SUCCESS", "NOT SUCCESS"] = Field(
        description="Final verdict for whether the web assistant completed the task or not."
    )


# ---- Single judge call -----------------------------------------------------

async def _judge(args: Any, sample: Sample, tool_parser: ToolParser) -> tuple[float | None, str, bool]:
    """One LLM call with retry.

    Returns (score, judge_text, saw_timeout). ``score`` is 1.0 / 0.0 for
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

    user_text = f"TASK:\n{task}\nResult Response:\n{answer}\n{len(screenshots)} screenshots"
    user_content: list[dict[str, Any]] = [{"type": "text", "text": user_text}]
    for i, b64 in enumerate(screenshots):
        url = b64 if b64.startswith("data:") else f"data:image/png;base64,{b64}"
        user_content.append({"type": "text", "text": f"Step {i + 1}"})
        user_content.append({"type": "image_url", "image_url": {"url": url}})

    if getattr(args, "log_judge_output", False):
        logger.info("Judge user prompt: %s", user_text)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    api_model = getattr(args, "judge_api_model")
    timeout_secs = getattr(args, "judge_timeout_secs", 120.0)
    client = _get_openai_client(method=getattr(args, "judge_api_mode", "token"))

    max_retries = 4
    saw_timeout = False
    for attempt in range(max_retries):
        try:
            coro = client.beta.chat.completions.parse(
                model=api_model,
                messages=messages,
                response_format=DeepShopVerdict,
                temperature=0,
                seed=42,
            )
            resp = await (asyncio.wait_for(coro, timeout=timeout_secs) if timeout_secs else coro)
            parsed: DeepShopVerdict | None = resp.choices[0].message.parsed
            if parsed is None:
                refusal = resp.choices[0].message.refusal
                logger.warning("Judge returned no parsed object (refusal=%s)", refusal)
                return None, f"Judge refused or unparsed: {refusal}", False
            score = 1.0 if parsed.verdict == "SUCCESS" else 0.0
            judge_text = f"verdict={parsed.verdict}\nthought={parsed.thought}"
            logger.info("Judge response: %s", judge_text)
            return score, judge_text, False

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
        raise ValueError("deepshop reward requires sample.metadata['intent']")

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
            "DeepShop reward: judge=%s (status=%s, task_id=%s)",
            score, sample.status, sample.metadata.get("task_id"),
        )

        sample.metadata["reward"] = {
            "judge": score,
            "combined": score,
            "judge_text": judge_text,
            "judge_timeout": judge_timeout,
            "protocol": "deepshop",
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
