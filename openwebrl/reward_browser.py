"""
Reward computation for browser agent tasks.

Two reward components:
1. Format reward  – deterministic check that the agent response adheres to
   the policy defined in system_prompt.md (<thinking>, <action>, <tool_call>).
2. LLM-as-a-Judge – async OpenAI-compatible call that evaluates whether the
   task was successfully completed, adapted from the original GPT-4V evaluator.

Usage:
    from reward_browser import reward_func

    score = await reward_func(args, sample)
"""

import asyncio
import json
import logging
import os
import random
import re
import time
from datetime import datetime
from typing import Any

from openai import AsyncAzureOpenAI, AsyncOpenAI

from openwebrl.base.utils import ToolParser
from openwebrl.feedback_utils import (
    DEFAULT_BROWSER_HISTORY_ENV_FEEDBACK_MAX_CHARS,
    normalize_feedback_text,
    truncate_feedback_text,
)
from openwebrl.response_format import get_browser_response_format_mode
from slime.utils.types import Sample

logger = logging.getLogger(__name__)

# ==============================================================================
# Constants
# ==============================================================================

# Load valid tool names from tool_list.json
_TOOL_LIST_PATH = os.path.join(os.path.dirname(__file__), "env", "prompts", "tool_list.json")
with open(_TOOL_LIST_PATH, "r") as _f:
    VALID_TOOL_NAMES = {tool["function"]["name"] for tool in json.load(_f)}

# Concurrency control (mirrors swe_reward.py)
_REWARD_CONCURRENCY = int(os.environ.get("BROWSER_REWARD_CONCURRENCY", "8"))
_reward_semaphore = asyncio.Semaphore(_REWARD_CONCURRENCY)


def _get_env_override(name: str) -> str | None:
    value = os.environ.get(name)
    if value is None:
        return None
    value = value.strip()
    return value if value else None


def _should_log_judge_output(args: Any) -> bool:
    value = _get_env_override("SLIME_BROWSER_LOG_JUDGE_OUTPUT")
    if value is not None:
        return value.lower() in {"1", "true", "yes", "on"}
    return bool(getattr(args, "log_judge_output", False))


def _get_judge_output_log_prob(args: Any) -> float:
    value = _get_env_override("SLIME_BROWSER_LOG_JUDGE_OUTPUT_PROB")
    if value is None:
        value = getattr(args, "log_judge_output_prob", 0.1)
    try:
        prob = float(value)
    except (TypeError, ValueError):
        return 0.1
    return max(0.0, min(1.0, prob))


def _should_sample_judge_output(args: Any) -> bool:
    return _should_log_judge_output(args) and random.random() < _get_judge_output_log_prob(args)


def _get_debug_trace_prob(args: Any) -> float:
    value = _get_env_override("SLIME_BROWSER_DEBUG_TRACE_PROB")
    if value is None:
        value = getattr(args, "debug_trace_prob", 0.0)
    try:
        prob = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, prob))


def _get_debug_trace_info(sample: Sample) -> tuple[bool, str | None]:
    metadata = sample.metadata or {}
    selected = metadata.get("debug_trace_selected")
    trace_id = metadata.get("debug_trace_id")
    if not isinstance(selected, bool):
        return False, None
    return selected, trace_id if isinstance(trace_id, str) and trace_id else None


def _get_debug_trace_dir(args: Any, trace_kind: str) -> str:
    save_dir = getattr(args, "save", "") or os.environ.get("SLIME_SAVE_DIR", "")
    if not save_dir:
        return ""
    return os.path.join(save_dir, "debug_traces", trace_kind)


def _to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_jsonable(v) for v in value]
    if hasattr(value, "value"):
        try:
            return value.value
        except Exception:
            pass
    if hasattr(value, "to_dict") and callable(value.to_dict):
        try:
            return _to_jsonable(value.to_dict())
        except Exception:
            pass
    if hasattr(value, "tolist") and callable(value.tolist):
        try:
            return _to_jsonable(value.tolist())
        except Exception:
            pass
    return str(value)


def _save_judge_trace(
    args: Any,
    sample: Sample,
    *,
    messages: list[dict[str, Any]],
    judge_text: str,
    judge_score: float,
    judge_timeout: bool,
    prompt_variant: str,
    answer: str,
    screenshots: list[str],
    action_history: str | None = None,
) -> None:
    debug_dir = _get_debug_trace_dir(args, "judge")
    selected, trace_id = _get_debug_trace_info(sample)
    if not debug_dir or not selected or not trace_id:
        return

    os.makedirs(debug_dir, exist_ok=True)
    filename = f"{trace_id}.json"

    payload = {
        "trace_id": trace_id,
        "task_id": (sample.metadata or {}).get("task_id"),
        "intent": (sample.metadata or {}).get("intent"),
        "status": _to_jsonable(sample.status),
        "terminate_reason": (sample.metadata or {}).get("terminate_reason"),
        "turn_index": (sample.metadata or {}).get("turn_index"),
        "is_last_turn": (sample.metadata or {}).get("is_last_turn"),
        "judge_prompt_variant": prompt_variant,
        "judge_score": judge_score,
        "judge_timeout": judge_timeout,
        "answer": answer,
        "action_history": action_history,
        "num_screenshots": len(screenshots),
        "messages": _to_jsonable(messages),
        "judge_text": judge_text,
    }

    with open(os.path.join(debug_dir, filename), "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

# ==============================================================================
# LLM Judge Prompts (adapted from WebVoyager)
# ==============================================================================

JUDGE_SYSTEM_PROMPT = (
    "As an evaluator, you will be presented with three primary components "
    "to assist you in your role:\n\n"
    "1. Web Task Instruction: This is a clear and specific directive provided "
    "in natural language, detailing the online activity to be carried out.\n\n"
    "2. Result Screenshots: This is a visual representation of the screen "
    "showing the result or intermediate state of performing a web task.\n\n"
    "3. Result Response: This is a textual response obtained after the "
    "execution of the web task.\n\n"
    "-- You DO NOT NEED to interact with web pages or perform actions.\n"
    "-- You SHOULD NOT make assumptions based on information not presented "
    "in the screenshot when comparing it to the instructions.\n"
    "-- Your primary responsibility is to conduct a thorough assessment of "
    "the web task instruction against the outcome depicted in the screenshot "
    "and in the response, evaluating whether the actions taken align with "
    "the given instructions.\n"
    "-- NOTE that the instruction may involve more than one task. Failing to "
    "complete either task should be considered unsuccessful.\n"
    "-- NOTE that the screenshot is authentic, but the response provided by "
    "LLM is generated at the end of web browsing; there may be discrepancies "
    "between the text and the screenshots.\n"
    "-- Note the difference: 1) Result response may contradict the screenshot, "
    "then the content of the screenshot prevails, 2) The content in the Result "
    "response is not mentioned on the screenshot, choose to believe the content.\n\n"
    "You should elaborate on how you arrived at your final evaluation and then "
    "provide a definitive verdict on whether the task has been successfully "
    "accomplished, either as 'SUCCESS' or 'NOT SUCCESS'."
)

JUDGE_SYSTEM_PROMPT_ACTION_HISTORY = (
    "As an evaluator, you will be presented with four primary components "
    "to assist you in your role:\n\n"
    "1. Web Task Instruction: This is a clear and specific directive provided "
    "in natural language, detailing the online activity to be carried out.\n\n"
    "2. Agent Action History: This is a chronological summary of the agent's "
    "observed actions across steps. Use it to understand what the agent tried "
    "to do, but do not treat it as ground truth if it conflicts with the "
    "screenshots.\n\n"
    "3. Result Screenshots: This is a visual representation of the screen "
    "showing the result or intermediate state of performing a web task. "
    "Each screenshot will be annotated with an inferred step index in text.\n\n"
    "4. Result Response: This is a textual response obtained after the "
    "execution of the web task.\n\n"
    "-- You DO NOT NEED to interact with web pages or perform actions.\n"
    "-- You SHOULD use the screenshots as the strongest evidence about the "
    "actual page state.\n"
    "-- You SHOULD use the action history to judge whether the agent followed "
    "the instruction and whether the final response is supported by what "
    "happened on screen.\n"
    "-- If the action history conflicts with screenshots, trust the screenshots.\n"
    "-- NOTE that the instruction may involve more than one task. Failing to "
    "complete either task should be considered unsuccessful.\n"
    "-- NOTE that the final response may contradict the screenshots; in that "
    "case the screenshots prevail. If the final response contains information "
    "not visible in the screenshots, you may still consider it only if it is "
    "consistent with the screenshots and action history.\n\n"
    "You should first explain your reasoning with explicit reference to the "
    "instruction, action history, screenshots, and final response. Then provide "
    "a definitive verdict as either 'SUCCESS' or 'NOT SUCCESS'."
)

JUDGE_USER_PROMPT = (
    "### TASK: {task}\n"
    "### Result Response: {answer}\n"
    "### {num} screenshots at the end: "
)

JUDGE_USER_PROMPT_ACTION_HISTORY = (
    "### TASK: {task}\n"
    "### Agent Action History:\n{action_history}\n\n"
    "### Result Response: {answer}\n"
    "### {num} screenshots from the trajectory are attached below with inferred step indices.\n"
)


# ==============================================================================
# 1. Format Reward
# ==============================================================================

# Module-level ToolParser for format reward checks.
# Uses regex mode (no tools_info) — same parsing as generate_browser.py L328-333.
_tool_parser = ToolParser(parser_type="regex")


def _get_browser_response_mode(args: Any):
    return get_browser_response_format_mode(getattr(args, "browser_response_format_mode", "slime"))


def _build_thinking_block_pattern(thinking_open_tag: str, thinking_close_tag: str) -> str:
    return rf"{re.escape(thinking_open_tag)}.*?{re.escape(thinking_close_tag)}"


def _is_tool_call_parseable(response: str) -> bool:
    """
    Check whether the response contains a successfully parsed tool call.

    Uses the same ToolParser that generate_browser.py uses:
        parsed = adapter.tool_parser.parse(llm_response)
        if parsed.success and len(parsed.calls) > 0: ...

    Args:
        response: The full assistant response text.

    Returns:
        True if at least one tool call was successfully parsed.
    """
    parsed = _tool_parser.parse(response)
    return parsed.success and len(parsed.calls) > 0


def _extract_tool_names_from_action(text: str) -> set:
    """
    Extract tool names from action block content (for partial credit).

    Detects the common failure pattern where the model puts bare tool names
    inside <action> instead of using <tool_call>:
        <action>
        click
        arguments:
        {"point_2d": [375, 80]}
        </action>
    """
    found = set()
    # Bare tool name on its own line, followed by "arguments:" on next line
    bare_names = re.findall(r'(?:^|\n)\s*(\w+)\s*\n\s*arguments?\s*:', text)
    found.update(bare_names)
    return found

def _action_contains_function_call(action_text: str) -> bool:
    """
    Detect whether the content inside <action>...</action> looks like a
    function call rather than a natural-language description.

    Common failure patterns caught:
        1. JSON object with "name" key:
              {"name": "click", "arguments": {...}}
        2. Python-style call with parenthesized arguments:
              click(point_2d=[375, 80])
        3. "arguments" keyword followed by = or : or { (function-call style):
              click
              arguments={"point_2d": [375, 80]}

    Note: bare tool names (e.g. "click") are NOT flagged, because they
    naturally appear in textual descriptions like "click on the search
    button".  The discriminator is the presence of structured argument
    syntax.

    Args:
        action_text: The raw text between <action> and </action>.

    Returns:
        True if the content resembles a function call.
    """
    text = action_text.strip()

    # 1. JSON-style: {"name": "...", ...}
    if re.search(r'[{"\']\s*name\s*["\']\s*:', text):
        return True

    # 2. Python-style function call: tool_name(key=value, ...)
    #    Must contain '=' inside parens (keyword args) to distinguish from
    #    natural parenthetical text like "English (UK)".
    if re.search(r'\b\w+\s*\([^)]*=.*\)', text, re.DOTALL):
        return True

    # 3. "arguments" keyword followed by = or : or { (function-call pattern)
    if re.search(r'\barguments?\s*[=:{]', text):
        return True

    return False


def _compute_turn_format_reward_slime(args: Any, response: str) -> float:
    """
    Compute a deterministic format reward based on policy adherence.

    Checks each assistant turn for the required tags defined in
    system_prompt.md and validates tool call parsing.

    Scoring rubric (each worth 0.20, total = 1.0):
        - <thinking> tag present                           → 0.20
        - <action> tag present with *textual* description  → 0.20
        - <tool_call> tag present                          → 0.20
        - Successfully parsed tool call inside <tool_call> → 0.20
        - Proper tag ordering (thinking→action→tool_call)  → 0.20

    The <action> block must contain a natural-language description of
    the intended action, NOT a function call.  If function-call-like
    content is detected inside <action> the score is capped at 0.80
    (the action criterion fails).

    Args:
        response: The assistant's response text (single turn).

    Returns:
        Float in [0.0, 1.0].
    """
    score = 0.0

    # Strip chat-template delimiters that may still be attached to the response.
    response = response.strip()
    if response.endswith("<|im_end|>"):
        response = response[: -len("<|im_end|>")].rstrip()

    response_mode = _get_browser_response_mode(args)
    thinking_block_pattern = _build_thinking_block_pattern(
        response_mode.thinking_open_tag,
        response_mode.thinking_close_tag,
    )

    # --- Tag presence checks ---
    has_thinking = bool(re.search(thinking_block_pattern, response, re.DOTALL))
    has_action = bool(re.search(r"<action>.*?</action>", response, re.DOTALL))
    has_tool_call = bool(re.search(r"<tool_call>.*?</tool_call>", response, re.DOTALL))

    # --- Check that <action> contains text, not a function call ---
    action_is_textual = False
    if has_action:
        action_match = re.search(r"<action>(.*?)</action>", response, re.DOTALL)
        if action_match and not _action_contains_function_call(action_match.group(1)):
            action_is_textual = True

    if has_thinking and has_action and action_is_textual and has_tool_call and _is_tool_call_parseable(response):
        strict_pattern = (
            rf"^\s*{thinking_block_pattern}"
            r"\s*<action>.*?</action>"
            r"\s*<tool_call>.*?</tool_call>\s*$"
        )
        if re.match(strict_pattern, response, re.DOTALL):
            score = 1.0   # all 5 criteria met

    return score


def _compute_turn_format_reward_browser_env(args: Any, response: str) -> float:
    response = response.strip()
    if response.endswith("<|im_end|>"):
        response = response[: -len("<|im_end|>")].rstrip()

    response_mode = _get_browser_response_mode(args)
    has_think_close = response_mode.thinking_close_tag in response
    parsed = _tool_parser.parse(response)
    has_tool_call = parsed.success and len(parsed.calls) > 0
    return 1.0 if has_think_close and has_tool_call else 0.0


def compute_format_reward(args: Any, response: str | None = None, turn_level: bool = False) -> float:
    """
    Compute format reward across (multiple) assistant turns in a
    sample response string.

    For trajectory-level response strings, each assistant turn is delimited by ``<|im_start|>assistant\\n`` …
    ``<|im_end|>``. All turns are extracted, scored individually, and
    averaged.

    For turn-level response strings, the full response is treated as a single turn.

    Args:
        response: Full trajectory response string.
        turn_level: Whether to treat the response as a single turn.

    Returns:
        1.0 only if every turn scores 1.0; otherwise 0.0.
    """
    if response is None:
        response = str(args)
        args = None

    response_mode = _get_browser_response_mode(args) if args is not None else get_browser_response_format_mode("slime")
    if turn_level:
        if response_mode.name == "slime":
            return _compute_turn_format_reward_slime(args, response)
        return _compute_turn_format_reward_browser_env(args, response)
    
    # Extract every assistant turn: content between <|im_start|>assistant\n
    # and <|im_end|>.  This is the standard chat format used in the data files.
    turn_pattern = r"<\|im_start\|>assistant\s*\n(.*?)<\|im_end\|>"
    turns = re.findall(turn_pattern, response, re.DOTALL)
    turns = [t.strip() for t in turns if t.strip()]

    if not turns:
        return 0.0

    if response_mode.name == "slime":
        scores = [_compute_turn_format_reward_slime(args, turn) for turn in turns]
    else:
        scores = [_compute_turn_format_reward_browser_env(args, turn) for turn in turns]
    if all(s == 1.0 for s in scores):
        return 1.0
    return 0.0


# ==============================================================================
# 2. LLM-as-a-Judge Reward
# ==============================================================================

def _extract_final_answer_slime(response: str) -> str:
    """
    Extract the final answer from the agent's response.

    Looks for the done() tool call, which contains the agent's final answer.
    Falls back to the last non-empty text block.

    Args:
        response: Agent response text.

    Returns:
        Extracted answer string.
    """
    # Look for done(response="...") style
    done_match = re.search(
        r'done\s*\(\s*response\s*=\s*["\'](.+?)["\']\s*\)',
        response,
        re.DOTALL,
    )
    if done_match:
        return done_match.group(1).strip()

    # Look for JSON-style done tool call: {"name": "done", "arguments": {"response": "..."}}
    done_json = re.search(
        r'"name"\s*:\s*"done".*?"response"\s*:\s*"(.+?)"',
        response,
        re.DOTALL,
    )
    if done_json:
        return done_json.group(1).strip()

    # Fallback: last non-empty line that isn't a tag
    lines = response.strip().split("\n")
    for line in reversed(lines):
        line = line.strip()
        if line and not line.startswith("<") and not line.startswith("["):
            return line

    return response.strip()[-500:] if response.strip() else ""


def _extract_final_answer_browser_env(response: str) -> str:
    parsed = _tool_parser.parse(response)
    if parsed.success and parsed.calls:
        last_call = parsed.calls[-1]
        if last_call.name != "done":
            return ""
        params = (
            json.loads(last_call.parameters)
            if isinstance(last_call.parameters, str)
            else last_call.parameters
        )
        if isinstance(params, dict):
            return str(params.get("response", "")).strip()
    return ""


def _extract_final_answer(args: Any, response: str) -> str:
    response_mode = _get_browser_response_mode(args)
    if response_mode.name == "browser_env":
        return _extract_final_answer_browser_env(response)
    return _extract_final_answer_slime(response)


def _get_judge_screenshots(args: Any, sample: Sample) -> list[str]:
    """
    Return the screenshots to attach to the judge prompt.

    Turn-level reward may inject a ``judge_images`` override containing
    trajectory-level recent screenshots; otherwise we fall back to the sample's
    own multimodal inputs.
    """
    multimodal_inputs = sample.multimodal_inputs or {}
    screenshots = multimodal_inputs.get("judge_images") or multimodal_inputs.get("images") or []

    max_imgs = getattr(args, "judge_max_attached_imgs")
    return screenshots[-max_imgs:] if screenshots else []


def _collect_recent_trajectory_images(samples: list[Sample], max_imgs: int) -> list[str]:
    """
    Aggregate one representative screenshot per turn sample and keep the most recent ones.

    For turn-level samples, ``multimodal_inputs["images"]`` may contain a rolling
    context window. We take only the last image from each turn sample, which
    corresponds to that turn's latest observation, then keep the most recent
    ``max_imgs`` of those turn-level screenshots.
    """
    ordered_samples = sorted(samples, key=lambda s: s.metadata.get("turn_index", -1))
    all_images: list[str] = []

    for sample in ordered_samples:
        multimodal_inputs = sample.multimodal_inputs or {}
        sample_images = multimodal_inputs.get("images", []) or []
        if sample_images:
            all_images.append(sample_images[-1])

    return all_images[-max_imgs:] if all_images else []


def _extract_assistant_turns(response: str) -> list[str]:
    turn_pattern = r"<\|im_start\|>assistant\s*\n(.*?)<\|im_end\|>"
    turns = re.findall(turn_pattern, response or "", re.DOTALL)
    turns = [t.strip() for t in turns if t.strip()]
    if turns:
        return turns
    response = (response or "").strip()
    return [response] if response else []


def _extract_action_history_slime(response: str, max_actions: int | None = None) -> str:
    turns = _extract_assistant_turns(response)
    history_lines: list[str] = []

    for idx, turn in enumerate(turns, start=1):
        action_match = re.search(r"<action>\s*(.*?)\s*</action>", turn, re.DOTALL)
        action_text = action_match.group(1).strip() if action_match else ""

        # tool_calls: list[str] = []
        # for match in re.findall(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", turn, re.DOTALL):
        #     try:
        #         payload = json.loads(match)
        #         name = payload.get("name", "unknown")
        #         args = payload.get("arguments", {}) or {}
        #         if args:
        #             tool_calls.append(f"{name}({json.dumps(args, ensure_ascii=False, sort_keys=True)})")
        #         else:
        #             tool_calls.append(name)
        #     except json.JSONDecodeError:
        #         tool_calls.append("unparseable_tool_call")

        parts = [f"Step {idx}:"]
        if action_text:
            parts.append(f"action={action_text}")
        # if tool_calls:
        #     parts.append(f"tool_calls=[{'; '.join(tool_calls)}]")
        if len(parts) > 1:
            history_lines.append(" ".join(parts))

    if max_actions is not None and max_actions > 0:
        history_lines = history_lines[-max_actions:]

    if not history_lines:
        return "No action history could be extracted from the agent response."
    return "\n".join(history_lines)


def _clean_think_from_action_history(text: str, args: Any) -> str:
    response_mode = _get_browser_response_mode(args)
    pattern = _build_thinking_block_pattern(
        response_mode.thinking_open_tag,
        response_mode.thinking_close_tag,
    )
    return re.sub(pattern, "", text, flags=re.DOTALL)


def _clean_img_placeholders_from_action_history(text: str) -> str:
    screenshot_placeholder = "screenshot:\n<|vision_start|><|image_pad|><|vision_end|>\n"
    return text.replace(screenshot_placeholder, "")


def _format_tool_call_for_action_history(name: str, parameters: Any) -> str:
    if not isinstance(name, str) or not name.strip():
        name = "unknown_tool"
    if parameters is None:
        parameters = {}
    if not isinstance(parameters, dict):
        return f"{name}({str(parameters)})"
    if not parameters:
        return f"{name}()"
    return f"{name}({json.dumps(parameters, ensure_ascii=False, sort_keys=True)})"


def _extract_browser_env_tool_calls(response: str) -> list[str]:
    parsed = _tool_parser.parse((response or "").strip())
    if parsed.success and parsed.calls:
        return [
            _format_tool_call_for_action_history(call.name, call.parameters)
            for call in parsed.calls
        ]

    tool_calls: list[str] = []
    for match in re.finditer(
        r'\{\s*"name"\s*:\s*"(?P<name>[^"]+)"\s*,\s*"arguments"\s*:\s*(?P<args>\{.*?\})\s*\}',
        response or "",
        re.DOTALL,
    ):
        name = match.group("name")
        try:
            parameters = json.loads(match.group("args"))
        except json.JSONDecodeError:
            parameters = match.group("args")
        tool_calls.append(_format_tool_call_for_action_history(name, parameters))
    return tool_calls


def _normalize_action_history_text(text: str) -> str:
    return normalize_feedback_text(text)


def _get_history_env_feedback_max_chars(args: Any) -> int:
    value = getattr(args, "browser_history_env_feedback_max_chars", None)
    if value is None:
        value = _get_env_override("SLIME_BROWSER_HISTORY_ENV_FEEDBACK_MAX_CHARS")
    if value is None:
        value = DEFAULT_BROWSER_HISTORY_ENV_FEEDBACK_MAX_CHARS
    try:
        return max(int(value), 1)
    except (TypeError, ValueError):
        return DEFAULT_BROWSER_HISTORY_ENV_FEEDBACK_MAX_CHARS


def _extract_browser_env_action_summary(args: Any, response: str) -> str:
    response = (response or "").strip()
    if not response:
        return ""

    tool_calls = _extract_browser_env_tool_calls(response)

    # Browser-env responses often interleave visible reasoning, malformed
    # closing tags, and raw JSON tool calls in the same text span. For judge
    # action history, the executed tool calls are the most reliable signal.
    if tool_calls:
        return f"[{'; '.join(tool_calls)}]"

    cleaned = _clean_think_from_action_history(response, args)
    cleaned = _clean_img_placeholders_from_action_history(cleaned)
    cleaned = re.sub(r"</?tool_call>", "", cleaned)
    cleaned = re.sub(r"<\|im_end\|>", "", cleaned)
    cleaned = re.sub(r"</think>|<think>|</thinking>|<thinking>", "", cleaned)
    cleaned = re.sub(r'\{\s*"name"\s*:\s*".*?"\s*,\s*"arguments"\s*:\s*\{.*?\}\s*\}', "", cleaned)
    return _normalize_action_history_text(cleaned)


def _extract_step_environment_feedback(sample: Sample, args: Any | None = None) -> str:
    metadata = sample.metadata or {}
    feedback_parts: list[str] = []
    max_chars = _get_history_env_feedback_max_chars(args)

    tool_responses = metadata.get("step_tool_responses")
    if isinstance(tool_responses, list):
        for entry in tool_responses:
            if not isinstance(entry, dict):
                continue
            tool_name = str(entry.get("tool_name", "")).strip()
            tool_response = truncate_feedback_text(entry.get("tool_response", ""), max_chars)
            if not tool_response:
                continue
            if tool_name:
                feedback_parts.append(f"{tool_name}: {tool_response}")
            else:
                feedback_parts.append(tool_response)

    return " | ".join(part for part in feedback_parts if part)


def _extract_action_history_browser_env(args: Any, text: str) -> str:
    turns = _extract_assistant_turns(text)
    history_lines: list[str] = []

    for idx, turn in enumerate(turns, start=1):
        action_summary = _extract_browser_env_action_summary(args, turn)
        if action_summary:
            history_lines.append(f"Step {idx}: action={action_summary}")

    if history_lines:
        return "\n".join(history_lines)

    cleaned = _clean_img_placeholders_from_action_history(text)
    cleaned = _clean_think_from_action_history(cleaned, args)
    cleaned = cleaned.strip()
    if not cleaned:
        return "No action history could be extracted from the agent response."
    return cleaned


def _build_turn_level_action_history_for_browser_env(args: Any, samples: list[Sample]) -> str:
    ordered_samples = sorted(samples, key=lambda s: s.metadata.get("turn_index", -1))
    history_lines: list[str] = []

    for idx, sample in enumerate(ordered_samples, start=1):
        turn_index = sample.metadata.get("turn_index")
        display_step = turn_index + 1 if isinstance(turn_index, int) and turn_index >= 0 else idx
        action_summary = _extract_browser_env_action_summary(args, sample.response or "")
        env_feedback = _extract_step_environment_feedback(sample, args)

        parts = [f"Step {display_step}:"]
        if action_summary:
            parts.append(f"action={action_summary}")
        if env_feedback:
            parts.append(f"environment feedback={env_feedback}")

        if len(parts) > 1:
            history_lines.append(" ".join(parts))

    if not history_lines:
        return "No action history could be extracted from the agent response."
    return "\n".join(history_lines)


def _build_turn_level_action_history_response(samples: list[Sample]) -> str:
    """
    Reconstruct a multi-turn assistant transcript from turn-level samples.

    Each turn-level sample stores only its own assistant response in
    ``sample.response``. To extract judge-facing action history across the full
    trajectory, we concatenate those responses in turn order using the same
    assistant delimiters handled by ``_extract_assistant_turns``.
    """
    ordered_samples = sorted(samples, key=lambda s: s.metadata.get("turn_index", -1))
    assistant_turns: list[str] = []

    for sample in ordered_samples:
        response = (sample.response or "").strip()
        if not response:
            continue
        assistant_turns.append(f"<|im_start|>assistant\n{response}\n<|im_end|>")

    return "\n".join(assistant_turns)


def _infer_recent_image_step_indices(sample: Sample, num_selected_images: int) -> list[int | None]:
    """
    Infer display step indices for judge-facing prompts.

    Notes:
    - ``sample.metadata['turn_index']`` is stored as 0-based in generate_browser.py.
    - We convert all step indices shown to the judge into 1-based numbering so
      action-history steps and screenshot labels follow the same convention.
    """
    if num_selected_images <= 0:
        return []

    turn_index = sample.metadata.get("turn_index")
    if isinstance(turn_index, int) and turn_index >= 0:
        end_display_step = turn_index + 1
        start_display_step = max(1, end_display_step - num_selected_images + 1)
        return list(range(start_display_step, end_display_step + 1))

    total_steps = sample.metadata.get("total_steps")
    if isinstance(total_steps, int) and total_steps > 0:
        start = max(1, total_steps - num_selected_images + 1)
        return list(range(start, start + num_selected_images))

    return [None] * num_selected_images


def _build_user_content_with_step_indices(
    sample: Sample,
    user_text: str,
    screenshots: list[str],
) -> list[dict[str, Any]]:
    user_content: list[dict[str, Any]] = [{"type": "text", "text": user_text}]
    step_indices = _infer_recent_image_step_indices(sample, len(screenshots))

    for img_idx, img_b64 in enumerate(screenshots, start=1):
        inferred_step = step_indices[img_idx - 1]
        if inferred_step is None:
            label = f"Screenshot {img_idx}."
        else:
            label = f"Screenshot {img_idx} (inferred step {inferred_step})."
        user_content.append({"type": "text", "text": label})

        if img_b64.startswith("data:"):
            url = img_b64
        else:
            url = f"data:image/png;base64,{img_b64}"
        user_content.append({
            "type": "image_url",
            "image_url": {"url": url},
        })

    user_content.append({"type": "text", "text": "Your verdict:\n"})
    return user_content


def _normalize_served_base_url(base_url: str) -> str:
    base_url = (base_url or "").strip()
    if not base_url:
        raise ValueError("Judge served mode requires JUDGE_API_BASE or JUDGE_API_HOST/JUDGE_API_PORT.")
    if not re.match(r"^https?://", base_url):
        base_url = f"http://{base_url}"
    base_url = base_url.rstrip("/")
    if not base_url.endswith("/v1"):
        base_url = f"{base_url}/v1"
    return base_url


def _get_openai_client(method="token") -> Any:
    """
    Create an AsyncAzureOpenAI client from environment variables.
    Supports both API key and Azure AD token authentication.
    """
    if method == "token":
        resource_name = os.environ.get("AZURE_RESOURCE_NAME")
        token_path = os.environ.get("AZURE_TOKEN_PATH")
        api_version = os.environ.get("AZURE_API_VERSION", "2025-02-01-preview")
        if not resource_name:
            raise ValueError("Judge token mode requires AZURE_RESOURCE_NAME.")
        if not token_path:
            raise ValueError("Judge token mode requires AZURE_TOKEN_PATH.")
        with open(token_path) as f:
            token = f.read().strip()
        return AsyncAzureOpenAI(
            azure_ad_token=token,
            api_version=api_version,
            azure_endpoint=f"https://{resource_name}.openai.azure.com/",
        )
    if method == "api_key":
        api_key = os.environ.get("OPENAI_API_KEY")
        base_url = os.environ.get("OPENAI_API_BASE")
        api_version = os.environ.get("OPENAI_API_VERSION", "2025-04-01-preview")

        kwargs: dict[str, Any] = {"api_key": api_key, "base_url": base_url, "api_version": api_version}
        return AsyncAzureOpenAI(**kwargs)

    if method == "served":
        base_url = os.environ.get("JUDGE_API_BASE")
        if not base_url:
            host = os.environ.get("JUDGE_API_HOST", "").strip()
            port = os.environ.get("JUDGE_API_PORT", "").strip()
            if host and port:
                base_url = f"{host}:{port}"
            elif host:
                base_url = host

        base_url = _normalize_served_base_url(base_url)
        api_key = os.environ.get("JUDGE_API_KEY") or os.environ.get("OPENAI_API_KEY") or "EMPTY"
        return AsyncOpenAI(api_key=api_key, base_url=base_url)

    raise ValueError(f"Unsupported judge_api_mode: {method}")


async def compute_judge_reward(
    args: Any,
    sample: Sample,
) -> tuple[float, str, bool]:
    """
    Compute LLM-as-a-Judge reward using an OpenAI-compatible API.

    Builds a multimodal evaluation prompt with the task instruction,
    the agent's final answer, and (optionally) screenshots, then
    parses SUCCESS / NOT SUCCESS from the judge's response.

    Args:
        args: Training arguments (may contain judge config overrides).
        sample: Sample with response and metadata.

    Returns:
        1.0 for SUCCESS, 0.0 for NOT SUCCESS or on error.
    """
    # --- Extract task and answer ---
    task = sample.metadata["intent"]

    answer = _extract_final_answer(args, sample.response or "")
    if not answer:
        logger.warning("No answer extracted from response; judge score = 0.0")
        return 0.0, "No answer extracted.", False

    # --- Collect screenshots ---
    screenshots = _get_judge_screenshots(args, sample)
    num_imgs = len(screenshots)

    # --- Build messages ---
    user_text = JUDGE_USER_PROMPT.format(
        task=task,
        answer=answer,
        num=str(num_imgs) if num_imgs > 0 else "0",
    )

    user_content: list[dict[str, Any]] = [{"type": "text", "text": user_text}]
    for img_b64 in screenshots:
        # Support both raw base64 and data-URL format
        if img_b64.startswith("data:"):
            url = img_b64
        else:
            url = f"data:image/png;base64,{img_b64}"
        user_content.append({
            "type": "image_url",
            "image_url": {"url": url},
        })
    user_content.append({"type": "text", "text": "Your verdict:\n"})

    messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    # --- Call LLM judge ---
    api_model = getattr(args, "judge_api_model")
    judge_timeout_secs = getattr(args, "judge_timeout_secs", 30.0)
    # max_tokens = getattr(args, "judge_max_tokens")
    # temperature = getattr(args, "judge_temperature")

    judge_api_mode = getattr(args, "judge_api_mode", "token")
    client = _get_openai_client(method=judge_api_mode)
    # if os.environ.get("OPENAI_API_BASE") and os.environ.get("OPENAI_API_KEY"):
    #     client = _get_openai_client(method="api_key")
    # else:
    #     raise EnvironmentError("OPENAI_API_BASE and OPENAI_API_KEY must be set for judge reward computation.")

    max_retries = 3
    saw_timeout = False
    for attempt in range(max_retries):
        try:
            judge_coro = client.chat.completions.create(
                model=api_model,
                messages=messages,
                # max_tokens=max_tokens,
                # temperature=temperature,
                seed=42,
            )
            if judge_timeout_secs is not None and judge_timeout_secs > 0:
                response = await asyncio.wait_for(judge_coro, timeout=judge_timeout_secs)
            else:
                response = await judge_coro
            judge_text = response.choices[0].message.content or ""
            logger.debug("Judge response: %s", judge_text)
            if _should_sample_judge_output(args):
                logger.info("Judge response: %s", judge_text)

            judge_score = 0.0
            if "NOT SUCCESS" in judge_text:
                judge_score = 0.0
            elif "SUCCESS" in judge_text:
                judge_score = 1.0
            else:
                logger.warning(f"Judge response does not contain SUCCESS/NOT SUCCESS: {judge_text}")
                judge_score = 0.0

            _save_judge_trace(
                args,
                sample,
                messages=messages,
                judge_text=judge_text,
                judge_score=judge_score,
                judge_timeout=False,
                prompt_variant="default",
                answer=answer,
                screenshots=screenshots,
            )
            return judge_score, judge_text, False

        except asyncio.TimeoutError as e:
            saw_timeout = True
            logger.warning(
                "Judge API call timed out (attempt %d/%d, timeout=%ss): %s",
                attempt + 1,
                max_retries,
                judge_timeout_secs,
                e,
            )
            if attempt < max_retries - 1:
                await asyncio.sleep(3 ** (attempt + 1))
            else:
                logger.error("All judge API retry attempts exhausted due to timeout.")
                return 0.0, f"Judge timeout exhausted after {max_retries} attempts.", True
        except Exception as e:
            logger.warning(
                "Judge API call failed (attempt %d/%d, timeout=%ss): %s",
                attempt + 1,
                max_retries,
                judge_timeout_secs,
                e,
            )
            if attempt < max_retries - 1:
                await asyncio.sleep(3 ** (attempt + 1))
            else:
                logger.error("All judge API retry attempts exhausted.")
                return 0.0, "All judge API retry attempts exhausted.", saw_timeout

    return 0.0, "Unexpected error occurred.", saw_timeout  # Should not reach here


async def compute_judge_reward_actionhistory(
    args: Any,
    sample: Sample,
    trajectory_action_history_response: str | None = None,
    trajectory_action_history_text: str | None = None,
) -> tuple[float, str, bool]:
    """
    Compute LLM-as-a-Judge reward with explicit action history and per-image step indices.
    """
    task = sample.metadata["intent"]

    answer = _extract_final_answer(args, sample.response or "")
    if not answer:
        logger.warning("No answer extracted from response; judge score = 0.0")
        return 0.0, "No answer extracted.", False

    screenshots = _get_judge_screenshots(args, sample)
    num_imgs = len(screenshots)

    response_mode = _get_browser_response_mode(args)
    if response_mode.name == "browser_env":
        action_history = trajectory_action_history_text
        if not action_history:
            action_history_source = (
                trajectory_action_history_response
                if trajectory_action_history_response is not None
                else f"{sample.prompt or ''}{sample.response or ''}"
            )
            action_history = _extract_action_history_browser_env(args, action_history_source)
    else:
        action_history_source = (
            trajectory_action_history_response
            if trajectory_action_history_response is not None
            else (sample.response or "")
        )
        action_history = _extract_action_history_slime(action_history_source, max_actions=None)

    user_text = JUDGE_USER_PROMPT_ACTION_HISTORY.format(
        task=task,
        action_history=action_history,
        answer=answer,
        num=str(num_imgs) if num_imgs > 0 else "0",
    )
    user_content = _build_user_content_with_step_indices(sample, user_text, screenshots)

    messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT_ACTION_HISTORY},
        {"role": "user", "content": user_content},
    ]

    api_model = getattr(args, "judge_api_model")
    judge_timeout_secs = getattr(args, "judge_timeout_secs", 30.0)

    judge_api_mode = getattr(args, "judge_api_mode", "token")
    client = _get_openai_client(method=judge_api_mode)

    max_retries = 3
    saw_timeout = False
    for attempt in range(max_retries):
        try:
            judge_coro = client.chat.completions.create(
                model=api_model,
                messages=messages,
                seed=42,
            )
            if judge_timeout_secs is not None and judge_timeout_secs > 0:
                response = await asyncio.wait_for(judge_coro, timeout=judge_timeout_secs)
            else:
                response = await judge_coro
            judge_text = response.choices[0].message.content or ""
            logger.debug("Judge response: %s", judge_text)
            if _should_sample_judge_output(args):
                logger.info("Judge response: %s", judge_text)

            judge_score = 0.0
            if "NOT SUCCESS" in judge_text:
                judge_score = 0.0
            elif "SUCCESS" in judge_text:
                judge_score = 1.0
            else:
                logger.warning(f"Judge response does not contain SUCCESS/NOT SUCCESS: {judge_text}")
                judge_score = 0.0

            _save_judge_trace(
                args,
                sample,
                messages=messages,
                judge_text=judge_text,
                judge_score=judge_score,
                judge_timeout=False,
                prompt_variant="action_history",
                answer=answer,
                screenshots=screenshots,
                action_history=action_history,
            )
            return judge_score, judge_text, False

        except asyncio.TimeoutError as e:
            saw_timeout = True
            logger.warning(
                "Judge API call timed out (attempt %d/%d, timeout=%ss): %s",
                attempt + 1,
                max_retries,
                judge_timeout_secs,
                e,
            )
            if attempt < max_retries - 1:
                await asyncio.sleep(3 ** (attempt + 1))
            else:
                logger.error("All judge API retry attempts exhausted due to timeout.")
                return 0.0, f"Judge timeout exhausted after {max_retries} attempts.", True
        except Exception as e:
            logger.warning(
                "Judge API call failed (attempt %d/%d, timeout=%ss): %s",
                attempt + 1,
                max_retries,
                judge_timeout_secs,
                e,
            )
            if attempt < max_retries - 1:
                await asyncio.sleep(3 ** (attempt + 1))
            else:
                logger.error("All judge API retry attempts exhausted.")
                return 0.0, "All judge API retry attempts exhausted.", saw_timeout

    return 0.0, "Unexpected error occurred.", saw_timeout


# ==============================================================================
# 3. Main Reward Function
# ==============================================================================

async def reward_func(args: Any, samples: Sample | list[Sample], **kwargs) -> float | list[float]:
    """
    Main async reward function for browser agent tasks (batch mode).

    Called by slime's ``batched_async_rm`` which passes the full list of samples.

    Combines two reward signals:
        reward = format_weight * format_score + judge_weight * judge_score

    Handles both turn-level and trajectory-level samples:
    - Turn-level: sample.response is a single assistant turn, and
      sample.metadata may contain 'status' and 'is_last_turn'.
      For turn-level samples, the reward is computed only on the last turn
      and then propagated to all turns in the trajectory.
    - Trajectory-level: sample.response contains the full multi-turn
      response string.

    Status values (from base.types.Status):
    - COMPLETED: agent called done(), episode finished successfully
    - TRUNCATED: agent reaches the `rollout_max_context_len` (only for trajectory-level samples).
                    got an "finish_type == length" from SGLang engine.
    - ABORTED:   got an "finish_type == abort" from SGLang engine.
    - FAILED:    trajectory sample: agent didn't call done() before reaching max steps.
                 turn-level sample: predicted tool calls cannot be parsed.

    Args:
        args: Training arguments.
        samples: List of Sample objects with response and metadata.
        **kwargs: Additional keyword arguments (unused).

    Returns:
        List of combined rewards (one float per sample).
    """
    if isinstance(samples, Sample):
        return await _score_single_sample(args, samples)

    # For turn-level samples (from generate_turn_sample), compute the reward
    # only on the last turn and share it across all turns in the trajectory.
    is_turn_level = any("turn_index" in s.metadata for s in samples)
    if is_turn_level:
        # Find the last turn sample
        last_turn_sample = None
        for s in samples:
            if s.metadata.get("is_last_turn", False):
                last_turn_sample = s
                break
        if last_turn_sample is None:
            raise ValueError("No turn sample with is_last_turn=True found in turn-level samples.")
            # Fallback: use the sample with the highest turn_index
            last_turn_sample = max(samples, key=lambda s: s.metadata.get("turn_index", 0))

        judge_max_attached_imgs = getattr(args, "judge_max_attached_imgs", 0) or 0 
        context_num_screenshots = getattr(args, "context_num_screenshots", 0) or 0 
        max_judge_imgs = max(judge_max_attached_imgs, context_num_screenshots)
        aggregated_judge_images = _collect_recent_trajectory_images(samples, max_judge_imgs)
        if aggregated_judge_images:
            if last_turn_sample.multimodal_inputs is None:
                last_turn_sample.multimodal_inputs = {}
            last_turn_sample.multimodal_inputs["judge_images"] = aggregated_judge_images

        trajectory_action_history_response = _build_turn_level_action_history_response(samples)
        if trajectory_action_history_response:
            last_turn_sample.metadata["trajectory_action_history_response"] = trajectory_action_history_response
        response_mode = _get_browser_response_mode(args)
        if response_mode.name == "browser_env":
            trajectory_action_history_text = _build_turn_level_action_history_for_browser_env(args, samples)
            if trajectory_action_history_text:
                last_turn_sample.metadata["trajectory_action_history_text"] = trajectory_action_history_text

        # Compute reward once on the last turn
        trajectory_reward = await _score_single_sample(args, last_turn_sample)
        # Propagate the same reward and reward metadata to all turns
        reward_meta = last_turn_sample.metadata.get("reward", {})
        for s in samples:
            s.metadata["reward"] = reward_meta
            if last_turn_sample.remove_sample:
                s.remove_sample = True
        return [trajectory_reward] * len(samples)

    tasks = [_score_single_sample(args, s) for s in samples]
    return await asyncio.gather(*tasks)


async def _score_single_sample(args: Any, sample: Sample) -> float:
    """Score a single sample (called in parallel by reward_func)."""
    if not isinstance(sample, Sample):
        raise TypeError(f"Expected Sample, got {type(sample)}")

    async with _reward_semaphore:
        # # Determine status from metadata (normalise to lowercase string)
        # raw_status = sample.metadata.get("status", "completed")
        # if isinstance(raw_status, str):
        #     # Handle "Status.ABORTED" → "aborted" format from data files
        #     status = raw_status.rsplit(".", 1)[-1].lower()
        # else:
        #     status = str(raw_status).lower()

        is_turn_level = True if "turn_index" in sample.metadata else False

        # --- Format reward (deterministic) ---
        format_score = compute_format_reward(args, sample.response, turn_level=is_turn_level)
    
        # --- LLM-as-a-Judge reward (async) ---
        # Only run judge on completed episodes (last turn with done() call)
        # or when is_last_turn is True.  For non-final turns, judge is 0.
        judge_timeout = False
        terminate_reason = str((sample.metadata or {}).get("terminate_reason", "")).lower()
        if terminate_reason == "format_error_failed":
            judge_score = 0.0
            judge_text = "Judge not run for terminate_reason=format_error_failed"
        elif sample.status != Sample.Status.COMPLETED:
            judge_score = 0.0
            judge_text = f"Judge not run for status={sample.status}"
        else:
            judge_variant = getattr(args, "judge_prompt_variant", "default")
            if judge_variant == "action_history":
                judge_score, judge_text, judge_timeout = await compute_judge_reward_actionhistory(
                    args,
                    sample,
                    trajectory_action_history_response=sample.metadata.get("trajectory_action_history_response"),
                    trajectory_action_history_text=sample.metadata.get("trajectory_action_history_text"),
                )
            else:
                judge_score, judge_text, judge_timeout = await compute_judge_reward(args, sample)
            if judge_timeout:
                sample.remove_sample = True
                sample.metadata["judge_timeout"] = True

        # --- Combine ---
        if terminate_reason == "format_error_failed":
            combined = -1.0
        elif format_score == 0.0:
            combined = 0.0
        else:
            combined = 1.0 if judge_score == 1.0 else 0.0

        logger.info(
            "Browser reward: format=%.3f judge=%.3f combined=%.3f (status=%s)",
            format_score, judge_score, combined, sample.status
        )

        sample.metadata["reward"] = {
            "format": format_score,
            "judge": judge_score,
            "combined": combined,
            "judge_text": judge_text,
            "judge_timeout": judge_timeout,
            "judge_prompt_variant": getattr(args, "judge_prompt_variant", "default"),
        }
        return combined
