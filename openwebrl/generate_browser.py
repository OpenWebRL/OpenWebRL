"""
Extensible Gym Environment Integration for slime Training

This module provides a standardized interface for training agents in various
gym-like environments using the slime framework. It uses an adapter pattern
to support multiple environments through a common interface.

Adapted from slime's multi-turn rollout interface.
"""

import base64
import io
import logging
import os
import sys
import yaml
import json
import random
import re
import time
from datetime import datetime
from contextlib import asynccontextmanager
from enum import Enum
from typing import Any
import asyncio
import torch

from PIL import Image

# Suppress noisy asyncio socket warnings that flood logs on connection failures.
# Set level to CRITICAL and install a filter as a safety net in case Ray or
# another library resets the level after import.
_asyncio_logger = logging.getLogger("asyncio")
_asyncio_logger.setLevel(logging.CRITICAL)
_asyncio_logger.addFilter(lambda record: "socket.send()" not in record.getMessage())

# Use fully-qualified imports so that importlib.import_module() in Ray workers
# can resolve them without sys.path hacks.
from loguru import logger
logger.remove()
logger.add(sys.stderr, level="INFO")

from openwebrl.base.adapter import BaseGymEnvAdapter
from openwebrl.base.registry import DEFAULT_REGISTRY as ENV_REGISTRY
from openwebrl.base.utils import ToolParser
from openwebrl.adapters.browser_adapter import BrowserEnvConfig, BrowserAdapter
from openwebrl.response_format import (
    ensure_prompt_ends_with_visible_thinking_tag,
    get_browser_response_format_mode,
    rewrite_policy_thinking_tags,
)

from slime.utils.types import Sample
from slime.utils.http_utils import post
from slime.utils.processing_utils import build_processor_kwargs
from slime.rollout.sglang_rollout import GenerateState

# Resolve the openwebrl/ directory once at import time.
_BROWSER_DIR = os.path.dirname(os.path.abspath(__file__))
_EXPT_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
_SANDBOX_CLEANUP_DONE = False
_BROWSER_USE_CLEANUP_DONE = False
_BROWSER_ROLLOUT_GATE: asyncio.Semaphore | None = None
_BROWSER_ROLLOUT_GATE_LIMIT: int | None = None
_BROWSER_ROLLOUT_GATE_LOCK = asyncio.Lock()
_BROWSER_ROLLOUT_GATE_WAITING = 0
_BROWSER_ROLLOUT_GATE_ACTIVE = 0
_BROWSER_HOST_BLACKLIST_PATH = os.path.join(_BROWSER_DIR, "data", "webgym_filtered_popular_blacklist_hosts.txt")
_TASK_JSONL_CACHE: dict[str, dict[str, Any]] = {}
_TASK_JSONL_INDEX_CACHE: dict[str, list[dict[str, Any]]] = {}


def _get_env_override(name: str) -> str | None:
    value = os.environ.get(name)
    if value is None:
        return None
    value = value.strip()
    return value if value else None


def _get_debug_trace_prob(args: Any) -> float:
    value = _get_env_override("SLIME_BROWSER_DEBUG_TRACE_PROB")
    if value is None:
        value = getattr(args, "debug_trace_prob", 0.0)
    try:
        prob = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, prob))


def _get_llm_output_log_prob(args: Any) -> float:
    value = _get_env_override("SLIME_BROWSER_LOG_LLM_OUTPUT_PROB")
    if value is None:
        value = getattr(args, "log_llm_output_prob", 0.1)
    try:
        prob = float(value)
    except (TypeError, ValueError):
        return 0.1
    return max(0.0, min(1.0, prob))


def _should_save_debug_trace(args: Any) -> bool:
    prob = _get_debug_trace_prob(args)
    return prob > 0.0 and random.random() < prob


def _get_or_create_debug_trace_info(args: Any, sample: Sample) -> tuple[bool, str | None]:
    metadata = sample.metadata if sample.metadata is not None else {}
    sample.metadata = metadata

    selected = metadata.get("debug_trace_selected")
    trace_id = metadata.get("debug_trace_id")
    if isinstance(selected, bool):
        return selected, trace_id if isinstance(trace_id, str) and trace_id else None

    selected = _should_save_debug_trace(args)
    if selected:
        task_id = str(metadata.get("task_id", "unknown")).replace("/", "_")
        trace_id = f"{task_id}_pid{os.getpid()}_{time.time_ns()}"
    else:
        trace_id = None

    metadata["debug_trace_selected"] = selected
    metadata["debug_trace_id"] = trace_id
    return selected, trace_id


def _get_debug_trace_dir(args: Any, trace_kind: str, sample_kind: str) -> str:
    save_dir = getattr(args, "save", "") or os.environ.get("SLIME_SAVE_DIR", "")
    if not save_dir:
        return ""
    return os.path.join(save_dir, "debug_traces", trace_kind, sample_kind)


def _should_log_llm_output(args: Any) -> bool:
    value = _get_env_override("SLIME_BROWSER_LOG_LLM_OUTPUT")
    if value is not None:
        return value.lower() in {"1", "true", "yes", "on"}
    return bool(getattr(args, "log_llm_output", False))


def _should_sample_llm_output(args: Any) -> bool:
    return _should_log_llm_output(args) and random.random() < _get_llm_output_log_prob(args)


def _should_log_task_start(args: Any) -> bool:
    return bool(getattr(args, "log_task_start", False))


def _make_turn_sample_index(parent_sample_index: int | str | None, turn_index: int) -> int | str:
    """Create a turn-level unique sample index while preserving trajectory_id separately."""
    if isinstance(parent_sample_index, int):
        return (parent_sample_index << 16) | int(turn_index)
    return f"{parent_sample_index}:{turn_index}"


def _extract_failed_navigation_url(error_text: str) -> str | None:
    match = re.search(r"Failed to navigate to (https?://[^\s\\'\"]+)", error_text)
    if match:
        return match.group(1)

    match = re.search(r"(https?://[^\s\\'\"]+)", error_text)
    return match.group(1) if match else None


def _should_blacklist_init_failure(error_text: str) -> bool:
    return (
        "env_server /reset error" in error_text
        and "Failed to navigate to" in error_text
        and any(
            marker in error_text
            for marker in (
                "ERR_NAME_NOT_RESOLVED",
                "ERR_HTTP2_PROTOCOL_ERROR",
                "ERR_CONNECTION_RESET",
                "ERR_CONNECTION_CLOSED",
                "ERR_CONNECTION_REFUSED",
                "ERR_SSL_PROTOCOL_ERROR",
            )
        )
    )


def _append_host_to_blacklist_if_needed(error_text: str) -> None:
    if not _should_blacklist_init_failure(error_text):
        return

    failed_url = _extract_failed_navigation_url(error_text)
    if not failed_url:
        return

    from urllib.parse import urlsplit

    host = urlsplit(failed_url).netloc.lower()
    if not host:
        return

    os.makedirs(os.path.dirname(_BROWSER_HOST_BLACKLIST_PATH), exist_ok=True)
    existing = set()
    trailing_newline = True
    if os.path.exists(_BROWSER_HOST_BLACKLIST_PATH):
        with open(_BROWSER_HOST_BLACKLIST_PATH, "rb") as f:
            content = f.read()
        trailing_newline = not content or content.endswith(b"\n")
        for raw_line in content.decode("utf-8", errors="ignore").splitlines():
            line = raw_line.strip().lower()
            if line and not line.startswith("#"):
                existing.add(line)

    if host in existing:
        return

    with open(_BROWSER_HOST_BLACKLIST_PATH, "a", encoding="utf-8") as f:
        if not trailing_newline:
            f.write("\n")
        f.write(f"{host}\n")
    logger.warning(f"Added host to browser blacklist after init failure: {host}")


def _get_browser_rollout_concurrency(args: Any) -> int:
    """Return the per-process browser rollout submission concurrency."""
    configured = getattr(args, "browser_rollout_concurrency", None)
    if configured is not None:
        return int(configured)

    env_limit = os.environ.get("SLIME_BROWSER_ROLLOUT_CONCURRENCY")
    if env_limit:
        return int(env_limit)

    sandbox_limit = os.environ.get("SLIME_BROWSER_SANDBOX_MAX_SANDBOXES")
    if sandbox_limit:
        return int(sandbox_limit)

    return 0


async def _get_browser_rollout_gate(args: Any) -> asyncio.Semaphore | None:
    """Lazily initialize an in-process gate for browser rollout submissions."""
    global _BROWSER_ROLLOUT_GATE, _BROWSER_ROLLOUT_GATE_LIMIT

    limit = _get_browser_rollout_concurrency(args)
    if limit <= 0:
        return None

    if _BROWSER_ROLLOUT_GATE is None:
        async with _BROWSER_ROLLOUT_GATE_LOCK:
            if _BROWSER_ROLLOUT_GATE is None:
                _BROWSER_ROLLOUT_GATE = asyncio.Semaphore(limit)
                _BROWSER_ROLLOUT_GATE_LIMIT = limit
            elif _BROWSER_ROLLOUT_GATE_LIMIT != limit:
                logger.warning(
                    "Browser rollout gate already initialized with concurrency=%d; ignoring new value %d in this process.",
                    _BROWSER_ROLLOUT_GATE_LIMIT,
                    limit,
                )
    return _BROWSER_ROLLOUT_GATE


@asynccontextmanager
async def _browser_rollout_submission_slot(args: Any, task_id: str):
    """Bound the number of browser tasks that may start concurrently in one process."""
    global _BROWSER_ROLLOUT_GATE_WAITING, _BROWSER_ROLLOUT_GATE_ACTIVE

    gate = await _get_browser_rollout_gate(args)
    if gate is None:
        yield
        return

    limit = _BROWSER_ROLLOUT_GATE_LIMIT or 0
    async with _BROWSER_ROLLOUT_GATE_LOCK:
        _BROWSER_ROLLOUT_GATE_WAITING += 1
        waiting = _BROWSER_ROLLOUT_GATE_WAITING
        active = _BROWSER_ROLLOUT_GATE_ACTIVE
    logger.info(
        "[BrowserRolloutGate] wait_start task_id=%s active=%d/%d waiting=%d",
        task_id,
        active,
        limit,
        waiting,
    )

    await gate.acquire()

    async with _BROWSER_ROLLOUT_GATE_LOCK:
        _BROWSER_ROLLOUT_GATE_WAITING = max(0, _BROWSER_ROLLOUT_GATE_WAITING - 1)
        _BROWSER_ROLLOUT_GATE_ACTIVE += 1
        waiting = _BROWSER_ROLLOUT_GATE_WAITING
        active = _BROWSER_ROLLOUT_GATE_ACTIVE
    logger.info(
        "[BrowserRolloutGate] acquired task_id=%s active=%d/%d waiting=%d",
        task_id,
        active,
        limit,
        waiting,
    )

    try:
        yield
    finally:
        gate.release()
        async with _BROWSER_ROLLOUT_GATE_LOCK:
            _BROWSER_ROLLOUT_GATE_ACTIVE = max(0, _BROWSER_ROLLOUT_GATE_ACTIVE - 1)
            waiting = _BROWSER_ROLLOUT_GATE_WAITING
            active = _BROWSER_ROLLOUT_GATE_ACTIVE
        logger.info(
            "[BrowserRolloutGate] released task_id=%s active=%d/%d waiting=%d",
            task_id,
            active,
            limit,
            waiting,
        )


_ACTION_BLOCK_RE = re.compile(r"<action>.*?</action>", re.DOTALL)
_TOOL_CALL_BLOCK_RE = re.compile(r"<tool_call>.*?</tool_call>", re.DOTALL)


def _get_browser_response_mode(args: Any):
    return get_browser_response_format_mode(getattr(args, "browser_response_format_mode", "slime"))


def _include_tool_response_in_rollout(args: Any) -> bool:
    value = getattr(args, "browser_include_tool_response", 1)
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no", "off"}
    return bool(value)


def _build_thinking_block_re(args: Any) -> re.Pattern[str]:
    mode = _get_browser_response_mode(args)
    return re.compile(
        rf"{re.escape(mode.thinking_open_tag)}.*?{re.escape(mode.thinking_close_tag)}\s*",
        re.DOTALL,
    )


def _compress_assistant_history_text(text: str, mode: str, args: Any) -> str:
    """
    Compress historical assistant turns in turn-level context.

    Modes:
    - ``full``: keep the text unchanged.
    - ``hide_thinking``: remove ``<thinking>...</thinking>`` blocks.
    - ``action_only``: keep only ``<action>`` / ``<tool_call>`` blocks.
    """
    if mode == "full":
        return text

    if mode == "hide_thinking":
        return _build_thinking_block_re(args).sub("", text).strip()

    if mode == "action_only":
        kept_blocks = []
        if _get_browser_response_mode(args).requires_action_block:
            kept_blocks.extend(match.group(0).strip() for match in _ACTION_BLOCK_RE.finditer(text))
        kept_blocks.extend(match.group(0).strip() for match in _TOOL_CALL_BLOCK_RE.finditer(text))
        return "\n".join(block for block in kept_blocks if block).strip()

    raise ValueError(
        f"Unsupported turn_history_reasoning_mode={mode!r}. "
        "Expected one of: 'full', 'hide_thinking', 'action_only'."
    )


def _compress_turn_history_messages(messages: list[dict[str, Any]], mode: str, args: Any) -> list[dict[str, Any]]:
    """
    Apply reasoning compression to historical assistant messages only.

    ``browser_history_reasoning_max_turns`` keeps action/env-feedback history
    intact while limiting full assistant reasoning to the latest N assistant
    turns. Older assistant turns hide thinking but keep non-thinking text.
    """
    reasoning_max_turns = max(int(getattr(args, "browser_history_reasoning_max_turns", 0) or 0), 0)
    assistant_indices = [
        idx
        for idx, msg in enumerate(messages)
        if msg.get("role") == "assistant" and isinstance(msg.get("content"), str)
    ]
    reasoning_keep_indices = (
        set(assistant_indices[-reasoning_max_turns:])
        if reasoning_max_turns > 0
        else set(assistant_indices)
    )

    if mode == "full" and reasoning_max_turns <= 0:
        return messages

    compressed_messages: list[dict[str, Any]] = []
    for idx, msg in enumerate(messages):
        if msg.get("role") == "assistant" and isinstance(msg.get("content"), str):
            compress_mode = mode if idx in reasoning_keep_indices else "hide_thinking"
            compressed_messages.append({
                **msg,
                "content": _compress_assistant_history_text(msg["content"], compress_mode, args),
            })
        else:
            compressed_messages.append(msg)
    return compressed_messages


def _assistant_prefill_opens_visible_thinking_block(
    prompt_text: str,
    response_mode: Any,
) -> bool:
    return prompt_text.rstrip().endswith(response_mode.thinking_open_tag)


def _restore_prefilled_thinking_tag_for_history(
    response_text: str,
    prompt_text: str,
    response_mode: Any,
) -> str:
    """
    Reconstruct assistant history so prior turns keep a visible opening
    thinking tag.

    During generation we may append `<think>` / `<thinking>` only as prompt
    prefill. In that case the model response starts *inside* the block, so the
    returned text lacks the opening tag. If we store that raw response into
    history, subsequent turns lose the prior opening tag entirely.
    """
    stripped = response_text.lstrip()
    if stripped.startswith(response_mode.thinking_open_tag):
        return response_text
    if _assistant_prefill_opens_visible_thinking_block(prompt_text, response_mode):
        return f"{response_mode.thinking_open_tag}\n{response_text}"
    return response_text


def _restore_assistant_history_blocks_in_prompt(
    prompt_text: str,
    messages: list[dict[str, Any]],
) -> str:
    """
    Re-inject historical assistant content after chat templating.

    Some Qwen chat templates discard visible `<think>...</think>` content from
    prior assistant turns and keep only tool-call blocks. For browser turn-level
    rollouts we want the full historical assistant reasoning to remain visible
    in the next-turn prompt, so we replace each serialized assistant block in
    the templated prompt with the original message content.
    """
    assistant_contents = [
        msg.get("content", "")
        for msg in messages
        if msg.get("role") == "assistant" and isinstance(msg.get("content"), str)
    ]
    if not assistant_contents:
        return prompt_text

    pattern = re.compile(r"(<\|im_start\|>assistant\s*\n)(.*?)(<\|im_end\|>)", re.DOTALL)
    assistant_idx = 0

    def _replace(match: re.Match[str]) -> str:
        nonlocal assistant_idx
        if assistant_idx >= len(assistant_contents):
            return match.group(0)
        content = assistant_contents[assistant_idx]
        assistant_idx += 1
        return f"{match.group(1)}{content}{match.group(3)}"

    return pattern.sub(_replace, prompt_text, count=len(assistant_contents))


def _build_fallback_task_data(task_id: str, task_metadata: dict[str, Any] | None) -> dict[str, Any]:
    task_metadata = task_metadata or {}
    start_url = task_metadata.get("start_url", "")
    intent = task_metadata.get("task") or task_metadata.get("intent", "")
    if not start_url or not intent:
        raise ValueError(
            f"❗  Task {task_id} not found locally and metadata fallback is incomplete: "
            f"start_url={bool(start_url)} intent={bool(intent)}"
        )

    numeric_task_id = task_id.split("/")[-1] if "/" in task_id else task_id
    return {
        "sites": task_metadata.get("sites", []),
        "task_id": numeric_task_id,
        "require_login": bool(task_metadata.get("require_login", False)),
        "storage_state": task_metadata.get("storage_state"),
        "start_url": start_url,
        "intent": intent,
        "require_reset": bool(task_metadata.get("require_reset", False)),
        "eval": task_metadata.get(
            "eval",
            {
                "eval_types": ["string_match"],
                "reference_answers": {"fuzzy_match": [""]},
                "reference_url": "",
                "program_html": [],
                "string_note": "",
                "reference_answer_raw_annotation": "",
            },
        ),
        "intent_template_id": task_metadata.get("intent_template_id", 0),
        "old_task_id": task_metadata.get("old_task_id", task_id),
    }


def _task_record_to_task_data(record: dict[str, Any], requested_task_id: str) -> dict[str, Any]:
    metadata = dict(record.get("metadata", {}) or {})
    resolved_task_id = (
        metadata.get("task_id")
        or record.get("id")
        or requested_task_id
    )
    start_url = metadata.get("start_url") or record.get("web", "")
    intent = metadata.get("intent") or record.get("ques", "")
    if not start_url or not intent:
        raise ValueError(
            f"❗  Task record for {requested_task_id} is missing required fields: "
            f"start_url={bool(start_url)} intent={bool(intent)}"
        )

    return {
        "sites": metadata.get("sites", [record.get("web_name")] if record.get("web_name") else []),
        "task_id": resolved_task_id,
        "require_login": bool(metadata.get("require_login", False)),
        "storage_state": metadata.get("storage_state"),
        "start_url": start_url,
        "intent": intent,
        "require_reset": bool(metadata.get("require_reset", False)),
        "eval": metadata.get(
            "eval",
            {
                "eval_types": ["string_match"],
                "reference_answers": {"fuzzy_match": [""]},
                "reference_url": "",
                "program_html": [],
                "string_note": "",
                "reference_answer_raw_annotation": "",
            },
        ),
        "intent_template_id": metadata.get("intent_template_id", 0),
        "old_task_id": record.get("id", resolved_task_id),
    }


def _load_task_jsonl_cache(path: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    cached_by_id = _TASK_JSONL_CACHE.get(path)
    cached_records = _TASK_JSONL_INDEX_CACHE.get(path)
    if cached_by_id is not None and cached_records is not None:
        return cached_by_id, cached_records

    by_id: dict[str, Any] = {}
    records: list[dict[str, Any]] = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            records.append(record)
            metadata = record.get("metadata", {}) or {}
            candidate_ids = [
                record.get("id"),
                metadata.get("task_id"),
            ]
            for candidate in candidate_ids:
                if isinstance(candidate, str) and candidate and candidate not in by_id:
                    by_id[candidate] = record

    _TASK_JSONL_CACHE[path] = by_id
    _TASK_JSONL_INDEX_CACHE[path] = records
    return by_id, records


def _collect_task_lookup_candidates(task_id: str, task_metadata: dict[str, Any] | None) -> list[str]:
    metadata = task_metadata or {}
    candidates: list[str] = []
    for value in (
        task_id,
        metadata.get("old_task_id"),
        metadata.get("task_id"),
        metadata.get("id"),
    ):
        if isinstance(value, str):
            value = value.strip()
            if value and value not in candidates:
                candidates.append(value)
    return candidates


def _try_load_task_from_jsonl(task_id: str, task_metadata: dict[str, Any] | None) -> dict[str, Any] | None:
    tasks_dir = os.path.join(_BROWSER_DIR, "env", "tasks")
    configured_task_file = None
    if task_metadata and isinstance(task_metadata.get("_browser_task_file"), str):
        configured_task_file = task_metadata["_browser_task_file"].strip() or None

    jsonl_paths: list[str] = []
    if configured_task_file:
        configured_path = configured_task_file
        if not os.path.isabs(configured_path):
            configured_path = os.path.join(_BROWSER_DIR, configured_path)
        if os.path.isfile(configured_path):
            jsonl_paths.append(configured_path)

    fallback_paths = []
    if os.path.isdir(tasks_dir):
        fallback_paths = sorted(
            path for path in (
                os.path.join(tasks_dir, name) for name in os.listdir(tasks_dir)
            )
            if path.endswith(".jsonl") and os.path.isfile(path)
        )
    for path in fallback_paths:
        if path not in jsonl_paths:
            jsonl_paths.append(path)

    if not jsonl_paths:
        return None

    candidates = _collect_task_lookup_candidates(task_id, task_metadata)
    for path in jsonl_paths:
        by_id, _ = _load_task_jsonl_cache(path)
        for candidate in candidates:
            record = by_id.get(candidate)
            if record is not None:
                logger.info(f"Loaded task {task_id} from JSONL {path} using key {candidate}")
                return _task_record_to_task_data(record, task_id)

    # Fallback for datasets that still use webvoyager/<numeric_id> while tasks are stored
    # in a single ordered webvoyager.jsonl file.
    match = re.fullmatch(r"webvoyager/(\d+)", task_id)
    if match:
        numeric_index = int(match.group(1))
        ordered_candidates = [
            os.path.join(tasks_dir, "webvoyager.jsonl"),
        ]
        for path in ordered_candidates:
            if not os.path.isfile(path):
                continue
            _, records = _load_task_jsonl_cache(path)
            if 0 <= numeric_index < len(records):
                logger.warning(
                    "Resolved task %s by positional index %d in %s; "
                    "prefer metadata.old_task_id/task_id for stable matching.",
                    task_id,
                    numeric_index,
                    path,
                )
                return _task_record_to_task_data(records[numeric_index], task_id)

    return None


def _load_local_resources(
    task_id: str,
    env_config: BrowserEnvConfig,
    task_metadata: dict[str, Any] | None = None,
    response_format_mode_name: str | None = None,
):
    """Load task_data, tool_list, and policy from the local filesystem."""
    tool_list = []
    if env_config.get("path_to_tool_list"):
        _tool_list_path = os.path.join(_BROWSER_DIR, env_config["path_to_tool_list"])
        if os.path.exists(_tool_list_path):
            with open(_tool_list_path, "r") as f:
                tool_list = json.load(f)

    response_mode = get_browser_response_format_mode(response_format_mode_name)

    policy = ""
    policy_relpath = response_mode.policy_relpath or env_config.get("path_to_policy")
    if policy_relpath:
        _policy_path = os.path.join(_BROWSER_DIR, policy_relpath)
        if os.path.exists(_policy_path):
            with open(_policy_path, "r") as f:
                policy = f.read()
    policy = rewrite_policy_thinking_tags(policy, response_mode)

    task_metadata = dict(task_metadata or {})
    configured_task_file = env_config.get("path_to_task_file")
    if configured_task_file and "_browser_task_file" not in task_metadata:
        task_metadata["_browser_task_file"] = configured_task_file

    _task_path = os.path.join(_BROWSER_DIR, "env", "tasks", f"{task_id}.json")
    if os.path.exists(_task_path):
        with open(_task_path, "r") as f:
            task_data = json.load(f)
    else:
        task_data = _try_load_task_from_jsonl(task_id, task_metadata)
        if task_data is None:
            logger.warning(f"Task {task_id} not found locally, falling back to sample metadata.")
            task_data = _build_fallback_task_data(task_id, task_metadata)

    return task_data, tool_list, policy


def _apply_browser_env_mode_override(env_config: BrowserEnvConfig) -> BrowserEnvConfig:
    """Allow launch scripts to select the browser env backend without editing config.yaml."""
    mode = os.environ.get("SLIME_BROWSER_ENV_MODE")
    if mode in (None, ""):
        return env_config
    if mode not in {"local_process", "sandbox", "browser-use"}:
        raise ValueError(
            "Unsupported SLIME_BROWSER_ENV_MODE="
            f"{mode!r}; expected one of local_process, sandbox, browser-use."
        )
    merged_config = dict(env_config)
    merged_config["mode"] = mode
    return merged_config


def _apply_sandbox_env_overrides(env_config: BrowserEnvConfig) -> BrowserEnvConfig:
    """Allow launch scripts to override sandbox settings via environment variables."""
    if env_config.get("mode", "sandbox") != "sandbox":
        return env_config

    sandbox_cfg = dict(env_config.get("sandbox", {}))

    max_sandboxes = os.environ.get("SLIME_BROWSER_SANDBOX_MAX_SANDBOXES")
    if max_sandboxes:
        sandbox_cfg["max_sandboxes"] = int(max_sandboxes)

    acquire_timeout = os.environ.get("SLIME_BROWSER_SANDBOX_ACQUIRE_TIMEOUT_SECS")
    if acquire_timeout:
        sandbox_cfg["acquire_timeout_secs"] = int(acquire_timeout)

    merged_config = dict(env_config)
    merged_config["sandbox"] = sandbox_cfg
    return merged_config


def _apply_local_process_env_overrides(env_config: BrowserEnvConfig) -> BrowserEnvConfig:
    """Allow launch scripts to override local_process settings via environment variables."""
    if env_config.get("mode", "sandbox") != "local_process":
        return env_config

    local_cfg = dict(env_config.get("local_process", {}))
    override_map = {
        "host": "SLIME_BROWSER_LOCAL_PROCESS_HOST",
        "max_processes": "SLIME_BROWSER_LOCAL_PROCESS_MAX_PROCESSES",
        "acquire_timeout_secs": "SLIME_BROWSER_LOCAL_PROCESS_ACQUIRE_TIMEOUT_SECS",
        "port_start": "SLIME_BROWSER_LOCAL_PROCESS_PORT_START",
        "port_end": "SLIME_BROWSER_LOCAL_PROCESS_PORT_END",
        "startup_timeout_secs": "SLIME_BROWSER_LOCAL_PROCESS_STARTUP_TIMEOUT_SECS",
        "startup_poll_secs": "SLIME_BROWSER_LOCAL_PROCESS_STARTUP_POLL_SECS",
        "request_timeout_secs": "SLIME_BROWSER_LOCAL_PROCESS_REQUEST_TIMEOUT_SECS",
        "exit_timeout_secs": "SLIME_BROWSER_LOCAL_PROCESS_EXIT_TIMEOUT_SECS",
        "kill_timeout_secs": "SLIME_BROWSER_LOCAL_PROCESS_KILL_TIMEOUT_SECS",
        "log_dir": "SLIME_BROWSER_LOCAL_PROCESS_LOG_DIR",
        "port_lock_dir": "SLIME_BROWSER_LOCAL_PROCESS_PORT_LOCK_DIR",
        "python_bin": "SLIME_BROWSER_LOCAL_PROCESS_PYTHON",
    }
    int_keys = {"max_processes", "port_start", "port_end"}
    float_keys = {
        "acquire_timeout_secs",
        "startup_timeout_secs",
        "startup_poll_secs",
        "request_timeout_secs",
        "exit_timeout_secs",
        "kill_timeout_secs",
    }
    for key, env_name in override_map.items():
        value = os.environ.get(env_name)
        if value in (None, ""):
            continue
        if key in int_keys:
            local_cfg[key] = int(value)
        elif key in float_keys:
            local_cfg[key] = float(value)
        else:
            local_cfg[key] = value

    merged_config = dict(env_config)
    merged_config["local_process"] = local_cfg
    return merged_config


async def _create_env(
    task_id: str,
    env_config: BrowserEnvConfig,
    task_metadata: dict[str, Any] | None = None,
    response_format_mode_name: str | None = None,
):
    """
    Create browser gym environment.

    Supports browser env modes (determined by ``env_config['mode']``):

    - ``"sandbox"``: K8s sandbox pods via orchestrator + exec/curl
    - ``"local_process"``: local env_server subprocess + HTTP
    - ``"browser-use"``: drive a remote browser on cloud.browser-use.com via CDP

    All config/task/policy data is loaded from the local filesystem and,
    in server-backed modes, sent to the env_server via the /reset request.

    Returns:
        (env, task_data) tuple
    """
    mode = env_config.get("mode", "sandbox")
    task_data, tool_list, policy = _load_local_resources(
        task_id,
        env_config,
        task_metadata,
        response_format_mode_name=response_format_mode_name,
    )

    # ---- Sandbox mode: either create a fresh sandbox or lease one from the pool ----
    if mode == "sandbox":
        sandbox_cfg = env_config.get("sandbox", {})

        from openwebrl.env.sandbox_env import create_sandbox_env, cleanup_existing_sandboxes

        # On first sandbox creation, clean up any leftover sandboxes
        # from a previous run that was aborted (e.g. Ctrl+C).
        global _SANDBOX_CLEANUP_DONE
        if not _SANDBOX_CLEANUP_DONE:
            await cleanup_existing_sandboxes(sandbox_cfg)
            _SANDBOX_CLEANUP_DONE = True

        env = None
        try:
            env = await create_sandbox_env(sandbox_cfg)
            await env.initialize(
                task_id=task_id,
                task_data=task_data,
                env_config=env_config,
                tool_list=tool_list,
                policy=policy,
            )
        except BaseException:
            # A cancellation can land after the sandbox slot is acquired and the
            # env is created, but before _initialize_resources() returns to the
            # caller. In that window the caller has no env handle yet, so we
            # must clean up here to avoid leaking the slot.
            if env is not None:
                try:
                    await env.exit()
                except Exception as cleanup_exc:
                    logger.warning(
                        "Task %s: failed to clean up sandbox env during initialization: %s",
                        task_id,
                        cleanup_exc,
                    )
            raise

        return env, task_data

    # ---- Browser-Use mode: create a fresh remote browser session per environment ----
    if mode == "browser-use":
        browser_use_cfg = env_config.get("browser_use", {}) or {}

        from openwebrl.env.browser_use_env import (
            create_browser_use_env,
            cleanup_existing_browser_use_sessions,
        )

        # On first session creation, stop any leftover remote sessions from a
        # previous run that was aborted (e.g. Ctrl+C). Browser-Use sessions bill
        # by the minute, so this sweep avoids charges until the server-side
        # timeout kicks in.
        global _BROWSER_USE_CLEANUP_DONE
        if not _BROWSER_USE_CLEANUP_DONE:
            await cleanup_existing_browser_use_sessions(browser_use_cfg)
            _BROWSER_USE_CLEANUP_DONE = True

        env = None
        try:
            env = await create_browser_use_env(
                browser_use_cfg=browser_use_cfg,
                env_config=env_config,
                tool_list=tool_list,
                policy=policy,
                start_url=task_data["start_url"],
            )
        except BaseException:
            if env is not None:
                try:
                    await env.exit()
                except Exception as cleanup_exc:
                    logger.warning(
                        "Task %s: failed to clean up browser-use env during initialization: %s",
                        task_id,
                        cleanup_exc,
                    )
            raise

        return env, task_data

    # ---- Local-process mode: fresh local env_server subprocess per environment ----
    if mode == "local_process":
        local_cfg = env_config.get("local_process", {})

        from openwebrl.env.local_process_env import create_local_process_env

        env = None
        try:
            env = await create_local_process_env(local_cfg)
            await env.initialize(
                task_id=task_id,
                task_data=task_data,
                env_config=env_config,
                tool_list=tool_list,
                policy=policy,
            )
        except BaseException:
            if env is not None:
                try:
                    await env.exit()
                except Exception as cleanup_exc:
                    logger.warning(
                        "Task %s: failed to clean up local_process env during initialization: %s",
                        task_id,
                        cleanup_exc,
                    )
            raise

        return env, task_data

    raise ValueError(f"Unsupported browser env mode={mode!r}; expected one of sandbox, local_process, browser-use.")

async def _initialize_resources(args: Any, task_id: str, task_metadata: dict[str, Any] | None = None):
    """Initialize adapter, generate state, and server URL."""
    env_name = "browser"
    
    # Register browser environment
    _this_dir = os.path.dirname(os.path.abspath(__file__))
    env_config_path = os.path.join(_this_dir, "env", "config.yaml")    
    if os.path.exists(env_config_path):
        with open(env_config_path, "r") as f:
            env_config = yaml.safe_load(f)
    else:
        raise ValueError(f"❗  Env config path {env_config_path} does not exist.")
    env_config = _apply_browser_env_mode_override(env_config)
    env_config = _apply_sandbox_env_overrides(env_config)
    env_config = _apply_local_process_env_overrides(env_config)

    ENV_REGISTRY.register(
        name=env_name,
        adapter_class=BrowserAdapter,
        config_class=BrowserEnvConfig,
        default_config=env_config,
    )

    env, task_data = await _create_env(
        task_id,
        env_config,
        task_metadata,
        response_format_mode_name=getattr(args, "browser_response_format_mode", "slime"),
    )

    # Create adapter for this environment
    adapter: BaseGymEnvAdapter = ENV_REGISTRY.create_adapter(name=env_name,
                                                            rollout_args=args,
                                                            config_overrides=None
                                                            )
    state = GenerateState(args)
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
    return env, adapter, state, url, task_data


async def _run_inference_step(
    url: str,
    text: str,
    sampling_params: dict,
    image_data: list,
    timeout_secs: float | None = None,
):
    """
    Run a single inference step via SGLang /generate endpoint.
    """
    payload = {
        "text": text,
        "sampling_params": sampling_params,
        "return_logprob": True,
    }
    if image_data:
        payload["image_data"] = image_data

    try:
        if timeout_secs is not None and timeout_secs > 0:
            output = await asyncio.wait_for(post(url, payload), timeout=timeout_secs)
        else:
            output = await post(url, payload)
    except asyncio.TimeoutError as exc:
        raise TimeoutError(
            f"SGLang /generate timed out after {timeout_secs}s for request to {url}"
        ) from exc
    response_text = output["text"]

    meta = output.get("meta_info", {})
    if "output_token_logprobs" in meta:
        new_tokens = [item[1] for item in meta["output_token_logprobs"]]
        new_log_probs = [item[0] for item in meta["output_token_logprobs"]]
    else:
        raise ValueError("❗  output_token_logprobs not found in meta_info")
        new_tokens, new_log_probs = [], []

    finish_type = meta.get("finish_reason", {}).get("type", "stop")

    response_text = _ensure_im_end_w_new_line(
        response_text, new_tokens, new_log_probs
    )

    return response_text, new_tokens, new_log_probs, finish_type


def _ensure_im_end_w_new_line(response_text: str, new_tokens: list[int], new_logprobs: list[float]) -> str:
    """
    Ensure the response text and tokens end with "<|im_end|>\n".
    Ensure the new tokens end with [151645, 198]
    """
    # Normalise text: strip trailing whitespace, append <|im_end|>\n
    response_text = response_text.rstrip()
    if not response_text.endswith("<|im_end|>"):
        response_text += "<|im_end|>"
    response_text += "\n"

    # Normalise tokens: ensure they end with [151645, 198]
    if new_tokens[-1] != 151645:
        new_tokens.extend([151645, 198])
        new_logprobs.extend([0.0, 0.0])
    else:
        new_tokens.append(198)
        new_logprobs.append(0.0)
    
    return response_text

def _ensure_correct_sample(sample: Sample, tokenizer, save_suffix: bool | None=None, debug: bool=False) -> None:
    """
    This function is for debugging: validates that ``sample.tokens`` has the correct total length and optionally
    writes a per-token debug file.

    The check accounts for the fact that ``tokenizer.encode()`` treats each
    ``<|image_pad|>`` placeholder as a single token, whereas ``sample.tokens``
    stores the fully-expanded visual tokens (``t * h * w // 4`` tokens per image).

    Args:
        sample: Sample whose token stream should be verified.
            - sample.prompt: empty string for trajectory samples; the text fed
            to the LLM for turn-level samples.
            - sample.response: the LLM's response text for turn-level samples;
            the complete templated text for trajectory samples.
            - sample.tokens: token IDs with each ``<|image_pad|>`` expanded to
            the actual number of visual tokens based on image dimensions.
            - sample.metadata["image_grid_thw"]: list of [temporal, height, width]
            for each image in the sample.
        tokenizer: Tokenizer used to re-encode prompt + response for comparison.
        save_sample: If True, writes a tab-separated debug file ``debug_sample.txt``
            where each line is: decoded_token \\t loss_mask \\t logprob \\t token_id.

    Raises:
        ValueError: If the length of ``sample.tokens`` does not equal
            len(tokenizer.encode(prompt + response)) + sum(t*h*w//4) - num_images.
    """
    # loss_mask and rollout_log_probs cover only the response portion (not prompt tokens).
    # Validate they match each other in length.
    if len(sample.loss_mask) != len(sample.rollout_log_probs):
        raise ValueError(
            f"❗  Length mismatch: loss_mask({len(sample.loss_mask)}) != "
            f"rollout_log_probs({len(sample.rollout_log_probs)}). They must be equal."
        )
    if sample.response_length != len(sample.loss_mask):
        raise ValueError(
            f"❗  response_length({sample.response_length}) != "
            f"len(loss_mask)({len(sample.loss_mask)}). They must be equal."
        )
    if debug:
        # compute the total number of image tokens after expansion
        cnt_img_tokens = []
        for thw in sample.metadata["image_grid_thw"]:
            cnt_img_tokens.append(thw[0] * thw[1] * thw[2] // 4)

        text_tokens = tokenizer.encode(sample.prompt + sample.response)
        num_images = len(sample.metadata["image_grid_thw"])
        expected = len(text_tokens) + sum(cnt_img_tokens) - num_images
        actual = len(sample.tokens)
        if actual != expected:
            print(f"{sample.response[-10:]!r}")
            img_tk = tokenizer.encode("<|image_pad|>", add_special_tokens=False)[0]
            text_token_str = "-".join([str(t) for t in text_tokens])
            actual_token_str = "-".join([str(t) for t in sample.tokens])
            # convert continuous multiple image tokens back to one image token
            for cnt_img_tk in cnt_img_tokens:
                actual_token_str = actual_token_str.replace("-".join([str(img_tk)] * cnt_img_tk), str(img_tk))
            print(f"Text tokens ({len(text_tokens)}): {text_token_str}")
            print(f"Actual tokens ({len(sample.tokens)}): {actual_token_str}")

            raise ValueError(
                f"❗  Token length mismatch: sample.tokens has {actual} tokens, "
                f"expected {expected} "
                f"(text_tokens={text_tokens}, cnt_img_tokens={sum(cnt_img_tokens)}, num_images={num_images})"
            )
    
    if save_suffix is not None:
        # save the sample to disk for debugging. each line is a [decoded(token_id), loss_mask, logprob, token_id]
        # loss_mask / rollout_log_probs only cover the response portion,
        # so pad with '-' for prompt tokens at the start.
        prompt_len = len(sample.tokens) - len(sample.loss_mask)
        with open(f"debug_sample{save_suffix}.txt", "w") as f:
            for i, token_id in enumerate(sample.tokens):
                decoded = tokenizer.decode([token_id])
                if i < prompt_len:
                    f.write(f"{decoded}\t-\t-\t{token_id}\n")
                else:
                    ri = i - prompt_len
                    f.write(f"{decoded}\t{sample.loss_mask[ri]}\t{sample.rollout_log_probs[ri]}\t{token_id}\n")
    
def _append_to_sample(
    sample: Sample,
    new_tokens: list[int],
    logprobs: list[float],
    loss_mask_val: int,
) -> None:
    """Append tokens to sample tracking."""
    if len(new_tokens) != len(logprobs):
        raise ValueError("❗  new_tokens and logprobs must have the same length")
    sample.tokens.extend(new_tokens)
    sample.loss_mask.extend([loss_mask_val] * len(new_tokens))
    if sample.rollout_log_probs is not None:
        sample.rollout_log_probs.extend(logprobs)


def _merge_multimodal_train_inputs(chunks: list[dict | None]) -> dict | None:
    """
    Merge per-turn multimodal_train_inputs by concatenating tensors along dim 0.

    Follows the same pattern as geo3k_vlm_multi_turn/rollout.py.
    Only torch.Tensor values are merged; non-tensor fields are ignored.

    Args:
        chunks: List of per-turn multimodal_train_inputs dicts (or None).

    Returns:
        Merged dict with concatenated tensors, or None if nothing to merge.
    """
    if not chunks:
        return None

    values_by_key: dict[str, list] = {}
    for chunk in chunks:
        if not chunk:
            continue
        for key, val in chunk.items():
            if val is None:
                continue
            values_by_key.setdefault(key, []).append(val)

    merged = {
        key: torch.cat(values, dim=0)
        for key, values in values_by_key.items()
        if all(isinstance(v, torch.Tensor) for v in values)
    }
    return merged or None


def _encode_with_processor(
    processor, tokenizer, text: str, image_data: list[str]
) -> tuple:
    """
    Encode text using the VL processor to correctly expand <|image_pad|> tokens.

    When text contains <|vision_start|><|image_pad|><|vision_end|> placeholders,
    ``tokenizer.encode()`` treats <|image_pad|> as a single token. The processor
    instead expands it to the correct number of tokens based on image dimensions.

    Falls back to ``tokenizer.encode()`` when no processor or no images.

    Args:
        processor: VL processor (e.g. Qwen2VLProcessor) or None
        tokenizer: tokenizer[] instance
        text: formatted text containing image placeholders
        image_data: list of base64 data-URL strings
                    (e.g. "data:image/png;base64,...")

    Returns:
        token_ids: Token IDs with image pad tokens correctly expanded.
        image_grid_thw_list: list of [temporal, height, width] for each image.
        pil_images: list of decoded PIL images (empty list if no images).
        multimodal_train_inputs: processor output minus input_ids/attention_mask,
            suitable for accumulation into sample.multimodal_train_inputs; None
            if no images.
    """
    if not image_data:
        return tokenizer.encode(text, add_special_tokens=False), [], [], None
    
    if processor is None:
        raise ValueError("❗  Processor is required for image encoding.")

    # Decode base64 data-URLs → PIL images
    pil_images = []
    for data_url in image_data:
        # Strip the data-URL prefix ("data:image/...;base64,")
        if "," in data_url:
            b64_str = data_url.split(",", 1)[1]
        else:
            b64_str = data_url
        raw_bytes = base64.b64decode(b64_str)
        pil_images.append(Image.open(io.BytesIO(raw_bytes)).convert("RGB"))

    processor_kwargs = build_processor_kwargs({"images": pil_images})
    processor_output = processor(text=[text], **processor_kwargs)
    token_ids = processor_output["input_ids"][0]
    image_grid_thw = processor_output["image_grid_thw"].tolist()

    # Keep only tensor-valued multimodal inputs so Megatron can move them to GPU
    # and concatenate them across samples without list/numpy edge cases.
    multimodal_train_inputs = {
        k: v for k, v in processor_output.items()
        if k not in ["input_ids", "attention_mask"] and isinstance(v, torch.Tensor)
    } or None


    return token_ids, image_grid_thw, pil_images, multimodal_train_inputs

# ---------------------------------------------------------------------------
# Sample data saving utilities
# ---------------------------------------------------------------------------
def _save_sample(
    samples_dir: str,
    sample_id: str,
    mm_messages: list[dict],
    sample: Sample,
) -> None:
    os.makedirs(samples_dir, exist_ok=True)

    def _to_jsonable(value: Any) -> Any:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, Enum):
            return value.value
        if isinstance(value, dict):
            return {str(k): _to_jsonable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [_to_jsonable(v) for v in value]
        if hasattr(value, "to_dict") and callable(value.to_dict):
            return _to_jsonable(value.to_dict())
        if hasattr(value, "tolist") and callable(value.tolist):
            try:
                return _to_jsonable(value.tolist())
            except Exception:
                pass
        return str(value)

    # Extract per-step images from user messages in mm_messages
    user_images: list[list[str]] = []
    for msg in mm_messages:
        if msg["role"] != "user" or not isinstance(msg.get("content"), list):
            continue
        imgs = [
            item["image_url"]
            for item in msg["content"]
            if isinstance(item, dict) and item.get("type") == "image_url"
        ]
        user_images.append(imgs)

    data = {
        "sample_id": f"{sample_id}",
        "total_steps": sample.metadata.get("total_steps", -1),
        "llm_input_texts": _to_jsonable(sample.prompt),
        "llm_response": _to_jsonable(sample.response),
        "images": _to_jsonable(user_images),
        "status": sample.status.value,
        "terminate_reason": _to_jsonable(sample.metadata.get("terminate_reason", "N/A")),
    }

    path_to_save = os.path.join(samples_dir, f"{sample_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{data['status']}_{data['total_steps']}steps.json")
    with open(path_to_save, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

# ---------------------------------------------------------------------------
# Main generate function
# ---------------------------------------------------------------------------

def _get_rollout_task_timeout_secs(args: Any) -> float | None:
    """Get the per-task rollout timeout from args, if configured."""
    timeout_secs = getattr(args, "rollout_task_timeout_secs", None)
    if timeout_secs is None:
        timeout_secs = getattr(args, "task_timeout_secs", None)
    if timeout_secs is None:
        timeout_secs = os.environ.get("SLIME_BROWSER_ROLLOUT_TASK_TIMEOUT_SECS")
    if timeout_secs is None or timeout_secs == "":
        return None
    return float(timeout_secs)


def _should_remove_browser_sample(sample: Sample) -> bool:
    """Filter out samples caused by infrastructure/system failures from training loss."""
    reason = str((sample.metadata or {}).get("terminate_reason", "")).lower()
    status = sample.status

    timeout_markers = (
        "rollout_task_timeout",
        "sglang /generate timed out",
        "sandbox exec timed out",
        "not healthy within",
    )

    sandbox_markers = (
        "env_step_error",
        "env_server /reset error",
        "env_server /step error",
    )

    if any(marker in reason for marker in timeout_markers):
        return True
    if status == Sample.Status.ABORTED and any(marker in reason for marker in sandbox_markers):
        return True
    return False


def _mark_remove_sample_if_needed(sample: Sample) -> Sample:
    """Mark browser samples that should not contribute to training loss."""
    if _should_remove_browser_sample(sample):
        sample.remove_sample = True
    return sample


def _get_env_exit_timeout_secs(args: Any) -> float:
    """Best-effort cleanup timeout for browser env.exit()."""
    timeout_secs = getattr(args, "browser_env_exit_timeout_secs", None)
    if timeout_secs is None:
        timeout_secs = getattr(args, "env_exit_timeout_secs", 15)
    env_timeout = os.environ.get("SLIME_BROWSER_ENV_EXIT_TIMEOUT_SECS")
    if env_timeout not in (None, ""):
        timeout_secs = env_timeout
    return float(timeout_secs)


async def _safe_exit_env(env: Any, task_id: str, args: Any) -> None:
    """Attempt env cleanup without letting slow cleanup stall the whole rollout."""
    timeout_secs = _get_env_exit_timeout_secs(args)
    cleanup_task = asyncio.create_task(env.exit())
    try:
        await asyncio.wait_for(asyncio.shield(cleanup_task), timeout=timeout_secs)
        logger.info(f"Task {task_id}: environment exited successfully.")
    except asyncio.TimeoutError:
        logger.warning(
            "Task %s: environment exit timed out after %ss; continuing without waiting for cleanup.",
            task_id,
            timeout_secs,
        )
    except asyncio.CancelledError:
        logger.warning(
            "Task %s: cleanup received cancellation; waiting up to %ss before detaching.",
            task_id,
            timeout_secs,
        )
        try:
            await asyncio.wait_for(asyncio.shield(cleanup_task), timeout=timeout_secs)
            logger.info(f"Task {task_id}: environment exited successfully after cancellation.")
        except asyncio.TimeoutError:
            logger.warning(
                "Task %s: environment exit remained pending after cancellation for %ss; detaching cleanup.",
                task_id,
                timeout_secs,
            )
    except Exception as e:
        logger.warning(f"Task {task_id}: environment exit failed: {e}")


async def _generate_trajectory_sample_impl(
    args: Any,
    sample: Sample,
    sampling_params: dict,
) -> Sample:
    """
    Generate a complete agent-environment interaction trajectory as a single Sample.

    The entire multi-turn conversation (system + user observations + LLM responses +
    tool messages) is tracked in one token stream. User/tool tokens get loss_mask=0,
    LLM response tokens get loss_mask=1.

    Args:
        args: Rollout arguments from slime training pipeline
        sample: Sample containing task info
        sampling_params: LLM sampling parameters

    Returns:
        Sample with the complete trajectory token stream.
    """
    # Re-apply asyncio suppression inside the Ray worker in case Ray's
    # logging setup overrode the module-level configuration.
    logging.getLogger("asyncio").setLevel(logging.CRITICAL)

    assert not getattr(args, "partial_rollout", False), \
        "Partial rollout is not supported for gym environment interactions."

    task_id = (sample.metadata or {}).get("task_id", "webvoyager/60")
    sample_id = task_id.replace('/', '_')
    if _should_log_task_start(args):
        logger.info(f"Generating trajectory for sample: {task_id}")

    mm_messages: list[dict] = []  # initialized early so finally block can always access it
    env = None  # initialized early so finally block can always clean up
    try:
        env, adapter, state, url, task_data = await _initialize_resources(args, task_id, sample.metadata)

        tokenizer = state.tokenizer
        processor = state.processor
        sampling_params = sampling_params.copy()

        observation, info = await env.reset()
        tools_info, policy = adapter.parse_env_info(info)
        adapter._tools_info = tools_info
        adapter._tool_parser = ToolParser(tools_info)
        response_mode = _get_browser_response_mode(args)

        # --- Initialize message tracking ---
        all_imgs: list[str] = []          # base64 data-URLs for the rollout engine
        # all_pil_imgs: list = []            # PIL images for sample.multimodal_inputs
        multimodal_train_inputs_buffer: list[dict | None] = []  # for sample.multimodal_train_inputs
        cat_all_msgs: str = ""

        mm_messages.clear()
        mm_messages.append({"role": "system", "content": policy})

        # we don't add generation prompt here, because it will be followed by user message
        prompt_text = adapter.token_handler.apply_chat_template(
            mm_messages,
            tools_info,
            add_generation_prompt=False,
            enable_thinking=response_mode.chat_template_enable_thinking,
        )
        cat_all_msgs += prompt_text

        sys_tokens = tokenizer.encode(prompt_text, add_special_tokens=False)
        sample = Sample(
            index=-1,
            group_index=-1,
            prompt=prompt_text,
            response_length=0,
            tokens=sys_tokens,
            # loss_mask and rollout_log_probs cover ONLY the response portion (post-prompt tokens).
            # The training backend's unpack_sequences slices model log_probs using: log_probs[end - 1 - response_length : end - 1]
            # If response_length == len(tokens), this underflows and returns only 1 element, causing split_with_sizes to fail.
            loss_mask=[],
            rollout_log_probs=[],
            metadata=task_data,
        )
        sample.metadata["image_grid_thw"] = []

        pending_user_msg = adapter._get_latest_user_message(task_data.get("intent", ""), observation, step_id=0)

        # --- Main interaction loop ---
        terminated = False
        truncated = False
        consecutive_parse_failures = 0

        for step in range(getattr(args, "max_steps", 16)):
            # 1. Append the next user-side turn.
            user_msg = pending_user_msg
            pending_user_msg = None
            mm_messages.append(user_msg)

            # 2. Separate text and images from all messages
            user_msg, user_img = adapter._process_multimodal_messages([user_msg])

            # 3. Apply chat template → text input for LLM
            user_text_input = adapter.token_handler.apply_chat_template(
                user_msg,
                add_generation_prompt=True,
                enable_thinking=response_mode.chat_template_enable_thinking,
            )
            user_text_input = ensure_prompt_ends_with_visible_thinking_tag(
                user_text_input,
                response_mode,
            )

            # 4. Compute delta (token ids) from latest user message tokens
            user_delta, image_grid_thw, pil_images, step_mm_train = _encode_with_processor(
                processor, tokenizer, user_text_input, user_img
            )

            # 5. Track tokens from user message
            cat_all_msgs += user_text_input
            all_imgs.extend(user_img)

            # Check max context length to avoid exceeding the LLM's context window.
            max_ctx_len = getattr(args, "rollout_max_context_len", None)
            if max_ctx_len and (len(user_delta) + len(sample.tokens)) > max_ctx_len:
                logger.info(f"Step {step}: total length {len(user_delta)} exceeds max_context_len {max_ctx_len}, truncating.")
                sample.status = Sample.Status.TRUNCATED
                sample.metadata["terminate_reason"] = "rollout_max_context_length_exceeded"
                sample.metadata["total_steps"] = step + 1
                break

            user_logprobs = [0.0] * len(user_delta)
            # append user message tokens with loss_mask=0
            _append_to_sample(sample, user_delta, user_logprobs, loss_mask_val=0)
            sample.metadata["image_grid_thw"].extend(image_grid_thw)
            # Accumulate PIL images and processor tensors for training.
            # all_pil_imgs.extend(pil_images)
            if step_mm_train:
                multimodal_train_inputs_buffer.append(step_mm_train)

            # 6. Run inference ---------------------------------------------
            llm_response, new_tokens, new_logprobs, finish_type = await _run_inference_step(
                url,
                cat_all_msgs,
                sampling_params,
                all_imgs,
                timeout_secs=getattr(args, "inference_step_timeout_secs", None),
            )
            # --------------------------------------------------------------
            if _should_sample_llm_output(args):
                logger.info(f"Task {task_id} step {step} llm_response={llm_response!r}")

            # 7. Append LLM response tokens (loss_mask=1)
            cat_all_msgs += llm_response
            _append_to_sample(sample, new_tokens, new_logprobs, loss_mask_val=1)

            llm_response_wo_end = adapter.preprocess_response(llm_response) # remove "<|im_end|>"
            assistant_history_text = _restore_prefilled_thinking_tag_for_history(
                llm_response_wo_end,
                user_text_input,
                response_mode,
            )
            mm_messages.append({"role": "assistant", "content": assistant_history_text})

            # 8. Check if generation should stop (from the SGLang side)
            if finish_type == "length":
                sample.status = Sample.Status.TRUNCATED
                sample.metadata["terminate_reason"] = "generation_length_limit"
                sample.metadata["total_steps"] = step + 1
                break
            if finish_type == "abort":
                sample.status = Sample.Status.ABORTED
                sample.metadata["terminate_reason"] = "generation_abort"
                sample.metadata["total_steps"] = step + 1
                break

            # 9. Parse LLM response to execute action
            parsed = adapter.tool_parser.parse(llm_response_wo_end)
            if parsed.success and len(parsed.calls) > 0:
                actions = adapter.actions_from_parsed(parsed.calls)

                try:
                    observation, _, terminated, _, info = await env.step(actions)
                except Exception as step_err:
                    # logger.warning(f"Task {task_id} step {step}: env.step() failed, marking as ABORTED: {step_err}")
                    logger.warning(
                        "Task %s step %d env.step failed | llm_response=%r | parsed_calls=%s | actions=%s",
                        task_id,
                        step,
                        llm_response_wo_end[:1000],
                        parsed.calls,
                        actions,
                    )
                    logger.warning(f"❗  Environment step failed: {step_err}")
                    sample.status = Sample.Status.ABORTED
                    sample.metadata["terminate_reason"] = "env_step_error"
                    sample.metadata["total_steps"] = step + 1
                    break

                if not terminated:
                    pending_user_msg = adapter._convert_tool_responses_to_msgs(
                        info["tool_responses"],
                        observation=observation,
                        include_tool_response=_include_tool_response_in_rollout(args),
                    )[0]
                consecutive_parse_failures = 0  # reset on success

                # Check if the agent called done
                if terminated:
                    sample.status = Sample.Status.COMPLETED
                    sample.metadata["terminate_reason"] = "task_completed"
                    sample.metadata["total_steps"] = step + 1
                    break

            else:
                consecutive_parse_failures += 1
                if consecutive_parse_failures >= getattr(args, "max_consecutive_parse_failures", 3):
                    logger.warning(f"Task {task_id} step {step}: {consecutive_parse_failures} consecutive parse failures, aborting.")
                    sample.status = Sample.Status.FAILED
                    sample.metadata["terminate_reason"] = "format_error_failed"
                    sample.metadata["total_steps"] = step + 1
                    break
                pending_user_msg = adapter._convert_tool_responses_to_msgs(
                    [{"tool_response": "Failed to parse tool calls from the response. Please try again with correct format."}],
                    observation=observation,
                    include_tool_response=_include_tool_response_in_rollout(args),
                )[0]

        # --- Finalization ---
        if not terminated:
            if sample.status is None or sample.status == Sample.Status.PENDING:
                sample.status = Sample.Status.FAILED
                sample.metadata["terminate_reason"] = "max_steps_exhausted"
                sample.metadata["total_steps"] = getattr(args, "max_steps", 16)
        if sample.status is None:
            raise ValueError("❗  Sample status is None")

        sample.response = cat_all_msgs[len(sample.prompt):]
        sample.response_length = len(sample.loss_mask)
        sample.multimodal_inputs = {"images": all_imgs}
        # sample.multimodal_inputs = {"images": all_pil_imgs}
        # Snapshot the trajectory for the eval judges (see the turn-level path):
        # online_mind2web / webvoyager / deepshop read metadata["messages"] and
        # metadata["full_image_list"].
        sample.metadata["messages"] = list(mm_messages)
        sample.metadata["full_image_list"] = list(all_imgs)
        sample.multimodal_train_inputs = _merge_multimodal_train_inputs(multimodal_train_inputs_buffer)
        _ensure_correct_sample(sample, tokenizer) # for debugging: verify final token stream is correct after merging multimodal inputs

        # log the final sample for debugging
        # mm_keys = list(sample.multimodal_train_inputs.keys()) if sample.multimodal_train_inputs else None
        logger.info(
            f"Task {task_id} done | status={sample.status} | steps={step} | "
            f"tokens={len(sample.tokens)} | " # mm_train_keys={mm_keys} | "
            f"response={sample.response[:200]!r}..."
        )

        return _mark_remove_sample_if_needed(sample)

    except Exception as e:
        _append_host_to_blacklist_if_needed(str(e))
        logger.opt(depth=0).warning("Task {}: generate_trajectory_sample failed: {}", task_id, e, exc_info=True)
        # raise ValueError(f"❗  Generation failed: {e}") from e
        sample.status = Sample.Status.ABORTED
        sample.metadata["terminate_reason"] = f"generation_error: {e}"
        return _mark_remove_sample_if_needed(sample)
    finally:
        try:
            path_to_save_generated_samples = getattr(args, "path_to_save_generated_samples", "")
            if path_to_save_generated_samples and mm_messages:
                _save_sample(
                    samples_dir=os.path.join(path_to_save_generated_samples, "trajectory", _EXPT_ID),
                    sample_id=sample_id,
                    mm_messages=mm_messages,
                    sample=sample,
                )
            debug_trace_dir = _get_debug_trace_dir(args, "rollout", "trajectory")
            save_debug_trace, trace_id = _get_or_create_debug_trace_info(args, sample)
            if save_debug_trace and trace_id and debug_trace_dir and mm_messages:
                _save_sample(
                    samples_dir=debug_trace_dir,
                    sample_id=trace_id,
                    mm_messages=mm_messages,
                    sample=sample,
                )
        except Exception as save_err:
            logger.warning(f"Task {task_id}: failed to save sample data: {save_err}")

        if env is not None:
            await _safe_exit_env(env, task_id, args)


async def generate_trajectory_sample(
    args: Any,
    sample: Sample,
    sampling_params: dict,
) -> Sample:
    """Generate one browser trajectory with an optional per-task timeout."""
    task_id = (sample.metadata or {}).get("task_id", "webvoyager/60")
    async with _browser_rollout_submission_slot(args, task_id):
        timeout_secs = _get_rollout_task_timeout_secs(args)
        try:
            if timeout_secs is not None and timeout_secs > 0:
                return await asyncio.wait_for(
                    _generate_trajectory_sample_impl(args, sample, sampling_params),
                    timeout=timeout_secs,
                )
            return await _generate_trajectory_sample_impl(args, sample, sampling_params)
        except asyncio.TimeoutError:
            logger.warning(f"Task {task_id}: generate_trajectory_sample timed out after {timeout_secs}s")
            sample.status = Sample.Status.ABORTED
            sample.metadata = sample.metadata or {}
            sample.metadata["terminate_reason"] = f"generation_error: rollout_task_timeout after {timeout_secs}s"
            sample.metadata["total_steps"] = sample.metadata.get("total_steps", 0)
            return _mark_remove_sample_if_needed(sample)


async def _generate_turn_sample_impl(
    args: Any,
    sample: Sample,
    sampling_params: dict,
) -> list[Sample]:
    """
    Generate a multi-turn agent-environment interaction, returning one Sample per turn.

    Each turn sample contains its own prompt (full context up to that turn) and
    response (the LLM output for that turn). Only LLM response tokens have loss_mask=1.

    Args:
        args: Rollout arguments from slime training pipeline
        sample: Sample containing task info
        sampling_params: LLM sampling parameters

    Returns:
        List of per-turn Sample objects.
    """
    logging.getLogger("asyncio").setLevel(logging.CRITICAL)

    assert not getattr(args, "partial_rollout", False), \
        "Partial rollout is not supported for gym environment interactions."

    context_num_screenshots = getattr(args, "context_num_screenshots", 1)
    turn_history_reasoning_mode = getattr(args, "turn_history_reasoning_mode", "full")

    task_id = (sample.metadata or {}).get("task_id", "webvoyager/60")
    sample_id = task_id.replace('/', '_')
    if _should_log_task_start(args):
        logger.info(f"Generating turn data for sample: {task_id}")

    mm_messages: list[dict] = []
    turn_samples: list[Sample] = []
    env = None  # initialized early so finally block can always clean up
    try:
        env, adapter, state, url, task_data = await _initialize_resources(args, task_id, sample.metadata)

        tokenizer = state.tokenizer
        processor = state.processor
        sampling_params = sampling_params.copy()

        observation, info = await env.reset()
        tools_info, policy = adapter.parse_env_info(info)
        adapter._tools_info = tools_info
        adapter._tool_parser = ToolParser(tools_info)
        response_mode = _get_browser_response_mode(args)

        # --- Initialize message tracking ---
        mm_messages.clear()
        mm_messages.append({"role": "system", "content": policy})

        _SCREENSHOT_PLACEHOLDER = "screenshot:\n<|vision_start|><|image_pad|><|vision_end|>"

        pending_user_msg = adapter._get_latest_user_message(task_data.get("intent", ""), observation, step_id=0)

        # --- Main interaction loop ---
        terminated = False
        truncated = False
        consecutive_parse_failures = 0

        for step in range(getattr(args, "max_steps", 20)):
            # 1. Append the next user-side turn.
            user_msg = pending_user_msg
            pending_user_msg = None
            mm_messages.append(user_msg)

            # 2. Filter mm_messages to keep only the last K screenshots.
            #    Identify user messages that contain screenshots, then for those outside the last-K window
            #    strip the placeholder text and image_url entry so the downstream pipeline sees only K images.
            user_img_indices = [
                i for i, msg in enumerate(mm_messages)
                if msg["role"] == "user"
                and isinstance(msg.get("content"), list)
                and any(item.get("type") == "image_url" for item in msg["content"])
            ]
            indices_to_keep = set(user_img_indices[-context_num_screenshots:])

            mm_messages_filtered = []
            for i, msg in enumerate(mm_messages):
                if i in user_img_indices and i not in indices_to_keep:
                    # Strip screenshot from this older user message.
                    new_content = []
                    for item in msg["content"]:
                        if item.get("type") == "text":
                            cleaned = item["text"].replace(_SCREENSHOT_PLACEHOLDER, "")
                            new_content.append({"type": "text", "text": cleaned})
                        # Drop image_url items entirely.
                    mm_messages_filtered.append({"role": msg["role"], "content": new_content})
                else:
                    mm_messages_filtered.append(msg)

            mm_messages_filtered = _compress_turn_history_messages(
                mm_messages_filtered,
                turn_history_reasoning_mode,
                args,
            )

            # 3. Separate text and images from filtered messages
            text_msg, img_list = adapter._process_multimodal_messages(mm_messages_filtered)

            # 4. Apply chat template → text input for LLM
            input_text = adapter.token_handler.apply_chat_template(
                text_msg,
                tools_info,
                add_generation_prompt=True,
                enable_thinking=response_mode.chat_template_enable_thinking,
            )
            input_text = _restore_assistant_history_blocks_in_prompt(
                input_text,
                mm_messages_filtered,
            )
            input_text = ensure_prompt_ends_with_visible_thinking_tag(
                input_text,
                response_mode,
            )

            # 5. Compute input tokens with image_pad correctly expanded
            input_tokens, image_grid_thw, pil_images, mm_train = _encode_with_processor(
                processor, tokenizer, input_text, img_list
            )

            # Check max context length
            max_ctx_len = getattr(args, "rollout_max_context_len", None)
            if max_ctx_len and (len(input_tokens) > max_ctx_len):
                logger.info(f"Step {step}: turn input length {len(input_tokens)} exceeds max_context_len {max_ctx_len}, stopping.")
                if turn_samples:
                    turn_samples[-1].metadata["is_last_turn"] = True
                    for ts in turn_samples:
                        ts.status = Sample.Status.TRUNCATED
                        ts.metadata["terminate_reason"] = "generation_length_limit"
                        ts.metadata["total_steps"] = step + 1
                break

            # 6. Create turn sample
            turn_sample = Sample(
                index=_make_turn_sample_index(sample.index, step),
                # Preserve the parent trajectory's prompt-group identity for
                # downstream reward normalization / training grouping. The
                # per-turn order stays in metadata["turn_index"].
                group_index=sample.group_index,
                prompt=input_text,
                response_length=0,
                tokens=input_tokens,
                loss_mask=[],
                rollout_log_probs=[],
                metadata=dict(task_data),
            )
            turn_sample.metadata["trajectory_id"] = sample.index
            turn_sample.metadata["turn_sample_id"] = turn_sample.index
            turn_sample.metadata["turn_index"] = step
            turn_sample.metadata["is_last_turn"] = False
            turn_sample.metadata["image_grid_thw"] = image_grid_thw
            turn_sample.multimodal_inputs = {"images": img_list}
            turn_sample.multimodal_train_inputs = mm_train

            # 7. Run inference ---------------------------------------------
            llm_response, new_tokens, new_logprobs, finish_type = await _run_inference_step(
                url,
                input_text,
                sampling_params,
                img_list,
                timeout_secs=getattr(args, "inference_step_timeout_secs", None),
            )
            # --------------------------------------------------------------
            if _should_sample_llm_output(args):
                logger.info(f"Task {task_id} step {step} llm_response={llm_response[:1000]!r}")

            # 8. Append LLM response tokens (loss_mask=1)
            _append_to_sample(turn_sample, new_tokens, new_logprobs, loss_mask_val=1)

            turn_sample.response = llm_response
            turn_sample.response_length = len(new_tokens)
            _ensure_correct_sample(turn_sample, tokenizer) # for debugging: verify final token stream is correct after merging multimodal inputs

            turn_samples.append(turn_sample)

            llm_response_wo_end = adapter.preprocess_response(llm_response) # remove "<|im_end|>"
            assistant_history_text = _restore_prefilled_thinking_tag_for_history(
                llm_response_wo_end,
                input_text,
                response_mode,
            )
            mm_messages.append({"role": "assistant", "content": assistant_history_text})

            # Snapshot the full conversation + screenshots onto the turn sample so
            # the eval judges (online_mind2web / webvoyager / deepshop) can read the
            # trajectory via metadata["messages"] / ["full_image_list"]. Whichever
            # turn ends up last then carries the complete trajectory.
            turn_sample.metadata["messages"] = list(mm_messages)
            turn_sample.metadata["full_image_list"] = list(img_list)

            # 9. Check if generation should stop (from the SGLang side)
            if finish_type == "length":
                turn_samples[-1].metadata["is_last_turn"] = True
                for ts in turn_samples:
                    ts.status = Sample.Status.TRUNCATED
                    ts.metadata["terminate_reason"] = "generation_length_limit"
                    ts.metadata["total_steps"] = step + 1
                break
            if finish_type == "abort":
                turn_samples[-1].metadata["is_last_turn"] = True
                for ts in turn_samples:
                    ts.status = Sample.Status.ABORTED
                    ts.metadata["terminate_reason"] = "generation_abort"
                    ts.metadata["total_steps"] = step + 1
                break

            # 10. Parse LLM_response to execute action
            parsed = adapter.tool_parser.parse(llm_response_wo_end)
            if parsed.success and len(parsed.calls) > 0:
                actions = adapter.actions_from_parsed(parsed.calls)

                try:
                    observation, _, terminated, _, info = await env.step(actions)
                except Exception as step_err:
                    logger.warning(f"Task {task_id} step {step}: env.step() failed, marking as ABORTED: {step_err}")
                    # raise ValueError(f"❗  Environment step failed: {step_err}") from step_err
                    turn_samples[-1].metadata["is_last_turn"] = True
                    for ts in turn_samples:
                        ts.status = Sample.Status.ABORTED
                        ts.metadata["terminate_reason"] = "env_step_error"
                        ts.metadata["total_steps"] = step + 1
                    break

                turn_samples[-1].metadata["step_tool_responses"] = [
                    {
                        "tool_name": item.get("tool_name", ""),
                        "tool_response": adapter._truncate_tool_response(item.get("tool_response", "")),
                    }
                    for item in info.get("tool_responses", []) or []
                    if isinstance(item, dict)
                ]
                if not terminated:
                    pending_user_msg = adapter._convert_tool_responses_to_msgs(
                        info["tool_responses"],
                        observation=observation,
                        include_tool_response=_include_tool_response_in_rollout(args),
                    )[0]
                consecutive_parse_failures = 0  # reset on success
                
                # Check if the agent called done
                if terminated:
                    turn_samples[-1].metadata["is_last_turn"] = True
                    for ts in turn_samples:
                        ts.status = Sample.Status.COMPLETED
                        ts.metadata["terminate_reason"] = "task_completed"
                        ts.metadata["total_steps"] = step + 1
                    break

            else:
                consecutive_parse_failures += 1
                if consecutive_parse_failures >= getattr(args, "max_consecutive_parse_failures", 3):
                    logger.warning(f"Task {task_id} step {step}: {consecutive_parse_failures} consecutive parse failures, aborting")
                    turn_samples[-1].metadata["is_last_turn"] = True
                    for ts in turn_samples:
                        ts.status = Sample.Status.FAILED
                        ts.metadata["terminate_reason"] = "format_error_failed"
                        ts.metadata["total_steps"] = step + 1
                    break
                pending_user_msg = adapter._convert_tool_responses_to_msgs(
                    [{"tool_response": "Failed to parse tool calls from the response. Please try again with correct format."}],
                    observation=observation,
                    include_tool_response=_include_tool_response_in_rollout(args),
                )[0]
        
        # --- Finalization ---
        if not terminated:
            if turn_samples:
                turn_samples[-1].metadata["is_last_turn"] = True
            for ts in turn_samples:
                if ts.status is None or ts.status == Sample.Status.PENDING:
                    ts.status = Sample.Status.FAILED
                    ts.metadata["terminate_reason"] = "max_steps_exhausted"
                    ts.metadata["total_steps"] = getattr(args, "max_steps", 16)
        
        num_turns_in_trajectory = len(turn_samples)
        for ts in turn_samples:
            ts.metadata["num_turns_in_trajectory"] = num_turns_in_trajectory

        return [_mark_remove_sample_if_needed(ts) for ts in turn_samples]

    except Exception as e:
        _append_host_to_blacklist_if_needed(str(e))
        logger.opt(depth=0).warning("Task {}: generate_turn_sample failed: {}", task_id, e, exc_info=True)
        # raise ValueError(f"❗  Generation failed: {e}") from e
        # Return a failed sample instead of crashing the whole batch.
        if turn_samples:
            turn_samples[-1].metadata["is_last_turn"] = True
            # set all previous turns to ABORTED to avoid introducing noisy tokens, and mark the last turn as the final one.
            for ts in turn_samples:
                ts.status = Sample.Status.ABORTED
                ts.metadata["terminate_reason"] = f"generation_error: {e}"
        else:
            # No turns were generated at all (e.g., sandbox 404 before first turn).
            # Return the original sample marked as ABORTED so downstream code
            # doesn't crash on an empty list.
            sample.status = Sample.Status.ABORTED
            sample.metadata = sample.metadata or {}
            sample.metadata["is_last_turn"] = True
            sample.metadata["terminate_reason"] = f"generation_error: {e}"
            sample.metadata["total_steps"] = 0
            turn_samples = [sample]
        num_turns_in_trajectory = len(turn_samples)
        for ts in turn_samples:
            ts.metadata["num_turns_in_trajectory"] = num_turns_in_trajectory

        return [_mark_remove_sample_if_needed(ts) for ts in turn_samples]
    finally:
        try:
            path_to_save_generated_samples = getattr(args, "path_to_save_generated_samples", "")
            if path_to_save_generated_samples and mm_messages:
                _save_sample(
                    samples_dir=os.path.join(path_to_save_generated_samples, "turn", _EXPT_ID),
                    sample_id=sample_id,
                    mm_messages=mm_messages,
                    sample=turn_samples[-1],
                )
            debug_trace_dir = _get_debug_trace_dir(args, "rollout", "turn")
            last_turn_sample = turn_samples[-1] if turn_samples else sample
            save_debug_trace, trace_id = _get_or_create_debug_trace_info(args, last_turn_sample)
            if save_debug_trace and trace_id and debug_trace_dir and mm_messages and turn_samples:
                _save_sample(
                    samples_dir=debug_trace_dir,
                    sample_id=trace_id,
                    mm_messages=mm_messages,
                    sample=turn_samples[-1],
                )
        except Exception as save_err:
            logger.warning(f"Task {task_id}: failed to save sample data: {save_err}")

        if env is not None:
            await _safe_exit_env(env, task_id, args)


async def generate_turn_sample(
    args: Any,
    sample: Sample,
    sampling_params: dict,
) -> list[Sample]:
    """Generate browser turn samples with an optional per-task timeout."""
    task_id = (sample.metadata or {}).get("task_id", "webvoyager/60")
    async with _browser_rollout_submission_slot(args, task_id):
        timeout_secs = _get_rollout_task_timeout_secs(args)
        try:
            if timeout_secs is not None and timeout_secs > 0:
                return await asyncio.wait_for(
                    _generate_turn_sample_impl(args, sample, sampling_params),
                    timeout=timeout_secs,
                )
            return await _generate_turn_sample_impl(args, sample, sampling_params)
        except asyncio.TimeoutError:
            logger.warning(f"Task {task_id}: generate_turn_sample timed out after {timeout_secs}s")
            sample.status = Sample.Status.ABORTED
            sample.metadata = sample.metadata or {}
            sample.metadata["is_last_turn"] = True
            sample.metadata["terminate_reason"] = f"generation_error: rollout_task_timeout after {timeout_secs}s"
            sample.metadata["total_steps"] = sample.metadata.get("total_steps", 0)
            return [_mark_remove_sample_if_needed(sample)]
