"""
Stage 1 of the OpenWebRL SFT data pipeline: convert the released per-turn
trajectories into a **model-agnostic, OpenAI chat-completion format**.

The released ``OpenWebRL_SFT_trajectories.jsonl`` bakes a model-specific
tool-calling format (Qwen3-VL / Hermes JSON) into pre-rendered prompt/response
text, which couples the data to a single base model. This script instead emits
the standard OpenAI shape used by agentic-SFT datasets::

    {
      "tools":    [ {"type": "function", "function": {...}}, ... ],
      "messages": [
        {"role": "system",    "content": "<policy>"},
        {"role": "user",      "content": [ {"type": "text", ...},
                                            {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}},
                                            {"type": "text", ...} ]},
        {"role": "assistant", "content": "<think>...</think>",
                              "tool_calls": [ {"id": "...", "type": "function",
                                               "function": {"name": "click",
                                                            "arguments": {"point_2d": [x, y]}}} ]},
        {"role": "tool",      "tool_call_id": "...", "name": "click",
                              "content": [ {"type": "text", ...}, {"type": "image_url", ...} ]},
        ...
      ],
      "metadata": { "task_id": ..., "rollout_idx": ..., ... }
    }

One record is emitted **per full episode** (grouped by ``(task_id,
rollout_idx)``), with **every screenshot kept** inline as a base64
``image_url``. Each released row stores only its current screenshot, so the
episode's screenshots are reassembled by ordering the rows by ``turn_index`` and
taking each row's single ``metadata.images[0]``; the longest (max-turn) row's
``metadata.messages`` provides the full message skeleton (its screenshot
placeholders line up 1:1 with the ordered per-turn screenshots).

This file is model-agnostic. Rendering it into trainer-specific data (and
choosing per-turn vs trajectory granularity) is Stage 2,
``prepare_openai_for_llamafactory.py``.

Usage::

    python sft/convert_to_openai_messages.py \
        --input-path /path/to/OpenWebRL_SFT_trajectories.jsonl
    # default output: <input_dir>/openwebrl_sft_openai.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("convert_to_openai_messages")

_VISION_PLACEHOLDER = "<|vision_start|><|image_pad|><|vision_end|>"

# Matches the real `<tools>\n...\n</tools>` listing (newline right after the tag),
# not the literal `<tools></tools>` mentioned in the QWEN tool-prompt prose.
_TOOLS_BLOCK_RE = re.compile(r"<tools>\n(.*?)\n</tools>", re.DOTALL)
# A single JSON-style tool call (same shape the OpenWebRL runtime parser uses).
_JSON_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)

# Episode-level metadata fields worth carrying forward (drop bulky per-turn
# fields like the embedded `images`/`messages`).
_EPISODE_META_FIELDS = (
    "task_id",
    "benchmark_name",
    "domain",
    "subdomain",
    "intent",
    "start_url",
    "difficulty",
    "total_steps",
    "terminate_reason",
    "is_last_turn",
    "evaluator_reference",
    "reward",
)


def load_jsonl(path: str, max_rows: int | None = None) -> list[dict]:
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
            if max_rows is not None and len(rows) >= max_rows:
                break
    return rows


def group_episodes(rows: list[dict]) -> dict[tuple, list[dict]]:
    """Group per-turn rows into episodes keyed by ``(task_id, rollout_idx)``.

    Each episode's rows are returned sorted by ``turn_index`` so screenshots and
    the message skeleton line up chronologically.
    """
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        meta = row.get("metadata") or {}
        key = (meta.get("task_id"), row.get("rollout_idx"))
        groups[key].append(row)
    for key, ep_rows in groups.items():
        ep_rows.sort(key=lambda r: (r.get("metadata") or {}).get("turn_index", 0))
    return groups


def parse_tools(prompt: str) -> list[dict]:
    """Extract the OpenAI tool definitions from a rendered prompt's <tools> block."""
    match = _TOOLS_BLOCK_RE.search(prompt)
    if not match:
        return []
    tools = []
    for line in match.group(1).split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            tools.append(json.loads(line))
        except json.JSONDecodeError:
            logger.warning("Skipping unparseable tool definition line")
    return tools


def split_tool_calls(content: str, call_id_prefix: str) -> tuple[str, list[dict], int]:
    """Split assistant text into (reasoning_text, structured_tool_calls, n_unparsed).

    Returns the content with all ``<tool_call>`` blocks removed (the
    ``<think>...`` reasoning), the structured OpenAI ``tool_calls`` list, and a
    count of tool calls whose JSON could not be parsed (kept inline in the text,
    mirroring the OpenWebRL runtime's tolerant fallback).
    """
    tool_calls: list[dict] = []
    n_unparsed = 0
    idx = 0

    def _repl(match: re.Match) -> str:
        nonlocal idx, n_unparsed
        raw = match.group(1)
        try:
            call = json.loads(raw)
            name = call["name"]
            arguments = call["arguments"]
        except (json.JSONDecodeError, KeyError, TypeError):
            n_unparsed += 1
            return match.group(0)  # leave the bad block in the reasoning text
        tool_calls.append(
            {
                "id": f"{call_id_prefix}_{idx}",
                "type": "function",
                "function": {"name": name, "arguments": arguments},
            }
        )
        idx += 1
        return ""

    text = _JSON_TOOL_CALL_RE.sub(_repl, content)
    # Collapse the whitespace the removed tool-call blocks leave behind.
    text = text.rstrip()
    return text, tool_calls, n_unparsed


def content_with_images(text: str, screenshots: list[str], cursor: list[int]) -> object:
    """Turn a string with vision placeholders into OpenAI multimodal content.

    Each ``_VISION_PLACEHOLDER`` is replaced, in document order, by the next
    screenshot from ``screenshots`` (tracked via the single-element ``cursor``
    list so the position is shared across all messages of an episode). Returns a
    plain string when there is no image, else a list of text/image_url parts.
    """
    if _VISION_PLACEHOLDER not in text:
        return text

    parts: list[dict] = []
    segments = text.split(_VISION_PLACEHOLDER)
    for i, segment in enumerate(segments):
        if segment:
            parts.append({"type": "text", "text": segment})
        if i < len(segments) - 1:
            url = screenshots[cursor[0]]
            cursor[0] += 1
            parts.append({"type": "image_url", "image_url": {"url": url}})
    return parts


def convert_episode(key: tuple, ep_rows: list[dict]) -> tuple[dict | None, dict]:
    """Convert one episode's per-turn rows into a single OpenAI-format record.

    Returns ``(record, stats)``; ``record`` is None if the episode is malformed
    (e.g. screenshot/placeholder mismatch).
    """
    stats = {"unparsed_tool_calls": 0, "skipped_episodes": 0}
    task_id, rollout_idx = key
    max_row = ep_rows[-1]
    skeleton = (max_row.get("metadata") or {}).get("messages") or []
    if not skeleton or skeleton[0].get("role") != "system":
        logger.warning(f"Episode {key}: missing system skeleton; skipping")
        stats["skipped_episodes"] = 1
        return None, stats

    # Chronological list of per-turn screenshots (one per row / turn).
    screenshots: list[str] = []
    for row in ep_rows:
        imgs = (row.get("metadata") or {}).get("images") or []
        if len(imgs) != 1:
            logger.warning(f"Episode {key}: a turn does not have exactly 1 image; skipping")
            stats["skipped_episodes"] = 1
            return None, stats
        screenshots.append(imgs[0])

    n_placeholders = sum(
        (m.get("content") or "").count(_VISION_PLACEHOLDER)
        for m in skeleton
        if isinstance(m.get("content"), str)
    )
    if n_placeholders != len(screenshots):
        logger.warning(
            f"Episode {key}: {n_placeholders} placeholders != {len(screenshots)} screenshots; skipping"
        )
        stats["skipped_episodes"] = 1
        return None, stats

    tools = parse_tools(max_row.get("prompt", ""))
    id_stub = f"call_{task_id}_{rollout_idx}"

    out_messages: list[dict] = []
    cursor = [0]  # shared screenshot position across the episode
    turn_idx = -1  # increments on each assistant message
    pending_calls: list[dict] = []  # tool_call ids awaiting their tool responses
    tool_slot = 0

    for msg in skeleton:
        role = msg.get("role")
        content = msg.get("content")

        if role == "system":
            out_messages.append({"role": "system", "content": content})
        elif role == "user":
            out_messages.append(
                {"role": "user", "content": content_with_images(content, screenshots, cursor)}
            )
        elif role == "assistant":
            turn_idx += 1
            text, tool_calls, n_unparsed = split_tool_calls(
                content, f"{id_stub}_turn{turn_idx}"
            )
            stats["unparsed_tool_calls"] += n_unparsed
            out_msg: dict = {"role": "assistant", "content": text}
            if tool_calls:
                out_msg["tool_calls"] = tool_calls
            out_messages.append(out_msg)
            pending_calls = tool_calls
            tool_slot = 0
        elif role == "tool":
            tool_msg: dict = {
                "role": "tool",
                "content": content_with_images(content, screenshots, cursor),
            }
            if msg.get("name"):
                tool_msg["name"] = msg["name"]
            # Link to the matching call from the preceding assistant turn (in order).
            if tool_slot < len(pending_calls):
                tool_msg["tool_call_id"] = pending_calls[tool_slot]["id"]
            tool_slot += 1
            out_messages.append(tool_msg)
        else:
            logger.warning(f"Episode {key}: unexpected role {role!r}; skipping message")

    if cursor[0] != len(screenshots):
        logger.warning(
            f"Episode {key}: consumed {cursor[0]}/{len(screenshots)} screenshots; skipping"
        )
        stats["skipped_episodes"] = 1
        return None, stats

    src_meta = max_row.get("metadata") or {}
    metadata = {"task_id": task_id, "rollout_idx": rollout_idx, "n_turns": len(ep_rows)}
    for field in _EPISODE_META_FIELDS:
        if field in src_meta:
            metadata[field] = src_meta[field]

    record = {"tools": tools, "messages": out_messages, "metadata": metadata}
    return record, stats


def main():
    parser = argparse.ArgumentParser(
        description="Convert OpenWebRL per-turn trajectories into canonical OpenAI-format episodes."
    )
    parser.add_argument("--input-path", required=True, help="Path to OpenWebRL_SFT_trajectories.jsonl")
    parser.add_argument(
        "--output-path",
        default=None,
        help="Output JSONL path (default: <input_dir>/openwebrl_sft_openai.jsonl)",
    )
    parser.add_argument("--max-rows", type=int, default=None, help="Cap input rows (for testing).")
    parser.add_argument("--max-episodes", type=int, default=None, help="Cap output episodes (for testing).")
    args = parser.parse_args()

    input_path = Path(args.input_path)
    output_path = Path(args.output_path) if args.output_path else input_path.parent / "openwebrl_sft_openai.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading {input_path}")
    rows = load_jsonl(str(input_path), max_rows=args.max_rows)
    logger.info(f"Loaded {len(rows)} rows")

    episodes = group_episodes(rows)
    logger.info(f"Grouped into {len(episodes)} episodes")

    totals = {"episodes": 0, "skipped_episodes": 0, "unparsed_tool_calls": 0}
    with open(output_path, "w", encoding="utf-8") as out_fp:
        for i, (key, ep_rows) in enumerate(tqdm(episodes.items(), desc="Converting episodes")):
            if args.max_episodes is not None and totals["episodes"] >= args.max_episodes:
                break
            record, stats = convert_episode(key, ep_rows)
            totals["skipped_episodes"] += stats["skipped_episodes"]
            totals["unparsed_tool_calls"] += stats["unparsed_tool_calls"]
            if record is None:
                continue
            out_fp.write(json.dumps(record, ensure_ascii=False) + "\n")
            totals["episodes"] += 1

    logger.info(
        f"Wrote {totals['episodes']} episodes to {output_path} "
        f"(skipped {totals['skipped_episodes']})"
    )
    if totals["unparsed_tool_calls"]:
        logger.warning(
            f"{totals['unparsed_tool_calls']} tool call(s) kept inline in reasoning "
            f"text because their JSON could not be parsed (e.g. illegal backslash in done.response)."
        )


if __name__ == "__main__":
    main()
