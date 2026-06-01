"""
Convert OpenWebRL SFT trajectory JSONL files into LLaMAFactory-ready
multimodal SFT JSONL + image files.

Each turn sample is converted into a messages list with image file paths,
suitable for LLaMAFactory's ShareGPT-style multimodal dataset format.

Screenshots are saved as PNG files next to the output JSONL and referenced
by path in the output JSONL (avoids OOM from loading all base64 images at once).

The released OpenWebRL SFT trajectories are already the curated experiment
data, so this script does not filter, resample, or deduplicate examples. It
only converts the JSONL rows into LLaMAFactory-compatible format.

When ``--input-path`` is a glob that matches multiple files, the script
streams per shard: load one file → convert → append to output → discard →
next file. Peak memory stays at ~one shard instead of the whole sum.

Usage:
    # Output defaults to <input_dir>/sft_data.jsonl
    python sft/prepare_llamafactory_sft_data.py \
        --input-path /path/to/OpenWebRL_SFT_trajectories.jsonl

    # Explicit output path
    python sft/prepare_llamafactory_sft_data.py \
        --input-path /path/to/OpenWebRL_SFT_trajectories.jsonl \
        --output-path /path/to/sft_data.jsonl \
        --max-rows 100
"""

from __future__ import annotations

import argparse
import base64
import glob
import json
import logging
import os
import sys
from pathlib import Path

from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("prepare_llamafactory_sft_data")

# Chat template markers (Qwen format)
_IM_START = "<|im_start|>"
_IM_END = "<|im_end|>"
_VISION_PLACEHOLDER = "<|vision_start|><|image_pad|><|vision_end|>"
_IMAGE_PLACEHOLDER = "<image>"


def resolve_input_paths(path: str) -> list[str]:
    """Expand a glob or return a single-file list. Raises if nothing matches.

    Glob expansion is separated from file loading so ``main()`` can iterate
    over shards and stream them one at a time when multiple files match.
    """
    paths = sorted(glob.glob(path)) if ("*" in path or "?" in path) else [path]
    if not paths:
        raise FileNotFoundError(f"No files matched: {path}")
    return paths


def load_jsonl(path: str, max_rows: int | None = None) -> list[dict]:
    """Load rows from a single JSONL file (no glob handling).

    Glob expansion lives in ``resolve_input_paths``; this function intentionally
    only reads one file so callers can stream shard-by-shard.
    """
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
                return rows
    return rows


def parse_templated_text_to_messages(text: str) -> list[dict]:
    """Parse a Qwen chat-templated string back into a list of message dicts.

    Splits on ``<|im_start|>`` boundaries, extracts the role from the first
    line and the content from the rest (stripping ``<|im_end|>``).

    Returns:
        List of ``{"role": str, "content": str}`` dicts.
    """
    messages = []
    # Split on <|im_start|>, skip the first empty segment
    segments = text.split(_IM_START)
    for segment in segments:
        segment = segment.strip()
        if not segment:
            continue

        # Remove trailing <|im_end|> and anything after it (e.g. newlines)
        if _IM_END in segment:
            segment = segment[: segment.rfind(_IM_END)]

        # First line is the role, rest is content
        newline_idx = segment.find("\n")
        if newline_idx == -1:
            role = segment.strip()
            content = ""
        else:
            role = segment[:newline_idx].strip()
            content = segment[newline_idx + 1 :]

        if role not in ("system", "user", "assistant", "tool"):
            logger.warning(f"Unexpected role '{role}', skipping segment")
            continue

        messages.append({"role": role, "content": content})

    return messages


def replace_content(messages: list[dict], images: list[str]) -> list[dict]:
    """Rewrite message content strings for the SFT pipeline.

    1. Validate that the number of vision placeholders equals ``len(images)``.
    2. Sanitize raw ``<video>``/``<image>``/``<audio>`` tags that may appear
       inside page text (e.g. HTML fragments in screenshots) to
       ``<vid>``/``<img>``/``<aud>`` so LlamaFactory does not interpret them
       as multimodal placeholders.
    3. Replace ``<|vision_start|><|image_pad|><|vision_end|>`` with ``<image>``.

    Order matters: sanitization runs BEFORE the vision-placeholder
    replacement, so real image markers (``<image>``) stay distinct from
    sanitized raw-text collisions (``<img>``).

    Args:
        messages: List of message dicts with string content.
        images: List of base64 data-URL strings for each image.

    Returns:
        Updated messages with content strings rewritten in place.

    Raises:
        ValueError: If placeholder count does not match image count.
    """
    total_placeholders = sum(
        msg["content"].count(_VISION_PLACEHOLDER)
        for msg in messages
        if isinstance(msg["content"], str)
    )

    if total_placeholders != len(images):
        raise ValueError(
            f"Placeholder count ({total_placeholders}) != image count ({len(images)}). "
            f"Messages may be corrupted."
        )

    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, str):
            continue
        # 1. Sanitize raw collisions first, before the vision-placeholder
        #    rewrite introduces real <image> markers.
        content = (
            content
            .replace("<video>", "<vid>").replace("</video>", "</vid>")
            .replace("<image>", "<img>").replace("</image>", "</img>")
            .replace("<audio>", "<aud>").replace("</audio>", "</aud>")
        )
        # 2. Then expand vision placeholders to the real image marker.
        content = content.replace(_VISION_PLACEHOLDER, _IMAGE_PLACEHOLDER)
        msg["content"] = content

    return messages


def apply_step_loss_mask(messages: list[dict]) -> list[dict]:
    """Set ``step_loss_mask=0`` on all assistant messages except the last.

    This ensures that only the final assistant response (the current turn's
    output) is supervised during SFT, not earlier assistant turns in the
    context window.
    """
    assistant_indices = [i for i, m in enumerate(messages) if m["role"] == "assistant"]

    if len(assistant_indices) <= 1:
        return messages

    for idx in assistant_indices[:-1]:
        messages[idx]["step_loss_mask"] = 0

    return messages


def _save_base64_image(data_url: str, path: str) -> None:
    """Decode a base64 data-URL and save as a PNG file."""
    if "," in data_url:
        b64_str = data_url.split(",", 1)[1]
    else:
        b64_str = data_url
    raw_bytes = base64.b64decode(b64_str)
    with open(path, "wb") as f:
        f.write(raw_bytes)


def convert_row(
    row: dict,
    image_dir: str,
    row_idx: int,
    *,
    image_path_mode: str = "relative",
) -> dict | None:
    """Convert a single JSONL row to an SFT-ready dict.

    Saves base64 images as PNG files under ``image_dir`` and stores
    file paths in the output ``images`` list (instead of base64 strings).
    This avoids OOM when the Dataset loads all images into PIL at init time.

    Returns:
        Dict with ``messages`` and ``images`` keys, or None if conversion fails.
    """
    prompt = row.get("prompt", "")
    response = row.get("response", "")

    # Get images — try multimodal_inputs first, fall back to metadata
    mm_inputs = row.get("multimodal_inputs")
    if isinstance(mm_inputs, str):
        try:
            mm_inputs = json.loads(mm_inputs)
        except json.JSONDecodeError:
            mm_inputs = None
    if mm_inputs and "images" in mm_inputs:
        raw_images = mm_inputs["images"]
    else:
        metadata = row.get("metadata")
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except json.JSONDecodeError:
                metadata = {}
        metadata = metadata or {}
        raw_images = metadata.get("images", [])

    # Concatenate prompt + response to get full templated text
    full_text = prompt + response

    # Parse back into message dicts
    try:
        messages = parse_templated_text_to_messages(full_text)
    except Exception as e:
        logger.warning(f"Failed to parse messages: {e}")
        return None

    if not messages:
        logger.warning("No messages parsed from text")
        return None

    # Sanitize raw collisions and expand vision placeholders in one pass.
    try:
        messages = replace_content(messages, raw_images)
    except ValueError as e:
        logger.warning(f"Image/placeholder mismatch: {e}")
        return None

    # Apply per-turn loss masking
    messages = apply_step_loss_mask(messages)

    # Save images as files and collect paths
    meta = _parse_row_metadata(row)
    task_id = str(meta.get("task_id", "unknown")).replace("/", "_").replace(" ", "_")
    rollout_idx = row.get("rollout_idx", 0)
    turn_index = meta.get("turn_index", 0)

    image_paths = []
    image_root = Path(image_dir).parent
    for img_idx, data_url in enumerate(raw_images):
        fname = f"{row_idx:06d}_task_{task_id}_rollout_{rollout_idx}_turn_{turn_index}_{img_idx}.png"
        fpath = os.path.join(image_dir, fname)
        try:
            _save_base64_image(data_url, fpath)
            if image_path_mode == "relative":
                image_paths.append(os.path.relpath(fpath, start=image_root))
            else:
                image_paths.append(os.path.abspath(fpath))
        except Exception as e:
            logger.warning(f"Row {row_idx}: failed to save image {img_idx}: {e}")
            return None

    return {"messages": messages, "images": image_paths}


def _parse_row_metadata(row: dict) -> dict:
    meta = row.get("metadata")
    if isinstance(meta, str):
        try:
            return json.loads(meta)
        except json.JSONDecodeError:
            return {}
    return meta or {}


def process_file(
    path: str,
    *,
    image_dir: str,
    output_fp,
    row_counter: list,
    max_rows_remaining: int | None,
    stats: dict,
    image_path_mode: str,
) -> None:
    """Load → convert → append one JSONL shard.

    ``row_counter`` is a single-element list used as a mutable int shared
    across shards so image filenames stay unique and ordered. ``stats`` is
    mutated in place: keys ``rows_loaded``, ``converted``, ``skipped``.

    Writes converted rows directly to ``output_fp`` so memory stays at
    ~one shard. The function does not close ``output_fp``.
    """
    logger.info(f"Loading {path} (max_rows={max_rows_remaining})")
    rows = load_jsonl(path, max_rows=max_rows_remaining)
    stats["rows_loaded"] += len(rows)
    logger.info(f"Loaded {len(rows)} rows from {os.path.basename(path)}")

    for row in tqdm(rows, desc=f"Converting {os.path.basename(path)}", leave=False):
        result = convert_row(
            row,
            image_dir=image_dir,
            row_idx=row_counter[0],
            image_path_mode=image_path_mode,
        )
        row_counter[0] += 1
        if result is None:
            stats["skipped"] += 1
            continue
        # Save as JSONL — pyarrow's iter_batches() cannot handle deeply nested
        # structs (messages with step_loss_mask), but slime's JSONL reader
        # in data.py handles them fine via json.loads().
        output_fp.write(json.dumps(result, ensure_ascii=False) + "\n")
        stats["converted"] += 1


def main():
    parser = argparse.ArgumentParser(description="Convert browser rollout JSONL to SFT JSONL")
    parser.add_argument("--input-path", required=True, help="Path to JSONL file or glob pattern (e.g. '/path/to/results-*.jsonl')")
    parser.add_argument("--output-path", default=None, help="Path to output JSONL file (default: <input_dir>/sft_data.jsonl)")
    parser.add_argument("--max-rows", type=int, default=None, help="Max JSONL rows to load (for testing, e.g. 100)")
    parser.add_argument(
        "--image-path-mode",
        choices=("relative", "absolute"),
        default="relative",
        help="How image paths are written into the output JSONL. Relative paths are rooted at the output file directory.",
    )
    args = parser.parse_args()

    # Resolve input paths and default output path
    input_paths = resolve_input_paths(args.input_path)
    if args.output_path:
        output_path = Path(args.output_path)
    else:
        output_path = Path(input_paths[0]).parent / "sft_data.jsonl"

    image_dir = str(output_path.parent / "images")
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(output_path.parent, exist_ok=True)
    logger.info(f"Saving images to {image_dir}")

    logger.info(f"Processing {len(input_paths)} file(s) → {output_path}")

    stats = {"rows_loaded": 0, "converted": 0, "skipped": 0}
    row_counter = [0]
    max_rows_budget = args.max_rows

    with open(output_path, "w", encoding="utf-8") as output_fp:
        for i, path in enumerate(input_paths):
            logger.info(f"[{i + 1}/{len(input_paths)}] {path}")
            if max_rows_budget is None:
                remaining = None
            else:
                remaining = max(0, max_rows_budget - stats["rows_loaded"])
                if remaining == 0:
                    logger.info(f"Reached --max-rows={max_rows_budget}, stopping")
                    break
            process_file(
                path,
                image_dir=image_dir,
                output_fp=output_fp,
                row_counter=row_counter,
                max_rows_remaining=remaining,
                stats=stats,
                image_path_mode=args.image_path_mode,
            )
            output_fp.flush()

    logger.info(
        f"Total: loaded={stats['rows_loaded']}, "
        f"converted={stats['converted']}, skipped={stats['skipped']}"
    )
    if stats["converted"] == 0:
        logger.error("No rows to save. Check the input data.")
        sys.exit(1)
    logger.info(f"Saved {stats['converted']} rows to {output_path}")


if __name__ == "__main__":
    main()
