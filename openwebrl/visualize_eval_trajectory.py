"""
Notebook-friendly helpers for visualizing browser evaluation trajectories.

Typical usage in Jupyter:

    from openwebrl.visualize_eval_trajectory import (
        list_trajectories,
        show_trajectory,
        show_result_summary,
    )

    eval_dir = "eval_outputs/browser_eval/<run_dir>"
    list_trajectories(eval_dir)[:5]
    show_result_summary(eval_dir, "webvoyager/83")
    show_trajectory(eval_dir, "webvoyager/83")

You can also run this as a script:

    python openwebrl/visualize_eval_trajectory.py \
      --eval-dir /path/to/eval_dir \
      --task-id webvoyager/83
"""

from __future__ import annotations

import argparse
import base64
import difflib
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from PIL import Image as PILImage
from PIL import ImageDraw


_CHAT_BLOCK_RE = re.compile(r"<\|im_start\|>(\w+)\n(.*?)<\|im_end\|>", re.DOTALL)
_SCREENSHOT_PLACEHOLDER_RE = re.compile(
    r"\nscreenshot:\n<\|vision_start\|><\|image_pad\|><\|vision_end\|>", re.DOTALL
)
_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
_ACTION_RE = re.compile(r"<action>\s*(.*?)\s*</action>", re.DOTALL)
_TOOL_RESPONSE_RE = re.compile(r"<tool_response>\s*(.*?)\s*</tool_response>", re.DOTALL)


def _require_notebook_display():
    try:
        from IPython.display import HTML, Image, Markdown, display
    except ImportError as exc:
        raise RuntimeError(
            "This helper is designed for Jupyter/IPython. Install IPython or run it in a notebook."
        ) from exc
    return display, Markdown, Image, HTML


def _load_json(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def _load_jsonl_first(path: Path) -> dict[str, Any]:
    with open(path) as f:
        for line in f:
            if line.strip():
                return json.loads(line)
    raise ValueError(f"No JSON object found in {path}")


def _load_jsonl_last(path: Path) -> dict[str, Any]:
    last_obj: dict[str, Any] | None = None
    with open(path) as f:
        for line in f:
            if line.strip():
                last_obj = json.loads(line)
    if last_obj is None:
        raise ValueError(f"No JSON object found in {path}")
    return last_obj


def _decode_data_url(data_url: str) -> bytes:
    if "base64," in data_url:
        data_url = data_url.split(",", 1)[1]
    return base64.b64decode(data_url)


def _parse_tool_calls(text: str) -> list[dict[str, Any]]:
    calls = []
    for match in _TOOL_CALL_RE.findall(text or ""):
        try:
            calls.append(json.loads(match))
        except json.JSONDecodeError:
            continue
    return calls


def _parse_action_text(text: str | None) -> str | None:
    if not text:
        return None
    match = _ACTION_RE.search(text)
    if match:
        return match.group(1).strip()
    return None


def _parse_tool_response_text(text: str | None) -> str | None:
    if not text:
        return None
    match = _TOOL_RESPONSE_RE.search(text)
    if match:
        return match.group(1).strip()
    return None


def _to_pixel(point_2d: list[float | int], width: int, height: int) -> tuple[int, int]:
    x_norm, y_norm = point_2d
    x = round(float(x_norm) / 1000.0 * width)
    y = round(float(y_norm) / 1000.0 * height)
    return max(0, min(width - 1, x)), max(0, min(height - 1, y))


def _draw_x(draw: ImageDraw.ImageDraw, x: int, y: int, size: int = 16, color: str = "red", width: int = 4) -> None:
    draw.line((x - size, y - size, x + size, y + size), fill=color, width=width)
    draw.line((x - size, y + size, x + size, y - size), fill=color, width=width)


def _annotate_image(img_bytes: bytes, assistant_text: str | None) -> tuple[bytes, list[str]]:
    """
    Annotate screenshot with action markers extracted from assistant tool calls.

    point_2d values are normalized to [0, 1000] and mapped to image pixels.
    """
    try:
        from io import BytesIO

        image = PILImage.open(BytesIO(img_bytes)).convert("RGB")
        draw = ImageDraw.Draw(image)
        width, height = image.size
        labels: list[str] = []

        for call in _parse_tool_calls(assistant_text or ""):
            name = call.get("name", "unknown")
            args = call.get("arguments", {}) or {}

            if isinstance(args.get("point_2d"), list) and len(args["point_2d"]) == 2:
                x, y = _to_pixel(args["point_2d"], width, height)
                _draw_x(draw, x, y)
                labels.append(f"{name}: point_2d={args['point_2d']} -> pixel=({x}, {y})")

            if isinstance(args.get("start_point_2d"), list) and len(args["start_point_2d"]) == 2:
                x, y = _to_pixel(args["start_point_2d"], width, height)
                _draw_x(draw, x, y, color="red")
                labels.append(f"{name}: start_point_2d={args['start_point_2d']} -> pixel=({x}, {y})")

            if isinstance(args.get("end_point_2d"), list) and len(args["end_point_2d"]) == 2:
                x, y = _to_pixel(args["end_point_2d"], width, height)
                _draw_x(draw, x, y, color="orange")
                labels.append(f"{name}: end_point_2d={args['end_point_2d']} -> pixel=({x}, {y})")

        output = BytesIO()
        image.save(output, format="PNG")
        return output.getvalue(), labels
    except Exception:
        return img_bytes, []


def _parse_chat_blocks(text: str) -> list[dict[str, str]]:
    blocks = []
    for role, content in _CHAT_BLOCK_RE.findall(text):
        blocks.append({"role": role, "content": content.strip()})
    return blocks


def _normalize_user_content(text: str) -> str:
    return _SCREENSHOT_PLACEHOLDER_RE.sub("", text).strip()


def _has_user_request(text: str | None) -> bool:
    return bool(text and "<user_request>" in text)


def _extract_observation_url(text: str | None) -> str | None:
    if not text:
        return None
    match = re.search(r"url:\s*(.*)", text)
    if match:
        return match.group(1).strip()
    return None


def _image_signature(data_url: str) -> str:
    img_bytes = _decode_data_url(data_url)
    return hashlib.md5(img_bytes).hexdigest()[:8]


def _build_steps(trajectory_data: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Build per-step view from the saved trajectory json.

    The saved format contains:
    - llm_response: full conversation after the initial system prompt
    - images: one list of screenshots per user turn
    """
    blocks = _parse_chat_blocks(_get_chat_transcript_text(trajectory_data))
    image_groups = trajectory_data.get("images", [])
    step_boundaries = [idx for idx, block in enumerate(blocks) if block["role"] == "user" and _has_user_request(block["content"])]

    steps: list[dict[str, Any]] = []
    for image_idx, start_idx in enumerate(step_boundaries):
        end_idx = step_boundaries[image_idx + 1] if image_idx + 1 < len(step_boundaries) else len(blocks)
        step_blocks = blocks[start_idx:end_idx]
        if not step_blocks:
            continue

        first_block = step_blocks[0]
        step: dict[str, Any] = {
            "observation": _normalize_user_content(first_block["content"]),
            "images": image_groups[image_idx] if image_idx < len(image_groups) else [],
            "assistant": None,
            "tool_feedback": [],
            "tool_calls": [],
        }

        for block in step_blocks[1:]:
            role = block["role"]
            content = block["content"]

            if role == "assistant":
                step["assistant"] = content
                step["action"] = _parse_action_text(content)
                step["tool_calls"] = _parse_tool_calls(content)
            elif role == "user":
                tool_response = _parse_tool_response_text(content)
                if tool_response is not None:
                    step["tool_feedback"].append(tool_response)
                else:
                    step.setdefault("extra_user_messages", []).append(_normalize_user_content(content))
            elif role == "tool":
                step["tool_feedback"].append(content)
            else:
                step.setdefault(role, []).append(content)

        steps.append(step)

    return steps


def _get_chat_transcript_text(trajectory_data: dict[str, Any]) -> str:
    """
    Return a parseable full chat transcript for both trajectory-level and turn-level samples.

    Trajectory-level saves the full multi-turn chat in ``llm_response``.
    Turn-level saves the conversation context in ``llm_input_texts`` and the final
    assistant turn separately in ``llm_response``.
    """
    llm_response = trajectory_data.get("llm_response", "") or ""
    llm_input_texts = trajectory_data.get("llm_input_texts", "") or ""

    if "<|im_start|>" in llm_response:
        return llm_response

    if llm_input_texts.rstrip().endswith("<|im_start|>assistant"):
        return llm_input_texts + llm_response + "<|im_end|>"

    if llm_input_texts.rstrip().endswith("<|im_start|>assistant\n"):
        return llm_input_texts + llm_response + "<|im_end|>"

    if "<|im_start|>" in llm_input_texts:
        return llm_input_texts

    return llm_response


def _extract_judge_info(result: dict[str, Any]) -> dict[str, Any]:
    reward_meta = (result.get("metadata") or {}).get("reward") or {}
    judge_score = reward_meta.get("judge")
    judge_text = reward_meta.get("judge_text")

    judge_result: str | None = None
    if isinstance(judge_score, (int, float)):
        judge_result = "SUCCESS" if float(judge_score) >= 0.5 else "FAILURE"
    elif isinstance(judge_text, str):
        upper_text = judge_text.upper()
        if "NOT SUCCESS" in upper_text:
            judge_result = "FAILURE"
        elif "SUCCESS" in upper_text:
            judge_result = "SUCCESS"

    return {
        "judge_score": judge_score,
        "judge_result": judge_result,
        "judge_reason": judge_text,
    }


def _iter_saved_sample_paths(eval_dir: str) -> list[Path]:
    root = Path(eval_dir)
    paths = list(root.glob("**/trajectory/**/*.json"))
    paths.extend(root.glob("**/turn/**/*.json"))

    # Also support passing a leaf directory that directly contains sample JSON files.
    # Restrict to browser sample naming convention to avoid picking up unrelated JSONs.
    if root.is_dir():
        paths.extend(root.glob("webvoyager_*.json"))

    return sorted(paths)


def _sample_kind_from_path(path: Path) -> str:
    if "turn" in path.parts:
        return "turn"
    if "trajectory" in path.parts:
        return "trajectory"
    return "sample"


def _task_id_from_filename(path: Path) -> str:
    parts = path.stem.split("_")
    if len(parts) >= 2:
        return f"{parts[0]}/{parts[1]}"
    return path.stem.replace("_", "/")


def _normalize_task_id(task_id: str) -> str:
    task_id = task_id.strip()
    if task_id.endswith(".json"):
        return _task_id_from_filename(Path(task_id))
    parts = task_id.split("/")
    if len(parts) == 1:
        underscore_parts = task_id.split("_")
        if len(underscore_parts) == 2:
            return f"{underscore_parts[0]}/{underscore_parts[1]}"
    return task_id


def _task_id_aliases(task_id: str) -> list[str]:
    normalized = _normalize_task_id(task_id)
    aliases = [normalized]
    if "/" in normalized:
        aliases.append(normalized.replace("/", "_"))
    if "_" in normalized:
        aliases.append(normalized.replace("_", "/"))
    return list(dict.fromkeys(aliases))


def _suggest_task_ids(eval_dir: str, task_id: str, limit: int = 8) -> list[str]:
    requested_aliases = _task_id_aliases(task_id)
    available = [item["task_id"] for item in list_trajectories(eval_dir)]
    if not available:
        return []

    normalized_available = {candidate: _normalize_task_id(candidate) for candidate in available}
    candidate_pool = set(available)
    for requested in requested_aliases:
        requested_normalized = _normalize_task_id(requested)
        requested_suffix = requested_normalized.split("/")[-1]
        for candidate, candidate_normalized in normalized_available.items():
            if candidate_normalized == requested_normalized:
                candidate_pool.add(candidate)
            elif requested_suffix and candidate_normalized.endswith(f"/{requested_suffix}"):
                candidate_pool.add(candidate)

        candidate_pool.update(
            difflib.get_close_matches(requested_normalized, list(normalized_available.values()), n=limit, cutoff=0.4)
        )
        candidate_pool.update(
            difflib.get_close_matches(requested.replace("/", "_"), available, n=limit, cutoff=0.4)
        )

    ordered: list[str] = []
    for candidate in available:
        if candidate in candidate_pool or normalized_available[candidate] in candidate_pool:
            ordered.append(candidate)
        if len(ordered) >= limit:
            break
    return ordered


def list_trajectories(eval_dir: str) -> list[dict[str, str]]:
    """
    Return discovered trajectory/turn JSON files under an evaluation directory.
    """
    out = []
    for path in _iter_saved_sample_paths(eval_dir):
        out.append(
            {
                "task_id": _task_id_from_filename(path),
                "path": str(path),
                "name": path.name,
                "kind": _sample_kind_from_path(path),
            }
        )
    return out


def find_trajectory_path(eval_dir: str, task_id: str) -> Path:
    """
    Find a saved trajectory/turn json for task_id like 'webvoyager/83'.
    """
    task_id = _normalize_task_id(task_id)
    matches = _iter_saved_sample_paths(eval_dir)
    task_prefixes = [alias.replace("/", "_") + "_" for alias in _task_id_aliases(task_id)]
    for path in matches:
        if any(path.name.startswith(prefix) for prefix in task_prefixes):
            return path
    suggestions = _suggest_task_ids(eval_dir, task_id)
    message = f"No trajectory/turn json found for {task_id} under {eval_dir}"
    if suggestions:
        message += "\nDid you mean one of: " + ", ".join(suggestions)
    raise FileNotFoundError(message)


def find_result_path(eval_dir: str, task_id: str) -> Path:
    """
    Find the corresponding results_task_*.jsonl file for task_id.
    """
    task_id = _normalize_task_id(task_id)
    suffix = task_id.replace("/", "_")
    filename = f"results_task_{suffix}.jsonl"
    root = Path(eval_dir)

    direct_path = root / filename
    if direct_path.exists():
        return direct_path

    for parent in [root, *root.parents]:
        candidate = parent / filename
        if candidate.exists():
            return candidate

    recursive_matches = sorted(root.glob(f"**/{filename}"))
    if recursive_matches:
        return recursive_matches[0]

    raise FileNotFoundError(f"No result jsonl found for {task_id} under {eval_dir}")


def show_result_summary(eval_dir: str, task_id: str) -> None:
    """
    Display high-level evaluation summary for one task in Jupyter.
    """
    display, Markdown, _, _ = _require_notebook_display()
    task_id = _normalize_task_id(task_id)
    try:
        result = _load_jsonl_last(find_result_path(eval_dir, task_id))
    except FileNotFoundError:
        display(Markdown(f"## Result Summary: `{task_id}`\n- result jsonl: not found"))
        return
    judge_info = _extract_judge_info(result)
    md = [
        f"## Result Summary: `{task_id}`",
        f"- status: `{result.get('status')}`",
        f"- reward: `{result.get('reward')}`",
        f"- total_steps: `{result.get('total_steps')}`",
        f"- terminate_reason: `{result.get('terminate_reason')}`",
    ]
    if judge_info["judge_result"] is not None:
        md.append(f"- judge_evaluation: `{judge_info['judge_result']}`")
    if judge_info["judge_score"] is not None:
        md.append(f"- judge_score: `{judge_info['judge_score']}`")
    reward_meta = (result.get("metadata") or {}).get("reward")
    if reward_meta:
        md.append(f"- reward_meta: `{reward_meta}`")
    if judge_info["judge_reason"]:
        md.append("")
        md.append("### Judge Reason")
        md.append(f"```text\n{judge_info['judge_reason']}\n```")
    display(Markdown("\n".join(md)))


def inspect_trajectory_alignment(eval_dir: str, task_id: str) -> list[dict[str, Any]]:
    """
    Return a compact per-step summary for checking observation/image/action alignment.
    """
    task_id = _normalize_task_id(task_id)
    path = find_trajectory_path(eval_dir, task_id)
    data = _load_json(path)
    steps = _build_steps(data)

    inspection: list[dict[str, Any]] = []
    for idx, step in enumerate(steps, start=1):
        inspection.append(
            {
                "step": idx,
                "url": _extract_observation_url(step.get("observation")),
                "image_signatures": [_image_signature(img) for img in step.get("images", [])],
                "action": step.get("action"),
                "tool_feedback": step.get("tool_feedback", []),
                "has_assistant": bool(step.get("assistant")),
            }
        )
    return inspection


def show_trajectory(eval_dir: str, task_id: str, max_steps: int | None = None) -> None:
    """
    Display a saved trajectory/turn sample step-by-step in Jupyter.
    """
    display, Markdown, Image, HTML = _require_notebook_display()
    task_id = _normalize_task_id(task_id)
    try:
        path = find_trajectory_path(eval_dir, task_id)
    except FileNotFoundError:
        suggestions = _suggest_task_ids(eval_dir, task_id)
        md = [
            f"## Trajectory: `{task_id}`",
            f"- trajectory json: not found under `{eval_dir}`",
            "",
            "Use `list_trajectories(eval_dir)` to inspect the available tasks.",
        ]
        if suggestions:
            md.append("")
            md.append("### Nearby Matches")
            md.extend([f"- `{candidate}`" for candidate in suggestions])
        else:
            md.append("")
            md.append("- no saved trajectories were discovered in this directory")
        display(Markdown("\n".join(md)))
        return
    data = _load_json(path)
    steps = _build_steps(data)
    sample_kind = _sample_kind_from_path(path)
    result: dict[str, Any] | None = None
    judge_info = {"judge_score": None, "judge_result": None, "judge_reason": None}
    try:
        result = _load_jsonl_last(find_result_path(eval_dir, task_id))
        judge_info = _extract_judge_info(result)
    except (FileNotFoundError, ValueError):
        pass

    summary = [
        f"## {sample_kind.capitalize()}: `{task_id}`",
        f"- file: `{path.name}`",
        f"- kind: `{sample_kind}`",
        f"- status: `{data.get('status')}`",
        f"- total_steps: `{data.get('total_steps')}`",
        f"- terminate_reason: `{data.get('terminate_reason')}`",
    ]
    if judge_info["judge_result"] is not None:
        summary.append(f"- judge_evaluation: `{judge_info['judge_result']}`")
    if judge_info["judge_score"] is not None:
        summary.append(f"- judge_score: `{judge_info['judge_score']}`")
    display(Markdown("\n".join(summary)))

    if max_steps is not None:
        steps = steps[:max_steps]

    for idx, step in enumerate(steps, start=1):
        display(Markdown(f"### Step {idx}"))

        observation_text = step.get("observation")
        if observation_text:
            display(Markdown("**Observation**"))
            display(Markdown(f"```text\n{observation_text}\n```"))

        images = step.get("images", [])
        if images:
            display(Markdown(f"**Screenshots ({len(images)})**"))
            for img_idx, img_url in enumerate(images, start=1):
                try:
                    img_bytes = _decode_data_url(img_url)
                    annotated, labels = _annotate_image(img_bytes, step.get("assistant"))
                    display(Markdown(f"Screenshot {img_idx}"))
                    display(Image(data=annotated))
                    if labels:
                        display(Markdown("Action markers:"))
                        for label in labels:
                            display(Markdown(f"- `{label}`"))
                except Exception as exc:
                    display(Markdown(f"`Failed to decode screenshot {img_idx}: {exc}`"))

        action_text = step.get("action")
        if action_text:
            display(Markdown("**Action**"))
            display(Markdown(f"```text\n{action_text}\n```"))

        assistant_text = step.get("assistant")
        if assistant_text:
            display(Markdown("**Assistant Full Response**"))
            display(Markdown(f"```text\n{assistant_text}\n```"))

        tool_calls = step.get("tool_calls", [])
        if tool_calls:
            display(Markdown("**Parsed Tool Calls**"))
            display(Markdown(f"```json\n{json.dumps(tool_calls, indent=2, ensure_ascii=False)}\n```"))

        tool_msgs = step.get("tool_feedback", [])
        if tool_msgs:
            display(Markdown("**Tool Feedback**"))
            for tool_msg in tool_msgs:
                display(Markdown(f"```text\n{tool_msg}\n```"))

        display(HTML("<hr>"))

    if result is not None and (
        judge_info["judge_reason"] is not None or judge_info["judge_result"] is not None
    ):
        judge_summary = ["## Judge Result"]
        if judge_info["judge_result"] is not None:
            judge_summary.append(f"- evaluation: `{judge_info['judge_result']}`")
        if judge_info["judge_score"] is not None:
            judge_summary.append(f"- score: `{judge_info['judge_score']}`")
        display(Markdown("\n".join(judge_summary)))
        if judge_info["judge_reason"]:
            display(Markdown("**Judge Reason**"))
            display(Markdown(f"```text\n{judge_info['judge_reason']}\n```"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize browser eval trajectories in Jupyter/IPython")
    parser.add_argument("--eval-dir", required=True, help="Path to browser eval directory")
    parser.add_argument("--task-id", help="Task id like webvoyager/83")
    parser.add_argument("--list", action="store_true", help="List discovered trajectory files")
    parser.add_argument("--max-steps", type=int, default=None, help="Optional max number of steps to render")
    args = parser.parse_args()

    if args.list:
        for item in list_trajectories(args.eval_dir):
            print(f"{item['task_id']:20s}  {item['path']}")
        return

    if not args.task_id:
        raise SystemExit("--task-id is required unless --list is used")

    show_result_summary(args.eval_dir, args.task_id)
    show_trajectory(args.eval_dir, args.task_id, max_steps=args.max_steps)


if __name__ == "__main__":
    main()
