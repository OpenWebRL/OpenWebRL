"""
Notebook-friendly helpers for visualizing browser rollout debug traces.

Typical usage in Jupyter:

    from pathlib import Path
    import importlib.util

    helper_path = Path("<repo-root>/openwebrl/visualize_debug_trace.py")
    spec = importlib.util.spec_from_file_location("visualize_debug_trace_helper", helper_path)
    helper = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(helper)

    sample_path = "/path/to/debug_traces/rollout/turn/sample.json"
    helper.show_debug_trace_summary(sample_path)
    helper.show_debug_trace(sample_path)
"""

from __future__ import annotations

import argparse
import base64
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


def _draw_x(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    size: int = 16,
    color: str = "red",
    width: int = 4,
) -> None:
    draw.line((x - size, y - size, x + size, y + size), fill=color, width=width)
    draw.line((x - size, y + size, x + size, y - size), fill=color, width=width)


def _annotate_image(img_bytes: bytes, assistant_text: str | None) -> tuple[bytes, list[str]]:
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
                labels.append(
                    f"{name}: start_point_2d={args['start_point_2d']} -> pixel=({x}, {y})"
                )

            if isinstance(args.get("end_point_2d"), list) and len(args["end_point_2d"]) == 2:
                x, y = _to_pixel(args["end_point_2d"], width, height)
                _draw_x(draw, x, y, color="orange")
                labels.append(
                    f"{name}: end_point_2d={args['end_point_2d']} -> pixel=({x}, {y})"
                )

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


def _get_chat_transcript_text(trajectory_data: dict[str, Any]) -> str:
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


def _build_steps(trace_data: dict[str, Any]) -> list[dict[str, Any]]:
    blocks = _parse_chat_blocks(_get_chat_transcript_text(trace_data))
    image_groups = trace_data.get("images", [])
    step_boundaries = [
        idx
        for idx, block in enumerate(blocks)
        if block["role"] == "user" and _has_user_request(block["content"])
    ]

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


def list_debug_traces(trace_dir: str) -> list[str]:
    root = Path(trace_dir)
    return sorted(str(path) for path in root.glob("*.json"))


def inspect_debug_trace_alignment(sample_path: str) -> list[dict[str, Any]]:
    data = _load_json(Path(sample_path))
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


def show_debug_trace_summary(sample_path: str) -> None:
    display, Markdown, _, _ = _require_notebook_display()
    path = Path(sample_path)
    data = _load_json(path)
    steps = _build_steps(data)

    summary = [
        f"## Debug Trace Summary: `{path.name}`",
        f"- sample_path: `{path}`",
        f"- sample_id: `{data.get('sample_id')}`",
        f"- status: `{data.get('status')}`",
        f"- total_steps: `{data.get('total_steps')}`",
        f"- parsed_steps: `{len(steps)}`",
        f"- terminate_reason: `{data.get('terminate_reason')}`",
        f"- image_groups: `{len(data.get('images', []))}`",
    ]
    display(Markdown("\n".join(summary)))


def show_debug_trace(sample_path: str, max_steps: int | None = None) -> None:
    display, Markdown, Image, HTML = _require_notebook_display()
    path = Path(sample_path)
    data = _load_json(path)
    steps = _build_steps(data)

    summary = [
        f"## Debug Trace: `{path.name}`",
        f"- status: `{data.get('status')}`",
        f"- total_steps: `{data.get('total_steps')}`",
        f"- parsed_steps: `{len(steps)}`",
        f"- terminate_reason: `{data.get('terminate_reason')}`",
    ]
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize browser rollout debug traces in Jupyter/IPython")
    parser.add_argument("--sample-path", required=True, help="Path to a debug trace json file")
    parser.add_argument("--max-steps", type=int, default=None, help="Optional max number of steps to render")
    args = parser.parse_args()

    show_debug_trace_summary(args.sample_path)
    show_debug_trace(args.sample_path, max_steps=args.max_steps)


if __name__ == "__main__":
    main()
