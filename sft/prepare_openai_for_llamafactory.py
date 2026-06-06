"""
Stage 2 of the OpenWebRL SFT data pipeline: render the model-agnostic canonical
OpenAI episodes (from ``convert_to_openai_messages.py``) into LLaMAFactory SFT
data **for a chosen base model**, using that model's **official chat template**.

Why the official template (instead of LLaMAFactory's tool rendering):

The OpenWebRL runtime builds every inference/RL prompt with
``tokenizer.apply_chat_template(messages, tools=tools_info, ...)`` — the model's
own template (see ``openwebrl/base/utils.py:TokenHandler``). LLaMAFactory's own
``qwen3_5`` template diverges from that official format (tool block placed after
the system prompt instead of before, merged ``<tool_response>`` blocks, etc.),
so training on LLaMAFactory-rendered tools would not match inference. To stay
byte-consistent with the official inference setup for any model, this stage calls
the same ``apply_chat_template`` the runtime uses, then parses the result back
into LLaMAFactory's ShareGPT format. LLaMAFactory then only re-wraps each turn
(``<|im_start|>role ... <|im_end|>``), computes loss masks, and expands image
tokens — it never re-renders the tool format.

Granularity is controlled by ``--per-turn``:

* default (trajectory-level): one record per episode, **all** screenshots kept.
  Train with ``mask_history: false`` to supervise every assistant turn.
* ``--per-turn``: each N-turn episode is expanded into N records; record ``k`` is
  the prefix ending at turn ``k``'s assistant, rendered in the exact context the
  runtime would generate that turn in, keeping only turn ``k``'s **current**
  screenshot (historical screenshots stripped). Train with ``mask_history: true``
  so the loss falls only on the current turn.

Usage (run from the LLaMAFactory env / wherever the model tokenizer is available)::

    python sft/prepare_openai_for_llamafactory.py \
        --input-path  /path/to/openwebrl_sft_openai.jsonl \
        --output-path /path/to/data/openwebrl_sft_llamafactory.jsonl \
        --model-name-or-path Qwen/Qwen3.5-9B [--per-turn]
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import os
from pathlib import Path

from tqdm import tqdm
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("prepare_openai_for_llamafactory")

_IM_START = "<|im_start|>"
_IM_END = "<|im_end|>"
_VISION_PLACEHOLDER = "<|vision_start|><|image_pad|><|vision_end|>"
_IMAGE_TOKEN = "<image>"
_VALID_ROLES = ("system", "user", "assistant", "tool")


def load_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _save_base64_image(data_url: str, path: str) -> None:
    b64 = data_url.split(",", 1)[1] if "," in data_url else data_url
    with open(path, "wb") as f:
        f.write(base64.b64decode(b64))


def _sanitize_text(text: str) -> str:
    """Neutralize raw <image>/<video>/<audio> tags from page text so LLaMAFactory
    does not mistake them for multimodal placeholders."""
    return (
        text.replace("<video>", "<vid>").replace("</video>", "</vid>")
        .replace("<image>", "<img>").replace("</image>", "</img>")
        .replace("<audio>", "<aud>").replace("</audio>", "</aud>")
    )


def _strip_extra_fields(messages: list[dict]) -> list[dict]:
    """Keep only the fields the chat template needs (role/content/tool_calls).

    Content (possibly a multimodal list with ``image_url`` parts) is passed
    through unchanged so the template emits a vision placeholder per image.
    """
    clean = []
    for m in messages:
        cm: dict = {"role": m["role"], "content": m.get("content", "")}
        if m["role"] == "assistant" and m.get("tool_calls"):
            cm["tool_calls"] = m["tool_calls"]
        clean.append(cm)
    return clean


def _image_urls(messages: list[dict]) -> list[str]:
    """Ordered base64 data-URLs of every image in the messages (document order)."""
    return [
        part["image_url"]["url"]
        for m in messages
        if isinstance(m.get("content"), list)
        for part in m["content"]
        if part.get("type") == "image_url"
    ]


def parse_turns(text: str) -> list[dict]:
    """Parse ``apply_chat_template`` output into ``[{role, content}]`` turns.

    Splits on ``<|im_start|>`` and drops the ``<|im_end|>`` suffix, mirroring the
    inverse of LLaMAFactory's per-turn wrapping so the content can be re-wrapped
    identically.
    """
    turns = []
    for segment in text.split(_IM_START):
        if not segment.strip():
            continue
        if _IM_END in segment:
            segment = segment[: segment.rfind(_IM_END)]
        newline = segment.find("\n")
        if newline == -1:
            role, content = segment.strip(), ""
        else:
            role, content = segment[:newline].strip(), segment[newline + 1 :]
        if role not in _VALID_ROLES:
            logger.warning(f"Unexpected role {role!r} in rendered text; skipping turn")
            continue
        turns.append({"role": role, "content": content})
    return turns


def _flatten_turn(text: str, keep_flags: list[bool], url_iter, image_sink: list[str]) -> str:
    """Replace vision placeholders in one rendered turn with ``<image>`` tokens.

    ``keep_flags`` is consumed (pop(0)) per placeholder in document order; a kept
    image emits ``<image>`` and its data-URL is appended to ``image_sink``; a
    dropped image is removed along with its ``screenshot:\\n...\\n`` wrapper
    (matching the online BrowserAdapter._clean_messages single-screenshot context).
    """
    text = _sanitize_text(text)
    parts = text.split(_VISION_PLACEHOLDER)
    out = ""
    drop_pending_newline = False
    for i, seg in enumerate(parts):
        if drop_pending_newline:
            seg = seg[1:] if seg.startswith("\n") else seg
            drop_pending_newline = False
        out += seg
        if i < len(parts) - 1:  # a placeholder sat here
            url = next(url_iter)
            if keep_flags.pop(0):
                out += _IMAGE_TOKEN
                image_sink.append(url)
            else:
                if out.endswith("screenshot:\n"):
                    out = out[: -len("screenshot:\n")]
                drop_pending_newline = True
    return out


def render_record(messages: list[dict], tools: list[dict], tokenizer, *, per_turn: bool) -> tuple[list[dict], list[str]]:
    """Render a prefix via the official chat template -> ShareGPT messages + images."""
    urls = _image_urls(messages)
    keep_flags = [(i == len(urls) - 1) for i in range(len(urls))] if per_turn else [True] * len(urls)

    text = tokenizer.apply_chat_template(
        _strip_extra_fields(messages), tools=tools, tokenize=False, add_generation_prompt=False
    )
    turns = parse_turns(text)

    url_iter = iter(urls)
    images: list[str] = []
    out_messages = [
        {"role": t["role"], "content": _flatten_turn(t["content"], keep_flags, url_iter, images)}
        for t in turns
    ]
    return out_messages, images


def iter_prefixes(episode: dict, per_turn: bool):
    """Yield message prefixes to render: the full episode, or one per assistant turn."""
    messages = episode["messages"]
    if not per_turn:
        yield messages
        return
    for i, m in enumerate(messages):
        if m["role"] == "assistant":
            yield messages[: i + 1]


def main():
    parser = argparse.ArgumentParser(
        description="Render canonical OpenAI episodes into LLaMAFactory ShareGPT data via the model's official chat template."
    )
    parser.add_argument("--input-path", required=True, help="Canonical OpenAI JSONL (Stage 1 output).")
    parser.add_argument("--output-path", required=True, help="Output LLaMAFactory JSONL path.")
    parser.add_argument(
        "--model-name-or-path",
        required=True,
        help="Base model (or tokenizer dir) whose official chat template renders the per-model format.",
    )
    parser.add_argument(
        "--per-turn",
        action="store_true",
        help="Expand each episode into per-turn examples (single current screenshot; pair with mask_history:true).",
    )
    parser.add_argument("--dataset-name", default="openwebrl_sft_trajectories", help="Dataset key in dataset_info.json.")
    parser.add_argument(
        "--image-path-mode", choices=("relative", "absolute"), default="relative",
        help="How image paths are written into the output JSONL.",
    )
    parser.add_argument("--max-episodes", type=int, default=None, help="Cap episodes (for testing).")
    args = parser.parse_args()

    output_path = Path(args.output_path)
    image_dir = output_path.parent / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading tokenizer/chat template from {args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    logger.info(f"Granularity: {'per-turn' if args.per_turn else 'trajectory'}")

    n_episodes = n_records = img_counter = 0
    with open(output_path, "w", encoding="utf-8") as out_fp:
        for episode in tqdm(load_jsonl(args.input_path), desc="Rendering episodes"):
            if args.max_episodes is not None and n_episodes >= args.max_episodes:
                break
            n_episodes += 1
            tools = episode.get("tools", [])
            for prefix in iter_prefixes(episode, args.per_turn):
                messages, images = render_record(prefix, tools, tokenizer, per_turn=args.per_turn)
                image_paths = []
                for data_url in images:
                    fpath = image_dir / f"{img_counter:07d}.png"
                    _save_base64_image(data_url, str(fpath))
                    img_counter += 1
                    image_paths.append(
                        os.path.relpath(fpath, start=output_path.parent)
                        if args.image_path_mode == "relative"
                        else os.path.abspath(fpath)
                    )
                out_fp.write(json.dumps({"messages": messages, "images": image_paths}, ensure_ascii=False) + "\n")
                n_records += 1

    logger.info(f"Wrote {n_records} records from {n_episodes} episodes to {output_path}")

    # ShareGPT dataset_info with simple role/content tags. The official tool
    # format is already baked into the text; the LLaMAFactory template only
    # supplies <|im_start|> wrapping, EOS, reasoning masking, and multimodal.
    dataset_info = {
        args.dataset_name: {
            "file_name": str(output_path),
            "formatting": "sharegpt",
            "columns": {"messages": "messages", "images": "images"},
            "tags": {
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant",
                "system_tag": "system",
            },
        }
    }
    info_path = output_path.parent / "dataset_info.json"
    info_path.write_text(json.dumps(dataset_info, indent=2) + "\n", encoding="utf-8")
    logger.info(f"Wrote dataset_info: {info_path}")


if __name__ == "__main__":
    main()
