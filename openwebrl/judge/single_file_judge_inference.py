#!/usr/bin/env python3
"""Standalone browser judge inference.

This file intentionally does not import any local `slime` modules. It only
depends on normal runtime packages used by the model stack: torch,
transformers, and Pillow.

Input data format
-----------------
Preferred format is one JSON object per trajectory, either in a `.json` file,
`.jsonl` file, or a directory of JSON/JSONL files. Each object should contain:

{
  "trace_id": "optional unique id",
  "task_id": "optional task id",
  "intent": "original web task",
  "messages": [
    {"role": "system", "content": "...judge system prompt..."},
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "### TASK: ..."},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}},
        {"type": "text", "text": "Your verdict:\\n"}
      ]
    }
  ],
  "judge_score": 1.0,
  "judge_text": "optional oracle judge output"
}

For convenience, if `messages` is absent the script can build the judge prompt
from raw trajectory fields:

{
  "trace_id": "optional unique id",
  "task_id": "optional task id",
  "intent": "web task instruction",
  "answer": "agent final answer",
  "action_history": "optional chronological action history",
  "screenshots": [
    "data:image/png;base64,...",
    "iVBORw0KGgoAAAANS..."
  ]
}

`screenshots` may also be named `images`; values may be raw base64 strings or
data URLs. The model output is parsed for its final `SUCCESS` or `NOT SUCCESS`
verdict.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import torch
from PIL import Image
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, AutoProcessor, AutoTokenizer
from transformers import PreTrainedTokenizerBase, ProcessorMixin


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

JUDGE_USER_PROMPT = "### TASK: {task}\n### Result Response: {answer}\n### {num} screenshots at the end: "
JUDGE_USER_PROMPT_ACTION_HISTORY = (
    "### TASK: {task}\n"
    "### Agent Action History:\n{action_history}\n\n"
    "### Result Response: {answer}\n"
    "### {num} screenshots from the trajectory are attached below with inferred step indices.\n"
)

DATA_URL_RE = re.compile(r"^data:image/[^;]+;base64,", re.IGNORECASE)
VERDICT_RE = re.compile(r"\b(NOT SUCCESS|SUCCESS)\b", re.IGNORECASE)
EXPLICIT_VERDICT_RE = re.compile(
    r"\b(?:final\s+|definitive\s+)?verdict\b[\s:*_`\-]*\b(NOT SUCCESS|SUCCESS)\b",
    re.IGNORECASE,
)
WEBJUDGE_STATUS_RE = re.compile(r"\bStatus\s*:\s*[\"']?(success|failure)[\"']?", re.IGNORECASE)
WEBJUDGE_ANY_STATUS_RE = re.compile(r"\b(success|failure)\b", re.IGNORECASE)

WEBJUDGE_SYSTEM_PROMPT = """You are an expert in evaluating the performance of a web navigation agent. The agent is designed to help a human user navigate a website to complete a task. Given the user's task, the agent's action history, key points for task completion, some potentially important web pages in the agent's trajectory and their reasons, your goal is to determine whether the agent has completed the task and achieved all requirements.

Your response must strictly follow the following evaluation criteria!
*Important Evaluation Criteria*:
1: The filtered results must be displayed correctly. If filters were not properly applied (i.e., missing selection, missing confirmation, or no visible effect in results), the task is not considered successful.
2: You must carefully check whether these snapshots and action history meet these key points. Ensure that specific filter conditions, such as "best," "highest," "cheapest," "latest," "most recent," "lowest," "closest," "highest-rated," "largest," and "newest" are correctly applied using the filter function(e.g., sort function).
3: Certain key points or requirements should be applied by the filter. Otherwise, a search with all requirements as input will be deemed a failure since it cannot guarantee that all results meet the requirements!
4: If the task requires filtering by a specific range of money, years, or the number of beds and bathrooms, the applied filter must exactly match the given requirement. Any deviation results in failure. To ensure the task is successful, the applied filter must precisely match the specified range without being too broad or too narrow.
Examples of Failure Cases:
- If the requirement is less than $50, but the applied filter is less than $25, it is a failure.
- If the requirement is $1500-$2500, but the applied filter is $2000-$2500, it is a failure.
- If the requirement is $25-$200, but the applied filter is $0-$200, it is a failure.
- If the required years are 2004-2012, but the filter applied is 2001-2012, it is a failure.
- If the required years are before 2015, but the applied filter is 2000-2014, it is a failure.
- If the task requires exactly 2 beds, but the filter applied is 2+ beds, it is a failure.
5: Some tasks require a submission action or a display of results to be considered successful.
6: If the retrieved information is invalid or empty(e.g., No match was found), but the agent has correctly performed the required action, it should still be considered successful.
7: If the current page already displays all available items, then applying a filter is not necessary. As long as the agent selects items that meet the requirements (e.g., the cheapest or lowest price), the task is still considered successful.

*IMPORTANT*
Format your response into two lines as shown below:

Thoughts: <your thoughts and reasoning process based on double-checking each key points and the evaluation criteria>
Status: "success" or "failure"
"""

WEBJUDGE_PROMPT = (
    "User Task: {task}\n\n"
    "Key Points:\n{key_points}\n\n"
    "Action History:\n{action_history}\n\n"
    "The potentially important snapshots of the webpage in the agent's trajectory "
    "and their reasons:\n{snapshot_reasons}"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a local judge model on browser judge trajectory data.")
    parser.add_argument("--model", required=True, help="Path or HF id of the judge model.")
    parser.add_argument("--input", required=True, help="JSON, JSONL, or directory containing trajectory records.")
    parser.add_argument("--output-jsonl", default=None, help="Where to write per-sample predictions.")
    parser.add_argument("--summary-json", default=None, help="Optional aggregate summary path.")
    parser.add_argument("--processor-path", default=None, help="Optional processor path. Defaults to --model.")
    parser.add_argument("--glob", default="*.json", help="Glob used when --input is a directory.")
    parser.add_argument("--jsonl-line", type=int, default=0, help="1-based single line to run from a JSONL file; 0 runs all.")
    parser.add_argument("--max-samples", type=int, default=0, help="Optional cap after discovery. 0 means all.")
    parser.add_argument("--sample-seed", type=int, default=0, help="Seed used when --max-samples subsamples.")
    parser.add_argument("--shard-id", type=int, default=0, help="0-based shard id for parallel evaluation.")
    parser.add_argument("--num-shards", type=int, default=1, help="Total number of parallel evaluation shards.")
    parser.add_argument("--batch-size", type=int, default=1, help="Per-process generation batch size.")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", choices=["auto", "float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument(
        "--prompt-template",
        choices=["native", "webjudge"],
        default="native",
        help="Use native stored messages/raw prompt fields, or rebuild inputs in WebJudge-7B format.",
    )
    parser.add_argument(
        "--webjudge-key-points-field",
        default="key_points",
        help=(
            "Record or metadata field used for WebJudge key points. If absent, the task text "
            "is used as a conservative fallback."
        ),
    )
    parser.add_argument(
        "--webjudge-status-retry",
        type=int,
        default=0,
        help="For WebJudge prompts, retry unparsed generations with a short status-only follow-up.",
    )
    parser.add_argument(
        "--webjudge-status-retry-max-new-tokens",
        type=int,
        default=64,
        help="Max new tokens for the WebJudge status-only retry.",
    )
    parser.add_argument("--print-output", action="store_true", help="Print full judge output for each sample.")
    return parser.parse_args()


def resolve_dtype(name: str):
    return {
        "auto": "auto",
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def load_processor(name_or_path: str):
    proc = AutoProcessor.from_pretrained(name_or_path, trust_remote_code=True)
    if isinstance(proc, PreTrainedTokenizerBase) or not isinstance(proc, ProcessorMixin):
        raise ValueError(f"{name_or_path} did not load as a multimodal processor.")
    if getattr(proc, "tokenizer", None) is not None:
        proc.tokenizer.padding_side = "left"
    return proc


def normalize_image_url(image_url: Any) -> str:
    if isinstance(image_url, dict):
        image_url = image_url.get("url")
    if not isinstance(image_url, str) or not image_url:
        raise ValueError(f"Invalid image_url item: {image_url!r}")
    return image_url


def decode_image(image_value: Any) -> Image.Image:
    url = normalize_image_url(image_value)
    if url.startswith(("http://", "https://", "file://")):
        raise ValueError(
            "This standalone runner only supports embedded base64/data-url images. "
            f"Got external image URL: {url[:120]}"
        )
    b64 = DATA_URL_RE.sub("", url.strip())
    image = Image.open(io.BytesIO(base64.b64decode(b64, validate=False)))
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image


def normalize_messages_for_qwen(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for message in messages:
        cloned = dict(message)
        content = cloned.get("content")
        if not isinstance(content, list):
            normalized.append(cloned)
            continue
        new_content = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "image_url":
                new_content.append({"type": "image_url", "image_url": normalize_image_url(item.get("image_url"))})
            else:
                new_content.append(item)
        cloned["content"] = new_content
        normalized.append(cloned)
    return normalized


def collect_images(messages: list[dict[str, Any]]) -> list[Image.Image]:
    images = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if isinstance(item, dict) and item.get("type") == "image_url":
                images.append(decode_image(item.get("image_url")))
    return images


def messages_to_text_only(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    converted = []
    for message in messages:
        cloned = dict(message)
        content = cloned.get("content")
        if not isinstance(content, list):
            converted.append(cloned)
            continue
        parts = []
        image_count = 0
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "text":
                parts.append(item.get("text", ""))
            elif item.get("type") == "image_url":
                image_count += 1
        if image_count:
            parts.append(f"[{image_count} image(s) attached]")
        cloned["content"] = "\n".join(part for part in parts if part)
        converted.append(cloned)
    return converted


def iter_message_content_items(messages: list[dict[str, Any]]) -> Iterable[dict[str, Any]]:
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if isinstance(item, dict):
                yield item


def extract_text_between(text: str, start_marker: str, end_marker: str) -> str:
    start = text.find(start_marker)
    if start < 0:
        return ""
    start += len(start_marker)
    end = text.find(end_marker, start)
    if end < 0:
        return text[start:].strip()
    return text[start:end].strip()


def record_text_blocks(record: dict[str, Any]) -> list[str]:
    messages = record.get("messages")
    if not isinstance(messages, list):
        return []
    texts = []
    for item in iter_message_content_items(messages):
        if item.get("type") == "text" and isinstance(item.get("text"), str):
            texts.append(item["text"])
    return texts


def record_image_items(record: dict[str, Any]) -> list[dict[str, Any]]:
    messages = record.get("messages")
    items = []
    if isinstance(messages, list):
        for item in iter_message_content_items(messages):
            if item.get("type") == "image_url":
                items.append({"type": "image_url", "image_url": item.get("image_url")})
    if items:
        return items

    screenshots = record.get("screenshots", record.get("images", []))
    if not isinstance(screenshots, list):
        return []
    for img in screenshots:
        if not isinstance(img, str):
            continue
        url = img if img.startswith("data:") else f"data:image/png;base64,{img}"
        items.append({"type": "image_url", "image_url": {"url": url}})
    return items


def infer_task(record: dict[str, Any]) -> str:
    metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
    for value in (
        record.get("intent"),
        record.get("task"),
        record.get("instruction"),
        metadata.get("intent"),
        metadata.get("task"),
        metadata.get("instruction"),
    ):
        if isinstance(value, str) and value.strip():
            return value.strip()

    for text in record_text_blocks(record):
        task = extract_text_between(text, "### TASK:", "### Agent Action History:")
        if task:
            return task
        task = extract_text_between(text, "### TASK:", "### Result Response:")
        if task:
            return task
    raise ValueError("Could not infer task/intent for WebJudge prompt.")


def infer_action_history(record: dict[str, Any]) -> str:
    metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
    for value in (
        record.get("action_history"),
        record.get("last_actions"),
        metadata.get("action_history"),
        metadata.get("last_actions"),
    ):
        if isinstance(value, list):
            return "\n".join(f"{idx}. {item}" for idx, item in enumerate(value, start=1))
        if isinstance(value, str) and value.strip():
            return value.strip()

    for text in record_text_blocks(record):
        action_history = extract_text_between(text, "### Agent Action History:", "### Result Response:")
        if action_history:
            return action_history
    return "No action history is available."


def infer_key_points(record: dict[str, Any], field_name: str) -> str:
    metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
    for value in (record.get(field_name), metadata.get(field_name), record.get("key_points"), metadata.get("key_points")):
        if isinstance(value, list) and value:
            return "\n".join(f"{idx}. {item}" for idx, item in enumerate(value, start=1))
        if isinstance(value, str) and value.strip():
            return value.strip()
    task = infer_task(record)
    return f"1. Complete every explicit requirement in the task: {task}"


def build_webjudge_messages(record: dict[str, Any], key_points_field: str) -> list[dict[str, Any]]:
    task = infer_task(record)
    action_history = infer_action_history(record)
    key_points = infer_key_points(record, key_points_field)

    image_items = record_image_items(record)
    if image_items:
        snapshot_reasons = "\n".join(
            f"{idx}. Screenshot {idx} is included as visual evidence from the trajectory."
            for idx in range(1, len(image_items) + 1)
        )
    else:
        snapshot_reasons = "No screenshots are available."

    text = WEBJUDGE_PROMPT.format(
        task=task,
        key_points=key_points,
        action_history=action_history,
        snapshot_reasons=snapshot_reasons,
    )
    return [
        {"role": "system", "content": WEBJUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": [{"type": "text", "text": text}] + image_items},
    ]


def build_messages_from_raw_record(record: dict[str, Any]) -> list[dict[str, Any]]:
    task = record.get("intent") or record.get("task") or record.get("instruction")
    answer = record.get("answer") or record.get("response") or record.get("final_answer")
    if not isinstance(task, str) or not task.strip():
        raise ValueError("Record without `messages` must contain `intent`, `task`, or `instruction`.")
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("Record without `messages` must contain `answer`, `response`, or `final_answer`.")

    screenshots = record.get("screenshots", record.get("images", []))
    if screenshots is None:
        screenshots = []
    if not isinstance(screenshots, list):
        raise ValueError("`screenshots`/`images` must be a list of base64 strings or data URLs.")

    action_history = record.get("action_history")
    use_action_history = isinstance(action_history, str) and action_history.strip()
    if use_action_history:
        system_prompt = JUDGE_SYSTEM_PROMPT_ACTION_HISTORY
        user_text = JUDGE_USER_PROMPT_ACTION_HISTORY.format(
            task=task,
            action_history=action_history,
            answer=answer,
            num=len(screenshots),
        )
    else:
        system_prompt = JUDGE_SYSTEM_PROMPT
        user_text = JUDGE_USER_PROMPT.format(task=task, answer=answer, num=len(screenshots))

    content: list[dict[str, Any]] = [{"type": "text", "text": user_text}]
    for idx, img in enumerate(screenshots, start=1):
        label = f"Screenshot {idx} (inferred step {idx})." if use_action_history else f"Screenshot {idx}."
        content.append({"type": "text", "text": label})
        url = img if isinstance(img, str) and img.startswith("data:") else f"data:image/png;base64,{img}"
        content.append({"type": "image_url", "image_url": {"url": url}})
    content.append({"type": "text", "text": "Your verdict:\n"})
    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": content}]


def record_messages(record: dict[str, Any]) -> list[dict[str, Any]]:
    messages = record.get("messages")
    if isinstance(messages, list):
        return messages
    return build_messages_from_raw_record(record)


def extract_native_verdict(text: str) -> str | None:
    if not text:
        return None

    explicit_matches = list(EXPLICIT_VERDICT_RE.finditer(text))
    if explicit_matches:
        return explicit_matches[-1].group(1).upper()

    matches = list(VERDICT_RE.finditer(text))
    if not matches:
        return None
    return matches[-1].group(1).upper()


def extract_webjudge_verdict(text: str) -> str | None:
    if not text:
        return None
    match = WEBJUDGE_STATUS_RE.search(text)
    if match is None:
        matches = WEBJUDGE_ANY_STATUS_RE.findall(text)
        if not matches:
            return None
        status = matches[-1]
    else:
        status = match.group(1)
    return "SUCCESS" if status.lower() == "success" else "NOT SUCCESS"


def extract_verdict(text: str, prompt_template: str = "native") -> str | None:
    if prompt_template == "webjudge":
        return extract_webjudge_verdict(text)
    return extract_native_verdict(text)


def oracle_verdict(record: dict[str, Any]) -> str | None:
    score = record.get("judge_score")
    metadata = record.get("metadata")
    if score is None and isinstance(metadata, dict):
        score = metadata.get("judge_score")
    if score == 1 or score == 1.0:
        return "SUCCESS"
    if score == 0 or score == 0.0:
        return "NOT SUCCESS"
    judge_text = record.get("judge_text") or record.get("label")
    return extract_native_verdict(judge_text) if isinstance(judge_text, str) else None


class InputRecord:
    def __init__(self, source: str, line_number: int | None, data: dict[str, Any]):
        self.source = source
        self.line_number = line_number
        self.data = data


def iter_jsonl(path: Path, jsonl_line: int = 0) -> Iterable[InputRecord]:
    with path.open(encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if jsonl_line and line_number != jsonl_line:
                continue
            line = line.strip()
            if not line:
                continue
            yield InputRecord(str(path), line_number, json.loads(line))


def load_input_records(input_path: Path, pattern: str, jsonl_line: int) -> list[InputRecord]:
    if input_path.is_dir():
        paths = sorted(input_path.glob(pattern))
    else:
        paths = [input_path]
    records = []
    for path in paths:
        if path.suffix == ".jsonl":
            records.extend(iter_jsonl(path, jsonl_line=jsonl_line))
        else:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                for idx, item in enumerate(data, start=1):
                    records.append(InputRecord(str(path), idx, item))
            elif isinstance(data, dict):
                records.append(InputRecord(str(path), None, data))
            else:
                raise ValueError(f"Unsupported JSON root in {path}: {type(data).__name__}")
    return records


def apply_sample_cap(records: list[InputRecord], max_samples: int, seed: int) -> list[InputRecord]:
    if max_samples <= 0 or len(records) <= max_samples:
        return records
    rng = random.Random(seed)
    return sorted(rng.sample(records, max_samples), key=lambda rec: (rec.source, rec.line_number or 0))


def apply_shard(records: list[InputRecord], shard_id: int, num_shards: int) -> list[InputRecord]:
    if num_shards <= 1:
        return records
    if shard_id < 0 or shard_id >= num_shards:
        raise ValueError(f"Invalid shard config: shard_id={shard_id}, num_shards={num_shards}")
    return [record for idx, record in enumerate(records) if idx % num_shards == shard_id]


def iter_batches(records: list[InputRecord], batch_size: int) -> Iterable[list[InputRecord]]:
    if batch_size <= 0:
        raise ValueError(f"--batch-size must be positive, got {batch_size}")
    for start in range(0, len(records), batch_size):
        yield records[start : start + batch_size]


class JudgeRunner:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.model_path = args.model
        self.dtype = resolve_dtype(args.torch_dtype)
        self.config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
        self.is_multimodal = self.config.model_type in {"qwen3_vl", "qwen3_vl_moe", "qwen2_5_vl", "qwen2_vl"}
        self.processor = None
        self.tokenizer = None
        if self.is_multimodal:
            self.processor = load_processor(args.processor_path or args.model)
            self.tokenizer = self.processor.tokenizer
            self.model = AutoModelForImageTextToText.from_pretrained(
                args.model,
                trust_remote_code=True,
                dtype=self.dtype,
                device_map=args.device_map,
            )
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(args.processor_path or args.model, trust_remote_code=True)
            self.tokenizer.padding_side = "left"
            self.model = AutoModelForCausalLM.from_pretrained(
                args.model,
                trust_remote_code=True,
                dtype=self.dtype,
                device_map=args.device_map,
            )
        self.model.eval()

    def prepare_inputs(self, messages: list[dict[str, Any]]) -> tuple[str, dict[str, Any]]:
        prompts, model_inputs = self.prepare_batch_inputs([messages])
        return prompts[0], model_inputs

    def prepare_batch_inputs(self, batch_messages: list[list[dict[str, Any]]]) -> tuple[list[str], dict[str, Any]]:
        if self.is_multimodal:
            normalized_messages = [normalize_messages_for_qwen(messages) for messages in batch_messages]
            prompts = [
                self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                for messages in normalized_messages
            ]
            batch_images = [collect_images(messages) for messages in normalized_messages]
            kwargs = {}
            if any(batch_images):
                kwargs["images"] = batch_images
            try:
                model_inputs = self.processor(text=prompts, return_tensors="pt", padding=True, **kwargs)
            except Exception:
                if "images" not in kwargs:
                    raise
                flat_images = [image for images in batch_images for image in images]
                model_inputs = self.processor(text=prompts, return_tensors="pt", padding=True, images=flat_images)
        else:
            text_messages = [messages_to_text_only(messages) for messages in batch_messages]
            prompts = [
                self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                for messages in text_messages
            ]
            model_inputs = self.tokenizer(prompts, return_tensors="pt", padding=True)
        return prompts, model_inputs

    def generate(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        return self.generate_batch([messages])[0]

    def generate_batch(self, batch_messages: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
        return self._generate_batch(batch_messages, max_new_tokens=self.args.max_new_tokens)

    def _generate_batch(self, batch_messages: list[list[dict[str, Any]]], max_new_tokens: int) -> list[dict[str, Any]]:
        prompts, model_inputs = self.prepare_batch_inputs(batch_messages)
        device = next(self.model.parameters()).device
        model_inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in model_inputs.items()}
        generate_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": self.args.temperature > 0,
        }
        if self.args.temperature > 0:
            generate_kwargs["temperature"] = self.args.temperature
            generate_kwargs["top_p"] = self.args.top_p
        with torch.inference_mode():
            output_ids = self.model.generate(**model_inputs, **generate_kwargs)
        prompt_len = model_inputs["input_ids"].shape[1]
        results = []
        attention_mask = model_inputs.get("attention_mask")
        pad_token_id = self.tokenizer.pad_token_id
        for row_idx in range(output_ids.shape[0]):
            generated_ids = output_ids[row_idx][prompt_len:]
            output_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            if attention_mask is not None:
                prompt_tokens = int(attention_mask[row_idx].sum().item())
            else:
                prompt_tokens = int(prompt_len)
            if pad_token_id is None:
                generated_tokens = int(generated_ids.numel())
            else:
                generated_tokens = int((generated_ids != pad_token_id).sum().item())
            results.append(
                {
                    "output_text": output_text,
                    "verdict": extract_verdict(output_text, self.args.prompt_template),
                    "prompt_tokens": prompt_tokens,
                    "generated_tokens": generated_tokens,
                    "prompt_text": prompts[row_idx],
                }
            )
        return results

    def retry_webjudge_status(self, messages: list[dict[str, Any]], output_text: str) -> dict[str, Any]:
        retry_messages = list(messages) + [
            {"role": "assistant", "content": output_text},
            {
                "role": "user",
                "content": (
                    "Your previous answer did not include a valid final status line. "
                    "Based only on your previous reasoning and the provided evidence, "
                    'output exactly one line in this format: Status: "success" or "failure"'
                ),
            },
        ]
        return self._generate_batch(
            [retry_messages],
            max_new_tokens=self.args.webjudge_status_retry_max_new_tokens,
        )[0]


def safe_ratio(num: int, den: int) -> float:
    return 0.0 if den == 0 else num / den


def classification_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    oracle_rows = [row for row in rows if row.get("oracle_verdict") is not None]
    tp = sum(row.get("verdict") == "SUCCESS" and row.get("oracle_verdict") == "SUCCESS" for row in oracle_rows)
    fp = sum(row.get("verdict") == "SUCCESS" and row.get("oracle_verdict") == "NOT SUCCESS" for row in oracle_rows)
    fn = sum(row.get("verdict") == "NOT SUCCESS" and row.get("oracle_verdict") == "SUCCESS" for row in oracle_rows)
    tn = sum(row.get("verdict") == "NOT SUCCESS" and row.get("oracle_verdict") == "NOT SUCCESS" for row in oracle_rows)
    precision = safe_ratio(tp, tp + fp)
    recall = safe_ratio(tp, tp + fn)
    f1 = safe_ratio(2 * precision * recall, precision + recall)
    return {
        "positive_label": "SUCCESS",
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "true_negative": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def build_summary(rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    total = len(rows)
    parsed = sum(row["verdict"] is not None for row in rows)
    oracle_rows = [row for row in rows if row.get("oracle_verdict") is not None]
    matches = sum(row.get("matches_oracle") is True for row in rows)
    summary = {
        "model": args.model,
        "input": args.input,
        "num_samples": total,
        "parsed_count": parsed,
        "unparsed_count": total - parsed,
        "success_count": sum(row["verdict"] == "SUCCESS" for row in rows),
        "not_success_count": sum(row["verdict"] == "NOT SUCCESS" for row in rows),
        "oracle_count": len(oracle_rows),
        "oracle_accuracy": safe_ratio(matches, len(oracle_rows)),
        "shard_id": args.shard_id,
        "num_shards": args.num_shards,
    }
    summary.update(classification_metrics(rows))
    return summary


def main() -> None:
    args = parse_args()
    records = load_input_records(Path(args.input), args.glob, args.jsonl_line)
    records = apply_sample_cap(records, args.max_samples, args.sample_seed)
    records = apply_shard(records, args.shard_id, args.num_shards)
    if not records:
        raise ValueError(f"No input records found under {args.input}")

    output_file = None
    if args.output_jsonl:
        output_path = Path(args.output_jsonl)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_file = output_path.open("w", encoding="utf-8")

    rows: list[dict[str, Any]] = []
    start = time.time()
    try:
        runner = JudgeRunner(args)
        next_index = 1
        for batch_records in iter_batches(records, args.batch_size):
            if args.prompt_template == "webjudge":
                batch_messages = [build_webjudge_messages(rec.data, args.webjudge_key_points_field) for rec in batch_records]
            else:
                batch_messages = [record_messages(rec.data) for rec in batch_records]
            batch_results = runner.generate_batch(batch_messages)
            for rec, messages, result in zip(batch_records, batch_messages, batch_results, strict=True):
                retry_result = None
                if (
                    args.prompt_template == "webjudge"
                    and args.webjudge_status_retry > 0
                    and result["verdict"] is None
                ):
                    retry_result = runner.retry_webjudge_status(messages, result["output_text"])
                    result["verdict"] = retry_result["verdict"]
                oracle = oracle_verdict(rec.data)
                row = {
                    "index": next_index,
                    "source": rec.source,
                    "line_number": rec.line_number,
                    "trace_id": rec.data.get("trace_id") or (rec.data.get("metadata") or {}).get("trace_id"),
                    "task_id": rec.data.get("task_id") or (rec.data.get("metadata") or {}).get("task_id"),
                    "intent": rec.data.get("intent")
                    or rec.data.get("task")
                    or rec.data.get("instruction")
                    or (rec.data.get("metadata") or {}).get("intent"),
                    "oracle_verdict": oracle,
                    "verdict": result["verdict"],
                    "matches_oracle": (result["verdict"] == oracle) if oracle is not None else None,
                    "output_text": result["output_text"],
                    "prompt_tokens": result["prompt_tokens"],
                    "generated_tokens": result["generated_tokens"],
                }
                if retry_result is not None:
                    row["status_retry_output_text"] = retry_result["output_text"]
                    row["status_retry_verdict"] = retry_result["verdict"]
                    row["status_retry_generated_tokens"] = retry_result["generated_tokens"]
                rows.append(row)
                if output_file is not None:
                    output_file.write(json.dumps(row, ensure_ascii=False) + "\n")
                    output_file.flush()
                print(
                    f"[{next_index}/{len(records)}] trace_id={row['trace_id']} verdict={row['verdict']} "
                    f"oracle={oracle} generated_tokens={row['generated_tokens']}",
                    flush=True,
                )
                if args.print_output:
                    print(result["output_text"], flush=True)
                next_index += 1
    finally:
        if output_file is not None:
            output_file.close()

    summary = build_summary(rows, args)
    summary["elapsed_seconds"] = round(time.time() - start, 3)
    if args.summary_json:
        summary_path = Path(args.summary_json)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), file=sys.stderr)


if __name__ == "__main__":
    main()
