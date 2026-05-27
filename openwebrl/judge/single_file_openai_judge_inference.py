#!/usr/bin/env python3
"""Run OpenAI-compatible multimodal judge models over browser judge JSONL data.

This mirrors ``single_file_judge_inference.py`` but calls an API model such as
``gpt-4o`` or ``gpt-5`` instead of loading a local/Hugging Face model.

The input is the judge SFT/eval JSONL format used in this directory. Records can
either contain prebuilt OpenAI-style ``messages`` or raw fields such as
``intent``, ``answer``, ``action_history``, and ``screenshots``.

Unparsed judge outputs are treated as ``NOT SUCCESS`` for metrics.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Iterable

from openai import AsyncAzureOpenAI, AsyncOpenAI


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
VERDICT_RE = re.compile(r"\b(NOT\s+SUCCESS(?:FUL)?|UNSUCCESSFUL|SUCCESS(?:FUL)?)\b", re.IGNORECASE)


class InputRecord:
    def __init__(self, source: str, line_number: int | None, data: dict[str, Any]):
        self.source = source
        self.line_number = line_number
        self.data = data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an OpenAI-compatible judge model on browser judge data.")
    parser.add_argument("--model", required=True, help="API model/deployment name, e.g. gpt-4o or gpt-5.")
    parser.add_argument("--input", required=True, help="JSON, JSONL, or directory containing judge records.")
    parser.add_argument("--output-jsonl", default=None, help="Where to write per-sample predictions.")
    parser.add_argument("--summary-json", default=None, help="Optional aggregate summary path.")
    parser.add_argument("--glob", default="*.json", help="Glob used when --input is a directory.")
    parser.add_argument("--jsonl-line", type=int, default=0, help="1-based single line to run from a JSONL file.")
    parser.add_argument("--max-samples", type=int, default=0, help="Optional cap after discovery. 0 means all.")
    parser.add_argument("--sample-seed", type=int, default=0, help="Seed used when --max-samples subsamples.")
    parser.add_argument("--shard-id", type=int, default=0, help="0-based shard id for parallel evaluation.")
    parser.add_argument("--num-shards", type=int, default=1, help="Total number of parallel evaluation shards.")
    parser.add_argument("--concurrency", type=int, default=8, help="Concurrent API calls per process.")
    parser.add_argument("--timeout-secs", type=float, default=180.0, help="Timeout for one API call. <=0 disables.")
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--max-completion-tokens", type=int, default=2048, help="0 means do not send a token cap.")
    parser.add_argument("--temperature", type=float, default=None, help="Only sent when explicitly provided.")
    parser.add_argument("--print-output", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Reuse existing output-jsonl rows for this shard.")
    parser.add_argument(
        "--api-mode",
        choices=["openai", "azure_token", "azure_api_key", "token", "api_key", "served"],
        default=os.environ.get("OPENAI_JUDGE_API_MODE", "openai"),
        help=(
            "openai: OPENAI_API_KEY with optional OPENAI_BASE_URL/OPENAI_API_BASE; "
            "azure_token/token: AZURE_RESOURCE_NAME plus AZURE_TOKEN_PATH; "
            "azure_api_key/api_key: AZURE_OPENAI_ENDPOINT or OPENAI_API_BASE plus OPENAI_API_KEY; "
            "served: OpenAI-compatible endpoint via JUDGE_API_BASE."
        ),
    )
    return parser.parse_args()


def iter_jsonl(path: Path, jsonl_line: int = 0) -> Iterable[InputRecord]:
    with path.open(encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if jsonl_line and line_number != jsonl_line:
                continue
            line = line.strip()
            if line:
                yield InputRecord(str(path), line_number, json.loads(line))


def load_input_records(input_path: Path, pattern: str, jsonl_line: int) -> list[InputRecord]:
    if input_path.is_dir():
        paths = sorted(input_path.glob(pattern))
    else:
        paths = [input_path]

    records: list[InputRecord] = []
    for path in paths:
        if path.suffix == ".jsonl":
            records.extend(iter_jsonl(path, jsonl_line=jsonl_line))
            continue

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


def normalize_image_url(image_url: Any) -> str:
    if isinstance(image_url, dict):
        image_url = image_url.get("url")
    if not isinstance(image_url, str) or not image_url:
        raise ValueError(f"Invalid image_url item: {image_url!r}")
    if image_url.startswith(("http://", "https://")):
        return image_url
    if image_url.startswith("data:"):
        return image_url
    return f"data:image/png;base64,{DATA_URL_RE.sub('', image_url.strip())}"


def normalize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
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
                new_content.append({"type": "image_url", "image_url": {"url": normalize_image_url(item.get("image_url"))}})
            else:
                new_content.append(item)
        cloned["content"] = new_content
        normalized.append(cloned)
    return normalized


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
        content.append({"type": "image_url", "image_url": {"url": normalize_image_url(img)}})
    content.append({"type": "text", "text": "Your verdict:\n"})
    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": content}]


def record_messages(record: dict[str, Any]) -> list[dict[str, Any]]:
    messages = record.get("messages")
    if isinstance(messages, list):
        return normalize_messages(messages)
    return build_messages_from_raw_record(record)


def extract_verdict(text: str) -> str | None:
    if not text:
        return None
    match = VERDICT_RE.search(text)
    if not match:
        return None
    verdict = re.sub(r"\s+", " ", match.group(1).upper()).strip()
    if verdict in {"NOT SUCCESS", "NOT SUCCESSFUL", "UNSUCCESSFUL"}:
        return "NOT SUCCESS"
    if verdict in {"SUCCESS", "SUCCESSFUL"}:
        return "SUCCESS"
    return None


def verdict_for_metrics(verdict: str | None) -> str:
    return "SUCCESS" if verdict == "SUCCESS" else "NOT SUCCESS"


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
    return extract_verdict(judge_text) if isinstance(judge_text, str) else None


def normalize_served_base_url(base_url: str) -> str:
    base_url = (base_url or "").strip()
    if not base_url:
        raise ValueError("served mode requires JUDGE_API_BASE or JUDGE_API_HOST/JUDGE_API_PORT.")
    if not re.match(r"^https?://", base_url):
        base_url = f"http://{base_url}"
    base_url = base_url.rstrip("/")
    if not base_url.endswith("/v1"):
        base_url = f"{base_url}/v1"
    return base_url


def make_client(api_mode: str) -> Any:
    if api_mode == "token":
        api_mode = "azure_token"
    elif api_mode == "api_key":
        api_mode = "azure_api_key"

    if api_mode == "openai":
        api_key = os.environ.get("OPENAI_API_KEY")
        base_url = os.environ.get("OPENAI_BASE_URL") or os.environ.get("OPENAI_API_BASE")
        kwargs: dict[str, Any] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        return AsyncOpenAI(**kwargs)

    if api_mode == "azure_token":
        resource_name = os.environ.get("AZURE_RESOURCE_NAME")
        api_version = os.environ.get("AZURE_API_VERSION", os.environ.get("OPENAI_API_VERSION", "2025-02-01-preview"))
        token_path = os.environ.get("AZURE_TOKEN_PATH")
        if not token_path:
            raise ValueError("API_MODE=azure_token requires AZURE_TOKEN_PATH.")
        with open(token_path, encoding="utf-8") as f:
            token = f.read().strip()
        return AsyncAzureOpenAI(
            azure_ad_token=token,
            api_version=api_version,
            azure_endpoint=f"https://{resource_name}.openai.azure.com/",
        )

    if api_mode == "azure_api_key":
        api_key = os.environ.get("AZURE_OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
        endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT") or os.environ.get("OPENAI_API_BASE")
        api_version = os.environ.get("AZURE_API_VERSION", os.environ.get("OPENAI_API_VERSION", "2025-02-01-preview"))
        return AsyncAzureOpenAI(api_key=api_key, azure_endpoint=endpoint, api_version=api_version)

    if api_mode == "served":
        base_url = os.environ.get("JUDGE_API_BASE")
        if not base_url:
            host = os.environ.get("JUDGE_API_HOST", "").strip()
            port = os.environ.get("JUDGE_API_PORT", "").strip()
            if host and port:
                base_url = f"{host}:{port}"
            elif host:
                base_url = host
        return AsyncOpenAI(
            api_key=os.environ.get("JUDGE_API_KEY") or os.environ.get("OPENAI_API_KEY") or "EMPTY",
            base_url=normalize_served_base_url(base_url or ""),
        )

    raise ValueError(f"Unsupported api_mode: {api_mode}")


def build_request_kwargs(
    args: argparse.Namespace,
    messages: list[dict[str, Any]],
    *,
    token_param: str | None = "max_completion_tokens",
    send_temperature: bool = True,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "model": args.model,
        "messages": messages,
    }
    if args.max_completion_tokens > 0 and token_param is not None:
        kwargs[token_param] = args.max_completion_tokens
    if args.temperature is not None and send_temperature:
        kwargs["temperature"] = args.temperature
    return kwargs


def is_unsupported_parameter_error(error_text: str, parameter: str) -> bool:
    lower = error_text.lower()
    return parameter.lower() in lower and any(
        marker in lower
        for marker in ("unsupported", "unrecognized", "unknown parameter", "invalid parameter", "not supported")
    )


async def call_one(
    client: Any,
    args: argparse.Namespace,
    sem: asyncio.Semaphore,
    record: InputRecord,
) -> dict[str, Any]:
    messages = record_messages(record.data)
    saw_timeout = False
    last_error = ""
    token_param = "max_completion_tokens"
    send_temperature = True

    for attempt in range(args.max_retries):
        try:
            async with sem:
                request_kwargs = build_request_kwargs(
                    args,
                    messages,
                    token_param=token_param,
                    send_temperature=send_temperature,
                )
                coro = client.chat.completions.create(**request_kwargs)
                if args.timeout_secs and args.timeout_secs > 0:
                    response = await asyncio.wait_for(coro, timeout=args.timeout_secs)
                else:
                    response = await coro

            text = response.choices[0].message.content or ""
            usage = getattr(response, "usage", None)
            prompt_tokens = getattr(usage, "prompt_tokens", None) if usage is not None else None
            completion_tokens = getattr(usage, "completion_tokens", None) if usage is not None else None
            verdict = extract_verdict(text)
            return {
                "verdict": verdict,
                "verdict_for_metrics": verdict_for_metrics(verdict),
                "output_text": text,
                "prompt_tokens": prompt_tokens,
                "generated_tokens": completion_tokens,
                "error": None,
                "timeout": False,
            }
        except asyncio.TimeoutError as exc:
            saw_timeout = True
            last_error = f"timeout: {exc}"
        except Exception as exc:
            last_error = repr(exc)
            if token_param == "max_completion_tokens" and is_unsupported_parameter_error(last_error, "max_completion_tokens"):
                token_param = "max_tokens"
                continue
            if token_param == "max_tokens" and is_unsupported_parameter_error(last_error, "max_tokens"):
                token_param = None
                continue
            if send_temperature and is_unsupported_parameter_error(last_error, "temperature"):
                send_temperature = False
                continue

        if attempt < args.max_retries - 1:
            await asyncio.sleep(3 ** (attempt + 1))

    return {
        "verdict": None,
        "verdict_for_metrics": "NOT SUCCESS",
        "output_text": "",
        "prompt_tokens": None,
        "generated_tokens": None,
        "error": last_error or "All retries exhausted.",
        "timeout": saw_timeout,
    }


def safe_ratio(num: int, den: int) -> float:
    return 0.0 if den == 0 else num / den


def classification_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    oracle_rows = [row for row in rows if row.get("oracle_verdict") is not None]
    tp = sum(row.get("verdict_for_metrics") == "SUCCESS" and row.get("oracle_verdict") == "SUCCESS" for row in oracle_rows)
    fp = sum(row.get("verdict_for_metrics") == "SUCCESS" and row.get("oracle_verdict") == "NOT SUCCESS" for row in oracle_rows)
    fn = sum(row.get("verdict_for_metrics") == "NOT SUCCESS" and row.get("oracle_verdict") == "SUCCESS" for row in oracle_rows)
    tn = sum(row.get("verdict_for_metrics") == "NOT SUCCESS" and row.get("oracle_verdict") == "NOT SUCCESS" for row in oracle_rows)
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


def build_summary(rows: list[dict[str, Any]], args: argparse.Namespace, elapsed_seconds: float) -> dict[str, Any]:
    rows = sorted(rows, key=lambda row: (row.get("source") or "", row.get("line_number") or 0, row.get("index") or 0))
    total = len(rows)
    parsed = sum(row.get("verdict") is not None for row in rows)
    oracle_rows = [row for row in rows if row.get("oracle_verdict") is not None]
    matches = sum(row.get("matches_oracle") is True for row in rows)
    summary = {
        "model": args.model,
        "input": args.input,
        "num_samples": total,
        "parsed_count": parsed,
        "unparsed_count": total - parsed,
        "unparsed_treated_as": "NOT SUCCESS",
        "success_count": sum(row.get("verdict_for_metrics") == "SUCCESS" for row in rows),
        "not_success_count": sum(row.get("verdict_for_metrics") == "NOT SUCCESS" for row in rows),
        "oracle_count": len(oracle_rows),
        "oracle_accuracy": safe_ratio(matches, len(oracle_rows)),
        "api_mode": args.api_mode,
        "shard_id": args.shard_id,
        "num_shards": args.num_shards,
        "elapsed_seconds": round(elapsed_seconds, 3),
    }
    summary.update(classification_metrics(rows))
    return summary


def existing_rows_by_key(output_path: Path | None) -> dict[tuple[str, int | None], dict[str, Any]]:
    if output_path is None or not output_path.exists():
        return {}

    rows: dict[tuple[str, int | None], dict[str, Any]] = {}
    with output_path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("verdict") is None and row.get("output_text"):
                row["verdict"] = extract_verdict(row.get("output_text") or "")
            if "verdict_for_metrics" not in row or row.get("verdict_for_metrics") is None:
                row["verdict_for_metrics"] = verdict_for_metrics(row.get("verdict"))
            if row.get("error") or row.get("verdict") is None:
                continue
            rows[(row.get("source"), row.get("line_number"))] = row
    return rows


async def run(args: argparse.Namespace) -> list[dict[str, Any]]:
    records = load_input_records(Path(args.input), args.glob, args.jsonl_line)
    records = apply_sample_cap(records, args.max_samples, args.sample_seed)
    records = apply_shard(records, args.shard_id, args.num_shards)
    if not records:
        raise ValueError(f"No input records found under {args.input}")

    output_path = Path(args.output_jsonl) if args.output_jsonl else None
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    rows_by_key = existing_rows_by_key(output_path) if args.resume else {}
    records_to_run = [rec for rec in records if (rec.source, rec.line_number) not in rows_by_key]

    output_file = output_path.open("a" if args.resume else "w", encoding="utf-8") if output_path else None
    client = make_client(args.api_mode)
    sem = asyncio.Semaphore(max(args.concurrency, 1))
    next_index = len(rows_by_key) + 1

    async def run_and_write(rec: InputRecord) -> dict[str, Any]:
        nonlocal next_index
        result = await call_one(client, args, sem, rec)
        oracle = oracle_verdict(rec.data)
        normalized = result["verdict_for_metrics"]
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
            "verdict_for_metrics": normalized,
            "matches_oracle": (normalized == oracle) if oracle is not None else None,
            "output_text": result["output_text"],
            "prompt_tokens": result["prompt_tokens"],
            "generated_tokens": result["generated_tokens"],
            "error": result["error"],
            "timeout": result["timeout"],
        }
        next_index += 1
        if output_file is not None:
            output_file.write(json.dumps(row, ensure_ascii=False) + "\n")
            output_file.flush()
        print(
            f"[{next_index - 1}/{len(records)}] trace_id={row['trace_id']} verdict={row['verdict']} "
            f"metrics_verdict={normalized} oracle={oracle} error={row['error']}",
            flush=True,
        )
        if args.print_output and row["output_text"]:
            print(row["output_text"], flush=True)
        return row

    try:
        tasks = [asyncio.create_task(run_and_write(rec)) for rec in records_to_run]
        new_rows = await asyncio.gather(*tasks)
    finally:
        if output_file is not None:
            output_file.close()
        close = getattr(client, "close", None)
        if close is not None:
            await close()

    return list(rows_by_key.values()) + new_rows


def main() -> None:
    args = parse_args()
    start = time.time()
    rows = asyncio.run(run(args))
    summary = build_summary(rows, args, time.time() - start)
    if args.summary_json:
        summary_path = Path(args.summary_json)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), file=sys.stderr)


if __name__ == "__main__":
    main()
