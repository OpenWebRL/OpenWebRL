"""
Convert browser benchmark JSONL data into the parquet format expected by slime.

Input JSONL rows are expected to contain fields like:
  - task_name
  - website
  - task_id

Output parquet rows use the schema expected by slime:
  - prompt:   [{"role": "user", "content": <task>}]
  - metadata: {"task_id": ..., "start_url": ..., "task": ...}
"""

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import pandas as pd


HOST_REWRITES = {
    "www.books.google.com": "books.google.com",
    "www.news.google.com": "news.google.com",
    "www.patents.google.com": "patents.google.com",
    "www.scholar.google.com": "scholar.google.com",
    "www.support.google.com": "support.google.com",
    "www.translate.google.com": "translate.google.com",
}


def load_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no}: {exc}") from exc
    return rows


def load_host_blacklist(path: str | None) -> set[str]:
    if not path:
        return set()

    hosts = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip().lower()
            if not line or line.startswith("#"):
                continue
            hosts.add(line)
    return hosts


def load_excluded_task_ids(path: str | None) -> set[str]:
    if not path:
        return set()

    task_ids = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(task_ids, list):
        raise ValueError(f"Excluded task file must contain a list, got {type(task_ids).__name__}: {path}")
    return {str(task_id) for task_id in task_ids}


def normalize_task_id(raw_task_id: object, task_prefix: str) -> str:
    return f"{task_prefix}/{raw_task_id}"


def normalize_start_url(url: str) -> str:
    if not url:
        return url

    parts = urlsplit(url)
    host = parts.netloc.lower()
    normalized_host = HOST_REWRITES.get(host, host)

    # Conservative cleanup for malformed Google subservice hosts like
    # www.books.google.com -> books.google.com.
    if normalized_host == host and host.startswith("www.") and host.endswith(".google.com") and host.count(".") >= 3:
        normalized_host = host[4:]

    if normalized_host == host:
        return url

    return urlunsplit((parts.scheme, normalized_host, parts.path, parts.query, parts.fragment))


def find_suspicious_hosts(records: list[dict]) -> list[tuple[str, int, str]]:
    hosts = Counter()
    for record in records:
        host = urlsplit(record.get("website", "")).netloc.lower()
        if host:
            hosts[host] += 1

    suspicious = []
    for host, count in sorted(hosts.items()):
        if host in HOST_REWRITES:
            suspicious.append((host, count, HOST_REWRITES[host]))
        elif host.startswith("www.") and host.endswith(".google.com") and host.count(".") >= 3:
            suspicious.append((host, count, host[4:]))
    return suspicious


def filter_records_by_host_blacklist(records: list[dict], host_blacklist: set[str]) -> tuple[list[dict], list[dict]]:
    if not host_blacklist:
        return records, []

    kept = []
    dropped = []
    for record in records:
        normalized_url = normalize_start_url(record.get("website", ""))
        host = urlsplit(normalized_url).netloc.lower()
        if host in host_blacklist:
            dropped.append(record)
        else:
            kept.append(record)
    return kept, dropped


def filter_records_by_excluded_task_ids(
    records: list[dict],
    task_prefix: str,
    excluded_task_ids: set[str],
) -> tuple[list[dict], list[dict]]:
    if not excluded_task_ids:
        return records, []

    kept = []
    dropped = []
    for record in records:
        normalized_task_id = normalize_task_id(record.get("task_id", ""), task_prefix)
        if normalized_task_id in excluded_task_ids:
            dropped.append(record)
        else:
            kept.append(record)
    return kept, dropped


def record_to_row(record: dict, task_prefix: str) -> dict:
    task = record.get("task_name", "")
    start_url = normalize_start_url(record.get("website", ""))
    task_id = normalize_task_id(record.get("task_id", ""), task_prefix)
    return {
        "prompt": [{"role": "user", "content": task}],
        "metadata": {
            "task_id": task_id,
            "start_url": start_url,
            "task": task,
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Convert benchmark JSONL into slime browser parquet format.")
    parser.add_argument("--input", required=True, help="Input JSONL path.")
    parser.add_argument("--output", required=True, help="Output parquet path.")
    parser.add_argument(
        "--task-prefix",
        default="webvoyager",
        help="Prefix used in metadata.task_id, e.g. webvoyager -> webvoyager/123.",
    )
    parser.add_argument(
        "--host-blacklist",
        default=None,
        help="Optional text file with one host per line to exclude after URL normalization.",
    )
    parser.add_argument(
        "--exclude-task-ids",
        default=None,
        help="Optional JSON file containing a list of task_ids to exclude, e.g. webvoyager/84.",
    )
    args = parser.parse_args()

    records = load_jsonl(args.input)
    suspicious_hosts = find_suspicious_hosts(records)
    if suspicious_hosts:
        print("Suspicious hosts detected and normalized when possible:")
        for host, count, normalized in suspicious_hosts:
            print(f"  {host} -> {normalized} ({count} rows)")
    host_blacklist = load_host_blacklist(args.host_blacklist)
    if host_blacklist:
        records, dropped_records = filter_records_by_host_blacklist(records, host_blacklist)
        print(f"Excluded {len(dropped_records)} rows using host blacklist: {sorted(host_blacklist)}")
    excluded_task_ids = load_excluded_task_ids(args.exclude_task_ids) if args.exclude_task_ids else set()
    if excluded_task_ids:
        records, dropped_records = filter_records_by_excluded_task_ids(records, args.task_prefix, excluded_task_ids)
        print(f"Excluded {len(dropped_records)} rows using excluded task ids from {args.exclude_task_ids}")
    rows = [record_to_row(record, args.task_prefix) for record in records]
    df = pd.DataFrame(rows)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df.to_parquet(args.output, index=False)
    print(f"Saved {len(df)} rows to {args.output}")


if __name__ == "__main__":
    main()
