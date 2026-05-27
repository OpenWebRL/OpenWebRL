import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def _load_jsonl_record(path: Path) -> dict[str, Any]:
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                return json.loads(line)
    raise ValueError(f"No JSON object found in {path}")


def _is_success(record: dict[str, Any]) -> bool:
    reward = record.get("reward")
    if isinstance(reward, (int, float)):
        return float(reward) == 1.0

    reward_meta = (record.get("metadata") or {}).get("reward") or {}
    combined = reward_meta.get("combined")
    if isinstance(combined, (int, float)):
        return float(combined) == 1.0

    return False


def compute_eval_stats(eval_dir: Path) -> dict[str, Any]:
    result_paths = sorted(eval_dir.glob("results_task_*.jsonl"))
    abnormal_paths = sorted(eval_dir.glob("abnormal_task_*.jsonl"))

    results = []
    empty_result_files: list[str] = []
    for path in result_paths:
        try:
            results.append(_load_jsonl_record(path))
        except ValueError:
            empty_result_files.append(path.name)
    abnormals = [_load_jsonl_record(path) for path in abnormal_paths]

    successes = [record for record in results if _is_success(record)]
    finished_total = len(results)
    abnormal_total = len(abnormals)
    empty_result_total = len(empty_result_files)
    total_including_abnormal = finished_total + abnormal_total + empty_result_total

    status_counter = Counter(str(record.get("status", "UNKNOWN")) for record in results)
    abnormal_reason_counter = Counter(str(record.get("abnormal_reason", "unknown")) for record in abnormals)
    if empty_result_total:
        abnormal_reason_counter["empty_result_file"] += empty_result_total

    finished_success_rate = len(successes) / finished_total if finished_total else 0.0
    overall_success_rate = len(successes) / total_including_abnormal if total_including_abnormal else 0.0

    return {
        "eval_dir": str(eval_dir),
        "num_success": len(successes),
        "num_finished": finished_total,
        "num_abnormal": abnormal_total,
        "num_empty_result_files": empty_result_total,
        "num_total_including_abnormal": total_including_abnormal,
        "finished_success_rate": finished_success_rate,
        "overall_success_rate_including_abnormal": overall_success_rate,
        "status_counts": dict(status_counter),
        "abnormal_reason_counts": dict(abnormal_reason_counter),
        "success_task_ids": [record.get("task_id") for record in successes],
        "empty_result_files": empty_result_files,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute browser eval success rate, including abnormal exits in the denominator."
    )
    parser.add_argument("eval_dir", help="Path to an evaluation output directory")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the full stats as JSON instead of a short text summary.",
    )
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)
    stats = compute_eval_stats(eval_dir)

    if args.json:
        print(json.dumps(stats, indent=2, ensure_ascii=False))
        return

    print(f"eval_dir: {stats['eval_dir']}")
    print(f"finished tasks: {stats['num_finished']}")
    print(f"abnormal tasks: {stats['num_abnormal']}")
    print(f"empty result files: {stats['num_empty_result_files']}")
    print(f"total tasks incl abnormal: {stats['num_total_including_abnormal']}")
    print(f"successes: {stats['num_success']}")
    print(f"finished success rate: {stats['finished_success_rate']:.4f} ({stats['finished_success_rate'] * 100:.2f}%)")
    print(
        "overall success rate incl abnormal: "
        f"{stats['overall_success_rate_including_abnormal']:.4f} "
        f"({stats['overall_success_rate_including_abnormal'] * 100:.2f}%)"
    )
    print(f"status counts: {stats['status_counts']}")
    print(f"abnormal reason counts: {stats['abnormal_reason_counts']}")


if __name__ == "__main__":
    main()
