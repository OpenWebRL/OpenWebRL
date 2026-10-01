"""
Turn the om2w pilot (150 tasks x 5 reps of the SFT model on Browserbase) into the
fixed 48-task RL sweep set.

Reads results_task_<base>__r<k>.jsonl / abnormal_task_*.jsonl from the eval dir,
groups by base task id, keeps tasks with 1-4 successes out of 5 (abnormal or
missing reps count as failures), samples 48 with a fixed seed, and writes the
matching rows of the source benchmark JSONL plus a slime parquet.

  python openwebrl/data/pilot/select_train48.py --eval-dir outputs/eval/.../eval_* \
      --out-dir openwebrl/data/sweep
"""
import argparse
import json
import random
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
SOURCE = REPO / "openwebrl/data/eval/online-mind2web.jsonl"


def _first_record(path: Path):
    for line in path.open():
        line = line.strip()
        if line:
            return json.loads(line)
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--eval-dir", required=True)
    p.add_argument("--out-dir", default=str(REPO / "openwebrl/data/sweep"))
    p.add_argument("--n", type=int, default=48)
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--seed", type=int, default=20260904)
    args = p.parse_args()

    eval_dir = Path(args.eval_dir)
    outcomes = defaultdict(dict)  # base -> {rep: 0/1}
    for path in eval_dir.glob("results_task_*.jsonl"):
        rec = _first_record(path)
        if rec is None:
            continue
        base, _, rep = str(rec["task_id"]).rpartition("__r")
        outcomes[base][int(rep)] = 1 if float(rec.get("reward") or 0.0) == 1.0 else 0
    for path in eval_dir.glob("abnormal_task_*.jsonl"):
        rec = _first_record(path)
        if rec is None:
            continue
        base, _, rep = str(rec["task_id"]).rpartition("__r")
        outcomes[base].setdefault(int(rep), 0)

    per_task = {b: sum(o.get(r, 0) for r in range(args.reps)) for b, o in outcomes.items()}
    observed = {b: len(o) for b, o in outcomes.items()}
    hist = defaultdict(int)
    for k in per_task.values():
        hist[k] += 1
    print(f"tasks seen: {len(per_task)}; reps observed: min={min(observed.values())} max={max(observed.values())}")
    print("successes/5 histogram:", dict(sorted(hist.items())))

    eligible = sorted(b for b, k in per_task.items() if 1 <= k <= args.reps - 1)
    print(f"eligible (1-{args.reps - 1}/{args.reps}): {len(eligible)}")
    if len(eligible) < args.n:
        sys.exit(f"only {len(eligible)} eligible tasks, need {args.n}")
    rng = random.Random(args.seed)
    chosen = set(rng.sample(eligible, args.n))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = out_dir / f"om2w_train{args.n}.jsonl"
    n_written = 0
    with out_jsonl.open("w") as fo:
        for line in SOURCE.open():
            if not line.strip():
                continue
            row = json.loads(line)
            if row["task_id"] in chosen:
                row["metadata"]["pilot_successes"] = per_task[row["task_id"]]
                fo.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_written += 1
    assert n_written == args.n, n_written
    (out_dir / f"om2w_train{args.n}_pilot_stats.json").write_text(
        json.dumps({"per_task_successes": per_task, "chosen": sorted(chosen), "seed": args.seed}, indent=1))
    mean_rate = sum(per_task[b] for b in chosen) / (args.n * args.reps)
    print(f"chosen {args.n}: mean pilot success {mean_rate:.3f} -> {out_jsonl}")

    out_parquet = out_dir / f"om2w_train{args.n}.parquet"
    subprocess.run([sys.executable, str(REPO / "openwebrl/data/convert_benchmark_jsonl_to_parquet.py"),
                    "--input", str(out_jsonl), "--output", str(out_parquet),
                    "--task-prefix", "om2w"], check=True)
    print(f"wrote {out_parquet}")


if __name__ == "__main__":
    main()
