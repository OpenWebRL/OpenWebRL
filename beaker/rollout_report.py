"""Index page for the molmoweb-harness RL rollouts of one run.

    python beaker/rollout_report.py <run_name | rollouts dir> [--out index.html]

Reads every ``<save>/rollouts/v<version>/<episode>/`` (episode.json written by
openwebrl/generate_molmoweb.py, reward.json by openwebrl/reward_browser.py, trajectory.html
by molmoweb's log_episode) and writes ``<save>/rollouts/index.html``: a per-policy-version
summary table (episodes, completed, judged success, mean steps) and one row per episode
linking to its trajectory viewer. Rendered with molmoweb's own html helpers.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import defaultdict

DEFAULT_ROOT = "/weka/oe-training-default/new_peters/OpenWebRL/outputs/rl_sweep"
MOLMOWEB_ROOT = os.environ.get("MOLMOWEB_ROOT", "/weka/oe-training-default/new_peters/molmoweb-rl")


def _load(path: str) -> dict:
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def collect(rollouts_dir: str) -> list[dict]:
    rows = []
    for ep_json in sorted(glob.glob(os.path.join(rollouts_dir, "v*", "*", "episode.json"))):
        ep_dir = os.path.dirname(ep_json)
        ep = _load(ep_json)
        reward = _load(os.path.join(ep_dir, "reward.json"))
        rows.append({
            "version": os.path.basename(os.path.dirname(ep_dir)),
            "episode": os.path.basename(ep_dir),
            "rel": os.path.relpath(ep_dir, rollouts_dir),
            "task_id": ep.get("task_id", ""),
            "intent": ep.get("intent", ""),
            "status": ep.get("status", ""),
            "terminate_reason": ep.get("terminate_reason", ""),
            "steps": ep.get("total_steps", 0),
            "wall_secs": ep.get("traj_wall_secs"),
            "final_answer": ep.get("final_answer", ""),
            "reward": reward.get("combined"),
            "judge": reward.get("judge"),
            "judge_text": reward.get("judge_text", ""),
            "has_html": os.path.exists(os.path.join(ep_dir, "trajectory.html")),
        })
    return rows


def summarize(rows: list[dict]) -> list[dict]:
    by_v: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_v[r["version"]].append(r)
    out = []
    for v in sorted(by_v):
        rs = by_v[v]
        judged = [r for r in rs if r["reward"] is not None]
        completed = [r for r in rs if r["status"] == "completed"]
        succ = [r for r in judged if r["judge"] == 1.0]
        out.append({
            "version": v,
            "episodes": len(rs),
            "completed": f"{len(completed)} ({100 * len(completed) / max(1, len(rs)):.0f}%)",
            "judged": len(judged),
            "success": f"{len(succ)} ({100 * len(succ) / max(1, len(judged)):.0f}% of judged)",
            "mean steps": f"{sum(r['steps'] for r in rs) / max(1, len(rs)):.1f}",
            "mean wall s": f"{sum(r['wall_secs'] or 0 for r in rs) / max(1, len(rs)):.0f}",
        })
    return out


def render(rows: list[dict], out_path: str, title: str) -> None:
    sys.path.insert(0, MOLMOWEB_ROOT)
    import fasthtml.common as ft
    from utils.vis_utils.html import create_table, save_html

    summary = summarize(rows)
    elements = [
        ft.H1(title),
        ft.P(f"{len(rows)} episodes, {len(summary)} policy versions"),
        ft.H3("Per policy version"),
        create_table(columns=["version", "episodes", "completed", "judged", "success", "mean steps", "mean wall s"], rows=summary),
        ft.Hr(),
        ft.H3("Episodes"),
    ]
    ep_rows = []
    for r in rows:
        link = ft.A(r["episode"], href=f"{r['rel']}/trajectory.html") if r["has_html"] else r["episode"]
        reward = "" if r["reward"] is None else f"{r['reward']:.0f}"
        ep_rows.append({
            "version": r["version"], "episode": link, "task": r["intent"][:160],
            "status": r["status"], "reason": r["terminate_reason"][:60], "steps": r["steps"],
            "reward": reward, "answer": (r["final_answer"] or "")[:120],
            "judge": ft.Details(ft.Summary("judge"), ft.P(r["judge_text"][:2000])) if r["judge_text"] else "",
        })
    elements.append(create_table(
        columns=["version", "episode", "task", "status", "reason", "steps", "reward", "answer", "judge"], rows=ep_rows
    ))
    save_html(elements, out_path)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("run", help="run name under outputs/rl_sweep, or a path to a rollouts dir")
    p.add_argument("--out", default="")
    a = p.parse_args()
    rollouts_dir = a.run if os.path.isdir(a.run) else os.path.join(DEFAULT_ROOT, a.run, "rollouts")
    rows = collect(rollouts_dir)
    if not rows:
        raise SystemExit(f"no episodes under {rollouts_dir}")
    out = a.out or os.path.join(rollouts_dir, "index.html")
    render(rows, out, title=f"RL rollouts: {os.path.basename(os.path.dirname(rollouts_dir))}")
    for s in summarize(rows):
        print(s)
    print("wrote", out)


if __name__ == "__main__":
    main()
