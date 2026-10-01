"""
Launch browser-benchmark evaluation on Beaker via Gantry.

One 8-GPU node per (checkpoint, benchmark) pair. The sglang/Megatron stack comes
from an existing Beaker image; the OpenWebRL repo is read from the Weka mount.
Container entry point is beaker/run_eval.sh.

Usage:
  # the authors' released post-RL checkpoint, all three benchmarks
  python beaker/launch_eval.py --ckpt OpenWebRL/OpenWebRL-4B --bench all

  # a local checkpoint, one benchmark
  python beaker/launch_eval.py --ckpt /weka/.../OpenWebRL-4B-SFT --bench om2w

  # full WebVoyager through the paper's Browser-Use stealth browsers, 20 at a time
  python beaker/launch_eval.py --ckpt OpenWebRL/OpenWebRL-4B --bench webvoyager \
      --env-mode browser-use --n-parallel 20 --run-tag wv_browseruse

  # a 12-task subset through Browserbase stealth browsers
  python beaker/launch_eval.py --ckpt OpenWebRL/OpenWebRL-4B --bench webvoyager \
      --env-mode browserbase --task-indices 227,239,469 --run-tag wv_blocked_bb

Prereqs:
  - `beaker account login`
  - Secrets in the workspace: PS_OPENAI_API_KEY, PS_HF_TOKEN
    (browserbase mode also needs PS_BROWSERBASE_API_KEY, PS_BROWSERBASE_PROJECT_ID)
"""
import argparse
import os
import re
import subprocess

OPENWEBRL_ROOT = "/weka/oe-training-default/new_peters/OpenWebRL"
BENCHES = ("om2w", "webvoyager", "deepshop")


def slug(ckpt: str) -> str:
    """Beaker experiment names allow [a-zA-Z0-9_.-] only."""
    if len(ckpt.split()) > 1:      # a sequential multi-checkpoint sweep
        return "multi"
    return re.sub(r"[^A-Za-z0-9_.-]", "-", os.path.basename(ckpt.rstrip("/"))).lower()


def browserbase_env_args(args):
    """Extra gantry args for Browserbase mode; empty otherwise."""
    if args.env_mode != "browserbase":
        return []
    return [
        "--env", "SLIME_BROWSER_ENV_MODE=browserbase",
        "--secret-env", "BROWSERBASE_API_KEY=" + args.browserbase_key_secret,
        "--secret-env", "BROWSERBASE_PROJECT_ID=" + args.browserbase_project_secret,
        "--env", "BROWSERBASE_PROXIES=" + ("true" if args.browserbase_proxies else "false"),
        "--env", "BROWSERBASE_ADVANCED_STEALTH=" + ("true" if args.browserbase_stealth else "false"),
    ]


def browser_use_env_args(args):
    """Extra gantry args for Browser-Use mode (the paper's stealth browsers)."""
    if args.env_mode != "browser-use":
        return []
    return [
        "--env", "SLIME_BROWSER_ENV_MODE=browser-use",
        "--secret-env", "BROWSER_USE_API_KEY=" + args.browser_use_key_secret,
        "--env", f"BU_MAX_CONCURRENCY={args.browser_use_max_concurrency}",
    ]


def env_mode_args(args):
    """Extra gantry args for whichever non-default browser env mode is selected."""
    if args.env_mode == "local_process":
        return []
    if args.env_mode == "browserbase":
        return browserbase_env_args(args)
    if args.env_mode == "browser-use":
        return browser_use_env_args(args)
    return sandbox_env_args(args)


def sandbox_env_args(args):
    """Extra gantry args for Orchard sandbox mode; empty for local_process."""
    if args.env_mode != "sandbox":
        return []
    if not args.sandbox_url or not args.sandbox_image:
        raise SystemExit("sandbox mode needs --sandbox-url and --sandbox-image "
                         "(or SANDBOX_ORCHESTRATOR_URL / BROWSER_SANDBOX_IMAGE in the environment)")
    return [
        "--env", "SLIME_BROWSER_ENV_MODE=sandbox",
        "--env", "SANDBOX_ORCHESTRATOR_URL=" + args.sandbox_url,
        "--env", "BROWSER_SANDBOX_IMAGE=" + args.sandbox_image,
        "--secret-env", "SANDBOX_API_KEY=" + args.sandbox_key_secret,
    ]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True,
                   help="Local checkpoint dir or HuggingFace repo id. May be a quoted, "
                        "space-separated list, which is evaluated sequentially in one job "
                        "(required when a cloud-browser concurrency cap is account-wide).")
    p.add_argument("--bench", default="all", choices=BENCHES + ("all",))
    p.add_argument("--image", default=os.environ.get("BEAKER_IMAGE", "peters/openwebrl-train"))
    p.add_argument("--script", default="beaker/run_eval.sh")
    p.add_argument("--cluster", action="append", default=None,
                   help="Beaker cluster; repeatable (default ai2/jupiter)")
    p.add_argument("--budget", default="ai2/oe-omai")
    p.add_argument("--workspace", default="ai2/general-tool-use")
    p.add_argument("--gpus", type=int, default=8)
    p.add_argument("--priority", default="urgent")
    p.add_argument("--min-runtime", default="2h",
                   help="Guaranteed runtime before preemption; non-zero is what puts the job "
                        "on allocated slots instead of preemptible backfill.")
    p.add_argument("--env-mode", default="local_process",
                   choices=("local_process", "sandbox", "browserbase", "browser-use"),
                   help="Browser env mode. 'sandbox' uses Orchard pods; 'browser-use' uses the "
                        "paper's Browser-Use Stealth Browsers; 'browserbase' uses Browserbase "
                        "stealth browsers (1280x720, proxies off).")
    p.add_argument("--browser-use-key-secret", default="MOLMOWEB_BROWSERUSE",
                   help="Beaker secret holding the Browser-Use API key.")
    p.add_argument("--browser-use-max-concurrency", type=int, default=25,
                   help="Account concurrency cap. run_eval.sh refuses to start if --n-parallel "
                        "reaches it, since each task holds one cloud browser.")
    p.add_argument("--score-mode", default="", choices=("", "official", "noaborted"),
                   help="Paper Table 8 decoding preset. 'official' = temp 0.6/top_p 0.95/top_k 20 "
                        "(the Table 2 headline); 'noaborted' = temp 0.0/top_p 1.0/top_k 1.")
    p.add_argument("--n-parallel", type=int, default=0,
                   help="Concurrent tasks (one browser session each). 0 keeps the script default "
                        "of 16. Must stay under the provider's concurrency cap.")
    p.add_argument("--browserbase-key-secret", default="PS_BROWSERBASE_API_KEY",
                   help="Beaker secret holding the Browserbase API key.")
    p.add_argument("--browserbase-project-secret", default="PS_BROWSERBASE_PROJECT_ID",
                   help="Beaker secret holding the Browserbase project id.")
    p.add_argument("--browserbase-no-stealth", dest="browserbase_stealth",
                   action="store_false", default=True,
                   help="Disable advanced stealth. Stealth renders 1280x720 whatever "
                        "viewport is requested, so this is the only way to get the "
                        "configured 1280x1000 -- at the cost of anti-bot fingerprinting.")
    p.add_argument("--browserbase-proxies", dest="browserbase_proxies",
                   action="store_true", default=False,
                   help="Enable Browserbase residential proxies. Off by default (standing "
                        "rule); they bill per GB.")
    p.add_argument("--task-file", default="",
                   help="Task JSONL relative to the repo root; overrides the benchmark default "
                        "(protocol/judge still come from --bench).")
    p.add_argument("--max-steps", type=int, default=0,
                   help="Browser step budget. 0 keeps the repo default of 30; only lower it "
                        "deliberately, e.g. a pilot matching stage-1 RL rollouts at 15.")
    p.add_argument("--temperature", type=float, default=None,
                   help="Override the score-mode sampling temperature (e.g. 0.8 to mimic RL rollouts).")
    p.add_argument("--judge-endpoint-file", default="",
                   help="Weka file where beaker/run_judge_server.sh publishes its URL; selects a "
                        "self-hosted judge instead of the OpenAI API. Needs --judge-model.")
    p.add_argument("--judge-model", default="",
                   help="Judge model name (served model id for a self-hosted judge).")
    p.add_argument("--task-indices", default="",
                   help="Comma-separated row indices of the task file to evaluate; empty runs "
                        "the whole benchmark. Subset runs get their own output tree.")
    p.add_argument("--run-tag", default="",
                   help="Output subdirectory under outputs/eval/<ckpt>/ and experiment-name "
                        "suffix. Defaults to the benchmark name.")
    p.add_argument("--sandbox-url", default=os.environ.get("SANDBOX_ORCHESTRATOR_URL", ""),
                   help="Orchard orchestrator base URL (sandbox mode).")
    p.add_argument("--sandbox-image", default=os.environ.get("BROWSER_SANDBOX_IMAGE", ""),
                   help="browser-env image in a registry the Orchard cluster can pull.")
    p.add_argument("--sandbox-key-secret", default="PS_SANDBOX_API_KEY",
                   help="Beaker secret holding the Orchard API key.")
    p.add_argument("--ref", default="",
                   help="Git ref to report to gantry. The job reads code from the Weka mount, "
                        "so this is provenance only -- useful when HEAD in a shared checkout "
                        "carries another session's unpushed commits.")
    p.add_argument("--extra-env", action="append", default=[], metavar="KEY=VAL",
                   help="Additional env var for run_eval.sh (repeatable), e.g. EVAL_TOP_P=1.0.")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    benches = BENCHES if args.bench == "all" else (args.bench,)

    for bench in benches:
        command = [
            "gantry", "run",
            "--name", f"owrl_eval_{slug(args.ckpt)}_{args.run_tag or bench}",
            "--budget", args.budget,
            "--workspace", args.workspace,
            "--priority", args.priority,
            *[a for c in (args.cluster or ["ai2/jupiter"]) for a in ("--cluster", c)],
            "--min-runtime", args.min_runtime,
            "--gpus", str(args.gpus),
            "--shared-memory", "100GiB",
            "--weka", "oe-training-default:/weka/oe-training-default",
            "--beaker-image", args.image,
            # Judge: o4-mini (om2w), gpt-4o (webvoyager/deepshop). HF for weights.
            "--secret-env", "OPENAI_API_KEY=PS_OPENAI_API_KEY",
            "--secret-env", "HUGGINGFACE_HUB_TOKEN=PS_HF_TOKEN",
            "--env", "OPENWEBRL_ROOT=" + OPENWEBRL_ROOT,
            "--env", "BENCH=" + bench,
            "--env", "CKPT=" + args.ckpt,
            *env_mode_args(args),
            *(["--env", f"SLIME_BROWSER_SANDBOX_MAX_SANDBOXES={args.n_parallel}"]
              if args.n_parallel else []),
            *(["--env", "EVAL_SCORE_MODE=" + args.score_mode] if args.score_mode else []),
            *(["--env", "TASK_INDICES=" + args.task_indices] if args.task_indices else []),
            *(["--env", "RUN_TAG=" + args.run_tag] if args.run_tag else []),
            *(["--env", "TASK_FILE=" + args.task_file] if args.task_file else []),
            *(["--env", f"BROWSER_MAX_STEPS={args.max_steps}"] if args.max_steps else []),
            *(["--env", f"EVAL_TEMPERATURE={args.temperature}"] if args.temperature is not None else []),
            # Self-hosted judge: cross-job traffic goes to the judge node's IP, which
            # is only guaranteed routable pod->node with host networking on both sides.
            *(["--env", "JUDGE_ENDPOINT_FILE=" + args.judge_endpoint_file, "--host-networking"]
              if args.judge_endpoint_file else []),
            *(["--env", "JUDGE_MODEL=" + args.judge_model] if args.judge_model else []),
            *(a for kv in args.extra_env for a in ("--env", kv)),
            *(["--ref", args.ref] if args.ref else []),
            "--no-python",
            "--allow-dirty",
            "--", "bash", f"{OPENWEBRL_ROOT}/{args.script}",
        ]
        print("[launch]", " ".join(command))
        if args.dry_run:
            continue
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
