"""
Launch one RL-step sweep run (MM-GRPO on the fixed 48-task om2w subset, Browserbase
rollouts, self-hosted WebJudge-7B) on Beaker via Gantry. Entry point:
beaker/run_rl_sweep.sh; every hyperparameter there is an env knob, settable with
--set KEY=VAL.

Usage:
  # paper stage-1 baseline, seed 1234
  python beaker/launch_rl_sweep.py --name sweep_base_s1234

  # a sweep point
  python beaker/launch_rl_sweep.py --name sweep_lr2e6_s7 --set RL_LR=2e-6 --set RL_SEED=7

Prereqs: a running judge (beaker/launch_judge_server.py) publishing the endpoint
file; secrets BROWSERBASE_API_KEY, BROWSERBASE_PROJECT_ID, PS_HF_TOKEN,
PS_WANDB_API_KEY in the workspace.
"""
import argparse
import os
import shutil
import subprocess

OPENWEBRL_ROOT = "/weka/oe-training-default/new_peters/OpenWebRL"
DEFAULT_JUDGE_ENDPOINT_FILE = f"{OPENWEBRL_ROOT}/outputs/judge/WebJudge-7B.endpoint"


def snapshot_script(script_rel, name):
    """Run an immutable per-run copy: bash reads scripts lazily by byte offset,
    so editing the shared file under a live job would corrupt it."""
    src = os.path.join(OPENWEBRL_ROOT, script_rel)
    # RL_SNAP_DIR: escape hatch when the default directory is wedged on Weka (stat hangs in D state)
    snap_dir = os.environ.get("RL_SNAP_DIR") or os.path.join(OPENWEBRL_ROOT, "outputs", "rl_sweep", "_launched")
    os.makedirs(snap_dir, exist_ok=True)
    dst = os.path.join(snap_dir, f"{name}_{os.path.basename(script_rel)}")
    shutil.copy2(src, dst)
    return dst


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True, help="RUN_NAME: save dir, W&B run, Beaker experiment.")
    p.add_argument("--beaker-name", default="", help="Beaker experiment name when it must differ from RUN_NAME (a resume of the same run).")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VAL",
                   help="Env knob for run_rl_sweep.sh (repeatable): RL_LR, RL_SEED, NUM_ROLLOUT, "
                        "PPO_EPOCHS, ROLLOUT_BATCH_SIZE, N_SAMPLES_PER_PROMPT, GLOBAL_BATCH_SIZE, "
                        "BROWSER_MAX_STEPS, SLIME_BROWSER_ROLLOUT_CONCURRENCY, RL_MAX_TOKENS_PER_GPU, "
                        "SGLANG_SERVER_CONCURRENCY, EVAL_INTERVAL, ...")
    p.add_argument("--judge-endpoint-file", default=DEFAULT_JUDGE_ENDPOINT_FILE)
    p.add_argument("--resume-from", default="", help="Existing save dir (SLIME_LOAD_CHECKPOINT).")
    p.add_argument("--ckpt-step", type=int, default=0)
    p.add_argument("--script", default="beaker/run_rl_sweep.sh")
    p.add_argument("--image", default=os.environ.get("BEAKER_IMAGE", "peters/openwebrl-train-sm103f"))
    p.add_argument("--cluster", default="ai2/holmes")
    p.add_argument("--budget", default="ai2/oe-omai")
    p.add_argument("--workspace", default="ai2/oe-agents-holmes")
    p.add_argument("--priority", default="urgent")
    p.add_argument("--min-runtime", default="2h")  # holmes only schedules allocated slots; 2h is what the allocated jobs in ai2/oe-agents-holmes use. 8h waits for a window that rarely exists, "" drops to backfill and sits pending
    p.add_argument("--gpus", type=int, default=8, help="GPUs per node.")
    p.add_argument("--nodes", type=int, default=1,
                   help="Beaker replicas; rank 0 runs the Ray head + train.py, the rest join as Ray workers.")
    p.add_argument("--browserbase-key-secret", default="BROWSERBASE_API_KEY")
    p.add_argument("--browserbase-project-secret", default="BROWSERBASE_PROJECT_ID")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    for kv in args.set:
        if "=" not in kv:
            raise SystemExit(f"--set expects KEY=VAL, got {kv!r}")

    command = [
        "gantry", "run",
        "--name", args.beaker_name or args.name,
        "--budget", args.budget,
        "--workspace", args.workspace,
        "--priority", args.priority,
        "--cluster", args.cluster,
        # A guaranteed minimum runtime stops the job from backfilling: it waits for a
        # slot that long, which can be hours on a busy cluster. Pass "" to backfill.
        *(["--min-runtime", args.min_runtime] if args.min_runtime else []),
        "--gpus", str(args.gpus),
        # --leader-selection is what populates BEAKER_LEADER_REPLICA_HOSTNAME;
        # --replicas alone does not. --propagate-failure so a dead worker stops the head.
        *(["--replicas", str(args.nodes), "--leader-selection", "--propagate-failure",
           "--synchronized-start-timeout", "30m"] if args.nodes > 1 else []),
        # /dev/shm backs Ray's object store (RAY_OBJECT_STORE_BYTES in run_rl_sweep.sh,
        # default 400 GB); 100 GiB spilled to disk every iteration. Node has 3 TB.
        "--shared-memory", "500GiB",
        # Judge traffic goes to another job's node IP; routable only with host networking.
        "--host-networking",
        "--weka", "oe-training-default:/weka/oe-training-default",
        "--beaker-image", args.image,
        "--secret-env", "HUGGINGFACE_HUB_TOKEN=PS_HF_TOKEN",
        "--secret-env", "WANDB_API_KEY=PS_WANDB_API_KEY",
        "--secret-env", "BROWSERBASE_API_KEY=" + args.browserbase_key_secret,
        "--secret-env", "BROWSERBASE_PROJECT_ID=" + args.browserbase_project_secret,
        "--secret-env", "OPENAI_API_KEY=PS_OPENAI_API_KEY",  # molmoweb_om2w judge (o4-mini)
        "--env", "OPENWEBRL_ROOT=" + OPENWEBRL_ROOT,
        "--env", "RUN_NAME=" + args.name,
        "--env", f"NUM_NODES={args.nodes}",
        "--env", f"NUM_GPUS={args.gpus}",
        "--env", "JUDGE_ENDPOINT_FILE=" + args.judge_endpoint_file,
        "--env", "SLIME_BROWSER_ENV_MODE=browserbase",
        "--env", "BROWSERBASE_PROXIES=false",
        *(["--env", "SLIME_LOAD_CHECKPOINT=" + args.resume_from] if args.resume_from else []),
        *(["--env", f"SLIME_CKPT_STEP={args.ckpt_step}"] if args.ckpt_step else []),
        *(a for kv in args.set for a in ("--env", kv)),
        "--no-python",
        "--allow-dirty",
        "--", "bash", snapshot_script(args.script, args.name),
    ]
    print("[launch]", " ".join(command))
    if not args.dry_run:
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
