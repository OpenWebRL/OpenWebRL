"""
Launch a standalone VLM judge server (WebJudge-7B by default) on Beaker via Gantry.

One GPU, host networking, runs until killed. Publishes its URL to a Weka file that
eval/RL jobs read via JUDGE_ENDPOINT_FILE (see beaker/run_judge_server.sh).

Usage:
  python beaker/launch_judge_server.py --cluster ai2/holmes --workspace ai2/oe-agents-holmes
"""
import argparse
import os
import subprocess

OPENWEBRL_ROOT = "/weka/oe-training-default/new_peters/OpenWebRL"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", default="/weka/oe-training-default/new_peters/models/WebJudge-7B")
    p.add_argument("--served-model-name", default="WebJudge-7B")
    p.add_argument("--port", type=int, default=8300)
    p.add_argument("--endpoint-file", default="",
                   help="Defaults to outputs/judge/<served-model-name>.endpoint under the repo.")
    p.add_argument("--image", default=os.environ.get("BEAKER_IMAGE", "peters/openwebrl-train-sm103f"))
    p.add_argument("--cluster", default="ai2/holmes")
    p.add_argument("--budget", default="ai2/oe-omai")
    p.add_argument("--workspace", default="ai2/oe-agents-holmes")
    p.add_argument("--priority", default="urgent")
    p.add_argument("--min-runtime", default="8h")  # Beaker cap
    p.add_argument("--name", default="")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    command = [
        "gantry", "run",
        "--name", args.name or f"owrl_judge_{args.served_model_name.lower()}",
        "--budget", args.budget,
        "--workspace", args.workspace,
        "--priority", args.priority,
        "--cluster", args.cluster,
        "--min-runtime", args.min_runtime,
        "--gpus", "1",
        "--shared-memory", "32GiB",
        "--host-networking",
        "--weka", "oe-training-default:/weka/oe-training-default",
        "--beaker-image", args.image,
        "--env", "OPENWEBRL_ROOT=" + OPENWEBRL_ROOT,
        "--env", "MODEL_PATH=" + args.model_path,
        "--env", "SERVED_MODEL_NAME=" + args.served_model_name,
        "--env", f"JUDGE_PORT={args.port}",
        # slow Weka reads (cold-tiered or degraded) make the 20-min default startup window fail
        "--env", f"SGLANG_STARTUP_TIMEOUT_SECS={os.environ.get('SGLANG_STARTUP_TIMEOUT_SECS', '7200')}",
        *(["--env", "JUDGE_ENDPOINT_FILE=" + args.endpoint_file] if args.endpoint_file else []),
        "--no-python",
        "--allow-dirty",
        "--", "bash", f"{OPENWEBRL_ROOT}/beaker/run_judge_server.sh",
    ]
    print("[launch]", " ".join(command))
    if not args.dry_run:
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
