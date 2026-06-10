"""
Launch a Tier-3 OpenWebRL GRPO smoke run on Beaker via Gantry.

One 8-GPU node on ai2/jupiter. The slime/Megatron/SGLang stack comes from an
existing Beaker image (pass --image or set BEAKER_IMAGE); the OpenWebRL repo is
read from the Weka mount, so no code is uploaded by gantry. The container entry
point is beaker/run_rl_smoke.sh.

Usage:
  python beaker/launch_rl_smoke.py --image <beaker-image-name-or-id>

Prereqs:
  - `beaker account login` (gantry uses your Beaker auth)
  - Secrets present in the workspace: PS_OPENAI_API_KEY, PS_HF_TOKEN, PS_WANDB_API_KEY
"""
import argparse
import os
import subprocess

OPENWEBRL_ROOT = "/weka/oe-training-default/new_peters/OpenWebRL"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", default=os.environ.get("BEAKER_IMAGE", ""),
                   help="Existing Beaker image with the slime/Megatron/SGLang stack.")
    p.add_argument("--name", default="openwebrl_rl_smoke")
    p.add_argument("--cluster", default="ai2/jupiter")
    p.add_argument("--budget", default="ai2/oe-omai")
    p.add_argument("--workspace", default="ai2/general-tool-use")
    p.add_argument("--gpus", type=int, default=8)
    p.add_argument("--priority", default="high")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    assert args.image, "Pass --image (or set BEAKER_IMAGE) with the existing slime/Megatron/SGLang Beaker image."

    command = [
        "gantry", "run",
        "--name", args.name,
        "--budget", args.budget,
        "--workspace", args.workspace,
        "--priority", args.priority,
        "--cluster", args.cluster,
        "--gpus", str(args.gpus),
        "--shared-memory", "100GiB",
        "--weka", "oe-training-default:/weka/oe-training-default",
        "--beaker-image", args.image,
        # Judge (standard OpenAI), HF download, optional W&B.
        "--secret-env", "OPENAI_API_KEY=PS_OPENAI_API_KEY",
        "--secret-env", "HUGGINGFACE_HUB_TOKEN=PS_HF_TOKEN",
        "--secret-env", "WANDB_API_KEY=PS_WANDB_API_KEY",
        "--env", "OPENWEBRL_ROOT=" + OPENWEBRL_ROOT,
        "--no-python",
        "--allow-dirty",
        "--", "bash", f"{OPENWEBRL_ROOT}/beaker/run_rl_smoke.sh",
    ]

    print("[launch]", " ".join(command))
    if args.dry_run:
        return
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
