# Clean up sandbox environments (best-effort; no-ops in local_process mode)
python openwebrl/env/sandbox_env.py --cleanup || true

# unset wandb environment variables
unset WANDB_RUN_ID
unset WANDB_RUN_GROUP
unset WANDB_PROJECT
unset WANDB_NOTES
unset WANDB_NAME

# for rerun the task
# pkill -9 sglang
sleep 3
ray stop --force
pkill -9 ray
# pkill -9 python
sleep 3
pkill -9 ray
# pkill -9 python
pkill -9 redis

# This is best-effort cleanup run after training; pkill returns non-zero when
# no matching process exists, which must not fail the (successful) job.
exit 0
