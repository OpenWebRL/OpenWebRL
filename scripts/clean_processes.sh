# Clean up sandbox environments
python openwebrl/env/sandbox_env.py --cleanup

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
