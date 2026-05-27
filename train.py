import logging
import time

import ray
from ray.exceptions import ActorUnavailableError
from tqdm.auto import tqdm

from slime.ray.placement_group import create_placement_groups, create_rollout_manager, create_training_models
from slime.utils.arguments import parse_args
from slime.utils.logging_utils import append_progress_log, configure_logger, finish_tracking, init_tracking
from slime.utils.misc import should_run_periodic_action

logger = logging.getLogger(__name__)


def _build_train_progress_bar(args, num_rollout_per_epoch: int):
    total_rollouts = max(args.num_rollout - args.start_rollout_id, 0)
    if total_rollouts <= 0:
        return None

    return tqdm(
        total=total_rollouts,
        desc="SFT/RL train",
        dynamic_ncols=True,
        leave=True,
    )


def _update_train_progress_bar(progress_bar, *, rollout_id: int, args, phase: str, num_rollout_per_epoch: int):
    if progress_bar is None:
        return

    postfix = {"phase": phase, "rollout": f"{rollout_id + 1}/{args.num_rollout}"}
    if num_rollout_per_epoch and args.num_epoch:
        epoch_id = rollout_id // num_rollout_per_epoch + 1
        epoch_rollout = rollout_id % num_rollout_per_epoch + 1
        postfix["epoch"] = f"{epoch_id}/{args.num_epoch}"
        postfix["epoch_step"] = f"{epoch_rollout}/{num_rollout_per_epoch}"

    progress_bar.set_postfix(postfix, refresh=False)


def _advance_train_progress_bar(progress_bar, *, amount: int = 1):
    if progress_bar is not None:
        progress_bar.update(amount)


def _log_train_progress(rollout_id: int, total_rollouts: int, phase: str) -> None:
    line = f"[TrainProgress] rollout={rollout_id + 1}/{total_rollouts} phase={phase}"
    logger.info(line)
    append_progress_log(getattr(_log_train_progress, "_args", None), line)


def _ray_get_with_actor_retry(obj_ref, *, label: str, max_attempts: int = 5, base_delay_secs: float = 5.0):
    attempt = 0
    while True:
        attempt += 1
        try:
            return ray.get(obj_ref)
        except ActorUnavailableError as exc:
            if attempt >= max_attempts:
                raise
            delay = base_delay_secs * attempt
            logger.warning(
                "Retrying ray.get for %s after ActorUnavailableError (attempt %d/%d, sleep %.1fs): %s",
                label,
                attempt,
                max_attempts,
                delay,
                exc,
            )
            time.sleep(delay)


def train(args):
    configure_logger()
    _log_train_progress._args = args
    # allocate the GPUs
    pgs = create_placement_groups(args)
    init_tracking(args)

    # create the rollout manager, with sglang engines inside.
    # need to initialize rollout manager first to calculate num_rollout
    rollout_manager, num_rollout_per_epoch = create_rollout_manager(args, pgs["rollout"])

    # create the actor and critic models
    actor_model, critic_model = create_training_models(args, pgs, rollout_manager)
    progress_bar = _build_train_progress_bar(args, num_rollout_per_epoch)

    if args.offload_rollout:
        _ray_get_with_actor_retry(rollout_manager.onload_weights.remote(), label="rollout_manager.onload_weights")

    # always update weight first so that sglang has the loaded weights from training.
    if not args.critic_train_only:
        actor_model.update_weights()

        if args.check_weight_update_equal:
            _ray_get_with_actor_retry(
                rollout_manager.check_weights.remote(action="compare"),
                label="rollout_manager.check_weights(compare)",
            )

    if args.offload_rollout:
        _ray_get_with_actor_retry(rollout_manager.onload_kv.remote(), label="rollout_manager.onload_kv")

    # special case for eval-only
    if args.num_rollout == 0 and args.eval_interval is not None:
        _ray_get_with_actor_retry(rollout_manager.eval.remote(rollout_id=0), label="rollout_manager.eval(0)")

    def offload_train(rollout_id):
        if args.offload_train:
            if args.use_critic:
                critic_model.offload()
                if rollout_id >= args.num_critic_only_steps and not args.critic_train_only:
                    actor_model.offload()
            else:
                actor_model.offload()
        else:
            if args.critic_train_only:
                critic_model.clear_memory()
            else:
                actor_model.clear_memory()

    def save(rollout_id):
        if (not args.use_critic) or (rollout_id >= args.num_critic_only_steps and not args.critic_train_only):
            actor_model.save_model(
                rollout_id,
                force_sync=rollout_id == args.num_rollout - 1,
            )
        if args.use_critic:
            critic_model.save_model(
                rollout_id,
                force_sync=rollout_id == args.num_rollout - 1,
            )
        if args.rollout_global_dataset:
            _ray_get_with_actor_retry(rollout_manager.save.remote(rollout_id), label=f"rollout_manager.save({rollout_id})")

    # train loop.
    # note that for async training, one can change the position of the sync operation(ray.get).
    try:
        for rollout_id in range(args.start_rollout_id, args.num_rollout):
            if args.eval_interval is not None and rollout_id == 0 and not args.skip_eval_before_train:
                _update_train_progress_bar(
                    progress_bar,
                    rollout_id=rollout_id,
                    args=args,
                    phase="eval",
                    num_rollout_per_epoch=num_rollout_per_epoch,
                )
                _log_train_progress(rollout_id, args.num_rollout, "eval")
                _ray_get_with_actor_retry(
                    rollout_manager.eval.remote(rollout_id),
                    label=f"rollout_manager.eval({rollout_id})",
                )

            _update_train_progress_bar(
                progress_bar,
                rollout_id=rollout_id,
                args=args,
                phase="generate",
                num_rollout_per_epoch=num_rollout_per_epoch,
            )
            _log_train_progress(rollout_id, args.num_rollout, "generate")
            rollout_data_ref = _ray_get_with_actor_retry(
                rollout_manager.generate.remote(rollout_id),
                label=f"rollout_manager.generate({rollout_id})",
            )

            if args.offload_rollout:
                _ray_get_with_actor_retry(rollout_manager.offload.remote(), label="rollout_manager.offload")

            if args.use_critic:
                critic_train_handle = critic_model.async_train(rollout_id, rollout_data_ref)
                if rollout_id >= args.num_critic_only_steps and not args.critic_train_only:
                    _update_train_progress_bar(
                        progress_bar,
                        rollout_id=rollout_id,
                        args=args,
                        phase="train_actor",
                        num_rollout_per_epoch=num_rollout_per_epoch,
                    )
                    _log_train_progress(rollout_id, args.num_rollout, "train_actor")
                    actor_train_handle = actor_model.async_train(rollout_id, rollout_data_ref)
                    _ray_get_with_actor_retry(
                        actor_train_handle,
                        label=f"actor_model.async_train({rollout_id})",
                    )
                _update_train_progress_bar(
                    progress_bar,
                    rollout_id=rollout_id,
                    args=args,
                    phase="train_critic",
                    num_rollout_per_epoch=num_rollout_per_epoch,
                )
                _log_train_progress(rollout_id, args.num_rollout, "train_critic")
                _ray_get_with_actor_retry(
                    critic_train_handle,
                    label=f"critic_model.async_train({rollout_id})",
                )
            else:
                _update_train_progress_bar(
                    progress_bar,
                    rollout_id=rollout_id,
                    args=args,
                    phase="train_actor",
                    num_rollout_per_epoch=num_rollout_per_epoch,
                )
                _log_train_progress(rollout_id, args.num_rollout, "train_actor")
                actor_train_handle = actor_model.async_train(rollout_id, rollout_data_ref)
                _ray_get_with_actor_retry(
                    actor_train_handle,
                    label=f"actor_model.async_train({rollout_id})",
                )

            if should_run_periodic_action(rollout_id, args.save_interval, num_rollout_per_epoch, args.num_rollout):
                _update_train_progress_bar(
                    progress_bar,
                    rollout_id=rollout_id,
                    args=args,
                    phase="save",
                    num_rollout_per_epoch=num_rollout_per_epoch,
                )
                _log_train_progress(rollout_id, args.num_rollout, "save")
                save(rollout_id)

            offload_train(rollout_id)
            if args.offload_rollout:
                _ray_get_with_actor_retry(
                    rollout_manager.onload_weights.remote(),
                    label="rollout_manager.onload_weights",
                )
            if not args.critic_train_only:
                actor_model.update_weights()
            if args.offload_rollout:
                _ray_get_with_actor_retry(rollout_manager.onload_kv.remote(), label="rollout_manager.onload_kv")

            if should_run_periodic_action(rollout_id, args.eval_interval, num_rollout_per_epoch):
                _update_train_progress_bar(
                    progress_bar,
                    rollout_id=rollout_id,
                    args=args,
                    phase="eval",
                    num_rollout_per_epoch=num_rollout_per_epoch,
                )
                _log_train_progress(rollout_id, args.num_rollout, "eval")
                _ray_get_with_actor_retry(
                    rollout_manager.eval.remote(rollout_id),
                    label=f"rollout_manager.eval({rollout_id})",
                )

            _advance_train_progress_bar(progress_bar)
    finally:
        if progress_bar is not None:
            progress_bar.close()

    _ray_get_with_actor_retry(rollout_manager.dispose.remote(), label="rollout_manager.dispose")
    finish_tracking(args)


if __name__ == "__main__":
    args = parse_args()
    train(args)
