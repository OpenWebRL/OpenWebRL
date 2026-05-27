import torch

from slime.rollout.filter_hub.base_types import DynamicFilterOutput
from slime.utils.types import Sample

__all__ = ["check_reward_nonzero_std", "check_reward_nonempty", "check_reward_nonempty_nonzero_std"]


def _terminal_sample(sample: Sample | list[Sample]) -> Sample:
    return sample[-1] if isinstance(sample, list) else sample


def _reward_value(args, sample: Sample):
    if sample.reward is None:
        return None
    return sample.get_reward_value(args)


def check_reward_nonzero_std(args, samples: list[Sample], **kwargs):
    rewards = [sample.get_reward_value(args) for sample in samples]
    keep = torch.tensor(rewards, dtype=torch.float64).std() > 1e-6
    return DynamicFilterOutput(
        keep=keep,
        reason=None if keep else f"zero_std_{round(rewards[0], 1)}",
    )


def check_reward_nonempty_nonzero_std(args, samples: list[Sample], **kwargs):
    kept_samples = []
    rewards = []
    num_none_rewards = 0
    for sample in samples:
        reward_sample = _terminal_sample(sample)
        reward = _reward_value(args, reward_sample)
        if reward is None:
            num_none_rewards += 1
            continue
        kept_samples.append(sample)
        rewards.append(reward)

    if not kept_samples:
        return DynamicFilterOutput(keep=False, reason="all_none_reward_in_group")

    if num_none_rewards > 0 and len(kept_samples) < 2:
        return DynamicFilterOutput(
            keep=False,
            reason=f"insufficient_nonempty_rewards_{len(kept_samples)}",
        )

    if len(kept_samples) == 1:
        return DynamicFilterOutput(
            keep=False,
            reason="single_valid_reward_in_group",
        )

    keep = torch.tensor(rewards, dtype=torch.float64).std() > 1e-6
    return DynamicFilterOutput(
        keep=keep,
        reason=None if keep else f"zero_std_{round(rewards[0], 1)}",
        samples=kept_samples if keep and len(kept_samples) != len(samples) else None,
    )


def check_reward_nonempty(args, samples: list[Sample], **kwargs):
    kept_samples = []
    for sample in samples:
        reward_sample = _terminal_sample(sample)
        if _reward_value(args, reward_sample) is None:
            continue
        kept_samples.append(sample)

    if not kept_samples:
        return DynamicFilterOutput(keep=False, reason="all_none_reward_in_group")

    return DynamicFilterOutput(
        keep=True,
        samples=kept_samples if len(kept_samples) != len(samples) else None,
    )
