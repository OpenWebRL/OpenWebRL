import abc
import copy
import json
import logging
import os
import random
from pathlib import Path
from urllib.parse import urlsplit

import torch

from slime.rollout.query_selector import (
    QueryStats,
    extract_completed_trajectory_length,
    extract_query_key,
    extract_terminal_sample,
)
from slime.utils.data import Dataset
from slime.utils.logging_utils import append_progress_log
from slime.utils.misc import load_function
from slime.utils.processing_utils import load_processor, load_tokenizer
from slime.utils.types import Sample

logger = logging.getLogger(__name__)


class DataSource(abc.ABC):
    @abc.abstractmethod
    def get_samples(self, num_samples: int) -> list[list[Sample]]:
        """
        Return num_samples samples
        """

    @abc.abstractmethod
    def add_samples(self, samples: list[list[Sample]]):
        """
        Add samples to the data source
        """

    @abc.abstractmethod
    def save(self, rollout_id):
        """
        Save the state of the data source
        """

    @abc.abstractmethod
    def load(self, rollout_id=None):
        """
        Load the state of the data source
        """

    @abc.abstractmethod
    def __len__(self) -> int:
        """
        Length of the data source. May change when samples are added/fetched.
        """

    def update_from_rollout(self, samples: list[list[Sample]], rollout_id: int | None = None):
        """Update data-source state using finished rollout groups."""
        return None


# TODO may further refactor data-loading part later
class RolloutDataSource(DataSource):
    def __init__(self, args):
        self.args = args

        self.epoch_id = 0
        self.sample_group_index = 0
        self.sample_index = 0
        self.sample_offset = 0
        # TODO remove this
        self.metadata = {}
        self._adaptive_rng = random.Random(getattr(args, "rollout_seed", 42))
        self._adaptive_query_stats: dict[str, QueryStats] = {}
        self._adaptive_select_step = 0
        self._adaptive_global_completed_traj_count = 0
        self._adaptive_global_completed_len_sum = 0.0
        self._adaptive_last_rollout_id = -1
        self._adaptive_last_update_status = "not_started"
        self._adaptive_last_updated_queries = 0
        self._adaptive_blacklist_hosts = self._load_adaptive_blacklist_hosts()
        self._adaptive_blacklist_logged_empty = False
        self._sequential_blacklist_logged_empty = False

        if args.rollout_global_dataset:
            tokenizer = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)
            processor = load_processor(args.hf_checkpoint, trust_remote_code=True)

            # TODO move (during the refactor)
            if (d := args.dump_details) is not None:
                tokenizer.save_pretrained(Path(d) / "tokenizer")
                if processor:
                    processor.save_pretrained(Path(d) / "processor")

            self.dataset = Dataset(
                args.prompt_data,
                tokenizer=tokenizer,
                processor=processor,
                max_length=args.rollout_max_prompt_len,
                prompt_key=args.input_key,
                multimodal_keys=args.multimodal_keys,
                label_key=args.label_key,
                metadata_key=args.metadata_key,
                tool_key=args.tool_key,
                apply_chat_template=args.apply_chat_template,
                apply_chat_template_kwargs=args.apply_chat_template_kwargs,
                seed=args.rollout_seed,
            )
            if self.args.rollout_shuffle:
                self.dataset.shuffle(self.epoch_id)
        else:
            self.dataset = None

    def _load_adaptive_blacklist_hosts(self) -> set[str]:
        """Load an optional host blacklist used to skip queries during adaptive sampling."""
        configured_path = (
            getattr(self.args, "adaptive_query_blacklist_path", None)
            or os.environ.get("SLIME_ADAPTIVE_QUERY_BLACKLIST_PATH")
            or os.environ.get("SLIME_BROWSER_QUERY_BLACKLIST_PATH")
        )

        if configured_path:
            path = Path(configured_path)
        else:
            repo_root = Path(__file__).resolve().parents[2]
            path = repo_root / "examples" / "browser" / "data" / "webgym_filtered_popular_blacklist_hosts.txt"

        if not path.exists():
            return set()

        hosts = {
            line.strip().lower()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }
        if hosts:
            logger.info("Loaded %d adaptive query blacklist host(s) from %s", len(hosts), path)
        return hosts

    def _extract_sample_host(self, sample: Sample) -> str:
        metadata = sample.metadata or {}
        start_url = metadata.get("start_url")
        if not isinstance(start_url, str) or not start_url:
            return ""
        try:
            return (urlsplit(start_url).hostname or "").lower()
        except Exception:
            return ""

    def _is_blacklisted_query_sample(self, sample: Sample) -> bool:
        if not self._adaptive_blacklist_hosts:
            return False
        host = self._extract_sample_host(sample)
        return bool(host and host in self._adaptive_blacklist_hosts)

    def get_samples(self, num_samples):
        if self._use_adaptive_query_sampling():
            prompt_samples = self._get_adaptive_prompt_samples(num_samples)
        elif self.dataset is not None:
            prompt_samples = self._get_sequential_prompt_samples(num_samples)
        else:
            prompt_samples = [Sample() for _ in range(num_samples)]

        samples = []
        for prompt_sample in prompt_samples:
            group = []
            for _ in range(self.args.n_samples_per_prompt):
                sample = copy.deepcopy(prompt_sample)
                sample.group_index = self.sample_group_index
                sample.index = self.sample_index
                self.sample_index += 1
                group.append(sample)
            self.sample_group_index += 1
            samples.append(group)
        return samples

    def add_samples(self, samples: list[list[Sample]]):
        raise RuntimeError(f"Cannot add samples to {self.__class__.__name__}. This is a read-only data source.")

    def _use_adaptive_query_sampling(self) -> bool:
        return bool(getattr(self.args, "enable_adaptive_query_sampling", False) and self.dataset is not None)

    def _get_sequential_prompt_samples(self, num_samples: int) -> list[Sample]:
        assert self.dataset is not None and len(self.dataset) > 0, "No dataset rows available for sequential sampling."

        if not self._adaptive_blacklist_hosts:
            if self.sample_offset + num_samples <= len(self.dataset):
                prompt_samples = self.dataset.samples[self.sample_offset : self.sample_offset + num_samples]
                self.sample_offset += num_samples
                return prompt_samples

            prompt_samples = self.dataset.samples[self.sample_offset :]
            remaining = num_samples - len(prompt_samples)
            self.epoch_id += 1
            if self.args.rollout_shuffle:
                self.dataset.shuffle(self.epoch_id)
            prompt_samples += self.dataset.samples[:remaining]
            self.sample_offset = remaining
            return prompt_samples

        prompt_samples: list[Sample] = []
        visited_since_yield = 0
        dataset_len = len(self.dataset)

        while len(prompt_samples) < num_samples:
            if visited_since_yield >= dataset_len:
                if not self._sequential_blacklist_logged_empty:
                    logger.warning(
                        "Sequential sampling blacklist filtered out every dataset row; falling back to unfiltered sequential sampling."
                    )
                    self._sequential_blacklist_logged_empty = True
                remaining = num_samples - len(prompt_samples)
                if self.sample_offset + remaining <= dataset_len:
                    prompt_samples.extend(self.dataset.samples[self.sample_offset : self.sample_offset + remaining])
                    self.sample_offset += remaining
                else:
                    prompt_samples.extend(self.dataset.samples[self.sample_offset :])
                    remaining = num_samples - len(prompt_samples)
                    self.epoch_id += 1
                    if self.args.rollout_shuffle:
                        self.dataset.shuffle(self.epoch_id)
                    prompt_samples.extend(self.dataset.samples[:remaining])
                    self.sample_offset = remaining
                break

            if self.sample_offset >= dataset_len:
                self.epoch_id += 1
                if self.args.rollout_shuffle:
                    self.dataset.shuffle(self.epoch_id)
                self.sample_offset = 0

            sample = self.dataset.samples[self.sample_offset]
            self.sample_offset += 1
            visited_since_yield += 1

            if self._is_blacklisted_query_sample(sample):
                continue

            prompt_samples.append(sample)
            visited_since_yield = 0

        return prompt_samples

    def _get_adaptive_prompt_samples(self, num_samples: int) -> list[Sample]:
        prompt_samples = []
        while len(prompt_samples) < num_samples:
            idx = self._sample_adaptive_index()
            prompt_sample = self.dataset.samples[idx]
            prompt_samples.append(prompt_sample)
            self._on_query_selected(prompt_sample)

        return prompt_samples

    def _sample_adaptive_index(self) -> int:
        assert self.dataset is not None and len(self.dataset) > 0, "No dataset rows available for adaptive sampling."

        candidate_indices = [
            idx
            for idx in range(len(self.dataset))
            if not self._is_blacklisted_query_sample(self.dataset.samples[idx])
        ]
        if not candidate_indices:
            if not self._adaptive_blacklist_logged_empty:
                logger.warning(
                    "Adaptive query blacklist filtered out every dataset row; falling back to the full dataset."
                )
                self._adaptive_blacklist_logged_empty = True
            candidate_indices = list(range(len(self.dataset)))

        weights = [self._compute_adaptive_weight(idx) for idx in candidate_indices]
        total_weight = sum(weights)
        if total_weight <= 0:
            return candidate_indices[self._adaptive_rng.randrange(len(candidate_indices))]

        threshold = self._adaptive_rng.random() * total_weight
        cumulative = 0.0
        choice_pos = len(weights) - 1
        for pos, weight in enumerate(weights):
            cumulative += weight
            if cumulative >= threshold:
                choice_pos = pos
                break

        return candidate_indices[choice_pos]

    def _compute_adaptive_weight(self, dataset_idx: int) -> float:
        sample = self.dataset.samples[dataset_idx]
        query_key = extract_query_key(sample)
        stats = self._adaptive_query_stats.get(query_key, QueryStats())

        sr_weight = self._compute_success_rate_bucket_weight(stats)
        stale_bonus = self._compute_stale_bonus(stats)
        length_bonus = self._compute_length_bonus(stats)
        return max(sr_weight * stale_bonus * length_bonus, 1e-8)

    def _compute_success_rate_bucket_weight(self, stats: QueryStats) -> float:
        if stats.total_rewarded_samples < self.args.adaptive_query_warmup_samples:
            return 1.0

        success_rate = stats.success_rate
        if success_rate == 0.0:
            return self.args.adaptive_query_weight_all_fail
        if success_rate < 0.3:
            return self.args.adaptive_query_weight_low
        if success_rate < 0.7:
            return self.args.adaptive_query_weight_mid
        if success_rate < 1.0:
            return self.args.adaptive_query_weight_high
        return self.args.adaptive_query_weight_all_success

    def _compute_stale_bonus(self, stats: QueryStats) -> float:
        stale_cap = max(int(self.args.adaptive_query_stale_cap), 1)
        if stats.last_selected_step is None:
            stale_ratio = 1.0
        else:
            staleness = max(self._adaptive_select_step - stats.last_selected_step, 0)
            stale_ratio = min(staleness / stale_cap, 1.0)
        return 1.0 + self.args.adaptive_query_stale_alpha * stale_ratio

    def _compute_length_bonus(self, stats: QueryStats) -> float:
        if stats.completed_traj_count == 0 or self._adaptive_global_completed_traj_count == 0:
            return 1.0

        ramp_rollouts = max(int(self.args.adaptive_query_length_ramp_rollouts), 1)
        progress = min(max(self._adaptive_last_rollout_id, 0) / ramp_rollouts, 1.0)
        if progress <= 0:
            return 1.0

        confidence_k = max(float(self.args.adaptive_query_length_confidence_k), 1.0)
        confidence = stats.completed_traj_count / (stats.completed_traj_count + confidence_k)
        global_avg_completed_len = self._adaptive_global_completed_len_sum / self._adaptive_global_completed_traj_count
        if global_avg_completed_len <= 0:
            return 1.0

        len_norm = stats.avg_completed_len / global_avg_completed_len
        len_norm = min(max(len_norm, 0.5), self.args.adaptive_query_length_cap)
        positive_excess = max(len_norm - 1.0, 0.0)
        return 1.0 + progress * confidence * self.args.adaptive_query_length_alpha * positive_excess

    def _on_query_selected(self, prompt_sample: Sample):
        query_key = extract_query_key(prompt_sample)
        stats = self._adaptive_query_stats.setdefault(query_key, QueryStats())
        stats.selected_count += 1
        stats.last_selected_step = self._adaptive_select_step
        self._adaptive_select_step += 1

    def _reward_to_binary_outcome(self, reward_value: float | int | None) -> int | None:
        if reward_value is None:
            return None
        return 1 if float(reward_value) > 0.5 else 0

    def _normalize_rollout_group(self, group: list[Sample] | list[list[Sample]]) -> list[list[Sample]]:
        if not group:
            return []
        if isinstance(group[0], list):
            return [trajectory for trajectory in group if trajectory]
        return [group]  # type: ignore[list-item]

    def update_from_rollout(self, samples: list[list[Sample]], rollout_id: int | None = None):
        if not self._use_adaptive_query_sampling():
            self._adaptive_last_update_status = "skipped"
            self._adaptive_last_updated_queries = 0
            line = (
                f"[AdaptiveQuerySampling] rollout={rollout_id + 1 if rollout_id is not None else 'unknown'} "
                f"skipped reason=disabled_or_no_dataset "
                f"enabled={bool(getattr(self.args, 'enable_adaptive_query_sampling', False))} "
                f"has_dataset={self.dataset is not None}"
            )
            logger.info(line)
            append_progress_log(self.args, line)
            return

        if rollout_id is not None:
            self._adaptive_last_rollout_id = rollout_id

        logger.info(
            "[AdaptiveQuerySampling] rollout_id=%s update_from_rollout_start groups=%d tracked_queries_before=%d",
            rollout_id,
            len(samples),
            len(self._adaptive_query_stats),
        )

        updated_queries = set()
        for group in samples:
            if not group:
                continue

            trajectories = self._normalize_rollout_group(group)  # one query group = multiple sampled trajectories
            if not trajectories:
                continue

            query_key = extract_query_key(trajectories[0][0])
            stats = self._adaptive_query_stats.setdefault(query_key, QueryStats())
            updated_queries.add(query_key)

            for trajectory in trajectories:
                terminal_sample = extract_terminal_sample(trajectory)
                reward_value = terminal_sample.get_reward_value(self.args) if terminal_sample.reward is not None else None
                outcome = self._reward_to_binary_outcome(reward_value)
                if outcome is None:
                    continue
                if outcome == 1:
                    stats.success_count += 1
                else:
                    stats.fail_count += 1

                terminal_outcome = outcome
                if terminal_outcome is None:
                    stats.invalid_traj_count += 1
                elif terminal_outcome == 1:
                    stats.success_traj_count += 1
                else:
                    stats.fail_traj_count += 1

                completed_len = extract_completed_trajectory_length(trajectory)
                if completed_len is None:
                    continue

                stats.completed_traj_count += 1
                stats.completed_len_sum += completed_len
                self._adaptive_global_completed_traj_count += 1
                self._adaptive_global_completed_len_sum += completed_len

                if terminal_outcome == 1:
                    stats.completed_success_traj_count += 1
                    stats.completed_success_len_sum += completed_len
                elif terminal_outcome == 0:
                    stats.completed_fail_traj_count += 1
                    stats.completed_fail_len_sum += completed_len

        if updated_queries:
            global_avg_completed_len = (
                self._adaptive_global_completed_len_sum / self._adaptive_global_completed_traj_count
                if self._adaptive_global_completed_traj_count > 0
                else 0.0
            )
            logger.info(
                "[AdaptiveQuerySampling] rollout_id=%s updated_queries=%d tracked_queries=%d global_avg_completed_len=%.3f",
                rollout_id,
                len(updated_queries),
                len(self._adaptive_query_stats),
                global_avg_completed_len,
            )
            self._adaptive_last_update_status = "updated"
            self._adaptive_last_updated_queries = len(updated_queries)
            if rollout_id is not None:
                summary_record = self._append_adaptive_query_summary_log(rollout_id)
                if summary_record is not None:
                    self._append_adaptive_query_progress_log(summary_record)
        else:
            self._adaptive_last_update_status = "no_updates"
            self._adaptive_last_updated_queries = 0
            line = (
                f"[AdaptiveQuerySampling] rollout={rollout_id + 1 if rollout_id is not None else 'unknown'} "
                "update_from_rollout_no_updates"
            )
            logger.info(line)
            append_progress_log(self.args, line)

    def get_adaptive_query_rollout_metrics(self) -> dict[str, float]:
        enabled = int(bool(getattr(self.args, "enable_adaptive_query_sampling", False)))
        active = int(self._use_adaptive_query_sampling())
        tracked_queries = len(self._adaptive_query_stats)
        global_avg_completed_len = (
            self._adaptive_global_completed_len_sum / self._adaptive_global_completed_traj_count
            if self._adaptive_global_completed_traj_count > 0
            else 0.0
        )

        bucket_counts = {key: 0 for key in ["warmup", "all_fail", "low", "mid", "high", "all_success"]}
        if active:
            for _, stats in self._adaptive_query_stats.items():
                bucket_counts[self._get_success_rate_bucket_name(stats)] += 1

        return {
            "rollout/adaptive_query_sampling/enabled": float(enabled),
            "rollout/adaptive_query_sampling/active": float(active),
            "rollout/adaptive_query_sampling/status_updated": float(self._adaptive_last_update_status == "updated"),
            "rollout/adaptive_query_sampling/status_no_updates": float(
                self._adaptive_last_update_status == "no_updates"
            ),
            "rollout/adaptive_query_sampling/status_skipped": float(self._adaptive_last_update_status == "skipped"),
            "rollout/adaptive_query_sampling/tracked_queries": float(tracked_queries),
            "rollout/adaptive_query_sampling/updated_queries": float(self._adaptive_last_updated_queries),
            "rollout/adaptive_query_sampling/select_step": float(self._adaptive_select_step),
            "rollout/adaptive_query_sampling/global_completed_traj_count": float(
                self._adaptive_global_completed_traj_count
            ),
            "rollout/adaptive_query_sampling/global_avg_completed_len": float(global_avg_completed_len),
            "rollout/adaptive_query_sampling/bucket_warmup": float(bucket_counts["warmup"]),
            "rollout/adaptive_query_sampling/bucket_all_fail": float(bucket_counts["all_fail"]),
            "rollout/adaptive_query_sampling/bucket_low": float(bucket_counts["low"]),
            "rollout/adaptive_query_sampling/bucket_mid": float(bucket_counts["mid"]),
            "rollout/adaptive_query_sampling/bucket_high": float(bucket_counts["high"]),
            "rollout/adaptive_query_sampling/bucket_all_success": float(bucket_counts["all_success"]),
        }

    def _iter_adaptive_query_stat_records(self):
        global_avg_completed_len = (
            self._adaptive_global_completed_len_sum / self._adaptive_global_completed_traj_count
            if self._adaptive_global_completed_traj_count > 0
            else 0.0
        )

        items = sorted(
            self._adaptive_query_stats.items(),
            key=lambda item: (
                -item[1].selected_count,
                -(item[1].success_count + item[1].fail_count),
                item[0],
            ),
        )
        for query_key, stats in items:
            yield {
                "query_key": query_key,
                "selected_count": stats.selected_count,
                "last_selected_step": stats.last_selected_step,
                "success_count": stats.success_count,
                "fail_count": stats.fail_count,
                "total_rewarded_samples": stats.total_rewarded_samples,
                "success_rate": stats.success_rate,
                "success_traj_count": stats.success_traj_count,
                "fail_traj_count": stats.fail_traj_count,
                "invalid_traj_count": stats.invalid_traj_count,
                "completed_traj_count": stats.completed_traj_count,
                "completed_len_sum": stats.completed_len_sum,
                "avg_completed_len": stats.avg_completed_len,
                "completed_success_traj_count": stats.completed_success_traj_count,
                "completed_success_len_sum": stats.completed_success_len_sum,
                "avg_completed_success_len": stats.avg_completed_success_len,
                "completed_fail_traj_count": stats.completed_fail_traj_count,
                "completed_fail_len_sum": stats.completed_fail_len_sum,
                "avg_completed_fail_len": stats.avg_completed_fail_len,
                "global_avg_completed_len": global_avg_completed_len,
            }

    def _build_adaptive_query_summary_record(self, rollout_id: int, records: list[dict]) -> dict:
        tracked_queries = len(records)
        global_avg_completed_len = (
            self._adaptive_global_completed_len_sum / self._adaptive_global_completed_traj_count
            if self._adaptive_global_completed_traj_count > 0
            else 0.0
        )
        bucket_counts = {
            "warmup": 0,
            "all_fail": 0,
            "low": 0,
            "mid": 0,
            "high": 0,
            "all_success": 0,
        }
        for stats in self._adaptive_query_stats.values():
            if stats.total_rewarded_samples < self.args.adaptive_query_warmup_samples:
                bucket_counts["warmup"] += 1
            elif stats.success_rate == 0.0:
                bucket_counts["all_fail"] += 1
            elif stats.success_rate < 0.3:
                bucket_counts["low"] += 1
            elif stats.success_rate < 0.7:
                bucket_counts["mid"] += 1
            elif stats.success_rate < 1.0:
                bucket_counts["high"] += 1
            else:
                bucket_counts["all_success"] += 1

        top_queries = [
            {
                "query_key": record["query_key"],
                "selected_count": record["selected_count"],
                "success_rate": record["success_rate"],
                "avg_completed_len": record["avg_completed_len"],
            }
            for record in records[:10]
        ]
        return {
            "rollout_id": rollout_id,
            "tracked_queries": tracked_queries,
            "select_step": self._adaptive_select_step,
            "global_completed_traj_count": self._adaptive_global_completed_traj_count,
            "global_completed_len_sum": self._adaptive_global_completed_len_sum,
            "global_avg_completed_len": global_avg_completed_len,
            "bucket_counts": bucket_counts,
            "top_queries": top_queries,
        }

    def _append_adaptive_query_summary_log(self, rollout_id: int) -> dict | None:
        if not self._use_adaptive_query_sampling():
            return None

        save_dir = getattr(self.args, "save", None)
        if not save_dir:
            return None

        base_dir = Path(save_dir) / "rollout" / "adaptive_query_sampling"
        base_dir.mkdir(parents=True, exist_ok=True)

        records = list(self._iter_adaptive_query_stat_records())
        summary_record = self._build_adaptive_query_summary_record(rollout_id, records)
        summary_path = base_dir / "summary.jsonl"
        with summary_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(summary_record, ensure_ascii=False, sort_keys=True) + "\n")

        return summary_record

    def _save_adaptive_query_detail_snapshot(self, rollout_id: int) -> None:
        if not self._use_adaptive_query_sampling():
            return

        save_dir = getattr(self.args, "save", None)
        if not save_dir:
            return

        base_dir = Path(save_dir) / "rollout" / "adaptive_query_sampling"
        base_dir.mkdir(parents=True, exist_ok=True)

        detail_path = base_dir / f"query_stats_{rollout_id}.jsonl"
        with detail_path.open("w", encoding="utf-8") as f:
            for record in self._iter_adaptive_query_stat_records():
                f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    def _append_adaptive_query_progress_log(self, summary_record: dict) -> None:
        bucket_counts = summary_record["bucket_counts"]
        line = (
            f"[AdaptiveQuerySampling] rollout={summary_record['rollout_id'] + 1} "
            f"tracked_queries={summary_record['tracked_queries']} "
            f"select_step={summary_record['select_step']} "
            f"completed_trajs={summary_record['global_completed_traj_count']} "
            f"global_avg_completed_len={summary_record['global_avg_completed_len']:.3f} "
            f"buckets=warmup:{bucket_counts['warmup']},all_fail:{bucket_counts['all_fail']},"
            f"low:{bucket_counts['low']},mid:{bucket_counts['mid']},high:{bucket_counts['high']},"
            f"all_success:{bucket_counts['all_success']}"
        )
        logger.info(line)
        append_progress_log(self.args, line)

    def save(self, rollout_id):
        if not self.args.rollout_global_dataset:
            return

        state_dict = {
            "sample_offset": self.sample_offset,
            "epoch_id": self.epoch_id,
            "sample_group_index": self.sample_group_index,
            "sample_index": self.sample_index,
            "metadata": self.metadata,
        }
        if self._use_adaptive_query_sampling():
            state_dict["adaptive_query_sampling"] = {
                "query_stats": {key: stats.to_dict() for key, stats in self._adaptive_query_stats.items()},
                "select_step": self._adaptive_select_step,
                "global_completed_traj_count": self._adaptive_global_completed_traj_count,
                "global_completed_len_sum": self._adaptive_global_completed_len_sum,
                "last_rollout_id": self._adaptive_last_rollout_id,
                "rng_state": self._adaptive_rng.getstate(),
            }
        path = os.path.join(self.args.save, f"rollout/global_dataset_state_dict_{rollout_id}.pt")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(state_dict, path)
        if self._use_adaptive_query_sampling():
            self._save_adaptive_query_detail_snapshot(rollout_id)

    def load(self, rollout_id=None):
        if not self.args.rollout_global_dataset:
            return

        if self.args.load is None:
            return

        path = os.path.join(self.args.load, f"rollout/global_dataset_state_dict_{rollout_id}.pt")
        if not os.path.exists(path):
            logger.info(f"Checkpoint {path} does not exist.")
            return

        logger.info(f"load metadata from {path}")
        logger.info(f"load metadata: {self.metadata}")
        state_dict = torch.load(path)
        self.sample_offset = state_dict.get("sample_offset", 0)
        self.epoch_id = state_dict.get("epoch_id", 0)
        self.sample_group_index = state_dict.get("sample_group_index", 0)
        self.sample_index = state_dict.get("sample_index", 0)
        self.metadata = state_dict.get("metadata", {})

        if self.args.rollout_global_dataset and self.args.rollout_shuffle:
            self.dataset.shuffle(self.epoch_id)

        if self._use_adaptive_query_sampling():
            adaptive_state = state_dict.get("adaptive_query_sampling", {})
            self._adaptive_query_stats = {
                key: QueryStats.from_dict(value) for key, value in adaptive_state.get("query_stats", {}).items()
            }
            self._adaptive_select_step = adaptive_state.get("select_step", 0)
            self._adaptive_global_completed_traj_count = adaptive_state.get("global_completed_traj_count", 0)
            self._adaptive_global_completed_len_sum = adaptive_state.get("global_completed_len_sum", 0.0)
            self._adaptive_last_rollout_id = adaptive_state.get("last_rollout_id", -1)
            rng_state = adaptive_state.get("rng_state")
            if rng_state is not None:
                self._adaptive_rng.setstate(rng_state)

    def __len__(self) -> int:
        return len(self.dataset)


class RolloutDataSourceWithBuffer(RolloutDataSource):
    def __init__(self, args):
        super().__init__(args)
        self.buffer = []
        if self.args.buffer_filter_path is None:
            self.buffer_filter = pop_first
        else:
            self.buffer_filter = load_function(self.args.buffer_filter_path)

    def get_samples(self, num_samples: int) -> list[list[Sample]]:
        """
        Return num_samples samples
        """

        samples = self._get_samples_from_buffer(num_samples)
        num_samples -= len(samples)

        if num_samples == 0:
            return samples

        samples += super().get_samples(num_samples=num_samples)
        return samples

    def _get_samples_from_buffer(self, num_samples: int) -> list[list[Sample]]:
        if len(self.buffer) == 0 or num_samples == 0:
            return []

        samples = self.buffer_filter(self.args, None, self.buffer, num_samples)
        return samples

    def add_samples(self, samples: list[list[Sample]]):
        """
        Add a sample group to buffer.
        """
        if not samples:
            return
        assert isinstance(samples, list), f"samples must be a list, got {type(samples)}"
        assert isinstance(samples[0], list), f"the elements of samples must be list, got {type(samples[0])}"
        for i in range(0, len(samples)):
            assert (
                len(samples[i]) == self.args.n_samples_per_prompt
            ), f"the length of the elements of samples must be equal to n_samples_per_prompt, got {len(samples[i])} != {self.args.n_samples_per_prompt}"
            group = samples[i]  # type: ignore
            self.buffer.append(group)

    # TODO remove
    def update_metadata(self, metadata: dict):
        self.metadata.update(metadata)

    # TODO remove
    def get_metadata(self):
        return self.metadata

    def get_buffer_length(self):
        return len(self.buffer)


def pop_first(args, rollout_id, buffer: list[list[Sample]], num_samples: int) -> list[list[Sample]]:
    num_to_pop = min(len(buffer), num_samples)
    samples = buffer[:num_to_pop]
    del buffer[:num_to_pop]
    return samples
