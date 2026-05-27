import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any

from slime.utils.types import Sample


@dataclass
class QueryStats:
    selected_count: int = 0
    last_selected_step: int | None = None
    success_count: int = 0
    fail_count: int = 0
    success_traj_count: int = 0
    fail_traj_count: int = 0
    invalid_traj_count: int = 0
    completed_traj_count: int = 0
    completed_len_sum: float = 0.0
    completed_success_traj_count: int = 0
    completed_success_len_sum: float = 0.0
    completed_fail_traj_count: int = 0
    completed_fail_len_sum: float = 0.0

    @property
    def total_rewarded_samples(self) -> int:
        return self.success_count + self.fail_count

    @property
    def success_rate(self) -> float:
        total = self.total_rewarded_samples
        return self.success_count / total if total > 0 else 0.0

    @property
    def avg_completed_len(self) -> float:
        return self.completed_len_sum / self.completed_traj_count if self.completed_traj_count > 0 else 0.0

    @property
    def avg_completed_success_len(self) -> float:
        return (
            self.completed_success_len_sum / self.completed_success_traj_count
            if self.completed_success_traj_count > 0
            else 0.0
        )

    @property
    def avg_completed_fail_len(self) -> float:
        return (
            self.completed_fail_len_sum / self.completed_fail_traj_count if self.completed_fail_traj_count > 0 else 0.0
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "QueryStats":
        return QueryStats(**data)


def extract_query_key(sample: Sample) -> str:
    metadata = sample.metadata or {}
    task_id = metadata.get("task_id")
    if task_id:
        return str(task_id)

    payload = {
        "prompt": sample.prompt,
        "label": sample.label,
        "metadata": metadata,
    }
    serialized = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    digest = hashlib.md5(serialized.encode("utf-8")).hexdigest()
    return f"prompt:{digest}"


def extract_terminal_sample(group: list[Sample]) -> Sample:
    last_turn_samples = [sample for sample in group if bool((sample.metadata or {}).get("is_last_turn", False))]
    if last_turn_samples:
        return max(last_turn_samples, key=lambda sample: int((sample.metadata or {}).get("turn_index", -1)))

    turn_samples = [sample for sample in group if "turn_index" in (sample.metadata or {})]
    if turn_samples:
        return max(turn_samples, key=lambda sample: int((sample.metadata or {}).get("turn_index", -1)))

    return group[-1]


def extract_completed_trajectory_length(group: list[Sample]) -> int | None:
    terminal_sample = extract_terminal_sample(group)
    if terminal_sample.status != Sample.Status.COMPLETED:
        return None

    metadata = terminal_sample.metadata or {}
    total_steps = metadata.get("total_steps")
    if isinstance(total_steps, int) and total_steps > 0:
        return total_steps

    if isinstance(terminal_sample.response_length, int) and terminal_sample.response_length > 0:
        return terminal_sample.response_length

    return None
