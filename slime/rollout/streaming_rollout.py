"""Continuous ("streaming") rollout for non-colocated training.

The stock ``sglang_rollout.generate_rollout`` runs one wave of groups and returns when
the batch is full, so every step pays rollout + train sequentially and the slowest
trajectory of the wave gates the whole step. This module keeps a fixed number of GRPO
groups in flight *at all times* on the background event loop (the same loop
``slime.utils.async_utils.run`` uses, so the pool survives across ``generate`` calls)
and hands the trainer the oldest ``rollout_batch_size`` finished groups on each call.
Trajectories keep generating while the actor trains, so the step is
``max(train, generate)`` instead of their sum.

Requires the inference engines on their own GPUs (no ``--colocate``): weights are
pushed with ``UpdateWeightFromDistributed`` (pause → broadcast → continue) while the
pool keeps running. Turns in flight during a sync are aborted by SGLang and retried by
``openwebrl.generate_browser._run_inference_step``.

Weight staleness: ``train.py`` calls ``RolloutManager.set_weight_version`` after every
sync; each turn records the version it was sampled under (``metadata["weight_version"]``)
and the batch metrics report the lag. ``SLIME_STREAM_MAX_VERSION_LAG`` (default -1 = off)
drops whole trajectories whose *last* turn is older than that many syncs.

Env knobs (read once at pool start):
  SLIME_STREAM_INFLIGHT_GROUPS      groups kept in flight (default 1.5 x rollout_batch_size)
  SLIME_STREAM_MAX_BACKLOG_GROUPS   stop submitting when this many kept groups wait (default rollout_batch_size)
  SLIME_STREAM_MIN_INFLIGHT_GROUPS  keep at least this many groups in flight even when the backlog is full (default 0)
  SLIME_STREAM_MAX_VERSION_LAG      drop trajectories older than this many syncs (default -1 = keep all)
  SLIME_STREAM_SHARDS               run the groups in N worker processes instead of this one (default 0 = off)
"""

import asyncio
import json
import logging
import os
import time
from argparse import Namespace
from collections import deque
from typing import Any

import numpy as np
import ray

from slime.rollout.base_types import RolloutFnEvalOutput, RolloutFnTrainOutput
from slime.rollout.filter_hub.base_types import MetricGatherer, call_dynamic_filter
from slime.rollout.sglang_rollout import GenerateState, _dump_wave_trajectories, eval_rollout, generate_and_rm_group
from slime.rollout.streaming_shard import shard_count
from slime.utils.async_utils import run
from slime.utils.logging_utils import append_progress_log
from slime.utils.misc import load_function
from slime.utils.types import Sample

__all__ = ["generate_rollout", "current_weight_version", "set_weight_version", "StreamingPool"]

logger = logging.getLogger(__name__)

# Number of weight syncs the rollout engines have received. Lives in the RolloutManager
# actor process, where both the pool and openwebrl.generate_browser run.
_WEIGHT_VERSION = 0


def current_weight_version() -> int:
    return _WEIGHT_VERSION


def set_weight_version(version: int) -> None:
    global _WEIGHT_VERSION
    _WEIGHT_VERSION = int(version)
    logger.info("streaming rollout: weight version -> %d", _WEIGHT_VERSION)
    pool = StreamingPool._instance
    if pool is not None and pool.shards:
        pool.set_weight_version_on_shards(_WEIGHT_VERSION)


def _last(traj):
    return traj[-1] if isinstance(traj, list) else traj


def _first(traj):
    return traj[0] if isinstance(traj, list) else traj


def _group_index(group):
    return _first(group[0]).index


def _env_int(name: str, default: int) -> int:
    val = os.environ.get(name)
    return int(val) if val not in (None, "") else default


class StreamingPool:
    """Keeps ``inflight_groups`` groups generating at all times; ``next_batch`` drains
    the oldest finished groups. One instance per process, created lazily on the
    background loop by ``generate_rollout``."""

    _instance: "StreamingPool | None" = None

    def __init__(self, args: Namespace, data_source: Any, group_fn=None) -> None:
        self.args = args
        self.data_source = data_source
        # Shards: run each group in a worker process with its own event loop and
        # Playwright driver. The pool keeps the data source, batching and metrics.
        self.shards = []
        self._next_shard = 0
        n_shards = shard_count()
        if group_fn is None and n_shards > 1:
            from slime.rollout.streaming_shard import make_shards

            self.shards = make_shards(args, n_shards)
        # injectable for unit tests
        self._group_fn = group_fn or (self._run_group_on_shard if self.shards else (
            lambda group: generate_and_rm_group(
                args, group, sampling_params=GenerateState(args).sampling_params.copy(), evaluation=False
            )
        ))
        bs = args.rollout_batch_size
        self.inflight_groups = _env_int("SLIME_STREAM_INFLIGHT_GROUPS", int(bs * 1.5))
        self.max_backlog = _env_int("SLIME_STREAM_MAX_BACKLOG_GROUPS", bs)
        self.min_inflight = _env_int("SLIME_STREAM_MIN_INFLIGHT_GROUPS", 0)
        self.max_version_lag = _env_int("SLIME_STREAM_MAX_VERSION_LAG", -1)
        self.dynamic_filter = (
            load_function(args.dynamic_sampling_filter_path) if args.dynamic_sampling_filter_path is not None else None
        )

        self.pending: set[asyncio.Task] = set()
        self.ready: deque[list] = deque()  # kept groups, oldest first
        self.finished_since_drain: list[list] = []  # every finished group (pre-filter) for the jsonl dump
        self.metric_gatherer = MetricGatherer()
        self.paused = False
        self.stopped = False
        self.start_ts = time.time()
        self.last_drain_mono = time.monotonic()
        self.n_submitted = 0
        self.n_finished = 0
        self.n_kept = 0
        self.n_errors = 0
        self._wake = asyncio.Event()
        # Event-loop starvation probe: every 0.5 s of sleep, how late did we wake?
        # All 240 browser sessions + the processor run on this one loop, so a large
        # lag here (not Browserbase) is what makes step_env grow with concurrency.
        self._loop_lag: list[float] = []
        self._loop_lag_task = asyncio.create_task(self._probe_loop_lag(), name="streaming_pool_lag")
        self._task = asyncio.create_task(self._run(), name="streaming_pool")
        logger.info(
            "streaming pool started: inflight_groups=%d max_backlog=%d min_inflight=%d max_version_lag=%d shards=%d",
            self.inflight_groups, self.max_backlog, self.min_inflight, self.max_version_lag, len(self.shards),
        )

    async def _run_group_on_shard(self, group: list):
        """Round-robin one group to a shard actor; awaiting the ObjectRef behaves like
        awaiting the local coroutine it replaces."""
        shard = self.shards[self._next_shard % len(self.shards)]
        self._next_shard += 1
        return await shard.run_group.remote(group)

    def set_weight_version_on_shards(self, version: int) -> None:
        for shard in self.shards:
            shard.set_weight_version.remote(version)

    async def _probe_loop_lag(self) -> None:
        loop = asyncio.get_running_loop()
        while not self.stopped:
            t0 = loop.time()
            await asyncio.sleep(0.5)
            self._loop_lag.append(loop.time() - t0 - 0.5)

    def _loop_lag_metrics(self) -> dict[str, float]:
        lag = np.array(self._loop_lag or [0.0], dtype=float)
        n = lag.size
        self._loop_lag = []
        return {
            "rollout/stream/loop_lag_ms_mean": float(lag.mean() * 1e3),
            "rollout/stream/loop_lag_ms_p99": float(np.percentile(lag, 99) * 1e3),
            "rollout/stream/loop_lag_ms_max": float(lag.max() * 1e3),
            # fraction of wall time the loop was busy (not sleeping when it should have)
            "rollout/stream/loop_busy_frac": float(lag.sum() / max(lag.sum() + 0.5 * n, 1e-6)),
        }

    # ------------------------------------------------------------------ producer
    def _want_submit(self) -> bool:
        return (
            not self.paused
            and not self.stopped
            and len(self.pending) < self.inflight_groups
            and (len(self.ready) < self.max_backlog or len(self.pending) < self.min_inflight)
        )

    def _submit_one(self) -> None:
        groups = self.data_source.get_samples(1)
        for group in groups:
            now = time.time()
            for sample in group:
                sample.metadata = sample.metadata or {}
                sample.metadata["stream_submit_ts"] = now
                sample.metadata["submit_weight_version"] = _WEIGHT_VERSION
            self.pending.add(asyncio.create_task(self._group_fn(group)))
            self.n_submitted += 1

    async def _run(self) -> None:
        last_log = time.monotonic()
        while not self.stopped:
            while self._want_submit():
                self._submit_one()
            if not self.pending:
                await asyncio.sleep(0.5)
                continue
            done, self.pending = await asyncio.wait(self.pending, timeout=1.0, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                try:
                    group = task.result()
                except asyncio.CancelledError:
                    continue
                except Exception as exc:  # a crashed group must not kill the pool
                    self.n_errors += 1
                    logger.exception("streaming pool: group task failed (%s); resubmitting", type(exc).__name__)
                    continue
                self._on_group_done(group)
            if done:
                self._wake.set()
            now = time.monotonic()
            if now - last_log >= 30:
                self._log("tick")
                last_log = now

    def _on_group_done(self, group: list) -> None:
        self.n_finished += 1
        self.finished_since_drain.append(group)
        out = call_dynamic_filter(self.dynamic_filter, self.args, group)
        if not out.keep:
            self.metric_gatherer.on_dynamic_filter_drop(reason=out.reason)
            return
        filtered = out.samples if out.samples is not None else group
        if len(filtered) == 0:
            self.metric_gatherer.on_dynamic_filter_drop(reason="empty_group_after_filter")
            return
        for traj in filtered:
            _last(traj).metadata["stream_ready_ts"] = time.time()
            _last(traj).metadata["ready_weight_version"] = _WEIGHT_VERSION
        self.ready.append(filtered)
        self.n_kept += 1

    def _log(self, event: str) -> None:
        line = (
            f"[StreamPool] event={event} inflight={len(self.pending)} ready={len(self.ready)} "
            f"submitted={self.n_submitted} finished={self.n_finished} kept={self.n_kept} errors={self.n_errors} "
            f"weight_version={_WEIGHT_VERSION} paused={int(self.paused)} elapsed_secs={time.time() - self.start_ts:.0f}"
        )
        logger.info(line)
        append_progress_log(self.args, line)

    # ------------------------------------------------------------------ consumer
    async def next_batch(self, rollout_id: int) -> RolloutFnTrainOutput:
        target = self.args.rollout_batch_size
        wait_start = time.monotonic()
        last_log = wait_start
        while len(self.ready) < target:
            self._wake.clear()
            try:
                await asyncio.wait_for(self._wake.wait(), timeout=10)
            except asyncio.TimeoutError:
                pass
            if time.monotonic() - last_log >= 30:
                self._log(f"wait rollout_id={rollout_id} need={target}")
                last_log = time.monotonic()
        wait_secs = time.monotonic() - wait_start
        now_mono = time.monotonic()
        batch_wall = now_mono - self.last_drain_mono
        self.last_drain_mono = now_mono

        batch = [self.ready.popleft() for _ in range(target)]
        n_before = sum(len(g) for g in batch)
        if self.max_version_lag >= 0:
            batch = self._drop_stale(batch)
        n_after = sum(len(g) for g in batch)

        metrics = self._batch_metrics(batch, rollout_id, wait_secs, batch_wall)
        metrics["rollout/stream/trajectories_dropped_stale"] = n_before - n_after
        # per-trajectory jsonl + group-outcome counts, for every group finished since the last drain
        finished, self.finished_since_drain = self.finished_since_drain, []
        kept_groups = batch + list(self.ready)
        metrics.update(_dump_wave_trajectories(self.args, rollout_id, finished, kept_groups, self.start_ts))
        metrics.update(self.metric_gatherer.collect())
        self.metric_gatherer = MetricGatherer()
        self._log(f"drain rollout_id={rollout_id} batch_wall={batch_wall:.0f}s wait={wait_secs:.0f}s")

        data = sorted(batch, key=_group_index)
        if self.args.rollout_sample_filter_path is not None:
            load_function(self.args.rollout_sample_filter_path)(self.args, data)
        return RolloutFnTrainOutput(samples=data, metrics=metrics)

    def _drop_stale(self, batch: list[list]) -> list[list]:
        out = []
        for group in batch:
            kept = [
                traj for traj in group
                if _WEIGHT_VERSION - _last(traj).metadata.get("weight_version", _WEIGHT_VERSION) <= self.max_version_lag
            ]
            if kept:
                out.append(kept)
        return out

    def _batch_metrics(self, batch: list[list], rollout_id: int, wait_secs: float, batch_wall: float) -> dict[str, float]:
        lags, turn_lags, traj_wall, queue_secs, n_turns = [], [], [], [], 0
        now = time.time()
        for group in batch:
            for traj in group:
                last = _last(traj)
                md = last.metadata or {}
                lags.append(_WEIGHT_VERSION - md.get("weight_version", _WEIGHT_VERSION))
                if "traj_wall_secs" in md:
                    traj_wall.append(md["traj_wall_secs"])
                if "stream_ready_ts" in md:
                    queue_secs.append(now - md["stream_ready_ts"])
                turns = traj if isinstance(traj, list) else [traj]
                n_turns += len(turns)
                for t in turns:
                    turn_lags.append(_WEIGHT_VERSION - (t.metadata or {}).get("weight_version", _WEIGHT_VERSION))
        lags = np.array(lags or [0], dtype=float)
        turn_lags = np.array(turn_lags or [0], dtype=float)
        elapsed_h = max(now - self.start_ts, 1e-6) / 3600
        m = {
            "rollout/stream/batch_wall_secs": batch_wall,
            "rollout/stream/wait_secs": wait_secs,
            "rollout/stream/inflight_groups": len(self.pending),
            "rollout/stream/backlog_groups": len(self.ready),
            "rollout/stream/groups": len(batch),
            "rollout/stream/trajectories": int(lags.size),
            "rollout/stream/turns": n_turns,
            "rollout/stream/weight_version": _WEIGHT_VERSION,
            "rollout/stream/version_lag_mean": float(lags.mean()),
            "rollout/stream/version_lag_max": float(lags.max()),
            "rollout/stream/version_lag_frac_ge2": float((lags >= 2).mean()),
            "rollout/stream/turn_version_lag_mean": float(turn_lags.mean()),
            "rollout/stream/turn_version_lag_frac_ge1": float((turn_lags >= 1).mean()),
            "rollout/stream/traj_wall_secs_mean": float(np.mean(traj_wall)) if traj_wall else 0.0,
            "rollout/stream/traj_wall_secs_p90": float(np.percentile(traj_wall, 90)) if traj_wall else 0.0,
            "rollout/stream/queue_secs_mean": float(np.mean(queue_secs)) if queue_secs else 0.0,
            "rollout/stream/queue_secs_max": float(np.max(queue_secs)) if queue_secs else 0.0,
            "rollout/stream/finished_groups_total": self.n_finished,
            "rollout/stream/finished_groups_per_hour": self.n_finished / elapsed_h,
            "rollout/stream/group_errors_total": self.n_errors,
            **self._loop_lag_metrics(),
        }
        logger.info("rollout %d stream batch: %s", rollout_id, json.dumps({k.split('/')[-1]: round(v, 3) for k, v in m.items()}))
        return m

    # ------------------------------------------------------------------ control
    def pause(self) -> None:
        self.paused = True

    def resume(self) -> None:
        self.paused = False

    async def stop(self) -> None:
        self.stopped = True
        for task in list(self.pending):
            task.cancel()
        if self.pending:
            await asyncio.gather(*self.pending, return_exceptions=True)
        self._task.cancel()
        for shard in self.shards:
            try:
                ray.get(shard.stop.remote(), timeout=60)
                ray.kill(shard)
            except Exception as exc:
                logger.warning("shard stop failed: %s", exc)


def _get_pool(args: Namespace, data_source: Any) -> StreamingPool:
    if StreamingPool._instance is None:
        StreamingPool._instance = StreamingPool(args, data_source)
    return StreamingPool._instance


async def _drain_pool_for_eval(pool: StreamingPool, timeout_secs: float) -> None:
    """Wait for the in-flight groups to finish before eval starts.

    Pausing only stops new submissions; the groups already running keep their browser
    sessions open and eval opens one more per task on top of them. Browserbase allows
    250 live sessions per account, so an overlap costs both sides their sessions.
    Drained groups stay in the pool's backlog, so nothing is thrown away.
    """
    deadline = time.monotonic() + timeout_secs
    while pool.pending and time.monotonic() < deadline:
        await asyncio.wait(list(pool.pending), timeout=5, return_when=asyncio.FIRST_COMPLETED)
    if pool.pending:
        logger.warning(
            "eval starting with %d groups still in flight after a %.0fs drain",
            len(pool.pending), timeout_secs,
        )


async def _eval_with_pool_paused(args: Namespace, rollout_id: int):
    pool = StreamingPool._instance
    if pool is not None:
        pool.pause()
        if _env_int("SLIME_STREAM_DRAIN_BEFORE_EVAL", 1):
            await _drain_pool_for_eval(pool, _env_int("SLIME_STREAM_EVAL_DRAIN_SECS", 900))
    try:
        return await eval_rollout(args, rollout_id)
    finally:
        if pool is not None:
            pool.resume()


async def _train_batch(args: Namespace, rollout_id: int, data_source: Any) -> RolloutFnTrainOutput:
    return await _get_pool(args, data_source).next_batch(rollout_id)


def generate_rollout(
    args: Namespace, rollout_id: int, data_source: Any, evaluation: bool = False
) -> RolloutFnTrainOutput | RolloutFnEvalOutput:
    """Drop-in for ``--rollout-function-path``. Train calls drain the pool; eval calls pause
    new submissions (in-flight groups keep going) and run the stock eval."""
    assert args.rollout_global_dataset
    assert not getattr(args, "colocate", False), "streaming rollout needs dedicated rollout GPUs (drop --colocate)"
    if evaluation:
        output, _ = run(_eval_with_pool_paused(args, rollout_id))
        return output
    return run(_train_batch(args, rollout_id, data_source))


def stop_pool() -> None:
    if StreamingPool._instance is not None:
        run(StreamingPool._instance.stop())
        StreamingPool._instance = None
