"""Shard the streaming rollout across several worker processes.

Every browser session used to run on the RolloutManager's single event loop, together
with the Playwright driver, the tokenizer/processor calls and the reward requests. At
240 sessions that loop lagged 450-570 ms per await and one environment step cost 16 s
that takes 0.5 s standalone -- the controller, not the page, set the pace.

``StreamingShardActor`` is a plain CPU Ray actor that runs ``generate_and_rm_group`` for
one group on its own event loop, with its own Playwright driver and its own copy of the
generation state. ``StreamingPool`` keeps the data source, the submit/drain logic and
the metrics, and routes each group to a shard round-robin; the returned ObjectRef is
awaited exactly like the local task it replaces.

Enable with ``SLIME_STREAM_SHARDS=<n>`` (0/1 = unsharded, the old behaviour).

Per-process limits are divided by the shard count so the totals stay what the launcher
asked for: ``browser_rollout_concurrency`` (browser sessions) and
``sglang_server_concurrency`` (in-flight inference requests).
"""

import logging
import os
from argparse import Namespace

import ray

logger = logging.getLogger(__name__)

__all__ = ["StreamingShardActor", "make_shards", "shard_count"]


def shard_count() -> int:
    val = os.environ.get("SLIME_STREAM_SHARDS")
    try:
        return max(0, int(val)) if val not in (None, "") else 0
    except ValueError:
        return 0


def _split_args(args: Namespace, n_shards: int, shard_id: int) -> Namespace:
    """Copy of args with the per-process concurrency limits divided by the shard count."""
    import copy

    a = copy.copy(args)
    conc = getattr(a, "browser_rollout_concurrency", 0) or 0
    if conc > 0:
        a.browser_rollout_concurrency = max(1, conc // n_shards)
    sconc = getattr(a, "sglang_server_concurrency", 0) or 0
    if sconc > 0:
        a.sglang_server_concurrency = max(1, sconc // n_shards)
    a.stream_shard_id = shard_id
    return a


@ray.remote(num_cpus=4)
class StreamingShardActor:
    """Runs rollout groups on its own process/event loop."""

    def __init__(self, args: Namespace, n_shards: int, shard_id: int, env: dict[str, str] | None = None) -> None:
        for k, v in (env or {}).items():
            os.environ[k] = v
        # Each shard releases only the Browserbase sessions it opened; the shared marker
        # directory is swept by whichever job starts, so keep them apart.
        os.environ["BROWSERBASE_MANIFEST_DIR"] = f"/tmp/bb_sessions_shard{shard_id}"
        # The per-process gate in generate_browser is read from the env first.
        self.args = _split_args(args, n_shards, shard_id)
        if getattr(self.args, "browser_rollout_concurrency", 0):
            os.environ["SLIME_BROWSER_ROLLOUT_CONCURRENCY"] = str(self.args.browser_rollout_concurrency)
        # Browserbase's slot pool and spawn-rate throttle are per process too; without
        # dividing them N shards would each allow the whole account limit (250 live
        # sessions, 100 creates/min) and the provider would answer 429.
        for var, default, floor in (("BROWSERBASE_MAX_SESSIONS", 240, 1), ("BROWSERBASE_SPAWN_PER_MINUTE", 90, 1)):
            try:
                total = int(os.environ.get(var, default))
            except ValueError:
                total = default
            os.environ[var] = str(max(floor, total // n_shards))
        self.shard_id = shard_id
        # Each worker process needs its own httpx client; the one the RolloutManager
        # builds in its own process is not inherited (the shard would post on None).
        from slime.utils.http_utils import init_http_client

        init_http_client(self.args)
        try:
            from slime.utils.logging_utils import configure_logger

            configure_logger()
        except Exception:
            logging.basicConfig(level=logging.INFO)
        logger.info(
            "streaming shard %d/%d: browser_concurrency=%s sglang_concurrency=%s bb_sessions=%s bb_spawn=%s",
            shard_id, n_shards, self.args.browser_rollout_concurrency, self.args.sglang_server_concurrency,
            os.environ["BROWSERBASE_MAX_SESSIONS"], os.environ["BROWSERBASE_SPAWN_PER_MINUTE"],
        )

    def ping(self) -> int:
        return self.shard_id

    def set_weight_version(self, version: int) -> None:
        from slime.rollout.streaming_rollout import set_weight_version

        set_weight_version(version)

    def run_group(self, group: list):
        """Generate + reward one group. Returns the finished group (or raises)."""
        from slime.rollout.sglang_rollout import GenerateState, generate_and_rm_group
        from slime.utils.async_utils import run

        state = GenerateState(self.args)
        return run(
            generate_and_rm_group(self.args, group, sampling_params=state.sampling_params.copy(), evaluation=False)
        )

    def stop(self) -> None:
        try:
            from openwebrl.env.browserbase_env import cleanup_existing_browserbase_sessions
            from slime.utils.async_utils import run

            run(cleanup_existing_browserbase_sessions({}))
        except Exception as exc:  # best effort
            logger.warning("shard %d cleanup failed: %s", self.shard_id, exc)


def make_shards(args: Namespace, n_shards: int) -> list:
    """Create ``n_shards`` shard actors and wait until they are alive."""
    passthrough = {
        k: v
        for k, v in os.environ.items()
        if k.startswith(("SLIME_BROWSER_", "BROWSERBASE_", "SLIME_MM_", "SLIME_ENCODE_", "SLIME_HF_", "SLIME_MW_", "MOLMOWEB_"))
        and k not in ("SLIME_BROWSER_ROLLOUT_CONCURRENCY", "BROWSERBASE_MANIFEST_DIR")
    }
    shards = [
        StreamingShardActor.options(name=f"stream_shard_{i}", max_concurrency=256).remote(
            args, n_shards, i, passthrough
        )
        for i in range(n_shards)
    ]
    ray.get([s.ping.remote() for s in shards])
    logger.info("streaming rollout sharded over %d worker processes", n_shards)
    return shards
