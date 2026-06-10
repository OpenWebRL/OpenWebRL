"""
Sandbox-based Browser Environment Client

Provides ``SandboxWebEnv`` -- a drop-in replacement for ``WebEnv`` that
communicates with a K8s sandbox pod through the sandbox orchestrator's
``exec()`` + ``curl`` commands.

Each generation creates a fresh sandbox via ``create_sandbox_env()`` and
deletes it when ``env.exit()`` is called.
"""

import argparse
import asyncio
import base64
import json
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Import sandbox client from the sandbox/ directory
# ---------------------------------------------------------------------------
_SANDBOX_CLIENT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "sandbox")
)
if _SANDBOX_CLIENT_DIR not in sys.path:
    sys.path.insert(0, _SANDBOX_CLIENT_DIR)

try:
    from client.sandbox_client import AsyncSandboxClient, AsyncSandboxInstance
except ModuleNotFoundError:
    # The Orchard sandbox client is only required for sandbox mode. In
    # local_process mode (or any environment where it isn't installed) the
    # module must still import so utilities like `--cleanup` can run and
    # no-op gracefully instead of crashing with ModuleNotFoundError.
    AsyncSandboxClient = None
    AsyncSandboxInstance = None

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_ENV_SERVER_PORT = 8100
_CURL_TIMEOUT = 300  # seconds for curl requests
_EXEC_CURL_MAX_RETRIES = 3  # retries for exec/curl failures inside sandbox
_EXEC_CURL_RETRY_DELAY = 2  # initial backoff delay in seconds
_ENV_EXIT_REQUEST_TIMEOUT = 15  # seconds to wait for best-effort /exit before forcing cleanup
_SANDBOX_HEARTBEAT_INTERVAL = 20  # seconds

_ENV_SERVER_STARTUP_TIMEOUT = 120  # seconds
_ENV_SERVER_STARTUP_POLL = 3  # seconds between health checks
_SLOT_READY_STUCK_THRESHOLD_SECS = 30.0
_SLOT_PROVISIONING_STUCK_THRESHOLD_SECS = 120.0
_SLOT_EXIT_STUCK_THRESHOLD_SECS = 20.0

_SLOT_REGISTRY: dict[str, dict[str, Any]] = {}


def _get_env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError:
        logger.warning("Invalid integer for %s=%r; using default=%s", name, value, default)
        return default


def _get_exec_curl_timeout(path: str) -> int:
    if path == "/step":
        return _get_env_int("SLIME_BROWSER_STEP_CURL_TIMEOUT", _CURL_TIMEOUT)
    return _get_env_int("SLIME_BROWSER_EXEC_CURL_TIMEOUT", _CURL_TIMEOUT)


def _get_exec_curl_max_retries(path: str) -> int:
    if path == "/step":
        return max(1, _get_env_int("SLIME_BROWSER_STEP_MAX_RETRIES", _EXEC_CURL_MAX_RETRIES))
    return max(1, _get_env_int("SLIME_BROWSER_EXEC_CURL_MAX_RETRIES", _EXEC_CURL_MAX_RETRIES))


def _get_exec_curl_retry_delay(path: str) -> int:
    if path == "/step":
        return max(0, _get_env_int("SLIME_BROWSER_STEP_RETRY_DELAY_SECS", _EXEC_CURL_RETRY_DELAY))
    return max(0, _get_env_int("SLIME_BROWSER_EXEC_CURL_RETRY_DELAY_SECS", _EXEC_CURL_RETRY_DELAY))


def _get_sandbox_heartbeat_interval() -> int:
    return max(1, _get_env_int("SLIME_BROWSER_SANDBOX_HEARTBEAT_INTERVAL_SECS", _SANDBOX_HEARTBEAT_INTERVAL))


def _is_missing_sandbox_error(exc: BaseException) -> bool:
    status = getattr(exc, "status", None)
    if status == 404:
        return True
    text = str(exc)
    return "404" in text and "/sandboxes/" in text and "/exec" in text


def _slot_registry_update(sandbox_id: str, **updates: Any) -> None:
    now = time.monotonic()
    record = _SLOT_REGISTRY.get(sandbox_id, {})
    if not record:
        record = {
            "sandbox_id": sandbox_id,
            "created_monotonic": now,
        }
    record.update(updates)
    record["last_event_monotonic"] = now
    _SLOT_REGISTRY[sandbox_id] = record


def _slot_registry_remove(sandbox_id: str) -> None:
    _SLOT_REGISTRY.pop(sandbox_id, None)


def _summarize_slot_registry(slot_snapshot: dict[str, float | int]) -> dict[str, Any]:
    now = time.monotonic()
    active_records = list(_SLOT_REGISTRY.values())

    running = 0
    provisioning = 0
    ready_not_initialized = 0
    exit_started = 0
    ready_stuck = 0
    provisioning_stuck = 0
    exit_stuck = 0
    oldest_ready_stuck: list[tuple[float, str]] = []

    for record in active_records:
        phase = record.get("phase", "unknown")
        phase_started = float(record.get("phase_started_monotonic", record.get("created_monotonic", now)))
        age_secs = max(0.0, now - phase_started)

        if phase == "initialized":
            running += 1
        elif phase in {"slot_acquired", "created"}:
            provisioning += 1
            if age_secs >= _SLOT_PROVISIONING_STUCK_THRESHOLD_SECS:
                provisioning_stuck += 1
        elif phase == "ready":
            ready_not_initialized += 1
            if age_secs >= _SLOT_READY_STUCK_THRESHOLD_SECS:
                ready_stuck += 1
                oldest_ready_stuck.append((age_secs, record["sandbox_id"]))
        elif phase == "exit_started":
            exit_started += 1
            if age_secs >= _SLOT_EXIT_STUCK_THRESHOLD_SECS:
                exit_stuck += 1

    oldest_ready_stuck.sort(reverse=True)
    ready_examples = [f"{sid}:{age:.1f}s" for age, sid in oldest_ready_stuck[:3]]
    tracked_active = len(active_records)
    untracked_active = max(0, int(slot_snapshot["active"]) - tracked_active)
    return {
        "tracked_active": tracked_active,
        "untracked_active": untracked_active,
        "running": running,
        "provisioning": provisioning,
        "ready_not_initialized": ready_not_initialized,
        "ready_stuck": ready_stuck,
        "provisioning_stuck": provisioning_stuck,
        "exit_started": exit_started,
        "exit_stuck": exit_stuck,
        "ready_examples": ready_examples,
    }


def _log_slot_registry_summary(slot_snapshot: dict[str, float | int], *, event: str) -> None:
    summary = _summarize_slot_registry(slot_snapshot)
    logger.info(
        "[SandboxSlotHealth] event=%s active=%d/%d tracked_active=%d untracked_active=%d "
        "running=%d provisioning=%d ready_not_initialized=%d ready_stuck=%d "
        "provisioning_stuck=%d exit_started=%d exit_stuck=%d ready_examples=%s",
        event,
        slot_snapshot["active"],
        slot_snapshot["max"],
        summary["tracked_active"],
        summary["untracked_active"],
        summary["running"],
        summary["provisioning"],
        summary["ready_not_initialized"],
        summary["ready_stuck"],
        summary["provisioning_stuck"],
        summary["exit_started"],
        summary["exit_stuck"],
        summary["ready_examples"] or "-",
    )


class SandboxLease:
    """A leased sandbox instance."""

    def __init__(
        self,
        sandbox: AsyncSandboxInstance,
        *,
        slot_pool: Optional["SandboxSlotPool"] = None,
        slot_wait_ms: float = 0.0,
        create_ms: float = 0.0,
        ready_ms: float = 0.0,
    ) -> None:
        self.sandbox = sandbox
        self.slot_pool = slot_pool
        self.slot_wait_ms = slot_wait_ms
        self.create_ms = create_ms
        self.ready_ms = ready_ms
        self.released = False

    @property
    def sandbox_id(self) -> str:
        return self.sandbox.sandbox_id

    @property
    def pooled(self) -> bool:
        return False

    async def release(self, healthy: bool) -> None:
        if self.released:
            return
        self.released = True
        await _destroy_sandbox(self.sandbox, release_slot=True, slot_pool=self.slot_pool)


class SandboxSlotPool:
    """Minimal in-process quota manager for sandbox creation.

    This first version only limits the number of concurrently *active*
    sandboxes in the current process. It does not reuse sandboxes; each
    acquired slot still creates a fresh sandbox and deletes it on exit.
    """

    _instance: "SandboxSlotPool | None" = None
    _instance_lock = asyncio.Lock()

    def __init__(self, max_sandboxes: int) -> None:
        self.max_sandboxes = max_sandboxes
        self._semaphore = asyncio.Semaphore(max_sandboxes)
        self._active = 0
        self._waiting = 0
        self._total_acquired = 0
        self._total_released = 0
        self._total_wait_ms = 0.0
        self._last_wait_ms = 0.0
        self._active_lock = asyncio.Lock()

    @classmethod
    async def get(cls, max_sandboxes: int) -> "SandboxSlotPool":
        if max_sandboxes <= 0:
            raise ValueError(f"max_sandboxes must be > 0, got {max_sandboxes}")

        if cls._instance is None:
            async with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls(max_sandboxes=max_sandboxes)
                elif cls._instance.max_sandboxes != max_sandboxes:
                    logger.warning(
                        "SandboxSlotPool already initialized with max_sandboxes=%d; "
                        "ignoring new value %d in this process.",
                        cls._instance.max_sandboxes,
                        max_sandboxes,
                    )
        return cls._instance

    def snapshot(self) -> dict[str, float | int]:
        avg_wait_ms = self._total_wait_ms / self._total_acquired if self._total_acquired > 0 else 0.0
        return {
            "active": self._active,
            "waiting": self._waiting,
            "max": self.max_sandboxes,
            "total_acquired": self._total_acquired,
            "total_released": self._total_released,
            "last_wait_ms": round(self._last_wait_ms, 1),
            "avg_wait_ms": round(avg_wait_ms, 1),
        }

    async def acquire(self, timeout: float | None = None) -> dict[str, float | int]:
        wait_start = time.monotonic()
        async with self._active_lock:
            self._waiting += 1
            snapshot = self.snapshot()
        logger.info(
            "[SandboxPool] acquire_wait_start active=%d/%d waiting=%d total_acquired=%d total_released=%d",
            snapshot["active"],
            snapshot["max"],
            snapshot["waiting"],
            snapshot["total_acquired"],
            snapshot["total_released"],
        )
        _log_slot_registry_summary(snapshot, event="acquire_wait_start")
        try:
            if timeout is None or timeout <= 0:
                await self._semaphore.acquire()
            else:
                await asyncio.wait_for(self._semaphore.acquire(), timeout=timeout)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            async with self._active_lock:
                self._waiting = max(0, self._waiting - 1)
                wait_ms = (time.monotonic() - wait_start) * 1000
                snapshot = self.snapshot()
            logger.warning(
                "[SandboxPool] acquire_wait_failed active=%d/%d waiting=%d wait_ms=%.1f timeout=%s total_acquired=%d total_released=%d",
                snapshot["active"],
                snapshot["max"],
                snapshot["waiting"],
                wait_ms,
                timeout,
                snapshot["total_acquired"],
                snapshot["total_released"],
            )
            _log_slot_registry_summary(snapshot, event="acquire_wait_failed")
            raise

        async with self._active_lock:
            self._waiting = max(0, self._waiting - 1)
            self._active += 1
            self._total_acquired += 1
            self._last_wait_ms = (time.monotonic() - wait_start) * 1000
            self._total_wait_ms += self._last_wait_ms
            snapshot = self.snapshot()
            logger.info(
                "[SandboxPool] acquire_ok active=%d/%d waiting=%d wait_ms=%.1f total_acquired=%d total_released=%d avg_wait_ms=%.1f",
                snapshot["active"],
                snapshot["max"],
                snapshot["waiting"],
                self._last_wait_ms,
                snapshot["total_acquired"],
                snapshot["total_released"],
                snapshot["avg_wait_ms"],
            )
            _log_slot_registry_summary(snapshot, event="acquire_ok")
        return snapshot

    async def release(self, sandbox_id: str | None = None) -> None:
        async with self._active_lock:
            self._active = max(0, self._active - 1)
            self._total_released += 1
            if sandbox_id is not None:
                _slot_registry_remove(sandbox_id)
            snapshot = self.snapshot()
            logger.info(
                "[SandboxPool] release active=%d/%d waiting=%d total_acquired=%d total_released=%d avg_wait_ms=%.1f",
                snapshot["active"],
                snapshot["max"],
                snapshot["waiting"],
                snapshot["total_acquired"],
                snapshot["total_released"],
                snapshot["avg_wait_ms"],
            )
            _log_slot_registry_summary(snapshot, event="release")
        self._semaphore.release()


def _build_curl_cmd(method: str, path: str, payload: Optional[dict] = None) -> str:
    """Build a curl command that avoids shell quoting issues.

    Uses base64-encoded stdin for POST payloads so that no special
    characters appear in the shell command string.
    """
    url = f"http://localhost:{_ENV_SERVER_PORT}{path}"

    if method == "GET":
        return f'curl -sf "{url}"'

    # POST: pipe base64-decoded JSON through stdin
    if payload is not None:
        payload_json = json.dumps(payload)
        b64 = base64.b64encode(payload_json.encode()).decode()
        return (
            f'echo {b64} | base64 -d | '
            f'curl -s -X POST "{url}" '
            f'-H "Content-Type: application/json" -d @-'
        )
    else:
        return (
            f'curl -s -X POST "{url}" '
            f'-H "Content-Type: application/json" -d "{{}}"'
        )


class SandboxWebEnv:
    """
    Drop-in replacement for WebEnv that communicates with
    a sandbox pod running env_server.

    Communication uses the sandbox ``exec()`` method to run ``curl``
    commands inside the pod, hitting ``localhost:8100``.

    Lifecycle:
        1. create_sandbox_env() -- creates sandbox, starts env_server, returns this env
        2. initialize(task_id) -- calls /reset on the env_server
        3. reset()    -- returns cached observation from initialize()
        4. step(actions) -- calls /step
        5. exit()     -- calls /exit, deletes the sandbox, closes the client
    """

    def __init__(
        self,
        lease: SandboxLease,
        *,
        log_successful_requests: bool = True,
    ) -> None:
        self._lease = lease
        self.sandbox = lease.sandbox
        self._slot_wait_ms = lease.slot_wait_ms
        self._create_ms = lease.create_ms
        self._ready_ms = lease.ready_ms
        self._log_successful_requests = log_successful_requests
        self._current_task_id: Optional[str] = None
        self._broken = False
        self._closed = False
        self._sandbox_missing = False

        self._cached_reset: Optional[Tuple[dict, dict]] = None
        self._task_data: Optional[dict] = None

    # ------------------------------------------------------------------
    # Public interface (mirrors WebEnv)
    # ------------------------------------------------------------------

    async def setup(self) -> None:
        """No-op -- sandbox and env_server are started by create_sandbox_env()."""
        pass

    async def initialize(
        self,
        task_id: str,
        task_data: dict,
        env_config: dict,
        tool_list: list | None = None,
        policy: str = "",
    ) -> None:
        """Reset env with a task via curl /reset. Caches observation for reset().

        All configuration is sent from the client — the server does not
        load any local config/task files.
        """
        payload: Dict[str, Any] = {
            "task_id": task_id,
            "task_data": task_data,
            "env_config": env_config,
            "tool_list": tool_list or [],
            "policy": policy,
        }

        try:
            data = await self._exec_curl("/reset", payload)
        except BaseException as exc:
            logger.error("❌ Failed to initialize environment with task_id %s: %s", task_id, exc)
            self._broken = True
            await self.exit()
            raise
        self._current_task_id = task_id
        _slot_registry_update(
            self.sandbox.sandbox_id,
            phase="initialized",
            phase_started_monotonic=time.monotonic(),
            task_id=task_id,
        )
        logger.info(
            "[SandboxTask] task_id=%s sandbox_id=%s initialized slot_wait_ms=%.1f create_ms=%.1f ready_ms=%.1f",
            task_id,
            self.sandbox.sandbox_id,
            self._slot_wait_ms,
            self._create_ms,
            self._ready_ms,
        )

        obs = data["observation"]
        obs["screenshot"] = self._decode_screenshot(obs.get("screenshot"))

        self._cached_reset = (obs, data["info"])
        self._task_data = task_data

    async def reset(
        self,
        url: Optional[str] = None,
        auth_info: Optional[dict] = None,
    ) -> Tuple[dict, dict]:
        """Return the cached observation from initialize()."""
        if self._cached_reset is not None:
            result = self._cached_reset
            self._cached_reset = None
            return result
        raise RuntimeError(
            "❌ SandboxWebEnv.reset() called without a prior initialize(). "
            "Call initialize(task_id) first."
        )

    async def step(
        self, action_list: List[Dict[str, Any]]
    ) -> Tuple[dict, float, bool, bool, dict]:
        """Execute actions via curl /step."""
        try:
            data = await self._exec_curl("/step", {"actions": action_list, "env_id": 0})
        except Exception as exc:
            if _is_missing_sandbox_error(exc):
                self._sandbox_missing = True
            logger.error("❌ Step failed: %s", exc)
            self._broken = True
            await self.exit()
            raise

        obs = data["observation"]
        if obs.get("screenshot") is None:
            print("========== Error: Action execution failed without a screenshot. ==========")
            print("Environment message:", data["info"]["env_message"])
            print("Diagnostics:", data.get("diagnostics"))
            self._broken = True
            await self.exit()
            raise ValueError("Action execution failed without a screenshot.")

        obs["screenshot"] = self._decode_screenshot(obs.get("screenshot"))

        info = data["info"]
        if "diagnostics" in data:
            info["server_diagnostics"] = data["diagnostics"]

        return (
            obs,
            data["reward"],
            data["terminated"],
            data["truncated"],
            info,
        )

    async def exit(self) -> None:
        """Close the browser env and either return or destroy the sandbox lease."""
        if self._closed:
            return
        self._closed = True
        sid = self.sandbox.sandbox_id
        _slot_registry_update(
            sid,
            phase="exit_started",
            phase_started_monotonic=time.monotonic(),
            task_id=self._current_task_id,
        )
        delete_start = time.monotonic()
        try:
            if self._sandbox_missing:
                self._broken = True
                logger.warning(
                    "❌ SandboxWebEnv skipping /exit because sandbox_id=%s is already missing.",
                    sid,
                )
            else:
                await asyncio.wait_for(
                    self._exec_curl("/exit", {"env_id": 0}),
                    timeout=_ENV_EXIT_REQUEST_TIMEOUT,
                )
        except asyncio.TimeoutError:
            self._broken = True
            logger.warning(
                "❌ SandboxWebEnv /exit timed out after %ss for sandbox_id=%s; forcing cleanup.",
                _ENV_EXIT_REQUEST_TIMEOUT,
                sid,
            )
        except Exception as exc:
            self._broken = True
            logger.warning("❌ SandboxWebEnv env/exit error (ignored): %s", exc)
        finally:
            logger.info(
                "[SandboxExit] task_id=%s sandbox_id=%s pooled=%s healthy=%s delete_ms=%.1f",
                self._current_task_id or "unknown",
                sid,
                self._lease.pooled,
                not self._broken,
                (time.monotonic() - delete_start) * 1000,
            )
            release_task = asyncio.create_task(self._lease.release(healthy=not self._broken))
            try:
                await asyncio.shield(release_task)
            except asyncio.CancelledError:
                logger.warning(
                    "❌ Sandbox lease release cancelled for sandbox_id=%s; waiting for cleanup to finish.",
                    sid,
                )
                await release_task
                raise

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _exec_curl(self, path: str, payload: dict) -> dict:
        """Execute a curl POST command inside the sandbox and parse the JSON response.

        Retries on transient exec/curl failures (exit_code=None, empty stdout)
        before raising.
        """
        cmd = _build_curl_cmd("POST", path, payload)
        sid = self.sandbox.sandbox_id
        last_error = None
        curl_timeout = _get_exec_curl_timeout(path)
        max_retries = _get_exec_curl_max_retries(path)
        retry_delay = _get_exec_curl_retry_delay(path)

        for attempt in range(max_retries):
            t0 = asyncio.get_event_loop().time()
            result = await self.sandbox.exec(
                cmd,
                timeout=curl_timeout,
                cwd="/app",
                login_shell=False,
            )
            client_elapsed = round(asyncio.get_event_loop().time() - t0, 3)

            if not result.succeeded:
                last_error = (
                    f"⁉️ curl {path} failed in sandbox {sid} "
                    f"(client_elapsed={client_elapsed}s, timeout={curl_timeout}s): "
                    f"exit_code={result.exit_code}, "
                    f"stderr={result.stderr!r}, "
                    f"stdout(first 500)={result.stdout[:500]!r}"
                )
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt)
                    logger.warning(
                        "[exec_curl retry %d/%d] %s — waiting %ds",
                        attempt + 1, max_retries, last_error, wait_time,
                    )
                    await asyncio.sleep(wait_time)
                    continue
                raise RuntimeError(last_error)

            stdout = result.stdout.strip()
            if not stdout:
                last_error = (
                    f"⁉️ curl {path} returned empty response in sandbox {sid} "
                    f"(client_elapsed={client_elapsed}s)"
                )
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt)
                    logger.warning(
                        "[exec_curl retry %d/%d] %s — waiting %ds",
                        attempt + 1, max_retries, last_error, wait_time,
                    )
                    await asyncio.sleep(wait_time)
                    continue
                raise RuntimeError(last_error)

            # Success — break out of retry loop
            break

        try:
            data = json.loads(stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"⁉️ curl {path} returned invalid JSON in sandbox "
                f"{sid}: {exc}\n"
                f"stdout: {stdout}"
            )

        # FastAPI HTTPException returns {"detail": "..."} on errors.
        # curl -s doesn't fail on HTTP 4xx/5xx, so catch it here.
        if "detail" in data:
            detail = data["detail"]
            # The /step endpoint now sends structured detail with diagnostics.
            if isinstance(detail, dict):
                diag = detail.get("diagnostics", {})
                logger.error(
                    "⁉️ env_server %s error in sandbox %s: %s | "
                    "server_diagnostics=%s",
                    path, sid, detail.get("message", detail), diag,
                )
                raise RuntimeError(
                    f"⁉️ env_server {path} error in sandbox {sid}: "
                    f"{detail.get('message', detail)} | "
                    f"server_elapsed={diag.get('elapsed_secs')}s, "
                    f"rss_mb={diag.get('rss_mb')}, "
                    f"server_error={diag.get('error')}, "
                    f"server_traceback={diag.get('error_traceback', '')[:1000]}"
                )
            raise RuntimeError(
                f"⁉️ env_server {path} error in sandbox {sid}: {detail}"
            )

        # Log diagnostics from successful responses
        diag = data.get("diagnostics")
        if diag and self._log_successful_requests:
            logger.info(
                "sandbox %s %s OK: server_elapsed=%.3fs, rss_mb=%s, "
                "step_count=%s, uptime=%ss",
                sid, path,
                diag.get("elapsed_secs", -1),
                diag.get("rss_mb"),
                diag.get("request_counts", {}).get("step"),
                diag.get("server_uptime_secs"),
            )

        return data

    @staticmethod
    def _decode_screenshot(value: Any) -> bytes:
        """Decode a base64-encoded screenshot back to bytes."""
        if isinstance(value, str):
            return base64.b64decode(value)
        if isinstance(value, (bytes, bytearray)):
            return bytes(value)
        return b""


# ---------------------------------------------------------------------------
# Public factory function (creates a sandbox, starts env_server, returns env)
# ---------------------------------------------------------------------------


_DEFAULT_SANDBOX_MANIFEST_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), ".sandboxes"
)


def _get_sandbox_manifest_dir() -> str:
    return os.environ.get("SLIME_BROWSER_SANDBOX_MANIFEST_DIR", _DEFAULT_SANDBOX_MANIFEST_DIR)


def _get_sandbox_owner_metadata() -> Dict[str, Any]:
    """Return owner metadata used to scope cleanup to the current run/session."""
    metadata: Dict[str, Any] = {
        "owner_pid": os.getpid(),
        "owner_ppid": os.getppid(),
        "owner_pgid": None,
        "owner_sid": None,
    }
    try:
        metadata["owner_pgid"] = os.getpgid(0)
    except OSError:
        pass
    try:
        metadata["owner_sid"] = os.getsid(0)
    except OSError:
        pass

    owner_tag = os.environ.get("SANDBOX_OWNER_TAG")
    if owner_tag:
        metadata["owner_tag"] = owner_tag
    return metadata


def _load_sandbox_marker(marker_path: str) -> Dict[str, Any]:
    """Load sandbox marker metadata, tolerating legacy empty marker files."""
    sandbox_id = os.path.basename(marker_path)
    try:
        with open(marker_path, "r") as f:
            raw = f.read().strip()
    except OSError:
        return {"sandbox_id": sandbox_id, "marker_format": "unreadable"}

    if not raw:
        return {"sandbox_id": sandbox_id, "marker_format": "legacy"}

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {"sandbox_id": sandbox_id, "marker_format": "invalid"}

    if not isinstance(data, dict):
        return {"sandbox_id": sandbox_id, "marker_format": "invalid"}

    data.setdefault("sandbox_id", sandbox_id)
    data.setdefault("marker_format", "json")
    return data


def _save_sandbox_id(sandbox_id: str) -> None:
    """Record a sandbox ID by creating a marker file.

    Each sandbox gets its own file so parallel workers never contend
    on the same file.
    """
    manifest_dir = _get_sandbox_manifest_dir()
    os.makedirs(manifest_dir, exist_ok=True)
    marker = os.path.join(manifest_dir, sandbox_id)
    with open(marker, "w") as f:
        json.dump(
            {
                "sandbox_id": sandbox_id,
                "created_at": time.time(),
                **_get_sandbox_owner_metadata(),
            },
            f,
        )


def _remove_sandbox_id(sandbox_id: str) -> None:
    """Remove the marker file for a sandbox ID."""
    marker = os.path.join(_get_sandbox_manifest_dir(), sandbox_id)
    try:
        os.remove(marker)
    except OSError:
        pass


def _list_sandbox_markers() -> List[Dict[str, Any]]:
    """Return sandbox marker metadata recorded in the manifest directory."""
    manifest_dir = _get_sandbox_manifest_dir()
    if not os.path.isdir(manifest_dir):
        return []
    try:
        return [
            _load_sandbox_marker(os.path.join(manifest_dir, entry))
            for entry in os.listdir(manifest_dir)
        ]
    except OSError:
        return []


async def cleanup_existing_sandboxes(
    sandbox_cfg: Dict[str, Any],
    *,
    cleanup_all: bool = False,
) -> None:
    """Delete leftover sandboxes recorded in the local manifest directory.

    Call this at startup to clean up sandboxes from a previous run that
    was aborted (e.g. via Ctrl+C).  Reads sandbox IDs from per-sandbox
    marker files and deletes them via the orchestrator API.

    Args:
        sandbox_cfg: The ``sandbox:`` section from config.yaml.
    """
    markers = _list_sandbox_markers()
    if not markers:
        logger.info("No leftover sandbox markers found, nothing to clean up.")
        return

    selected_markers = markers

    orchestrator_url = (
        os.environ.get("SANDBOX_ORCHESTRATOR_URL")
        or sandbox_cfg.get("orchestrator_url")
    )
    if not orchestrator_url:
        raise ValueError("Set SANDBOX_ORCHESTRATOR_URL env var")

    api_key = os.environ.get("SANDBOX_API_KEY") or sandbox_cfg.get("api_key")

    logger.info(
        "Found %d sandbox(es) to delete from manifest_dir=%s%s.",
        len(selected_markers),
        _get_sandbox_manifest_dir(),
        " with --cleanup-all" if cleanup_all else "",
    )

    try:
        async with AsyncSandboxClient(
            base_url=orchestrator_url,
            api_key=api_key,
            auto_cleanup=True,
        ) as client:

            for marker in selected_markers:
                sid = str(marker["sandbox_id"])
                try:
                    await client.delete_sandbox(sid)
                    logger.info(f"♻️  [Deleted {sid}]")
                    _remove_sandbox_id(sid)
                except Exception as exc:
                    logger.warning(f"❌ Failed to delete sandbox {sid} (may already be gone): {exc}")

    except Exception as exc:
        logger.warning(f"❌ Failed to cleanup existing sandboxes: {exc}")


def _get_sandbox_connection_config(sandbox_cfg: Dict[str, Any]) -> tuple[str, Optional[str]]:
    orchestrator_url = (
        os.environ.get("SANDBOX_ORCHESTRATOR_URL")
        or sandbox_cfg.get("orchestrator_url")
    )
    if not orchestrator_url:
        raise ValueError("Set SANDBOX_ORCHESTRATOR_URL env var or sandbox.orchestrator_url in config.yaml")
    api_key = os.environ.get("SANDBOX_API_KEY") or sandbox_cfg.get("api_key")
    return orchestrator_url, api_key


async def _destroy_sandbox(
    sandbox: AsyncSandboxInstance,
    *,
    release_slot: bool,
    slot_pool: Optional[SandboxSlotPool],
) -> None:
    sid = sandbox.sandbox_id
    try:
        try:
            await sandbox.delete()
            print(f"♻️  [Deleted {sid}]")
            _remove_sandbox_id(sid)
        except asyncio.CancelledError:
            logger.warning("❌ Sandbox delete cancelled for sandbox_id=%s during cleanup", sid)
            raise
        except Exception as exc:
            logger.warning("❌ Failed to delete sandbox %s: %s", sid, exc)

        try:
            await sandbox._client.close()
        except asyncio.CancelledError:
            logger.warning("❌ Client close cancelled for sandbox_id=%s during cleanup", sid)
            raise
        except Exception as exc:
            logger.warning("❌ Failed to close client session: %s", exc)
    finally:
        if release_slot and slot_pool is not None:
            try:
                await slot_pool.release(sandbox_id=sid)
            except Exception as exc:
                logger.warning("❌ Failed to release slot for sandbox_id=%s: %s", sid, exc)
                _slot_registry_remove(sid)
        else:
            _slot_registry_remove(sid)


async def _provision_sandbox(
    sandbox_cfg: Dict[str, Any],
    *,
    slot_pool: Optional[SandboxSlotPool],
    acquire_slot: bool,
) -> SandboxLease:
    orchestrator_url, api_key = _get_sandbox_connection_config(sandbox_cfg)

    # On success, SandboxWebEnv takes ownership of the sandbox (which
    # internally holds a client reference) and closes it in exit().
    # On failure, we clean up the client here via try/except.
    client = AsyncSandboxClient(
        base_url=orchestrator_url,
        api_key=api_key,
        auto_cleanup=True,
        max_retries=int(sandbox_cfg.get("max_retries", 3) or 3),
        retry_delay=int(sandbox_cfg.get("retry_delay", 1) or 1),
    )

    sandbox = None
    slot_acquired = False
    slot_wait_ms = 0.0
    create_ms = 0.0
    ready_ms = 0.0
    try:
        acquire_timeout = sandbox_cfg.get("acquire_timeout_secs")
        if acquire_slot:
            if slot_pool is None:
                max_sandboxes = int(sandbox_cfg.get("max_sandboxes", 0) or 0)
                if max_sandboxes <= 0:
                    raise ValueError("max_sandboxes must be > 0 when acquire_slot=True")
                slot_pool = await SandboxSlotPool.get(max_sandboxes=max_sandboxes)
            try:
                slot_snapshot = await slot_pool.acquire(timeout=acquire_timeout)
            except asyncio.TimeoutError as exc:
                snapshot = slot_pool.snapshot()
                raise TimeoutError(
                    "Timed out waiting for a sandbox slot "
                    f"after {acquire_timeout}s "
                    f"(active={snapshot['active']}/{snapshot['max']}, waiting={snapshot['waiting']})"
                ) from exc
            slot_acquired = True
            slot_wait_ms = float(slot_snapshot["last_wait_ms"])

        image = sandbox_cfg.get("image", os.environ.get("BROWSER_SANDBOX_IMAGE", "browser-env:latest"))
        cpu = sandbox_cfg.get("cpu")
        memory = sandbox_cfg.get("memory")
        block_network = sandbox_cfg.get("block_network", False)

        start_time = time.monotonic()
        print(f"⏱️  Creating sandbox with image={image}, cpu={cpu}, memory={memory}, block_network={block_network}...")
        logger.info(
            "[SandboxCreate] start slot_wait_ms=%.1f image=%s cpu=%s memory=%s block_network=%s",
            slot_wait_ms,
            image,
            cpu,
            memory,
            block_network,
        )
        sandbox = await client.create_sandbox(
            image=image,
            block_network=block_network,
            cpu=cpu,
            memory=memory,
            timeout=600,
        )
        sandbox.start_heartbeat(interval=_get_sandbox_heartbeat_interval())
        sid = sandbox.sandbox_id
        _slot_registry_update(
            sid,
            phase="created",
            phase_started_monotonic=time.monotonic(),
            slot_wait_ms=slot_wait_ms,
            pooled=False,
        )
        elapsed = time.monotonic() - start_time
        create_ms = elapsed * 1000
        print(f"✅ [Created {sid}] {elapsed:.1f}s - Sandbox starting env_server...")
        logger.info("[SandboxCreate] created sandbox_id=%s create_ms=%.1f", sid, create_ms)
        _save_sandbox_id(sid)

        # Start env_server — the sandbox orchestrator overrides the image CMD
        # with its own agent, so we must launch the server explicitly.
        start_cmd = (
            "nohup python -m openwebrl.docker.env_server "
            f"--host 0.0.0.0 --port {_ENV_SERVER_PORT} "
            "> /tmp/env_server.log 2>&1 &"
        )
        await sandbox.exec(start_cmd, timeout=30, cwd="/app", login_shell=False)

        # Poll health until ready (reset timer — don't count sandbox creation time)
        health_start = time.monotonic()
        health_cmd = f'curl -sf "http://localhost:{_ENV_SERVER_PORT}/health"'

        while time.monotonic() - health_start < _ENV_SERVER_STARTUP_TIMEOUT:
            await asyncio.sleep(_ENV_SERVER_STARTUP_POLL)
            try:
                hresult = await sandbox.exec(
                    health_cmd, timeout=10, cwd="/app", login_shell=False
                )
                if hresult.succeeded and hresult.stdout.strip():
                    ready_ms = (time.monotonic() - health_start) * 1000
                    _slot_registry_update(
                        sid,
                        phase="ready",
                        phase_started_monotonic=time.monotonic(),
                        create_ms=create_ms,
                        ready_ms=ready_ms,
                    )
                    logger.info(
                        "[SandboxCreate] ready sandbox_id=%s ready_ms=%.1f total_startup_ms=%.1f",
                        sid,
                        ready_ms,
                        slot_wait_ms + create_ms + ready_ms,
                    )
                    return SandboxLease(
                        sandbox=sandbox,
                        slot_pool=slot_pool,
                        slot_wait_ms=slot_wait_ms,
                        create_ms=create_ms,
                        ready_ms=ready_ms,
                    )
            except Exception:
                pass

        # Timed out — dump diagnostics before cleanup
        try:
            ps_result = await sandbox.exec("ps aux", timeout=10, cwd="/app", login_shell=False)
            print(f"⏱️ [{sid}] Timed out. Processes:\n{ps_result.stdout[:2000]}")
            log_result = await sandbox.exec("cat /tmp/env_server.log 2>/dev/null || journalctl -u env_server --no-pager -n 50 2>/dev/null || echo 'no logs found'", timeout=10, cwd="/app", login_shell=False)
            print(f"⏱️ [{sid}] Server logs:\n{log_result.stdout[:2000]}")
        except Exception:
            pass
        raise TimeoutError(f"❌ env_server in {sid} not healthy within {_ENV_SERVER_STARTUP_TIMEOUT}s")

    except asyncio.CancelledError:
        # Timeout-driven rollout aborts cancel provisioning mid-request.
        # Treat that as cooperative cancellation rather than a sandbox error.
        if sandbox is not None:
            try:
                await asyncio.shield(_destroy_sandbox(sandbox, release_slot=False, slot_pool=None))
            except Exception as exc:
                logger.warning("❌ Failed to delete sandbox during cancellation cleanup: %s", exc)
        try:
            await asyncio.shield(client.close(cleanup=True))
        except Exception:
            pass
        if slot_acquired and slot_pool is not None:
            await asyncio.shield(slot_pool.release())
        logger.info("Sandbox provisioning cancelled")
        raise
    except BaseException:
        # Clean up on any failure
        if sandbox is not None:
            try:
                await asyncio.shield(_destroy_sandbox(sandbox, release_slot=False, slot_pool=None))
            except Exception as exc:
                raise RuntimeError("❌ Failed to delete sandbox during cleanup") from exc
        try:
            await asyncio.shield(client.close(cleanup=True))
        except Exception:
            pass
        if slot_acquired and slot_pool is not None:
            await asyncio.shield(slot_pool.release())
        logger.error("❌ Failed to create sandbox environment", exc_info=True)
        raise


async def create_sandbox_env(sandbox_cfg: Dict[str, Any]) -> SandboxWebEnv:
    """Create a sandbox-backed environment with optional per-process slot limiting."""
    log_successful_requests = bool(sandbox_cfg.get("log_successful_requests", True))

    max_sandboxes = int(sandbox_cfg.get("max_sandboxes", 0) or 0)
    slot_pool = None
    acquire_slot = max_sandboxes > 0
    if acquire_slot:
        slot_pool = await SandboxSlotPool.get(max_sandboxes=max_sandboxes)
    lease = await _provision_sandbox(sandbox_cfg, slot_pool=slot_pool, acquire_slot=acquire_slot)
    return SandboxWebEnv(lease=lease, log_successful_requests=log_successful_requests)


# ---------------------------------------------------------------------------
# CLI entry point: python sandbox_env.py --cleanup
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sandbox environment utilities")
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Delete leftover sandboxes recorded in the current manifest directory.",
    )
    parser.add_argument(
        "--cleanup-all",
        action="store_true",
        help="Alias of --cleanup kept for compatibility.",
    )
    parser.add_argument(
        "--manifest-dir",
        type=str,
        default=None,
        help="Override the sandbox marker directory for this invocation.",
    )
    args = parser.parse_args()

    if args.manifest_dir:
        os.environ["SLIME_BROWSER_SANDBOX_MANIFEST_DIR"] = os.path.abspath(args.manifest_dir)

    if args.cleanup or args.cleanup_all:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
        manifest_dir = _get_sandbox_manifest_dir()
        print("Sandbox manifest directory:", manifest_dir)

        markers = _list_sandbox_markers()
        if not markers:
            print("No sandbox markers found in", manifest_dir)
            sys.exit(0)
        selected_markers = markers
        print(f"Found {len(selected_markers)} sandbox(es) to clean up from {manifest_dir}:")

        for marker in selected_markers:
            sandbox_id = str(marker["sandbox_id"])
            marker_scope = f"tag={marker.get('owner_tag')!r}, sid={marker.get('owner_sid')!r}"
            print(f"  - {sandbox_id} ({marker_scope})")

        # Build a minimal sandbox_cfg from environment variables
        sandbox_cfg = {
            "orchestrator_url": os.environ.get("SANDBOX_ORCHESTRATOR_URL", ""),
            "api_key": os.environ.get("SANDBOX_API_KEY"),
        }

        asyncio.run(cleanup_existing_sandboxes(sandbox_cfg, cleanup_all=args.cleanup_all))
        print("Cleanup complete.")
    else:
        parser.print_help()
