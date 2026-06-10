"""
Local subprocess-backed browser environment client.

This is a lightweight alternative to the K8s sandbox path.  Each browser
environment gets its own local env_server subprocess and process group, so
cleanup can forcibly terminate the env_server and its Chromium children.
"""

import asyncio
import base64
import contextlib
import fcntl
import logging
import os
import signal
import socket
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import aiohttp

logger = logging.getLogger(__name__)

_DEFAULT_HOST = "127.0.0.1"
_DEFAULT_PORT_START = 18000
_DEFAULT_PORT_END = 18999
_DEFAULT_STARTUP_TIMEOUT_SECS = 30
_DEFAULT_STARTUP_POLL_SECS = 0.5
_DEFAULT_REQUEST_TIMEOUT_SECS = 300
_DEFAULT_EXIT_TIMEOUT_SECS = 10
_DEFAULT_KILL_TIMEOUT_SECS = 5
_DEFAULT_MAX_PROCESSES = 8
_DEFAULT_LOG_DIR = "/tmp/slime_browser_local_process_env_logs"
_DEFAULT_PORT_LOCK_DIR = "/tmp/slime_browser_local_process_ports"

_ENV_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_ENV_DIR, "..", "..", ".."))

_child_watcher_ready = False


def _ensure_asyncio_child_watcher() -> None:
    """Guarantee asyncio.create_subprocess_exec works from this event loop.

    Inside Ray rollout actors the loop runs on a non-main thread whose active
    event-loop policy lacks a child watcher, so create_subprocess_exec raises
    NotImplementedError. Install a ThreadedChildWatcher (safe from any thread)
    and force the policy to return it. Idempotent.
    """
    global _child_watcher_ready
    if _child_watcher_ready:
        return
    try:
        existing = asyncio.get_event_loop_policy().get_child_watcher()
        if existing is not None:
            _child_watcher_ready = True
            return
    except Exception:
        pass
    watcher = asyncio.ThreadedChildWatcher()
    try:
        watcher.attach_loop(asyncio.get_running_loop())
    except Exception:
        pass
    pol = asyncio.get_event_loop_policy()
    try:
        pol.set_child_watcher(watcher)
    except Exception:
        pass
    # Belt-and-suspenders: ensure events.get_child_watcher() returns a working
    # watcher even if the policy doesn't support set_child_watcher.
    try:
        pol.get_child_watcher = lambda: watcher  # type: ignore[method-assign]
    except Exception:
        pass
    _child_watcher_ready = True


def _get_env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError:
        logger.warning("Invalid integer for %s=%r; using default=%s", name, value, default)
        return default


def _get_env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError:
        logger.warning("Invalid float for %s=%r; using default=%s", name, value, default)
        return default


def _is_port_free(host: str, port: int) -> bool:
    with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
        except OSError:
            return False
    return True


class LocalProcessPortLease:
    """Best-effort process-safe port lease using an exclusive lock file."""

    def __init__(self, host: str, port: int, lock_path: str, lock_fd: int) -> None:
        self.host = host
        self.port = port
        self.lock_path = lock_path
        self.lock_fd = lock_fd
        self.released = False

    def release(self) -> None:
        if self.released:
            return
        self.released = True
        try:
            fcntl.flock(self.lock_fd, fcntl.LOCK_UN)
        except Exception:
            pass
        try:
            os.close(self.lock_fd)
        except Exception:
            pass
        try:
            os.unlink(self.lock_path)
        except FileNotFoundError:
            pass
        except Exception as exc:
            logger.warning("Failed to remove local_process port lock %s: %s", self.lock_path, exc)


class LocalProcessSlotPool:
    """In-process quota manager for local env_server subprocesses."""

    _instance: "LocalProcessSlotPool | None" = None
    _instance_lock = asyncio.Lock()

    def __init__(self, max_processes: int) -> None:
        if max_processes <= 0:
            raise ValueError(f"max_processes must be > 0, got {max_processes}")
        self.max_processes = max_processes
        self._semaphore = asyncio.Semaphore(max_processes)
        self._active = 0
        self._waiting = 0
        self._total_acquired = 0
        self._total_released = 0
        self._total_wait_ms = 0.0
        self._last_wait_ms = 0.0
        self._active_lock = asyncio.Lock()

    @classmethod
    async def get(cls, max_processes: int) -> "LocalProcessSlotPool":
        if cls._instance is None:
            async with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls(max_processes=max_processes)
                elif cls._instance.max_processes != max_processes:
                    logger.warning(
                        "LocalProcessSlotPool already initialized with max_processes=%d; "
                        "ignoring new value %d in this process.",
                        cls._instance.max_processes,
                        max_processes,
                    )
        return cls._instance

    def snapshot(self) -> dict[str, float | int]:
        avg_wait_ms = self._total_wait_ms / self._total_acquired if self._total_acquired > 0 else 0.0
        return {
            "active": self._active,
            "waiting": self._waiting,
            "max": self.max_processes,
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
            "[LocalProcessPool] acquire_wait_start active=%d/%d waiting=%d total_acquired=%d total_released=%d",
            snapshot["active"],
            snapshot["max"],
            snapshot["waiting"],
            snapshot["total_acquired"],
            snapshot["total_released"],
        )
        try:
            if timeout is None or timeout <= 0:
                await self._semaphore.acquire()
            else:
                await asyncio.wait_for(self._semaphore.acquire(), timeout=timeout)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            async with self._active_lock:
                self._waiting = max(0, self._waiting - 1)
            raise

        async with self._active_lock:
            self._waiting = max(0, self._waiting - 1)
            self._active += 1
            self._total_acquired += 1
            self._last_wait_ms = (time.monotonic() - wait_start) * 1000
            self._total_wait_ms += self._last_wait_ms
            snapshot = self.snapshot()
        logger.info(
            "[LocalProcessPool] acquire_ok active=%d/%d waiting=%d wait_ms=%.1f total_acquired=%d total_released=%d avg_wait_ms=%.1f",
            snapshot["active"],
            snapshot["max"],
            snapshot["waiting"],
            snapshot["last_wait_ms"],
            snapshot["total_acquired"],
            snapshot["total_released"],
            snapshot["avg_wait_ms"],
        )
        return snapshot

    async def release(self) -> None:
        async with self._active_lock:
            self._active = max(0, self._active - 1)
            self._total_released += 1
            snapshot = self.snapshot()
        logger.info(
            "[LocalProcessPool] release active=%d/%d waiting=%d total_acquired=%d total_released=%d avg_wait_ms=%.1f",
            snapshot["active"],
            snapshot["max"],
            snapshot["waiting"],
            snapshot["total_acquired"],
            snapshot["total_released"],
            snapshot["avg_wait_ms"],
        )
        self._semaphore.release()


def _cfg_int(local_cfg: Dict[str, Any], key: str, env_name: str, default: int) -> int:
    value = os.environ.get(env_name)
    if value not in (None, ""):
        return _get_env_int(env_name, default)
    return int(local_cfg.get(key, default) or default)


def _cfg_float(local_cfg: Dict[str, Any], key: str, env_name: str, default: float) -> float:
    value = os.environ.get(env_name)
    if value not in (None, ""):
        return _get_env_float(env_name, default)
    return float(local_cfg.get(key, default) or default)


def _cfg_str(local_cfg: Dict[str, Any], key: str, env_name: str, default: str) -> str:
    value = os.environ.get(env_name)
    if value not in (None, ""):
        return str(value)
    return str(local_cfg.get(key, default) or default)


def _acquire_port(host: str, port_start: int, port_end: int, lock_dir: str) -> LocalProcessPortLease:
    os.makedirs(lock_dir, exist_ok=True)
    for port in range(port_start, port_end + 1):
        lock_path = os.path.join(lock_dir, f"{host.replace('.', '_')}_{port}.lock")
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            with contextlib.suppress(Exception):
                os.close(fd)
            continue
        except Exception:
            with contextlib.suppress(Exception):
                os.close(fd)
            continue
        try:
            os.ftruncate(fd, 0)
            os.write(fd, f"pid={os.getpid()}\nport={port}\n".encode())
        except Exception:
            with contextlib.suppress(Exception):
                fcntl.flock(fd, fcntl.LOCK_UN)
            with contextlib.suppress(Exception):
                os.close(fd)
            continue

        try:
            if _is_port_free(host, port):
                return LocalProcessPortLease(host, port, lock_path, fd)
        except Exception as exc:
            logger.warning("Failed to probe local_process port %s:%s: %s", host, port, exc)

        with contextlib.suppress(Exception):
            fcntl.flock(fd, fcntl.LOCK_UN)
        with contextlib.suppress(Exception):
            os.close(fd)
    raise RuntimeError(f"No free local_process env_server port found in range {port_start}-{port_end}")


class LocalProcessWebEnv:
    """Drop-in WebEnv replacement backed by a local env_server subprocess."""

    def __init__(
        self,
        *,
        proc: asyncio.subprocess.Process,
        port_lease: LocalProcessPortLease,
        slot_pool: Optional[LocalProcessSlotPool],
        log_file,
        request_timeout_secs: float,
        exit_timeout_secs: float,
        kill_timeout_secs: float,
        startup_ms: float,
    ) -> None:
        self.proc = proc
        self.port_lease = port_lease
        self.slot_pool = slot_pool
        self.log_file = log_file
        self.request_timeout_secs = request_timeout_secs
        self.exit_timeout_secs = exit_timeout_secs
        self.kill_timeout_secs = kill_timeout_secs
        self.startup_ms = startup_ms
        self.base_url = f"http://{port_lease.host}:{port_lease.port}"
        self._current_task_id: Optional[str] = None
        self._cached_reset: Optional[Tuple[dict, dict]] = None
        self._task_data: Optional[dict] = None
        self._closed = False
        self._broken = False

    async def setup(self) -> None:
        """No-op -- env_server is started by create_local_process_env()."""
        pass

    async def initialize(
        self,
        task_id: str,
        task_data: dict,
        env_config: dict,
        tool_list: list | None = None,
        policy: str = "",
    ) -> None:
        payload = {
            "task_id": task_id,
            "task_data": task_data,
            "env_config": env_config,
            "tool_list": tool_list or [],
            "policy": policy,
        }
        try:
            data = await self._post("/reset", payload, timeout_secs=self.request_timeout_secs)
        except BaseException:
            self._broken = True
            await self.exit()
            raise

        self._current_task_id = task_id
        obs = data["observation"]
        obs["screenshot"] = self._decode_screenshot(obs.get("screenshot"))
        info = data["info"]
        if "diagnostics" in data:
            info["server_diagnostics"] = data["diagnostics"]
        self._cached_reset = (obs, info)
        self._task_data = task_data
        logger.info(
            "[LocalProcessTask] task_id=%s pid=%s port=%s initialized startup_ms=%.1f",
            task_id,
            self.proc.pid,
            self.port_lease.port,
            self.startup_ms,
        )

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
            "LocalProcessWebEnv.reset() called without a prior initialize(). "
            "Call initialize(task_id) first."
        )

    async def step(
        self, action_list: List[Dict[str, Any]]
    ) -> Tuple[dict, float, bool, bool, dict]:
        try:
            data = await self._post(
                "/step",
                {"actions": action_list, "env_id": 0},
                timeout_secs=self.request_timeout_secs,
            )
        except Exception:
            self._broken = True
            await self.exit()
            raise

        obs = data["observation"]
        if obs.get("screenshot") is None:
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
        """Best-effort server exit followed by process-group termination."""
        if self._closed:
            return
        self._closed = True
        t0 = time.monotonic()
        try:
            if not self._broken and self.proc.returncode is None:
                try:
                    await self._post(
                        "/exit",
                        {"env_id": 0},
                        timeout_secs=self.exit_timeout_secs,
                    )
                except Exception as exc:
                    self._broken = True
                    logger.warning(
                        "LocalProcessWebEnv /exit error for task_id=%s pid=%s port=%s: %s",
                        self._current_task_id,
                        self.proc.pid,
                        self.port_lease.port,
                        exc,
                    )
        finally:
            await self._terminate_process_group()
            self.port_lease.release()
            if self.slot_pool is not None:
                await self.slot_pool.release()
            try:
                self.log_file.close()
            except Exception:
                pass
            logger.info(
                "[LocalProcessExit] task_id=%s pid=%s port=%s healthy=%s elapsed_ms=%.1f",
                self._current_task_id or "unknown",
                self.proc.pid,
                self.port_lease.port,
                not self._broken,
                (time.monotonic() - t0) * 1000,
            )

    async def _post(self, path: str, payload: dict, *, timeout_secs: float) -> dict:
        url = f"{self.base_url}{path}"
        timeout = aiohttp.ClientTimeout(total=timeout_secs)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, json=payload) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    raise RuntimeError(
                        f"local_process env_server error [{resp.status}] {path}: {text}"
                    )
                data = await resp.json()
        if "detail" in data:
            raise RuntimeError(f"local_process env_server {path} error: {data['detail']}")
        return data

    async def _terminate_process_group(self) -> None:
        if self.proc.returncode is not None:
            return

        pgid = None
        try:
            pgid = os.getpgid(self.proc.pid)
        except ProcessLookupError:
            pass

        if pgid is not None:
            try:
                os.killpg(pgid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            except Exception as exc:
                logger.warning("Failed to SIGTERM local_process pgid=%s: %s", pgid, exc)
        else:
            with contextlib.suppress(ProcessLookupError):
                self.proc.terminate()

        try:
            await asyncio.wait_for(self.proc.wait(), timeout=self.kill_timeout_secs)
            return
        except asyncio.TimeoutError:
            pass

        if pgid is not None:
            try:
                os.killpg(pgid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            except Exception as exc:
                logger.warning("Failed to SIGKILL local_process pgid=%s: %s", pgid, exc)
        else:
            with contextlib.suppress(ProcessLookupError):
                self.proc.kill()
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(self.proc.wait(), timeout=max(1.0, self.kill_timeout_secs))

    @staticmethod
    def _decode_screenshot(value: Any) -> bytes:
        if isinstance(value, str):
            return base64.b64decode(value)
        if isinstance(value, (bytes, bytearray)):
            return bytes(value)
        return b""


async def _wait_until_healthy(
    base_url: str,
    proc: asyncio.subprocess.Process,
    *,
    startup_timeout_secs: float,
    startup_poll_secs: float,
) -> None:
    deadline = time.monotonic() + startup_timeout_secs
    last_error: Exception | None = None
    timeout = aiohttp.ClientTimeout(total=max(1.0, startup_poll_secs))
    while time.monotonic() < deadline:
        if proc.returncode is not None:
            raise RuntimeError(f"local_process env_server exited early with code {proc.returncode}")
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(f"{base_url}/health") as resp:
                    if resp.status == 200:
                        await resp.read()
                        return
        except Exception as exc:
            last_error = exc
        await asyncio.sleep(startup_poll_secs)
    raise TimeoutError(
        f"local_process env_server at {base_url} was not healthy within "
        f"{startup_timeout_secs}s; last_error={last_error}"
    )


async def create_local_process_env(local_cfg: Dict[str, Any]) -> LocalProcessWebEnv:
    """Create a local env_server subprocess and return a LocalProcessWebEnv."""
    host = _cfg_str(local_cfg, "host", "SLIME_BROWSER_LOCAL_PROCESS_HOST", _DEFAULT_HOST)
    port_start = _cfg_int(local_cfg, "port_start", "SLIME_BROWSER_LOCAL_PROCESS_PORT_START", _DEFAULT_PORT_START)
    port_end = _cfg_int(local_cfg, "port_end", "SLIME_BROWSER_LOCAL_PROCESS_PORT_END", _DEFAULT_PORT_END)
    startup_timeout_secs = _cfg_float(
        local_cfg,
        "startup_timeout_secs",
        "SLIME_BROWSER_LOCAL_PROCESS_STARTUP_TIMEOUT_SECS",
        _DEFAULT_STARTUP_TIMEOUT_SECS,
    )
    startup_poll_secs = _cfg_float(
        local_cfg,
        "startup_poll_secs",
        "SLIME_BROWSER_LOCAL_PROCESS_STARTUP_POLL_SECS",
        _DEFAULT_STARTUP_POLL_SECS,
    )
    request_timeout_secs = _cfg_float(
        local_cfg,
        "request_timeout_secs",
        "SLIME_BROWSER_LOCAL_PROCESS_REQUEST_TIMEOUT_SECS",
        _DEFAULT_REQUEST_TIMEOUT_SECS,
    )
    exit_timeout_secs = _cfg_float(
        local_cfg,
        "exit_timeout_secs",
        "SLIME_BROWSER_LOCAL_PROCESS_EXIT_TIMEOUT_SECS",
        _DEFAULT_EXIT_TIMEOUT_SECS,
    )
    kill_timeout_secs = _cfg_float(
        local_cfg,
        "kill_timeout_secs",
        "SLIME_BROWSER_LOCAL_PROCESS_KILL_TIMEOUT_SECS",
        _DEFAULT_KILL_TIMEOUT_SECS,
    )
    max_processes = _cfg_int(
        local_cfg,
        "max_processes",
        "SLIME_BROWSER_LOCAL_PROCESS_MAX_PROCESSES",
        _DEFAULT_MAX_PROCESSES,
    )
    acquire_timeout_secs = _cfg_float(
        local_cfg,
        "acquire_timeout_secs",
        "SLIME_BROWSER_LOCAL_PROCESS_ACQUIRE_TIMEOUT_SECS",
        0.0,
    )
    log_dir = _cfg_str(local_cfg, "log_dir", "SLIME_BROWSER_LOCAL_PROCESS_LOG_DIR", _DEFAULT_LOG_DIR)
    port_lock_dir = _cfg_str(
        local_cfg,
        "port_lock_dir",
        "SLIME_BROWSER_LOCAL_PROCESS_PORT_LOCK_DIR",
        _DEFAULT_PORT_LOCK_DIR,
    )
    python_bin = _cfg_str(local_cfg, "python_bin", "SLIME_BROWSER_LOCAL_PROCESS_PYTHON", sys.executable)

    slot_pool = await LocalProcessSlotPool.get(max_processes=max_processes)
    slot_acquired = False
    port_lease: LocalProcessPortLease | None = None
    log_file = None
    proc: asyncio.subprocess.Process | None = None
    startup_start = time.monotonic()
    try:
        await slot_pool.acquire(timeout=acquire_timeout_secs)
        slot_acquired = True
        port_lease = _acquire_port(host, port_start, port_end, port_lock_dir)
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, f"env_server_{port_lease.port}_{int(time.time() * 1000)}.log")
        log_file = open(log_path, "ab", buffering=0)

        # Inside Ray rollout actors the event loop runs on a non-main thread whose
        # active policy has no asyncio child watcher, so create_subprocess_exec
        # raises NotImplementedError. Install a ThreadedChildWatcher (works from
        # any thread) and make the policy return it, so subprocess spawning works.
        _ensure_asyncio_child_watcher()

        proc = await asyncio.create_subprocess_exec(
            python_bin,
            "-m",
            "openwebrl.docker.env_server",
            "--host",
            host,
            "--port",
            str(port_lease.port),
            stdout=log_file,
            stderr=log_file,
            cwd=_PROJECT_ROOT,
            start_new_session=True,
        )
        base_url = f"http://{host}:{port_lease.port}"
        logger.info(
            "[LocalProcessCreate] start pid=%s port=%s log=%s max_processes=%s",
            proc.pid,
            port_lease.port,
            log_path,
            max_processes,
        )
        await _wait_until_healthy(
            base_url,
            proc,
            startup_timeout_secs=startup_timeout_secs,
            startup_poll_secs=startup_poll_secs,
        )
        startup_ms = (time.monotonic() - startup_start) * 1000
        logger.info("[LocalProcessCreate] ready pid=%s port=%s startup_ms=%.1f", proc.pid, port_lease.port, startup_ms)
        return LocalProcessWebEnv(
            proc=proc,
            port_lease=port_lease,
            slot_pool=slot_pool,
            log_file=log_file,
            request_timeout_secs=request_timeout_secs,
            exit_timeout_secs=exit_timeout_secs,
            kill_timeout_secs=kill_timeout_secs,
            startup_ms=startup_ms,
        )
    except BaseException:
        if proc is not None and proc.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            with contextlib.suppress(Exception):
                await proc.wait()
        if port_lease is not None:
            port_lease.release()
        if log_file is not None:
            with contextlib.suppress(Exception):
                log_file.close()
        if slot_acquired:
            await slot_pool.release()
        raise
