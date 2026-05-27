"""
Remote Browser Environment Client

Provides ``RemoteWebEnv`` – a drop-in replacement for ``WebEnv`` that
communicates with the Browser Environment Server (``env_server.py``) via
HTTP POST requests.

Also provides ``RemoteEnvPool`` for managing env-id allocation across
concurrent generate() calls.  Supports **multiple** env servers — the pool
auto-discovers the number of envs on each server via ``GET /health`` and
distributes work across all of them transparently.

Usage in generate_browser.py:
    # Single server
    BROWSER_ENV_SERVER_URL=http://browser-env:8100

    # Multiple servers (comma-separated)
    BROWSER_ENV_SERVER_URL=http://server-a:8100,http://server-b:8100
"""

import asyncio
import base64
import logging
from typing import Any, Dict, List, Optional, Tuple

import aiohttp

logger = logging.getLogger(__name__)

# Default timeout for HTTP requests (seconds).
_HTTP_TIMEOUT = aiohttp.ClientTimeout(total=300)


class RemoteWebEnv:
    """
    Drop-in replacement for ``WebEnv`` that forwards all operations to the
    Browser Environment Server via HTTP.

    The interface mirrors WebEnv: ``setup()``, ``reset()``, ``step()``,
    ``exit()``.
    """

    def __init__(
        self,
        server_url: str,
        env_id: int,
        pool: Optional["RemoteEnvPool"] = None,
    ) -> None:
        self.server_url = server_url.rstrip("/")
        self.env_id = env_id
        self.pool = pool

        # Cached result from ``initialize()`` so that the first
        # ``reset()`` call returns it without an extra round-trip.
        self._cached_reset: Optional[Tuple[dict, dict]] = None
        self._task_data: Optional[dict] = None

    # ------------------------------------------------------------------
    # Public interface (mirrors WebEnv)
    # ------------------------------------------------------------------

    async def setup(self) -> None:
        """No-op – the server manages Playwright setup."""
        pass

    async def initialize(
        self,
        task_id: str,
        config_overrides: Optional[Dict[str, Any]] = None,
    ) -> dict:
        """
        Send ``POST /reset`` to the server, creating / resetting the
        environment for *task_id*.  Caches the observation so that the
        next ``reset()`` call returns it immediately.

        Returns:
            task_data dict loaded by the server.
        """
        payload = {
            "env_id": self.env_id,
            "task_id": task_id,
        }
        if config_overrides:
            payload["config_overrides"] = config_overrides

        data = await self._post("/reset", payload)

        obs = data["observation"]
        obs["screenshot"] = self._decode_screenshot(obs.get("screenshot"))
        # screen_size comes back as a list; keep as-is (callers use [0]/[1])

        self._cached_reset = (obs, data["info"])
        self._task_data = data["task_data"]
        return data["task_data"]

    async def reset(
        self,
        url: Optional[str] = None,
        auth_info: Optional[dict] = None,
    ) -> Tuple[dict, dict]:
        """
        Return the cached observation from ``initialize()``.

        This mirrors the WebEnv call in ``generate_browser.generate()``:
            ``observation, info = await env.reset()``
        """
        if self._cached_reset is not None:
            result = self._cached_reset
            self._cached_reset = None
            return result

        raise RuntimeError(
            "RemoteWebEnv.reset() called without a prior initialize(). "
            "Call initialize(task_id) first."
        )

    async def step(
        self, action_list: List[Dict[str, Any]]
    ) -> Tuple[dict, float, bool, bool, dict]:
        """Execute actions via ``POST /step`` and return gym-style tuple."""
        data = await self._post("/step", {
            "env_id": self.env_id,
            "actions": action_list,
        })

        obs = data["observation"]
        obs["screenshot"] = self._decode_screenshot(obs.get("screenshot"))

        return (
            obs,
            data["reward"],
            data["terminated"],
            data["truncated"],
            data["info"],
        )

    async def exit(self) -> None:
        """Close the environment via ``POST /exit``, then release the env-id."""
        try:
            await self._post("/exit", {"env_id": self.env_id})
        except Exception as exc:
            logger.warning("RemoteWebEnv exit error (ignored): %s", exc)
        finally:
            if self.pool is not None:
                self.pool.release(self.server_url, self.env_id)
            else:
                raise RuntimeError(
                    "RemoteWebEnv created without a pool; cannot release env_id."
                )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _post(self, path: str, payload: dict) -> dict:
        """Send a JSON POST request and return the parsed response."""
        url = f"{self.server_url}{path}"
        async with aiohttp.ClientSession(timeout=_HTTP_TIMEOUT) as session:
            async with session.post(url, json=payload) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    raise RuntimeError(
                        f"Browser env server error [{resp.status}] "
                        f"{path}: {text}"
                    )
                return await resp.json()

    @staticmethod
    def _decode_screenshot(value: Any) -> bytes:
        """Decode a base64-encoded screenshot back to bytes."""
        if isinstance(value, str):
            return base64.b64decode(value)
        if isinstance(value, (bytes, bytearray)):
            return bytes(value)
        return b""


# ---------------------------------------------------------------------------
# Environment pool (supports multiple servers)
# ---------------------------------------------------------------------------


class RemoteEnvPool:
    """
    Manages a pool of ``(server_url, env_id)`` pairs across one or more
    browser env servers.

    Each ``generate()`` call acquires a slot, uses it for the full
    interaction, then releases it.  The pool auto-discovers each server's
    capacity via ``GET /health``.
    """

    def __init__(self, server_urls: List[str]) -> None:
        self.server_urls = [url.rstrip("/") for url in server_urls]
        self.total_envs = 0
        self._available: asyncio.Queue[Tuple[str, int]] = asyncio.Queue()

    async def discover(self) -> None:
        """Query ``/health`` on each server to discover available env slots."""
        for url in self.server_urls:
            try:
                async with aiohttp.ClientSession(timeout=_HTTP_TIMEOUT) as session:
                    async with session.get(f"{url}/health") as resp:
                        if resp.status != 200:
                            logger.warning("Server %s /health returned %d, skipping", url, resp.status)
                            continue
                        data = await resp.json()
                        num_envs = data["num_envs"]
                        for i in range(num_envs):
                            self._available.put_nowait((url, i))
                        logger.info("Discovered %d envs on %s", num_envs, url)
            except Exception as exc:
                logger.warning("Failed to discover envs on %s: %s", url, exc)
        self.total_envs = self._available.qsize()
        if self.total_envs == 0:
            raise RuntimeError(
                f"No browser envs discovered from servers: {self.server_urls}"
            )

    async def acquire(self) -> Tuple[str, int]:
        """Block until a ``(server_url, env_id)`` slot is available."""
        return await self._available.get()

    def release(self, server_url: str, env_id: int) -> None:
        """Return a slot to the pool."""
        self._available.put_nowait((server_url, env_id))

    def create_remote_env(self, server_url: str, env_id: int) -> RemoteWebEnv:
        """Create a ``RemoteWebEnv`` bound to the given server and env-id."""
        return RemoteWebEnv(server_url, env_id, pool=self)


# ---------------------------------------------------------------------------
# Singleton pool accessor
# ---------------------------------------------------------------------------

_pool: Optional[RemoteEnvPool] = None
_pool_lock = asyncio.Lock()


async def get_env_pool(server_urls: str) -> RemoteEnvPool:
    """
    Get or create the singleton ``RemoteEnvPool``.

    Args:
        server_urls: Comma-separated list of server URLs,
            e.g. ``"http://host:8100"`` or ``"http://a:8100,http://b:8100"``.
    """
    global _pool
    if _pool is None:
        async with _pool_lock:
            if _pool is None:
                urls = [u.strip() for u in server_urls.split(",") if u.strip()]
                _pool = RemoteEnvPool(urls)
                await _pool.discover()
                logger.info(
                    "Created RemoteEnvPool: servers=%s, total_envs=%d",
                    urls, _pool.total_envs,
                )
    return _pool


async def close_env_pool() -> None:
    """Close the singleton RemoteEnvPool and release resources."""
    global _pool
    _pool = None
