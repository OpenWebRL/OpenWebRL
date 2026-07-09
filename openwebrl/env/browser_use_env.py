"""Browser-Use remote browser environment. See ``notes/browser_use_env.md``."""

import argparse
import asyncio
import logging
import os
import sys
from typing import Any, Dict, List, Optional

# Allow running as a script (`python openwebrl/env/browser_use_env.py
# --cleanup`) in addition to `python -m openwebrl.env.browser_use_env`.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from playwright.async_api import async_playwright

from openwebrl.env.web_env import WebEnv

logger = logging.getLogger(__name__)

# Poll for cdp_url up to _CDP_POLL_TIMEOUT seconds after create().
_CDP_POLL_TIMEOUT = 60
_CDP_POLL_INTERVAL = 2

# ---------------------------------------------------------------------------
# Session marker manifest (Ctrl+C-safe cleanup)
#
# Mirrors the ``.sandboxes/`` pattern in ``sandbox_env.py``: create one empty
# marker file per live remote session; remove it on successful stop; sweep
# leftovers at the start of the next run (or via ``--cleanup`` CLI).
# ---------------------------------------------------------------------------
_BROWSER_USE_MANIFEST_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), ".browser_use_sessions"
)
print("Browser-Use session manifest directory:", _BROWSER_USE_MANIFEST_DIR)


def _save_session_id(session_id: Any) -> None:
    os.makedirs(_BROWSER_USE_MANIFEST_DIR, exist_ok=True)
    with open(os.path.join(_BROWSER_USE_MANIFEST_DIR, str(session_id)), "w") as f:
        f.write("")


def _remove_session_id(session_id: Any) -> None:
    try:
        os.remove(os.path.join(_BROWSER_USE_MANIFEST_DIR, str(session_id)))
    except OSError:
        pass


def _list_session_ids() -> List[str]:
    if not os.path.isdir(_BROWSER_USE_MANIFEST_DIR):
        return []
    try:
        return os.listdir(_BROWSER_USE_MANIFEST_DIR)
    except OSError:
        return []


async def cleanup_existing_browser_use_sessions(
    browser_use_cfg: Optional[Dict[str, Any]] = None,
) -> None:
    """Stop any leftover sessions recorded in the manifest directory."""
    session_ids = _list_session_ids()
    if not session_ids:
        logger.info("No leftover Browser-Use session markers found.")
        return

    from browser_use_sdk import AsyncBrowserUse

    api_key = (
        os.environ.get("BROWSER_USE_API_KEY")
        or (browser_use_cfg or {}).get("api_key")
    )
    if not api_key:
        raise ValueError(
            "Set BROWSER_USE_API_KEY env var or browser_use.api_key in config.yaml"
        )

    logger.info(f"Found {len(session_ids)} leftover Browser-Use session(s), stopping...")

    client = AsyncBrowserUse(api_key=api_key)
    for sid in session_ids:
        try:
            await client.browsers.stop(sid)
            print(f"♻️  [Stopped Browser-Use session {sid}]")
        except Exception as exc:
            # session_not_found / already-stopped → same outcome, drop marker too.
            logger.warning("Stop failed for %s (may already be gone): %s", sid, exc)
        _remove_session_id(sid)


class BrowserUseWebEnv(WebEnv):
    """WebEnv variant that drives a remote Browser-Use browser via CDP."""

    def __init__(
        self,
        cdp_url: str,
        session_id: str,
        bu_client: Any,
        **web_env_kwargs: Any,
    ) -> None:
        super().__init__(**web_env_kwargs)
        self.cdp_url = cdp_url
        self.session_id = session_id
        self._bu_client = bu_client

    async def setup(self) -> None:
        self.playwright = await async_playwright().start()
        self.browser_type = self.playwright.chromium
        self.browser = await self.browser_type.connect_over_cdp(self.cdp_url)

        await self._initialize_context(
            enable_recording=False,
            start_url=self.start_url,
            auth_info=self.auth_info,
        )

        # Browser-Use may ignore our requested browser_screen_width/height and
        # launch Chromium at its own default. The resize_scale → pixel transform
        # in WebEnv.execute_single_action relies on self.screen_size / self.dpr
        # matching the REAL viewport, or clicks land off-target. Realign to
        # whatever the remote actually gave us.
        try:
            dims = await self.page.evaluate(
                "() => ({w: window.innerWidth, h: window.innerHeight, dpr: window.devicePixelRatio})"
            )
            real_dpr = max(1, int(dims["dpr"] or 1))
            real_w = int(dims["w"]) * real_dpr
            real_h = int(dims["h"]) * real_dpr
            if (real_w, real_h, real_dpr) != (self.screen_size[0], self.screen_size[1], self.dpr):
                logger.warning(
                    "Remote viewport %dx%d @dpr=%d differs from configured "
                    "%dx%d @dpr=%d; using actual dimensions for coord transforms.",
                    real_w, real_h, real_dpr,
                    self.screen_size[0], self.screen_size[1], self.dpr,
                )
                self.screen_size = (real_w, real_h)
                self.dpr = real_dpr
                self.css_width = int(dims["w"])
                self.css_height = int(dims["h"])
        except Exception as exc:
            logger.warning("Could not query remote viewport; keeping configured dims: %s", exc)

        logger.info(
            f"BrowserUseWebEnv ready (session_id={self.session_id}, "
            f"viewport={self.screen_size}, DPR={self.dpr})"
        )

    async def _initialize_context(
        self,
        enable_recording: bool = False,
        start_url: str = "about:blank",
        auth_info: Optional[dict] = None,
    ) -> None:
        # Reuse the CDP browser's default context — remote browsers expose
        # a single shared context that can't be replaced or closed.
        self.context = (
            self.browser.contexts[0]
            if self.browser.contexts
            else await self.browser.new_context()
        )
        self.context.set_default_timeout(self.timeout)

        assert start_url
        start_urls = start_url.split(" |AND| ")
        existing_pages = list(self.context.pages)

        for i, url in enumerate(start_urls):
            page = existing_pages[i] if i < len(existing_pages) else await self.context.new_page()
            for attempt in range(self.max_retries):
                try:
                    # Return as soon as the DOM is ready — waiting for "load"
                    # times out on sites with heavy third-party resources.
                    await page.goto(url, timeout=self.timeout, wait_until="domcontentloaded")
                    # Then best-effort wait for "load" so later page.evaluate
                    # calls don't race with in-flight client-side redirects.
                    try:
                        await page.wait_for_load_state("load", timeout=5000)
                    except Exception:
                        pass
                    break
                except Exception as e:
                    if attempt == self.max_retries - 1:
                        raise EnvironmentError(
                            f"Failed to navigate to {url} after {self.max_retries} attempts: {e}"
                        )
                    logger.warning(f"Navigate to {url} failed (attempt {attempt + 1}): {e}")
                    await asyncio.sleep(2)

        self.page = self.context.pages[0]
        await self.page.bring_to_front()
        self.setup_global_page_listener()
        self.setup_dialog_interceptor()
        await asyncio.sleep(2)

    async def get_screen_size(self) -> "tuple[int, int]":
        # Playwright's page.viewport_size is None for CDP-attached browsers
        # (Playwright didn't launch this browser, so it has no viewport to
        # report). Fall back to the size we requested from Browser-Use.
        viewport = self.page.viewport_size
        if viewport:
            return int(viewport["width"] * self.dpr), int(viewport["height"] * self.dpr)
        return self.screen_size[0], self.screen_size[1]

    async def exit(self) -> None:
        # Never close context/browser — they live on the remote side. Stop
        # the session first so billing halts even if the cost query later hangs.
        try:
            await self._bu_client.browsers.stop(self.session_id)
            print(f"♻️  [Stopped Browser-Use session {self.session_id}]")
        except Exception as exc:
            logger.warning("Stop failed for %s: %s", self.session_id, exc)

        # Read the finalized billing for this session and surface it. The
        # browser/proxy split makes it easy to spot a regression of the
        # "proxy was silently on" bug: proxy_cost should be $0.0000 by default.
        try:
            view = await self._bu_client.browsers.get(self.session_id)
            bc = float(view.browser_cost or 0)
            pc = float(view.proxy_cost or 0)
            print(
                f"💵 [COST] session={str(self.session_id)[:8]} "
                f"browser=${bc:.4f} proxy=${pc:.4f} total=${bc + pc:.4f}"
            )
        except Exception as exc:
            logger.warning("Cost query failed for %s: %s", self.session_id, exc)

        _remove_session_id(self.session_id)

        if self.playwright:
            try:
                await self.playwright.stop()
            except Exception as exc:
                logger.warning("playwright.stop() failed: %s", exc)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

async def create_browser_use_env(
    browser_use_cfg: Dict[str, Any],
    env_config: Dict[str, Any],
    tool_list: Optional[List[Dict[str, Any]]],
    policy: Optional[str],
    start_url: str,
) -> BrowserUseWebEnv:
    """Create a remote Browser-Use session and wrap it in a ``BrowserUseWebEnv``."""
    from browser_use_sdk import AsyncBrowserUse

    browser_use_cfg = browser_use_cfg or {}
    api_key = os.environ.get("BROWSER_USE_API_KEY") or browser_use_cfg.get("api_key")
    if not api_key:
        raise ValueError(
            "Set BROWSER_USE_API_KEY env var or browser_use.api_key in config.yaml"
        )

    client = AsyncBrowserUse(api_key=api_key)

    # Remote screen size is in CSS pixels (pre-DPR).
    dpr = max(1, int(env_config.get("dpr", 1)))
    create_kwargs: Dict[str, Any] = {
        "browser_screen_width": int(env_config["width"]) // dpr,
        "browser_screen_height": int(env_config["height"]) // dpr,
    }
    for key in ("timeout", "profile_id"):
        if browser_use_cfg.get(key):
            create_kwargs[key] = browser_use_cfg[key]
    # Always pass proxy_country_code so YAML null/missing actually disables
    # the proxy. The SDK uses an `_UNSET` sentinel — if we omit the kwarg, the
    # field is dropped from the request body and the API defaults to "us"
    # (which then bills ~$5/GB of proxy egress).
    create_kwargs["proxy_country_code"] = browser_use_cfg.get("proxy_country_code") or None

    session = None
    try:
        print(
            f"⏱️  Creating Browser-Use session (screen="
            f"{create_kwargs['browser_screen_width']}x{create_kwargs['browser_screen_height']}, "
            f"timeout_min={create_kwargs.get('timeout')}, "
            f"proxy={create_kwargs['proxy_country_code'] or 'disabled'})..."
        )
        session = await client.browsers.create(**create_kwargs)
        # Save marker BEFORE polling for cdp_url so a Ctrl+C during the
        # poll still leaves a trace for the next-run sweep.
        _save_session_id(session.id)
        print(f"✅ [Created Browser-Use session {session.id}]")

        cdp_url = session.cdp_url
        live_url = session.live_url
        waited = 0
        while not cdp_url and waited < _CDP_POLL_TIMEOUT:
            await asyncio.sleep(_CDP_POLL_INTERVAL)
            waited += _CDP_POLL_INTERVAL
            s = await client.browsers.get(session.id)
            cdp_url = s.cdp_url
            live_url = live_url or s.live_url
        if not cdp_url:
            raise RuntimeError(
                f"Browser-Use session {session.id} returned no cdp_url within {_CDP_POLL_TIMEOUT}s"
            )
        if live_url:
            print(f"👀 [Browser-Use live view: {live_url}]")

        env = BrowserUseWebEnv(
            cdp_url=cdp_url,
            session_id=session.id,
            bu_client=client,
            width=int(env_config["width"]),
            height=int(env_config["height"]),
            dpr=dpr,
            max_retries=int(env_config["max_retries"]),
            wait_timeout=int(env_config["wait_timeout"]),
            # OpenWebRL's WebEnv requires screenshot_timeout (slime's does not);
            # None lets WebEnv fall back to max(wait_timeout, 15000).
            screenshot_timeout=env_config.get("screenshot_timeout"),
            start_url=start_url,
            resize_output_coords=bool(env_config["resize_output_coords"]),
            resize_scale=int(env_config["resize_scale"]),
            image_patch_size=int(env_config["image_patch_size"]),
            tool_list=tool_list,
            policy=policy,
        )
        await env.setup()
        return env

    except BaseException:
        if session is not None:
            try:
                await client.browsers.stop(session.id)
                print(f"♻️  [Stopped Browser-Use session {session.id} after failure]")
            except Exception as exc:
                logger.warning("Stop-on-failure failed: %s", exc)
            _remove_session_id(session.id)
        raise


# ---------------------------------------------------------------------------
# CLI entry point: python -m openwebrl.env.browser_use_env --cleanup
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Browser-Use environment utilities")
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Stop all leftover sessions recorded in .browser_use_sessions/",
    )
    args = parser.parse_args()

    if not args.cleanup:
        parser.print_help()
        sys.exit(0)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    session_ids = _list_session_ids()
    if not session_ids:
        print("No session markers found in", _BROWSER_USE_MANIFEST_DIR)
        sys.exit(0)

    print(f"Found {len(session_ids)} Browser-Use session(s) to clean up:")
    for sid in session_ids:
        print(f"  - {sid}")

    asyncio.run(cleanup_existing_browser_use_sessions(
        {"api_key": os.environ.get("BROWSER_USE_API_KEY", "")}
    ))
    print("Cleanup complete.")
