import asyncio
import os
import threading

__all__ = ["get_async_loop", "run"]


def _create_background_loop() -> asyncio.AbstractEventLoop:
    """Create a stable loop for the background helper thread.

    By default we force the stdlib selector loop on Unix to avoid uvloop/libuv
    transport races in long-lived background threads under heavy cancellation.
    Set ``SLIME_ASYNC_USE_STDLIB_LOOP=0`` to restore the active loop policy.
    """
    use_stdlib_loop = os.environ.get("SLIME_ASYNC_USE_STDLIB_LOOP", "1") == "1"
    if use_stdlib_loop and os.name != "nt" and hasattr(asyncio, "SelectorEventLoop"):
        return asyncio.SelectorEventLoop()
    return asyncio.new_event_loop()


# Create a background event loop thread
class AsyncLoopThread:
    def __init__(self):
        self.loop = _create_background_loop()
        self._thread = threading.Thread(target=self._start_loop, daemon=True)
        self._thread.start()

    def _start_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def run(self, coro):
        # Schedule a coroutine onto the loop and block until it's done
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result()


# Create one global instance
async_loop = None


def get_async_loop():
    global async_loop
    if async_loop is None:
        async_loop = AsyncLoopThread()
    return async_loop


def run(coro):
    """Run a coroutine in the background event loop."""
    return get_async_loop().run(coro)
