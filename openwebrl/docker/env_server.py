"""
Browser Environment HTTP Server

Serves a single browser environment via HTTP POST endpoints.
Requests are serialized via an async lock.

All configuration (env_config, tool_list, policy, task_data) is provided
by the client in the /reset request — the server does not load any local
config/task files.

Endpoints:
    POST /reset   - Reset/create env with a new task
    POST /step    - Execute actions
    POST /exit    - Close the env
    GET  /health  - Health check
"""

import argparse
import asyncio
import base64
import json
import logging
import os
import resource
import sys
import time
import traceback
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_BROWSER_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
_PROJECT_ROOT = os.path.abspath(os.path.join(_BROWSER_DIR, "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from openwebrl.env.web_env import WebEnv

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("env_server")


# ---------------------------------------------------------------------------
# Single environment state
# ---------------------------------------------------------------------------
_env: Optional[WebEnv] = None
_lock = asyncio.Lock()
_task_id: Optional[str] = None
_task_data: Optional[dict] = None
_last_used_at: Optional[float] = None


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------
def _serialize_observation(obs: dict) -> dict:
    result = {}
    for key, value in obs.items():
        if key == "screenshot" and isinstance(value, (bytes, bytearray)):
            result[key] = base64.b64encode(value).decode("utf-8")
        else:
            result[key] = value
    return result


# ---------------------------------------------------------------------------
# Server diagnostics
# ---------------------------------------------------------------------------
_server_start_time: float = time.time()
_request_counter: Dict[str, int] = {"reset": 0, "step": 0, "exit": 0}


def _get_server_diagnostics(endpoint: str, elapsed_secs: float) -> dict:
    try:
        rusage = resource.getrusage(resource.RUSAGE_SELF)
        rss_mb = round(rusage.ru_maxrss / 1024, 1)
    except Exception:
        rss_mb = None

    return {
        "endpoint": endpoint,
        "elapsed_secs": round(elapsed_secs, 4),
        "timestamp": time.time(),
        "server_uptime_secs": round(time.time() - _server_start_time, 1),
        "rss_mb": rss_mb,
        "request_counts": dict(_request_counter),
        "initialized": _env is not None,
        "task_id": _task_id,
    }


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------
class ResetRequest(BaseModel):
    task_id: str
    task_data: Dict[str, Any]
    env_config: Dict[str, Any]
    tool_list: List[Dict[str, Any]] = []
    policy: str = ""


class StepRequest(BaseModel):
    actions: List[Dict[str, Any]]


class EnvIdRequest(BaseModel):
    env_id: int = 0


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="Browser Environment Server")


@app.on_event("startup")
async def _startup() -> None:
    logger.info("Environment server started (single-env mode)")


# ---- POST /reset ----------------------------------------------------------

@app.post("/reset")
async def reset_env(request: ResetRequest):
    global _env, _task_id, _task_data, _last_used_at

    t0 = time.monotonic()
    _request_counter["reset"] += 1

    async with _lock:
        if _env is not None:
            raise HTTPException(status_code=400, detail="Unexpected /reset call: Env already exists.")

        try:
            task_data = request.task_data
            config = request.env_config

            env = WebEnv(
                width=config["width"],
                height=config["height"],
                dpr=config["dpr"],
                max_retries=config["max_retries"],
                wait_timeout=config["wait_timeout"],
                screenshot_timeout=config.get("screenshot_timeout"),
                start_url=task_data["start_url"],
                resize_output_coords=config["resize_output_coords"],
                resize_scale=config["resize_scale"],
                image_patch_size=config["image_patch_size"],
                tool_list=request.tool_list,
                policy=request.policy,
            )
            await env.setup()
            observation, info = await env.reset()
            
            _env = env
            _task_id = request.task_id
            _task_data = task_data
            _last_used_at = time.monotonic()

            return {
                "observation": _serialize_observation(observation),
                "info": info,
                "diagnostics": _get_server_diagnostics("reset", time.monotonic() - t0),
            }
        except Exception as exc:
            logger.error("/reset failed for task %s: %r",
                         request.task_id, exc, exc_info=True)
            diag = _get_server_diagnostics("reset", time.monotonic() - t0)
            diag["error"] = repr(exc)
            diag["error_traceback"] = traceback.format_exc()
            raise HTTPException(500, {
                "message": f"/reset failed for task {request.task_id}: {repr(exc)}",
                "diagnostics": diag,
            })


# ---- POST /step -----------------------------------------------------------

@app.post("/step")
async def step_env(request: StepRequest):
    global _last_used_at

    if _env is None:
        raise HTTPException(status_code=400, detail="env not initialized – call /reset first")

    t0 = time.monotonic()
    _request_counter["step"] += 1

    async with _lock:
        try:
            observation, reward, terminated, truncated, info = await _env.step(request.actions)
        
        except Exception as exc:
            logger.error("step failed: %r", exc, exc_info=True)
            diag = _get_server_diagnostics("step", time.monotonic() - t0)
            diag["error"] = repr(exc)
            diag["error_traceback"] = traceback.format_exc()
            raise HTTPException(500, detail={
                "message": f"step failed: {repr(exc)}",
                "diagnostics": diag,
            })

        _last_used_at = time.monotonic()

        return {
            "observation": _serialize_observation(observation),
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
            "info": info,
            "diagnostics": _get_server_diagnostics("step", time.monotonic() - t0),
        }


# ---- POST /exit -----------------------------------------------------------

@app.post("/exit")
async def exit_env(request: EnvIdRequest):
    global _env, _task_id, _task_data, _last_used_at

    t0 = time.monotonic()
    _request_counter["exit"] += 1

    async with _lock:
        if _env is None:
            raise HTTPException(status_code=400, detail="env not initialized – call /reset first")

        try:
            await _env.exit()
        except Exception as exc:
            raise HTTPException(500, detail={
                "error": f"exit failed: {repr(exc)}",
                "diagnostics": _get_server_diagnostics("exit", time.monotonic() - t0),
            })
        finally:
            _env = None
            _task_id = None
            _task_data = None
            _last_used_at = None

        return {
            "status": "ok",
            "diagnostics": _get_server_diagnostics("exit", time.monotonic() - t0),
        }


# ---- GET /health ----------------------------------------------------------

@app.get("/health")
async def health():
    return {
        "initialized": _env is not None,
        "task_id": _task_id,
        "locked": _lock.locked(),
        "idle_secs": (
            round(time.monotonic() - _last_used_at, 1)
            if _last_used_at
            else None
        ),
    }


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Browser Environment Server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8100)
    parser.add_argument(
        "--log-level",
        default=os.environ.get("BROWSER_ENV_SERVER_LOG_LEVEL", "warning"),
    )
    parser.add_argument(
        "--access-log",
        action=argparse.BooleanOptionalAction,
        default=os.environ.get("BROWSER_ENV_SERVER_ACCESS_LOG", "0").strip().lower() in {"1", "true", "yes", "on"},
    )
    args = parser.parse_args()

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level=args.log_level,
        access_log=args.access_log,
    )


if __name__ == "__main__":
    main()
