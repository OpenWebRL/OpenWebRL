# OpenWebRL Browser Environments

This directory contains the browser environment runtime used by OpenWebRL rollouts. It wraps a Playwright-based `WebEnv` behind a small HTTP server, then lets the rollout code launch that server either as a local subprocess or inside sandbox pods.

The root [README.md](../README.md) is the main project entry point. This document focuses on the WebEnv pieces under `openwebrl/`.

## Directory Overview

| Path | Purpose |
|---|---|
| `generate_browser.py` | Main browser rollout generator used by training and evaluation scripts. |
| `env/web_env.py` | Playwright browser environment implementation. |
| `env/local_process_env.py` | Starts one local `env_server` subprocess per browser environment. |
| `env/sandbox_env.py` | Creates sandbox pods, starts `env_server` inside each pod, and communicates with it through exec/curl. |
| `docker/env_server.py` | FastAPI HTTP server around `WebEnv`. |
| `docker/Dockerfile.browser` | Browser environment image for sandbox or standalone HTTP-server testing. |
| `env/config.yaml` | Browser environment defaults. |
| `env/prompts/` | System prompt and browser tool definitions. |
| `data/` | Browser task files and parquet training/evaluation data. |
| `reward_browser.py` | Browser-task reward and answer checking utilities. |
| `response_format.py` | Browser action/response formatting helpers. |

## Supported Runtime Modes

`generate_browser.py` currently supports three modes. You can set them in `openwebrl/env/config.yaml` or override with `SLIME_BROWSER_ENV_MODE`.

| Mode | Description | Typical Use |
|---|---|---|
| `local_process` | Starts local `openwebrl.docker.env_server` subprocesses and talks to them over HTTP. | Local smoke tests and small evaluations. |
| `sandbox` | Creates fresh sandbox pods through the sandbox orchestrator ([Orchard](https://github.com/microsoft/Orchard)), starts `env_server` inside each pod, and deletes pods on exit. | Scalable training and evaluation. |
| `browser-use` | Creates a fresh remote browser session on [cloud.browser-use.com](https://cloud.browser-use.com) per environment and drives it via CDP. Requires the `browser_use_sdk` package and `BROWSER_USE_API_KEY`. | Running against a hosted browser without a sandbox cluster. |

The Docker server can also be run directly for validating the browser image and HTTP API. The training/evaluation launchers use `local_process`, `sandbox`, or `browser-use`.

## Required Environment

Install the project dependencies from the repository root:

```bash
pip install -r requirements.txt
playwright install chromium
```

For Linux containers, Chromium may also need system browser libraries. The browser Dockerfile installs the expected runtime packages for the container path.

Sandbox mode additionally requires:

```bash
export SANDBOX_ORCHESTRATOR_URL=http://<orchestrator-host>
export SANDBOX_API_KEY=<optional-api-key>
export BROWSER_SANDBOX_IMAGE=<registry-or-local-image>/browser-env:latest
```

Browser-Use mode additionally requires the `browser_use_sdk` package and an API key:

```bash
export BROWSER_USE_API_KEY=<your-browser-use-api-key>
```

Do not commit real orchestrator URLs, tokens, API keys, or private registry names. Keep them in environment variables or your cluster secret manager.

## Quick Start

### Local Process Smoke Test

Use this path when you want to verify the browser environment without a sandbox cluster:

```bash
export SLIME_BROWSER_ENV_MODE=local_process
export MODEL_PATH=<path-or-hf-id>
export TASK_FILE=openwebrl/data/online-mind2web.jsonl

bash scripts/run_evaluation_local.sh
```

`local_process` allocates ports from `openwebrl/env/config.yaml` and writes logs under `/tmp/slime_browser_local_process_env_logs` by default. You can override the port/log directories with:

```bash
export SLIME_BROWSER_LOCAL_PROCESS_PORT_START=18000
export SLIME_BROWSER_LOCAL_PROCESS_PORT_END=18999
export SLIME_BROWSER_LOCAL_PROCESS_LOG_DIR=/tmp/slime_browser_local_process_env_logs
export SLIME_BROWSER_LOCAL_PROCESS_PORT_LOCK_DIR=/tmp/slime_browser_local_process_ports
```

### Sandbox Training/Evaluation

OpenWebRL's sandbox mode is designed to work with [Orchard](https://github.com/microsoft/Orchard), an open-source Kubernetes-native sandbox framework that provides per-episode network isolation and scales to hundreds of concurrent browser instances. Compared to `local_process`, sandbox isolation significantly reduces the rate at which websites block agent traffic — particularly important during online RL where GRPO-style group rollouts repeatedly query the same site within a single training step.

Build and publish a browser environment image that your sandbox cluster can pull:

```bash
docker build -f openwebrl/docker/Dockerfile.browser -t browser-env .
```

Then configure the sandbox connection and launch training or evaluation:

```bash
export SLIME_BROWSER_ENV_MODE=sandbox
export SANDBOX_ORCHESTRATOR_URL=http://<orchestrator-host>
export BROWSER_SANDBOX_IMAGE=<registry-or-local-image>/browser-env:latest

bash scripts/run_browser_Qwen3VL_4B_Instruct.sh
# or
bash scripts/run_evaluation.sh
```

Sandbox concurrency is controlled by environment variables used by the launch scripts and `env/config.yaml`:

```bash
export SLIME_BROWSER_SANDBOX_MAX_SANDBOXES=32
export SLIME_BROWSER_SANDBOX_ACQUIRE_TIMEOUT_SECS=1800
export SLIME_BROWSER_SANDBOX_MANIFEST_DIR=/tmp/slime_browser_sandboxes
```

Each browser environment creates a sandbox pod, starts `python -m openwebrl.docker.env_server` inside it, and deletes the sandbox during `env.exit()`. If a run is interrupted, clean leftover sandboxes with:

```bash
bash scripts/clean_processes.sh
```

or directly:

```bash
python openwebrl/env/sandbox_env.py --cleanup --cleanup-all
```

## Standalone Docker Server

The standalone server is useful for checking the browser image, debugging the HTTP API, or testing `env_server.py` outside the rollout launcher:

```bash
docker build -f openwebrl/docker/Dockerfile.browser -t browser-env .

docker run --rm \
  -p 8100:8100 \
  -e NUM_BROWSER_ENVS=4 \
  -e BROWSER_ENV_IDLE_TIMEOUT_SECS=600 \
  --shm-size=2g \
  browser-env
```

Health check:

```bash
curl http://localhost:8100/health
```

Docker Compose is also available:

```bash
docker compose -f openwebrl/docker/docker-compose.browser.yaml up --build
```

## Configuration

The main browser config lives in `openwebrl/env/config.yaml`.

Important fields:

| Field | Meaning |
|---|---|
| `mode` | `local_process`, `sandbox`, or `browser-use`. Can be overridden by `SLIME_BROWSER_ENV_MODE`. |
| `sandbox.orchestrator_url` | Sandbox orchestrator URL. Prefer `SANDBOX_ORCHESTRATOR_URL`. |
| `sandbox.api_key` | Sandbox API key. Prefer `SANDBOX_API_KEY`. |
| `sandbox.image` | Browser image. Prefer `BROWSER_SANDBOX_IMAGE`. |
| `sandbox.max_sandboxes` | Maximum concurrent sandbox environments in one rollout process. |
| `local_process.port_start` / `port_end` | Port range for local env servers. |
| `browser_use.api_key` | Browser-Use API key. Prefer `BROWSER_USE_API_KEY`. |
| `browser_use.timeout` | Remote session timeout in minutes (free: ≤15, paid: ≤240). |
| `browser_use.profile_id` / `proxy_country_code` | Optional Browser-Use profile and proxy country. |
| `path_to_policy` | Browser system prompt path. |
| `path_to_tool_list` | Browser tool schema path. |
| `path_to_task_file` | Default task file. |
| `use_screenshot` | Include screenshots in observations. |
| `use_a11ytree` | Include accessibility tree observations. |

## Data

The default task file in `env/config.yaml` is:

```yaml
path_to_task_file: data/online-mind2web.jsonl
```

Training and validation parquet files live under `openwebrl/data/`. Keep the release data small enough for GitHub, and move larger generated artifacts to a separate hosting location if needed.

## HTTP API

`openwebrl/docker/env_server.py` exposes the following endpoints:

| Endpoint | Purpose |
|---|---|
| `GET /health` | Check that the server is alive. |
| `POST /reset` | Reset or initialize an environment for a task. |
| `POST /step` | Execute one browser action. |
| `POST /exit` | Close an environment. |

Example reset request:

```bash
curl -X POST http://localhost:8100/reset \
  -H "Content-Type: application/json" \
  -d '{"env_id": 0, "task_id": "0"}'
```

Example step request:

```bash
curl -X POST http://localhost:8100/step \
  -H "Content-Type: application/json" \
  -d '{"env_id": 0, "action": "click(100, 200)"}'
```

## Troubleshooting

If Chromium is missing, run:

```bash
playwright install chromium
```

If local subprocess mode cannot allocate a port, adjust:

```bash
export SLIME_BROWSER_LOCAL_PROCESS_PORT_START=18000
export SLIME_BROWSER_LOCAL_PROCESS_PORT_END=18999
```

If sandbox pods are left behind after an interrupted run, run:

```bash
bash scripts/clean_processes.sh
```

If Chromium crashes inside Docker, increase shared memory:

```bash
docker run --shm-size=2g ...
```
