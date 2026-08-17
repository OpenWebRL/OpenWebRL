<div align="center">

# OpenWebRL: Online Multi-Turn Reinforcement Learning for Visual Web Agents

[Rui Yang](https://yangrui2015.github.io/)<sup>1*</sup>, [Qianhui Wu](https://qianhuiwu.github.io/)<sup>2*†</sup>, [Yuxi Chen](https://www.linkedin.com/in/yuxichen6/)<sup>1</sup>, [Hao Bai](https://biechi.github.io/)<sup>1</sup>, [Wenlin Yao](https://wenlinyao.github.io)<sup>2</sup>, [Hao Cheng](https://sites.google.com/site/hcheng2site/Home)<sup>2</sup>,<br>
[Baolin Peng](https://www.microsoft.com/en-us/research/people/baolinpeng/)<sup>2</sup>, [Huan Zhang](https://huan-zhang.com)<sup>1</sup>, [Tong Zhang](https://tongzhang-ml.org)<sup>1</sup>, [Jianfeng Gao](https://www.microsoft.com/en-us/research/people/jfgao/)<sup>2</sup>

<sup>1</sup> University of Illinois Urbana-Champaign, <sup>2</sup> Microsoft Research



*Equal contribution. † Project lead 

Links: [Paper](https://arxiv.org/abs/2606.02031) | [Website](https://openwebrl.github.io) | [Hugging Face](https://huggingface.co/OpenWebRL) | [Orchard Env](https://github.com/microsoft/Orchard)

</div>

OpenWebRL is a framework for training visual web agents with online multi-turn reinforcement learning on live websites. The repository builds on top of the Megatron / SGLang-based `slime` training stack and adds the browser rollout, reward, data, and evaluation components needed for web-agent RL. For large-scale parallel rollouts, OpenWebRL integrates with [Orchard](https://github.com/microsoft/Orchard), an open-source sandbox environment that provides network-isolated browser instances at scale. We also support local process for web environments.

The main browser-agent implementation lives in [`openwebrl/`](openwebrl/).
It supports Playwright-based browser interaction, multi-turn multimodal
rollouts, tool-call parsing, textual environment feedback, VLM-as-a-judge rewards, and training/evaluation scripts for Qwen3-VL style visual language models.

## 📋 TODO

- [x] Supported [SFT](sft/README.md) with Qwen3.5 (2026.06.06)
- [x] Supported [browser-use env](openwebrl/env/browser_use_env.py) for interaction (2026.07.08)
- [x] Update [evaluation script](scripts/run_evaluation.sh) for [WebVoyager](openwebrl/eval/reward_webvoyager.py), [Online-Mind2Web](openwebrl/eval/reward_online_mind2web.py), and [Deepshop](openwebrl/eval/reward_deepshop.py). (2026.07.08)
- [x] Support [RL with Qwen3.5](https://github.com/MSR-Orchard/slime/blob/main/examples/orchard_gui/scripts/run_browser_qwen3.5_9b.sh) (2026.08.04)
- [x] [Fixed native judge verdict parsing](https://github.com/OpenWebRL/OpenWebRL/issues/5). (2026.08.17)
- [ ] Release demo

## 🔭 Method At A Glance

| Stage | What it does | Main entry point |
|---|---|---|
| Data preparation | Uses curated browser tasks in parquet / JSONL format and can convert benchmark JSONL files to the training parquet schema. | [`openwebrl/data/`](openwebrl/data/) |
| Browser rollout | Runs multi-turn browser episodes with screenshots, tool calls, environment feedback, and response-format checks. | [`openwebrl/generate_browser.py`](openwebrl/generate_browser.py) |
| SFT warm start (optional, recommended) | Trains a supervised fine-tuned checkpoint to initialize online RL, via LLaMAFactory on released OpenWebRL trajectories. RL can also start directly from the base VLM. See [`sft/README.md`](sft/README.md). | [`sft/run_sft_with_llamafactory.sh`](sft/run_sft_with_llamafactory.sh) |
| Online RL training | Trains a visual web agent with turn-level or trajectory-level browser rollouts and judge-based rewards, starting from the base VLM or, recommended, an SFT warm-start checkpoint. See [Training Scripts](#training-scripts) for available launchers. | [`scripts/run_browser_Qwen3VL_4B_Instruct.sh`](scripts/run_browser_Qwen3VL_4B_Instruct.sh) |
| Reward / judge | Combines format rewards with VLM-as-a-judge success evaluation through OpenAI-compatible, Azure, or self-hosted endpoints. | [`openwebrl/reward_browser.py`](openwebrl/reward_browser.py) |
| Evaluation | Evaluates converted Hugging Face checkpoints on browser benchmark task files. | [`scripts/run_evaluation.sh`](scripts/run_evaluation.sh), [`openwebrl/run_evaluate.py`](openwebrl/run_evaluate.py) |
| Optional judge SFT | Contains utilities for training and evaluating a smaller browser judge model. | [`openwebrl/judge/`](openwebrl/judge/) |

## 📂 Repository Layout

```text
.
├── README.md
├── requirements.txt
├── set_up.sh
├── setup.py / pyproject.toml
├── train.py
├── slime/                         # Modified RL training framework
├── slime_plugins/                 # Remaining model/plugin extensions
├── tools/                         # Checkpoint conversion helper used by scripts/run_convert_hf.sh
├── docker/                        # Base training Dockerfile and patches
├── scripts/                       # OpenWebRL training/evaluation/conversion entrypoints
│   └── model_configs/             # Model argument presets used by conversion/training
├── sft/                           # SFT warm-start workflow (LLaMAFactory) producing the RL init checkpoint
│   ├── README.md                  # SFT pipeline notes
│   ├── run_sft_with_llamafactory.sh
│   ├── convert_to_openai_messages.py        # Stage 1: trajectories -> canonical OpenAI format
│   ├── prepare_openai_for_llamafactory.py   # Stage 2: canonical -> LLaMAFactory data (official chat template)
│   └── post_process_llamafactory_ckpt.py    # restore base-model config/tokenizer onto the trained checkpoint for serving/RL.
└── openwebrl/
    ├── README.md                  # Detailed browser-environment notes
    ├── generate_browser.py        # Multi-turn browser rollout driver
    ├── reward_browser.py          # Format reward + VLM judge reward
    ├── response_format.py         # Browser-agent response-format handling
    ├── browser_training_config.yaml
    ├── run_evaluate.py
    ├── data/
    │   ├── webgym_filtered_popular_2102_cleaned.parquet
    │   ├── webvoyager_val.parquet
    │   ├── online-mind2web.jsonl
    │   └── convert_benchmark_jsonl_to_parquet.py
    ├── docker/                    # Browser env server Dockerfile / compose / FastAPI server
    ├── env/                       # Playwright browser env and local/sandbox clients
    └── judge/                     # Optional judge-evaluation utilities
```

## 📊 Included Data

The release includes small browser datasets under [`openwebrl/data/`](openwebrl/data/):

| File | Use |
|---|---|
| `webgym_filtered_popular_2102_cleaned.parquet` | Default RL training prompts used by the browser training launcher. |
| `webvoyager_val.parquet` | Default validation prompts. |
| `online-mind2web.jsonl` | Online-Mind2Web task file for evaluation. |



## 🛠️ Installation

OpenWebRL expects Python 3.10 or newer and a CUDA-capable environment for the
full training stack.

```bash
# 1. Install Python dependencies, CUDA/SGLang pins, and Playwright Chromium.
bash set_up.sh

# 2. Install this repository in editable mode.
pip install -e .
```

The browser environment can run in three modes configured in
[`openwebrl/env/config.yaml`](openwebrl/env/config.yaml):

| Mode | Description |
|---|---|
| `local_process` | Starts local `env_server.py` subprocesses on this machine. Useful for debugging and small evaluation runs. |
| `sandbox` | Uses [Orchard Env](https://github.com/microsoft/Orchard) to create isolated browser pods for large-scale parallel rollout. |
| `browser-use` | Drives a remote browser hosted on [cloud.browser-use.com](https://cloud.browser-use.com) via CDP. Requires `BROWSER_USE_API_KEY` and the `browser_use_sdk` package. |

We recommend sandbox mode for large-scale rollouts. OpenWebRL integrates with [Orchard](https://github.com/microsoft/Orchard), an open-source Kubernetes-native sandbox framework that provides per-episode network isolation and scales to hundreds of concurrent browser instances. Isolation significantly reduces the rate at which real websites block agent traffic — in our Online-Mind2Web evaluation without browser base service, the block rate dropped from **25.7% (local process) to 17.7% (Orchard sandbox)**. This advantage is amplified during online RL, where GRPO-style group rollouts repeatedly query the same site within a single training step, making per-site rate limiting a much more severe bottleneck.

For sandbox mode, build and publish the browser environment image to a registry
your cluster can pull from:

```bash
docker build -f openwebrl/docker/Dockerfile.browser \
  -t <your-registry>/browser-env:latest .

docker push <your-registry>/browser-env:latest
```

Then set `BROWSER_SANDBOX_IMAGE`, `SANDBOX_ORCHESTRATOR_URL`, and
`SANDBOX_API_KEY` in your environment.

## 🔑 Required Environment

Start from the template:

```bash
cp .env.example .env
$EDITOR .env
source .env
```

Core paths:

| Variable | Description |
|---|---|
| `SLIME_REPO_ROOT` | Absolute path to this repository. |
| `SLIME_MODEL_ROOT` | Directory containing pretrained and converted model checkpoints. |
| `SLIME_SAVE_ROOT` | Directory where training writes checkpoints and logs. |
| `SLIME_OUTPUT_ROOT` | Directory for evaluation outputs and debug traces. |
| `PYTHONPATH` | Should include this repo and your Megatron-LM checkout. |

Browser environment:

| Variable | Description |
|---|---|
| `SLIME_BROWSER_ENV_MODE` | `sandbox`, `local_process`, or `browser-use`. |
| `BROWSER_SANDBOX_IMAGE` | Browser container image for sandbox mode. |
| `SANDBOX_ORCHESTRATOR_URL` | Sandbox orchestrator URL for sandbox mode. |
| `SANDBOX_API_KEY` | Sandbox orchestrator API key. |
| `BROWSER_USE_API_KEY` | [Browser-Use](https://cloud.browser-use.com) API key (only for `browser-use` mode). |

Judge / reward:

| Variable | Description |
|---|---|
| `JUDGE_API_MODE` | `token`, `api_key`, or `served`. |
| `JUDGE_MODEL` | Judge model name, deployment name, or local/self-hosted model id. |
| `JUDGE_API_BASE` | OpenAI-compatible endpoint for `served` or `api_key` modes. |
| `JUDGE_API_KEY` | Judge endpoint key if needed. |
| `OPENAI_API_KEY` | Optional OpenAI-compatible API key. |
| `AZURE_RESOURCE_NAME` | Optional Azure resource name for token-mode judge calls. |
| `AZURE_API_VERSION` | Optional Azure API version. |

Experiment tracking:

| Variable | Description |
|---|---|
| `WANDB_API_KEY` | Optional. If unset, W&B logging stays disabled in the launcher scripts. |
| `WANDB_ENTITY` | Optional W&B entity/team. |

## 🚀 Quick Start

### 1. Prepare the environment

```bash
bash set_up.sh
pip install -e .
cp .env.example .env
$EDITOR .env
source .env
```

### 2. Check browser config

Edit [`openwebrl/env/config.yaml`](openwebrl/env/config.yaml):

```yaml
mode: sandbox        # or local_process
path_to_task_file: data/online-mind2web.jsonl
use_screenshot: true
use_a11ytree: false
```

For quick local debugging, set:

```yaml
mode: local_process
```

### 3. Train with online browser RL

By default the launcher's `MODEL_NAME` points to a model under `SLIME_MODEL_ROOT`, and online RL can start directly from a base VLM. For best results we **recommend** an SFT warm start first and initializing RL from that checkpoint. The SFT workflow lives in [`sft/`](sft/) — it trains an OpenWebRL SFT checkpoint from the released trajectories with LLaMAFactory and post-processes it for RL reuse (see [`sft/README.md`](sft/README.md)). To include the SFT stage, produce the checkpoint, place it under `SLIME_MODEL_ROOT`, and set `MODEL_NAME` to its path before launching RL.

#### Training Scripts

| Script | Description |
|---|---|
| `run_browser_Qwen3VL_4B_Instruct.sh` | Main MM-GRPO training launcher for Qwen3-VL-4B. Default entry point for reproducing OpenWebRL-4B. |
| `run_browser_Qwen3VL_8B_Instruct.sh` | MM-GRPO training launcher for Qwen3-VL-8B. |

The main launcher is:

```bash
bash scripts/run_browser_Qwen3VL_4B_Instruct.sh
```

Before running, set at least:

```bash
export SLIME_MODEL_ROOT=<path-to-model-checkpoints>
export SLIME_SAVE_ROOT=<path-for-training-outputs>
export JUDGE_API_MODE=<token|api_key|served>
export JUDGE_MODEL=<judge-model-or-deployment>
```

The launcher reads the included training data by default:

```text
openwebrl/data/webgym_filtered_popular_2102_cleaned.parquet
```

### 4. Evaluate a checkpoint

Convert a Megatron checkpoint to Hugging Face format if needed:

```bash
bash scripts/run_convert_hf.sh <path-to-iter-dir> <origin-hf-model-dir>
```

Then run evaluation:

```bash
MODEL_PATH=<path-to-hf-checkpoint> \
TASK_FILE=openwebrl/data/online-mind2web.jsonl \
bash scripts/run_evaluation.sh
```

For local-process smoke tests:

```bash
MODEL_PATH=<path-to-hf-checkpoint> \
TASK_FILE=openwebrl/data/online-mind2web.jsonl \
bash scripts/run_evaluation_local.sh
```

## 🙏 Acknowledgements

This repository builds on [slime](https://github.com/THUDM/slime),
[SGLang](https://github.com/sgl-project/sglang), Megatron-LM,
Megatron-Bridge, Playwright, and the open-source VLM/web-agent ecosystem.
We also thank Qwen for releasing the base VLMs used in our experiments, and
WebGym for providing the initial browser-task data source.

## 📚 Citation

```bibtex
@misc{yang2026openwebrldemystifyingonlinemultiturn,
      title={OpenWebRL: Demystifying Online Multi-turn Reinforcement Learning for Visual Web Agents},
      author={Rui Yang and Qianhui Wu and Yuxi Chen and Hao Bai and Wenlin Yao and Hao Cheng and Baolin Peng and Huan Zhang and Tong Zhang and Jianfeng Gao},
      year={2026},
      eprint={2606.02031},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2606.02031},
}
```
