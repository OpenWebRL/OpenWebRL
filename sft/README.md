# 🔥 OpenWebRL SFT Warm Start

This directory contains the supervised fine-tuning workflow used to train the
OpenWebRL SFT checkpoint before online RL.

The intended pipeline is:

```text
OpenWebRL SFT trajectories
  -> canonical OpenAI-format episodes (model-agnostic)
  -> LLaMAFactory SFT data for a chosen base model (per-model tool format)
  -> LLaMAFactory SFT checkpoint
  -> OpenWebRL online RL initialization
```

The workflow defaults to the released experiment data:

```text
OpenWebRL/OpenWebRL-SFT-Trajectories
OpenWebRL_SFT_trajectories.jsonl
```

The released trajectories pre-render one model's tool-calling format into text,
which couples the data to a single base model. To decouple it, data preparation
is split into two stages: **Stage 1** converts the trajectories once into a
model-agnostic canonical OpenAI message format (`tools` + structured
`tool_calls` + `tool` role + screenshots as base64 `image_url`); **Stage 2**
renders that canonical data into LLaMAFactory SFT data for the chosen base model
using **that model's official chat template** (`tokenizer.apply_chat_template`,
tools included) — the exact call the OpenWebRL runtime makes at inference — so
the SFT data is byte-consistent with the model's inference format. Switching base
model is just a different Stage 2 `--model-name-or-path` plus the matching
LLaMAFactory `template:`.

## 📁 Files

```text
sft/
├── run_sft_with_llamafactory.sh        # One-command wrapper: download → Stage 1 + Stage 2 → SFT → post-process. MODEL_FAMILY picks qwen3_vl (default) / qwen3_5.
├── convert_to_openai_messages.py       # Stage 1: trajectory JSONL → model-agnostic canonical OpenAI episodes (one per episode, screenshots inline as base64). No filtering/resampling.
├── prepare_openai_for_llamafactory.py  # Stage 2: canonical → LLaMAFactory ShareGPT data via the model's official chat template (--model-name-or-path), per-turn/trajectory (--per-turn). Writes PNGs + dataset_info.json.
├── post_process_llamafactory_ckpt.py   # Restore base-model config/tokenizer onto the trained checkpoint for serving/RL (and fix MoE expert layout when needed).
└── configs/
    └── qwen3_5_full_sft.example.yaml   # Reference Qwen3.5 train config (the wrapper generates a concrete one per run).
```

## 🚀 Quick Start

First make sure the LLaMAFactory uv environment has already been created under
`LLAMAFACTORY_ROOT`. A quick check is:

```bash
cd /root/LlamaFactory
uv run python -c "import llamafactory; print('ok')"
```

`MODEL_FAMILY` selects a preset. Both families share Stage 1 (the same
model-agnostic canonical episodes); the preset only changes the base model, chat
template, and checkpoint naming — Stage 2 then renders each model's own native
tool format from its official chat template. Run from the OpenWebRL root.

**Qwen3-VL-4B** — `qwen3_vl` (default), Qwen3-VL/Hermes JSON tool calls:

```bash
LLAMAFACTORY_ROOT=/root/LlamaFactory \
MODEL_NAME_OR_PATH=Qwen/Qwen3-VL-4B-Thinking \
bash sft/run_sft_with_llamafactory.sh
```

**Qwen3.5-9B** — `qwen3_5`, Qwen3.5 XML `<function=...>` tool calls:

```bash
MODEL_FAMILY=qwen3_5 \
LLAMAFACTORY_ROOT=/root/LlamaFactory \
MODEL_NAME_OR_PATH=Qwen/Qwen3.5-9B \
bash sft/run_sft_with_llamafactory.sh
```

The two presets differ only in these defaults (each still overridable):

| Default | `qwen3_vl` | `qwen3_5` |
| --- | --- | --- |
| `MODEL_NAME_OR_PATH` | `Qwen/Qwen3-VL-4B-Thinking` | `Qwen/Qwen3.5-9B` |
| `TEMPLATE` | `qwen3_vl` | `qwen3_5` |
| tool format (from chat template) | Hermes JSON, tools-end | XML `<function=…>`, tools-first |
| checkpoint dir | `qwen3-vl-4b-thinking-openwebrl-sft` | `qwen3.5-9b-openwebrl-sft` |
| config name | `openwebrl_qwen3_vl_sft.yaml` | `openwebrl_qwen3_5_sft.yaml` |

The trained checkpoint is written to
`outputs/sft/llamafactory/checkpoints/<base-model>-openwebrl-sft/` by default — the
dir name is derived from the base model, e.g. `Qwen/Qwen3.5-27B` →
`qwen3.5-27b-openwebrl-sft`. The train config YAML is saved inside that dir. Pass
`OUTPUT_DIR=/your/path` to override — but give it a real path, not a literal
placeholder.

The wrapper runs dataset download, data preparation, LLaMAFactory training, and
checkpoint post-processing through `uv run` from `LLAMAFACTORY_ROOT`.
Post-processing runs by default (it makes the checkpoint ready for serving and
for OpenWebRL online RL initialization); set `RUN_POST_PROCESS=0` to skip it.

By default, the script downloads `OpenWebRL_SFT_trajectories.jsonl` from the
Hugging Face dataset repo. To use an already-downloaded copy:

```bash
OPENWEBRL_SFT_RAW_DATA=/path/to/OpenWebRL_SFT_trajectories.jsonl \
bash sft/run_sft_with_llamafactory.sh
```

For a cheap data-preparation smoke test without training:

```bash
RUN_TRAIN=0 MAX_ROWS=10 bash sft/run_sft_with_llamafactory.sh
```

The script writes generated data under `outputs/sft/llamafactory/data/`, and the
trained checkpoint — with its train config YAML saved **inside** it — under
`outputs/sft/llamafactory/checkpoints/<slug>/`. This workspace
(`outputs/sft/llamafactory/`) is ignored by git.

## ⚙️ Important Defaults

The default training config follows the settings used for the OpenWebRL SFT
warm start:

- `stage: sft`
- `finetuning_type: full`
- `template: qwen3_vl`
- `cutoff_len: 36864`
- `mask_history: true` (set automatically: `true` for `PER_TURN=1`, `false` for `PER_TURN=0`)
- `freeze_vision_tower: true`
- `freeze_multi_modal_projector: true`
- `freeze_language_model: false`

For the default per-turn granularity, each row contains the full browser
conversation context up to the current turn, so `mask_history: true` makes
LLaMAFactory train on the current assistant turn instead of re-supervising
earlier assistant turns. For `PER_TURN=0` the wrapper sets `mask_history: false`
so all assistant turns in a full-episode record are supervised in one pass.

## 🔁 Turn-level vs Trajectory-level (`PER_TURN`)

Stage 2 renders either granularity from the same canonical episodes:

- `PER_TURN=1` (default): each episode is expanded into per-turn examples. Each
  example carries the full prior context but only the **current** screenshot
  (historical screenshots stripped), and trains with `mask_history: true` so the
  loss falls only on the current turn — the history assistant responses are
  masked. This reproduces the released single-screenshot recipe.
- `PER_TURN=0`: one full-episode record per trajectory with **all** screenshots,
  trained with `mask_history: false` to supervise every assistant turn in a
  single pass (the model sees the full screenshot history). More efficient, but
  a different visual-context recipe than the released checkpoint.

A reference config is `configs/qwen3_5_full_sft.example.yaml`. The vision-tower
freezing options (`freeze_vision_tower`, `freeze_multi_modal_projector`,
`freeze_language_model`) and `image_max_pixels` still apply because the model
consumes screenshots through the `qwen3_5` template's vision plugin. After
training, post-process the checkpoint the same way, pointing
`--original-model-path` at the Qwen3.5 base model:

```bash
MODEL_FAMILY=qwen3_5 RUN_DATA_PREPARE=0 RUN_TRAIN=0 RUN_POST_PROCESS=1 \
bash sft/run_sft_with_llamafactory.sh
```

## 🔧 Common Environment Overrides

| Variable | Default | Meaning |
| --- | --- | --- |
| `LLAMAFACTORY_ROOT` | `/root/LlamaFactory` | Local LLaMAFactory checkout. |
| `UV` | `uv` | uv executable used to run commands inside the LLaMAFactory environment. |
| `HF_DATASET_REPO` | `OpenWebRL/OpenWebRL-SFT-Trajectories` | Hugging Face dataset repo. |
| `HF_DATASET_FILE` | `OpenWebRL_SFT_trajectories.jsonl` | Dataset file to download. |
| `OPENWEBRL_SFT_RAW_DATA` | unset | Local raw JSONL override. |
| `OPENWEBRL_SFT_WORK_DIR` | `outputs/sft/llamafactory` | Generated data/config/checkpoint workspace. |
| `MODEL_FAMILY` | `qwen3_vl` | Model-family preset: `qwen3_vl` or `qwen3_5`. Sets the model, template, and checkpoint-naming defaults below. |
| `MODEL_NAME_OR_PATH` | family default (`Qwen/Qwen3-VL-4B-Thinking`; `qwen3_5` → `Qwen/Qwen3.5-9B`) | Base model or local model path for SFT. |
| `TEMPLATE` | family default (`qwen3_vl`; `qwen3_5` → `qwen3_5`) | LLaMAFactory chat template. |
| `PER_TURN` | `1` | `1` = turn-level examples + `mask_history: true`; `0` = full-episode trajectories + `mask_history: false`. |
| `OPENWEBRL_SFT_CANONICAL_DATA` | `<work>/data/openwebrl_sft_openai.jsonl` | Stage 1 canonical output (shared across models; cached). |
| `OUTPUT_DIR` | `outputs/sft/llamafactory/checkpoints/<base-model>-openwebrl-sft` (derived from `MODEL_NAME_OR_PATH`) | LLaMAFactory output checkpoint dir (the train config YAML is saved inside it). |
| `REPORT_TO` | `wandb` | LLaMAFactory reporting backend. Defaults to `wandb` and prompts for `WANDB_API_KEY` if it is unset. Set `REPORT_TO=none` to disable. |
| `RUN_TRAIN` | `1` | Set to `0` to only prepare data and configs. |
| `RUN_POST_PROCESS` | `1` | Post-process the output checkpoint after training (restores base-model config/tokenizer for serving/RL). Set to `0` to skip. |
| `FIX_MOE_EXPERTS` | `0` | Set to `1` for MoE checkpoints that need expert tensor transposition. |

Distributed training options such as `FORCE_TORCHRUN`, `NNODES`,
`NODE_RANK`, `MASTER_ADDR`, `MASTER_PORT`, and `CUDA_VISIBLE_DEVICES` are passed
through to LLaMAFactory if set in the environment.

## 📦 Output Data Format

**Stage 1** (`convert_to_openai_messages.py`) writes one canonical, model-agnostic
record per episode:

```json
{
  "tools": [{"type": "function", "function": {"name": "click", ...}}, ...],
  "messages": [
    {"role": "system", "content": "<policy>"},
    {"role": "user", "content": [{"type": "text", "text": "..."},
                                 {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}]},
    {"role": "assistant", "content": "<think>...</think>",
     "tool_calls": [{"id": "...", "type": "function",
                     "function": {"name": "click", "arguments": {"point_2d": [374, 100]}}}]},
    {"role": "tool", "tool_call_id": "...", "name": "click", "content": [ ... ]}
  ],
  "metadata": {"task_id": ..., "rollout_idx": ..., "n_turns": ...}
}
```

**Stage 2** (`prepare_openai_for_llamafactory.py`) renders that into LLaMAFactory
ShareGPT multimodal JSONL for the chosen model by calling the model's official
`apply_chat_template` (tools included) and parsing the result back into ShareGPT
turns — string content with `<image>` tokens, the model's native tool format in
the assistant turns, and `<tool_response>` user turns — plus one PNG per kept
screenshot and a matching `dataset_info.json`:

```json
{
  "messages": [
    {"role": "system", "content": "...# Tools..."},
    {"role": "user", "content": "... <image> ..."},
    {"role": "assistant", "content": "<think>...</think>\n\n<tool_call>...</tool_call>"},
    {"role": "user", "content": "<tool_response>\n...<image>...\n</tool_response>"}
  ],
  "images": ["images/0000000.png"]
}
```

```yaml
formatting: sharegpt
columns: { messages: messages, images: images }
tags: { role_tag: role, content_tag: content, user_tag: user, assistant_tag: assistant, system_tag: system }
```

> Stage 2 renders via the model's official `apply_chat_template` (and reparses)
> rather than passing structured `tool_calls` to LLaMAFactory's `openai`
> formatting, for two reasons: LLaMAFactory's OpenAI converter discards the
> assistant `<think>` reasoning on tool-call turns and the Arrow loader null-fills
> the heterogeneous tool arguments; and LLaMAFactory's own templates diverge from
> the official format (e.g. `qwen3_5` places the tool block after the system
> prompt and merges `<tool_response>` blocks). Using the official template makes
> the SFT data byte-consistent with inference and preserves the reasoning;
> LLaMAFactory then only re-wraps turns, masks loss, and expands image tokens.

The released `OpenWebRL_SFT_trajectories.jsonl` file is already the curated
experiment data. Neither stage filters by reward, captcha text, task id, or
rollout metadata.

> **Serving / online-RL note.** Because Stage 2 uses the same `apply_chat_template`
> the runtime uses, the SFT *prompt* format already matches inference. The
> remaining piece is the response **parser**: `openwebrl/base/utils.py`
> (`ToolParser`) parses the Qwen3-VL/Hermes JSON format (sglang `qwen25` parser
> plus a JSON-call regex fallback). For a model whose native format differs (e.g.
> `qwen3_5` XML `<function=…>` calls), align the runtime parser and system
> prompt before serving.

## 🧩 Checkpoint Post-Processing

Post-processing is the compatibility step after SFT. LLaMAFactory writes the
trained model weights correctly, but the config, tokenizer, processor, and chat
template files saved beside those weights may differ from the original
Hugging Face model files. Those differences can make the checkpoint harder to
load in SGLang, vLLM, or the OpenWebRL online RL pipeline, especially for
multimodal Qwen checkpoints where the processor and chat template must match the
base model.

The post-processing script restores the non-weight files from the original base
model while preserving the trained SFT weights. It also backs up the
LLaMAFactory-saved files under `llamafactory_saved/`, so the step is reversible
at the file level. You should run it when:

- you plan to initialize OpenWebRL online RL from the SFT checkpoint;
- you plan to serve the checkpoint with SGLang, vLLM, or another inference
  engine outside LLaMAFactory;
- the checkpoint fails to load because of tokenizer, processor, chat template,
  or config mismatches;
- you trained an MoE checkpoint whose expert tensors need the HF-compatible
  layout fix.

Post-processing runs by default at the end of a wrapper run
(`RUN_POST_PROCESS=1`); pass `RUN_POST_PROCESS=0` to skip it.

To post-process an already-trained checkpoint without re-running data
preparation or training:

```bash
RUN_DATA_PREPARE=0 RUN_TRAIN=0 RUN_POST_PROCESS=1 \
bash sft/run_sft_with_llamafactory.sh
```

You can also call the script directly:

```bash
cd /root/LlamaFactory
uv run python /root/OpenWebRL/sft/post_process_llamafactory_ckpt.py \
  --ckpt-path /path/to/openwebrl_sft_ckpt \
  --original-model-path Qwen/Qwen3-VL-4B-Thinking
```

Add `--include-substeps` if you want to post-process intermediate checkpoint
directories such as `checkpoint-25`, `checkpoint-50`, and `checkpoint-75` in
addition to the final output directory.

For MoE checkpoints that need expert tensor layout fixes:

```bash
cd /root/LlamaFactory
uv run python /root/OpenWebRL/sft/post_process_llamafactory_ckpt.py \
  --ckpt-path /path/to/openwebrl_sft_ckpt \
  --original-model-path Qwen/Qwen3-VL-30B-A3B-Thinking \
  --fix-moe-experts
```
