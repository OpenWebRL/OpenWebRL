# OpenWebRL SFT Warm Start

This directory contains the supervised fine-tuning workflow used to train the
OpenWebRL SFT checkpoint before online RL.

The intended pipeline is:

```text
OpenWebRL SFT trajectories -> LLaMAFactory SFT checkpoint -> OpenWebRL online RL initialization
```

The workflow defaults to the released experiment data:

```text
OpenWebRL/OpenWebRL-SFT-Trajectories
OpenWebRL_SFT_trajectories.jsonl
```

## Files

| File | Purpose |
| --- | --- |
| `run_sft_with_llamafactory.sh` | One-command wrapper that downloads the HF dataset, prepares LLaMAFactory data, generates config files, launches SFT, and optionally post-processes the checkpoint. |
| `prepare_llamafactory_sft_data.py` | Converts OpenWebRL trajectory JSONL into LLaMAFactory ShareGPT-style multimodal JSONL and extracts screenshots to PNG files. It does not filter or resample the released SFT trajectories. |
| `post_process_llamafactory_ckpt.py` | Makes a trained LLaMAFactory checkpoint easier to serve and reuse in OpenWebRL by restoring base-model config/tokenizer files while keeping the trained weights. It can also fix MoE expert tensor layout when needed. |

## Quick Start

First make sure the LLaMAFactory uv environment has already been created under
`LLAMAFACTORY_ROOT`. A quick check is:

```bash
cd /root/LlamaFactory
uv run python -c "import llamafactory; print('ok')"
```

Then set the LLaMAFactory checkout and model, and run from the OpenWebRL root:

```bash
LLAMAFACTORY_ROOT=/root/LlamaFactory \
MODEL_NAME_OR_PATH=Qwen/Qwen3-VL-4B-Thinking \
OUTPUT_DIR=/path/to/openwebrl_sft_ckpt \
bash sft/run_sft_with_llamafactory.sh
```

The wrapper runs dataset download, data preparation, LLaMAFactory training, and
optional checkpoint post-processing through `uv run` from `LLAMAFACTORY_ROOT`.
Set `RUN_POST_PROCESS=1` if the checkpoint will be used directly for serving or
as the initialization checkpoint for OpenWebRL online RL.

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

The script writes generated data and configs under:

```text
outputs/sft/llamafactory/
```

This path is ignored by git.

## Important Defaults

The default training config follows the settings used for the OpenWebRL SFT
warm start:

- `stage: sft`
- `finetuning_type: full`
- `template: qwen3_vl`
- `cutoff_len: 36864`
- `mask_history: true`
- `freeze_vision_tower: true`
- `freeze_multi_modal_projector: true`
- `freeze_language_model: false`

`mask_history: true` is important because each row contains the full browser
conversation context up to the current turn. With history masking, LLaMAFactory
trains on the current assistant turn instead of re-supervising earlier assistant
turns in the context.

## Common Environment Overrides

| Variable | Default | Meaning |
| --- | --- | --- |
| `LLAMAFACTORY_ROOT` | `/root/LlamaFactory` | Local LLaMAFactory checkout. |
| `UV` | `uv` | uv executable used to run commands inside the LLaMAFactory environment. |
| `HF_DATASET_REPO` | `OpenWebRL/OpenWebRL-SFT-Trajectories` | Hugging Face dataset repo. |
| `HF_DATASET_FILE` | `OpenWebRL_SFT_trajectories.jsonl` | Dataset file to download. |
| `OPENWEBRL_SFT_RAW_DATA` | unset | Local raw JSONL override. |
| `OPENWEBRL_SFT_WORK_DIR` | `outputs/sft/llamafactory` | Generated data/config/checkpoint workspace. |
| `MODEL_NAME_OR_PATH` | `Qwen/Qwen3-VL-4B-Thinking` | Base model or local model path for SFT. |
| `OUTPUT_DIR` | `outputs/sft/llamafactory/checkpoints/qwen3-vl-openwebrl-sft` | LLaMAFactory output checkpoint dir. |
| `REPORT_TO` | `none` | LLaMAFactory reporting backend, e.g. `wandb`. |
| `RUN_TRAIN` | `1` | Set to `0` to only prepare data and configs. |
| `RUN_POST_PROCESS` | `0` | Set to `1` to post-process the output checkpoint after training. |
| `FIX_MOE_EXPERTS` | `0` | Set to `1` for MoE checkpoints that need expert tensor transposition. |

Distributed training options such as `FORCE_TORCHRUN`, `NNODES`,
`NODE_RANK`, `MASTER_ADDR`, `MASTER_PORT`, and `CUDA_VISIBLE_DEVICES` are passed
through to LLaMAFactory if set in the environment.

## Output Data Format

`prepare_llamafactory_sft_data.py` creates a JSONL file where each row has:

```json
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "... <image> ..."},
    {"role": "assistant", "content": "..."}
  ],
  "images": ["images/000000_task_...png"]
}
```

It also generates one PNG screenshot file per example. The wrapper writes a
matching `dataset_info.json` so LLaMAFactory can load the data with:

```yaml
formatting: sharegpt
columns:
  messages: messages
  images: images
tags:
  role_tag: role
  content_tag: content
  user_tag: user
  assistant_tag: assistant
  system_tag: system
```

The released `OpenWebRL_SFT_trajectories.jsonl` file is already the curated
experiment data. The preparation script only changes format and image storage;
it does not filter by reward, captcha text, task id, or rollout metadata.

## Checkpoint Post-Processing

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

For the default wrapper, enable post-processing in the same run with:

```bash
RUN_POST_PROCESS=1 bash sft/run_sft_with_llamafactory.sh
```

To post-process an already-trained checkpoint without re-running data
preparation or training:

```bash
RUN_PREPARE=0 RUN_TRAIN=0 RUN_POST_PROCESS=1 \
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
