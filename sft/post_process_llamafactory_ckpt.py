"""
Post-process a LLaMAFactory checkpoint to fix known compatibility issues
before serving with sglang, vllm, or other inference engines.

LLaMAFactory saves config files (config.json, tokenizer_config.json, etc.)
with a different transformers version, which can introduce incompatible
changes (missing fields, renamed keys, wrong types). The safest fix is to
restore the original config files from the base model while keeping the
trained weights.

Usage:
    # Auto-detect original model from checkpoint's config.json (easiest)
    python post_process_ckpt.py --ckpt-path /path/to/checkpoint

    # Specify a HF model ID (auto-resolves to cache or downloads)
    python post_process_ckpt.py --ckpt-path /path/to/checkpoint \\
        --original-model-path Qwen/Qwen3-VL-4B-Thinking

    # Specify a local path directly
    python post_process_ckpt.py --ckpt-path /path/to/checkpoint \\
        --original-model-path /path/to/original/model

    # Also fix intermediate checkpoints
    python post_process_ckpt.py --ckpt-path /path/to/checkpoint --include-substeps

    # Dry run (show what would be changed without modifying files)
    python post_process_ckpt.py --ckpt-path /path/to/checkpoint --dry-run
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sys
from collections import defaultdict

# MoE expert weight names saved by LlamaFactory with the last two dims swapped
# relative to the HF canonical layout. Transposing on dims (1, 2) fixes them.
_MOE_EXPERT_SUFFIXES = ("experts.down_proj", "experts.gate_up_proj")

# Config files to restore from original model (skip weights and training artifacts)
_CONFIG_FILES = [
    "config.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "generation_config.json",
    "preprocessor_config.json",
    "processor_config.json",
    "chat_template.json",
    "chat_template.jinja",
    "vocab.json",
    "video_preprocessor_config.json",
    "merges.txt",
    "special_tokens_map.json",
]


def resolve_model_path(model_name_or_path: str) -> str:
    """Resolve a model name or path to a local directory.

    If the path is already a local directory, return it as-is.
    Otherwise, treat it as a HF model ID and resolve via huggingface_hub
    (uses local cache if available, downloads otherwise).
    """
    if os.path.isdir(model_name_or_path):
        return model_name_or_path

    print(f"  Resolving '{model_name_or_path}' via huggingface_hub ...")
    from huggingface_hub import snapshot_download
    # local_files_only=True first to avoid unnecessary network calls
    try:
        path = snapshot_download(model_name_or_path, local_files_only=True)
        print(f"  Found in cache: {path}")
        return path
    except Exception:
        pass
    # Not in cache — download
    print(f"  Not in cache, downloading ...")
    path = snapshot_download(model_name_or_path)
    print(f"  Downloaded to: {path}")
    return path


def detect_original_model(ckpt_path: str) -> str | None:
    """Try to detect the original model name from the checkpoint's config.json."""
    config_path = os.path.join(ckpt_path, "config.json")
    if not os.path.exists(config_path):
        return None
    try:
        with open(config_path) as f:
            config = json.load(f)
        name = config.get("_name_or_path", "")
        if name and name != ckpt_path:
            return name
    except (json.JSONDecodeError, OSError):
        pass
    return None


def restore_configs_from_original(ckpt_path: str, original_path: str, dry_run: bool = False) -> bool:
    """Copy config files from the original model to the checkpoint.

    This overwrites LlamaFactory-saved configs with the originals so that
    inference engines (sglang, vllm) can load the checkpoint without errors.
    Model weights in the checkpoint are not touched.
    """
    if not os.path.isdir(original_path):
        print(f"  [error] Original model path not found: {original_path}")
        return False

    # Back up LlamaFactory-saved configs before overwriting
    backup_dir = os.path.join(ckpt_path, "llamafactory_saved")
    if not dry_run:
        os.makedirs(backup_dir, exist_ok=True)

    copied = 0
    for fname in _CONFIG_FILES:
        src = os.path.join(original_path, fname)
        dst = os.path.join(ckpt_path, fname)
        if os.path.exists(src):
            if not dry_run:
                # Back up the LlamaFactory version if it exists
                if os.path.exists(dst):
                    shutil.copy2(dst, os.path.join(backup_dir, fname))
                shutil.copy2(src, dst)
            print(f"  [restore] {fname}")
            copied += 1

    if copied > 0:
        print(f"  Backed up LlamaFactory configs to {backup_dir}/")
        print(f"  Restored {copied} config file(s) from {original_path}")
    else:
        print(f"  [warn] No config files found in {original_path}")

    return copied > 0


def _is_moe_expert_key(key: str) -> bool:
    return any(key.endswith(suffix) for suffix in _MOE_EXPERT_SUFFIXES)


def _load_original_expert_shapes(original_path: str) -> dict[str, tuple[int, ...]]:
    """Return {weight_name: shape} for every MoE expert tensor in the original model."""
    from safetensors import safe_open

    shapes: dict[str, tuple[int, ...]] = {}
    index_path = os.path.join(original_path, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path) as f:
            wm = json.load(f)["weight_map"]
        shards = {wm[k] for k in wm if _is_moe_expert_key(k)}
        for shard in shards:
            with safe_open(os.path.join(original_path, shard), framework="pt") as f:
                for k in f.keys():
                    if _is_moe_expert_key(k):
                        shapes[k] = tuple(f.get_slice(k).get_shape())
    else:
        # single-file checkpoint
        single = os.path.join(original_path, "model.safetensors")
        if os.path.exists(single):
            with safe_open(single, framework="pt") as f:
                for k in f.keys():
                    if _is_moe_expert_key(k):
                        shapes[k] = tuple(f.get_slice(k).get_shape())
    return shapes


def fix_moe_expert_weights(ckpt_path: str, original_path: str, dry_run: bool = False) -> bool:
    """Transpose LlamaFactory-saved MoE expert weights back to HF layout.

    LlamaFactory saves `experts.down_proj` and `experts.gate_up_proj` with the
    last two dims swapped relative to the original HF checkpoint, which makes
    SGLang's fused-MoE loader fail with a tensor-size mismatch. This rewrites
    each affected shard in place after transposing the offending tensors on
    dims (1, 2).
    """
    from safetensors import safe_open
    from safetensors.torch import save_file

    index_path = os.path.join(ckpt_path, "model.safetensors.index.json")
    single_path = os.path.join(ckpt_path, "model.safetensors")

    if os.path.exists(index_path):
        with open(index_path) as f:
            wm = json.load(f)["weight_map"]
        expert_to_shard: dict[str, str] = {k: wm[k] for k in wm if _is_moe_expert_key(k)}
    elif os.path.exists(single_path):
        with safe_open(single_path, framework="pt") as f:
            expert_to_shard = {k: "model.safetensors" for k in f.keys() if _is_moe_expert_key(k)}
    else:
        print("  [skip] No safetensors index or single-file checkpoint found")
        return False

    if not expert_to_shard:
        print("  [skip] No MoE expert weights found — not a MoE checkpoint")
        return False

    orig_shapes = _load_original_expert_shapes(original_path)
    if not orig_shapes:
        print(f"  [error] Could not read expert shapes from original model: {original_path}")
        return False

    # Group keys-to-fix by shard, after deciding per-shard whether fix is needed.
    shard_to_keys: dict[str, list[str]] = defaultdict(list)
    already_ok = 0
    for key, shard in expert_to_shard.items():
        shard_path = os.path.join(ckpt_path, shard)
        with safe_open(shard_path, framework="pt") as f:
            cur_shape = tuple(f.get_slice(key).get_shape())
        target = orig_shapes.get(key)
        if target is None:
            print(f"  [warn] {key}: no shape in original model, skipping")
            continue
        if cur_shape == target:
            already_ok += 1
            continue
        if len(cur_shape) == 3 and (cur_shape[0], cur_shape[2], cur_shape[1]) == target:
            shard_to_keys[shard].append(key)
        else:
            print(f"  [error] {key}: shape {cur_shape} is neither target {target} nor its transpose; aborting")
            return False

    if not shard_to_keys:
        print(f"  [ok] All {already_ok} MoE expert tensor(s) already match HF layout — nothing to do")
        return True

    total_fix = sum(len(v) for v in shard_to_keys.values())
    print(f"  Need to transpose {total_fix} expert tensor(s) across {len(shard_to_keys)} shard(s) "
          f"({already_ok} already correct)")

    for shard, keys_to_fix in sorted(shard_to_keys.items()):
        shard_path = os.path.join(ckpt_path, shard)
        fix_set = set(keys_to_fix)
        print(f"  [shard] {shard}: fixing {len(keys_to_fix)} tensor(s)")
        if dry_run:
            continue

        # Load all tensors from this shard, transpose the targeted ones, write back atomically.
        with safe_open(shard_path, framework="pt") as f:
            metadata = f.metadata() or {}
            tensors: dict = {}
            for k in f.keys():
                t = f.get_tensor(k)
                if k in fix_set:
                    t = t.transpose(1, 2).contiguous()
                tensors[k] = t

        tmp_path = shard_path + ".tmp"
        save_file(tensors, tmp_path, metadata=metadata)
        os.replace(tmp_path, shard_path)
        # Free the tensor dict before processing the next (possibly large) shard.
        del tensors

    print(f"  [done] Transposed {total_fix} MoE expert tensor(s)")
    return True


def process_checkpoint(
    ckpt_path: str,
    original_path: str | None = None,
    dry_run: bool = False,
    fix_moe: bool = False,
) -> bool:
    """Process a single checkpoint directory."""
    print(f"\nProcessing: {ckpt_path}")

    if not os.path.isdir(ckpt_path):
        print(f"  [error] Not a directory: {ckpt_path}")
        return False

    # Auto-detect original model if not specified
    if not original_path:
        detected = detect_original_model(ckpt_path)
        if detected:
            print(f"  Auto-detected original model: {detected}")
            original_path = detected
        else:
            print("  [skip] Could not detect original model from config.json")
            return False

    # Resolve HF model ID to local path
    original_path = resolve_model_path(original_path)
    ok = restore_configs_from_original(ckpt_path, original_path, dry_run)
    if fix_moe:
        ok = fix_moe_expert_weights(ckpt_path, original_path, dry_run) and ok
    return ok


def main():
    parser = argparse.ArgumentParser(
        description="Post-process LLaMAFactory checkpoints for serving compatibility"
    )
    parser.add_argument(
        "--ckpt-path", required=True,
        help="Path to the checkpoint directory (e.g. /data/.../qwen3-vl-4b-thinking)",
    )
    parser.add_argument(
        "--original-model-path", default=None,
        help="HF model ID (e.g. Qwen/Qwen3-VL-4B-Thinking) or local path. "
             "If omitted, auto-detected from checkpoint's config.json '_name_or_path' field.",
    )
    parser.add_argument(
        "--include-substeps", action="store_true",
        help="Also process intermediate checkpoint-* subdirectories",
    )
    parser.add_argument(
        "--fix-moe-experts", action="store_true",
        help="Also transpose MoE expert weights (experts.down_proj, experts.gate_up_proj) "
             "on dims (1, 2) back to HF layout. Required for Qwen3-VL MoE checkpoints "
             "saved by LlamaFactory.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be changed without modifying files",
    )
    args = parser.parse_args()

    if args.dry_run:
        print("=== DRY RUN (no files will be modified) ===")

    # Process main checkpoint
    total_fixed = 0
    if process_checkpoint(args.ckpt_path, args.original_model_path, args.dry_run, args.fix_moe_experts):
        total_fixed += 1

    # Process intermediate checkpoints
    if args.include_substeps:
        pattern = os.path.join(args.ckpt_path, "checkpoint-*")
        for sub_ckpt in sorted(glob.glob(pattern)):
            if os.path.isdir(sub_ckpt):
                if process_checkpoint(sub_ckpt, args.original_model_path, args.dry_run, args.fix_moe_experts):
                    total_fixed += 1

    print(f"\nDone. Fixed {total_fixed} checkpoint(s).")


if __name__ == "__main__":
    main()
