import argparse
import os
import tempfile
from contextlib import contextmanager

import megatron.bridge.training.model_load_save as _model_load_save_module
from megatron.bridge import AutoBridge
from megatron.core import parallel_state
import torch
import torch.distributed as dist


# Here we need to patch Megatron Bridge's `load_model_config`, since the checkpoint is saved
# by Megatron and lack of provider information.
_provider_override = {}
_original_load_model_config = _model_load_save_module.load_model_config


def _patched_load_model_config(checkpoint_path):
    model_cfg, mlm_args = _original_load_model_config(checkpoint_path)
    provider = _provider_override.get("provider")
    if provider is not None:
        from megatron.bridge.models.model_provider import ModelProviderMixin

        if not isinstance(model_cfg, ModelProviderMixin):
            print(f"[convert] Overriding MLM TransformerConfig with Bridge provider: " f"{type(provider).__name__}")
            return provider, mlm_args
    return model_cfg, mlm_args


_model_load_save_module.load_model_config = _patched_load_model_config


@contextmanager
def _file_store_distributed_context(backend: str = "gloo"):
    """Initialize Bridge's one-process context without opening a TCP rendezvous port."""
    if dist.is_initialized():
        yield
        return

    with tempfile.TemporaryDirectory(prefix="mbridge_dist_") as tmpdir:
        init_method = f"file://{os.path.join(tmpdir, 'init')}"
        os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
        dist.init_process_group(backend=backend, init_method=init_method, world_size=1, rank=0)
        parallel_state.initialize_model_parallel()

        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            try:
                from megatron.core.tensor_parallel import model_parallel_cuda_manual_seed

                model_parallel_cuda_manual_seed(0)
            except Exception:
                pass

        try:
            yield
        finally:
            parallel_state.destroy_model_parallel()
            dist.destroy_process_group()


_model_load_save_module.temporary_distributed_context = _file_store_distributed_context


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert torch distributed checkpoint to HuggingFace format using Megatron Bridge"
    )
    parser.add_argument(
        "--input-dir", type=str, required=True, help="Path to the torch distributed checkpoint directory"
    )
    parser.add_argument("--output-dir", type=str, required=True, help="Path to save the HuggingFace checkpoint")
    parser.add_argument(
        "--origin-hf-dir",
        type=str,
        required=True,
        help="Path to the original HuggingFace model directory (for config)",
    )
    parser.add_argument(
        "-f", "--force", action="store_true", help="Force overwrite the output directory if it exists."
    )
    args = parser.parse_args()

    if os.path.exists(args.output_dir) and not args.force:
        raise ValueError(f"Output directory {args.output_dir} already exists. Use --force to overwrite it.")

    print(f"Loading config from {args.origin_hf_dir}")
    bridge = AutoBridge.from_hf_pretrained(args.origin_hf_dir, trust_remote_code=True)

    # Use Bridge's provider so the correct model class is created (e.g., Qwen3VLModel
    # instead of GPTModel). This is needed because MLM checkpoints lack run_config.yaml.
    provider = bridge.to_megatron_provider(load_weights=False)
    _provider_override["provider"] = provider
    print(f"[convert] Using Bridge provider: {type(provider).__name__}")

    print(f"Exporting checkpoint from {args.input_dir} to {args.output_dir}")
    bridge.export_ckpt(args.input_dir, args.output_dir)

    print("Done!")
