"""Worker-process side of SLIME_ENCODE_EXECUTOR=process (see openwebrl/generate_browser.py _encode_async).

Kept import-light on purpose: the spawned workers import this module, not openwebrl (whose __init__ pulls in the whole rollout stack).
"""

import base64
import io
import os

import torch
from PIL import Image

from slime.utils.processing_utils import build_processor_kwargs, load_processor

_PROCESSOR = None
_PIXEL_DTYPE = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": None}[
    os.environ.get("SLIME_MM_PIXEL_DTYPE", "bf16").lower()
]


def init(hf_checkpoint: str) -> None:
    global _PROCESSOR
    torch.set_num_threads(1)
    _PROCESSOR = load_processor(hf_checkpoint, trust_remote_code=True)


def encode(text: str, image_data: list[str]) -> tuple:
    """Same contract as generate_browser._encode_with_processor, minus the PIL images."""
    pil_images = []
    for data_url in image_data:
        b64_str = data_url.split(",", 1)[1] if "," in data_url else data_url
        pil_images.append(Image.open(io.BytesIO(base64.b64decode(b64_str))).convert("RGB"))
    out = _PROCESSOR(text=[text], **build_processor_kwargs({"images": pil_images}))
    token_ids = out["input_ids"][0]
    image_grid_thw = out["image_grid_thw"].tolist()
    mm = {k: v for k, v in out.items() if k not in ["input_ids", "attention_mask"] and isinstance(v, torch.Tensor)} or None
    if mm and _PIXEL_DTYPE is not None:
        mm = {k: (v.to(_PIXEL_DTYPE) if v.is_floating_point() else v) for k, v in mm.items()}
    return token_ids, image_grid_thw, [], mm
