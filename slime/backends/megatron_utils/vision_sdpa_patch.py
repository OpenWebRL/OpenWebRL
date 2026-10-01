"""Run the vision tower's attention through PyTorch SDPA instead of TE's unfused kernel.

Qwen3.5-VL's vision tower packs every image of a micro-batch into one thd sequence
(``cu_seqlens`` marks image boundaries) and attends bidirectionally within each image. With
``--attention-backend unfused`` -- the only TE backend this image can run backward for --
TE unpacks that to padded bshd per image in Python (``ConvertBSHDtoTHD`` and friends), which
was 10.0 s of attention time per optimizer step against 0.4 s for the whole language model
(b3ft_prof6). This routes those calls, and only those (thd layout, head_dim <= 128), to
``torch.nn.functional.scaled_dot_product_attention`` one image at a time. Exact attention,
no dropout, so it is the same computation on PyTorch's own kernels; the tower stays trainable.

Enabled with ``SLIME_VISION_SDPA=1``. The language model's bshd calls are untouched.
"""

import logging
import math
import os

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

__all__ = ["enabled", "install", "sdpa_thd"]

_MAX_HEAD_DIM = 128
# cu_seqlens -> python offsets, keyed by the tensor's storage; every vision layer of one
# micro-batch shares the same cu_seqlens object, so the device sync happens once per
# micro-batch instead of once per image per layer.
_offsets_cache: dict[tuple[int, int], list[int]] = {}


def enabled() -> bool:
    return os.environ.get("SLIME_VISION_SDPA", "0") == "1"


def _offsets(cu_seqlens: torch.Tensor) -> list[int]:
    key = (cu_seqlens.data_ptr(), cu_seqlens.numel())
    offs = _offsets_cache.get(key)
    if offs is None:
        if len(_offsets_cache) > 64:
            _offsets_cache.clear()
        offs = cu_seqlens.tolist()
        _offsets_cache[key] = offs
    return offs


def sdpa_thd(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, cu_seqlens: torch.Tensor) -> torch.Tensor:
    """Bidirectional attention within each segment of a packed sequence.

    ``query``/``key``/``value``: ``[t, h, d]`` (or ``[t, 1, h, d]``). Returns ``[t, h * d]``.
    """
    squeeze = query.dim() == 4
    if squeeze:
        query, key, value = (x.squeeze(1) for x in (query, key, value))
    t, h, d = query.shape
    out = torch.empty((t, h, d), dtype=query.dtype, device=query.device)
    offs = _offsets(cu_seqlens)
    scale = 1.0 / math.sqrt(d)
    for start, end in zip(offs[:-1], offs[1:]):
        if end <= start:
            continue
        q = query[start:end].transpose(0, 1).unsqueeze(0)  # [1, h, n, d]
        k = key[start:end].transpose(0, 1).unsqueeze(0)
        v = value[start:end].transpose(0, 1).unsqueeze(0)
        o = F.scaled_dot_product_attention(q, k, v, is_causal=False, scale=scale)  # [1, h, n, d]
        out[start:end] = o.squeeze(0).transpose(0, 1)
    return out.reshape(t, h * d)


def install() -> None:
    if not enabled():
        return
    from megatron.core.extensions.transformer_engine import TEDotProductAttention

    if getattr(TEDotProductAttention, "_slime_vision_sdpa", False):
        return
    # TE's own cuDNN attention has no backward plan in this image; keep PyTorch off it too.
    try:
        torch.backends.cuda.enable_cudnn_sdp(False)
    except Exception:  # older torch
        pass
    inner_forward = TEDotProductAttention.forward

    def forward(self, query, key, value, attention_mask, attn_mask_type, *args, **kwargs):
        packed_seq_params = kwargs.get("packed_seq_params")
        if packed_seq_params is None and len(args) >= 2:
            packed_seq_params = args[1]
        head_dim = query.shape[-1]
        if (
            packed_seq_params is not None
            and getattr(packed_seq_params, "qkv_format", None) == "thd"
            and head_dim <= _MAX_HEAD_DIM
            and "causal" not in getattr(attn_mask_type, "name", str(attn_mask_type))
            and getattr(self, "attention_dropout", 0.0) in (0, 0.0, None)
        ):
            out = sdpa_thd(query, key, value, packed_seq_params.cu_seqlens_q)
            # TE returns [sq, b, hidden]; thd has b == 1.
            return out.unsqueeze(1) if query.dim() == 4 else out
        return inner_forward(self, query, key, value, attention_mask, attn_mask_type, *args, **kwargs)

    TEDotProductAttention.forward = forward
    TEDotProductAttention._slime_vision_sdpa = True
    logger.warning("vision-tower thd attention (head_dim <= %d) routed to torch SDPA", _MAX_HEAD_DIM)
