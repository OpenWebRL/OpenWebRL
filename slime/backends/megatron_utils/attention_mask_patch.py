"""Stop handing TransformerEngine a padding mask that carries no information.

In bshd mode megatron-bridge's VL model replaces a missing attention mask with
``torch.ones_like(input_ids)``. slime always passes ``attention_mask=None``, so every
attention layer receives an all-ones mask. TE treats any present mask as a *padding*
mask: it derives per-sequence lengths from it (``get_cu_seqlens`` / ``get_indices``) on
every layer of every micro-batch, and the unfused backend wraps the kernel in
``ConvertBSHDtoTHD`` / ``ConvertTHDtoBSHD``. In the TE shipped in the training image those
are per-sequence Python loops with ``.item()``: ~64k GPU->CPU syncs per optimizer step,
~70% of its wall time (see junk/perf/analyze_syncs.py on b3ft_prof4).

With the mask dropped TE runs plain causal attention. That is numerically identical for
the loss because slime right-pads every sample and ``get_responses`` never reads a pad
position; under causal attention a real token cannot see the later pad tokens. It is NOT
valid for left padding or for a loss that touches pad positions.

Enabled with ``SLIME_BSHD_DROP_PADDING_MASK=1``. Only affects the non-packed (bshd/sbhd)
path; thd (``packed_seq_params`` given) is left alone.
"""

import logging
import os

logger = logging.getLogger(__name__)

__all__ = ["enabled", "install"]


def enabled() -> bool:
    return os.environ.get("SLIME_BSHD_DROP_PADDING_MASK", "0") == "1"


def install() -> None:
    if not enabled():
        return
    from megatron.core.extensions.transformer_engine import TEDotProductAttention
    from megatron.core.transformer.enums import AttnMaskType

    if getattr(TEDotProductAttention, "_slime_drop_padding_mask", False):
        return
    inner_forward = TEDotProductAttention.forward

    def forward(self, query, key, value, attention_mask, attn_mask_type, *args, **kwargs):
        packed_seq_params = kwargs.get("packed_seq_params")
        if packed_seq_params is None and len(args) >= 2:
            packed_seq_params = args[1]  # (attention_bias, packed_seq_params, ...) positional
        if attention_mask is not None and packed_seq_params is None:
            attention_mask = None
            if attn_mask_type == AttnMaskType.padding_causal:
                attn_mask_type = AttnMaskType.causal
            elif attn_mask_type == AttnMaskType.padding:
                attn_mask_type = AttnMaskType.no_mask
        return inner_forward(self, query, key, value, attention_mask, attn_mask_type, *args, **kwargs)

    TEDotProductAttention.forward = forward
    TEDotProductAttention._slime_drop_padding_mask = True
    logger.warning("bshd padding mask dropped before TransformerEngine attention (causal only)")
