"""Let TransformerEngine pick attention kernels per call, except for very wide heads.

``--attention-backend`` is process-global in Megatron. Qwen3.5's language model has 256-wide
heads, for which this image's cuDNN has no backward plan ("No valid execution plans built")
and no flash kernel, so it needs ``unfused`` -- but that setting also forces the vision
tower's 27 layers (72-wide heads) onto the unfused path, which was 10.0 s of attention time
per optimizer step against 0.4 s for the whole language model (b3ft_prof6), plus most of the
host syncs.

With ``--attention-backend auto`` TE's selector chooses per call. This patch wraps that
selector and vetoes flash/fused only when ``head_dim_qk`` exceeds
``SLIME_TE_UNFUSED_MIN_HEAD_DIM`` (set it to 129 for Qwen3.5), leaving the vision tower on
real kernels. Off unless the variable is set.
"""

import logging
import os

logger = logging.getLogger(__name__)

__all__ = ["enabled", "install"]


def enabled() -> bool:
    return bool(os.environ.get("SLIME_TE_UNFUSED_MIN_HEAD_DIM", "").strip()) or no_fused()


def no_fused() -> bool:
    """SLIME_TE_NO_FUSED=1: never let TE pick cuDNN fused attention.

    This image's cuDNN has no fused-attention backward plan for any shape used here
    (both `auto` runs died in fused_attn_bwd), yet TE ranks fused above flash, so with
    `auto` it never reaches flash. Vetoing fused at selection time lets TE fall through to
    flash-attn where installed and unfused otherwise, without touching the NVTE_* env vars
    Megatron asserts on.
    """
    return os.environ.get("SLIME_TE_NO_FUSED", "0") == "1"


def install() -> None:
    if not enabled():
        return
    raw_min = os.environ.get("SLIME_TE_UNFUSED_MIN_HEAD_DIM", "").strip()
    min_head_dim = int(raw_min) if raw_min else 10**9  # unset -> never force unfused by head size
    from transformer_engine.pytorch.attention.dot_product_attention import utils as dpa_utils

    if getattr(dpa_utils, "_slime_head_dim_veto", False):
        return
    inner = dpa_utils.get_attention_backend

    veto_fused = no_fused()
    if veto_fused:
        # TE re-reads NVTE_FUSED_ATTN on every backend selection, and its "prefer fused over
        # flash on Hopper+" step only runs when fused is still a candidate. Megatron's
        # language_module asserts this variable is unset or 1 for --attention-backend auto
        # at model build; install() runs after the build, so setting 0 here passes that
        # check and still removes fused before TE ranks the candidates.
        os.environ["NVTE_FUSED_ATTN"] = "0"

    def get_attention_backend(attention_params, *args, **kwargs):
        result = inner(attention_params, *args, **kwargs)
        head_dim = getattr(attention_params, "head_dim_qk", 0) or 0
        if veto_fused and result[2]:
            result = list(result)
            result[2] = False
            if not result[0]:
                result[4] = True
            logger.warning("TE attention: head_dim %d, fused vetoed -> %s", head_dim, "flash" if result[0] else "unfused")
            result = tuple(result)
        if head_dim < min_head_dim:
            return result
        # Actual layout (see DotProductAttention.forward's unpack, not the docstring):
        # (use_flash_attention, flash_attention_backend, use_fused_attention,
        #  fused_attention_backend, use_unfused_attention, available_backends)
        result = list(result)
        if result[0] or result[2]:
            logger.warning(
                "TE attention: head_dim %d >= %d, vetoing flash/fused (was flash=%s fused=%s) -> unfused",
                head_dim, min_head_dim, result[0], result[2],
            )
        result[0] = False
        result[2] = False
        result[4] = True
        return tuple(result)

    dpa_utils.get_attention_backend = get_attention_backend
    dpa_utils._slime_head_dim_veto = True
    logger.warning("TE attention backend selector patched: no_fused=%s, unfused for head_dim >= %s", veto_fused, min_head_dim if min_head_dim < 10**9 else "off")
