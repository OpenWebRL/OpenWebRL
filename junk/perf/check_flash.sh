#!/usr/bin/env bash
# What attention kernels does the training image actually have?
python - <<'PY'
import importlib, torch
print("torch", torch.__version__, "cuda", torch.version.cuda)
for m in ("flash_attn", "flash_attn_interface", "flash_attn_3", "flash_attn_4", "transformer_engine", "fla", "triton"):
    try:
        mod = importlib.import_module(m); print(m, getattr(mod, "__version__", "present"))
    except Exception as e:
        print(m, "MISSING:", type(e).__name__, str(e)[:80])
try:
    from transformer_engine.pytorch.attention.dot_product_attention import utils as u
    F = u.FlashAttentionUtils
    print("TE sees FA2:", F.is_installed, F.version, "| FA3:", F.v3_is_installed, "| FA4:", getattr(F, "v4_is_installed", None))
except Exception as e:
    print("TE FA utils:", e)
PY
