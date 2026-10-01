"""Fix SGLang's Qwen3.5 attention for multimodal (mrope) positions.

On CUDA, Qwen3_5AttentionDecoderLayer.self_attention uses fused_qk_gemma_rmsnorm_rope_gate,
whose Triton kernel indexes `positions_ptr + token`: it assumes a 1-D [T] position vector.
With images, positions is the mrope (3, T) tensor, so the kernel reads the first T values,
the temporal row, for every token. Text tokens have t == h == w and are unaffected; every
image patch of a frame shares one temporal position, so all image tokens get the same rotary
angle and lose their (h, w) components. Symptom (measured 2026-09-17): text log-probs match HF
to 0.02 nats, click-coordinate digits differ by 10-35 nats.

This routes 2-D positions through forward_prepare_native, whose MRotaryEmbedding.forward
applies the interleaved mrope sections correctly. Run once per container before launching
SGLang (idempotent). Usage: python beaker/patch_sglang_qwen35_mrope.py
"""
import os, sys
import sglang

path = os.path.join(os.path.dirname(sglang.__file__), "srt", "models", "qwen3_5.py")
src = open(path).read()
old = "        if _is_cuda and self.attn_output_gate:\n            q, k, v, gate = self.forward_prepare_cuda_fused("
new = ("        # mrope (3, T) positions: the fused Triton kernel reads positions as 1-D and would\n"
       "        # apply only the temporal row; use the native path (MRotaryEmbedding handles 2-D).\n"
       "        if _is_cuda and self.attn_output_gate and positions.ndim == 1:\n"
       "            q, k, v, gate = self.forward_prepare_cuda_fused(")
if new in src:
    print(f"already patched: {path}")
    sys.exit(0)
if src.count(old) != 1:
    print(f"ERROR: anchor found {src.count(old)} times in {path} (sglang {sglang.__version__}); not patching")
    sys.exit(1)
open(path, "w").write(src.replace(old, new, 1))
print(f"patched {path} (sglang {sglang.__version__})")
