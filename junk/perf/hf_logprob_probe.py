"""Reference log-probs from HF transformers for saved rollout samples.

For a few samples of a saved rollout batch, run the HF Qwen3.5 model on the exact tokens and
pixel_values the trainer sees and compare, token by token, against the log-probs SGLang
returned at rollout time. Prints where HF sides with SGLang (=> trainer path wrong) or with
the trainer (=> SGLang path wrong), with digits/coordinates called out.
"""
import sys, torch, math
from collections import Counter
bank = sys.argv[1] if len(sys.argv) > 1 else "/weka/oe-training-default/new_peters/OpenWebRL/outputs/rollout_bank/b3ft/rollout_0.pt"
ckpt = "/weka/oe-training-default/new_peters/LlamaFactory/saves/baseline_3_ft/full/sft/checkpoint-1000"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 6
TEMP = 0.7

print("loading bank ...", flush=True)
data = torch.load(bank, weights_only=False)
samples = data["samples"] if isinstance(data, dict) else data
print("num samples", len(samples), "keys:", sorted(samples[0].keys()) if isinstance(samples[0], dict) else type(samples[0]), flush=True)

from transformers import AutoTokenizer, AutoModelForImageTextToText, AutoProcessor
tok = AutoTokenizer.from_pretrained(ckpt, trust_remote_code=True)
model = AutoModelForImageTextToText.from_pretrained(ckpt, torch_dtype=torch.bfloat16, trust_remote_code=True).cuda().eval()
print("model loaded", type(model).__name__, flush=True)

def g(s, *ks):
    for k in ks:
        if isinstance(s, dict) and k in s and s[k] is not None: return s[k]
    md = s.get("metadata") or {}
    for k in ks:
        if k in md: return md[k]
    return None

picked = [s for s in samples if g(s, "rollout_log_probs") and g(s, "multimodal_train_inputs")][:N]
agg_all, agg_digit = [], []
for i, s in enumerate(picked):
    tokens = torch.tensor(g(s, "tokens"), dtype=torch.long)
    rl = int(g(s, "response_length"))
    rlp = torch.tensor(g(s, "rollout_log_probs"), dtype=torch.float32)
    mm = g(s, "multimodal_train_inputs") or {}
    kwargs = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in mm.items()}
    if "pixel_values" in kwargs and kwargs["pixel_values"].is_floating_point():
        kwargs["pixel_values"] = kwargs["pixel_values"].to(torch.bfloat16)
    if "image_grid_thw" in kwargs:
        # newer transformers derives M-RoPE positions from this instead of scanning for image tokens
        kwargs["mm_token_type_ids"] = (tokens == model.config.image_token_id).long()[None].cuda()
    with torch.no_grad():
        out = model(input_ids=tokens[None].cuda(), **kwargs)
    logits = out.logits[0].float()                     # [T, V]
    T = tokens.numel()
    # position p predicts token p+1; response tokens are the last rl tokens
    resp_pos = torch.arange(T - rl - 1, T - 1)
    lp = torch.log_softmax(logits[resp_pos] / TEMP, dim=-1)
    hf = lp.gather(1, tokens[-rl:].cuda()[:, None]).squeeze(1).cpu()
    lp1 = torch.log_softmax(logits[resp_pos], dim=-1).gather(1, tokens[-rl:].cuda()[:, None]).squeeze(1).cpu()
    d_t = hf - rlp[:rl]          # HF(temp 0.7) - SGLang
    d_1 = lp1 - rlp[:rl]         # HF(temp 1.0) - SGLang
    resp = tokens[-rl:].tolist()
    strs = [tok.decode([t]) for t in resp]
    is_digit = torch.tensor([st.strip().isdigit() for st in strs])
    agg_all.append(d_t); agg_digit.append(d_t[is_digit])
    print(f"\nsample {i}: resp_len {rl} | HF@0.7-SGL mean {d_t.mean():.3f} median {d_t.median():.3f} | HF@1.0-SGL mean {d_1.mean():.3f} median {d_1.median():.3f} | "
          f"digits: n={int(is_digit.sum())} mean diff {d_t[is_digit].mean() if is_digit.any() else float('nan'):.3f} | frac |diff|>1 {(d_t.abs()>1).float().mean():.3f}")
    worst = torch.argsort(d_t)[:5].tolist()
    print("  worst:", " || ".join(f"pos {j} {strs[j]!r} sgl {rlp[j]:.2f} hf {hf[j]:.2f}" for j in worst))
    print("  head:", repr(tok.decode(resp[:30]))[:150])
A = torch.cat(agg_all); D = torch.cat([x for x in agg_digit if x.numel()])
print(f"\nOVERALL HF(temp0.7) - SGLang: mean {A.mean():.3f} median {A.median():.3f} frac|>1| {(A.abs()>1).float().mean():.3f} | digits only: n={D.numel()} mean {D.mean():.3f} median {D.median():.3f}")
print("Reading: trainer showed mean ~-1.1 with digit tokens at -20. If HF matches SGLang here (digits ~0), the Megatron-bridge trainer is wrong on vision; if HF also puts digits at -20, SGLang's rollout log-probs are wrong.")
