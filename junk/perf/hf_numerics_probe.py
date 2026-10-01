"""How fragile are the coordinate-digit log-probs to numerics alone? Same HF model, same inputs,
three numeric regimes (bf16 sdpa, bf16 eager, fp32). Pairwise digit outliers (>3 nats)."""
import os, gc, torch
bank = "/weka/oe-training-default/new_peters/OpenWebRL/outputs/rollout_bank/b3ft/rollout_0.pt"
ckpt = "/weka/oe-training-default/new_peters/LlamaFactory/saves/baseline_3_ft/full/sft/checkpoint-1000"
N = int(os.environ.get("N", "40"))
data = torch.load(bank, weights_only=False)
samples = data["samples"] if isinstance(data, dict) else data
g = lambda s, k: s.get(k) if isinstance(s, dict) and s.get(k) is not None else (s.get("metadata") or {}).get(k)
ok = [s for s in samples if g(s, "rollout_log_probs") and g(s, "multimodal_train_inputs")]
stride = max(1, len(ok) // N)
picked = [{"tokens": list(g(s, "tokens"))[:-2], "rl": int(g(s, "response_length")) - 2,
           "rec": [float(x) for x in g(s, "rollout_log_probs")], "mm": dict(g(s, "multimodal_train_inputs"))} for s in ok[::stride][:N]]
del data, samples, ok; gc.collect()
from transformers import AutoTokenizer, AutoModelForImageTextToText
tok = AutoTokenizer.from_pretrained(ckpt)
regimes = [("bf16_sdpa", torch.bfloat16, "sdpa"), ("bf16_eager", torch.bfloat16, "eager"), ("fp32_sdpa", torch.float32, "sdpa")]
lps = {r[0]: [] for r in regimes}
for name, dt, attn in regimes:
    model = AutoModelForImageTextToText.from_pretrained(ckpt, torch_dtype=dt, attn_implementation=attn).cuda().eval()
    for s in picked:
        tokens = torch.tensor(s["tokens"]); rl = s["rl"]; T = tokens.numel()
        kw = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in s["mm"].items()}
        kw["pixel_values"] = kw["pixel_values"].to(dt)
        kw["mm_token_type_ids"] = (tokens == model.config.image_token_id).long()[None].cuda()
        with torch.no_grad():
            logits = model(input_ids=tokens[None].cuda(), **kw).logits[0].float()
        pos = torch.arange(T - rl - 1, T - 1)
        lps[name].append(torch.log_softmax(logits[pos], -1).gather(1, tokens[-rl:].cuda()[:, None]).squeeze(1).cpu())
    del model; gc.collect(); torch.cuda.empty_cache()
    print(f"{name} done", flush=True)
isd_all, rec_all = [], []
for s in picked:
    resp = s["tokens"][-s["rl"]:]
    isd_all.append(torch.tensor([tok.decode([t]).strip().isdigit() for t in resp])); rec_all.append(torch.tensor(s["rec"][:s["rl"]]))
isd = torch.cat(isd_all); rec = torch.cat(rec_all)
L = {k: torch.cat(v) for k, v in lps.items()}
print(f"\n==== {isd.sum().item()} digit tokens of {isd.numel()} response tokens, {len(picked)} samples")
names = list(L)
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        d = L[names[i]] - L[names[j]]
        print(f"{names[i]:10s} vs {names[j]:10s}: all frac|>1| {(d.abs()>1).float().mean():.4f} | digits frac|>3| {(d[isd].abs()>3).float().mean():.4f} max {d[isd].abs().max():.1f}")
for n in names:
    d = L[n] - rec
    print(f"{n:10s} vs recorded : all frac|>1| {(d.abs()>1).float().mean():.4f} | digits frac|>3| {(d[isd].abs()>3).float().mean():.4f}")
# how sharp are the digit predictions? top-1 prob under fp32
