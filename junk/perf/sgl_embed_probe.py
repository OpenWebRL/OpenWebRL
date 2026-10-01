"""Bisect SGLang's image path: score the recorded rollout tokens with a fresh SGLang engine
(a) from the raw screenshot, and (b) from HF's own vision-tower embeddings passed in as
precomputed image features. Compared against HF log-probs on the same samples.
If (b) matches HF, SGLang's vision tower is what differs; if (b) still disagrees, it is the LM side."""
import os, sys, gc, torch
bank = "/weka/oe-training-default/new_peters/OpenWebRL/outputs/rollout_bank/b3ft/rollout_0.pt"
ckpt = "/weka/oe-training-default/new_peters/LlamaFactory/saves/baseline_3_ft/full/sft/checkpoint-1000"
N = int(os.environ.get("N", "40"))
print("loading bank ...", flush=True)
data = torch.load(bank, weights_only=False)
samples = data["samples"] if isinstance(data, dict) else data
g = lambda s, k: s.get(k) if isinstance(s, dict) and s.get(k) is not None else (s.get("metadata") or {}).get(k)
ok = [s for s in samples if g(s, "rollout_log_probs") and g(s, "multimodal_train_inputs") and (g(s, "multimodal_inputs") or {}).get("images")]
stride = max(1, len(ok) // N)
picked = [{"tokens": list(g(s, "tokens"))[:-2], "rl": int(g(s, "response_length")) - 2,
           "rec": [float(x) for x in g(s, "rollout_log_probs")], "images": list(g(s, "multimodal_inputs")["images"]),
           "mm": dict(g(s, "multimodal_train_inputs"))} for s in ok[::stride][:N]]
del data, samples, ok; gc.collect()
print(f"picked {len(picked)}", flush=True)

from transformers import AutoTokenizer, AutoModelForImageTextToText
tok = AutoTokenizer.from_pretrained(ckpt)
model = AutoModelForImageTextToText.from_pretrained(ckpt, torch_dtype=torch.bfloat16).cuda().eval()
for s in picked:
    tokens = torch.tensor(s["tokens"]); rl = s["rl"]; T = tokens.numel()
    kw = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in s["mm"].items()}
    kw["pixel_values"] = kw["pixel_values"].to(torch.bfloat16)
    kw["mm_token_type_ids"] = (tokens == model.config.image_token_id).long()[None].cuda()
    with torch.no_grad():
        logits = model(input_ids=tokens[None].cuda(), **kw).logits[0].float()
        feats = model.model.get_image_features(kw["pixel_values"], kw["image_grid_thw"])
    if hasattr(feats, "pooler_output"):
        feats = feats.pooler_output
    if isinstance(feats, (tuple, list)):
        feats = torch.cat(list(feats), dim=0)
    pos = torch.arange(T - rl - 1, T - 1)
    s["hf"] = torch.log_softmax(logits[pos], -1).gather(1, tokens[-rl:].cuda()[:, None]).squeeze(1).cpu()
    s["feat"] = feats.detach().to(torch.bfloat16).cpu()
    s["grid"] = s["mm"]["image_grid_thw"].clone()
    s["mm"] = None
print("HF done; feature shape", tuple(picked[0]["feat"].shape), flush=True)
del model; gc.collect(); torch.cuda.empty_cache()

import sglang
llm = sglang.Engine(model_path=ckpt, mem_fraction_static=0.6, mm_attention_backend="triton_attn",
                    log_level="warning", trust_remote_code=True, chunked_prefill_size=8192)
print("engine up", flush=True)

def score(s, image_data):
    tokens = s["tokens"]; rl = s["rl"]; T = len(tokens)
    out = llm.generate(input_ids=tokens, image_data=image_data,
                       sampling_params={"max_new_tokens": 1, "temperature": 1.0},
                       return_logprob=True, logprob_start_len=max(0, T - rl - 1))
    itl = out["meta_info"]["input_token_logprobs"]
    ids = [x[1] for x in itl]
    if ids[-rl:] != tokens[-rl:]:
        return None
    return torch.tensor([x[0] if x[0] is not None else float("nan") for x in itl[-rl:]])

agg = {"img-rec": [], "hfemb-rec": [], "hfemb-hf": [], "img-hf": []}; aggd = {k: [] for k in agg}
bad_n = 0
for i, s in enumerate(picked):
    rl = s["rl"]; resp = s["tokens"][-rl:]
    a = score(s, s["images"])
    try:
        b = score(s, [{"format": "precomputed_embedding", "feature": s["feat"].cuda(), "image_grid_thw": s["grid"]}])
    except Exception as e:
        print(f"sample {i}: precomputed path failed: {e!r}", flush=True); b = None
    if a is None or b is None:
        bad_n += 1; print(f"sample {i}: skipped (a={a is not None} b={b is not None})", flush=True); continue
    rec = torch.tensor(s["rec"][:rl]); hf = s["hf"][:rl]
    strs = [tok.decode([t]) for t in resp]; isd = torch.tensor([st.strip().isdigit() for st in strs])
    d = {"img-rec": a - rec, "hfemb-rec": b - rec, "hfemb-hf": b - hf, "img-hf": a - hf}
    for k, v in d.items():
        agg[k].append(v); aggd[k].append(v[isd])
    bad = lambda v: int((v[isd].abs() > 3).sum())
    print(f"sample {i}: rl {rl} digits {int(isd.sum())} | digit outliers(>3): img-rec {bad(d['img-rec'])} hfemb-rec {bad(d['hfemb-rec'])} hfemb-hf {bad(d['hfemb-hf'])} img-hf {bad(d['img-hf'])}", flush=True)
    worst = torch.argsort(d["hfemb-hf"].abs(), descending=True)[:3].tolist()
    print("   worst hfemb-vs-hf:", " || ".join(f"pos {j} {strs[j]!r} rec {rec[j]:.2f} img {a[j]:.2f} hfemb {b[j]:.2f} hf {hf[j]:.2f}" for j in worst), flush=True)
print(f"\n==== OVERALL skipped {bad_n}/{len(picked)}")
for k in agg:
    if not agg[k]: print(k, "none"); continue
    A = torch.cat(agg[k]); D = torch.cat(aggd[k])
    print(f"{k:10s} all: mean {A.mean():.3f} median {A.median():.3f} frac|>1| {(A.abs()>1).float().mean():.4f} | digits n={D.numel()} mean {D.mean():.3f} frac|>3| {(D.abs()>3).float().mean():.4f}")
llm.shutdown()
