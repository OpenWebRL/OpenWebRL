"""3-way log-prob check on saved rollout samples: recorded SGLang (rollout time) vs a fresh
SGLang server (flags from env) vs HF transformers. Answers whether SGLang's image path
(fast GPU image processor / mm attention backend) is what moves coordinate digits.

env: MM_BACKEND (default triton_attn, "" = server default), DISABLE_FAST_IMG=1, N (samples)
"""
import os, sys, time, json, subprocess, torch, requests

bank = "/weka/oe-training-default/new_peters/OpenWebRL/outputs/rollout_bank/b3ft/rollout_0.pt"
ckpt = "/weka/oe-training-default/new_peters/LlamaFactory/saves/baseline_3_ft/full/sft/checkpoint-1000"
N = int(os.environ.get("N", "40"))
MM_BACKEND = os.environ.get("MM_BACKEND", "triton_attn")
if MM_BACKEND.lower() in ("none", "default"):
    MM_BACKEND = ""
DISABLE_FAST = os.environ.get("DISABLE_FAST_IMG", "0") == "1"
PRECISE_INTERP = os.environ.get("PRECISE_INTERP", "0") == "1"
ON_POLICY = os.environ.get("RL_ON_POLICY", "0") == "1"
EXTRA_ARGS = os.environ.get("SGL_EXTRA_ARGS", "").split()
PORT = 30123
print(f"config: mm_backend={MM_BACKEND or 'default'} disable_fast_image_processor={DISABLE_FAST} precise_embedding_interpolation={PRECISE_INTERP} on_policy={ON_POLICY} extra={EXTRA_ARGS} N={N}", flush=True)

print("loading bank ...", flush=True)
data = torch.load(bank, weights_only=False)
samples = data["samples"] if isinstance(data, dict) else data
def g(s, k):
    v = s.get(k) if isinstance(s, dict) else None
    if v is None and isinstance(s, dict):
        v = (s.get("metadata") or {}).get(k)
    return v
ok = [s for s in samples if g(s, "rollout_log_probs") and g(s, "multimodal_train_inputs") and (g(s, "multimodal_inputs") or {}).get("images")]
stride = max(1, len(ok) // N)
picked = ok[::stride][:N]
print(f"{len(ok)} usable samples, picked {len(picked)} (stride {stride})", flush=True)
# keep only what we need, drop the 42 GB of pixel tensors
picked = [{"tokens": list(g(s, "tokens")), "rl": int(g(s, "response_length")),
           "rec": [float(x) for x in g(s, "rollout_log_probs")],
           "images": list(g(s, "multimodal_inputs")["images"]),
           "mm": {k: v for k, v in g(s, "multimodal_train_inputs").items()}} for s in picked]
del data, samples, ok
import gc; gc.collect()

# ---------- HF reference (temperature 1) ----------
from transformers import AutoTokenizer, AutoModelForImageTextToText
tok = AutoTokenizer.from_pretrained(ckpt)
model = AutoModelForImageTextToText.from_pretrained(ckpt, torch_dtype=torch.bfloat16).cuda().eval()
for s in picked:
    tokens = torch.tensor(s["tokens"], dtype=torch.long)
    rl = s["rl"]; T = tokens.numel()
    kw = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in s["mm"].items()}
    if "pixel_values" in kw and kw["pixel_values"].is_floating_point():
        kw["pixel_values"] = kw["pixel_values"].to(torch.bfloat16)
    if "image_grid_thw" in kw:
        kw["mm_token_type_ids"] = (tokens == model.config.image_token_id).long()[None].cuda()
    with torch.no_grad():
        logits = model(input_ids=tokens[None].cuda(), **kw).logits[0].float()
    pos = torch.arange(T - rl - 1, T - 1)
    lp = torch.log_softmax(logits[pos], dim=-1).gather(1, tokens[-rl:].cuda()[:, None]).squeeze(1)
    s["hf"] = lp.cpu().tolist()
    s["mm"] = None
del model; gc.collect(); torch.cuda.empty_cache()
print("HF done", flush=True)

# ---------- fresh SGLang server ----------
cmd = [sys.executable, "-m", "sglang.launch_server", "--model-path", ckpt, "--port", str(PORT),
       "--mem-fraction-static", "0.6", "--log-level", "warning", "--chunked-prefill-size", "8192",
       "--trust-remote-code"]
if MM_BACKEND:
    cmd += ["--mm-attention-backend", MM_BACKEND]
if DISABLE_FAST:
    cmd += ["--disable-fast-image-processor"]
if PRECISE_INTERP:
    cmd += ["--enable-precise-embedding-interpolation"]
if ON_POLICY:
    cmd += ["--rl-on-policy-target", "fsdp"]
cmd += EXTRA_ARGS
print("launching:", " ".join(cmd), flush=True)
srv = subprocess.Popen(cmd, stdout=open("/tmp/sgl_server.log", "w"), stderr=subprocess.STDOUT)
t0 = time.time()
while True:
    try:
        if requests.get(f"http://127.0.0.1:{PORT}/health", timeout=5).status_code == 200:
            break
    except Exception:
        pass
    if srv.poll() is not None or time.time() - t0 > 1500:
        print("SERVER FAILED:\n" + open("/tmp/sgl_server.log").read()[-6000:], flush=True); sys.exit(1)
    time.sleep(5)
print(f"server up in {time.time()-t0:.0f}s", flush=True)

def digits(strs):
    return torch.tensor([st.strip().isdigit() for st in strs])

agg = {"fresh-rec": [], "hf-rec": [], "fresh-hf": []}
aggd = {k: [] for k in agg}
misaligned = 0
for i, s in enumerate(picked):
    # drop the two appended tokens (im_end + newline): fabricated recorded lp, and they retokenize differently
    tokens = s["tokens"][:-2]; rl = s["rl"] - 2; T = len(tokens)
    payload = {"input_ids": tokens, "image_data": s["images"],
               "sampling_params": {"max_new_tokens": 1, "temperature": 1.0},
               "return_logprob": True, "logprob_start_len": max(0, T - rl - 1)}
    r = requests.post(f"http://127.0.0.1:{PORT}/generate", json=payload, timeout=600).json()
    meta = r.get("meta_info", {})
    itl = meta.get("input_token_logprobs") or []
    ids = [x[1] for x in itl]
    resp = tokens[-rl:]
    # locate the response ids as a suffix of what the server tokenized
    if ids[-rl:] != resp:
        misaligned += 1
        if misaligned <= 3:
            print(f"sample {i}: misaligned. server returned {len(itl)} entries; server tail {ids[-rl:][:12]} vs expected {resp[:12]}; "
                  f"server tail text {tok.decode(ids[-rl:])[:120]!r} | expected {tok.decode(resp)[:120]!r} | keys {list(meta.keys())[:8]}", flush=True)
        else:
            print(f"sample {i}: misaligned ({len(itl)} entries)", flush=True)
        continue
    tail = itl[-rl:]
    fresh = torch.tensor([x[0] if x[0] is not None else float("nan") for x in tail])
    rec = torch.tensor(s["rec"][:rl]); hf = torch.tensor(s["hf"][:rl])
    strs = [tok.decode([t]) for t in resp]
    isd = digits(strs)
    keep = torch.ones(rl, dtype=torch.bool)
    d = {"fresh-rec": fresh - rec, "hf-rec": hf - rec, "fresh-hf": fresh - hf}
    for k, v in d.items():
        agg[k].append(v[keep]); aggd[k].append(v[keep & isd])
    bad = lambda v: int(((v[keep & isd]).abs() > 3).sum())
    print(f"sample {i}: rl {rl} digits {int((keep&isd).sum())} | digit outliers(>3): fresh-rec {bad(d['fresh-rec'])} hf-rec {bad(d['hf-rec'])} fresh-hf {bad(d['fresh-hf'])} | "
          f"mean fresh-rec {d['fresh-rec'][keep].mean():.3f} hf-rec {d['hf-rec'][keep].mean():.3f} fresh-hf {d['fresh-hf'][keep].mean():.3f}", flush=True)
    worst = torch.argsort(d["fresh-hf"][keep].abs(), descending=True)[:3].tolist()
    print("   worst fresh-vs-hf:", " || ".join(f"pos {j} {strs[j]!r} rec {rec[j]:.2f} fresh {fresh[j]:.2f} hf {hf[j]:.2f}" for j in worst), flush=True)

print("\n==== OVERALL", f"mm_backend={MM_BACKEND or 'default'} disable_fast_image_processor={DISABLE_FAST} precise_embedding_interpolation={PRECISE_INTERP} on_policy={ON_POLICY} extra={EXTRA_ARGS}", f"misaligned {misaligned}/{len(picked)}")
for k in agg:
    if not agg[k]:
        print(k, "no aligned samples"); continue
    A = torch.cat(agg[k]); D = torch.cat(aggd[k])
    print(f"{k:10s} all: mean {A.mean():.3f} median {A.median():.3f} frac|>1| {(A.abs()>1).float().mean():.4f} | digits n={D.numel()} mean {D.mean():.3f} frac|>3| {(D.abs()>3).float().mean():.4f}")
srv.terminate()
