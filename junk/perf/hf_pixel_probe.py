"""Does the trainer-side pixel_values tensor match what a fast / slow HF image processor
produces from the same screenshot, and which one makes HF agree with SGLang's recorded log-probs?"""
import os, io, base64, gc, torch
from PIL import Image
bank = "/weka/oe-training-default/new_peters/OpenWebRL/outputs/rollout_bank/b3ft/rollout_0.pt"
ckpt = "/weka/oe-training-default/new_peters/LlamaFactory/saves/baseline_3_ft/full/sft/checkpoint-1000"
N = int(os.environ.get("N", "40"))
data = torch.load(bank, weights_only=False)
samples = data["samples"] if isinstance(data, dict) else data
g = lambda s, k: s.get(k) if isinstance(s, dict) and s.get(k) is not None else (s.get("metadata") or {}).get(k)
ok = [s for s in samples if g(s, "rollout_log_probs") and g(s, "multimodal_train_inputs") and (g(s, "multimodal_inputs") or {}).get("images")]
stride = max(1, len(ok) // N)
picked = [{"tokens": list(g(s, "tokens"))[:-2], "rl": int(g(s, "response_length")) - 2,
           "rec": [float(x) for x in g(s, "rollout_log_probs")], "images": list(g(s, "multimodal_inputs")["images"]),
           "mm": dict(g(s, "multimodal_train_inputs"))} for s in ok[::stride][:N]]
del data, samples, ok; gc.collect()

from transformers import AutoProcessor, AutoTokenizer, AutoModelForImageTextToText
from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor
from transformers.models.qwen2_vl.image_processing_qwen2_vl_fast import Qwen2VLImageProcessorFast
auto = AutoProcessor.from_pretrained(ckpt)
print("AutoProcessor ->", type(auto).__name__, "image_processor:", type(auto.image_processor).__name__, flush=True)
try:
    from sglang.srt.utils.hf_transformers_utils import get_processor
    sp = get_processor(ckpt)
    print("sglang get_processor ->", type(sp).__name__, "image_processor:", type(sp.image_processor).__name__, flush=True)
except Exception as e:
    print("sglang get_processor failed:", repr(e), flush=True)
fast = Qwen2VLImageProcessorFast.from_pretrained(ckpt)
slow = Qwen2VLImageProcessor.from_pretrained(ckpt)
print("fast size", fast.size, "| slow size", slow.size, "| auto size", auto.image_processor.size, flush=True)

def decode(url):
    b = url.split(",", 1)[1] if "," in url else url
    return Image.open(io.BytesIO(base64.b64decode(b))).convert("RGB")

tok = AutoTokenizer.from_pretrained(ckpt)
model = AutoModelForImageTextToText.from_pretrained(ckpt, torch_dtype=torch.bfloat16).cuda().eval()
variants = ["stored", "fast", "slow"]
agg = {v: [] for v in variants}; pvdiff = {"fast": [], "slow": []}
for i, s in enumerate(picked):
    tokens = torch.tensor(s["tokens"]); rl = s["rl"]; T = tokens.numel()
    img = decode(s["images"][-1]) if len(s["images"]) == 1 else None
    if img is None:
        print(f"sample {i}: {len(s['images'])} images, skipping", flush=True); continue
    pvs = {"stored": s["mm"]["pixel_values"].float()}
    pvs["fast"] = fast(images=[img], return_tensors="pt")["pixel_values"].float()
    pvs["slow"] = slow(images=[img], return_tensors="pt")["pixel_values"].float()
    grid = s["mm"]["image_grid_thw"]
    shapes = {k: tuple(v.shape) for k, v in pvs.items()}
    if len({v for v in shapes.values()}) != 1:
        print(f"sample {i}: pixel_values shape mismatch {shapes} (stored grid {grid.tolist()}) -- skipping", flush=True); continue
    for k in ("fast", "slow"):
        pvdiff[k].append((pvs[k] - pvs["stored"]).abs().max().item())
    rec = torch.tensor(s["rec"][:rl]); resp = tokens[-rl:]
    strs = [tok.decode([t]) for t in resp.tolist()]; isd = torch.tensor([st.strip().isdigit() for st in strs])
    line = f"sample {i}: rl {rl} digits {int(isd.sum())} | max|pv-stored| fast {pvdiff['fast'][-1]:.4f} slow {pvdiff['slow'][-1]:.4f} | digit outliers(>3) vs recorded:"
    for v in variants:
        kw = {"pixel_values": pvs[v].to(torch.bfloat16).cuda(), "image_grid_thw": grid.cuda(),
              "mm_token_type_ids": (tokens == model.config.image_token_id).long()[None].cuda()}
        with torch.no_grad():
            logits = model(input_ids=tokens[None].cuda(), **kw).logits[0].float()
        pos = torch.arange(T - rl - 1, T - 1)
        lp = torch.log_softmax(logits[pos], -1).gather(1, resp.cuda()[:, None]).squeeze(1).cpu()
        d = lp - rec
        agg[v].append(d[isd])
        line += f" {v} {int((d[isd].abs() > 3).sum())}"
    print(line, flush=True)
print("\n==== OVERALL digits: frac |HF - recorded| > 3 by pixel source")
for v in variants:
    D = torch.cat(agg[v]); print(f"{v:7s} n={D.numel()} mean {D.mean():.3f} frac|>3| {(D.abs()>3).float().mean():.4f}")
for k in ("fast", "slow"):
    print(f"max|pv_{k} - pv_stored| over samples: max {max(pvdiff[k]):.4f} median {sorted(pvdiff[k])[len(pvdiff[k])//2]:.4f}")
