"""Compare the trainer's per-token log-probs (SLIME_LOGPROB_DUMP files) with HF transformers on
the same samples, matched by token ids against the rollout bank. Digits called out."""
import sys, os, glob, torch
dump_dir = sys.argv[1]
bank = "/weka/oe-training-default/new_peters/OpenWebRL/outputs/rollout_bank/b3ft/rollout_0.pt"
ckpt = "/weka/oe-training-default/new_peters/LlamaFactory/saves/baseline_3_ft/full/sft/checkpoint-1000"
N = int(os.environ.get("N", "60"))
recs = []
for f in sorted(glob.glob(os.path.join(dump_dir, "*.pt"))):
    recs += torch.load(f)
print(f"{len(recs)} dumped samples from {dump_dir}", flush=True)
print("loading bank ...", flush=True)
data = torch.load(bank, weights_only=False)
samples = data["samples"] if isinstance(data, dict) else data
index = {}
for s in samples:
    t = s.get("tokens") if isinstance(s, dict) else None
    if t is not None:
        index[tuple(int(x) for x in t)] = s
matched = [(r, index[tuple(r["tokens"])]) for r in recs if tuple(r["tokens"]) in index]
print(f"matched {len(matched)}/{len(recs)}", flush=True)
stride = max(1, len(matched) // N)
matched = matched[::stride][:N]
from transformers import AutoTokenizer, AutoModelForImageTextToText
tok = AutoTokenizer.from_pretrained(ckpt)
model = AutoModelForImageTextToText.from_pretrained(ckpt, torch_dtype=torch.bfloat16).cuda().eval()
TEMP = float(os.environ.get("TEMP", "0.7"))  # trainer divides logits by rollout temperature
agg = {"trainer-hf": [], "trainer-rec": [], "hf-rec": []}; aggd = {k: [] for k in agg}
for i, (r, s) in enumerate(matched):
    tokens = torch.tensor(r["tokens"], dtype=torch.long); rl = r["rl"]; T = tokens.numel()
    mm = s.get("multimodal_train_inputs") or {}
    kw = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in mm.items()}
    if "pixel_values" in kw and kw["pixel_values"].is_floating_point():
        kw["pixel_values"] = kw["pixel_values"].to(torch.bfloat16)
    if "image_grid_thw" in kw:
        kw["mm_token_type_ids"] = (tokens == model.config.image_token_id).long()[None].cuda()
    with torch.no_grad():
        logits = model(input_ids=tokens[None].cuda(), **kw).logits[0].float()
    pos = torch.arange(T - rl - 1, T - 1)
    hf = torch.log_softmax(logits[pos] / TEMP, dim=-1).gather(1, tokens[-rl:].cuda()[:, None]).squeeze(1).cpu()
    tr = torch.tensor(r["trainer_lp"]); rec = torch.tensor(r["rollout_lp"])
    strs = [tok.decode([t]) for t in tokens[-rl:].tolist()]
    isd = torch.tensor([st.strip().isdigit() for st in strs])
    keep = torch.ones(rl, dtype=torch.bool); keep[-2:] = False   # appended im_end + newline
    d = {"trainer-hf": tr - hf, "trainer-rec": tr - rec, "hf-rec": hf - rec}
    for k, v in d.items():
        agg[k].append(v[keep]); aggd[k].append(v[keep & isd])
    bad = lambda v: int((v[keep & isd].abs() > 3).sum())
    print(f"sample {i}: rl {rl} digits {int((keep&isd).sum())} | digit outliers(>3): trainer-hf {bad(d['trainer-hf'])} trainer-rec {bad(d['trainer-rec'])} hf-rec {bad(d['hf-rec'])} | "
          f"mean trainer-hf {d['trainer-hf'][keep].mean():.3f} max|trainer-hf| {d['trainer-hf'][keep].abs().max():.2f}", flush=True)
    worst = torch.argsort(d["trainer-hf"][keep].abs(), descending=True)[:3].tolist()
    print("   worst trainer-vs-hf:", " || ".join(f"pos {j} {strs[j]!r} rec {rec[j]:.2f} trainer {tr[j]:.2f} hf {hf[j]:.2f}" for j in worst), flush=True)
print("\n==== OVERALL trainer vs HF (same samples, same pixel_values)")
for k in agg:
    A = torch.cat(agg[k]); D = torch.cat(aggd[k])
    print(f"{k:12s} all: mean {A.mean():.3f} median {A.median():.3f} frac|>1| {(A.abs()>1).float().mean():.4f} max {A.abs().max():.2f} | digits n={D.numel()} mean {D.mean():.3f} frac|>3| {(D.abs()>3).float().mean():.4f}")
