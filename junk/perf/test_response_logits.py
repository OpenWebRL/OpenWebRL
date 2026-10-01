"""The compacted logits must contain exactly what get_responses would have sliced.

Runs the real set_selection out of response_logits.py and the real index arithmetic out
of loss.py's get_responses, with torch stubbed, and compares the two element by element.
"""
import ast, os, random, types

os.environ["SLIME_RESPONSE_ONLY_LOGITS"] = "1"

SRC = "slime/backends/megatron_utils/response_logits.py"
tree = ast.parse(open(SRC).read())
keep = [n for n in tree.body if getattr(n, "name", None) in {"enabled", "set_selection", "clear_selection", "selection_offsets"}]
keep += [n for n in tree.body if isinstance(n, ast.AnnAssign) and getattr(n.target, "id", "").startswith("_")]
ns = {
    "os": os,
    "torch": types.SimpleNamespace(tensor=lambda v, **k: list(v), long=None, Tensor=list),
    "logger": types.SimpleNamespace(info=lambda *a, **k: None),
}
ns["_selection"] = None
ns["_offsets"] = None
exec(compile(ast.Module(body=keep, type_ignores=[]), SRC, "exec"), ns)
set_selection = ns["set_selection"]
selection_offsets = ns["selection_offsets"]

PAD = 256
random.seed(0)
failures = []

for trial in range(500):
    n = random.randint(1, 12)
    lengths = [random.randint(900, 3000) for _ in range(n)]
    responses = [random.randint(1, 200) for _ in range(n)]
    responses = [min(r, l - 1) for r, l in zip(responses, lengths)]
    msl = (max(lengths) + PAD - 1) // PAD * PAD
    max_seq_lens = [msl] * n
    V = 5  # stand-in vocab width

    # a full [b*msl, V] logits tensor where every row is identifiable by its flat position
    full = [[p * 10 + c for c in range(V)] for p in range(n * msl)]

    # what the unmodified get_responses reads: logits[start-1 : end-1]
    want = []
    for i, (tl, rl) in enumerate(zip(lengths, responses)):
        end = msl * i + tl
        start = end - rl
        want.append(full[start - 1 : end - 1])

    # what the compact path produces
    set_selection(lengths, responses, max_seq_lens, device=None)
    selection = ns["_selection"]
    offsets = selection_offsets()
    compact = [full[p] for p in selection]

    if len(selection) != sum(responses):
        failures.append(f"trial {trial}: selected {len(selection)} rows, expected {sum(responses)}")
        continue
    for i, rl in enumerate(responses):
        got = compact[offsets[i] : offsets[i] + rl]
        if got != want[i]:
            failures.append(f"trial {trial}: sample {i} mismatch (first row {got[:1]} vs {want[i][:1]})")
            break
    # no position may be selected twice, and none may fall outside the sample's own row
    if len(set(selection)) != len(selection):
        failures.append(f"trial {trial}: duplicate positions selected")
    for i, (tl, rl) in enumerate(zip(lengths, responses)):
        lo, hi = msl * i, msl * (i + 1)
        if not all(lo <= p < hi for p in selection[offsets[i] : offsets[i] + rl]):
            failures.append(f"trial {trial}: sample {i} selected outside its own row")
            break

# how much the output layer stops computing, on the measured distribution
lengths = [1949] * 1000
responses = [69] * 1000
print(f"positions projected into the vocab: {sum(lengths):,} -> {sum(responses):,} "
      f"({sum(lengths)/sum(responses):.1f}x fewer)")
print(f"\n{len(failures)} failures")
for f in failures[:8]:
    print(" ", f)
