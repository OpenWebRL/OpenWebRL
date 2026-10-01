"""Name the Python call sites of GPU->CPU syncs in a torch trace recorded with_stack=True.

usage: python analyze_syncs.py <trace.json.gz>
"""
import gzip, json, sys
from collections import defaultdict

SYNC_OPS = {"aten::item", "aten::_local_scalar_dense", "aten::is_nonzero"}
SKIP = ("torch/", "site-packages/torch", "<built-in", "torch._", "autograd", "profiler", "threading.py", "typing.py")

def load(p):
    with gzip.open(p, "rt") if p.endswith(".gz") else open(p) as f:
        return json.load(f)

def main(path):
    ev = load(path)["traceEvents"]
    pyf = sorted((e for e in ev if e.get("cat") == "python_function" and "dur" in e), key=lambda e: (e["ts"], -e["dur"]))
    syncs = [e for e in ev if e.get("cat") == "cpu_op" and e.get("name") in SYNC_OPS and e["name"] != "aten::_local_scalar_dense"]
    print(f"python_function events: {len(pyf):,}   sync ops: {len(syncs):,}")
    if not pyf:
        print("no python stacks in this trace (was SLIME_PROFILE_STACKS=1 set?)"); return
    # sweep: maintain the open python frame stack; at each sync record the innermost non-torch frames
    events = [(e["ts"], 1, e) for e in pyf] + [(e["ts"] + e["dur"], 0, e) for e in pyf] + [(e["ts"], 2, e) for e in syncs]
    events.sort(key=lambda x: (x[0], x[1]))
    stack = []; by_site = defaultdict(lambda: [0, 0.0]); by_chain = defaultdict(lambda: [0, 0.0])
    for ts, kind, e in events:
        if kind == 1: stack.append(e)
        elif kind == 0:
            if stack and stack[-1] is e: stack.pop()
            else:
                try: stack.remove(e)
                except ValueError: pass
        else:
            frames = [f["name"] for f in stack]
            user = [f for f in frames if not any(s in f for s in SKIP)]
            site = user[-1] if user else (frames[-1] if frames else "<none>")
            chain = " <- ".join(reversed(user[-3:])) if user else site
            by_site[site][0] += 1; by_site[site][1] += e["dur"] / 1e6
            by_chain[chain][0] += 1; by_chain[chain][1] += e["dur"] / 1e6
    print("\nsync call sites (innermost non-torch frame):")
    for site, (c, t) in sorted(by_site.items(), key=lambda kv: -kv[1][1])[:15]:
        print(f"  {t:6.1f}s {c:>8,}x  {site[:120]}")
    print("\nwith two callers of context:")
    for chain, (c, t) in sorted(by_chain.items(), key=lambda kv: -kv[1][1])[:10]:
        print(f"  {t:6.1f}s {c:>8,}x  {chain[:200]}")

if __name__ == "__main__":
    main(sys.argv[1])
