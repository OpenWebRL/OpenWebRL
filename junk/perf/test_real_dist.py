"""Size the micro-batch packer on the length distribution the run actually logged:
2350 samples, mean 1949, p50 2111, p99 2843, max 2998."""
import ast, random, types
from argparse import Namespace
from collections.abc import Sequence

SRC = "slime/backends/megatron_utils/data.py"
WANTED = {"DataIterator", "get_local_global_batch_size", "sort_samples_by_length",
          "build_padded_micro_batches", "compute_micro_batch_max_seq_lens", "_NON_PER_SAMPLE_KEYS"}
tree = ast.parse(open(SRC).read())
keep = [n for n in tree.body if getattr(n, "name", None) in WANTED
        or (isinstance(n, ast.Assign) and any(getattr(t, "id", None) in WANTED for t in n.targets))]
ns = {"Namespace": Namespace, "Sequence": Sequence, "RolloutBatch": dict,
      "logger": types.SimpleNamespace(info=lambda *a, **k: None),
      "torch": types.SimpleNamespace(tensor=lambda v, **k: types.SimpleNamespace(tolist=lambda: list(v)),
                                     int=None, cuda=types.SimpleNamespace(current_device=lambda: 0)),
      "dist": types.SimpleNamespace(is_initialized=lambda: False, ReduceOp=types.SimpleNamespace(MAX=0)),
      "mpu": types.SimpleNamespace(get_data_parallel_group=lambda **k: None)}
exec(compile(ast.Module(body=keep, type_ignores=[]), SRC, "exec"), ns)

PAD, DP, GBS = 256, 2, 792
random.seed(7)
# left-skewed toward ~2.1k with a hard ceiling at 2998, matching the logged quantiles
lengths = [min(2998, max(900, int(random.gauss(1949, 430)))) for _ in range(2350)]
lengths.sort(key=lambda _: random.random())

print(f"mean {sum(lengths)//len(lengths)}  max {max(lengths)}  (logged: mean 1949, max 2998)\n")
print(f"{'config':<34}{'micro-batches':>14}{'tokens/mb':>12}{'pad':>7}")
for label, budget, mbs in [("fixed micro_batch_size 1", 0, 1), ("fixed micro_batch_size 2", 0, 2),
                           ("token budget 16384", 16384, 2), ("token budget 32768", 32768, 2),
                           ("token budget 65536", 65536, 2)]:
    data = {"total_lengths": list(lengths), "tokens": [[0] * L for L in lengths]}
    args = Namespace(micro_batch_size=mbs, global_batch_size=GBS, use_dynamic_batch_size=False)
    ns["sort_samples_by_length"](args, data, DP)
    num_local_gbs = GBS // DP
    trained = (len(lengths) // num_local_gbs) * num_local_gbs
    if budget:
        mb, msl, _ = ns["build_padded_micro_batches"](args, data, DP, PAD, budget)
        count = len(mb)
        padded = sum(msl[i] for g in mb for i in g)
        real = sum(data["total_lengths"][i] for g in mb for i in g)
    else:
        msl = ns["compute_micro_batch_max_seq_lens"](args, data, PAD)
        count = trained // mbs
        padded = sum(msl[:trained]); real = sum(data["total_lengths"][:trained])
    print(f"{label:<34}{count:>14}{padded//count:>12,}{padded/real:>7.2f}x")
print(f"\ntrained samples per rank: {trained}")
