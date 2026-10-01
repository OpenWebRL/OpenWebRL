"""Check the token-budget micro-batch packer against what DataIterator will yield."""
import ast, random, types
from argparse import Namespace
from collections.abc import Sequence

SRC = "slime/backends/megatron_utils/data.py"
WANTED = {
    "DataIterator",
    "get_local_global_batch_size",
    "sort_samples_by_length",
    "build_padded_micro_batches",
    "_NON_PER_SAMPLE_KEYS",
}
tree = ast.parse(open(SRC).read())
keep = [
    n for n in tree.body
    if getattr(n, "name", None) in WANTED
    or (isinstance(n, ast.Assign) and any(getattr(t, "id", None) in WANTED for t in n.targets))
]
ns = {
    "Namespace": Namespace, "Sequence": Sequence, "RolloutBatch": dict,
    "logger": types.SimpleNamespace(info=lambda *a, **k: None),
    "torch": types.SimpleNamespace(
        tensor=lambda v, **k: types.SimpleNamespace(tolist=lambda: list(v)),
        int=None, cuda=types.SimpleNamespace(current_device=lambda: 0),
    ),
    "dist": types.SimpleNamespace(is_initialized=lambda: False, ReduceOp=types.SimpleNamespace(MAX=0)),
    "mpu": types.SimpleNamespace(get_data_parallel_group=lambda **k: None),
}
exec(compile(ast.Module(body=keep, type_ignores=[]), SRC, "exec"), ns)
DataIterator = ns["DataIterator"]
sort_samples_by_length = ns["sort_samples_by_length"]
build_padded_micro_batches = ns["build_padded_micro_batches"]

PAD = 256
random.seed(0)
failures = []

for trial in range(300):
    dp_size = random.choice([1, 2, 4, 6])
    gbs = dp_size * random.randint(20, 200)
    num_local_gbs = gbs // dp_size
    n = random.randint(num_local_gbs, num_local_gbs * 4)
    budget = random.choice([8192, 16384, 32768, 65536])
    lengths = [
        random.randint(800, 3000) if random.random() < 0.97 else random.randint(8000, 30000)
        for _ in range(n)
    ]
    data = {
        "total_lengths": list(lengths),
        "tokens": [[i] * L for i, L in enumerate(lengths)],
        "sample_indices": list(range(n)),
    }
    args = Namespace(micro_batch_size=2, global_batch_size=gbs, use_dynamic_batch_size=False)

    sort_samples_by_length(args, data, dp_size)
    mb_indices, max_seq_lens, per_step = build_padded_micro_batches(args, data, dp_size, PAD, budget)
    data["max_seq_lens"] = max_seq_lens

    num_steps = n // num_local_gbs
    covered = [i for g in mb_indices for i in g]
    if len(covered) != len(set(covered)):
        failures.append(f"trial {trial}: a sample is in two micro-batches")
    if set(covered) != set(range(num_steps * num_local_gbs)):
        failures.append(f"trial {trial}: micro-batches do not cover exactly the trained samples")
    if sum(per_step) != len(mb_indices):
        failures.append(f"trial {trial}: per-step counts disagree with the schedule")

    it = DataIterator(data, None, mb_indices)
    for _ in range(len(mb_indices)):
        b = it.get_next(["tokens", "max_seq_lens"])
        m = b["max_seq_lens"]
        if len(set(m)) != 1:
            failures.append(f"trial {trial}: micro-batch spans two padded lengths"); break
        if max(len(t) for t in b["tokens"]) > m[0]:
            failures.append(f"trial {trial}: sample longer than its padded length"); break
        if m[0] % PAD:
            failures.append(f"trial {trial}: padded length not a multiple of {PAD}"); break
        # a group is allowed to exceed the budget only when it is a single sample
        if len(m) > 1 and len(m) * m[0] > budget:
            failures.append(f"trial {trial}: multi-sample micro-batch {len(m)}x{m[0]} over budget {budget}"); break

# what it does to one rank's share of a real rollout, against fixed micro_batch_size=2
random.seed(1)
lengths = [
    random.randint(800, 3000) if random.random() < 0.97 else random.randint(8000, 30000)
    for _ in range(4688 // 4)
]
data = {"total_lengths": list(lengths), "tokens": [[0] * L for L in lengths]}
args = Namespace(micro_batch_size=2, global_batch_size=792, use_dynamic_batch_size=False)
sort_samples_by_length(args, data, 4)
mb, msl, _ = build_padded_micro_batches(args, data, 4, PAD, 32768)
num_local_gbs = 792 // 4
trained = (len(lengths) // num_local_gbs) * num_local_gbs
fixed_count = trained // 2
print(f"samples/rank {len(lengths)}  trained {trained}")
print(f"micro-batches: fixed size 2 -> {fixed_count},  32k budget -> {len(mb)}  ({fixed_count/len(mb):.1f}x fewer)")
print(f"mean samples per micro-batch: {trained/len(mb):.1f}")
print(f"padded tokens: {sum(msl[i] for g in mb for i in g):,} vs real {sum(lengths[i] for g in mb for i in g):,}")
print(f"\n{len(failures)} failures")
for f in failures[:8]:
    print(" ", f)
