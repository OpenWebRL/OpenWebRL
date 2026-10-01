"""Check that per-micro-batch padding lines up with the micro-batches DataIterator yields.

Runs the real definitions out of slime/backends/megatron_utils/data.py without
importing megatron, by exec'ing just the pieces under test.
"""
import ast, random, types
from argparse import Namespace
from collections.abc import Sequence

SRC = "slime/backends/megatron_utils/data.py"
WANTED = {
    "DataIterator",
    "get_local_global_batch_size",
    "sort_samples_by_length",
    "compute_micro_batch_max_seq_lens",
    "log_padding_efficiency",
    "_NON_PER_SAMPLE_KEYS",
}

tree = ast.parse(open(SRC).read())
keep = []
for node in tree.body:
    name = getattr(node, "name", None)
    if name in WANTED:
        keep.append(node)
    elif isinstance(node, ast.Assign) and any(
        getattr(t, "id", None) in WANTED for t in node.targets
    ):
        keep.append(node)
assert len(keep) >= 5, f"only found {[getattr(n,'name',n) for n in keep]}"

ns = {
    "Namespace": Namespace,
    "Sequence": Sequence,
    "RolloutBatch": dict,
    "frozenset": frozenset,
    "logger": types.SimpleNamespace(info=lambda *a, **k: None),
    "mpu": types.SimpleNamespace(
        get_data_parallel_rank=lambda **k: 0,
        is_pipeline_last_stage=lambda: True,
    ),
}
exec(compile(ast.Module(body=keep, type_ignores=[]), SRC, "exec"), ns)

DataIterator = ns["DataIterator"]
sort_samples_by_length = ns["sort_samples_by_length"]
compute_micro_batch_max_seq_lens = ns["compute_micro_batch_max_seq_lens"]

PAD = 256


def make_lengths(n):
    """Turn-level browser samples: most near 2k, a thin tail an order of magnitude longer."""
    return [
        random.randint(800, 3000) if random.random() < 0.97 else random.randint(8000, 30000)
        for _ in range(n)
    ]


random.seed(0)
failures = []

for trial in range(300):
    dp_size = random.choice([1, 2, 5, 6])
    mbs = random.choice([1, 2, 4])
    gbs = dp_size * mbs * random.randint(1, 8)
    num_local_gbs = gbs // dp_size
    n = random.randint(num_local_gbs, num_local_gbs * 5)

    lengths = make_lengths(n)
    data = {
        "total_lengths": list(lengths),
        "tokens": [[i] * L for i, L in enumerate(lengths)],
        "loss_masks": [[1] * L for L in lengths],
        "sample_indices": list(range(n)),
        "effective_global_batch_size": gbs,
    }
    args = Namespace(micro_batch_size=mbs, global_batch_size=gbs, use_dynamic_batch_size=False)

    sort_samples_by_length(args, data, dp_size)
    data["max_seq_lens"] = compute_micro_batch_max_seq_lens(args, data, PAD)

    # the permutation must keep every per-sample field on the same sample
    for i, tok in enumerate(data["tokens"]):
        if len(tok) != data["total_lengths"][i] or len(data["loss_masks"][i]) != data["total_lengths"][i]:
            failures.append(f"trial {trial}: per-sample fields desynced at {i}")
            break
    if sorted(data["sample_indices"]) != list(range(n)):
        failures.append(f"trial {trial}: samples lost or duplicated")
    if sorted(data["total_lengths"]) != sorted(lengths):
        failures.append(f"trial {trial}: lengths changed")
    if data["effective_global_batch_size"] != gbs:
        failures.append(f"trial {trial}: scalar field was permuted")

    # sorting must stay inside one optimizer step, so each window holds the same samples
    for start in range(0, n, num_local_gbs):
        window = slice(start, min(start + num_local_gbs, n))
        if sorted(data["sample_indices"][window]) != sorted(range(start, min(start + num_local_gbs, n))):
            failures.append(f"trial {trial}: sort crossed an optimizer-step boundary")
            break

    # replay what get_data_iterator + DataIterator do in the static path
    num_steps = n // num_local_gbs
    it = DataIterator(data, mbs)
    for _ in range(num_steps * (num_local_gbs // mbs)):
        b = it.get_next(["tokens", "max_seq_lens", "total_lengths"])
        m = b["max_seq_lens"]
        if len(set(m)) != 1:
            failures.append(f"trial {trial}: micro-batch spans two padded lengths {set(m)}")
            break
        if max(len(t) for t in b["tokens"]) > m[0]:  # the assert inside get_batch
            failures.append(f"trial {trial}: a sample is longer than its padded length")
            break
        if m[0] % PAD:
            failures.append(f"trial {trial}: padded length {m[0]} is not a multiple of {PAD}")
            break

# what it saves on one rank's share of a real rollout
random.seed(1)
lengths = make_lengths(4688 // 6)
data = {"total_lengths": list(lengths), "tokens": [[0] * L for L in lengths]}
args = Namespace(micro_batch_size=2, global_batch_size=792, use_dynamic_batch_size=False)
before = len(lengths) * ((max(lengths) + PAD - 1) // PAD * PAD)
sort_samples_by_length(args, data, 6)
after = sum(compute_micro_batch_max_seq_lens(args, data, PAD))
real = sum(lengths)

print(f"samples {len(lengths)}  mean {real // len(lengths)}  max {max(lengths)}")
print(f"padded tokens: global max {before:,} -> per micro-batch {after:,}   ({before / after:.1f}x less work)")
print(f"remaining waste over real tokens: {after / real:.2f}x (was {before / real:.1f}x)")
print(f"\n{len(failures)} failures")
for f in failures[:10]:
    print(" ", f)
