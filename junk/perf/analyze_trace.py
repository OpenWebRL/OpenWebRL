"""Aggregate GPU kernel time in a torch-profiler trace by what the kernel is doing.

usage: python analyze_trace.py <trace.pt.trace.json[.gz]> [more traces...]
"""
import gzip, json, re, sys
from collections import defaultdict

BUCKETS = [
    ("fla / linear attention (triton)", re.compile(r"chunk_|gated_delta|fwd_prepare|bwd_prepare|recurrent|l2norm|causal_conv|triton_", re.I)),
    ("attention (unfused sdpa/softmax/bmm)", re.compile(r"softmax|bmm|baddbmm|attention|attn|flash|fmha|cudnn", re.I)),
    ("gemm (linear layers)", re.compile(r"gemm|cutlass|nvjet|sm90|sm100|sm103|xmma|cublas|wgmma|matmul|Kernel_", re.I)),
    ("nccl / tp comm", re.compile(r"nccl|ncclDev|AllGather|ReduceScatter|AllReduce|Broadcast", re.I)),
    ("layernorm / rmsnorm", re.compile(r"norm|LayerNorm|RMSNorm", re.I)),
    ("cross-entropy / logits", re.compile(r"log_softmax|logsumexp|cross_entropy|nll|gather_kernel|index_select", re.I)),
    ("elementwise / copy / cast", re.compile(r"elementwise|vectorized|copy_|cast|fill|add_|mul_|silu|gelu|sigmoid|rotary|rope|transpose|cat_|Memcpy|Memset|reduce_kernel|unrolled|CatArray", re.I)),
    ("optimizer", re.compile(r"adam|multi_tensor|fused_adam|lamb|sgd", re.I)),
]

def load(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as f:
        return json.load(f)

def main(paths):
    for path in paths:
        data = load(path)
        events = data["traceEvents"] if isinstance(data, dict) else data
        kern = [e for e in events if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset") and "dur" in e]
        if not kern:
            print(path, ": no GPU kernel events"); continue
        t0 = min(e["ts"] for e in kern); t1 = max(e["ts"] + e["dur"] for e in kern)
        wall = (t1 - t0) / 1e6
        busy = sum(e["dur"] for e in kern) / 1e6
        by_bucket = defaultdict(float); by_name = defaultdict(float)
        for e in kern:
            name = e["name"]; by_name[name] += e["dur"] / 1e6
            for label, rx in BUCKETS:
                if rx.search(name):
                    by_bucket[label] += e["dur"] / 1e6; break
            else:
                by_bucket["other"] += e["dur"] / 1e6
        print(f"\n== {path.split('/')[-1]}")
        print(f"profiled window {wall:.1f}s wall, GPU busy {busy:.1f}s ({100*busy/max(wall,1e-9):.0f}%), idle {wall-busy:.1f}s")
        print(f"{'bucket':<40}{'gpu s':>8}{'% busy':>8}")
        for label, t in sorted(by_bucket.items(), key=lambda kv: -kv[1]):
            print(f"{label:<40}{t:>8.1f}{100*t/busy:>7.0f}%")
        print("\ntop kernels:")
        for name, t in sorted(by_name.items(), key=lambda kv: -kv[1])[:15]:
            print(f"  {t:6.1f}s  {name[:110]}")
        # CPU-side: how much of the window has NO kernel running (launch-bound / python-bound)
        cpu = [e for e in events if e.get("cat") == "cpu_op" and "dur" in e]
        if cpu:
            top = defaultdict(float)
            for e in cpu: top[e["name"]] += e["dur"] / 1e6
            print("\ntop cpu ops (self+children, may overlap):")
            for name, t in sorted(top.items(), key=lambda kv: -kv[1])[:8]:
                print(f"  {t:6.1f}s  {name[:100]}")

if __name__ == "__main__":
    main(sys.argv[1:])
