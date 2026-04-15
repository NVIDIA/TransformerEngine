import transformer_engine.pytorch as te
import transformer_engine_torch as tex
from transformer_engine.pytorch import NVFP4Quantizer
import torch
import torch.cuda.nvtx as nvtx

N = 7168
num_experts = 64


def make_quantizer():
    q = NVFP4Quantizer(rowwise=True, columnwise=True, with_rht=True, with_post_rht_amax=True)
    q.optimize_for_gemm = True
    return q


def bench(fn, label, iters=100):
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    nvtx.range_push(label)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    nvtx.range_pop()
    torch.cuda.synchronize()
    print(f"{label}: {start.elapsed_time(end) / iters * 1000:.1f} us")


for M in [16384, 65536, 131072]:
    x = torch.randn(M, N, dtype=torch.bfloat16, device="cuda")

    # 1. graph-safe + equal splits -> O(1) division (SAME_BOTH_DIMS)
    equal_splits = [M // num_experts] * num_experts
    equal_tensor = torch.tensor(equal_splits, dtype=torch.int64, device="cuda")
    q1 = make_quantizer()
    bench(
        lambda: tex.group_quantize(x, q1, num_experts, equal_tensor), f"[M={M}] graph_safe_equal_O1"
    )

    # 2. graph-safe + unequal splits -> binary search (VARYING_FIRST_DIM)
    base = M // num_experts
    unequal_splits = [base - 128 if i % 2 == 0 else base + 128 for i in range(num_experts)]
    unequal_tensor = torch.tensor(unequal_splits, dtype=torch.int64, device="cuda")
    q2 = make_quantizer()
    bench(
        lambda: tex.group_quantize(x, q2, num_experts, unequal_tensor),
        f"[M={M}] graph_safe_unequal_binary_search",
    )

    # 3. non-graph-safe + linear scan (GetGroupIdx)
    q_list = [
        NVFP4Quantizer(rowwise=True, columnwise=True, with_rht=True, with_post_rht_amax=True)
        for _ in range(num_experts)
    ]
    bench(
        lambda: tex.split_quantize(x, equal_splits, q_list), f"[M={M}] non_graph_safe_linear_scan"
    )

    print()
