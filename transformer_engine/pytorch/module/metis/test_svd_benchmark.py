#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import torch

# 这里假设你把上面那一大坨实现放在同目录下的 lowrank_ops.py
# 如果文件名不同，把 lowrank_ops 改成对应的模块名即可
import lowrank_eig as lr


def get_torch_svd_lowrank():
    """兼容不同 PyTorch 版本，优先用 torch.svd_lowrank，没有就用 torch.linalg.svd_lowrank。"""
    if hasattr(torch, "svd_lowrank"):
        return torch.svd_lowrank
    else:
        from torch.linalg import svd_lowrank

        return svd_lowrank


def time_fn(label, fn, *args, warmup=2, runs=5, device="cuda"):
    """简单计时函数：先 warmup，再跑 runs 次取平均时间（毫秒）"""
    print(f"\n==== {label} ====")

    # 预热
    for _ in range(warmup):
        fn(*args)
        if device == "cuda":
            torch.cuda.synchronize()

    times = []
    for _ in range(runs):
        if device == "cuda":
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            fn(*args)
            end.record()
            torch.cuda.synchronize()
            elapsed_ms = start.elapsed_time(end)
        else:
            t0 = time.perf_counter()
            fn(*args)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
        times.append(elapsed_ms)

    avg_ms = sum(times) / len(times)
    print(f"{label}: mean = {avg_ms:.3f} ms, min = {min(times):.3f} ms, max = {max(times):.3f} ms")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ----------------- 构造 (512, 2048, 2048) 然后前两维合并 -----------------
    B, M, N = 1, 2048, 2048  # 原始三维
    rank = 32  # 低秩 q / rank
    dtype = torch.float32

    # A3: (512, 2048, 2048)
    A3 = torch.randn(B, M, N, device=device, dtype=dtype)
    # 合并前两维：A: (512*2048, 2048)
    A = A3.reshape(-1, N).contiguous()
    print(f"Input 3D shape: {tuple(A3.shape)}, merged 2D shape: {tuple(A.shape)}")

    torch_svd_lowrank = get_torch_svd_lowrank()

    # ----------------- 定义各个待测函数 -----------------

    def fn_torch_svd_lowrank():
        # 原生 torch 版本
        U, S, V = torch_svd_lowrank(A, q=rank, niter=0)
        return U, S, V

    def fn_svd_lowrank_eig():
        # 你的 eig 版（非 graph）
        U, S, V = lr.svd_lowrank_eig(A, q=rank, niter=1)
        return U, S, V

    def fn_svd_lowrank_eig_graph():
        # 你的 CUDA graph 版
        U, S, V = lr.svd_lowrank_eig_graph(A, q=rank, niter=1)
        return U, S, V

    def fn_svd_lowrank_eig_graph_pipelined():
        # 你的 CUDA graph + pipelined CPU eig 版
        U, S, V = lr.svd_lowrank_eig_graph_pipelined(A, q=rank, niter=1)
        return U, S, V

    # ----------------- 先各跑一次，触发 graph capture / engine 缓存 -----------------
    print("\nWarmup single call for each method (graph capture, kernel init, etc.) ...")
    fn_torch_svd_lowrank()
    if device.type == "cuda":
        fn_svd_lowrank_eig()
        fn_svd_lowrank_eig_graph()
        fn_svd_lowrank_eig_graph_pipelined()
        torch.cuda.synchronize()

    # ----------------- 正式计时 -----------------
    dev_str = "cuda" if device.type == "cuda" else "cpu"

    time_fn("torch.svd_lowrank", fn_torch_svd_lowrank, device=dev_str)
    time_fn("svd_lowrank_eig", fn_svd_lowrank_eig, device=dev_str)

    if device.type == "cuda":
        time_fn("svd_lowrank_eig_graph", fn_svd_lowrank_eig_graph, device=dev_str)
        time_fn(
            "svd_lowrank_eig_graph_pipelined", fn_svd_lowrank_eig_graph_pipelined, device=dev_str
        )
    else:
        print("\n[Skip] CUDA graph variants require GPU (device != cuda).")


if __name__ == "__main__":
    main()
