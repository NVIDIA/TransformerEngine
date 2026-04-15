#!/usr/bin/env python3
"""
Benchmark all five top-k implementations using a batched timing methodology:
all iterations are submitted before a single sync, so per-iteration overhead
reflects pure kernel throughput with dispatch overhead amortised away.

Implementations:
  jax.lax.top_k   – JAX built-in
  CUB TopK        – TransformerEngine JAX FFI
  torch.topk      – PyTorch built-in
  AIR TopK        – standalone_air_topk via air_topk_wrapper
  topk_per_row    – histogram kernel via topk_per_row (float32 only)

Modes:
  1D  – input shape (N,),        one top-k per call
  2D  – input shape (bs, seqlen), top-k per row; enabled with --bs_list
"""

import argparse
import os
import sys
import time

_repo_root = os.path.dirname(os.path.abspath(__file__))
# topk/ first (air_topk_wrapper, topk_per_row), then repo root for the
# PR-branch transformer_engine that has the CUB topk JAX FFI.
sys.path.insert(0, os.path.join(_repo_root, "topk"))
sys.path.insert(1, _repo_root)

import jax
import jax.numpy as jnp
from jax import random
import torch
import numpy as np

from transformer_engine.jax.cpp_extensions.cub import topk as te_cub_topk
import air_topk_wrapper
import topk_per_row


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batched top-k benchmark across all implementations",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument(
        "--n_list", type=str, required=True, help="Comma-separated N values (seqlen for 2D)"
    )
    parser.add_argument("--k_list", type=str, required=True, help="Comma-separated K values")
    parser.add_argument(
        "--bs_list",
        type=str,
        default=None,
        help="Comma-separated batch sizes; enables 2D benchmark",
    )
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=200)
    return parser.parse_args()


JAX_DTYPE_MAP = {"float16": jnp.float16, "bfloat16": jnp.bfloat16, "float32": jnp.float32}
TORCH_DTYPE_MAP = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}


# ---------------------------------------------------------------------------
# Benchmark helpers – batched methodology: submit all iters, sync once
# ---------------------------------------------------------------------------


def bench_jax_lax(x_jax, k, warmup, iters):
    """Works for both 1D (N,) and 2D (bs, seqlen) – lax.top_k operates on last axis."""
    f = jax.jit(lambda x: jax.lax.top_k(x, k))
    for _ in range(warmup):
        f(x_jax)[0].block_until_ready()
    t0 = time.perf_counter()
    for _ in range(iters):
        out = f(x_jax)
    out[0].block_until_ready()
    return (time.perf_counter() - t0) * 1e3 / iters


def bench_cub(x_jax, k, warmup, iters):
    """1D only."""
    f = jax.jit(lambda x: te_cub_topk(x, k))
    for _ in range(warmup):
        f(x_jax)[0].block_until_ready()
    t0 = time.perf_counter()
    for _ in range(iters):
        out = f(x_jax)
    out[0].block_until_ready()
    return (time.perf_counter() - t0) * 1e3 / iters


def bench_cub_2d(x_jax, k, warmup, iters):
    """2D (bs, seqlen): CUB TopK with native batched dispatch (one call per row)."""
    f = jax.jit(lambda x: te_cub_topk(x, k))
    try:
        for _ in range(warmup):
            f(x_jax)[0].block_until_ready()
    except Exception:
        return None
    t0 = time.perf_counter()
    for _ in range(iters):
        out = f(x_jax)
    out[0].block_until_ready()
    return (time.perf_counter() - t0) * 1e3 / iters


def bench_torch(x_torch, k, warmup, iters):
    """Works for both 1D and 2D – topk on last axis."""
    for _ in range(warmup):
        torch.topk(x_torch, k, dim=-1)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        torch.topk(x_torch, k, dim=-1)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1e3 / iters


def bench_air(x_2d, lengths, buf, out_idx, k, warmup, iters):
    """Works for both [1, N] and [bs, seqlen]."""
    for _ in range(warmup):
        air_topk_wrapper.topk_kernel(x_2d, lengths, k, buf, out_idx, False)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        air_topk_wrapper.topk_kernel(x_2d, lengths, k, buf, out_idx, False)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1e3 / iters


def bench_per_row(x_f32, lengths, out_aux, logits_aux, out_idx, k, warmup, iters):
    """Works for both [1, N] and [bs, seqlen]."""
    try:
        topk_per_row.topk_kernel(x_f32, lengths, k, out_aux, logits_aux, out_idx, False)
        torch.cuda.synchronize()
    except RuntimeError:
        torch.cuda.synchronize()
        return None
    for _ in range(warmup - 1):
        topk_per_row.topk_kernel(x_f32, lengths, k, out_aux, logits_aux, out_idx, False)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        topk_per_row.topk_kernel(x_f32, lengths, k, out_aux, logits_aux, out_idx, False)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1e3 / iters


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def fmt(val, width):
    return f"{val:{width}.4f}" if val is not None else f"{'N/A':>{width}}"


def run_1d(n_list, k_list, jax_dtype, args, col, sep):
    print(
        "── 1D"
        " ──────────────────────────────────────────────────────────────────────────────────────────────────────"
    )
    print(
        f"{'dtype':<{col[0]}} {'N':>{col[1]}} {'K':>{col[2]}}"
        f" {'jax.lax.top_k':>{col[3]}} {'torch.topk':>{col[4]}}"
        f" {'CUB TopK':>{col[5]}} {'AIR TopK':>{col[6]}} {'topk_per_row':>{col[7]}}"
    )
    print(sep)

    rng = jax.random.PRNGKey(42)
    for N in n_list:
        for K in k_list:
            if K > N:
                continue
            x_jax = jax.random.uniform(rng, shape=(N,), dtype=jax_dtype)
            x_jax.block_until_ready()
            x_torch = torch.from_dlpack(x_jax).clone()
            x_2d = x_torch.unsqueeze(0).contiguous()
            x_f32 = x_torch.float().unsqueeze(0).contiguous()
            lengths = torch.tensor([N], dtype=torch.int32, device="cuda")
            out_aux, logits_aux, out_idx_pr = topk_per_row.allocate_buffers(x_f32, K)
            air_buf, out_idx_air = air_topk_wrapper.allocate_buffers(x_2d, K, False)

            t_jax = bench_jax_lax(x_jax, K, args.warmup, args.iterations)
            t_tor = bench_torch(x_torch, K, args.warmup, args.iterations)
            t_cub = bench_cub(x_jax, K, args.warmup, args.iterations)
            t_air = bench_air(x_2d, lengths, air_buf, out_idx_air, K, args.warmup, args.iterations)
            t_pr = bench_per_row(
                x_f32, lengths, out_aux, logits_aux, out_idx_pr, K, args.warmup, args.iterations
            )

            print(
                f"{args.dtype:<{col[0]}} {N:>{col[1]},} {K:>{col[2]},}"
                f" {fmt(t_jax, col[3])} {fmt(t_tor, col[4])}"
                f" {fmt(t_cub, col[5])} {fmt(t_air, col[6])} {fmt(t_pr, col[7])}"
            )
    print(sep)


def run_2d(n_list, k_list, bs_list, jax_dtype, args, sep):
    col = [10, 6, 10, 8, 16, 12, 12, 14, 18]
    print(
        "\n── 2D (top-k per row)"
        " ──────────────────────────────────────────────────────────────────────────────────────"
    )
    print(
        f"{'dtype':<{col[0]}} {'bs':>{col[1]}} {'seqlen':>{col[2]}} {'K':>{col[3]}}"
        f" {'jax.lax.top_k':>{col[4]}} {'torch.topk':>{col[5]}}"
        f" {'CUB TopK':>{col[6]}} {'AIR TopK':>{col[7]}} {'topk_per_row':>{col[8]}}"
    )
    sep2d = "-" * (sum(col) + len(col))
    print(sep2d)

    rng = jax.random.PRNGKey(42)
    for BS in bs_list:
        for seqlen in n_list:
            for K in k_list:
                if K > seqlen:
                    continue
                x_jax = jax.random.uniform(rng, shape=(BS, seqlen), dtype=jax_dtype)
                x_jax.block_until_ready()
                x_torch = torch.from_dlpack(x_jax).clone()  # (bs, seqlen)
                x_f32 = x_torch.float().contiguous()
                lengths = torch.full((BS,), seqlen, dtype=torch.int32, device="cuda")
                out_aux, logits_aux, out_idx_pr = topk_per_row.allocate_buffers(x_f32, K)
                air_buf, out_idx_air = air_topk_wrapper.allocate_buffers(x_torch, K, False)

                t_jax = bench_jax_lax(x_jax, K, args.warmup, args.iterations)
                t_tor = bench_torch(x_torch, K, args.warmup, args.iterations)
                t_cub = bench_cub_2d(x_jax, K, args.warmup, args.iterations)
                t_air = bench_air(
                    x_torch, lengths, air_buf, out_idx_air, K, args.warmup, args.iterations
                )
                t_pr = bench_per_row(
                    x_f32, lengths, out_aux, logits_aux, out_idx_pr, K, args.warmup, args.iterations
                )

                print(
                    f"{args.dtype:<{col[0]}} {BS:>{col[1]}} {seqlen:>{col[2]},} {K:>{col[3]},}"
                    f" {fmt(t_jax, col[4])} {fmt(t_tor, col[5])}"
                    f" {fmt(t_cub, col[6])} {fmt(t_air, col[7])} {fmt(t_pr, col[8])}"
                )
    print(sep2d)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = parse_args()
    n_list = [int(x.strip()) for x in args.n_list.split(",")]
    k_list = [int(x.strip()) for x in args.k_list.split(",")]
    bs_list = [int(x.strip()) for x in args.bs_list.split(",")] if args.bs_list else None
    jax_dtype = JAX_DTYPE_MAP[args.dtype]

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}  |  JAX: {jax.__version__}")
    print(f"Warmup: {args.warmup}  |  Bench: {args.iterations} iters  |  Metric: mean ms (batched)")
    pr_note = "" if args.dtype == "float32" else " (float32-only; input cast pre-benchmark)"
    print(f"topk_per_row dtype: float32{pr_note}\n")

    col = [10, 10, 8, 16, 12, 12, 14, 18]
    sep = "-" * (sum(col) + len(col))

    run_1d(n_list, k_list, jax_dtype, args, col, sep)

    if bs_list:
        run_2d(n_list, k_list, bs_list, jax_dtype, args, sep)

    print("\nAll times in milliseconds (mean).")
    if args.dtype != "float32":
        print(f"topk_per_row column uses float32 input; all others use {args.dtype}.")


if __name__ == "__main__":
    main()
