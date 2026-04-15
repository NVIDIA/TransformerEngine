#!/usr/bin/env python3
"""
Benchmark all five top-k implementations using a batched timing methodology:
all iterations are submitted before a single sync, so per-iteration overhead
reflects pure kernel throughput with dispatch overhead amortised away.

Implementations:
  jax.lax.top_k   – JAX built-in (bfloat16)
  CUB TopK        – TransformerEngine JAX FFI (bfloat16)
  torch.topk      – PyTorch built-in (bfloat16)
  AIR TopK        – standalone_air_topk via air_topk_wrapper (bfloat16)
  topk_per_row    – histogram kernel via topk_per_row (float32)
"""

import argparse
import os
import sys
import time

_repo_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _repo_root)
sys.path.insert(0, os.path.join(_repo_root, "topk"))

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
    parser.add_argument("--n_list", type=str, required=True)
    parser.add_argument("--k_list", type=str, required=True)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=200)
    return parser.parse_args()


JAX_DTYPE_MAP = {"float16": jnp.float16, "bfloat16": jnp.bfloat16, "float32": jnp.float32}
TORCH_DTYPE_MAP = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}


# ---------------------------------------------------------------------------
# Benchmark helpers – batched methodology: submit all iters, sync once
# ---------------------------------------------------------------------------


def bench_jax_lax(x_jax, k, warmup, iters):
    f = jax.jit(lambda x: jax.lax.top_k(x, k))
    for _ in range(warmup):
        f(x_jax)[0].block_until_ready()
    t0 = time.perf_counter()
    for _ in range(iters):
        out = f(x_jax)
    out[0].block_until_ready()
    return (time.perf_counter() - t0) * 1e3 / iters


def bench_cub(x_jax, k, warmup, iters):
    f = jax.jit(lambda x: te_cub_topk(x, k))
    for _ in range(warmup):
        f(x_jax)[0].block_until_ready()
    t0 = time.perf_counter()
    for _ in range(iters):
        out = f(x_jax)
    out[0].block_until_ready()
    return (time.perf_counter() - t0) * 1e3 / iters


def bench_torch(x_torch, k, warmup, iters):
    for _ in range(warmup):
        torch.topk(x_torch, k)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        torch.topk(x_torch, k)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1e3 / iters


def bench_air(x_bf16_2d, lengths, buf, out_idx, k, warmup, iters):
    for _ in range(warmup):
        air_topk_wrapper.topk_kernel(x_bf16_2d, lengths, k, buf, out_idx, False)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        air_topk_wrapper.topk_kernel(x_bf16_2d, lengths, k, buf, out_idx, False)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1e3 / iters


def bench_per_row(x_f32, lengths, out_aux, logits_aux, out_idx, k, warmup, iters):
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
# Main
# ---------------------------------------------------------------------------


def main():
    args = parse_args()
    n_list = [int(x.strip()) for x in args.n_list.split(",")]
    k_list = [int(x.strip()) for x in args.k_list.split(",")]
    jax_dtype = JAX_DTYPE_MAP[args.dtype]
    torch_dtype = TORCH_DTYPE_MAP[args.dtype]

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}  |  JAX: {jax.__version__}")
    print(f"Warmup: {args.warmup}  |  Bench: {args.iterations} iters  |  Metric: mean ms (batched)")
    print(f"topk_per_row dtype: float32 (float32-only; input cast pre-benchmark)\n")

    col = [10, 10, 8, 16, 12, 12, 14, 18]
    header = (
        f"{'dtype':<{col[0]}} {'N':>{col[1]}} {'K':>{col[2]}}"
        f" {'jax.lax.top_k':>{col[3]}} {'torch.topk':>{col[4]}}"
        f" {'CUB TopK':>{col[5]}} {'AIR TopK':>{col[6]}} {'topk_per_row':>{col[7]}}"
    )
    sep = "-" * len(header)
    print(header)
    print(sep)

    rng = jax.random.PRNGKey(42)

    for N in n_list:
        for K in k_list:
            if K > N:
                continue

            x_jax = jax.random.uniform(rng, shape=(N,), dtype=jax_dtype)
            x_jax.block_until_ready()
            x_torch = torch.from_dlpack(x_jax).clone()

            x_bf16_2d = x_torch.unsqueeze(0).contiguous()
            x_f32 = x_torch.float().unsqueeze(0).contiguous()
            lengths = torch.tensor([N], dtype=torch.int32, device="cuda")
            out_aux, logits_aux, out_idx_pr = topk_per_row.allocate_buffers(x_f32, K)
            air_buf, out_idx_air = air_topk_wrapper.allocate_buffers(x_bf16_2d, K, False)

            t_jax = bench_jax_lax(x_jax, K, args.warmup, args.iterations)
            t_tor = bench_torch(x_torch, K, args.warmup, args.iterations)
            t_cub = bench_cub(x_jax, K, args.warmup, args.iterations)
            t_air = bench_air(
                x_bf16_2d, lengths, air_buf, out_idx_air, K, args.warmup, args.iterations
            )
            t_pr = bench_per_row(
                x_f32, lengths, out_aux, logits_aux, out_idx_pr, K, args.warmup, args.iterations
            )
            pr_str = f"{t_pr:>{col[7]}.4f}" if t_pr is not None else f"{'N/A':>{col[7]}}"

            print(
                f"{'bfloat16':<{col[0]}} {N:>{col[1]},} {K:>{col[2]},}"
                f" {t_jax:>{col[3]}.4f} {t_tor:>{col[4]}.4f}"
                f" {t_cub:>{col[5]}.4f} {t_air:>{col[6]}.4f} {pr_str}"
            )

    print(sep)
    print("All times in milliseconds (mean).")
    print("topk_per_row column uses float32 input; all others use bfloat16.")


if __name__ == "__main__":
    main()
