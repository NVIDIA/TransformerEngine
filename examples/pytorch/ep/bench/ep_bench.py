# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""PyTorch EP perf bench: raw and autograd dispatch/combine on a single EP group.

One process per GPU; launched via run_ep_bench.sh (torchrun).

Stages (each timed in its own loop):
  - dispatch_raw:        _ep_dispatch_raw (no autograd, no prepare)
  - ep_dispatch_fwd:     ep_dispatch forward only
  - ep_dispatch_fwd_bwd: ep_dispatch + backward on 0.5 * ||recv||^2
  - combine_raw:         _ep_combine_raw (no autograd)
  - ep_combine_fwd:      ep_combine forward only
  - ep_combine_fwd_bwd:  ep_combine + backward

ep_prepare runs once outside the timed loops. --kineto DIR dumps a Chrome
trace plus a per-kernel summary on rank 0.
"""

import argparse
import gc
import os
import sys
import time
from contextlib import nullcontext

import numpy as np
import torch
import torch.distributed as dist

from transformer_engine.pytorch.ep import (
    EpBuffer,
    ep_bootstrap,
    ep_combine,
    ep_dispatch,
    ep_finalize,
    ep_prepare,
    _ep_combine_raw,
    _ep_dispatch_raw,
)


def _parse_args():
    p = argparse.ArgumentParser(description="TE-PyTorch EP perf bench")
    p.add_argument("--tokens-per-rank", type=int, default=8192)
    p.add_argument("--hidden", type=int, default=7168)
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--num-experts", type=int, default=256)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--iters", type=int, default=10)
    p.add_argument(
        "--max-num-sms",
        type=int,
        default=0,
        help="Max SMs for dispatch/combine/preprocess kernels (0 = auto).",
    )
    p.add_argument(
        "--kineto",
        default=None,
        help="If set, dump a Kineto Chrome trace + per-kernel summary into this dir (rank 0).",
    )
    p.add_argument(
        "--cuda-graph",
        action="store_true",
        default=False,
        help=(
            "Capture each stage into a CUDA graph and time replay() instead of the eager call. "
            "Raw + fwd-only stages use torch.cuda.graph; fwd+bwd stages use "
            "torch.cuda.make_graphed_callables to capture forward and backward together."
        ),
    )
    p.add_argument(
        "--mode-label",
        default=None,
        help="Optional suffix for NVTX range names (e.g. 'fused' / 'unfused').",
    )
    p.add_argument(
        "--caller-provides-dispatch-recv-tokens",
        action="store_true",
        default=False,
        help="Supply recv_tokens to ep_dispatch instead of letting EpBuffer own it.",
    )
    p.add_argument(
        "--caller-provides-grad-expert-out",
        action="store_true",
        default=False,
        help="Supply the combine backward grad buffer to ep_combine.",
    )
    return p.parse_args()


def _nvtx_funcs():
    """Return push/pop helpers using torch.cuda.nvtx if available."""
    try:
        push = torch.cuda.nvtx.range_push
        pop = torch.cuda.nvtx.range_pop
        return push, pop
    except AttributeError:
        return lambda _name: None, lambda: None


def _device_sm() -> int:
    major, minor = torch.cuda.get_device_capability()
    return major * 10 + minor


def _make_inputs(rank, world_size, T, H, K, E, device):
    """Round-robin identity routing + uniform top-k weights."""
    topk_idx = np.empty((T, K), dtype=np.int64)
    for t in range(T):
        for k in range(K):
            topk_idx[t, k] = ((rank * T + t) * K + k) % E
    rng = np.random.default_rng(seed=42 + rank)
    tokens_np = (rng.standard_normal((T, H), dtype=np.float32) * 0.5).astype(np.float32)
    return (
        torch.from_numpy(topk_idx).to(device),
        torch.from_numpy(tokens_np).to(device=device, dtype=torch.bfloat16),
        torch.full((T, K), 1.0 / K, dtype=torch.float32, device=device),
    )


def _time_stage_us(name, fn, iters, nvtx_suffix, push, pop):
    """Time fn for iters iterations after one untimed warmup; returns mean us."""
    # Run iters+1 times; drop the first (autotune outlier) and frame NVTX from iter 1.
    total_ns = 0
    counted = 0
    for i in range(iters + 1):
        if i == 1:
            push(f"{name}{nvtx_suffix}")
        torch.cuda.synchronize()
        t0 = time.perf_counter_ns()
        fn()
        torch.cuda.synchronize()
        dt = time.perf_counter_ns() - t0
        if i == 0:
            continue
        total_ns += dt
        counted += 1
    pop()
    return total_ns / 1e3 / counted


def main():
    args = _parse_args()
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", rank)))
    device = torch.device("cuda", torch.cuda.current_device())

    if _device_sm() < 90:
        if rank == 0:
            print(f"[ep_bench] SKIPPED: EP requires SM>=90 (got SM{_device_sm()})")
        dist.destroy_process_group()
        return
    if world_size < 4:
        if rank == 0:
            print(f"[ep_bench] SKIPPED: EP requires >=4 ranks (got {world_size})")
        dist.destroy_process_group()
        return

    ep_size = world_size
    E = args.num_experts
    assert E % ep_size == 0, f"num_experts ({E}) must be divisible by ep_size ({ep_size})"
    num_local_experts = E // ep_size
    T = args.tokens_per_rank
    H = args.hidden
    K = args.top_k
    # Conservative cap: every token could land on every local expert.
    recv_pr = world_size * T * K // 2
    if rank == 0:
        print(
            f"[ep_bench] world={world_size} ep={ep_size} T={T} H={H} K={K} "
            f"E={E} (local={num_local_experts}) recv_pr={recv_pr}"
            + (f" mode={args.mode_label}" if args.mode_label else ""),
            flush=True,
        )

    ep_group = dist.new_group(ranks=list(range(world_size)), backend="nccl")
    ep_bootstrap(
        ep_group,
        num_experts=E,
        max_tokens_per_rank=T,
        recv_capacity_per_rank=recv_pr,
        hidden_dim=H,
        max_num_sms=args.max_num_sms,
    )

    topk_idx, tokens_hbm, topk_w_hbm = _make_inputs(rank, world_size, T, H, K, E, device)

    # Caller-supplied buffers for the autograd ep_dispatch/ep_combine stages
    # (normal mode -> plain tensors), reused across iters. None when not opted in.
    caller_recv_tokens = (
        torch.empty(recv_pr, H, dtype=torch.bfloat16, device=device)
        if args.caller_provides_dispatch_recv_tokens
        else None
    )
    caller_grad_expert_out = (
        torch.empty(recv_pr, H, dtype=torch.bfloat16, device=device)
        if args.caller_provides_grad_expert_out
        else None
    )

    buffer = EpBuffer(
        top_k=K,
        max_tokens_per_rank=T,
        recv_capacity_per_rank=recv_pr,
        hidden_dim=H,
        num_local_experts=num_local_experts,
        dispatch_recv_tokens=caller_recv_tokens,
        combine_grad_expert_out=caller_grad_expert_out,
    )

    tokens = tokens_hbm
    topk_w = topk_w_hbm
    recv_tokens = torch.empty(recv_pr, H, dtype=torch.bfloat16, device=device)
    recv_w = torch.empty(recv_pr, dtype=torch.float32, device=device)

    # -- Prepare once outside the timed loops ------------------------------
    ep_prepare(buffer, topk_idx)
    torch.cuda.synchronize()

    # Pre-dispatch a steady recv_tokens / recv_w so combine stages have valid input.
    _ep_dispatch_raw(buffer, topk_idx, tokens, topk_w, recv_tokens, recv_w)
    torch.cuda.synchronize()
    # fp-equivalent stand-in for an MLP output.
    expert_out = recv_tokens.clone()

    nvtx_suffix = f"[{args.mode_label}]" if args.mode_label else ""
    push, pop = _nvtx_funcs()

    # -- Stage closures ----------------------------------------------------
    # Persistent fwd+bwd inputs (make_graphed_callables needs stable storage).
    tokens_p = tokens.detach().clone().requires_grad_(True)
    eo_p = recv_tokens.detach().clone().requires_grad_(True)

    # Stand-in callables; the cuda-graph branch below swaps in graphed versions.
    fwd_bwd_dispatch_fn = lambda x: ep_dispatch(buffer, x, topk_idx, topk_w)[0]  # noqa: E731
    fwd_bwd_combine_fn = lambda expert_out: ep_combine(buffer, expert_out)  # noqa: E731

    def _dispatch_raw():
        _ep_dispatch_raw(buffer, topk_idx, tokens, topk_w, recv_tokens, recv_w)

    def _combine_raw():
        out_buf = torch.empty(T, H, dtype=torch.bfloat16, device=device)
        _ep_combine_raw(buffer, expert_out, out_buf)

    def _ep_dispatch_fwd():
        ep_dispatch(buffer, tokens.detach(), topk_idx, topk_w)

    def _ep_dispatch_fwd_bwd():
        tokens_p.grad = None
        r = fwd_bwd_dispatch_fn(tokens_p)
        (0.5 * (r * r).sum(dtype=torch.float32)).backward()

    def _ep_combine_fwd():
        ep_combine(buffer, recv_tokens)

    def _ep_combine_fwd_bwd():
        eo_p.grad = None
        out = fwd_bwd_combine_fn(eo_p)
        (0.5 * (out * out).sum(dtype=torch.float32)).backward()

    stages = [
        ("dispatch_raw", _dispatch_raw, True),
        ("ep_dispatch_fwd", _ep_dispatch_fwd, True),
        ("ep_dispatch_fwd_bwd", _ep_dispatch_fwd_bwd, False),
        ("combine_raw", _combine_raw, True),
        ("ep_combine_fwd", _ep_combine_fwd, True),
        ("ep_combine_fwd_bwd", _ep_combine_fwd_bwd, False),
    ]
    # Third tuple element: True = direct torch.cuda.graph capture; False = use
    # make_graphed_callables (autograd-aware) instead.

    # -- Warmup -----------------------------------------------------------
    for _ in range(args.warmup):
        for _name, fn, _capt in stages:
            fn()
    torch.cuda.synchronize()

    # -- Optional CUDA-graph capture --------------------------------------
    # Capture each capturable stage on a side stream and time .replay()
    # instead of the eager call. Outputs allocated inside the
    # autograd.Function's forward go through the per-capture private pool
    # so addresses stay stable across replays.
    captured_runners = {}
    if args.cuda_graph:
        # Graph fwd+bwd of the autograd-wrapped ops via make_graphed_callables.
        class _DispatchMod(torch.nn.Module):
            def forward(self, x):
                return ep_dispatch(buffer, x, topk_idx, topk_w)[0]

        class _CombineMod(torch.nn.Module):
            def forward(self, expert_out):
                return ep_combine(buffer, expert_out)

        disp_mod = _DispatchMod().cuda()
        comb_mod = _CombineMod().cuda()
        g_disp, g_comb = torch.cuda.make_graphed_callables(
            (disp_mod, comb_mod),
            ((tokens_p,), (eo_p,)),
        )
        fwd_bwd_dispatch_fn = g_disp
        fwd_bwd_combine_fn = g_comb

        # Direct torch.cuda.graph capture for raw + fwd-only stages.
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for name, fn, direct_capturable in stages:
                if not direct_capturable:
                    continue
                fn()  # prime the allocator for stable replay addresses
                torch.cuda.synchronize()
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    fn()
                captured_runners[name] = g
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()

    # -- Optional Kineto profiling ----------------------------------------
    kineto_ctx = nullcontext()
    if args.kineto and rank == 0:
        os.makedirs(args.kineto, exist_ok=True)
        kineto_ctx = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=False,
            with_stack=False,
        )

    # -- Timed loops ------------------------------------------------------
    results = {}
    with kineto_ctx as prof:
        for name, fn, _ in stages:
            runner = fn
            if name in captured_runners:
                # Time replay() instead of the eager call.
                graph = captured_runners[name]
                runner = graph.replay
            results[name] = _time_stage_us(name, runner, args.iters, nvtx_suffix, push, pop)

    if rank == 0:
        label = f" [{args.mode_label}]" if args.mode_label else ""
        print("", flush=True)
        print(f"| stage                | mean wall (us){label} |", flush=True)
        print("|----------------------|---------------:|", flush=True)
        for name in (
            "dispatch_raw",
            "ep_dispatch_fwd",
            "ep_dispatch_fwd_bwd",
            "combine_raw",
            "ep_combine_fwd",
            "ep_combine_fwd_bwd",
        ):
            print(f"| {name:20s} | {results[name]:14.1f} |", flush=True)
        print(
            "| (dispatch fwd-raw)   |"
            f" {results['ep_dispatch_fwd'] - results['dispatch_raw']:14.1f} |",
            flush=True,
        )
        print(
            "| (dispatch bwd-fwd)   |"
            f" {results['ep_dispatch_fwd_bwd'] - results['ep_dispatch_fwd']:14.1f} |",
            flush=True,
        )
        print(
            "| (combine fwd-raw)    |"
            f" {results['ep_combine_fwd'] - results['combine_raw']:14.1f} |",
            flush=True,
        )
        print(
            "| (combine bwd-fwd)    |"
            f" {results['ep_combine_fwd_bwd'] - results['ep_combine_fwd']:14.1f} |",
            flush=True,
        )
        print("", flush=True)

    if args.kineto and rank == 0 and prof is not None:
        trace_path = os.path.join(args.kineto, "ep_bench_trace.json")
        prof.export_chrome_trace(trace_path)
        print(f"[ep_bench] kineto trace: {trace_path}", flush=True)
        print(
            prof.key_averages().table(sort_by="cuda_time_total", row_limit=30),
            flush=True,
        )
        kern_csv = os.path.join(args.kineto, "ep_bench_kernels.csv")
        with open(kern_csv, "w") as f:
            f.write("name,cuda_time_us,cpu_time_us,count\n")
            for evt in prof.key_averages():
                if evt.device_time_total == 0 and evt.cpu_time_total == 0:
                    continue
                f.write(f"{evt.key},{evt.device_time_total},{evt.cpu_time_total},{evt.count}\n")
        print(f"[ep_bench] per-kernel CSV: {kern_csv}", flush=True)

    # Captured CUDA graphs (when --cuda-graph) hold references to NCCL EP
    # handles and per-pool streams; drop them and sync before ep_finalize,
    # otherwise the post-finalize dist.barrier can deadlock against pending
    # graph state.
    torch.cuda.synchronize()
    if args.cuda_graph:
        fwd_bwd_dispatch_fn = None
        fwd_bwd_combine_fn = None
        captured_runners.clear()
        del g_disp, g_comb, disp_mod, comb_mod
    del tokens_p, eo_p, buffer, recv_tokens, recv_w, tokens, topk_w, expert_out
    gc.collect()
    torch.cuda.synchronize()
    # Release NCCL EP's borrowed comm before torch destroys it.
    ep_finalize()
    dist.barrier()
    dist.destroy_process_group()
    sys.stdout.flush()
    sys.stderr.flush()


if __name__ == "__main__":
    main()
