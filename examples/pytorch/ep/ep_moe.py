# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""End-to-end MoE example: dispatch -> batched expert linear -> combine, fwd + bwd.

One process per GPU; launched via run_test_ep.sh (torchrun).
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.distributed as dist

from transformer_engine.pytorch.ep import (
    EpBuffer,
    ep_bootstrap,
    ep_combine,
    ep_dispatch,
    ep_finalize,
)


def _parse_args():
    p = argparse.ArgumentParser(description="TE-PyTorch EP MoE example (fwd + bwd)")
    p.add_argument("--num-tokens", type=int, default=8, help="Per-rank token count.")
    p.add_argument("--top-k", type=int, default=2)
    p.add_argument("--hidden", type=int, default=32)
    p.add_argument("--hidden-out", type=int, default=32)
    p.add_argument("--num-experts", type=int, default=None)
    p.add_argument("--check", action="store_true", default=True)
    p.add_argument(
        "--benchmark",
        action="store_true",
        help="Time fwd over HBM buffers.",
    )
    p.add_argument("--benchmark-iters", type=int, default=20)
    p.add_argument("--benchmark-warmup", type=int, default=5)
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


def _make_routing(rank, T, K, E, num_local_experts):
    """Deterministic routing: topk_idx[t, k] = (rank*NLE + t*K + k) % E."""
    topk_idx = np.empty((T, K), dtype=np.int64)
    for t in range(T):
        for k in range(K):
            topk_idx[t, k] = (rank * num_local_experts + t * K + k) % E
    return topk_idx


def _batched_expert_linear(recv_tokens, kernels, num_local_experts):
    """Per-expert linear via bmm; ``recv_pr // num_local_experts`` slots per expert."""
    recv_pr, _H = recv_tokens.shape
    H_out = kernels.shape[-1]
    slots_per_expert = recv_pr // num_local_experts
    grouped = recv_tokens.view(num_local_experts, slots_per_expert, recv_tokens.shape[-1])
    out = torch.bmm(grouped, kernels.to(grouped.dtype))
    return out.view(recv_pr, H_out)


def _reference_moe(tokens, topk_idx, topk_w, kernels):
    T, K = topk_idx.shape
    H_out = kernels.shape[-1]
    out = np.zeros((T, H_out), dtype=np.float32)
    for t in range(T):
        tok = tokens[t].astype(np.float32)
        for k in range(K):
            e = int(topk_idx[t, k])
            out[t] += float(topk_w[t, k]) * (tok @ kernels[e].astype(np.float32))
    return out


def _reference_grad(tokens, topk_idx, topk_w, kernels):
    T, K = topk_idx.shape
    H = tokens.shape[-1]
    ref_out = _reference_moe(tokens, topk_idx, topk_w, kernels)
    grad = np.zeros((T, H), dtype=np.float32)
    for t in range(T):
        mixed = np.zeros_like(kernels[0])
        for k in range(K):
            mixed = mixed + float(topk_w[t, k]) * kernels[int(topk_idx[t, k])]
        grad[t] = ref_out[t] @ mixed.T
    return ref_out, grad


def main():
    args = _parse_args()

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", rank)))
    device = torch.device("cuda", torch.cuda.current_device())

    major, minor = torch.cuda.get_device_capability()
    if major * 10 + minor < 90:
        if rank == 0:
            print(f"[ep_moe] SKIPPED: EP requires SM>=90 (got SM{major}{minor})")
        dist.destroy_process_group()
        return

    if world_size < 4:
        if rank == 0:
            print(f"[ep_moe] SKIPPED: EP requires >= 4 ranks (got {world_size})")
        dist.destroy_process_group()
        return

    ep_size = world_size
    num_experts = args.num_experts if args.num_experts is not None else world_size
    assert num_experts % ep_size == 0
    num_local_experts = num_experts // ep_size
    T = args.num_tokens
    recv_pr = ep_size * T * args.top_k

    ep_group = dist.new_group(ranks=list(range(world_size)), backend="nccl")
    ep_bootstrap(
        ep_group,
        num_experts=num_experts,
        max_tokens_per_rank=T,
        recv_capacity_per_rank=recv_pr,
        hidden_dim=args.hidden,
    )
    try:
        _run_layer(
            args, rank, world_size, ep_size, num_experts, num_local_experts, T, recv_pr, device
        )
    finally:
        ep_finalize()
        dist.destroy_process_group()


def _run_layer(args, rank, world_size, ep_size, num_experts, num_local_experts, T, recv_pr, device):
    rng = np.random.default_rng(seed=42 + rank)
    tokens_np = (rng.standard_normal((T, args.hidden), dtype=np.float32) * 0.5).astype(np.float32)
    topk_idx_np = _make_routing(rank, T, args.top_k, num_experts, num_local_experts)
    w_np = np.full((T, args.top_k), 1.0 / args.top_k, dtype=np.float32)
    # Same seed across ranks -> identical kernel array everywhere.
    kr = np.random.default_rng(seed=42)
    kernels_np = (
        kr.standard_normal((num_experts, args.hidden, args.hidden_out), dtype=np.float32)
        * (1.0 / np.sqrt(args.hidden))
    ).astype(np.float32)

    tokens = (
        torch.from_numpy(tokens_np).to(device=device, dtype=torch.bfloat16).requires_grad_(True)
    )
    topk_idx = torch.from_numpy(topk_idx_np).to(device)
    topk_w = torch.from_numpy(w_np).to(device)
    kernels_local = torch.from_numpy(
        kernels_np[rank * num_local_experts : (rank + 1) * num_local_experts]
    ).to(device=device, dtype=torch.bfloat16)

    # Caller-supplied buffers (normal mode -> plain tensors), reused across iters.
    recv_tokens = (
        torch.empty(recv_pr, args.hidden, dtype=torch.bfloat16, device=device)
        if args.caller_provides_dispatch_recv_tokens
        else None
    )
    grad_expert_out = (
        torch.empty(recv_pr, args.hidden, dtype=torch.bfloat16, device=device)
        if args.caller_provides_grad_expert_out
        else None
    )

    buffer = EpBuffer(
        top_k=args.top_k,
        max_tokens_per_rank=T,
        recv_capacity_per_rank=recv_pr,
        hidden_dim=args.hidden,
        num_local_experts=num_local_experts,
        dispatch_recv_tokens=recv_tokens,
        combine_grad_expert_out=grad_expert_out,
    )

    recv_t, recv_w_out, _tc = ep_dispatch(buffer, tokens, topk_idx, topk_w)
    expert_out = _batched_expert_linear(recv_t, kernels_local, num_local_experts)
    # Apply per-slot topk weighting before combine.
    expert_out = expert_out * recv_w_out.unsqueeze(-1).to(expert_out.dtype)
    out = ep_combine(buffer, expert_out)

    loss = 0.5 * (out.float() ** 2).sum()
    loss.backward()
    torch.cuda.synchronize()

    if rank == 0:
        print(
            f"[ep_moe] loss={loss.item():.4f} grad_tokens.shape={tuple(tokens.grad.shape)} "
            f"ep={ep_size} num_experts={num_experts} recv_pr={recv_pr}"
        )

    if args.benchmark:
        # Time dispatch + expert + combine over HBM buffers.
        import time

        torch.cuda.synchronize()
        dist.barrier()
        for _ in range(args.benchmark_warmup):
            rt, rw, _tc = ep_dispatch(buffer, tokens.detach(), topk_idx, topk_w)
            expert_out = _batched_expert_linear(rt, kernels_local, num_local_experts)
            expert_out = expert_out * rw.unsqueeze(-1).to(expert_out.dtype)
            ep_combine(buffer, expert_out)
        torch.cuda.synchronize()
        dist.barrier()
        t0 = time.perf_counter()
        for _ in range(args.benchmark_iters):
            rt, rw, _tc = ep_dispatch(buffer, tokens.detach(), topk_idx, topk_w)
            expert_out = _batched_expert_linear(rt, kernels_local, num_local_experts)
            expert_out = expert_out * rw.unsqueeze(-1).to(expert_out.dtype)
            ep_combine(buffer, expert_out)
        torch.cuda.synchronize()
        dt_ms = (time.perf_counter() - t0) * 1000.0 / args.benchmark_iters
        if rank == 0:
            print(f"[ep_moe --benchmark] HBM: {dt_ms:.3f} ms/iter (iters={args.benchmark_iters})")

    if args.check:
        # All-gather inputs/outputs/grads for a global reference comparison.
        global_tokens = [torch.empty_like(tokens) for _ in range(world_size)]
        global_topk_idx = [torch.empty_like(topk_idx) for _ in range(world_size)]
        global_topk_w = [torch.empty_like(topk_w) for _ in range(world_size)]
        global_out = [torch.empty_like(out) for _ in range(world_size)]
        global_grad = [torch.empty_like(tokens.grad) for _ in range(world_size)]
        dist.all_gather(global_tokens, tokens.detach())
        dist.all_gather(global_topk_idx, topk_idx)
        dist.all_gather(global_topk_w, topk_w)
        dist.all_gather(global_out, out.detach())
        dist.all_gather(global_grad, tokens.grad)
        if rank == 0:
            all_tokens = torch.cat(global_tokens).float().cpu().numpy()
            all_idx = torch.cat(global_topk_idx).cpu().numpy()
            all_w = torch.cat(global_topk_w).cpu().numpy()
            all_out = torch.cat(global_out).float().cpu().numpy()
            all_grad = torch.cat(global_grad).float().cpu().numpy()
            ref_out, ref_grad = _reference_grad(all_tokens, all_idx, all_w, kernels_np)
            np.testing.assert_allclose(all_out, ref_out, rtol=5e-2, atol=5e-2)
            np.testing.assert_allclose(all_grad, ref_grad, rtol=5e-2, atol=5e-2)
            print(f"[ep_moe] --check PASSED (ref_out.sum()={float(ref_out.sum()):.4f})")


if __name__ == "__main__":
    main()
    sys.exit(0)
