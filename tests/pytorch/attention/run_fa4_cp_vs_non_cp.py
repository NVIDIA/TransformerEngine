# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Compare FlashAttention v4 context parallelism against full-sequence attention.

Example:
    torchrun --nproc_per_node=2 tests/pytorch/attention/run_fa4_cp_vs_non_cp.py
"""

import argparse
import os

import torch
import torch.distributed as dist


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--attn-mask-type", choices=["causal", "no_mask"], default="causal")
    parser.add_argument("--cp-comm-type", choices=["p2p"], default="p2p")
    parser.add_argument("--atol", type=float, default=5.0e-2)
    parser.add_argument("--rtol", type=float, default=5.0e-2)
    parser.add_argument("--rmse-tol", type=float, default=2.5e-2)
    return parser.parse_args()


def _cp_slice(tensor, rank, world_size, seq_dim):
    """Select the two load-balanced sequence chunks owned by a CP rank."""
    chunked = tensor.view(
        *tensor.shape[:seq_dim],
        2 * world_size,
        tensor.shape[seq_dim] // (2 * world_size),
        *tensor.shape[(seq_dim + 1) :],
    )
    chunk_ids = torch.tensor([rank, 2 * world_size - rank - 1], device=tensor.device)
    local = chunked.index_select(seq_dim, chunk_ids)
    return local.reshape(
        *local.shape[:seq_dim],
        -1,
        *local.shape[(seq_dim + 2) :],
    ).contiguous()


def _metric(actual, expected):
    diff = (actual.float() - expected.float()).abs()
    max_abs = diff.max()
    rmse = torch.sqrt(torch.mean(diff * diff))
    return max_abs, rmse


def _assert_close(name, actual, expected, atol, rtol, rmse_tol):
    max_abs, rmse = _metric(actual, expected)
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    if rmse.item() > rmse_tol:
        raise AssertionError(f"{name} RMSE {rmse.item():.6g} exceeds {rmse_tol:.6g}")
    return max_abs.detach(), rmse.detach()


def _make_attention(num_heads, head_dim, qkv_format, attn_mask_type):
    from transformer_engine.pytorch import DotProductAttention

    return DotProductAttention(
        num_heads,
        head_dim,
        num_gqa_groups=num_heads,
        attention_dropout=0.0,
        qkv_format=qkv_format,
        attn_mask_type=attn_mask_type,
    ).cuda()


def _force_fa4():
    os.environ["NVTE_FLASH_ATTN"] = "1"
    os.environ["NVTE_FUSED_ATTN"] = "0"
    os.environ["NVTE_ALLOW_NONDETERMINISTIC_ALGO"] = "1"

    from transformer_engine.pytorch.attention.dot_product_attention.backends import (
        _flash_attn_bwd_v4,
        _flash_attn_fwd_v4,
    )
    from transformer_engine.pytorch.attention.dot_product_attention.utils import FlashAttentionUtils

    if (
        not FlashAttentionUtils.v4_is_installed
        or _flash_attn_fwd_v4 is None
        or _flash_attn_bwd_v4 is None
    ):
        raise RuntimeError(
            "FlashAttention v4 with raw cute _flash_attn_fwd/_flash_attn_bwd APIs is required."
        )

    # Keep the backend selector on FA4 even when FA2/FA3 are also installed.
    FlashAttentionUtils.is_installed = False
    FlashAttentionUtils.v3_is_installed = False


def main():
    args = _parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    _force_fa4()

    if args.seq_len % (2 * world_size) != 0:
        raise ValueError("--seq-len must be divisible by 2 * WORLD_SIZE")
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[args.dtype]
    qkv_format = "bshd"
    seq_dim = qkv_format.index("s")
    device = torch.device("cuda", local_rank)

    torch.manual_seed(1234)
    q_full = torch.randn(
        args.batch_size, args.seq_len, args.num_heads, args.head_dim, device=device, dtype=dtype
    ).clamp_(-1, 1)
    k_full = torch.randn_like(q_full).clamp_(-1, 1)
    v_full = torch.randn_like(q_full).clamp_(-1, 1)
    dout_full = torch.randn(
        args.batch_size,
        args.seq_len,
        args.num_heads * args.head_dim,
        device=device,
        dtype=dtype,
    ).clamp_(-1, 1)

    q_ref, k_ref, v_ref = [x.detach().clone().requires_grad_() for x in (q_full, k_full, v_full)]
    attn_ref = _make_attention(args.num_heads, args.head_dim, qkv_format, args.attn_mask_type)
    out_ref = attn_ref(q_ref, k_ref, v_ref)
    out_ref.backward(dout_full)

    q_cp, k_cp, v_cp = [
        _cp_slice(x.detach(), rank, world_size, seq_dim).requires_grad_()
        for x in (q_full, k_full, v_full)
    ]
    dout_cp = _cp_slice(dout_full, rank, world_size, seq_dim)

    cp_group = dist.new_group(list(range(world_size)), backend="nccl")
    cp_ranks = list(range(world_size))
    attn_cp = _make_attention(args.num_heads, args.head_dim, qkv_format, args.attn_mask_type)
    attn_cp.set_context_parallel_group(cp_group, cp_ranks, torch.cuda.Stream(), args.cp_comm_type)
    out_cp = attn_cp(q_cp, k_cp, v_cp)
    out_cp.backward(dout_cp)

    torch.cuda.synchronize()

    checks = {
        "out": (out_cp.detach(), _cp_slice(out_ref.detach(), rank, world_size, seq_dim)),
        "dq": (q_cp.grad.detach(), _cp_slice(q_ref.grad.detach(), rank, world_size, seq_dim)),
        "dk": (k_cp.grad.detach(), _cp_slice(k_ref.grad.detach(), rank, world_size, seq_dim)),
        "dv": (v_cp.grad.detach(), _cp_slice(v_ref.grad.detach(), rank, world_size, seq_dim)),
    }

    local_metrics = []
    for name, (actual, expected) in checks.items():
        max_abs, rmse = _assert_close(name, actual, expected, args.atol, args.rtol, args.rmse_tol)
        local_metrics.append((name, max_abs, rmse))
        print(
            f"[rank {rank}] {name}: max_abs={max_abs.item():.6g}, rmse={rmse.item():.6g}",
            flush=True,
        )

    metric_tensor = torch.stack([torch.stack([m[1], m[2]]) for m in local_metrics])
    dist.all_reduce(metric_tensor, op=dist.ReduceOp.MAX)
    if rank == 0:
        for idx, (name, _, _) in enumerate(local_metrics):
            print(
                f"[global max] {name}: max_abs={metric_tensor[idx, 0].item():.6g}, "
                f"rmse={metric_tensor[idx, 1].item():.6g}",
                flush=True,
            )
        print("FA4 CP matches non-CP for local sequence chunks.", flush=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
