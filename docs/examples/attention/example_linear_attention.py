# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Gated DeltaNet forward and backward, with optional contiguous context parallelism.

Run from the repository root:
    python docs/examples/attention/example_linear_attention.py
    python -m torch.distributed.run --standalone --nproc-per-node=2 \
        docs/examples/attention/example_linear_attention.py --cp --qkv-format thd
"""

import argparse
import os

import torch
import torch.distributed as dist

import transformer_engine.pytorch as te


def make_inputs(qkv_format):
    """Create the same batch on each rank for comparison with a non-CP run."""
    torch.manual_seed(1234)
    batch, sequence, heads, head_dim = 16, 128, 8, 64
    if qkv_format == "thd":
        # Sample document lengths from 32 to 128 tokens, then pack without padding.
        lengths = torch.randint(32, 129, (batch,), device="cpu").tolist()
        offsets = [0]
        for length in lengths:
            offsets.append(offsets[-1] + length)
        shape = (offsets[-1], heads, head_dim)
    else:
        shape = (batch, sequence, heads, head_dim)
        offsets = [row * sequence for row in range(batch + 1)]
    q, k, v = (torch.randn(shape, device="cuda", dtype=torch.bfloat16) for _ in range(3))
    # Per-token, per-head FP32 gates: g is the log decay; beta is the write strength.
    g = -torch.rand(shape[:-1], device="cuda", dtype=torch.float32) * 0.02
    beta = torch.rand_like(g) * 0.7
    inputs = tuple(t.requires_grad_() for t in (q, k, v, g, beta))
    initial_state = (
        torch.randn(batch, heads, head_dim, head_dim, device="cuda", dtype=torch.float32) * 0.1
    ).requires_grad_()
    cu_seqlens = torch.tensor(offsets, device="cuda", dtype=torch.int32)
    return inputs, initial_state, cu_seqlens


def run_single_gpu(inputs, initial_state, cu_seqlens, qkv_format):
    """Run on complete documents and differentiate through outputs and final states."""
    q, k, v, g, beta = inputs
    attention = te.GatedDeltaNetAttention(
        num_attention_heads=q.shape[-2],
        kv_channels=q.shape[-1],
        qkv_format=qkv_format,
    )
    output, final_state = attention(
        q,
        k,
        v,
        g=g,
        beta=beta,
        cu_seqlens=cu_seqlens if qkv_format == "thd" else None,
        initial_state=initial_state,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
    )
    loss = output.float().square().mean() + final_state.square().mean()
    loss.backward()
    return output, final_state


def local_tokens(tensor, qkv_format, cp_group):
    """Select this rank's contiguous interval of the packed buffer or each batch row."""
    rank, size = dist.get_rank(cp_group), dist.get_world_size(cp_group)
    if qkv_format == "thd":
        length = tensor.shape[0]
        return tensor[rank * length // size : (rank + 1) * length // size]
    length = tensor.shape[1]
    return tensor[:, rank * length // size : (rank + 1) * length // size]


def run_context_parallel(inputs, initial_state, cu_seqlens, qkv_format, cp_group):
    """Supply local tokens and global document metadata to GDN CP."""
    local_inputs = tuple(
        local_tokens(t, qkv_format, cp_group).detach().contiguous().requires_grad_() for t in inputs
    )
    q, k, v, g, beta = local_inputs
    state = initial_state.detach().clone().requires_grad_()
    attention = te.GatedDeltaNetAttention(
        num_attention_heads=q.shape[-2],
        kv_channels=q.shape[-1],
        qkv_format=qkv_format,
        cp_group=cp_group,
    )
    output, final_state = attention(
        q,
        k,
        v,
        g=g,
        beta=beta,
        cu_seqlens=cu_seqlens,
        initial_state=state,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
    )
    # Normalize by the global output size, so rank-local losses sum to the
    # non-CP loss. Count the replicated final-state loss only once.
    cp_size = dist.get_world_size(cp_group)
    loss = output.float().square().sum() / inputs[2].numel()
    loss = loss + final_state.square().mean() / cp_size
    loss.backward()
    return output, final_state, local_inputs, state


def check_finite(tensors):
    """Fail on missing gradients or non-finite results, including in CI."""
    for tensor in tensors:
        assert tensor is not None, "Missing gradient"
        assert torch.isfinite(tensor).all(), "Non-finite result"


def check_close(actual, expected, tolerance):
    """Compare BF16 results using relative RMS error, as in the GDN tests."""
    actual, expected = actual.detach().double(), expected.detach().double()
    error = (actual - expected).square().mean().sqrt()
    reference = expected.square().mean().sqrt().clamp_min(1e-12)
    assert error / reference < tolerance, f"Relative RMS error: {(error / reference).item()}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cp", action="store_true", help="Run with torch.distributed.run on two or more GPUs"
    )
    parser.add_argument("--qkv-format", choices=("bshd", "thd"), default="bshd")
    args = parser.parse_args()
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    if args.cp:
        # NCCL is PyTorch's recommended backend for CUDA distributed training.
        # CP also accepts other backends that support its CUDA collectives.
        dist.init_process_group(backend="nccl")
    try:
        if args.cp and dist.get_world_size() < 2:
            raise ValueError("--cp requires torch.distributed.run with at least two GPUs")
        inputs, initial_state, cu_seqlens = make_inputs(args.qkv_format)
        output, final_state = run_single_gpu(inputs, initial_state, cu_seqlens, args.qkv_format)
        assert output.shape == (*inputs[2].shape[:-2], inputs[2].shape[-2:].numel())
        assert final_state.shape == initial_state.shape
        check_finite((output, final_state, initial_state.grad, *(t.grad for t in inputs)))
        if args.cp:
            cp_group = dist.group.WORLD
            local_output, states, local_inputs, state = run_context_parallel(
                inputs, initial_state, cu_seqlens, args.qkv_format, cp_group
            )
            check_close(local_output, local_tokens(output, args.qkv_format, cp_group), 2e-2)
            check_close(states, final_state, 2e-2)
            for local, full in zip(local_inputs, inputs):
                check_close(local.grad, local_tokens(full.grad, args.qkv_format, cp_group), 4e-2)
            # Initial-state gradients belong to each document's first owning rank.
            dist.all_reduce(state.grad, group=cp_group)
            check_close(state.grad, initial_state.grad, 4e-2)
        if not args.cp or dist.get_rank() == 0:
            print(f"GDN {args.qkv_format}: forward and backward passed (CP={args.cp})")
    finally:
        if args.cp:
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
