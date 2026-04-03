"""
Standalone test for THD AllGather-based Context Parallelism (forward + backward).

Run with:
    torchrun --nproc_per_node=2 tests/pytorch/attention/test_thd_ag_cp.py
"""

import os
import sys
import logging
import torch
import torch.distributed as dist

# Force fused attention backend
os.environ["NVTE_FLASH_ATTN"] = "0"
os.environ["NVTE_FUSED_ATTN"] = "1"

import transformer_engine.pytorch as te
import transformer_engine_torch as tex

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def run_test():
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    assert world_size == 2, "This test requires exactly 2 GPUs"

    device_count = torch.cuda.device_count()
    torch.cuda.set_device(rank % device_count)
    dist.init_process_group(backend="nccl", world_size=world_size, rank=rank)

    cp_group = dist.new_group(range(world_size), backend="nccl")
    cp_stream = torch.cuda.Stream()

    # Config
    batch_size = 3
    num_heads = 16
    head_dim = 64
    dtype = torch.bfloat16
    atol, rtol = 2.5e-2, 2.5e-2

    # Sequence lengths must be divisible by 2*world_size=4
    seqlens = torch.tensor([256, 512, 1024], dtype=torch.int32)
    assert all(s % (2 * world_size) == 0 for s in seqlens)

    # Build cu_seqlens (no padding between seqs)
    cu_seqlens = torch.zeros(batch_size + 1, dtype=torch.int32)
    cu_seqlens[1:] = seqlens.cumsum(0)
    cu_seqlens = cu_seqlens.cuda()
    cu_seqlens_padded = cu_seqlens.clone()
    total_tokens = cu_seqlens[-1].item()

    # Create global Q/K/V data (same on all ranks via same seed)
    torch.manual_seed(42)
    q_global = 0.02 * torch.randn(total_tokens, num_heads, head_dim, dtype=dtype, device="cuda")
    k_global = 0.02 * torch.randn(total_tokens, num_heads, head_dim, dtype=dtype, device="cuda")
    v_global = 0.02 * torch.randn(total_tokens, num_heads, head_dim, dtype=dtype, device="cuda")
    dout_global = 0.02 * torch.randn(total_tokens, num_heads * head_dim, dtype=dtype, device="cuda")

    # ============ Run without CP (single-GPU reference) ============
    log.info(f"[Rank {rank}] Running without CP (reference)")
    core_attn = te.DotProductAttention(
        num_heads,
        head_dim,
        num_gqa_groups=num_heads,
        attention_dropout=0.0,
        qkv_format="thd",
        attn_mask_type="padding_causal",
    ).cuda()

    q_ref = q_global.clone().requires_grad_()
    k_ref = k_global.clone().requires_grad_()
    v_ref = v_global.clone().requires_grad_()

    out_ref = core_attn(
        q_ref,
        k_ref,
        v_ref,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_kv=cu_seqlens,
        cu_seqlens_q_padded=cu_seqlens_padded,
        cu_seqlens_kv_padded=cu_seqlens_padded,
    )
    out_ref.backward(dout_global.clone())
    dq_ref, dk_ref, dv_ref = q_ref.grad, k_ref.grad, v_ref.grad
    log.info(f"[Rank {rank}] Reference out shape: {out_ref.shape}")

    # ============ Run with CP (AllGather) ============
    log.info(f"[Rank {rank}] Running with CP (all_gather)")

    # Partition Q/K/V for this CP rank
    seq_idx = tex.thd_get_partitioned_indices(cu_seqlens_padded, total_tokens, world_size, rank)
    seq_idx_kv = seq_idx  # same since self-attention

    q_cp = q_global.index_select(0, seq_idx).contiguous().requires_grad_()
    k_cp = k_global.index_select(0, seq_idx_kv).contiguous().requires_grad_()
    v_cp = v_global.index_select(0, seq_idx_kv).contiguous().requires_grad_()
    dout_cp = dout_global.index_select(0, seq_idx).contiguous()

    # Set up CP group
    core_attn.set_context_parallel_group(
        cp_group,
        list(range(world_size)),
        cp_stream,
        "all_gather",
    )

    out_cp = core_attn(
        q_cp,
        k_cp,
        v_cp,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_kv=cu_seqlens,
        cu_seqlens_q_padded=cu_seqlens_padded,
        cu_seqlens_kv_padded=cu_seqlens_padded,
    )
    out_cp.backward(dout_cp)
    dq_cp, dk_cp, dv_cp = q_cp.grad, k_cp.grad, v_cp.grad
    log.info(f"[Rank {rank}] CP out shape: {out_cp.shape}")

    # ============ Compare ============
    # Extract reference outputs for this rank's partition
    out_ref_part = out_ref.detach().index_select(0, seq_idx).contiguous()
    dq_ref_part = dq_ref.index_select(0, seq_idx).contiguous()
    dk_ref_part = dk_ref.index_select(0, seq_idx_kv).contiguous()
    dv_ref_part = dv_ref.index_select(0, seq_idx_kv).contiguous()

    passed = True
    for name, ref, cp in [
        ("out", out_ref_part, out_cp.detach()),
        ("dq", dq_ref_part, dq_cp),
        ("dk", dk_ref_part, dk_cp),
        ("dv", dv_ref_part, dv_cp),
    ]:
        max_diff = (ref - cp).abs().max().item()
        log.info(f"[Rank {rank}] {name}: max_diff = {max_diff}")
        try:
            torch.testing.assert_close(ref, cp, atol=atol, rtol=rtol)
            log.info(f"[Rank {rank}] {name}: PASSED")
        except AssertionError as e:
            log.error(f"[Rank {rank}] {name}: FAILED - {e}")
            passed = False

    dist.destroy_process_group()
    if not passed:
        log.error(f"[Rank {rank}] TEST FAILED")
        sys.exit(1)
    log.info(f"[Rank {rank}] ALL TESTS PASSED")


if __name__ == "__main__":
    run_test()
