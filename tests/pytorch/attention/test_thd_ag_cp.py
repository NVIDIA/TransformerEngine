"""
Standalone test for THD AllGather-based Context Parallelism (forward only).

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

    # Sequence lengths must be divisible by 2*world_size=4
    # Test with asymmetric sequence lengths
    seqlens = torch.tensor([256, 512, 1024], dtype=torch.int32)
    assert all(s % (2 * world_size) == 0 for s in seqlens), (
        "Sequence lengths must be divisible by 2*world_size"
    )

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
    core_attn.eval()

    q_ref = q_global.clone()
    k_ref = k_global.clone()
    v_ref = v_global.clone()

    with torch.no_grad():
        out_ref = core_attn(
            q_ref, k_ref, v_ref,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_kv=cu_seqlens,
            cu_seqlens_q_padded=cu_seqlens_padded,
            cu_seqlens_kv_padded=cu_seqlens_padded,
        )
    log.info(f"[Rank {rank}] Reference output shape: {out_ref.shape}")

    # ============ Run with CP (AllGather) ============
    log.info(f"[Rank {rank}] Running with CP (all_gather)")

    # Partition Q/K/V for this CP rank using tex.thd_get_partitioned_indices
    seq_idx = tex.thd_get_partitioned_indices(
        cu_seqlens_padded, total_tokens, world_size, rank
    )
    q_cp = q_global.index_select(0, seq_idx).contiguous()
    k_cp = k_global.index_select(0, seq_idx).contiguous()
    v_cp = v_global.index_select(0, seq_idx).contiguous()
    log.info(f"[Rank {rank}] CP Q shape: {q_cp.shape}, seq_idx[:10]: {seq_idx[:10]}")

    # Set up CP group
    core_attn.set_context_parallel_group(
        cp_group,
        list(range(world_size)),
        cp_stream,
        "all_gather",
    )

    with torch.no_grad():
        out_cp = core_attn(
            q_cp, k_cp, v_cp,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_kv=cu_seqlens,
            cu_seqlens_q_padded=cu_seqlens_padded,
            cu_seqlens_kv_padded=cu_seqlens_padded,
        )
    log.info(f"[Rank {rank}] CP output shape: {out_cp.shape}")

    # ============ Compare ============
    # Extract same partition from reference output
    out_ref_part = out_ref.index_select(0, seq_idx).contiguous()

    # Compare
    max_diff = (out_ref_part - out_cp).abs().max().item()
    log.info(f"[Rank {rank}] Max absolute diff: {max_diff}")

    try:
        torch.testing.assert_close(
            out_ref_part, out_cp,
            atol=2.5e-2, rtol=2.5e-2,
        )
        log.info(f"[Rank {rank}] PASSED: CP output matches reference!")
    except AssertionError as e:
        log.error(f"[Rank {rank}] FAILED: {e}")
        sys.exit(1)

    dist.destroy_process_group()
    log.info(f"[Rank {rank}] Test completed successfully.")


if __name__ == "__main__":
    run_test()
