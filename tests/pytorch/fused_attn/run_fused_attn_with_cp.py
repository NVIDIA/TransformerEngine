# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import os, sys
import torch
import torch.distributed as dist
from transformer_engine.pytorch.attention import DotProductAttention
import transformer_engine_torch as tex
from test_fused_attn_with_cp import model_configs_flash_attn, model_configs_fused_attn

dtypes = {"fp16": torch.float16, "bf16": torch.bfloat16}


def run_dpa_with_cp(dtype="bf16", model=None, qkv_format="bshd", kernel_backend="FlashAttention"):
    """Test DotProductAttention module with context parallelism"""

    os.environ["NVTE_FLASH_ATTN"] = "0"
    os.environ["NVTE_FUSED_ATTN"] = "0"
    if kernel_backend == "FlashAttention":
        os.environ["NVTE_FLASH_ATTN"] = "1"
        config = model_configs_flash_attn[model]
    if kernel_backend == "FusedAttention":
        os.environ["NVTE_FUSED_ATTN"] = "1"
        config = model_configs_fused_attn[model]
        if qkv_format == "thd" and (
            config.num_heads != config.num_gqa_groups or config.attn_bias_type == "post_scale_bias"
        ):
            return

    rank = int(os.getenv("RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))

    if dist.is_initialized():
        world_size = dist.get_world_size()
        rank = dist.get_rank()
    else:
        device_count = torch.cuda.device_count()
        device = rank % device_count
        torch.cuda.set_device(device)

    print(f"[INFO] world_size:{world_size}, rank:{rank}")

    dist.init_process_group(backend="nccl", world_size=world_size, rank=rank)

    # create flash attn comm group for CP
    cp_comm_ranks = range(world_size)
    assert rank in cp_comm_ranks
    cp_comm_group = dist.new_group(cp_comm_ranks, backend="nccl")

    assert config.attn_mask_type in [
        "causal",
        "no_mask",
    ], f"{config.attn_mask_type} is an unsupported attention mask type!"

    if kernel_backend == "FusedAttention" and qkv_format == "thd":
        if "causal" in config.attn_mask_type:
            config.attn_mask_type = "padding_causal"
        else:
            config.attn_mask_type = "padding"

    # instantiate core attn module
    core_attn = DotProductAttention(
        config.num_heads,
        config.head_dim,
        num_gqa_groups=config.num_gqa_groups,
        attention_dropout=config.dropout_p,
        qkv_format=qkv_format,
        attn_mask_type=config.attn_mask_type,
    )
    core_attn = core_attn.cuda()

    # create flash attn inputs
    if qkv_format == "bshd":
        q_input_shape = (config.batch_size, config.max_seqlen_q, config.num_heads, config.head_dim)
        kv_input_shape = (
            config.batch_size,
            config.max_seqlen_kv,
            config.num_gqa_groups,
            config.head_dim,
        )
        attn_output_shape = (
            config.batch_size,
            config.max_seqlen_q,
            config.num_heads * config.head_dim,
        )
        cu_seqlens_q = None
        cu_seqlens_kv = None
    elif qkv_format == "sbhd":
        q_input_shape = (config.max_seqlen_q, config.batch_size, config.num_heads, config.head_dim)
        kv_input_shape = (
            config.max_seqlen_kv,
            config.batch_size,
            config.num_gqa_groups,
            config.head_dim,
        )
        attn_output_shape = (
            config.max_seqlen_q,
            config.batch_size,
            config.num_heads * config.head_dim,
        )
        cu_seqlens_q = None
        cu_seqlens_kv = None
    elif qkv_format == "thd":
        seqlens_q = torch.randint(world_size * 2, config.max_seqlen_q + 1, [config.batch_size]).to(
            torch.int32
        )
        seqlens_q = seqlens_q - seqlens_q % (world_size * 2)
        cu_seqlens_q = torch.cat([torch.zeros([1], dtype=torch.int32), seqlens_q.cumsum(0)])
        cu_seqlens_kv = cu_seqlens_q
        q_input_shape = (cu_seqlens_q[-1], config.num_heads, config.head_dim)
        kv_input_shape = (cu_seqlens_kv[-1], config.num_gqa_groups, config.head_dim)
        attn_output_shape = (cu_seqlens_q[-1], config.num_heads * config.head_dim)
        cu_seqlens_q = cu_seqlens_q.to(torch.int32).cuda()
        cu_seqlens_kv = cu_seqlens_kv.to(torch.int32).cuda()
    else:
        assert False, f"{qkv_format} is an unsupported qkv_format!"

    q = torch.randn(q_input_shape, dtype=dtypes[dtype]).cuda()
    k = torch.randn(kv_input_shape, dtype=dtypes[dtype]).cuda()
    v = torch.randn(kv_input_shape, dtype=dtypes[dtype]).cuda()
    dout = torch.randn(attn_output_shape, dtype=dtypes[dtype]).cuda()

    # create flash attention bias
    if config.attn_bias_type not in ["no_bias", "alibi"]:
        attn_bias_shape = (1, 1, config.max_seqlen_q, config.max_seqlen_kv)
        bias = torch.randn(*attn_bias_shape, dtype=dtypes[dtype]).cuda()
    else:
        bias = None

    # make sure all GPU ranks have same inputs
    for x in [q, k, v, dout] + ([] if bias is None else [bias]):
        dist.broadcast(x, 0, group=cp_comm_group)
    if qkv_format == "thd":
        for x in [cu_seqlens_q, cu_seqlens_kv]:
            dist.broadcast(x, 0, group=cp_comm_group)

    # run core_attn without CP
    for x in [q, k, v]:
        x.requires_grad = True
    out = core_attn(
        q,
        k,
        v,
        core_attention_bias_type=config.attn_bias_type,
        core_attention_bias=bias,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_kv=cu_seqlens_kv,
    )
    out.backward(dout)

    # run core_attn wit CP
    q_, k_, v_, dout_, *rest = [
        x.clone().detach() for x in [q, k, v, dout] + ([] if bias is None else [bias])
    ]
    bias_ = rest[0] if len(rest) else None
    if qkv_format == "bshd" or qkv_format == "sbhd":
        seq_dim = qkv_format.index("s")
        q_, k_, v_, dout_ = [
            x.view(
                *x.shape[:seq_dim],
                2 * world_size,
                x.shape[seq_dim] // (2 * world_size),
                *x.shape[(seq_dim + 1) :],
            )
            for x in [q_, k_, v_, dout_]
        ]
        seq_idx = torch.tensor([rank, 2 * world_size - rank - 1], device=q_.device)
        q_, k_, v_, dout_ = [x.index_select(seq_dim, seq_idx) for x in [q_, k_, v_, dout_]]
        q_, k_, v_, dout_ = [
            x.view(*x.shape[:seq_dim], -1, *x.shape[(seq_dim + 2) :]) for x in [q_, k_, v_, dout_]
        ]
    elif qkv_format == "thd":
        seq_idx_q = tex.thd_get_partitioned_indices(cu_seqlens_q, q_.size(0), world_size, rank)
        seq_idx_kv = tex.thd_get_partitioned_indices(cu_seqlens_kv, k_.size(0), world_size, rank)
        q_, dout_ = [x.index_select(0, seq_idx_q) for x in [q_, dout_]]
        k_, v_ = [x.index_select(0, seq_idx_kv) for x in [k_, v_]]
        cu_seqlens_q = cu_seqlens_q // world_size
        cu_seqlens_kv = cu_seqlens_kv // world_size
    else:
        assert False, f"{qkv_format} is an unsupported qkv_format!"
    q_, k_, v_ = [x.requires_grad_() for x in [q_, k_, v_]]
    if bias_ is not None:
        bias_ = bias_.view(
            *bias_.shape[:-2], 2 * world_size, bias_.shape[-2] // (2 * world_size), bias_.shape[-1]
        )
        bias_ = bias_.index_select(2, seq_idx)
        bias_ = bias_.view(*bias_.shape[:2], -1, bias_.shape[-1])
    core_attn.set_context_parallel_group(cp_comm_group, cp_comm_ranks, torch.cuda.Stream())
    max_seqlen_q = config.max_seqlen_q
    max_seqlen_kv = config.max_seqlen_kv
    out_ = core_attn(
        q_,
        k_,
        v_,
        core_attention_bias_type=config.attn_bias_type,
        core_attention_bias=bias_,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_kv=cu_seqlens_kv,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_kv=max_seqlen_kv,
    )
    out_.backward(dout_)

    for x in [out_, q_.grad, k_.grad, v_.grad]:
        assert torch.all(~torch.isnan(x))
        assert torch.all(~torch.isinf(x))

    # compare results with and without CP
    tols = dict(atol=5e-3, rtol=5e-3)
    if dtype == "bf16":
        if config.num_heads == config.num_gqa_groups:
            tols = dict(atol=2.5e-2, rtol=2.5e-2)
        else:
            tols = dict(atol=3.5e-2, rtol=3.5e-2)

    if qkv_format == "bshd" or qkv_format == "sbhd":
        dq, dk, dv, out = [
            x.view(
                *x.shape[:seq_dim],
                2 * world_size,
                x.shape[seq_dim] // (2 * world_size),
                *x.shape[(seq_dim + 1) :],
            )
            for x in [q.grad, k.grad, v.grad, out]
        ]
        dq, dk, dv, out = [x.index_select(seq_dim, seq_idx) for x in [dq, dk, dv, out]]
        dq_, dk_, dv_, out_ = [
            x.view(*x.shape[:seq_dim], 2, x.shape[seq_dim] // 2, *x.shape[(seq_dim + 1) :])
            for x in [q_.grad, k_.grad, v_.grad, out_]
        ]
    elif qkv_format == "thd":
        dq, out = [x.index_select(0, seq_idx_q).contiguous().view(-1) for x in [q.grad, out]]
        dk, dv = [x.index_select(0, seq_idx_kv).contiguous().view(-1) for x in [k.grad, v.grad]]
        dq_, dk_, dv_, out_ = [x.view(-1) for x in [q_.grad, k_.grad, v_.grad, out_]]
    else:
        assert False, f"{qkv_format} is an unsupported qkv_format!"

    if qkv_format == "bshd":
        torch.testing.assert_close(out_[:, 0], out[:, 0], **tols)
        torch.testing.assert_close(dq_[:, 0], dq[:, 0], **tols)
        torch.testing.assert_close(dk_[:, 0], dk[:, 0], **tols)
        torch.testing.assert_close(dv_[:, 0], dv[:, 0], **tols)
        torch.testing.assert_close(out_[:, 1], out[:, 1], **tols)
        torch.testing.assert_close(dq_[:, 1], dq[:, 1], **tols)
        torch.testing.assert_close(dk_[:, 1], dk[:, 1], **tols)
        torch.testing.assert_close(dv_[:, 1], dv[:, 1], **tols)
    elif qkv_format == "sbhd":
        torch.testing.assert_close(out_[0], out[0], **tols)
        torch.testing.assert_close(dq_[0], dq[0], **tols)
        torch.testing.assert_close(dk_[0], dk[0], **tols)
        torch.testing.assert_close(dv_[0], dv[0], **tols)
        torch.testing.assert_close(out_[1], out[1], **tols)
        torch.testing.assert_close(dq_[1], dq[1], **tols)
        torch.testing.assert_close(dk_[1], dk[1], **tols)
        torch.testing.assert_close(dv_[1], dv[1], **tols)
    elif qkv_format == "thd":
        torch.testing.assert_close(out_, out, **tols)
        torch.testing.assert_close(dq_, dq, **tols)
        torch.testing.assert_close(dk_, dk, **tols)
        torch.testing.assert_close(dv_, dv, **tols)
    else:
        assert False, f"{qkv_format} is an unsupported qkv_format!"


def main(**kwargs):
    run_dpa_with_cp(**kwargs)


if __name__ == "__main__":
    kwargs = dict(arg.split("=") for arg in sys.argv[2:])
    main(**kwargs)
