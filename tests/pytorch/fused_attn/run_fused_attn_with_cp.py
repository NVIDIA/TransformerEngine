# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import os, sys
import torch
import torch.distributed as dist
from transformer_engine.pytorch.attention import DotProductAttention
from test_fused_attn_with_cp import model_configs

dtypes={'fp16' : torch.float16, 'bf16' : torch.bfloat16}

def run_dpa_with_cp(dtype='bf16', model=None, qkv_format='bshd', kernel_backend='FlashAttention'):
    """Test DotProductAttention module with context parallelism"""

    os.environ["NVTE_FLASH_ATTN"] = "0"
    os.environ["NVTE_FUSED_ATTN"] = "0"
    if kernel_backend == "FlashAttention":
        os.environ["NVTE_FLASH_ATTN"] = "1"
    if kernel_backend == "FusedAttention":
        os.environ["NVTE_FUSED_ATTN"] = "1"

    rank = int(os.getenv('RANK', '0'))
    world_size = int(os.getenv('WORLD_SIZE', '1'))

    if dist.is_initialized():
        world_size = dist.get_world_size()
        rank = dist.get_rank()
    else:
        device_count = torch.cuda.device_count()
        device = rank % device_count
        torch.cuda.set_device(device)

    print(f"[INFO] world_size:{world_size}, rank:{rank}")

    dist.init_process_group(backend='nccl', world_size=world_size, rank=rank)

    # create flash attn comm group for CP
    cp_comm_ranks = range(world_size)
    assert(rank in cp_comm_ranks)
    cp_comm_group = dist.new_group(cp_comm_ranks, backend='nccl')

    config = model_configs[model]

    assert config.attn_mask_type in ['causal', 'no_mask'], f"{config.attn_mask_type} is an unsupported attention mask type!"

    # instantiate core attn module
    core_attn = DotProductAttention(config.num_heads,
                                    config.head_dim,
                                    num_gqa_groups=config.num_gqa_groups,
                                    attention_dropout=config.dropout_p,
                                    qkv_format=qkv_format,
                                    attn_mask_type=config.attn_mask_type)
    core_attn = core_attn.cuda()

    # create flash attn inputs
    if qkv_format == "bshd":
        q_input_shape = (config.batch_size, config.max_seqlen_q, config.num_heads, config.head_dim)
        kv_input_shape = (config.batch_size, config.max_seqlen_kv, config.num_gqa_groups, config.head_dim)
        attn_output_shape = (config.batch_size, config.max_seqlen_q, config.num_heads*config.head_dim)
    elif qkv_format == "sbhd":
        q_input_shape = (config.max_seqlen_q, config.batch_size, config.num_heads, config.head_dim)
        kv_input_shape = (config.max_seqlen_kv, config.batch_size, config.num_gqa_groups, config.head_dim)
        attn_output_shape = (config.max_seqlen_q, config.batch_size, config.num_heads*config.head_dim)
    else:
        assert False, f"{qkv_format} is an unsupported qkv_format!"
    q = torch.randn(q_input_shape, dtype=dtypes[dtype]).cuda()
    k = torch.randn(kv_input_shape, dtype=dtypes[dtype]).cuda()
    v = torch.randn(kv_input_shape, dtype=dtypes[dtype]).cuda()
    dout = torch.randn(attn_output_shape, dtype=dtypes[dtype]).cuda()

    # make sure all GPU ranks have same inputs
    for x in [q, k, v, dout]:
        dist.broadcast(x, 0, group=cp_comm_group)

    # run core_attn without CP
    for x in [q, k, v]:
        x.requires_grad = True
    out = core_attn(q, k, v)
    out.backward(dout)

    # run core_attn wit CP
    q_, k_, v_, dout_ = [x.clone().detach() for x in [q, k, v, dout]]
    seq_dim = qkv_format.index('s')
    q_, k_, v_, dout_ = [x.view(*x.shape[:seq_dim], 2*world_size, x.shape[seq_dim]//(2*world_size), *x.shape[(seq_dim+1):]) \
        for x in [q_, k_, v_, dout_]]
    seq_idx = torch.tensor([rank, 2*world_size-rank-1], device=q_.device)
    q_, k_, v_, dout_ = [x.index_select(seq_dim, seq_idx) for x in [q_, k_, v_, dout_]]
    q_, k_, v_, dout_ = [x.view(*x.shape[:seq_dim], -1, *x.shape[(seq_dim+2):]) for x in [q_, k_, v_, dout_]]
    q_, k_, v_ = [x.requires_grad_() for x in [q_, k_, v_]]
    core_attn.set_context_parallel_group(cp_comm_group, cp_comm_ranks, torch.cuda.Stream())
    out_ = core_attn(q_, k_, v_)
    out_.backward(dout_)

    for x in [out_, q_.grad, k_.grad, v_.grad]:
        assert(torch.all(~torch.isnan(x)))
        assert(torch.all(~torch.isinf(x)))

    # compare results with and without CP
    tols = dict(atol=5e-3, rtol=5e-3)
    if dtype == 'bf16':
        tols = dict(atol=2.5e-2, rtol=2.5e-2)
    dq, dk, dv, out = [x.view(*x.shape[:seq_dim], 2*world_size, x.shape[seq_dim]//(2*world_size), *x.shape[(seq_dim+1):]) \
        for x in [q.grad, k.grad, v.grad, out]]
    dq, dk, dv, out = [x.index_select(seq_dim, seq_idx) for x in [dq, dk, dv, out]]
    dq_, dk_, dv_, out_ = [x.view(*x.shape[:seq_dim], 2, x.shape[seq_dim]//2, *x.shape[(seq_dim+1):]) \
        for x in [q_.grad, k_.grad, v_.grad, out_]]
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
    else:
        assert False, f"{qkv_format} is an unsupported qkv_format!"

def main(**kwargs):
    run_dpa_with_cp(**kwargs)

if __name__ == "__main__":
    kwargs = dict(arg.split('=') for arg in sys.argv[2:])
    main(**kwargs)
