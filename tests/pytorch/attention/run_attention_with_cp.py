# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import os
import sys
import logging
from contextlib import nullcontext
import torch
import torch.distributed as dist
from transformer_engine.pytorch.attention.dot_product_attention.context_parallel import (
    get_cu_seqlens_on_cp_rank,
)
from transformer_engine.pytorch.attention.dot_product_attention.utils import combine_and_quantize
import transformer_engine_torch as tex
from test_attention_with_cp import model_configs_flash_attn, model_configs_fused_attn
from transformer_engine.pytorch import (
    autocast,
    DotProductAttention,
    Float8Quantizer,
    Float8CurrentScalingQuantizer,
)
from transformer_engine.common.recipe import DelayedScaling, Float8CurrentScaling
from utils import ModelConfig, compare_and_assert

dtypes = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp8": torch.bfloat16}


def generate_input_shapes(
    qkv_format: str,
    config: ModelConfig,
    world_size: int,
    kernel_backend: str,
):
    if qkv_format == "bshd":
        q_input_shape = (
            config.batch_size,
            config.max_seqlen_q,
            config.num_heads,
            config.head_dim_qk,
        )
        k_input_shape = (
            config.batch_size,
            config.max_seqlen_kv,
            config.num_gqa_groups,
            config.head_dim_qk,
        )
        v_input_shape = (
            config.batch_size,
            config.max_seqlen_kv,
            config.num_gqa_groups,
            config.head_dim_v,
        )
        attn_output_shape = (
            config.batch_size,
            config.max_seqlen_q,
            config.num_heads * config.head_dim_v,
        )
        cu_seqlens_q = None
        cu_seqlens_kv = None
        cu_seqlens_q_padded = None
        cu_seqlens_kv_padded = None
    elif qkv_format == "sbhd":
        q_input_shape = (
            config.max_seqlen_q,
            config.batch_size,
            config.num_heads,
            config.head_dim_qk,
        )
        k_input_shape = (
            config.max_seqlen_kv,
            config.batch_size,
            config.num_gqa_groups,
            config.head_dim_qk,
        )
        v_input_shape = (
            config.max_seqlen_kv,
            config.batch_size,
            config.num_gqa_groups,
            config.head_dim_v,
        )
        attn_output_shape = (
            config.max_seqlen_q,
            config.batch_size,
            config.num_heads * config.head_dim_v,
        )
        cu_seqlens_q = None
        cu_seqlens_kv = None
        cu_seqlens_q_padded = None
        cu_seqlens_kv_padded = None
    elif qkv_format == "thd":
        q_input_shape = (
            config.batch_size * config.max_seqlen_q,
            config.num_heads,
            config.head_dim_qk,
        )
        k_input_shape = (
            config.batch_size * config.max_seqlen_q,
            config.num_gqa_groups,
            config.head_dim_qk,
        )
        v_input_shape = (
            config.batch_size * config.max_seqlen_q,
            config.num_gqa_groups,
            config.head_dim_v,
        )
        attn_output_shape = (
            config.batch_size * config.max_seqlen_q,
            config.num_heads * config.head_dim_v,
        )
        seqlens_q = torch.randint(0, config.max_seqlen_q + 1, [config.batch_size]).to(torch.int32)
        seqlens_q_padded = (seqlens_q + 2 * world_size - 1) // (world_size * 2) * (world_size * 2)
        cu_seqlens_q_padded = torch.cat(
            [
                torch.zeros([1], dtype=torch.int32),
                seqlens_q_padded.cumsum(0, dtype=torch.int32),
                torch.tensor([q_input_shape[0]], dtype=torch.int32),
            ]
        ).cuda()
        cu_seqlens_q = torch.clone(cu_seqlens_q_padded)
        if kernel_backend == "FusedAttention":
            cu_seqlens_q[1:-1] = seqlens_q.cumsum(0, dtype=torch.int32).cuda()
        cu_seqlens_q[-1] = cu_seqlens_q[-2]
        cu_seqlens_kv = cu_seqlens_q
        cu_seqlens_kv_padded = cu_seqlens_q_padded
    else:
        assert False, f"{qkv_format=} is not supported!"

    return (
        q_input_shape,
        k_input_shape,
        v_input_shape,
        attn_output_shape,
        cu_seqlens_q,
        cu_seqlens_kv,
        cu_seqlens_q_padded,
        cu_seqlens_kv_padded,
    )


def get_tols(config, dtype):
    if dtype == "bf16":
        if config.num_heads == config.num_gqa_groups:
            atol = 2.5e-2
            rtol = 2.5e-2
        else:
            atol = 3.5e-2
            rtol = 3.5e-2
        rmse_tol = 0.01
    elif dtype == "fp16":
        atol = 5e-3
        rtol = 5e-3
        rmse_tol = 0.01
    elif dtype == "fp8":
        atol = 5e-1
        rtol = 5e-1
        rmse_tol = 0.15
    else:
        assert False, f"{dtype=} is not supported!"

    return atol, rtol, rmse_tol


def run_dpa_with_cp(
    dtype="bf16",
    model=None,
    qkv_format="bshd",
    kernel_backend="FlashAttention",
    cp_comm_type="p2p",
    fp8_bwd="True",
    fp8_dpa="False",
    fp8_mha="False",
    scaling_mode="delayed",
    f16_O="False",
    log_level=logging.WARNING,
):
    """Test DotProductAttention module with context parallelism"""
    logging.root.setLevel(log_level)

    # set up environment variables and config
    fp8_bwd = fp8_bwd == "True" and dtype == "fp8"
    os.environ["NVTE_FP8_DPA_BWD"] = "1" if fp8_bwd else "0"
    fp8_dpa = fp8_dpa == "True" and dtype == "fp8"
    fp8_mha = fp8_mha == "True" and dtype == "fp8"
    f16_O = dtype == "fp8" and scaling_mode == "current" and f16_O == "True"
    os.environ["NVTE_DPA_FP8CS_O_in_F16"] = "1" if f16_O else "0"
    os.environ["NVTE_FLASH_ATTN"] = "0"
    os.environ["NVTE_FUSED_ATTN"] = "0"
    if kernel_backend == "FlashAttention":
        os.environ["NVTE_FLASH_ATTN"] = "1"
        config = model_configs_flash_attn[model]
    if kernel_backend == "FusedAttention":
        os.environ["NVTE_FUSED_ATTN"] = "1"
        config = model_configs_fused_attn[model]
    assert config.attn_mask_type in [
        "causal",
        "no_mask",
    ], f"{config.attn_mask_type=} is not supported!"
    if qkv_format == "thd":
        if "causal" in config.attn_mask_type:
            config.attn_mask_type = "padding_causal"
        else:
            config.attn_mask_type = "padding"

    # set up distributed group
    rank = int(os.getenv("RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    if dist.is_initialized():
        world_size = dist.get_world_size()
        rank = dist.get_rank()
    else:
        device_count = torch.cuda.device_count()
        device = rank % device_count
        torch.cuda.set_device(device)
    logging.info(f"[Rank {rank}] Setup: world_size {world_size}")
    dist.init_process_group(backend="nccl", world_size=world_size, rank=rank)

    # set up communication group for CP
    cp_comm_ranks = range(world_size)
    assert rank in cp_comm_ranks
    cp_comm_group = dist.new_group(cp_comm_ranks, backend="nccl")
    if cp_comm_type == "a2a+p2p":
        assert world_size % 2 == 0, (
            "{cp_comm_type=} requires world_size % 2 = 0 as it assumes the a2a level has cp_size"
            " = 2."
        )
        cp_comm_sub_ranks = [range(i * 2, (i + 1) * 2) for i in range(world_size // 2)]
        cp_comm_sub_ranks += [range(i, world_size, 2) for i in range(2)]
        cp_comm_sub_groups = []
        for sub_ranks in cp_comm_sub_ranks:
            sub_group = dist.new_group(sub_ranks, backend="nccl")
            if rank in sub_ranks:
                cp_comm_sub_groups.append(sub_group)

    if dtype == "fp8":
        if scaling_mode == "delayed":
            fp8_recipe = DelayedScaling(fp8_dpa=fp8_dpa, fp8_mha=fp8_mha)
        if scaling_mode == "current":
            fp8_recipe = Float8CurrentScaling(fp8_dpa=fp8_dpa, fp8_mha=fp8_mha)

    # instantiate attention module
    core_attn = DotProductAttention(
        config.num_heads,
        (config.head_dim_qk, config.head_dim_v),
        num_gqa_groups=config.num_gqa_groups,
        attention_dropout=config.dropout_p,
        qkv_format=qkv_format,
        attn_mask_type=config.attn_mask_type,
        window_size=config.window_size,
        softmax_type=config.softmax_type,
        return_max_logit=config.return_max_logit,
    ).cuda()
    if config.softmax_type != "vanilla":
        core_attn.softmax_offset.requires_grad = True

    # generate attention inputs
    (
        q_input_shape,
        k_input_shape,
        v_input_shape,
        attn_output_shape,
        cu_seqlens_q,
        cu_seqlens_kv,
        cu_seqlens_q_padded,
        cu_seqlens_kv_padded,
    ) = generate_input_shapes(qkv_format, config, world_size, kernel_backend)
    q_orig = torch.clamp(torch.randn(q_input_shape, dtype=dtypes[dtype]), min=-1, max=1).cuda()
    k_orig = torch.clamp(torch.randn(k_input_shape, dtype=dtypes[dtype]), min=-1, max=1).cuda()
    v_orig = torch.clamp(torch.randn(v_input_shape, dtype=dtypes[dtype]), min=-1, max=1).cuda()
    dout_orig = torch.clamp(
        torch.randn(attn_output_shape, dtype=dtypes[dtype]), min=-1, max=1
    ).cuda()
    if scaling_mode == "delayed":
        qkv_quantizer = Float8Quantizer(
            fp8_dtype=tex.DType.kFloat8E4M3,
            scale=torch.tensor([1], dtype=torch.float32).cuda(),
            amax=torch.tensor([0], dtype=torch.float32).cuda(),
        )
        dout_quantizer = Float8Quantizer(
            fp8_dtype=tex.DType.kFloat8E5M2,
            scale=torch.tensor([1], dtype=torch.float32).cuda(),
            amax=torch.tensor([0], dtype=torch.float32).cuda(),
        )
    if scaling_mode == "current":
        qkv_quantizer = Float8CurrentScalingQuantizer(
            fp8_dtype=tex.DType.kFloat8E4M3,
            device="cuda",
        )
        dout_quantizer = Float8CurrentScalingQuantizer(
            fp8_dtype=tex.DType.kFloat8E5M2,
            device="cuda",
        )
    qkv_layout = "_".join([qkv_format] * 3)
    q, k, v, dout = [x.clone().detach() for x in [q_orig, k_orig, v_orig, dout_orig]]
    if fp8_mha:
        q, k, v = combine_and_quantize(qkv_layout, q, k, v, qkv_quantizer)
    for x in [q, k, v]:
        x.requires_grad = True

    if config.attn_bias_type not in ["no_bias", "alibi"]:
        attn_bias_shape = (1, 1, config.max_seqlen_q, config.max_seqlen_kv)
        bias = torch.randn(*attn_bias_shape, dtype=dtypes[dtype]).cuda()
    else:
        bias = None

    ############ run without CP ############
    logging.info(f"[Rank {rank}] Run without context parallelism")
    if dtype == "fp8":
        fp8_context = autocast(enabled=True, recipe=fp8_recipe, amax_reduction_group=cp_comm_group)
    else:
        fp8_context = nullcontext()
    max_logit = None
    with fp8_context:
        # q, k, v, out in FP8; dout in F16
        out = core_attn(
            q,
            k,
            v,
            core_attention_bias_type=config.attn_bias_type,
            core_attention_bias=bias,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_kv=cu_seqlens_kv,
            cu_seqlens_q_padded=cu_seqlens_q_padded,
            cu_seqlens_kv_padded=cu_seqlens_kv_padded,
            fp8_output=fp8_mha,
        )
        if config.return_max_logit:
            out, max_logit = out
        if fp8_bwd and fp8_mha:
            dout_fp8 = dout_quantizer(dout)
            out.backward(dout_fp8)
        else:
            out.backward(dout)
    dq, dk, dv = q.grad, k.grad, v.grad
    d_softmax_offset = None
    if config.softmax_type != "vanilla":
        d_softmax_offset = core_attn.softmax_offset.grad

    ############ run with CP ############
    logging.info(f"[Rank {rank}] Run with context parallelism")

    # set up inputs
    q_, k_, v_, dout_, *rest = [
        x.clone().detach()
        for x in [q_orig, k_orig, v_orig, dout_orig] + ([] if bias is None else [bias])
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
        seq_idx_q = tex.thd_get_partitioned_indices(
            cu_seqlens_q_padded, q_.shape[0], world_size, rank
        )
        seq_idx_kv = tex.thd_get_partitioned_indices(
            cu_seqlens_kv_padded, k_.shape[0], world_size, rank
        )
        q_, dout_ = [x.index_select(0, seq_idx_q) for x in [q_, dout_]]
        k_, v_ = [x.index_select(0, seq_idx_kv) for x in [k_, v_]]
    else:
        assert False, f"{qkv_format} is an unsupported qkv_format!"
    q_, k_, v_, dout_ = [x.contiguous() for x in [q_, k_, v_, dout_]]
    if scaling_mode == "delayed":
        qkv_quantizer.scale.fill_(1.0)
        qkv_quantizer.amax.fill_(0.0)
        dout_quantizer.scale.fill_(1.0)
        dout_quantizer.amax.fill_(0.0)
    if fp8_mha:
        q_, k_, v_ = combine_and_quantize(qkv_layout, q_, k_, v_, qkv_quantizer)
    q_, k_, v_ = [x.requires_grad_() for x in [q_, k_, v_]]
    if bias_ is not None:
        bias_ = bias_.view(
            *bias_.shape[:-2], 2 * world_size, bias_.shape[-2] // (2 * world_size), bias_.shape[-1]
        )
        bias_ = bias_.index_select(2, seq_idx)
        bias_ = bias_.view(*bias_.shape[:2], -1, bias_.shape[-1])
    # set up environment
    core_attn.set_context_parallel_group(
        cp_comm_sub_groups if cp_comm_type == "a2a+p2p" else cp_comm_group,
        cp_comm_ranks,
        torch.cuda.Stream(),
        cp_comm_type,
    )
    if config.softmax_type != "vanilla":
        core_attn.softmax_offset.grad.zero_()
    if dtype == "fp8":
        core_attn.fp8_initialized = False
        core_attn.fp8_meta_tensors_initialized = False
        fp8_context = autocast(enabled=True, recipe=fp8_recipe, amax_reduction_group=cp_comm_group)
    else:
        fp8_context = nullcontext()

    # run attention
    max_logit_ = None
    with fp8_context:
        # q, k, v, out in FP8; dout in F16
        out_ = core_attn(
            q_,
            k_,
            v_,
            core_attention_bias_type=config.attn_bias_type,
            core_attention_bias=bias_,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_kv=cu_seqlens_kv,
            cu_seqlens_q_padded=cu_seqlens_q_padded,
            cu_seqlens_kv_padded=cu_seqlens_kv_padded,
            fp8_output=fp8_mha,
        )
        if config.return_max_logit:
            out_, max_logit_ = out_
        if fp8_bwd and fp8_mha:
            dout_fp8_ = dout_quantizer(dout_)
            out_.backward(dout_fp8_)
        else:
            out_.backward(dout_)
    dq_, dk_, dv_ = q_.grad, k_.grad, v_.grad
    d_softmax_offset_ = None
    if config.softmax_type != "vanilla":
        d_softmax_offset_ = core_attn.softmax_offset.grad.clone()

    # get outputs
    tensors = [out, dq, dk, dv, out_, dq_, dk_, dv_]
    if fp8_mha:
        tensors_to_deq = [out, out_] if not fp8_bwd else tensors
        for i, tensor in enumerate(tensors_to_deq):
            tensors_to_deq[i] = tensor.dequantize()
        if not fp8_bwd:
            tensors[0], tensors[4] = tensors_to_deq
    for tensor in tensors:
        assert torch.all(~torch.isnan(tensor))
        assert torch.all(~torch.isinf(tensor))
    out, dq, dk, dv, out_, dq_, dk_, dv_ = tensors

    ############  compare results between CP and no-CP ############
    if qkv_format == "bshd" or qkv_format == "sbhd":
        dq, dk, dv, out = [
            x.view(
                *x.shape[:seq_dim],
                2 * world_size,
                x.shape[seq_dim] // (2 * world_size),
                *x.shape[(seq_dim + 1) :],
            )
            for x in [dq, dk, dv, out]
        ]
        dq, dk, dv, out = [x.index_select(seq_dim, seq_idx) for x in [dq, dk, dv, out]]
        dq_, dk_, dv_, out_ = [
            x.view(*x.shape[:seq_dim], 2, x.shape[seq_dim] // 2, *x.shape[(seq_dim + 1) :])
            for x in [dq_, dk_, dv_, out_]
        ]
    elif qkv_format == "thd":
        dq, out = [x.index_select(0, seq_idx_q).contiguous() for x in [dq, out]]
        dk, dv = [x.index_select(0, seq_idx_kv).contiguous() for x in [dk, dv]]
        dq_, dk_, dv_, out_ = [dq_, dk_, dv_, out_]
        cu_seqlens_q_padded = cu_seqlens_q_padded // world_size
        cu_seqlens_q = get_cu_seqlens_on_cp_rank(
            cu_seqlens_q, cu_seqlens_q_padded, world_size, rank, True, True
        )
        cu_pads_q = cu_seqlens_q_padded - cu_seqlens_q
        num_pads_q = cu_pads_q[1:] - cu_pads_q[:-1]
        for x in [dq, out, dq_, out_]:
            assert torch.count_nonzero(x[cu_seqlens_q_padded[-1] :]).item() == 0
            for b in range(config.batch_size):
                assert (
                    num_pads_q[b] == 0
                    or torch.count_nonzero(
                        x[(cu_seqlens_q_padded[b + 1] - num_pads_q[b]) : cu_seqlens_q_padded[b + 1]]
                    ).item()
                    == 0
                )
        cu_seqlens_kv_padded = cu_seqlens_kv_padded // world_size
        cu_seqlens_kv = get_cu_seqlens_on_cp_rank(
            cu_seqlens_kv, cu_seqlens_kv_padded, world_size, rank, True, True
        )
        cu_pads_kv = cu_seqlens_kv_padded - cu_seqlens_kv
        num_pads_kv = cu_pads_kv[1:] - cu_pads_kv[:-1]
        for x in [dk, dv, dk_, dv_]:
            assert torch.count_nonzero(x[cu_seqlens_kv_padded[-1] :]).item() == 0
            for b in range(config.batch_size):
                assert (
                    num_pads_kv[b] == 0
                    or torch.count_nonzero(
                        x[
                            (cu_seqlens_kv_padded[b + 1] - num_pads_kv[b]) : cu_seqlens_kv_padded[
                                b + 1
                            ]
                        ]
                    ).item()
                    == 0
                )

    atol, rtol, rmse_tol = get_tols(config, dtype)
    tensors_cp = [out_, dq_, dk_, dv_, d_softmax_offset_, max_logit_]
    tensors_no_cp = [out, dq, dk, dv, d_softmax_offset, max_logit]
    names = ["out", "dq", "dk", "dv", "d_softmax_offset", "max_logit"]
    names_cp = [x + "_cp" for x in names]
    names_no_cp = [x + "_no_cp" for x in names]
    is_fp8 = dtype == "fp8"
    for i, t in enumerate(tensors_no_cp):
        if t is not None:
            if "softmax_offset" not in names[i] and "max_logit" not in names[i]:
                if qkv_format == "bshd":
                    compare_and_assert(
                        t[:, 0],
                        tensors_cp[i][:, 0],
                        names_no_cp[i],
                        names_cp[i],
                        atol,
                        rtol,
                        rmse_tol,
                        is_fp8,
                    )
                    compare_and_assert(
                        t[:, 1],
                        tensors_cp[i][:, 1],
                        names_no_cp[i],
                        names_cp[i],
                        atol,
                        rtol,
                        rmse_tol,
                        is_fp8,
                    )
                elif qkv_format == "sbhd":
                    compare_and_assert(
                        t[0],
                        tensors_cp[i][0],
                        names_no_cp[i],
                        names_cp[i],
                        atol,
                        rtol,
                        rmse_tol,
                        is_fp8,
                    )
                    compare_and_assert(
                        t[1],
                        tensors_cp[i][1],
                        names_no_cp[i],
                        names_cp[i],
                        atol,
                        rtol,
                        rmse_tol,
                        is_fp8,
                    )
                elif qkv_format == "thd":
                    compare_and_assert(
                        t, tensors_cp[i], names_no_cp[i], names_cp[i], atol, rtol, rmse_tol, is_fp8
                    )
            else:
                compare_and_assert(
                    t, tensors_cp[i], names_no_cp[i], names_cp[i], atol, rtol, rmse_tol, is_fp8
                )
            logging.info(f"[Rank {rank}] CP vs no-CP: {names[i]} matches")

    # destroy distribution group
    dist.destroy_process_group()


def main(**kwargs):
    run_dpa_with_cp(**kwargs)


if __name__ == "__main__":
    kwargs = dict(arg.split("=") for arg in sys.argv[2:])
    main(**kwargs)
