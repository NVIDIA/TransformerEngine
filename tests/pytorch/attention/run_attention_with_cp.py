# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import copy
import os
import sys
import time
import logging
from contextlib import nullcontext
import torch
import torch.distributed as dist
from transformer_engine.pytorch.attention.dot_product_attention.context_parallel import (
    get_cu_seqlens_on_cp_rank,
    get_thd_partitioned_indices,
    validate_packed_contiguous_thd_metadata,
)
from transformer_engine.pytorch.attention.dot_product_attention.utils import combine_and_quantize
import transformer_engine_torch as tex
from transformer_engine.pytorch import DType
from test_attention_with_cp import model_configs_flash_attn, model_configs_fused_attn
from transformer_engine.pytorch import (
    autocast,
    DotProductAttention,
    Float8Quantizer,
    Float8CurrentScalingQuantizer,
    MXFP8Quantizer,
)
from transformer_engine.common.recipe import (
    DelayedScaling,
    Float8CurrentScaling,
    MXFP8BlockScaling,
    Format,
)
from utils import ModelConfig, compare_and_assert

# Pool mode (NVTE_CP_POOL_PG=1) only: shared CP collective groups, created once
# per pool by run_attention_with_cp_pool.main() and reused across every case in
# that pool. world_size and the rank set don't change per case, so re-creating
# these per call would be wasted NCCL setup (~50-100 ms each). Single-shot
# subprocess mode leaves these None / [] and run_dpa_with_cp creates/destroys
# its own groups inline.
_pool_cp_comm_group = None
_pool_cp_comm_sub_groups: list = []

_PACKED_CONTIGUOUS_ENV = "NVTE_EXPERIMENTAL_CP_AG_THD_PACKED_CONTIGUOUS"

dtypes = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp8": torch.bfloat16}

uniform_thd_benchmark_configs = {
    "uniform_4x128k": ModelConfig(
        4, 131072, 32, 128, num_gqa_groups=8, attn_mask_type="causal"
    ),
    "uniform_8x64k": ModelConfig(
        8, 65536, 32, 128, num_gqa_groups=8, attn_mask_type="causal"
    ),
    "uniform_16x32k": ModelConfig(
        16, 32768, 32, 128, num_gqa_groups=8, attn_mask_type="causal"
    ),
    "uneven_8docs_512k": ModelConfig(
        8, 131072, 32, 128, num_gqa_groups=8, attn_mask_type="causal"
    ),
}


def generate_input_shapes(
    qkv_format: str,
    config: ModelConfig,
    world_size: int,
    kernel_backend: str,
    fa_pad_between_seqs: str = "False",
    thd_seqlen_pattern: str = "random",
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
        packed_contiguous = os.getenv(_PACKED_CONTIGUOUS_ENV, "0") == "1"
        partition_divisor = world_size if packed_contiguous else 2 * world_size
        if thd_seqlen_pattern == "max":
            seqlens_q = torch.full(
                [config.batch_size], config.max_seqlen_q, dtype=torch.int32
            )
            assert torch.all(seqlens_q.remainder(partition_divisor) == 0), (
                "Matched uniform THD benchmarks require every sequence length "
                "to be divisible by the selected CP partition count."
            )
            seqlens_q_padded = seqlens_q.clone()
        elif "," in thd_seqlen_pattern:
            values = [int(value) for value in thd_seqlen_pattern.split(",")]
            assert len(values) == config.batch_size
            assert all(0 < value <= config.max_seqlen_q for value in values)
            seqlens_q = torch.tensor(values, dtype=torch.int32)
            if packed_contiguous:
                seqlens_q_padded = seqlens_q.clone()
                assert seqlens_q.sum().remainder(partition_divisor) == 0
            else:
                seqlens_q_padded = (
                    (seqlens_q + 2 * world_size - 1) // (2 * world_size)
                ) * (2 * world_size)
        elif packed_contiguous:
            assert thd_seqlen_pattern == "random"
            assert config.batch_size == 2
            seqlens_q_padded = torch.tensor(
                [config.max_seqlen_q - 1, config.max_seqlen_q - (partition_divisor - 1)],
                dtype=torch.int32,
            )
            assert torch.all(seqlens_q_padded.remainder(partition_divisor) != 0)
            assert seqlens_q_padded.sum().remainder(partition_divisor) == 0
            seqlens_q = seqlens_q_padded.clone()
            if fa_pad_between_seqs == "True":
                seqlens_q -= torch.tensor([1, 2], dtype=torch.int32)
        else:
            assert thd_seqlen_pattern == "random"
            seqlens_q = torch.randint(
                0, config.max_seqlen_q + 1, [config.batch_size]
            ).to(torch.int32)
            seqlens_q_padded = (
                (seqlens_q + 2 * world_size - 1) // (world_size * 2) * (world_size * 2)
            )
        cu_seqlens_q_padded = torch.cat(
            [
                torch.zeros([1], dtype=torch.int32),
                seqlens_q_padded.cumsum(0, dtype=torch.int32),
            ]
        ).cuda()
        cu_seqlens_q = torch.clone(cu_seqlens_q_padded)

        # Generate padded data (cu_seqlens_q reflects non-padded lengths, so it
        # differs from cu_seqlens_q_padded) for FusedAttention always, and for
        # FlashAttention only when its test param requests it. DPA auto-detects
        # pad_between_seqs downstream from the cu_seqlens_q vs cu_seqlens_q_padded
        # mismatch.
        if kernel_backend == "FusedAttention" or fa_pad_between_seqs == "True":
            cu_seqlens_q[1:] = seqlens_q.cumsum(0, dtype=torch.int32).cuda()

        # NOTE: In case of Cross-Attention, `cu_seqlens_kv` and `cu_seqlens_kv_padded`
        # will not be the same as `cu_seqlens_q` and `cu_seqlens_q_padded` respectively.
        cu_seqlens_kv = cu_seqlens_q
        cu_seqlens_kv_padded = cu_seqlens_q_padded

        total_tokens = cu_seqlens_q_padded[-1]

        q_input_shape = (
            total_tokens,
            config.num_heads,
            config.head_dim_qk,
        )
        k_input_shape = (
            total_tokens,
            config.num_gqa_groups,
            config.head_dim_qk,
        )
        v_input_shape = (
            total_tokens,
            config.num_gqa_groups,
            config.head_dim_v,
        )
        attn_output_shape = (
            total_tokens,
            config.num_heads * config.head_dim_v,
        )
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
    is_training="True",
    fa_pad_between_seqs="False",
    deterministic="False",
    log_level=logging.WARNING,
    benchmark="0",
    thd_seqlen_pattern="random",
):
    """Test DotProductAttention module with context parallelism"""
    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)
    logging.root.setLevel(log_level)
    packed_contiguous = os.getenv(_PACKED_CONTIGUOUS_ENV, "0") == "1"
    partition = "packed_contiguous" if packed_contiguous else "per_document"
    if packed_contiguous:
        assert qkv_format == "thd" and cp_comm_type == "all_gather"
    # When is_training is False, gradient outputs are None.
    is_training = is_training == "True"
    benchmark_iters = int(benchmark)
    assert benchmark_iters >= 0
    if benchmark_iters:
        assert dtype == "bf16" and is_training
        assert qkv_format == "thd" and kernel_backend in [
            "FusedAttention",
            "FlashAttention",
        ]
        assert cp_comm_type == "all_gather"
        assert thd_seqlen_pattern == "max" or "," in thd_seqlen_pattern

    # set up environment variables and config
    if deterministic == "True":
        os.environ["NVTE_ALLOW_NONDETERMINISTIC_ALGO"] = "0"
    else:
        os.environ["NVTE_ALLOW_NONDETERMINISTIC_ALGO"] = "1"
    fp8_bwd = fp8_bwd == "True" and dtype == "fp8"
    os.environ["NVTE_FP8_DPA_BWD"] = "1" if fp8_bwd else "0"
    fp8_dpa = fp8_dpa == "True" and dtype == "fp8"
    fp8_mha = fp8_mha == "True" and dtype == "fp8" and scaling_mode != "mxfp8"
    f16_O = dtype == "fp8" and scaling_mode in ["current", "mxfp8"] and f16_O == "True"
    os.environ["NVTE_DPA_FP8CS_O_in_F16"] = "1" if f16_O else "0"
    os.environ["NVTE_FLASH_ATTN"] = "0"
    os.environ["NVTE_FUSED_ATTN"] = "0"
    if kernel_backend == "FlashAttention":
        os.environ["NVTE_FLASH_ATTN"] = "1"
        # Deep-copy: the module-level dict is shared across pool cases; the
        # THD branch below rewrites attn_mask_type in place, which would
        # otherwise leak into subsequent cases reusing the same model key.
        configs = (
            uniform_thd_benchmark_configs
            if model in uniform_thd_benchmark_configs
            else model_configs_flash_attn
        )
        config = copy.deepcopy(configs[model])
    if kernel_backend == "FusedAttention":
        os.environ["NVTE_FUSED_ATTN"] = "1"
        configs = (
            uniform_thd_benchmark_configs
            if model in uniform_thd_benchmark_configs
            else model_configs_fused_attn
        )
        config = copy.deepcopy(configs[model])
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
    # When NVTE_CP_POOL_PG=1, the pool runner owns the lifecycle of the main
    # process group across many cases; here we only reuse it.
    _pool_managed_pg = os.getenv("NVTE_CP_POOL_PG", "0") == "1"
    if dist.is_initialized():
        world_size = dist.get_world_size()
        rank = dist.get_rank()
    else:
        device_count = torch.cuda.device_count()
        device = rank % device_count
        torch.cuda.set_device(device)
    logging.info(f"[Rank {rank}] Setup: world_size {world_size}")
    if not _pool_managed_pg:
        dist.init_process_group(backend="nccl", world_size=world_size, rank=rank)

    # Set up communication group for CP. In pool mode, the pool worker has
    # already pre-created world-scoped and a2a+p2p sub-groups once and stashed
    # them in module-level pointers; we reuse those and the pool destroys them
    # at shutdown. In single-shot mode we create them per call and destroy in
    # the finally below.
    cp_comm_ranks = range(world_size)
    assert rank in cp_comm_ranks
    _reusing_pool_groups = _pool_managed_pg and _pool_cp_comm_group is not None
    cp_comm_group = None
    cp_comm_sub_groups: list = []
    if _reusing_pool_groups:
        cp_comm_group = _pool_cp_comm_group
        cp_comm_sub_groups = _pool_cp_comm_sub_groups if cp_comm_type == "a2a+p2p" else []
    else:
        cp_comm_group = dist.new_group(cp_comm_ranks, backend="nccl")
        if cp_comm_type == "a2a+p2p":
            assert world_size % 2 == 0, (
                "{cp_comm_type=} requires world_size % 2 = 0 as it assumes the a2a level has"
                " cp_size = 2."
            )
            cp_comm_sub_ranks = [range(i * 2, (i + 1) * 2) for i in range(world_size // 2)]
            cp_comm_sub_ranks += [range(i, world_size, 2) for i in range(2)]
            for sub_ranks in cp_comm_sub_ranks:
                sub_group = dist.new_group(sub_ranks, backend="nccl")
                if rank in sub_ranks:
                    cp_comm_sub_groups.append(sub_group)
    if dtype == "fp8":
        if scaling_mode == "delayed":
            fp8_recipe = DelayedScaling(fp8_dpa=fp8_dpa, fp8_mha=fp8_mha)
        if scaling_mode == "current":
            fp8_recipe = Float8CurrentScaling(fp8_dpa=fp8_dpa, fp8_mha=fp8_mha)
        if scaling_mode == "mxfp8":
            fp8_recipe = MXFP8BlockScaling(fp8_format=Format.E4M3, fp8_dpa=fp8_dpa, fp8_mha=fp8_mha)

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
    if not is_training:
        core_attn.eval()
    if is_training and config.softmax_type != "vanilla":
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
    ) = generate_input_shapes(
        qkv_format,
        config,
        world_size,
        kernel_backend,
        fa_pad_between_seqs,
        thd_seqlen_pattern,
    )
    if packed_contiguous:
        validate_packed_contiguous_thd_metadata(
            cu_seqlens_q,
            cu_seqlens_q_padded,
            q_input_shape[0],
            world_size,
        )
    has_inter_sequence_padding = None
    if qkv_format == "thd":
        # Resolve this once during setup so reference, CP, and timed replay use
        # the same physical-layout contract without synchronizing in the loop.
        has_inter_sequence_padding = not (
            torch.equal(cu_seqlens_q, cu_seqlens_q_padded)
            and torch.equal(cu_seqlens_kv, cu_seqlens_kv_padded)
        )
    if qkv_format == "thd" and rank == 0:
        effective_seqlens = (cu_seqlens_q[1:] - cu_seqlens_q[:-1]).cpu().tolist()
        print(
            f"BENCH_INPUT model={model} partition={partition} cp={world_size} "
            f"seqlens={effective_seqlens} logical_tokens={sum(effective_seqlens)} "
            f"physical_tokens={q_input_shape[0]}",
            flush=True,
        )
    total_tokens = q_input_shape[0] if qkv_format == "thd" else None
    q_orig = torch.clamp(torch.randn(q_input_shape, dtype=dtypes[dtype]), min=-1, max=1).cuda()
    k_orig = torch.clamp(torch.randn(k_input_shape, dtype=dtypes[dtype]), min=-1, max=1).cuda()
    v_orig = torch.clamp(torch.randn(v_input_shape, dtype=dtypes[dtype]), min=-1, max=1).cuda()
    dout_orig = torch.clamp(
        torch.randn(attn_output_shape, dtype=dtypes[dtype]), min=-1, max=1
    ).cuda()
    if scaling_mode == "delayed":
        qkv_quantizer = Float8Quantizer(
            fp8_dtype=DType.kFloat8E4M3,
            scale=torch.tensor([1], dtype=torch.float32).cuda(),
            amax=torch.tensor([0], dtype=torch.float32).cuda(),
        )
        dout_quantizer = Float8Quantizer(
            fp8_dtype=DType.kFloat8E5M2,
            scale=torch.tensor([1], dtype=torch.float32).cuda(),
            amax=torch.tensor([0], dtype=torch.float32).cuda(),
        )
    if scaling_mode == "current":
        qkv_quantizer = Float8CurrentScalingQuantizer(
            fp8_dtype=DType.kFloat8E4M3,
            device="cuda",
        )
        dout_quantizer = Float8CurrentScalingQuantizer(
            fp8_dtype=DType.kFloat8E5M2,
            device="cuda",
        )
    if scaling_mode == "mxfp8":
        qkv_quantizer = MXFP8Quantizer(
            fp8_dtype=DType.kFloat8E4M3,
            rowwise=True,
            columnwise=True,
        )
        qkv_quantizer.optimize_for_gemm = True
        qkv_quantizer.internal = False
        dout_quantizer = MXFP8Quantizer(
            fp8_dtype=DType.kFloat8E5M2,
            rowwise=True,
            columnwise=True,
        )
        dout_quantizer.optimize_for_gemm = True
        dout_quantizer.internal = False
    qkv_layout = "_".join([qkv_format] * 3)
    q, k, v, dout = [x.detach() for x in [q_orig, k_orig, v_orig, dout_orig]]
    if fp8_mha:
        q, k, v, qkv_layout, _ = combine_and_quantize(qkv_layout, q, k, v, qkv_quantizer)
    for x in [q, k, v]:
        x.requires_grad = True

    if config.attn_bias_type not in ["no_bias", "alibi"]:
        bias_shape_map = {
            "1hss": (1, config.num_heads, config.max_seqlen_q, config.max_seqlen_kv),
            "11ss": (1, 1, config.max_seqlen_q, config.max_seqlen_kv),
            "b1ss": (config.batch_size, 1, config.max_seqlen_q, config.max_seqlen_kv),
            "bhss": (
                config.batch_size,
                config.num_heads,
                config.max_seqlen_q,
                config.max_seqlen_kv,
            ),
            "111s": (1, 1, 1, config.max_seqlen_kv),
        }
        attn_bias_shape = bias_shape_map.get(config.bias_shape)
        if attn_bias_shape is None:
            assert False, f"cuDNN does not support {config.bias_shape=}"
        bias = torch.randn(*attn_bias_shape, dtype=dtypes[dtype]).cuda()
        # cuDNN does not support dbias calculation for 111s as of cuDNN 9.18
        # TODO(KshitijLakhani): Set requires_grad to True for all shapes once 111s is supported
        bias.requires_grad = True if config.bias_shape != "111s" else False
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
            pad_between_seqs=has_inter_sequence_padding,
            fp8_output=fp8_mha,
        )
        if config.return_max_logit:
            out, max_logit = out
        if is_training:
            if fp8_bwd and fp8_mha:
                dout_fp8 = dout_quantizer(dout)
                out.backward(dout_fp8)
            else:
                out.backward(dout)
    if is_training:
        dq, dk, dv, dbias = q.grad, k.grad, v.grad, bias.grad if bias is not None else None
        d_softmax_offset = (
            core_attn.softmax_offset.grad if config.softmax_type != "vanilla" else None
        )
    else:
        dq, dk, dv, dbias = None, None, None, None
        d_softmax_offset = None

    ############ run with CP ############
    logging.info(f"[Rank {rank}] Run with context parallelism")

    # set up inputs
    q_, k_, v_, dout_ = [x.detach() for x in [q_orig, k_orig, v_orig, dout_orig]]
    bias_ = bias.clone().detach() if bias is not None else None
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
        seq_idx_q = get_thd_partitioned_indices(
            cu_seqlens_q_padded,
            q_.shape[0],
            world_size,
            rank,
            device=q_.device,
        )
        seq_idx_kv = get_thd_partitioned_indices(
            cu_seqlens_kv_padded,
            k_.shape[0],
            world_size,
            rank,
            device=k_.device,
        )
        q_, dout_ = [x.index_select(0, seq_idx_q) for x in [q_, dout_]]
        k_, v_ = [x.index_select(0, seq_idx_kv) for x in [k_, v_]]
    else:
        assert False, f"{qkv_format} is an unsupported qkv_format!"
    q_, k_, v_, dout_ = [x.contiguous() for x in [q_, k_, v_, dout_]]
    out = out.detach()
    if max_logit is not None:
        max_logit = max_logit.detach()
    del q, k, v, dout, q_orig, k_orig, v_orig, dout_orig
    if scaling_mode == "delayed":
        qkv_quantizer.scale.fill_(1.0)
        qkv_quantizer.amax.fill_(0.0)
        dout_quantizer.scale.fill_(1.0)
        dout_quantizer.amax.fill_(0.0)
    if fp8_mha:
        q_, k_, v_, qkv_layout, _ = combine_and_quantize(qkv_layout, q_, k_, v_, qkv_quantizer)
    if is_training:
        q_, k_, v_ = [x.requires_grad_() for x in [q_, k_, v_]]
    if bias_ is not None:
        ndim = bias_.ndim
        seq_q_dim = ndim - 2
        if qkv_format == "thd":
            bias_seq_idx = seq_idx_q
        else:
            bias_seq_idx = seq_idx
        shape_before_seq = bias_.shape[:seq_q_dim]
        seq_q_size = bias_.shape[seq_q_dim]
        seq_kv_size = bias_.shape[-1]
        if seq_q_size == 1:
            # TODO(KshitijLakhani): Set to True always once cuDNN supports dbias for 111s
            bias_.requires_grad = False
            # Bias is broadcast, no need to partition along sequence dimension
            pass
        else:
            bias_ = bias_.view(
                *shape_before_seq, 2 * world_size, seq_q_size // (2 * world_size), seq_kv_size
            )
            bias_ = bias_.index_select(seq_q_dim, bias_seq_idx)
            bias_ = bias_.view(*shape_before_seq, -1, seq_kv_size)
            bias_.requires_grad = True
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
            pad_between_seqs=has_inter_sequence_padding,
            fp8_output=fp8_mha,
        )
        if config.return_max_logit:
            out_, max_logit_ = out_
        if is_training:
            if fp8_bwd and fp8_mha:
                dout_fp8_ = dout_quantizer(dout_)
                out_.backward(dout_fp8_)
            else:
                out_.backward(dout_)
    if is_training:
        dq_, dk_, dv_, dbias_ = (
            q_.grad,
            k_.grad,
            v_.grad,
            bias_.grad if bias_ is not None else None,
        )
        d_softmax_offset_ = (
            core_attn.softmax_offset.grad.clone() if config.softmax_type != "vanilla" else None
        )
    else:
        dq_, dk_, dv_, dbias_ = None, None, None, None
        d_softmax_offset_ = None

    save_dir = os.environ.get("CP_PARTITION_SAVE_DIR")
    if save_dir:
        assert qkv_format == "thd" and thd_seqlen_pattern == "max"
        os.makedirs(save_dir, exist_ok=True)
        torch.save(
            {
                "out": out_.detach().cpu(),
                "dq": dq_.detach().cpu() if dq_ is not None else None,
                "dk": dk_.detach().cpu() if dk_ is not None else None,
                "dv": dv_.detach().cpu() if dv_ is not None else None,
                "seq_idx_q": seq_idx_q.detach().cpu(),
                "seq_idx_kv": seq_idx_kv.detach().cpu(),
                "total_tokens": total_tokens,
                "cu_seqlens_q": cu_seqlens_q.detach().cpu(),
                "cu_seqlens_q_padded": cu_seqlens_q_padded.detach().cpu(),
                "partition": partition,
                "cp_size": world_size,
                "model": model,
                "dtype": dtype,
                "seed": 1234,
                "thd_seqlen_pattern": thd_seqlen_pattern,
            },
            os.path.join(save_dir, f"rank{rank}.pt"),
        )

    benchmark_metadata = (
        cu_seqlens_q,
        cu_seqlens_kv,
        cu_seqlens_q_padded,
        cu_seqlens_kv_padded,
    )

    # get outputs
    tensors = [out, dq, dk, dv, dbias, out_, dq_, dk_, dv_, dbias_]
    names = ["out", "dq", "dk", "dv", "dbias", "out_cp", "dq_cp", "dk_cp", "dv_cp", "dbias_cp"]
    if fp8_mha:
        tensors_to_deq = [out, out_] if not fp8_bwd else tensors
        for i, tensor in enumerate(tensors_to_deq):
            # dbias/dbias_ could be None, so skip check for it
            if tensor is not None:
                tensors_to_deq[i] = tensor.dequantize()
        if not fp8_bwd:
            tensors[0], tensors[5] = tensors_to_deq
    for tensor, name in zip(tensors, names):
        # dbias/dbias_ could be None, so skip check for it
        if tensor is not None:
            assert torch.all(~torch.isnan(tensor)), f"{name} has nan values"
            assert torch.all(~torch.isinf(tensor)), f"{name} has inf values"
    out, dq, dk, dv, dbias, out_, dq_, dk_, dv_, dbias_ = tensors

    ############  compare results between CP and no-CP ############
    if qkv_format == "bshd" or qkv_format == "sbhd":
        if is_training:
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
            if dbias is not None and dbias_ is not None:
                ndim = dbias.ndim
                # Query seq is at dim -2
                seq_q_dim = ndim - 2
                shape_before_seq = dbias.shape[:seq_q_dim]
                seq_q_size = dbias.shape[seq_q_dim]
                seq_kv_size = dbias.shape[-1]
                # Reshape to split seq_q dimension
                dbias = dbias.view(
                    *shape_before_seq,
                    2 * world_size,
                    seq_q_size // (2 * world_size),
                    seq_kv_size,
                )
                # Index select on the newly created dimension (now at position seq_q_dim)
                dbias = dbias.index_select(seq_q_dim, seq_idx)
                dbias_ = dbias_.view(
                    *shape_before_seq, 2, dbias_.shape[seq_q_dim] // 2, seq_kv_size
                )
        else:
            # Forward-only: reshape only out/out_ for comparison
            out = out.view(
                *out.shape[:seq_dim],
                2 * world_size,
                out.shape[seq_dim] // (2 * world_size),
                *out.shape[(seq_dim + 1) :],
            )
            out = out.index_select(seq_dim, seq_idx)
            out_ = out_.view(
                *out_.shape[:seq_dim], 2, out_.shape[seq_dim] // 2, *out_.shape[(seq_dim + 1) :]
            )

    thd_valid_mask = None
    if qkv_format == "thd":
        if is_training:
            dq, out = [x.index_select(0, seq_idx_q).contiguous() for x in [dq, out]]
            dk, dv = [x.index_select(0, seq_idx_kv).contiguous() for x in [dk, dv]]
        else:
            out = out.index_select(0, seq_idx_q).contiguous()

        if packed_contiguous:
            global_valid_mask = torch.zeros(total_tokens, dtype=torch.bool, device=out_.device)
            actual_seqlens = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
            for seq_start, seq_len in zip(cu_seqlens_q_padded[:-1], actual_seqlens):
                global_valid_mask[seq_start : seq_start + seq_len] = True
            thd_valid_mask = global_valid_mask.index_select(0, seq_idx_q)

            cp_tensors = [out_] + ([dq_, dk_, dv_] if is_training else [])
            for name, tensor in zip(["out_", "dq_", "dk_", "dv_"], cp_tensors):
                nnz = torch.count_nonzero(tensor[~thd_valid_mask]).item()
                assert nnz == 0, f"{name} has {nnz} nonzero values in packed THD padding"
        elif is_training:
            cu_seqlens_q_padded = cu_seqlens_q_padded // world_size
            cu_seqlens_q = get_cu_seqlens_on_cp_rank(
                cu_seqlens_q, cu_seqlens_q_padded, world_size, rank, True, True
            )
            cu_pads_q = cu_seqlens_q_padded - cu_seqlens_q
            num_pads_q = cu_pads_q[1:] - cu_pads_q[:-1]
            cu_seqlens_kv_padded = cu_seqlens_kv_padded // world_size
            cu_seqlens_kv = get_cu_seqlens_on_cp_rank(
                cu_seqlens_kv, cu_seqlens_kv_padded, world_size, rank, True, True
            )
            num_pads_kv = (cu_seqlens_kv_padded - cu_seqlens_kv)[1:] - (
                cu_seqlens_kv_padded - cu_seqlens_kv
            )[:-1]
            # FA3 leaves garbage at padding positions despite seqused_q/k (tile spillover).
            # Forward out_ can't be pre-zeroed because FA3's custom op returns out_ as an
            # output rather than mutating it in-place, triggering PyTorch's aliasing constraint.
            # Backward dq/dk/dv CAN be pre-zeroed because FA3 marks them as mutated inputs.
            if fa_pad_between_seqs == "True":
                # out_ is a view inside the CP custom autograd Function, so in-place
                # zeroing is blocked by PyTorch. Clone to break the view relationship.
                out_ = out_.clone()
                for x in [out, out_, dq]:
                    for b in range(config.batch_size):
                        x[
                            cu_seqlens_q_padded[b + 1] - num_pads_q[b] : cu_seqlens_q_padded[b + 1]
                        ] = 0.0
                    x[cu_seqlens_q_padded[-1] :] = 0.0
                for x in [dk, dv]:
                    for b in range(config.batch_size):
                        x[
                            cu_seqlens_kv_padded[b + 1]
                            - num_pads_kv[b] : cu_seqlens_kv_padded[b + 1]
                        ] = 0.0
                    x[cu_seqlens_kv_padded[-1] :] = 0.0
                # Verify CP backward tensors have clean padding (pre-zeroed in context_parallel.py).
                for xname, x, cu, np_ in [
                    ("dq_", dq_, cu_seqlens_q_padded, num_pads_q),
                    ("dk_", dk_, cu_seqlens_kv_padded, num_pads_kv),
                    ("dv_", dv_, cu_seqlens_kv_padded, num_pads_kv),
                ]:
                    nnz = torch.count_nonzero(x[cu[-1] :]).item()
                    assert nnz == 0, (
                        f"{xname} has {nnz} nonzero values in tail padding — "
                        "context_parallel.py should zero padding positions"
                    )
                    for b in range(config.batch_size):
                        if np_[b] > 0:
                            nnz = torch.count_nonzero(x[cu[b + 1] - np_[b] : cu[b + 1]]).item()
                            assert nnz == 0, (
                                f"{xname} has {nnz} nonzero values in batch {b} padding — "
                                "context_parallel.py should zero padding positions"
                            )
    atol, rtol, rmse_tol = get_tols(config, dtype)
    tensors_cp = [out_, dq_, dk_, dv_, dbias_, d_softmax_offset_, max_logit_]
    tensors_no_cp = [out, dq, dk, dv, dbias, d_softmax_offset, max_logit]
    names = ["out", "dq", "dk", "dv", "dbias", "d_softmax_offset", "max_logit"]
    names_cp = [x + "_cp" for x in names]
    names_no_cp = [x + "_no_cp" for x in names]
    for i, t in enumerate(tensors_no_cp):
        if t is not None:
            # Uneven CP decompositions produced sparse BF16 dQ outliers from
            # accumulation order. Keep every other tensor on the strict check.
            use_rmse_tolerance = dtype == "fp8" or (
                names[i] == "dq" and benchmark_iters and "," in thd_seqlen_pattern
            )
            if "softmax_offset" not in names[i] and "max_logit" not in names[i]:
                if qkv_format == "bshd":
                    # Compare the two sequence chunks separately
                    # Compare dbias
                    if names[i] == "dbias":
                        # Compare the two chunks along dimension 2 (the split sequence dimension)
                        seq_q_dim_bias = 2
                        ndim_bias = t.ndim
                        slice_0 = [slice(None)] * ndim_bias
                        slice_0[seq_q_dim_bias] = 0
                        slice_1 = [slice(None)] * ndim_bias
                        slice_1[seq_q_dim_bias] = 1
                        compare_and_assert(
                            t[tuple(slice_0)],
                            tensors_cp[i][tuple(slice_0)],
                            names_no_cp[i],
                            names_cp[i],
                            atol,
                            rtol,
                            rmse_tol,
                            use_rmse_tolerance,
                        )
                        compare_and_assert(
                            t[tuple(slice_1)],
                            tensors_cp[i][tuple(slice_1)],
                            names_no_cp[i],
                            names_cp[i],
                            atol,
                            rtol,
                            rmse_tol,
                            use_rmse_tolerance,
                        )
                    # Compare Q/K/V/out
                    else:
                        #  Compare the two chunks along dimension 1 (the split sequence dimension)
                        compare_and_assert(
                            t[:, 0],
                            tensors_cp[i][:, 0],
                            names_no_cp[i],
                            names_cp[i],
                            atol,
                            rtol,
                            rmse_tol,
                            use_rmse_tolerance,
                        )
                        compare_and_assert(
                            t[:, 1],
                            tensors_cp[i][:, 1],
                            names_no_cp[i],
                            names_cp[i],
                            atol,
                            rtol,
                            rmse_tol,
                            use_rmse_tolerance,
                        )
                elif qkv_format == "sbhd":
                    # Compare the two sequence chunks separately
                    # Compare dbias (same as BSHD)
                    if names[i] == "dbias":
                        # Same as bshd: Compare the two chunks along dimension 2 (the split sequence dimension)
                        seq_q_dim_bias = 2
                        ndim_bias = t.ndim
                        slice_0 = [slice(None)] * ndim_bias
                        slice_0[seq_q_dim_bias] = 0
                        slice_1 = [slice(None)] * ndim_bias
                        slice_1[seq_q_dim_bias] = 1
                        compare_and_assert(
                            t[tuple(slice_0)],
                            tensors_cp[i][tuple(slice_0)],
                            names_no_cp[i],
                            names_cp[i],
                            atol,
                            rtol,
                            rmse_tol,
                            use_rmse_tolerance,
                        )
                        compare_and_assert(
                            t[tuple(slice_1)],
                            tensors_cp[i][tuple(slice_1)],
                            names_no_cp[i],
                            names_cp[i],
                            atol,
                            rtol,
                            rmse_tol,
                            use_rmse_tolerance,
                        )
                    # Compare Q/K/V/out
                    else:
                        #  Compare the two chunks along dimension 0 (the split sequence dimension)
                        compare_and_assert(
                            t[0],
                            tensors_cp[i][0],
                            names_no_cp[i],
                            names_cp[i],
                            atol,
                            rtol,
                            rmse_tol,
                            use_rmse_tolerance,
                        )
                        compare_and_assert(
                            t[1],
                            tensors_cp[i][1],
                            names_no_cp[i],
                            names_cp[i],
                            atol,
                            rtol,
                            rmse_tol,
                            use_rmse_tolerance,
                        )
                elif qkv_format == "thd":
                    if thd_valid_mask is not None:
                        t = t[thd_valid_mask]
                        tensors_cp[i] = tensors_cp[i][thd_valid_mask]
                    compare_and_assert(
                        t,
                        tensors_cp[i],
                        names_no_cp[i],
                        names_cp[i],
                        atol,
                        rtol,
                        rmse_tol,
                        use_rmse_tolerance,
                    )
            else:
                compare_and_assert(
                    t,
                    tensors_cp[i],
                    names_no_cp[i],
                    names_cp[i],
                    atol,
                    rtol,
                    rmse_tol,
                    use_rmse_tolerance,
                )
            logging.info(f"[Rank {rank}] CP vs no-CP: {names[i]} matches")

    if benchmark_iters:
        (
            benchmark_cu_seqlens_q,
            benchmark_cu_seqlens_kv,
            benchmark_cu_seqlens_q_padded,
            benchmark_cu_seqlens_kv_padded,
        ) = benchmark_metadata

        # Correctness tensors are not part of the timed workload. Release them
        # before allocating the reusable benchmark leaves.
        if thd_valid_mask is not None:
            del cp_tensors
        del tensors, tensors_cp, tensors_no_cp
        del out, dq, dk, dv, dbias, out_, dq_, dk_, dv_, dbias_
        del d_softmax_offset, d_softmax_offset_, max_logit, max_logit_
        for tensor in [q_, k_, v_]:
            tensor.grad = None

        warmup_iters = 10
        q_b, k_b, v_b = [x.detach().requires_grad_() for x in [q_, k_, v_]]
        elapsed_ms = []
        for iteration in range(warmup_iters + benchmark_iters):
            for tensor in [q_b, k_b, v_b]:
                tensor.grad = None
            dist.barrier()
            torch.cuda.synchronize()
            start = time.perf_counter()
            with fp8_context:
                out_b = core_attn(
                    q_b,
                    k_b,
                    v_b,
                    core_attention_bias_type=config.attn_bias_type,
                    core_attention_bias=bias_,
                    cu_seqlens_q=benchmark_cu_seqlens_q,
                    cu_seqlens_kv=benchmark_cu_seqlens_kv,
                    cu_seqlens_q_padded=benchmark_cu_seqlens_q_padded,
                    cu_seqlens_kv_padded=benchmark_cu_seqlens_kv_padded,
                    pad_between_seqs=has_inter_sequence_padding,
                    fp8_output=False,
                )
                if isinstance(out_b, tuple):
                    out_b = out_b[0]
                out_b.backward(dout_)
            torch.cuda.synchronize()
            local_ms = (time.perf_counter() - start) * 1000

            # Aggregate outside the timed interval. Every rank reports the same
            # per-iteration distributed latency sample.
            global_ms = torch.tensor(local_ms, dtype=torch.float32, device=q_b.device)
            dist.all_reduce(global_ms, op=dist.ReduceOp.MAX)
            if iteration >= warmup_iters:
                elapsed_ms.append(global_ms.item())
            del out_b

        ordered_ms = sorted(elapsed_ms)
        middle = len(ordered_ms) // 2
        median_ms = (
            ordered_ms[middle]
            if len(ordered_ms) % 2
            else (ordered_ms[middle - 1] + ordered_ms[middle]) / 2
        )
        mean_ms = sum(elapsed_ms) / len(elapsed_ms)
        print(
            f"BENCH_RESULT rank={rank} model={model} partition={partition} "
            f"cp={world_size} median_ms={median_ms:.3f} mean_ms={mean_ms:.3f} "
            f"min_ms={min(elapsed_ms):.3f} max_ms={max(elapsed_ms):.3f} "
            f"warmup={warmup_iters} iters={benchmark_iters}",
            flush=True,
        )

    # Teardown on the success path. Pool mode: cp_comm_group / cp_comm_sub_groups
    # point at pool-shared groups owned by the pool runner (which destroys them
    # at pool shutdown), and the main PG is also pool-owned — both branches
    # below are no-ops. Single-shot mode: destroy what we created here. If the
    # body above raises, we skip this — the subprocess dies at function return
    # and NCCL releases the communicators with the process.
    if not _reusing_pool_groups:
        if cp_comm_group is not None:
            try:
                dist.destroy_process_group(cp_comm_group)
            except Exception:
                pass
        for g in cp_comm_sub_groups:
            try:
                dist.destroy_process_group(g)
            except Exception:
                pass
    if not _pool_managed_pg:
        try:
            dist.destroy_process_group()
        except Exception:
            pass


def main(**kwargs):
    run_dpa_with_cp(**kwargs)


if __name__ == "__main__":
    kwargs = dict(arg.split("=") for arg in sys.argv[2:])
    main(**kwargs)
