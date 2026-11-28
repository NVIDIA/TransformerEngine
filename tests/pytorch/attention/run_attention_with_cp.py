# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import copy
import os
import sys
import logging
from contextlib import nullcontext
import torch
import torch.distributed as dist
from transformer_engine.pytorch.attention.dot_product_attention.context_parallel import (
    get_cu_seqlens_on_cp_rank,
    get_thd_partitioned_indices,
)
from transformer_engine.pytorch.attention.dot_product_attention.utils import (
    combine_and_quantize,
    get_thd_padding_mask,
)
from transformer_engine.pytorch import DType
from test_attention_with_cp import (
    model_configs_flash_attn,
    model_configs_fused_attn,
)
from transformer_engine.pytorch import (
    autocast,
    CPLoadBalancingStrategy,
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

dtypes = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp8": torch.bfloat16}


def generate_input_shapes(
    qkv_format: str,
    config: ModelConfig,
    world_size: int,
    kernel_backend: str,
    fa_pad_between_seqs: str = "False",
    load_balancing_strategy=CPLoadBalancingStrategy.DUAL_CHUNK_SWAP,
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
        if load_balancing_strategy is CPLoadBalancingStrategy.NO_LOAD_BALANCE:
            assert config.batch_size == 2
            if kernel_backend == "FlashAttention" and fa_pad_between_seqs == "False":
                seqlens_q = torch.tensor(
                    [config.max_seqlen_q - 2, config.max_seqlen_q], dtype=torch.int32
                )
                assert seqlens_q.sum().item() % world_size == 0
                seqlens_q_padded = seqlens_q
            else:
                # Exercise both document padding and a CP chunk boundary inside a document.
                seqlens_q = torch.tensor(
                    [config.max_seqlen_q - 2, config.max_seqlen_q - 1],
                    dtype=torch.int32,
                )
                padded_total = 2 * config.max_seqlen_q
                assert padded_total % world_size == 0
                seqlens_q_padded = torch.tensor(
                    [config.max_seqlen_q - 1, config.max_seqlen_q + 1],
                    dtype=torch.int32,
                )
        else:
            seqlens_q = torch.randint(0, config.max_seqlen_q + 1, [config.batch_size]).to(
                torch.int32
            )
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


def assert_matches_independent_reference(out, q, k, v, config, qkv_format, dtype, softmax_scale):
    """Check the non-CP output against a float64 reference instead of against another run.

    The CP suite compares a context-parallel run with a non-CP run *of the same backend*
    (``tensors_no_cp`` vs ``tensors_cp``). That validates the context-parallel plumbing and nothing
    about the kernel beneath it: any error affecting both sides equally cancels and the comparison
    passes. It is not a theoretical gap -- a backend was found returning an all-zero output while
    two of three ``cp_comm_type`` values reported PASS, because zeros matched zeros. The same
    structure then reported a partial *fix* as a regression, since correcting one side made the two
    disagree.

    This anchors one (batch, head) slice of the non-CP output to an independently computed
    reference, so a systematically wrong kernel cannot cancel itself out. It is a sanity bound, not
    a precision test -- the CP/no-CP comparison already covers precision. It is meant to catch
    catastrophic wrongness: zeros, NaNs, a mask applied in the wrong place.

    float64 rather than float32: torch computes fp32 matmuls in TF32 on Ampere and newer, whose
    significand is 11 bits -- the same as fp16 -- so an fp32 reference cannot judge a bf16 kernel.

    Configurations the reference does not model (fp8, bias, non-vanilla softmax, thd/ragged,
    padding masks) are skipped rather than approximated, so this never fails for a reason it cannot
    explain.
    """
    if qkv_format not in ("bshd", "sbhd"):
        return
    if dtype == "fp8" or config.attn_bias_type != "no_bias" or config.softmax_type != "vanilla":
        return
    # Softcap rescales the scores through tanh before the softmax. Modelling it here would mean
    # duplicating that formula in the reference, which is how a reference starts reimplementing the
    # thing it is supposed to check independently. Skipped instead.
    if getattr(config, "softcap", 0.0):
        return
    if "padding" in config.attn_mask_type:
        return
    if out is None or q is None:
        return
    # DotProductAttention returns [..., h * d_v], i.e. 3D. Restore the head axis from q and v --
    # without this the guard below would return early and the whole check would silently do
    # nothing, which is the failure mode this function exists to catch in the first place.
    if out.dim() == 3:
        out = out.view(*out.shape[:2], q.shape[2], v.shape[-1])
    if out.dim() != 4:
        return

    seq_dim = 1 if qkv_format == "bshd" else 0
    # One (batch, head) slice: the full float64 score matrix would be gigabytes.
    take = (
        lambda t, head: (t[0, :, head] if qkv_format == "bshd" else t[:, 0, head]).detach().double()
    )
    qhead, kvhead = 0, 0  # GQA groups are contiguous, so q head 0 belongs to kv group 0
    qs, ks, vs = take(q, qhead), take(k, kvhead), take(v, kvhead)
    got = take(out, qhead)
    sq, skv = qs.shape[0], ks.shape[0]

    scores = (qs @ ks.transpose(-1, -2)) * softmax_scale
    window = getattr(config, "window_size", None)
    left, right = window if window is not None else (-1, -1)
    if "causal" in config.attn_mask_type:
        right = 0
    # Bottom-right alignment when the lengths differ; identical to top-left when they do not.
    offset = (skv - sq) if "bottom_right" in config.attn_mask_type else 0
    blocked = torch.zeros(sq, skv, dtype=torch.bool, device=qs.device)
    if right != -1:
        blocked |= torch.ones(sq, skv, dtype=torch.bool, device=qs.device).triu(offset + right + 1)
    if left != -1:
        blocked |= torch.ones(sq, skv, dtype=torch.bool, device=qs.device).tril(offset - left - 1)
    scores = scores.masked_fill(blocked, float("-inf"))
    reference = torch.softmax(scores, dim=-1) @ vs

    # Reported separately from the tolerance: an all-zero or non-finite output is the specific
    # failure this exists to catch, and it should be named rather than shown as a large error.
    assert torch.isfinite(out).all(), "non-CP output is not finite"
    assert out.abs().max() > 0, (
        "non-CP output is all zeros -- the backend computed nothing. The CP/no-CP comparison"
        " cannot see this, because both sides are equally zero."
    )
    denom = reference.abs().max().clamp_min(1e-12)
    error = float((got - reference).abs().max() / denom)
    assert error < 5e-2, (
        "non-CP output disagrees with an independent float64 reference: relative error"
        f" {error:.3e}. This is a loose sanity bound, so exceeding it points at a wrong mask or a"
        " wrong kernel rather than a precision issue."
    )


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
    load_balancing_strategy="DUAL_CHUNK_SWAP",
    softcap="0.0",
    log_level=logging.WARNING,
):
    """Test DotProductAttention module with context parallelism"""
    logging.root.setLevel(log_level)
    load_balancing_strategy = CPLoadBalancingStrategy[load_balancing_strategy]
    # When is_training is False, gradient outputs are None.
    is_training = is_training == "True"
    pad_between_seqs = None
    if qkv_format == "thd":
        # Keep this in sync with generate_input_shapes so DPA gets the explicit
        # padding state without a GPU-to-CPU sync.
        pad_between_seqs = kernel_backend == "FusedAttention" or fa_pad_between_seqs == "True"

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
        config = copy.deepcopy(model_configs_flash_attn[model])
    if kernel_backend == "FusedAttention":
        os.environ["NVTE_FUSED_ATTN"] = "1"
        if model in model_configs_fused_attn:
            config = copy.deepcopy(model_configs_fused_attn[model])
        else:
            assert False, f"{model=} is not a known FusedAttention CP config!"
    assert config.attn_mask_type in [
        "causal",
        "no_mask",
    ], f"{config.attn_mask_type=} is not supported!"
    if qkv_format == "thd":
        if "causal" in config.attn_mask_type:
            config.attn_mask_type = "padding_causal"
        else:
            config.attn_mask_type = "padding"
    config.softcap = float(softcap)

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
        softcap=config.softcap,
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
        load_balancing_strategy,
    )
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
    q, k, v, dout = [x.clone().detach() for x in [q_orig, k_orig, v_orig, dout_orig]]
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
            pad_between_seqs=pad_between_seqs,
            fp8_output=fp8_mha,
        )
        if config.return_max_logit:
            out, max_logit = out
        # Anchor the non-CP result to something other than another run of the same backend, before
        # it is reshaped for the CP comparison. See the function docstring for why this exists.
        assert_matches_independent_reference(
            out, q, k, v, config, qkv_format, dtype, core_attn.softmax_scale
        )
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
    q_, k_, v_, dout_, *rest = [
        x.clone().detach()
        for x in [q_orig, k_orig, v_orig, dout_orig] + ([] if bias is None else [bias])
    ]
    bias_ = rest[0] if len(rest) else None
    if qkv_format in ("bshd", "sbhd"):
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
            load_balancing_strategy=load_balancing_strategy,
        )
        seq_idx_kv = get_thd_partitioned_indices(
            cu_seqlens_kv_padded,
            k_.shape[0],
            world_size,
            rank,
            device=k_.device,
            load_balancing_strategy=load_balancing_strategy,
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
        load_balancing_strategy,
    )
    if is_training and config.softmax_type != "vanilla":
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
            pad_between_seqs=pad_between_seqs,
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
    if qkv_format in ("bshd", "sbhd"):
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

    elif qkv_format == "thd":
        if is_training:
            dq, out = [x.index_select(0, seq_idx_q).contiguous() for x in [dq, out]]
            dk, dv = [x.index_select(0, seq_idx_kv).contiguous() for x in [dk, dv]]
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
            # FA3/FA4 kernels may leave physically padded rows unwritten. The CP
            # implementation must clean them after partial-output/gradient merges.
            if fa_pad_between_seqs == "True":
                out_padding_mask = get_thd_padding_mask(
                    out_.shape[0], cu_seqlens_q, cu_seqlens_q_padded
                )
                nnz = torch.count_nonzero(out_[out_padding_mask]).item()
                assert nnz == 0, (
                    f"out_ has {nnz} nonzero values in THD padding — "
                    "context_parallel.py should zero padding positions"
                )

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
        else:
            out = out.index_select(0, seq_idx_q).contiguous()
            out_ = out_

    atol, rtol, rmse_tol = get_tols(config, dtype)
    tensors_cp = [out_, dq_, dk_, dv_, dbias_, d_softmax_offset_, max_logit_]
    tensors_no_cp = [out, dq, dk, dv, dbias, d_softmax_offset, max_logit]
    names = ["out", "dq", "dk", "dv", "dbias", "d_softmax_offset", "max_logit"]
    names_cp = [x + "_cp" for x in names]
    names_no_cp = [x + "_no_cp" for x in names]
    is_fp8 = dtype == "fp8"
    for i, t in enumerate(tensors_no_cp):
        if t is not None:
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
                            is_fp8,
                        )
                        compare_and_assert(
                            t[tuple(slice_1)],
                            tensors_cp[i][tuple(slice_1)],
                            names_no_cp[i],
                            names_cp[i],
                            atol,
                            rtol,
                            rmse_tol,
                            is_fp8,
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
                            is_fp8,
                        )
                        compare_and_assert(
                            t[tuple(slice_1)],
                            tensors_cp[i][tuple(slice_1)],
                            names_no_cp[i],
                            names_cp[i],
                            atol,
                            rtol,
                            rmse_tol,
                            is_fp8,
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
                        t,
                        tensors_cp[i],
                        names_no_cp[i],
                        names_cp[i],
                        atol,
                        rtol,
                        rmse_tol,
                        is_fp8,
                    )
            else:
                compare_and_assert(
                    t, tensors_cp[i], names_no_cp[i], names_cp[i], atol, rtol, rmse_tol, is_fp8
                )
            logging.info(f"[Rank {rank}] CP vs no-CP: {names[i]} matches")

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


def _check_megaup_trusted_state(candidate, reference, x, dy, rope, metadata, compare, group):
    """Trusted fixed contracts reject changes locally; coordinated resets revalidate."""
    from unittest.mock import patch
    import mega_up.ops as ops
    from transformer_engine.pytorch.attention import _megaup as implementation

    cp = dist.get_world_size(group)
    rank = dist.get_rank(group)
    cu = metadata["cu_seqlens_q"].clone()
    maximum = metadata["max_seqlen_q"]
    object.__setattr__(candidate, "_megaup_thd_metadata_cache", None)
    object.__setattr__(candidate, "_megaup_validated_contract", None)

    def arguments(bounds, max_length=maximum):
        return dict(
            cu_seqlens_q=bounds,
            cu_seqlens_kv=bounds,
            cu_seqlens_q_padded=bounds,
            cu_seqlens_kv_padded=bounds,
            max_seqlen_q=max_length,
            max_seqlen_kv=max_length,
        )

    def step(bounds, expected, max_length=maximum, rows=None):
        source = x if rows is None else x[:rows]
        gradient = dy if rows is None else dy[:rows]
        xr, xc = (source.detach().clone().requires_grad_() for _ in range(2))
        reference.zero_grad(set_to_none=True)
        candidate.zero_grad(set_to_none=True)
        kwargs = arguments(bounds, max_length)
        with patch.dict(os.environ, NVTE_MHA_MEGAUP="0"):
            yr = reference(xr, rotary_pos_emb=rope, **kwargs)
            yr.backward(gradient)
        with patch.dict(os.environ, NVTE_MHA_MEGAUP="1", NVTE_MHA_MEGAUP_TRUSTED="1"), patch.object(
            dist, "all_gather_object", wraps=dist.all_gather_object
        ) as gathers, patch.object(
            ops, "_collective_validate", wraps=ops._collective_validate
        ) as validates, patch.object(
            implementation, "_thd_values", wraps=implementation._thd_values
        ) as reads:
            yc = candidate(xc, rotary_pos_emb=rope, **kwargs)
            yc.backward(gradient)
            counts = (gathers.call_count, validates.call_count, reads.call_count)
        assert counts == expected, (rank, counts, expected)
        compare(yc, yr, "trusted dynamic output")
        compare(xc.grad, xr.grad, "trusted dynamic dInput")
        for (name, actual), (_, wanted) in zip(
            candidate.named_parameters(), reference.named_parameters()
        ):
            compare(actual.grad, wanted.grad, f"trusted dynamic gradient {name}")
        assert candidate._megaup_validated_contract["ready"]

    def reset():
        # Tests call this on every rank, never while peers are in a payload.
        object.__setattr__(candidate, "_megaup_validated_contract", None)

    def reject(bounds, max_length=maximum, rows=None):
        with patch.object(dist, "all_gather_object", wraps=dist.all_gather_object) as gathers:
            try:
                implementation.prepare_megaup(
                    candidate,
                    x if rows is None else x[:rows],
                    eligible_call=True,
                    rotary_pos_emb=rope,
                    attn_mask_type="padding_causal",
                    **arguments(bounds, max_length),
                )
            except RuntimeError as error:
                assert "Trusted MegaUP" in str(error)
            else:
                raise AssertionError("Initialized trusted contract accepted a changed input")
            assert gathers.call_count == 0

    step(cu, (16, 14, 1))
    step(cu, (0, 0, 0))
    cu = cu.clone()
    step(cu, (0, 0, 1))  # Equivalent new identity rereads without changing the contract.
    cu.add_(0)
    step(cu, (0, 0, 1))  # Version-tracked edit with identical values also stays valid.
    cu[1] += 2 * cp
    reject(cu)
    reset()
    step(cu, (16, 14, 0))  # Rejected call already refreshed the metadata cache.
    step(cu, (0, 0, 0))
    one_rank_copy = cu.clone() if rank == 0 else cu
    step(one_rank_copy, (0, 0, int(rank == 0)))
    reject(one_rank_copy, max_length=maximum + 2 * cp)
    reset()
    step(one_rank_copy, (16, 14, 0), max_length=maximum + 2 * cp)
    # Deliberate rank-local errors are tested without sending other ranks into payloads.
    dist.barrier(group=group)
    if rank == 0:
        reject(one_rank_copy.to(torch.int64), max_length=maximum + 2 * cp)
        original_group = candidate.core_attention.cp_group
        candidate.core_attention.cp_group = None
        try:
            reject(one_rank_copy, max_length=maximum + 2 * cp)
        finally:
            candidate.core_attention.cp_group = original_group
    dist.barrier(group=group)

    # Invalid maxima cannot compare equal to an int inside a warmed cache key.
    with patch.object(implementation, "_thd_values", wraps=implementation._thd_values) as reads:
        try:
            implementation._cached_thd_metadata(
                candidate,
                x,
                cp,
                (one_rank_copy,) * 4,
                (torch.tensor(maximum + 2 * cp, device="cuda"), maximum + 2 * cp),
            )
        except ValueError:
            pass
        else:
            raise AssertionError("Tensor maximum bypassed validation via a cache hit")
        assert reads.call_count == 1
    with torch.inference_mode():
        inference_cu = cu.clone()
    with patch.object(implementation, "_thd_values", wraps=implementation._thd_values) as reads:
        for _ in range(2):
            implementation._cached_thd_metadata(
                candidate, x, cp, (inference_cu,) * 4, (maximum, maximum)
            )
        assert reads.call_count == 2
    short = torch.tensor([0, 32 * cp, 64 * cp], device="cuda", dtype=torch.int32)
    reject(short, rows=64)
    reset()
    step(short, (16, 14, 0), rows=64)

    # Same-contract retained forwards share readiness, but a graph from a reset
    # state cannot certify a new state. Only this test deliberately resets while
    # old graphs exist; real callers reset between completed steps.
    a = torch.tensor([0, 32 * cp, 80 * cp], device="cuda", dtype=torch.int32)
    reset()
    candidate.zero_grad(set_to_none=True)
    with patch.dict(os.environ, NVTE_MHA_MEGAUP="1", NVTE_MHA_MEGAUP_TRUSTED="1"):
        ya = candidate(x.clone().requires_grad_(), rotary_pos_emb=rope, **arguments(a))
        state_a = candidate._megaup_validated_contract
        with patch.object(dist, "all_gather_object", wraps=dist.all_gather_object) as gathers:
            yb = candidate(x.clone().requires_grad_(), rotary_pos_emb=rope, **arguments(a))
            assert gathers.call_count == 7  # Nested forward checks only, no TE checks.
        assert candidate._megaup_validated_contract is state_a and not state_a["ready"]
        reset()
        yc = candidate(x.clone().requires_grad_(), rotary_pos_emb=rope, **arguments(a))
        state_c = candidate._megaup_validated_contract
    with patch.dict(os.environ, NVTE_MHA_MEGAUP_TRUSTED="0"):
        yb.backward(dy)
    assert state_a["ready"] and not state_c["ready"]
    ya.backward(dy)
    assert not state_c["ready"]
    yc.backward(dy)
    assert state_c["ready"]
    print(
        f"Trusted state passed: rank={rank}, CP={cp}; cold16/14, warm0/0/0, fresh0/0/1"
        " (gathers/library/metadata reads)",
        flush=True,
    )


def run_mha_with_megaup(gqa_only=False):
    """Check packed rectangular MHA against native CP, including retained graphs."""
    from datetime import timedelta
    from inspect import signature
    from itertools import accumulate
    from unittest.mock import patch

    import mega_up.ops as ops
    from mega_up.ops import (
        PreparedHeadShardPullGemm,
        PreparedHeadShardPullGemmTHD,
        PreparedPeerWriteGemm,
        PreparedPeerWriteGemmTHD,
    )

    from transformer_engine.pytorch import MultiheadAttention
    from transformer_engine.pytorch.attention import RotaryPositionEmbedding
    from transformer_engine.pytorch.attention.dot_product_attention import (
        context_parallel as cp_ops,
    )
    from transformer_engine.pytorch.attention.dot_product_attention.dot_product_attention import (
        _attention_backends,
    )

    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl", timeout=timedelta(minutes=10))
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    assert world_size in (2, 4, 8)
    cases = [(list(range(world_size)), 2, 34, "sbhd", True)]
    cases.append((list(range(world_size)), 2, 64, "bshd", False))
    # Fresh groups isolate each fixed-capacity arena, including unequal THD samples.
    cases.extend(
        (list(range(world_size)), len(lengths), lengths, "thd", rope)
        for lengths, rope in [((34, 34), True), ((32, 48), True), ((64,), False)]
    )
    # Preserve subgroup-last ordering: the subgroup-to-world transition stalled
    # during validation; the mechanism remains unproven.
    if world_size == 4:
        cases.append(([1, 3], 2, 34, "bshd", True))
    cases = [(*case, 2048, 16, 16) for case in cases]
    if gqa_only:
        # Model width greater/smaller than Q width, then packed THD groups.
        cases = [
            (list(range(world_size)), 2, 32, "sbhd", True, 5120, 32, 8),
            (list(range(world_size)), 2, 32, "bshd", False, 5120, 64, 8),
            (list(range(world_size)), 2, (32, 48), "thd", True, 5120, 64, 8),
        ]
        if world_size == 2:
            cases.extend(
                (list(range(world_size)), 2, (32, 48), "thd", True, width, heads, 2)
                for width, heads in ((4096, 32), (8192, 64))
            )
    cp_calls = [0, 0]
    cp_function = cp_ops.AttnFuncWithCPAndQKVOA2A
    original_cp_forward, original_cp_backward = cp_function.forward, cp_function.backward

    def cp_forward(*args, **kwargs):
        call = signature(original_cp_forward).bind(*args, **kwargs).arguments
        if call["cp_external_exchange"] is None:
            return original_cp_forward(*args, **kwargs)
        cp_calls[0] += 1
        assert dist.get_world_size(call["cp_group"]) == len(ranks)
        assert call["q"].shape[-2] == heads // len(ranks)
        assert call["k"].shape[-2] == call["v"].shape[-2] == groups // len(ranks)
        # Trusted-state tests deliberately change maximum lengths between calls.
        selected = _attention_backends["attention_params"]
        assert call["max_seqlen_q"] == selected.max_seqlen_q
        assert call["max_seqlen_kv"] == selected.max_seqlen_kv
        # The real CP autograd Function must execute, with no inner payload exchange.
        with patch.object(
            cp_ops, "flash_attn_a2a_communicate", side_effect=AssertionError("duplicate A2A")
        ):
            return original_cp_forward(*args, **kwargs)

    def cp_backward(ctx, *args):
        if not ctx.cp_exchanges_external:
            return original_cp_backward(ctx, *args)
        cp_calls[1] += 1
        with patch.object(
            cp_ops, "flash_attn_a2a_communicate", side_effect=AssertionError("duplicate A2A")
        ):
            return original_cp_backward(ctx, *args)

    peer_calls = 0
    pull_calls = 0
    original_peer = PreparedPeerWriteGemm.__call__
    original_pull = PreparedHeadShardPullGemm.__call__
    original_peer_thd = PreparedPeerWriteGemmTHD.__call__
    original_pull_thd = PreparedHeadShardPullGemmTHD.__call__

    def peer(*args, **kwargs):
        nonlocal peer_calls
        peer_calls += 1
        return original_peer(*args, **kwargs)

    def pull(prepared, *args, **kwargs):
        nonlocal pull_calls
        pull_calls += 1
        return original_pull(prepared, *args, **kwargs)

    def peer_thd(*args, **kwargs):
        nonlocal peer_calls
        peer_calls += 1
        return original_peer_thd(*args, **kwargs)

    def pull_thd(prepared, *args, **kwargs):
        nonlocal pull_calls
        pull_calls += 1
        return original_pull_thd(prepared, *args, **kwargs)

    def uniform(condition, name):
        flag = torch.tensor(int(condition), device="cuda", dtype=torch.int32)
        dist.all_reduce(flag, op=dist.ReduceOp.MIN, group=group)
        assert flag.item(), name

    def compare(actual, expected, name):
        # Match the established BF16 CP tolerance, with a relative-norm guard
        # so small tensors cannot pass solely on the absolute tolerance.
        torch.testing.assert_close(
            actual, expected, atol=2.5e-2, rtol=2.5e-2, msg=lambda detail: f"{name}: {detail}"
        )
        relative = (actual.float() - expected.float()).norm() / expected.float().norm().clamp_min(
            1e-12
        )
        assert relative < 2.5e-2, f"{name}: relative L2={relative.item()}"

    with patch.object(PreparedPeerWriteGemm, "__call__", peer), patch.object(
        PreparedHeadShardPullGemm, "__call__", pull
    ), patch.object(PreparedPeerWriteGemmTHD, "__call__", peer_thd), patch.object(
        PreparedHeadShardPullGemmTHD, "__call__", pull_thd
    ), patch.object(
        cp_function, "forward", cp_forward
    ), patch.object(
        cp_function, "backward", cp_backward
    ):
        for ranks, batch, sequence, qkv_format, use_rope, width, heads, groups in cases:
            group = dist.new_group(ranks)
            if rank in ranks:
                torch.manual_seed(1234)
                thd = qkv_format == "thd"
                lengths = sequence if thd else (sequence,) * batch
                maximum = max(lengths) * len(ranks)
                metadata = {}
                if thd:
                    cu = torch.tensor(
                        [0] + list(accumulate(length * len(ranks) for length in lengths)),
                        device="cuda",
                        dtype=torch.int32,
                    )
                    metadata = dict(
                        cu_seqlens_q=cu,
                        cu_seqlens_kv=cu,
                        cu_seqlens_q_padded=cu,
                        cu_seqlens_kv_padded=cu,
                        max_seqlen_q=maximum,
                        max_seqlen_kv=maximum,
                    )
                    # Independent token IDs prove both native permutations, including
                    # each sample's second (reverse-rank) half in the CP partition.
                    from transformer_engine.pytorch.attention.dot_product_attention import (
                        context_parallel as cp_ops,
                    )

                    bounds = cu.tolist()
                    rank_ids = torch.cat(
                        [
                            torch.arange(
                                a + chunk * ((z - a) // (2 * len(ranks))),
                                a + (chunk + 1) * ((z - a) // (2 * len(ranks))),
                                device="cuda",
                            )
                            for r in range(len(ranks))
                            for a, z in zip(bounds, bounds[1:])
                            for chunk in (r, 2 * len(ranks) - 1 - r)
                        ]
                    )
                    ids = (
                        torch.arange(bounds[-1], device="cuda", dtype=torch.float32)
                        .view(-1, 1, 1)
                        .repeat(1, 1, 4)
                    )
                    ordered = cp_ops.thd_sequence_order_to_cp_rank_order(ids, cu, len(ranks))
                    torch.testing.assert_close(ordered, ids[rank_ids], atol=0, rtol=0)
                    torch.testing.assert_close(
                        cp_ops.thd_cp_rank_order_to_sequence_order(ids, cu, len(ranks)),
                        ids[torch.argsort(rank_ids)],
                        atol=0,
                        rtol=0,
                    )
                arguments = dict(
                    hidden_size=width,
                    num_attention_heads=heads,
                    num_gqa_groups=groups,
                    kv_channels=128,
                    attention_dropout=0.0,
                    attn_mask_type=(
                        ("padding_causal" if use_rope else "padding")
                        if thd
                        else ("causal" if use_rope else "no_mask")
                    ),
                    params_dtype=torch.bfloat16,
                    bias=False,
                    input_layernorm=False,
                    fuse_qkv_params=True,
                    qkv_weight_interleaved=(groups != heads or use_rope),
                    qkv_format=qkv_format,
                )
                reference = MultiheadAttention(**arguments).cuda()
                candidate = MultiheadAttention(**arguments).cuda()
                candidate.load_state_dict(reference.state_dict())
                original_keys = set(candidate.state_dict())
                original_parameters = {
                    name: id(value) for name, value in candidate.named_parameters()
                }
                for module in (reference, candidate):
                    module.set_context_parallel_group(group, ranks, torch.cuda.Stream(), "a2a")
                rope = RotaryPositionEmbedding(128)(maximum).cuda() if use_rope else None
                shape = (
                    (sum(lengths), width)
                    if thd
                    else (
                        (batch, sequence, width)
                        if qkv_format == "bshd"
                        else (sequence, batch, width)
                    )
                )
                if groups != heads:
                    from transformer_engine.pytorch.attention._megaup import prepare_megaup

                    # Independent head IDs expose actual packed group ownership:
                    # rank r must receive its complete (queries,K,V) groups.
                    probe = torch.zeros(shape, device="cuda", dtype=torch.bfloat16)
                    probe[..., 0] = dist.get_rank(group) + 1
                    plan = prepare_megaup(
                        candidate,
                        probe,
                        eligible_call=True,
                        rotary_pos_emb=rope,
                        attn_mask_type=arguments["attn_mask_type"],
                        **metadata,
                    )
                    assert plan is not None
                    parameter = candidate.qkv.weight
                    original = parameter.detach().clone()
                    head_ids = torch.arange(1, heads + 2 * groups + 1, device="cuda")
                    with torch.no_grad():
                        parameter.zero_()
                        parameter[:, 0].copy_(head_ids.repeat_interleave(128))
                        actual = candidate.qkv(plan.local_rows(probe), _cp_projection=plan.peer)
                        parameter.copy_(original)
                    del original
                    shard = head_ids.chunk(len(ranks))[dist.get_rank(group)].repeat_interleave(128)
                    factors = torch.arange(1, len(ranks) + 1, device="cuda").repeat_interleave(
                        plan.local_rows(probe).shape[0]
                    )
                    expected = (factors[:, None] * shard).to(torch.bfloat16)
                    uniform(torch.equal(actual, expected), "GQA head-group routing")
                    # Rectangular packed weights retain original Parameter version checks.
                    result = candidate.qkv(plan.local_rows(probe), _cp_projection=plan.peer)
                    with torch.no_grad():
                        parameter.add_(0)
                    version_rejected = False
                    try:
                        result.sum().backward()
                    except RuntimeError as error:
                        version_rejected = "modified by an inplace operation" in str(error)
                    uniform(version_rejected, "GQA projection ignored original weight mutation")
                    del result, actual, expected, probe, plan
                # The second step must consume updated weights, not a cached transpose.
                for step in range(2):
                    reference.zero_grad(set_to_none=True)
                    candidate.zero_grad(set_to_none=True)
                    before = (peer_calls, pull_calls)
                    cp_before = tuple(cp_calls)
                    graphs = []
                    for microbatch in range(2):
                        torch.manual_seed(1000 + rank * 100 + step * 10 + microbatch)
                        x = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
                        xr = x.clone().requires_grad_()
                        xc = x.clone().requires_grad_()
                        dy = torch.randn_like(x) * 0.1
                        with patch.dict(os.environ, NVTE_MHA_MEGAUP="0"):
                            yr = reference(xr, rotary_pos_emb=rope, **metadata)
                        with patch.dict(os.environ, NVTE_MHA_MEGAUP="1"):
                            yc = candidate(xc, rotary_pos_emb=rope, **metadata)
                        assert not hasattr(candidate, "_megaup_local_attention")
                        assert candidate.core_attention.num_attention_heads == heads
                        selected = _attention_backends["attention_params"]
                        assert selected.context_parallel and selected.cp_size == 1
                        assert selected.num_heads == heads // len(ranks)
                        assert selected.num_gqa_groups == groups // len(ranks)
                        assert selected.max_seqlen_q == selected.max_seqlen_kv == maximum
                        if not thd and step == 0 and microbatch == 0:
                            # Independent source-rank/early-or-late token oracle, B>1.
                            p, half = len(ranks), sequence // 2
                            coordinates = (
                                ((b, token) for b in range(batch) for token in range(p * sequence))
                                if qkv_format == "bshd"
                                else (
                                    (b, token)
                                    for token in range(p * sequence)
                                    for b in range(batch)
                                )
                            )
                            expected = []
                            for b, token in coordinates:
                                chunk, offset = divmod(token, half)
                                source = chunk if chunk < p else 2 * p - 1 - chunk
                                expected.append(
                                    (source * batch + b) * sequence
                                    + offset
                                    + (half if chunk >= p else 0)
                                )
                            expected = torch.tensor(expected, device="cuda")
                            to_sequence, to_rank = candidate._megaup_dense_rows[1:]
                            torch.testing.assert_close(to_sequence, expected, atol=0, rtol=0)
                            torch.testing.assert_close(to_rank, expected.argsort(), atol=0, rtol=0)
                        if microbatch == 0:
                            residual = torch.randn_like(yc)
                            yr.add_(residual)
                            yc.add_(residual)
                        graphs.append((xr, xc, yr, yc, dy))
                    # Keep both graphs outstanding, but validate each gradient before
                    # accumulation: two valid contributions may nearly cancel.
                    ref_parameters = dict(reference.named_parameters())
                    gradient_sums = {}
                    for graph_index, (xr, xc, yr, yc, dy) in enumerate(reversed(graphs)):
                        reference.zero_grad(set_to_none=True)
                        candidate.zero_grad(set_to_none=True)
                        compare(yc, yr, "output")
                        yr.backward(dy)
                        if graph_index == 0:
                            yc.backward(dy, retain_graph=True)
                            compare(xc.grad, xr.grad, "retained dInput")
                            for name, parameter in candidate.named_parameters():
                                compare(
                                    parameter.grad,
                                    ref_parameters[name].grad,
                                    f"retained {name}",
                                )
                            candidate.zero_grad(set_to_none=True)
                            xc.grad = None
                        yc.backward(dy)
                        compare(xc.grad, xr.grad, "dInput")
                        for name, parameter in candidate.named_parameters():
                            other = ref_parameters[name]
                            compare(parameter.grad, other.grad, f"graph {graph_index} local {name}")
                            actual, expected = parameter.grad.float(), other.grad.float()
                            if name not in gradient_sums:
                                gradient_sums[name] = (actual, expected, expected.abs())
                            else:
                                for total, value in zip(
                                    gradient_sums[name], (actual, expected, expected.abs())
                                ):
                                    total.add_(value)
                        del actual, expected
                    assert (peer_calls - before[0], pull_calls - before[1]) == (5, 5)
                    assert tuple(x - y for x, y in zip(cp_calls, cp_before)) == (2, 3)
                    # Metadata checks use initialized weights: unnormalized SGD can
                    # amplify cancellation-sensitive GEMM accumulation differences.
                    # Saved FP32 contributions below restore the main SGD gradients.
                    if (
                        step == 0
                        and thd
                        and len(set(lengths)) > 1
                        and os.getenv("NVTE_MHA_MEGAUP_TRUSTED", "0") == "1"
                    ):
                        _check_megaup_trusted_state(
                            candidate, reference, x, dy, rope, metadata, compare, group
                        )
                    for name, parameter in candidate.named_parameters():
                        other = ref_parameters[name]
                        # Both microbatch accumulation and rank reduction are FP32
                        # in this fixture; all individual BF16 gradients passed above.
                        reduced, wanted, abs_partials = gradient_sums.pop(name)
                        for value in (reduced, wanted, abs_partials):
                            dist.all_reduce(value, group=group)
                        error = (reduced - wanted).abs()
                        bound = len(graphs) * len(ranks) * 2.5e-2 + 2.5e-2 * abs_partials
                        strict = int((error > (2.5e-2 + 2.5e-2 * wanted.abs())).sum().item())
                        adjusted = int((error > bound).sum().item())
                        relative = (reduced - wanted).norm() / wanted.norm().clamp_min(1e-12)
                        print(
                            f"MegaUP reduced gradient {name}: accumulation/reduction=FP32,"
                            f" strict_mismatches={strict}, bound_mismatches={adjusted},"
                            f" relative_L2={relative.item():.6g}, max_abs={error.max().item():.6g}",
                            flush=True,
                        )
                        uniform(adjusted == 0 and relative.item() < 2.5e-2, f"reduced {name}")
                        parameter.grad.copy_(reduced)
                        other.grad.copy_(wanted)
                        del reduced, wanted, abs_partials, error, bound
                        with torch.no_grad():
                            parameter.add_(parameter.grad, alpha=-0.01)
                            other.add_(other.grad, alpha=-0.01)
                        compare(parameter, other, f"updated parameter {name}")
                        print(
                            f"MegaUP updated {name}: unequal_before_sync="
                            f"{int((parameter != other).sum().item())}, max_abs_before_sync="
                            f"{(parameter.float() - other.float()).abs().max().item():.6g}",
                            flush=True,
                        )
                        # The next live-weight check must compare the same function,
                        # not two trajectories with independently rounded updates.
                        with torch.no_grad():
                            parameter.copy_(other)
                    assert set(candidate.state_dict()) == original_keys
                    assert {
                        name: id(value) for name, value in candidate.named_parameters()
                    } == original_parameters
                if thd and len(set(lengths)) == 1:
                    # Same local dual-chunk rows, weights and dy: format alone changes.
                    dense = MultiheadAttention(
                        **{
                            **arguments,
                            "qkv_format": "bshd",
                            "attn_mask_type": "causal" if use_rope else "no_mask",
                        }
                    ).cuda()
                    dense.load_state_dict(candidate.state_dict())
                    dense.set_context_parallel_group(group, ranks, torch.cuda.Stream(), "a2a")
                    for enabled in ("0", "1"):
                        results = []
                        for module, value, kwargs in (
                            (candidate, x, metadata),
                            (dense, x.view(batch, lengths[0], width), {}),
                        ):
                            module.zero_grad(set_to_none=True)
                            leaf = value.detach().clone().requires_grad_()
                            with patch.dict(os.environ, NVTE_MHA_MEGAUP=enabled):
                                result = module(leaf, rotary_pos_emb=rope, **kwargs)
                            result.backward(dy.view_as(leaf))
                            results.append(
                                [result.detach().reshape(-1, width), leaf.grad.reshape(-1, width)]
                                + [p.grad for p in module.parameters()]
                            )
                        for thd_value, dense_value in zip(*results):
                            compare(thd_value, dense_value, f"THD/BSHD enabled={enabled}")
                    del dense, results
                if thd:
                    from transformer_engine.pytorch.attention._megaup import prepare_megaup

                    # These exercise uninitialized coordinated fallback, not a fixed contract.
                    object.__setattr__(candidate, "_megaup_validated_contract", None)
                    # Malformed metadata must return uniformly before kernel barriers;
                    # do not send invalid inputs down native attention afterwards.
                    before_guard = (peer_calls, pull_calls)
                    bad_dtype = {**metadata, "cu_seqlens_q": cu.to(torch.int64)}
                    mismatch = cu.clone()
                    if rank == ranks[0]:
                        mismatch[1] += 2 * len(ranks)
                    bad_rank = {
                        **metadata,
                        "cu_seqlens_q": mismatch,
                        "cu_seqlens_kv": mismatch,
                        "cu_seqlens_q_padded": mismatch,
                        "cu_seqlens_kv_padded": mismatch,
                        "max_seqlen_q": maximum + 2 * len(ranks),
                        "max_seqlen_kv": maximum + 2 * len(ranks),
                    }
                    for bad in (bad_dtype, bad_rank, {**metadata, "max_seqlen_q": 1}):
                        assert (
                            prepare_megaup(
                                candidate,
                                x,
                                eligible_call=True,
                                rotary_pos_emb=rope,
                                attn_mask_type=arguments["attn_mask_type"],
                                **bad,
                            )
                            is None
                        )
                    assert (peer_calls, pull_calls) == before_guard
                    # A real padding-gap input is valid native THD but outside fusion.
                    logical = cu.clone()
                    logical[1:] -= torch.arange(1, batch + 1, device="cuda", dtype=torch.int32) * (
                        2 * len(ranks)
                    )
                    padded = {**metadata, "cu_seqlens_q": logical, "cu_seqlens_kv": logical}
                    with torch.no_grad(), patch.dict(os.environ, NVTE_MHA_MEGAUP="0"):
                        expected = candidate(
                            x, rotary_pos_emb=rope, pad_between_seqs=True, **padded
                        )
                    with torch.no_grad(), patch.dict(os.environ, NVTE_MHA_MEGAUP="1"):
                        actual = candidate(x, rotary_pos_emb=rope, pad_between_seqs=True, **padded)
                    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                    assert (peer_calls, pull_calls) == before_guard
                # Unsupported policies must take the rank-uniform native path.
                fallback_options = [{"bias": True}]
                if groups != heads:
                    fallback_options.append({"qkv_weight_interleaved": False})
                for options in fallback_options:
                    fallback = MultiheadAttention(**{**arguments, **options}).cuda()
                    fallback.set_context_parallel_group(group, ranks, torch.cuda.Stream(), "a2a")
                    before = (peer_calls, pull_calls)
                    with torch.no_grad(), patch.dict(os.environ, NVTE_MHA_MEGAUP="0"):
                        expected = fallback(x, rotary_pos_emb=rope, **metadata)
                    with torch.no_grad(), patch.dict(os.environ, NVTE_MHA_MEGAUP="1"):
                        actual = fallback(x, rotary_pos_emb=rope, **metadata)
                    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                    assert (peer_calls, pull_calls) == before
                print(
                    f"MegaUP passed: rank={rank}, CP={len(ranks)}, B={batch}, S={sequence},"
                    f" {qkv_format}, RoPE={use_rope}, W={width}, Hq={heads}, Hkv={groups}",
                    flush=True,
                )
            dist.barrier()
    dist.destroy_process_group()


def run_megaup_linear_hooks():
    """Check both projection hooks against explicit rank/feature routing and GEMMs."""
    import weakref
    from unittest.mock import patch
    import pytest
    import mega_up
    from transformer_engine.pytorch import Linear
    from transformer_engine.pytorch.attention._megaup import _CPProjection

    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    group, rank, cp = dist.group.WORLD, dist.get_rank(), dist.get_world_size()

    def compare(actual, expected):
        error = None
        try:
            torch.testing.assert_close(actual, expected, rtol=0.025, atol=0.025)
            relative = (
                actual.float() - expected.float()
            ).norm() / expected.float().norm().clamp_min(1e-12)
            assert relative < 0.025, f"relative L2={relative.item()}"
        except AssertionError as failure:
            error = str(failure)
        passed = torch.tensor(int(error is None), device="cuda", dtype=torch.int32)
        dist.all_reduce(passed, op=dist.ReduceOp.MIN)
        assert passed.item(), error or "Projection comparison failed on another rank"

    m, k, d = 128, 1024, 128
    for peer, interleave in ((True, False), (True, True), (False, False)):
        n = 3 * k if peer else k
        runtime = mega_up.initialize_runtime(
            group, hidden=n if peer else k, max_rows=m, device=torch.device("cuda")
        )
        hook = _CPProjection(runtime, peer, d, k // d if interleave else None)
        linear = Linear(k, n, bias=False, params_dtype=torch.bfloat16).cuda()
        torch.manual_seed(941)
        with torch.no_grad():
            linear.weight.copy_(torch.randn_like(linear.weight) * 0.02)
        w = linear.weight.detach().clone()
        for need_dx, need_dw in ((True, True), (False, True), (True, False)):
            torch.manual_seed(713)
            linear.weight.requires_grad_(need_dw)
            linear.weight.grad = None
            if peer:
                all_x = torch.randn(cp, m, k, device="cuda", dtype=torch.bfloat16) * 0.1
                all_dy = torch.randn(cp, cp * m, n // cp, device="cuda", dtype=torch.bfloat16) * 0.1
                all_y = all_x @ w.t()
                if interleave:
                    all_y = all_y.view(cp, m, 3, k // d, d).transpose(2, 3).reshape(cp, m, n)
                expected_y = all_y[:, :, rank * (n // cp) : (rank + 1) * (n // cp)].reshape(
                    cp * m, n // cp
                )
                full_dy = all_dy[:, rank * m : (rank + 1) * m].movedim(0, 1).reshape(m, n)
                if interleave:
                    full_dy = full_dy.view(m, k // d, 3, d).transpose(1, 2).reshape(m, n)
                expected_dx, expected_dw = full_dy @ w, full_dy.t() @ all_x[rank]
            else:
                all_x = torch.randn(cp, cp * m, k // cp, device="cuda", dtype=torch.bfloat16) * 0.1
                all_dy = torch.randn(cp, m, n, device="cuda", dtype=torch.bfloat16) * 0.1
                materialized = all_x[:, rank * m : (rank + 1) * m].movedim(0, 1).reshape(m, k)
                expected_y = materialized @ w.t()
                full_dx = all_dy @ w
                expected_dx = full_dx[:, :, rank * (k // cp) : (rank + 1) * (k // cp)].reshape(
                    cp * m, k // cp
                )
                expected_dw = all_dy[rank].t() @ materialized
            x = all_x[rank].clone().requires_grad_(need_dx)
            with patch.object(
                mega_up, "peer_write_gemm_thd", wraps=mega_up.peer_write_gemm_thd
            ) as peers, patch.object(
                mega_up,
                "prepare_headshard_pull_gemm_thd",
                wraps=mega_up.prepare_headshard_pull_gemm_thd,
            ) as pulls:
                y = linear(x, _cp_projection=hook)
                saved = weakref.ref(y.grad_fn.saved_tensors[0]) if not peer and need_dw else None
                y.add_(0.125)
                compare(y, expected_y + 0.125)
                for repeat in (True, False):
                    y.backward(all_dy[rank], retain_graph=repeat)
                    if need_dx:
                        compare(x.grad, expected_dx)
                    if need_dw:
                        compare(linear.weight.grad, expected_dw)
                    if saved is not None:
                        assert (saved() is not None) == repeat
                    x.grad = linear.weight.grad = None
                expected_calls = (1, 2 if need_dx else 0) if peer else (2 if need_dx else 0, 1)
                assert (peers.call_count, pulls.call_count) == expected_calls

        linear.weight.requires_grad_(True)
        x = x.detach().requires_grad_()
        with torch.autocast("cuda", dtype=torch.float16), pytest.raises(ValueError, match="BF16"):
            linear(x, _cp_projection=hook)
        with pytest.raises((ValueError, torch._dynamo.exc.Unsupported)):
            torch.compile(
                lambda z: linear(z, _cp_projection=hook), backend="eager", fullgraph=True
            )(x)
        # Packing must not hide an in-place update of the original Parameter.
        y = linear(x, _cp_projection=hook)
        with torch.no_grad():
            linear.weight.add_(0.001)
        with pytest.raises(RuntimeError, match="modified by an inplace operation"):
            y.backward(torch.ones_like(y))
        if interleave:
            y = linear(x, _cp_projection=hook)
            with torch.no_grad():
                y.grad_fn.saved_tensors[1].add_(0.001)
            with pytest.raises(RuntimeError, match="modified by an inplace operation"):
                y.backward(torch.ones_like(y))
    if rank == 0:
        print(
            "MegaUP Linear hooks passed: routing, frozen gradients, retained graphs, ownership,"
            " guards",
            flush=True,
        )
    dist.destroy_process_group()


def main(**kwargs):
    run_dpa_with_cp(**kwargs)


if __name__ == "__main__":
    if sys.argv[1:2] == ["megaup"]:
        run_mha_with_megaup()
    elif sys.argv[1:2] == ["megaup-gqa"]:
        run_mha_with_megaup(gqa_only=True)
    elif sys.argv[1:2] == ["megaup-linear"]:
        run_megaup_linear_hooks()
    else:
        kwargs = dict(arg.split("=") for arg in sys.argv[2:])
        main(**kwargs)
