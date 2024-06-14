# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Utils for testing"""

import random
from typing import Union

import numpy as np
import paddle
from paddle.distributed import fleet
from paddle.distributed.fleet.meta_parallel import get_rng_state_tracker

import transformer_engine  # pylint: disable=unused-import
from transformer_engine.paddle.constants import (
    TE_DType,
    AttnBiasType,
    AttnMaskType,
    FusedAttnBackend,
)
from transformer_engine.paddle.fp8 import FP8TensorMeta
from transformer_engine import (
    transformer_engine_paddle as tex,
)  # pylint: disable=wrong-import-order


def create_fp8_meta(num_gemms=1, amax_history_len=10):
    """
    Create and initialize FP8TensorMeta
    """
    fp8_meta = FP8TensorMeta(is_forward=True)
    fp8_meta.prepare(num_gemms, amax_history_len)
    return fp8_meta


def assert_allclose(
    actual, desired, rtol=1e-05, atol=1e-08, equal_nan=True, err_msg="", verbose=True
):
    """Compare two input paddle tensors"""
    if isinstance(actual, paddle.Tensor):
        actual = paddle.cast(actual, "float32")
    if isinstance(desired, paddle.Tensor):
        desired = paddle.cast(desired, "float32")
    if len(actual.shape) == 0:
        actual = actual.item()
        desired = desired.item()
    else:
        actual = actual.numpy()
        desired = desired.numpy()
    np.testing.assert_allclose(actual, desired, rtol, atol, equal_nan, err_msg, verbose)


def assert_shape(inp, expected_shape):
    """Assert the shape of input tensor equals to expected shape"""
    assert (
        inp.shape == expected_shape
    ), f"Expected tensor shape: {expected_shape} != actual tensor shape: {inp.shape}"


def is_devices_enough(required):
    """If the number of device is enough"""
    return paddle.device.cuda.device_count() >= required


def set_random_seed(seed):
    """Set random seed for reproducability."""
    fleet.meta_parallel.model_parallel_random_seed(seed)

    hcg = fleet.get_hybrid_communicate_group()
    if paddle.distributed.get_world_size() > 1:
        # obtain rank message of hybrid parallel

        mp_rank = hcg.get_model_parallel_rank()
        mp_size = hcg.get_model_parallel_world_size()

        pp_rank = hcg.get_stage_id()
        pp_size = hcg.get_pipe_parallel_world_size()

        dp_rank = hcg.get_data_parallel_rank()
        dp_size = hcg.get_data_parallel_world_size()

        sharding_rank = hcg.get_sharding_parallel_rank()
    else:
        mp_rank, mp_size = 0, 1
        pp_rank, pp_size = 0, 1
        dp_rank, dp_size = 0, 1
        sharding_rank, _ = 0, 1

    random.seed(seed + 100 * pp_rank)
    np.random.seed(seed + 100 * pp_rank)

    seed_offset = seed + 1024 + paddle.distributed.get_world_size()
    global_seed = (
        seed_offset
        + pp_rank * (mp_size)
        + dp_rank * (mp_size * pp_size)
        + sharding_rank * (mp_size * pp_size * dp_size)
    )

    seed_offset += paddle.distributed.get_world_size()
    local_seed = (
        seed_offset
        + mp_rank
        + pp_rank * (mp_size)
        + dp_rank * (mp_size * pp_size)
        + sharding_rank * (mp_size * pp_size * dp_size)
    )

    tracker = get_rng_state_tracker()
    # tracker.reset()
    if "global_seed" not in tracker.states_:
        tracker.add("global_seed", global_seed)
    if "local_seed" not in tracker.states_:
        tracker.add("local_seed", local_seed)

    paddle.seed(global_seed)


def get_fused_attention_backend(
    num_heads: int,
    num_gqa_groups: int,
    q_seqlen: int,
    kv_seqlen: int,
    head_size: int,
    dtype: Union[paddle.dtype, str],
    dropout: float,
    qkv_layout: str = "bs3hd",
    bias_type: str = "no_bias",
    mask_type: str = "causal",
) -> tex.NVTE_Fused_Attn_Backend:
    """Get cuDNN fused attention backend for attention config"""
    if isinstance(dtype, str):
        dtype = dict(
            float32=paddle.float32,
            bfloat16=paddle.bfloat16,
            float16=paddle.float16,
        )[dtype]
    return tex.get_fused_attn_backend(
        TE_DType[dtype],
        TE_DType[dtype],
        tex.get_nvte_qkv_layout(qkv_layout),
        AttnBiasType[bias_type],
        AttnMaskType[mask_type],
        dropout,
        num_heads,
        num_gqa_groups,
        q_seqlen,
        kv_seqlen,
        head_size,
    )


def is_fused_attention_supported(
    num_heads: int,
    num_gqa_groups: int,
    q_seqlen: int,
    kv_seqlen: int,
    head_size: int,
    dtype: Union[paddle.dtype, str],
    dropout: float,
    qkv_layout: str = "bs3hd",
    bias_type: str = "no_bias",
    mask_type: str = "causal",
) -> bool:
    """Check if cuDNN fused attention is supported for attention config"""
    backend = get_fused_attention_backend(
        num_heads=num_heads,
        num_gqa_groups=num_gqa_groups,
        q_seqlen=q_seqlen,
        kv_seqlen=kv_seqlen,
        head_size=head_size,
        dtype=dtype,
        dropout=dropout,
        qkv_layout=qkv_layout,
        bias_type=bias_type,
        mask_type=mask_type,
    )
    return backend != FusedAttnBackend["No_Backend"]


def register_sequence_parallel_allreduce_hooks(model, accumulation_steps) -> None:
    """Register allreduce hooks for sequence parallel tensors"""

    def is_sequence_parallel_parameter(parameter):
        """If input tensor is marked as sequence parallel tensor"""
        out = getattr(parameter, "sequence_parallel", False)
        return out

    def create_allreduce_gradient_hook(param, accumulation_steps):
        """Create allreduce gradient hook"""
        hcg = fleet.get_hybrid_communicate_group()
        pg = hcg.get_model_parallel_group().process_group
        step = [0]

        @paddle.autograd.no_grad()
        def __impl__():
            step[0] += 1
            if (step[0] % accumulation_steps) == 0:
                if hasattr(param, "main_grad"):
                    pg.allreduce(param.main_grad).wait()
                else:
                    pg.allreduce(param.grad).wait()

        return __impl__

    if accumulation_steps <= 0 or not paddle.distributed.is_initialized():
        return

    hcg = fleet.get_hybrid_communicate_group()
    mp_group = hcg.get_model_parallel_group()
    if mp_group.nranks <= 1:
        return

    params = []
    for p in model.parameters():
        if is_sequence_parallel_parameter(p):
            params.append(p)

    for p in params:
        hook = create_allreduce_gradient_hook(p, accumulation_steps)
        p._register_backward_hook(hook)
