# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import pathlib
import sys
import pytest
import torch
import transformer_engine
from transformer_engine.pytorch import (
    DotProductAttention,
    TransformerLayer,
    Linear,
    GroupedLinear,
    NVFP4Quantizer,
    autocast,
    is_nvfp4_available,
)
from transformer_engine.common import recipe

_current_file = pathlib.Path(__file__).resolve()
sys.path.append(str(_current_file.parent.parent))
from utils import ModelConfig

model_configs = {
    "small": ModelConfig(2, 10, 2, 16),
}

nvfp4_available, reason_for_no_nvfp4 = is_nvfp4_available(return_reason=True)


@pytest.mark.parametrize("model", ["small"])
@pytest.mark.parametrize(
    "module", ["TransformerLayer", "DotProductAttention", "Linear", "GroupedLinear"]
)
def test_current_device(model, module):
    """Test cases where current device is different from tensor device"""

    num_devices = torch.cuda.device_count()
    assert num_devices > 1, "This test requires more than one GPU!"
    tensor_device = num_devices - 1
    dtype = torch.bfloat16
    config = model_configs[model]

    args = []
    kwargs = {}
    bwd_args = []
    if module == "TransformerLayer":
        model = TransformerLayer(
            config.hidden_size,
            4 * config.hidden_size,
            config.num_heads,
            params_dtype=dtype,
            attn_input_format="thd",
            self_attn_mask_type="padding",
            device=f"cuda:{tensor_device}",
        )
        seqlens_q = torch.randint(
            1,
            config.max_seqlen_q,
            [config.batch_size],
            dtype=torch.int32,
            device=f"cuda:{tensor_device}",
        )
        cu_seqlens_q = torch.zeros(
            config.batch_size + 1, dtype=torch.int32, device=f"cuda:{tensor_device}"
        )
        cu_seqlens_q[1:] = torch.cumsum(seqlens_q, dim=0)
        seqlens_kv = torch.randint(
            1,
            config.max_seqlen_kv,
            [config.batch_size],
            dtype=torch.int32,
            device=f"cuda:{tensor_device}",
        )
        cu_seqlens_kv = torch.zeros(
            config.batch_size + 1, dtype=torch.int32, device=f"cuda:{tensor_device}"
        )
        cu_seqlens_kv[1:] = torch.cumsum(seqlens_kv, dim=0)
        num_tokens = cu_seqlens_q[-1]
        args = [
            torch.randn(
                (num_tokens, config.hidden_size),
                dtype=dtype,
                device=f"cuda:{tensor_device}",
                requires_grad=True,
            )
        ]
        kwargs["cu_seqlens_q"] = cu_seqlens_q
        kwargs["cu_seqlens_kv"] = cu_seqlens_kv
        kwargs["max_seqlen_q"] = config.max_seqlen_q
        kwargs["max_seqlen_kv"] = config.max_seqlen_kv
    elif module == "DotProductAttention":
        model = DotProductAttention(
            config.num_heads, config.head_dim_qk, qkv_format="thd", attn_mask_type="padding"
        )
        seqlens_q = torch.randint(
            1,
            config.max_seqlen_q,
            [config.batch_size],
            dtype=torch.int32,
            device=f"cuda:{tensor_device}",
        )
        cu_seqlens_q = torch.zeros(
            config.batch_size + 1, dtype=torch.int32, device=f"cuda:{tensor_device}"
        )
        cu_seqlens_q[1:] = torch.cumsum(seqlens_q, dim=0)
        seqlens_kv = torch.randint(
            1,
            config.max_seqlen_kv,
            [config.batch_size],
            dtype=torch.int32,
            device=f"cuda:{tensor_device}",
        )
        cu_seqlens_kv = torch.zeros(
            config.batch_size + 1, dtype=torch.int32, device=f"cuda:{tensor_device}"
        )
        cu_seqlens_kv[1:] = torch.cumsum(seqlens_kv, dim=0)
        num_tokens = cu_seqlens_q[-1]
        args = [
            torch.randn(
                num_tokens,
                config.num_heads,
                config.head_dim_qk,
                dtype=dtype,
                device=f"cuda:{tensor_device}",
                requires_grad=True,
            )
            for _ in range(3)
        ]
        kwargs["cu_seqlens_q"] = cu_seqlens_q
        kwargs["cu_seqlens_kv"] = cu_seqlens_kv
        kwargs["max_seqlen_q"] = config.max_seqlen_q
        kwargs["max_seqlen_kv"] = config.max_seqlen_kv
        bwd_args = [
            torch.randn(num_tokens, config.hidden_size, dtype=dtype, device=f"cuda:{tensor_device}")
        ]
    elif module == "Linear":
        model = Linear(
            config.hidden_size,
            4 * config.hidden_size,
            params_dtype=dtype,
            device=f"cuda:{tensor_device}",
        )
        args = [
            torch.randn(
                (config.max_seqlen_q, config.batch_size, config.hidden_size),
                dtype=dtype,
                device=f"cuda:{tensor_device}",
                requires_grad=True,
            )
        ]
    elif module == "GroupedLinear":
        num_gemms = 4
        model = GroupedLinear(
            num_gemms,
            config.hidden_size,
            4 * config.hidden_size,
            params_dtype=dtype,
            device=f"cuda:{tensor_device}",
        )
        args = [
            torch.randn(
                (config.max_seqlen_q * config.batch_size * (num_gemms - 1), config.hidden_size),
                dtype=dtype,
                device=f"cuda:{tensor_device}",
                requires_grad=True,
            ),
            [0] + [config.max_seqlen_q * config.batch_size] * (num_gemms - 1),  # Empty first split.
        ]

    current_device_before = torch.cuda.current_device()
    out = model(*args, **kwargs)
    if module == "DotProductAttention":
        out.backward(*bwd_args)
    else:
        loss = out.sum()
        loss.backward()
    current_device_after = torch.cuda.current_device()
    tensor_device_out = out.get_device()
    tensor_device_grad = args[0].grad.get_device()

    assert (
        current_device_after == current_device_before
    ), "The current device should not have changed!"
    assert (
        tensor_device_out == tensor_device
    ), "The output tensor should be the same as the input tensors!"
    assert (
        tensor_device_grad == tensor_device
    ), "The gradient tensor should be the same as the input tensors!"


@pytest.mark.skipif(not nvfp4_available, reason=reason_for_no_nvfp4)
def test_nvfp4_rht_cache():
    """Ensure correct RHT cache for NVFP4."""

    num_devices = torch.cuda.device_count()
    assert num_devices > 1, "This test requires more than one GPU!"

    # Populate cache on last device.
    with torch.cuda.device(num_devices - 1):
        _ = NVFP4Quantizer()

    hidden_size = 128
    dtype = torch.bfloat16

    model = Linear(hidden_size, hidden_size, params_dtype=dtype)
    inp = torch.randn(hidden_size, hidden_size, device=torch.cuda.current_device(), dtype=dtype)
    fp4_recipe = recipe.NVFP4BlockScaling()
    with autocast(recipe=fp4_recipe):
        _ = model(inp)
