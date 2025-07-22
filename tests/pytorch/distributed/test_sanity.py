# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import pathlib
import sys
import pytest
import torch
import transformer_engine
from transformer_engine.pytorch.attention.dot_product_attention import DotProductAttention
from transformer_engine.pytorch import TransformerLayer, Linear

_current_file = pathlib.Path(__file__).resolve()
sys.path.append(str(_current_file.parent.parent))
from utils import ModelConfig

model_configs = {
    "small": ModelConfig(2, 10, 2, 16),
}


@pytest.mark.parametrize("model", ["small"])
@pytest.mark.parametrize("module", ["TransformerLayer", "DotProductAttention", "Linear"])
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
        num_tokens = torch.randint(0, config.max_seqlen_q, (1,)).item()
        args = [
            torch.randn(
                (num_tokens, config.hidden_size),
                dtype=dtype,
                device=f"cuda:{tensor_device}",
                requires_grad=True,
            )
        ]
        cu_seqlens_q, cu_seqlens_kv = [
            torch.Tensor([0, 2, 3]).to(dtype=torch.int32, device=tensor_device) for _ in range(2)
        ]
        kwargs["cu_seqlens_q"] = cu_seqlens_q
        kwargs["cu_seqlens_kv"] = cu_seqlens_kv
        kwargs["max_seqlen_q"] = config.max_seqlen_q
        kwargs["max_seqlen_kv"] = config.max_seqlen_kv
    if module == "DotProductAttention":
        model = DotProductAttention(
            config.num_heads, config.head_dim_qk, qkv_format="thd", attn_mask_type="padding"
        )
        num_tokens = torch.randint(0, config.max_seqlen_q, (1,)).item()
        args = [
            torch.randn(
                num_tokens,
                config.num_heads,
                config.head_dim_qk,
                dtype=dtype,
                device=tensor_device,
                requires_grad=True,
            )
            for _ in range(3)
        ]
        cu_seqlens_q, cu_seqlens_kv = [
            torch.Tensor([0, 2, 3]).to(dtype=torch.int32, device=tensor_device) for _ in range(2)
        ]
        kwargs["cu_seqlens_q"] = cu_seqlens_q
        kwargs["cu_seqlens_kv"] = cu_seqlens_kv
        kwargs["max_seqlen_q"] = config.max_seqlen_q
        kwargs["max_seqlen_kv"] = config.max_seqlen_kv
        bwd_args = [torch.randn(num_tokens, config.hidden_size, dtype=dtype, device=tensor_device)]
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
