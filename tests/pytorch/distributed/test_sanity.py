# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import torch
import transformer_engine
from transformer_engine.pytorch.attention.dot_product_attention import DotProductAttention


def test_current_device():
    """Test cases where current device is different from tensor device"""

    num_devices = torch.cuda.device_count()
    assert num_devices > 1, "This test requires more than one GPU!"
    num_tokens = 5
    num_heads = 2
    head_dim = 16
    tensor_device = num_devices - 1

    max_seqlen_q, max_seqlen_kv = 10, 10
    cu_seqlens_q, cu_seqlens_kv = [torch.Tensor([0, 2, 3]).to(dtype=torch.int32, device=tensor_device) for _ in range(2)]
    q, k, v = [torch.randn(num_tokens, num_heads, head_dim, dtype=torch.bfloat16, device=tensor_device) for _ in range(3)]
    module = DotProductAttention(num_heads, head_dim, qkv_format='thd', attn_mask_type="padding")
    current_device_before = torch.cuda.current_device()
    out = module(q, k, v, cu_seqlens_q=cu_seqlens_q, cu_seqlens_kv=cu_seqlens_kv, max_seqlen_q=max_seqlen_q, max_seqlen_kv=max_seqlen_kv)
    current_device_after = torch.cuda.current_device()
    tensor_device_out = out.get_device()
    assert current_device_after == current_device_before, "The current device should not have changed!"
    assert tensor_device_out == tensor_device, "The output tensor should be the same as the input tensors!"
