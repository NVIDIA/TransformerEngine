# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import pytest
import torch

import transformer_engine.pytorch as te

batch_size = 32
seq_length = 2048
num_heads = 16
head_dim = 64
dtype = torch.bfloat16
num_attn_head = 16
ffn_hidden_size=1024

@pytest.mark.parametrize("kv_channels", [128, 256])
@pytest.mark.parametrize("hidden_size", [128, 256])
@pytest.mark.parametrize("num_gqa_groups", [1, 2, 4, 8, 16])
def test_gqa(
    kv_channels,
    hidden_size,
    num_gqa_groups
) -> None:
    
    model = te.TransformerLayer(
        hidden_size,
        ffn_hidden_size,
        num_attn_head,
        num_gqa_groups,
        kv_channels=kv_channels
    )

    # Run forward pass
    x = torch.randn((batch_size, 1, hidden_size)).cuda()
    model(x)

