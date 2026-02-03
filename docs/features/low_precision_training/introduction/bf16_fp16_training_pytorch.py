# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

# START_BF16_FP16_TRAINING

import torch
import transformer_engine.pytorch as te
from contextlib import nullcontext


def run_forward_backward(params_dtype, autocast_precision, grad_scaler_enabled):
    if grad_scaler_enabled:
        grad_scaler = torch.amp.GradScaler("cuda")

    layer = te.TransformerLayer(
        hidden_size=1024,
        ffn_hidden_size=4096,
        num_attention_heads=16,
        params_dtype=params_dtype,
    )
    x = torch.randn(32, 128, 1024, dtype=params_dtype, device="cuda")

    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=autocast_precision)
        if autocast_precision is not None
        else nullcontext()
    )
    with autocast_ctx:
        output = layer(x)
        assert (
            output.dtype == autocast_precision if autocast_precision is not None else params_dtype
        )
        loss = output.sum()
    if grad_scaler_enabled:
        grad_scaler.scale(loss).backward()
    else:
        loss.backward()


run_forward_backward(torch.float32, torch.float32, False)  # high precision training
run_forward_backward(
    torch.float32, torch.bfloat16, False
)  # bfloat16 training with master weights in FP32
run_forward_backward(
    torch.float32, torch.float16, True
)  # fp16 training with master weights in FP32, needs loss scaling
run_forward_backward(
    torch.bfloat16, torch.bfloat16, False
)  # bfloat16 training with weights in BF16

# END_BF16_FP16_TRAINING
