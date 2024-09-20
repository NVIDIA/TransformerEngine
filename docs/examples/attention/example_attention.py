# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import os, sys, time
import subprocess
import pandas as pd
import numpy as np
import torch
import nvtx
import transformer_engine
from tests.pytorch.fused_attn.test_fused_attn import (
    ModelConfig,
    _get_attention_backends,
    _run_dot_product_attention,
)

# data type
dtype = torch.bfloat16
# number of iterations after 3 warmup iterations
num_iters = 3
# checkpointing
ckpt_attn = False
# workspace optimization path for cuDNN attention
workspace_opt = True
# QKV memory layout
qkv_layout = "bshd_bshd_bshd"
# sliding window attention
swa = False
# padding between sequences for qkv_format=thd
pad_between_seqs = False
# training mode
is_training = True

model_configs = {
    #   test:             b,  h, hg,   d,   sq,  skv,   p,     mask,              bias
    "test_0": ModelConfig(2, 16, 16, 64, 512, 512, 0.0, "no_mask", "no_bias"),  # short seq
    "test_1": ModelConfig(2, 16, 16, 128, 2048, 2048, 0.0, "causal", "no_bias"),  # longer seq, mask
    "test_2": ModelConfig(2, 16, 16, 128, 2048, 2048, 0.0, "causal", "post_scale_bias"),  # bias
    "test_3": ModelConfig(2, 32, 4, 128, 8192, 8192, 0.0, "causal", "no_bias"),  # GQA
}


def example_attention(model, fused_attn_supported, flash_attn_supported):
    config = model_configs[model]
    if dtype == torch.bfloat16:
        tols = dict(atol=2.5e-2, rtol=2.5e-2)
    else:
        tols = dict(atol=5e-3, rtol=5e-3)

    if fused_attn_supported:
        print()
        print("Run cuDNN attention...")
        fused_attn_fwd, fused_attn_bwd = _run_dot_product_attention(
            dtype,
            config,
            "FusedAttention",
            ckpt_attn,
            qkv_layout,
            workspace_opt,
            pad_between_seqs,
            is_training,
        )

    if flash_attn_supported:
        print()
        print("Run flash-attention...")
        flash_attn_fwd, flash_attn_bwd = _run_dot_product_attention(
            dtype,
            config,
            "FlashAttention",
            ckpt_attn,
            qkv_layout,
            workspace_opt,
            pad_between_seqs,
            is_training,
        )

    if fused_attn_supported and flash_attn_supported:
        torch.testing.assert_close(fused_attn_fwd, flash_attn_fwd, **tols)
        for i, _ in enumerate(flash_attn_bwd):
            torch.testing.assert_close(fused_attn_bwd[i], flash_attn_bwd[i], **tols)

    print()
    print("Test passed.")


def main():

    models = ["test_0"]
    for model in models:
        config = model_configs[model]
        available_backends, fused_attn_backends = _get_attention_backends(
            config,
            qkv_dtype=dtype,
            qkv_layout=qkv_layout,
            window_size=config.window_size,
            pad_between_seqs=pad_between_seqs,
        )
        flash_attn_supported, fused_attn_supported, unfused_attn_supported = available_backends

        example_attention(model, fused_attn_supported, flash_attn_supported)


if __name__ == "__main__":
    main()
