# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import os
import pytest
import subprocess
from test_fused_attn import (
    ModelConfig,
    _is_flash_attention_2_available,
)
from transformer_engine.pytorch.utils import (
    get_device_compute_capability,
    get_cudnn_version,
)

model_configs_flash_attn = {
    #   test:             b,  h, hg,   d,   sq,  skv,   p,      mask,      bias
    "cp_1_0": ModelConfig(2, 12, 12, 128, 4096, 4096, 0.0, "causal", "no_bias"),  # MHA
    "cp_1_1": ModelConfig(2, 12, 12, 128, 4096, 4096, 0.0, "no_mask", "no_bias"),  # MHA
    "cp_2_0": ModelConfig(2, 12, 1, 128, 4096, 4096, 0.0, "causal", "no_bias"),  # GQA
    "cp_2_1": ModelConfig(2, 12, 1, 128, 4096, 4096, 0.0, "no_mask", "no_bias"),  # GQA
}


def get_bash_arguments(**kwargs):
    args = ["python", "-m", "torch.distributed.launch", "--nproc-per-node=2"]
    te_path = os.getenv("TE_PATH", "/opt/transformerengine")
    script_path = os.path.join(te_path, "tests/pytorch/fused_attn/run_fused_attn_with_cp.py")
    args.append(script_path)
    for k, v in kwargs.items():
        args.append(f"{k}={v}")
    return args


@pytest.mark.skipif(not _is_flash_attention_2_available(), reason="Flash-attn 2.0+ is required.")
@pytest.mark.skipif(get_device_compute_capability() < (8, 0), reason="CP tests require sm80+.")
@pytest.mark.parametrize("dtype", ["bf16", "fp16"])
@pytest.mark.parametrize("model", model_configs_flash_attn.keys())
@pytest.mark.parametrize("qkv_format", ["bshd", "sbhd", "thd"])
def test_cp_with_flash_attention(dtype, model, qkv_format):
    subprocess.run(
        get_bash_arguments(
            dtype=dtype, model=model, qkv_format=qkv_format, kernel_backend="FlashAttention"
        ),
        check=True,
    )


model_configs_fused_attn = {
    #   test:             b,  h, hg,   d,   sq,  skv,   p,      mask,              bias
    "cp_1_0": ModelConfig(2, 12, 12, 128, 4096, 4096, 0.0, "causal", "no_bias"),  # MHA
    "cp_1_1": ModelConfig(2, 12, 12, 128, 4096, 4096, 0.0, "no_mask", "no_bias"),  # MHA
    "cp_1_2": ModelConfig(2, 12, 12, 128, 4096, 4096, 0.0, "causal", "post_scale_bias"),  # MHA
    "cp_1_3": ModelConfig(2, 12, 12, 128, 4096, 4096, 0.0, "no_mask", "post_scale_bias"),  # MHA
    "cp_2_0": ModelConfig(2, 12, 1, 128, 4096, 4096, 0.0, "causal", "no_bias"),  # GQA
    "cp_2_1": ModelConfig(2, 12, 1, 128, 4096, 4096, 0.0, "no_mask", "no_bias"),  # GQA
    "cp_2_2": ModelConfig(2, 12, 1, 128, 4096, 4096, 0.0, "causal", "post_scale_bias"),  # GQA
    "cp_2_3": ModelConfig(2, 12, 1, 128, 4096, 4096, 0.0, "no_mask", "post_scale_bias"),  # GQA
}


@pytest.mark.skipif(get_cudnn_version() < (8, 9, 7), reason="cuDNN 8.9.7+ is required.")
@pytest.mark.skipif(get_device_compute_capability() < (8, 0), reason="CP tests require sm80+.")
@pytest.mark.parametrize("dtype", ["bf16", "fp16"])
@pytest.mark.parametrize("model", model_configs_fused_attn.keys())
@pytest.mark.parametrize("qkv_format", ["bshd", "sbhd", "thd"])
def test_cp_with_fused_attention(dtype, model, qkv_format):
    if qkv_format == "thd" and get_device_compute_capability() < (9, 0):
        pytest.skip("THD format is only supported on sm90+.")
    subprocess.run(
        get_bash_arguments(
            dtype=dtype, model=model, qkv_format=qkv_format, kernel_backend="FusedAttention"
        ),
        check=True,
    )
