# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import os
import pytest
import subprocess
from test_fused_attn import ModelConfig
from transformer_engine.pytorch.attention import (
    _flash_attn_2_plus,
    _flash_attn_2_3_plus,
)
from transformer_engine.pytorch.utils import (
    get_device_compute_capability,
    get_cudnn_version,
)

model_configs_flash_attn = {
    #   test:             b,  h, hg,   d,   sq,  skv,   p,     mask,      bias
    "cp_1_0": ModelConfig(2, 12, 12, 128, 4096, 4096, 0.0, "causal", "no_bias"),  # MHA
    "cp_1_1": ModelConfig(2, 12, 12, 128, 4096, 4096, 0.0, "no_mask", "no_bias"),  # MHA
    "cp_1_2": ModelConfig(
        2, 12, 12, 128, 4096, 4096, 0.0, "causal", "no_bias", window_size=(512, 0)
    ),  # MHA
    "cp_2_0": ModelConfig(2, 12, 1, 128, 4096, 4096, 0.0, "causal", "no_bias"),  # GQA
    "cp_2_1": ModelConfig(2, 12, 1, 128, 4096, 4096, 0.0, "no_mask", "no_bias"),  # GQA
    "cp_2_2": ModelConfig(
        2, 12, 1, 128, 4096, 4096, 0.0, "causal", "no_bias", window_size=(512, 0)
    ),  # GQA
}


def get_bash_arguments(**kwargs):
    args = ["python", "-m", "torch.distributed.launch", "--nproc-per-node=2"]
    te_path = os.getenv("TE_PATH", "/opt/transformerengine")
    script_path = os.path.join(te_path, "tests/pytorch/fused_attn/run_fused_attn_with_cp.py")
    args.append(script_path)
    for k, v in kwargs.items():
        args.append(f"{k}={v}")
    return args


@pytest.mark.skipif(not _flash_attn_2_plus, reason="Flash-attn 2.0+ is required.")
@pytest.mark.skipif(get_device_compute_capability() < (8, 0), reason="CP tests require sm80+.")
@pytest.mark.parametrize("dtype", ["bf16", "fp16"])
@pytest.mark.parametrize("model", model_configs_flash_attn.keys())
@pytest.mark.parametrize("qkv_format", ["bshd", "sbhd", "thd"])
@pytest.mark.parametrize("cp_comm_type", ["p2p", "all_gather"])
def test_cp_with_flash_attention(dtype, model, qkv_format, cp_comm_type):
    config = model_configs_flash_attn[model]
    if cp_comm_type == "all_gather" and qkv_format == "thd":
        pytest.skip(
            f"CP implementation with KV all-gather does not support {qkv_format} format yet!"
        )
    if cp_comm_type == "all_gather" and "causal" not in config.attn_mask_type:
        pytest.skip(
            f"CP implementation with KV all-gather does not support {config.attn_mask_type} mask"
            " type yet!"
        )
    if cp_comm_type == "all_gather" and config.attn_bias_type != "no_bias":
        pytest.skip(
            f"CP implementation with KV all-gather does not support {config.attn_bias_type} bias"
            " type yet!"
        )
    if cp_comm_type == "p2p" and config.window_size != (-1, 0) and config.window_size != (-1, -1):
        pytest.skip(
            f"CP implementation with KV P2P does not support window size {config.window_size} yet!"
        )

    subprocess.run(
        get_bash_arguments(
            dtype=dtype, model=model, qkv_format=qkv_format, kernel_backend="FlashAttention"
        ),
        check=True,
    )


model_configs_fused_attn = {
    #   test:             b,  h, hg,   d,   sq,  skv,   p,     mask,      bias
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
@pytest.mark.parametrize("dtype", ["bf16", "fp16", "fp8"])
@pytest.mark.parametrize("model", model_configs_fused_attn.keys())
@pytest.mark.parametrize("qkv_format", ["bshd", "sbhd", "thd"])
@pytest.mark.parametrize("cp_comm_type", ["p2p", "all_gather"])
def test_cp_with_fused_attention(dtype, model, qkv_format, cp_comm_type):
    if qkv_format == "thd" and get_device_compute_capability() < (9, 0):
        pytest.skip("THD format is only supported on sm90+.")
    if cp_comm_type == "all_gather" and get_cudnn_version() < (9, 3, 0):
        pytest.skip("CP implementation with KV all-gather is only supported with cuDNN >= 9.3.0")

    config = model_configs_fused_attn[model]
    if qkv_format == "thd" and config.num_heads != config.num_gqa_groups:
        pytest.skip(f"{qkv_format} format does not support QGA/MQA yet!")
    if qkv_format == "thd" and config.attn_bias_type == "post_scale_bias":
        pytest.skip(f"{qkv_format} format does not support {config.attn_bias_type} bias type yet!")
    if cp_comm_type == "all_gather" and qkv_format == "thd":
        pytest.skip(
            f"CP implementation with KV all-gather does not support {qkv_format} format yet!"
        )
    if cp_comm_type == "all_gather" and "causal" not in config.attn_mask_type:
        pytest.skip(
            f"CP implementation with KV all-gather does not support {config.attn_mask_type} mask"
            " type yet!"
        )
    if cp_comm_type == "all_gather" and config.attn_bias_type != "no_bias":
        pytest.skip(
            f"CP implementation with KV all-gather does not support {config.attn_bias_type} bias"
            " type yet!"
        )
    if config.window_size != (-1, 0) and config.window_size != (-1, -1):
        pytest.skip(
            f"Fused attention does not support sliding window attention + context parallelism yet!"
        )
    if cp_comm_type == "all_gather" and dtype == "fp8":
        pytest.skip(
            f"CP implementation with KV all-gather does not support FP8 + context parallelism yet!"
        )
    if dtype == "fp8" and qkv_format == "thd":
        pytest.skip(
            f"FP8 attention cannot work with THD format yet!"
        )
    if dtype == "fp8" and config.attn_bias_type != "no_bias":
        pytest.skip(
            f"FP8 attention cannot work with bias yet!"
        )


    subprocess.run(
        get_bash_arguments(
            dtype=dtype, model=model, qkv_format=qkv_format, kernel_backend="FusedAttention"
        ),
        check=True,
    )
