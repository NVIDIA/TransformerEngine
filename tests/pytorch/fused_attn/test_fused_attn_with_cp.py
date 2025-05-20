# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import os
import subprocess

import pytest
import torch
from transformer_engine.pytorch.utils import (
    get_device_compute_capability,
    get_cudnn_version,
)
from transformer_engine.pytorch.attention.dot_product_attention.utils import FlashAttentionUtils
from test_fused_attn import ModelConfig

model_configs_flash_attn = {
    #   test:             b,  h, hg,   d,   sq,  skv,   p,     mask,      bias
    "cp_1_0": ModelConfig(2, 12, 12, 128, 4096, 4096, 0.0, "causal", "no_bias"),  # MHA
    "cp_1_1": ModelConfig(2, 12, 12, 128, 4096, 4096, 0.0, "no_mask", "no_bias"),  # MHA
    "cp_1_2": ModelConfig(
        2, 12, 12, 128, 4096, 4096, 0.0, "causal", "no_bias", window_size=(512, 0)
    ),  # MHA
    "cp_1_3": ModelConfig(
        2, 12, 12, 128, 4096, 4096, 0.0, "no_mask", "no_bias", window_size=(512, 512)
    ),  # MHA
    "cp_2_0": ModelConfig(2, 12, 2, 128, 4096, 4096, 0.0, "causal", "no_bias"),  # GQA
    "cp_2_1": ModelConfig(2, 12, 2, 128, 4096, 4096, 0.0, "no_mask", "no_bias"),  # GQA
    "cp_2_2": ModelConfig(
        2, 12, 2, 128, 4096, 4096, 0.0, "causal", "no_bias", window_size=(512, 0)
    ),  # GQA
    "cp_2_3": ModelConfig(
        2, 12, 2, 128, 4096, 4096, 0.0, "no_mask", "no_bias", window_size=(512, 512)
    ),  # GQA
}


def get_bash_arguments(num_gpus_per_node, **kwargs):
    args = [
        "python3",
        "-m",
        "torch.distributed.launch",
        "--nproc-per-node=" + str(num_gpus_per_node),
    ]
    te_path = os.getenv("TE_PATH", "/opt/transformerengine")
    script_path = os.path.join(te_path, "tests/pytorch/fused_attn/run_fused_attn_with_cp.py")
    args.append(script_path)
    for k, v in kwargs.items():
        args.append(f"{k}={v}")
    return args


@pytest.mark.skipif(not FlashAttentionUtils.v2_plus, reason="Flash-attn 2.0+ is required.")
@pytest.mark.skipif(get_device_compute_capability() < (8, 0), reason="CP tests require sm80+.")
@pytest.mark.parametrize("dtype", ["bf16", "fp16"])
@pytest.mark.parametrize("model", model_configs_flash_attn.keys())
@pytest.mark.parametrize("qkv_format", ["bshd", "sbhd", "thd"])
@pytest.mark.parametrize("cp_comm_type", ["p2p", "all_gather", "a2a", "a2a+p2p"])
def test_cp_with_flash_attention(dtype, model, qkv_format, cp_comm_type):
    num_gpus = 4 if cp_comm_type == "a2a+p2p" else 2
    if num_gpus > torch.cuda.device_count():
        pytest.skip(f"Test requires {num_gpus} GPUs, but found {torch.cuda.device_count()}")

    config = model_configs_flash_attn[model]
    if "p2p" in cp_comm_type and config.window_size != (-1, 0) and config.window_size != (-1, -1):
        pytest.skip("CP implementation with KV P2P does not support sliding window yet!")
    if cp_comm_type == "all_gather" and qkv_format == "thd":
        pytest.skip("CP implementation with KV all-gather does not support THD format yet!")
    if cp_comm_type == "all_gather" and config.attn_bias_type != "no_bias":
        pytest.skip("CP implementation with KV all-gather does not support bias yet!")
    if "a2a" in cp_comm_type and qkv_format == "thd":
        pytest.skip("CP implementation with QKVO A2A does not support THD format yet!")
    if "a2a" in cp_comm_type and config.attn_bias_type != "no_bias":
        pytest.skip("CP implementation with QKVO A2A does not support bias yet!")
    if "a2a" in cp_comm_type and (config.num_heads % 2 != 0 or config.num_gqa_groups % 2 != 0):
        pytest.skip(
            f"CP implementation with QKVO A2A requires num_heads ({config.num_heads}) and"
            f" num_gqa_groups ({config.num_gqa_groups}) to be divisible by cp_size (2)!"
        )

    subprocess.run(
        get_bash_arguments(
            num_gpus_per_node=num_gpus,
            dtype=dtype,
            model=model,
            qkv_format=qkv_format,
            kernel_backend="FlashAttention",
            cp_comm_type=cp_comm_type,
        ),
        check=True,
    )


model_configs_fused_attn = {
    #   test:             b,  h, hg,   d,   sq,  skv,   p,     mask,      bias
    "cp_1_0": ModelConfig(2, 12, 12, 128, 4096, 4096, 0.0, "causal", "no_bias"),  # MHA
    "cp_1_1": ModelConfig(2, 12, 12, 128, 4096, 4096, 0.0, "no_mask", "no_bias"),  # MHA
    "cp_1_2": ModelConfig(2, 12, 12, 128, 4096, 4096, 0.0, "causal", "post_scale_bias"),  # MHA
    "cp_1_3": ModelConfig(2, 12, 12, 128, 4096, 4096, 0.0, "no_mask", "post_scale_bias"),  # MHA
    "cp_1_4": ModelConfig(
        2, 12, 12, 128, 4096, 4096, 0.0, "causal", "no_bias", window_size=(512, 0)
    ),  # MHA
    "cp_2_0": ModelConfig(2, 12, 2, 128, 4096, 4096, 0.0, "causal", "no_bias"),  # GQA
    "cp_2_1": ModelConfig(2, 12, 2, 128, 4096, 4096, 0.0, "no_mask", "no_bias"),  # GQA
    "cp_2_2": ModelConfig(2, 12, 2, 128, 4096, 4096, 0.0, "causal", "post_scale_bias"),  # GQA
    "cp_2_3": ModelConfig(2, 12, 2, 128, 4096, 4096, 0.0, "no_mask", "post_scale_bias"),  # GQA
    "cp_2_4": ModelConfig(
        2, 12, 2, 128, 4096, 4096, 0.0, "causal", "no_bias", window_size=(512, 0)
    ),  # GQA
}


@pytest.mark.skipif(get_cudnn_version() < (8, 9, 7), reason="cuDNN 8.9.7+ is required.")
@pytest.mark.skipif(get_device_compute_capability() < (8, 0), reason="CP tests require sm80+.")
@pytest.mark.parametrize("dtype", ["bf16", "fp16", "fp8"])
@pytest.mark.parametrize("model", model_configs_fused_attn.keys())
@pytest.mark.parametrize("qkv_format", ["bshd", "sbhd", "thd"])
@pytest.mark.parametrize("cp_comm_type", ["p2p", "all_gather", "a2a", "a2a+p2p"])
@pytest.mark.parametrize("fp8_mha", [False, True])
def test_cp_with_fused_attention(dtype, model, qkv_format, cp_comm_type, fp8_mha):
    num_gpus = 4 if cp_comm_type == "a2a+p2p" else 2
    if num_gpus > torch.cuda.device_count():
        pytest.skip(f"Test requires {num_gpus} GPUs, but found {torch.cuda.device_count()}")

    if qkv_format == "thd" and get_device_compute_capability() < (9, 0):
        pytest.skip("THD format is only supported on sm90+!")
    if cp_comm_type == "all_gather" and get_cudnn_version() < (9, 3, 0):
        pytest.skip("CP implementation with KV all-gather is only supported with cuDNN >= 9.3.0!")
    if dtype == "fp8" and get_device_compute_capability() < (9, 0):
        pytest.skip("FP8 attention is only supported on sm90+!")

    config = model_configs_fused_attn[model]
    if qkv_format == "thd" and config.attn_bias_type == "post_scale_bias":
        pytest.skip("THD format does not support post_scale_bias yet!")
    if qkv_format == "thd" and cp_comm_type == "all_gather":
        pytest.skip("CP implementation with KV all-gather does not support THD format yet!")
    if qkv_format == "thd" and "a2a" in cp_comm_type:
        pytest.skip("CP implementation with QKVO A2A does not support THD format yet!")
    if dtype == "fp8" and cp_comm_type == "all_gather":
        pytest.skip(
            "CP implementation with KV all-gather does not support FP8 + context parallelism yet!"
        )
    if dtype == "fp8" and qkv_format == "thd":
        pytest.skip("FP8 attention cannot work with THD format yet!")
    if dtype == "fp8" and config.attn_bias_type != "no_bias":
        pytest.skip("FP8 attention cannot work with bias yet!")
    if dtype == "fp8" and config.window_size != (-1, 0) and config.window_size != (-1, -1):
        pytest.skip("FP8 attention cannot work with sliding window yet!")
    if "p2p" in cp_comm_type and config.window_size != (-1, 0) and config.window_size != (-1, -1):
        pytest.skip("CP implementation with KV P2P does not support sliding window yet!")
    if cp_comm_type == "all_gather" and config.attn_bias_type != "no_bias":
        pytest.skip("CP implementation with KV all-gather does not support bias yet!")
    if "a2a" in cp_comm_type and config.attn_bias_type != "no_bias":
        pytest.skip("CP implementation with QKVO A2A does not support bias yet!")
    if "a2a" in cp_comm_type and (config.num_heads % 2 != 0 or config.num_gqa_groups % 2 != 0):
        pytest.skip(
            f"CP implementation with QKVO A2A requires num_heads ({config.num_heads}) and"
            f" num_gqa_groups ({config.num_gqa_groups}) to be divisible by cp_size (2)!"
        )
    if dtype != "fp8" and fp8_mha:
        pytest.skip("Only fp8 works with fp8_mha=True!")

    subprocess.run(
        get_bash_arguments(
            num_gpus_per_node=num_gpus,
            dtype=dtype,
            model=model,
            qkv_format=qkv_format,
            kernel_backend="FusedAttention",
            cp_comm_type=cp_comm_type,
            fp8_mha=fp8_mha,
        ),
        check=True,
    )
