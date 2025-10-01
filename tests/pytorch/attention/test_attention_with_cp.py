# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import os
import subprocess
import sys
import pathlib
import logging

import pytest
import torch
from transformer_engine.pytorch.utils import (
    get_device_compute_capability,
    get_cudnn_version,
)
from transformer_engine.common.recipe import (
    DelayedScaling,
    Float8CurrentScaling,
)
from transformer_engine.pytorch.attention.dot_product_attention.utils import FlashAttentionUtils

_current_file = pathlib.Path(__file__).resolve()
sys.path.append(str(_current_file.parent.parent))
from utils import ModelConfig, get_available_attention_backends

pytest_logging_level = logging.getLevelName(logging.root.level)

# Initialize RNG state
seed = 1234
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

test_essential = True

model_configs_flash_attn = {
    # test: ModelConfig(b, sq, hq, dqk)
    "cp_1_0": ModelConfig(2, 4096, 12, 128, attn_mask_type="causal"),  # MHA
    "cp_1_1": ModelConfig(2, 4096, 12, 128),  # MHA
    "cp_1_2": ModelConfig(2, 4096, 12, 128, attn_mask_type="causal", window_size=(512, 0)),  # MHA
    "cp_1_3": ModelConfig(2, 4096, 12, 128, window_size=(512, 512)),  # MHA
    "cp_2_0": ModelConfig(2, 4096, 12, 128, num_gqa_groups=2, attn_mask_type="causal"),  # GQA
    "cp_2_1": ModelConfig(2, 4096, 12, 128, num_gqa_groups=2),  # GQA
    "cp_2_2": ModelConfig(
        2, 4096, 12, 128, num_gqa_groups=2, attn_mask_type="causal", window_size=(512, 0)
    ),  # GQA
    "cp_2_3": ModelConfig(2, 4096, 12, 128, num_gqa_groups=2, window_size=(512, 512)),  # GQA
    "cp_3_0": ModelConfig(2, 4096, 12, 192, attn_mask_type="causal", head_dim_v=128),  # MLA
    "cp_3_1": ModelConfig(2, 4096, 12, 192, head_dim_v=128),  # MLA
    "cp_3_2": ModelConfig(
        2, 4096, 12, 192, attn_mask_type="causal", window_size=(512, 0), head_dim_v=128
    ),  # MLA
    "cp_3_3": ModelConfig(2, 4096, 12, 192, window_size=(512, 512), head_dim_v=128),  # MLA
}


def get_bash_arguments(num_gpus_per_node, **kwargs):
    args = [
        "python3",
        "-m",
        "torch.distributed.launch",
        "--nproc-per-node=" + str(num_gpus_per_node),
    ]
    te_path = os.getenv("TE_PATH", "/opt/transformerengine")
    script_path = os.path.join(te_path, "tests/pytorch/attention/run_attention_with_cp.py")
    args.append(script_path)
    for k, v in kwargs.items():
        args.append(f"{k}={v}")
    return args


dtypes = ["bf16", "fp16"]
qkv_formats = ["bshd", "sbhd", "thd"]
cp_comm_types = ["p2p", "all_gather", "a2a", "a2a+p2p"]
if test_essential:
    configs = ["cp_1_0", "cp_2_1", "cp_3_2", "cp_3_3"]
    model_configs_flash_attn = {k: model_configs_flash_attn[k] for k in configs}
    dtypes = ["bf16"]
    qkv_formats = ["sbhd", "thd"]


@pytest.mark.skipif(not FlashAttentionUtils.v2_plus, reason="Flash-attn 2.0+ is required.")
@pytest.mark.skipif(get_device_compute_capability() < (8, 0), reason="CP tests require sm80+.")
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("model", model_configs_flash_attn.keys())
@pytest.mark.parametrize("qkv_format", qkv_formats)
@pytest.mark.parametrize("cp_comm_type", cp_comm_types)
def test_cp_with_flash_attention(dtype, model, qkv_format, cp_comm_type):
    num_gpus = 4 if cp_comm_type == "a2a+p2p" else 2
    if num_gpus > torch.cuda.device_count():
        pytest.skip(f"Test requires {num_gpus} GPUs, but found {torch.cuda.device_count()}")

    config = model_configs_flash_attn[model]
    config.context_parallel = True
    config.cp_comm_type = cp_comm_type

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
    if "p2p" not in cp_comm_type and config.head_dim_qk != config.head_dim_v:
        pytest.skip("MLA CP currently only support KV P2P!")
    dtypes = {"fp16": torch.float16, "bf16": torch.bfloat16}
    available_backends, *_ = get_available_attention_backends(
        config,
        qkv_dtype=dtypes[dtype],
        qkv_layout="_".join([qkv_format] * 3),
    )
    flash_attn_supported, *_ = available_backends
    if not flash_attn_supported:
        pytest.skip("No attention backend available.")

    subprocess.run(
        get_bash_arguments(
            num_gpus_per_node=num_gpus,
            dtype=dtype,
            model=model,
            qkv_format=qkv_format,
            kernel_backend="FlashAttention",
            cp_comm_type=cp_comm_type,
            log_level=pytest_logging_level,
        ),
        check=True,
    )


model_configs_fused_attn = {
    # test: ModelConfig(b, sq, hq, dqk)
    "cp_1_0": ModelConfig(2, 4096, 12, 128, attn_mask_type="causal"),  # MHA
    "cp_1_1": ModelConfig(2, 4096, 12, 128),  # MHA
    "cp_1_2": ModelConfig(
        2, 4096, 12, 128, attn_mask_type="causal", attn_bias_type="post_scale_bias"
    ),  # MHA
    "cp_1_3": ModelConfig(2, 4096, 12, 128, attn_bias_type="post_scale_bias"),  # MHA
    "cp_1_4": ModelConfig(2, 4096, 12, 128, attn_mask_type="causal", window_size=(512, 0)),  # MHA
    "cp_2_0": ModelConfig(2, 4096, 12, 128, num_gqa_groups=2, attn_mask_type="causal"),  # GQA
    "cp_2_1": ModelConfig(2, 4096, 12, 128, num_gqa_groups=2),  # GQA
    "cp_2_2": ModelConfig(
        2,
        4096,
        12,
        128,
        num_gqa_groups=2,
        attn_mask_type="causal",
        attn_bias_type="post_scale_bias",
    ),  # GQA
    "cp_2_3": ModelConfig(
        2, 4096, 12, 128, num_gqa_groups=2, attn_bias_type="post_scale_bias"
    ),  # GQA
    "cp_2_4": ModelConfig(
        2, 4096, 12, 128, num_gqa_groups=2, attn_mask_type="causal", window_size=(512, 0)
    ),  # GQA
    "cp_3_0": ModelConfig(2, 4096, 12, 128, attn_mask_type="causal", head_dim_v=64),  # MLA
    "cp_3_1": ModelConfig(2, 4096, 12, 128, head_dim_v=64),  # MLA
    "cp_3_2": ModelConfig(
        2, 4096, 12, 128, attn_mask_type="causal", attn_bias_type="post_scale_bias", head_dim_v=64
    ),  # MLA
    "cp_3_3": ModelConfig(2, 4096, 12, 128, attn_bias_type="post_scale_bias", head_dim_v=64),  # MLA
    "cp_4_0": ModelConfig(
        2, 4096, 64, 64, num_gqa_groups=8, attn_mask_type="causal", softmax_type="vanilla"
    ),  # GQA
    "cp_4_1": ModelConfig(
        2, 4096, 64, 64, num_gqa_groups=8, attn_mask_type="causal", softmax_type="off-by-one"
    ),  # GQA
    "cp_4_2": ModelConfig(
        2, 4096, 64, 64, num_gqa_groups=8, attn_mask_type="causal", softmax_type="learnable"
    ),  # GQA
}


dtypes = ["bf16", "fp16", "fp8"]
qkv_formats = ["bshd", "sbhd", "thd"]
cp_comm_types = ["p2p", "all_gather", "a2a", "a2a+p2p"]
if test_essential:
    configs = ["cp_1_0", "cp_2_0", "cp_2_2", "cp_3_2", "cp_4_2"]
    model_configs_fused_attn = {k: model_configs_fused_attn[k] for k in configs}
    dtypes = ["bf16", "fp8"]
    qkv_formats = ["sbhd", "thd"]


@pytest.mark.skipif(get_cudnn_version() < (8, 9, 7), reason="cuDNN 8.9.7+ is required.")
@pytest.mark.skipif(get_device_compute_capability() < (8, 0), reason="CP tests require sm80+.")
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("model", model_configs_fused_attn.keys())
@pytest.mark.parametrize("qkv_format", qkv_formats)
@pytest.mark.parametrize("cp_comm_type", cp_comm_types)
@pytest.mark.parametrize("fp8_bwd", [True, False])
@pytest.mark.parametrize("fp8_mha", [True, False])
@pytest.mark.parametrize("fp8_dpa", [True, False])
@pytest.mark.parametrize("scaling_mode", [None, "delayed", "current"])
@pytest.mark.parametrize("f16_O", [True, False])
def test_cp_with_fused_attention(
    dtype, model, qkv_format, cp_comm_type, fp8_bwd, fp8_mha, fp8_dpa, scaling_mode, f16_O
):
    num_gpus = 4 if cp_comm_type == "a2a+p2p" else 2
    if num_gpus > torch.cuda.device_count():
        pytest.skip(f"Test requires {num_gpus} GPUs, but found {torch.cuda.device_count()}")

    if qkv_format == "thd" and get_device_compute_capability() < (9, 0):
        pytest.skip("THD format is only supported on sm90+!")
    if cp_comm_type == "all_gather" and get_cudnn_version() < (9, 3, 0):
        pytest.skip("CP implementation with KV all-gather is only supported with cuDNN >= 9.3.0!")
    if dtype == "fp8" and get_device_compute_capability() < (9, 0):
        pytest.skip("FP8 attention is only supported on sm90+!")
    if dtype == "fp8" and not fp8_dpa and fp8_mha:
        pytest.skip("Duplicate tests to fp8_dpa=True and fp8_mha=True!")
    if dtype != "fp8" and fp8_bwd:
        pytest.skip("Only fp8 works with fp8_bwd=True!")

    config = model_configs_fused_attn[model]
    config.context_parallel = True
    config.cp_comm_type = cp_comm_type

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
    if dtype != "fp8" and (fp8_mha or fp8_dpa):
        pytest.skip("Only fp8 works with fp8_dpa=True or fp8_mha=True!")
    if dtype == "fp8" and not (fp8_mha or fp8_dpa):
        pytest.skip("fp8 only works with fp8_dpa=True or fp8_mha=True!")
    if dtype != "fp8" and scaling_mode is not None:
        pytest.skip("Only fp8 works with scaling_mode != None!")
    if dtype == "fp8" and scaling_mode is None:
        pytest.skip("fp8 only works with scaling_mode != None!")
    if (
        dtype == "fp8"
        and scaling_mode == "current"
        and cp_comm_type not in ["p2p", "a2a+p2p", "a2a"]
    ):
        pytest.skip("fp8 only works with P2P, A2A and A2A+P2P for scaling_mode = current!")
    if f16_O and (dtype != "fp8" or scaling_mode != "current"):
        pytest.skip("f16_O only needs to be tested for dtype = fp8 and scaling_mode = current!")
    if "p2p" not in cp_comm_type and config.head_dim_qk != config.head_dim_v:
        pytest.skip("MLA CP currently only support KV P2P!")
    if dtype == "fp8" and config.head_dim_qk != config.head_dim_v:
        pytest.skip("MLA CP currently does not support FP8 attention!")
    if dtype == "fp8" and config.softmax_type != "vanilla":
        pytest.skip("CP implementation does not support non-vanilla softmax types in FP8!")
    if config.softmax_type != "vanilla" and cp_comm_type != "a2a":
        pytest.skip(
            "CP implementation only supports cp_comm_type=a2a for non-vanilla softmax types!"
        )
    if config.softmax_type != "vanilla" and qkv_format == "thd":
        pytest.skip(
            "CP implementation does not support qkv_format=thd for non-vanilla softmax types!"
        )

    dtypes = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp8": torch.bfloat16}
    fp8_meta = {}
    fp8_meta["recipe"] = None
    fp8_meta["local_recipes"] = []
    fp8 = dtype == "fp8" and (fp8_dpa or fp8_mha)
    if fp8 and scaling_mode == "delayed":
        fp8_meta["recipe"] = DelayedScaling(fp8_dpa=True)
        fp8_meta["local_recipes"] = [DelayedScaling(fp8_dpa=True)]
    if fp8 and scaling_mode == "current":
        fp8_meta["recipe"] = DelayedScaling(fp8_dpa=True)
        fp8_meta["local_recipes"] = [
            Float8CurrentScaling(fp8_dpa=True),
            DelayedScaling(fp8_dpa=True),
        ]
    available_backends, _, fused_attn_backends = get_available_attention_backends(
        config,
        qkv_dtype=dtypes[dtype] if dtype != "fp8" else torch.float8_e4m3fn,
        qkv_layout="_".join([qkv_format] * 3),
        fp8=fp8,
        fp8_meta=fp8_meta,
    )
    _, fused_attn_supported, _ = available_backends
    if not fused_attn_supported:
        pytest.skip("No attention backend available.")

    subprocess.run(
        get_bash_arguments(
            num_gpus_per_node=num_gpus,
            dtype=dtype,
            model=model,
            qkv_format=qkv_format,
            kernel_backend="FusedAttention",
            cp_comm_type=cp_comm_type,
            fp8_bwd=fp8_bwd,
            fp8_dpa=fp8_dpa,
            fp8_mha=fp8_mha,
            scaling_mode=scaling_mode,
            f16_O=f16_O,
            log_level=pytest_logging_level,
        ),
        check=True,
    )
