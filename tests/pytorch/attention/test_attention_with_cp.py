# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import os
import subprocess
import sys
import pathlib
import logging
import copy
import pytest
import torch
from transformer_engine.pytorch import (
    get_device_compute_capability,
    get_cudnn_version,
)
from transformer_engine.common.recipe import (
    DelayedScaling,
    Float8CurrentScaling,
    MXFP8BlockScaling,
    Format,
)
from transformer_engine.pytorch.attention.dot_product_attention.utils import FlashAttentionUtils

_current_file = pathlib.Path(__file__).resolve()
sys.path.append(str(_current_file.parent.parent))
from utils import ModelConfig, get_available_attention_backends, run_distributed

pytest_logging_level = logging.getLevelName(logging.root.level)

# Get determinism
_deterministic = (
    not bool(int(os.getenv("NVTE_ALLOW_NONDETERMINISTIC_ALGO", "1")))
    or torch.are_deterministic_algorithms_enabled()
)

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
    "cp_2_0": ModelConfig(2, 4096, 32, 128, num_gqa_groups=4, attn_mask_type="causal"),  # GQA
    "cp_2_1": ModelConfig(2, 4096, 12, 128, num_gqa_groups=2),  # GQA
    "cp_2_2": ModelConfig(2, 4096, 32, 128, attn_mask_type="causal", window_size=(128, 0)),  # GQA
    "cp_2_3": ModelConfig(2, 4096, 12, 128, num_gqa_groups=2, window_size=(512, 512)),  # GQA
    "cp_3_0": ModelConfig(2, 4096, 128, 192, attn_mask_type="causal", head_dim_v=128),  # MLA
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
    configs = ["cp_2_0", "cp_2_2", "cp_3_0", "cp_3_3"]
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

    if config.attn_bias_type != "no_bias" and qkv_format == "thd":
        pytest.skip("No support for bias with THD format!")
    if config.attn_bias_type != "no_bias" and cp_comm_type in ["all_gather", "a2a", "a2a+p2p"]:
        pytest.skip("No support for bias with cp_comm_type={all_gather, a2a, a2a+p2p}!")

    if qkv_format == "thd" and cp_comm_type in ["all_gather", "a2a+p2p"]:
        pytest.skip("No support for THD format with cp_comm_type={all_gather, a2a+p2p}!")

    if (
        config.window_size != (-1, 0)
        and config.window_size != (-1, -1)
        and cp_comm_type
        in [
            "p2p",
            "a2a+p2p",
        ]
    ):
        pytest.skip("No support for SWA with cp_comm_type={p2p, a2a+p2p}!")

    if cp_comm_type in ["a2a", "a2a+p2p"] and (
        config.num_heads % 2 != 0 or config.num_gqa_groups % 2 != 0
    ):
        pytest.skip(
            f"cp_comm_type=a2a requires num_heads ({config.num_heads}) and"
            f" num_gqa_groups ({config.num_gqa_groups}) divisible by 2!"
        )

    # FlashAttention / CP implementation specific: MLA only with KV P2P
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

    run_distributed(
        get_bash_arguments(
            num_gpus_per_node=num_gpus,
            dtype=dtype,
            model=model,
            qkv_format=qkv_format,
            kernel_backend="FlashAttention",
            cp_comm_type=cp_comm_type,
            log_level=pytest_logging_level,
        ),
    )


model_configs_fused_attn = {
    # test: ModelConfig(b, sq, hq, dqk)
    "cp_1_0": ModelConfig(2, 4096, 12, 128, attn_mask_type="causal", return_max_logit=True),  # MHA
    "cp_1_1": ModelConfig(2, 4096, 12, 128, return_max_logit=True),  # MHA
    "cp_1_2": ModelConfig(
        2, 4096, 12, 128, attn_mask_type="causal", attn_bias_type="post_scale_bias"
    ),  # MHA
    "cp_1_3": ModelConfig(2, 4096, 12, 128, attn_bias_type="post_scale_bias"),  # MHA
    "cp_1_4": ModelConfig(
        2, 4096, 12, 128, attn_bias_type="post_scale_bias", bias_shape="bhss"
    ),  # MHA
    "cp_1_5": ModelConfig(2, 4096, 12, 128, attn_mask_type="causal", window_size=(512, 512)),  # MHA
    "cp_2_0": ModelConfig(
        2,
        4096,
        32,
        128,
        num_gqa_groups=4,
        attn_mask_type="causal",
    ),  # GQA
    "cp_2_1": ModelConfig(
        2,
        4096,
        32,
        128,
        attn_mask_type="causal",
        window_size=(128, 0),
    ),  # GQA
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
        2,
        4096,
        12,
        128,
        num_gqa_groups=2,
        attn_mask_type="causal",
        attn_bias_type="post_scale_bias",
        bias_shape="11ss",
    ),  # GQA
    "cp_2_4": ModelConfig(
        2,
        4096,
        12,
        128,
        num_gqa_groups=2,
        attn_mask_type="causal",
        attn_bias_type="post_scale_bias",
        bias_shape="111s",
        return_max_logit=True,
    ),  # GQA
    "cp_2_5": ModelConfig(
        2, 4096, 12, 128, num_gqa_groups=2, attn_bias_type="post_scale_bias"
    ),  # GQA
    "cp_2_6": ModelConfig(
        2, 4096, 12, 128, num_gqa_groups=2, attn_mask_type="causal", window_size=(512, 512)
    ),  # GQA
    "cp_3_0": ModelConfig(2, 4096, 12, 128, attn_mask_type="causal", head_dim_v=64),  # MLA
    "cp_3_1": ModelConfig(2, 4096, 128, 192, head_dim_v=128, attn_mask_type="causal"),  # MLA
    "cp_3_2": ModelConfig(
        2, 4096, 12, 128, attn_mask_type="causal", attn_bias_type="post_scale_bias", head_dim_v=64
    ),  # MLA
    "cp_3_3": ModelConfig(2, 4096, 12, 128, attn_bias_type="post_scale_bias", head_dim_v=64),  # MLA
    "cp_3_4": ModelConfig(
        2, 4096, 12, 128, attn_bias_type="post_scale_bias", bias_shape="b1ss", head_dim_v=64
    ),  # MLA
    "cp_4_0": ModelConfig(
        2, 4096, 64, 64, num_gqa_groups=8, attn_mask_type="causal", softmax_type="vanilla"
    ),  # GQA
    "cp_4_1": ModelConfig(
        2, 4096, 64, 64, num_gqa_groups=8, attn_mask_type="causal", softmax_type="off-by-one"
    ),  # GQA
    "cp_4_2": ModelConfig(
        2, 4096, 64, 64, num_gqa_groups=8, attn_mask_type="causal", softmax_type="learnable"
    ),  # GQA
    "cp_4_3": ModelConfig(
        2, 4096, 64, 64, attn_mask_type="causal", window_size=(128, 0), softmax_type="learnable"
    ),  # GQA
}


dtypes = ["bf16", "fp16", "fp8"]
qkv_formats = ["bshd", "sbhd", "thd"]
cp_comm_types = ["p2p", "all_gather", "a2a", "a2a+p2p"]
if test_essential:
    configs = [
        "cp_1_0",
        "cp_2_0",
        "cp_2_1",
        "cp_2_2",
        "cp_2_4",
        "cp_3_1",
        "cp_3_2",
        "cp_3_4",
        "cp_4_2",
        "cp_4_3",
    ]
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
@pytest.mark.parametrize("scaling_mode", [None, "delayed", "current", "mxfp8"])
@pytest.mark.parametrize("f16_O", [True, False])
def test_cp_with_fused_attention(
    dtype, model, qkv_format, cp_comm_type, fp8_bwd, fp8_mha, fp8_dpa, scaling_mode, f16_O
):
    config = model_configs_fused_attn[model]
    config.context_parallel = True
    config.cp_comm_type = cp_comm_type

    num_gpus = 4 if cp_comm_type == "a2a+p2p" else 2
    if num_gpus > torch.cuda.device_count():
        pytest.skip(f"Test requires {num_gpus} GPUs, but found {torch.cuda.device_count()} GPUs.")

    if get_device_compute_capability() < (9, 0) and qkv_format == "thd":
        pytest.skip("Only sm90+ architectures support THD format!")
    if get_device_compute_capability() < (9, 0) and dtype == "fp8":
        pytest.skip("Only sm90+ architectures support FP8 attention!")

    if dtype == "fp8" and not (fp8_mha or fp8_dpa):
        pytest.skip("dtype=fp8 requires fp8_dpa=True or fp8_mha=True!")
    if dtype == "fp8" and not fp8_dpa and fp8_mha:
        pytest.skip("Duplicate tests to fp8_dpa=True and fp8_mha=True!")
    if dtype != "fp8" and fp8_bwd:
        pytest.skip("fp8_bwd=True requires dtype=fp8!")
    if dtype != "fp8" and (fp8_mha or fp8_dpa):
        pytest.skip("dtype!=fp8 requires fp8_dpa=False and fp8_mha=False!")

    if dtype == "fp8" and qkv_format == "thd":
        pytest.skip("No support for FP8 attention with THD format!")
    if dtype == "fp8" and config.attn_bias_type != "no_bias":
        pytest.skip("No support for FP8 attention with bias!")

    if config.attn_bias_type != "no_bias" and qkv_format == "thd":
        pytest.skip("No support for bias with THD format!")
    if config.attn_bias_type != "no_bias" and cp_comm_type in ["all_gather", "a2a", "a2a+p2p"]:
        pytest.skip("No support for bias with cp_comm_type={all_gather, a2a, a2a+p2p}!")

    if qkv_format == "thd" and cp_comm_type in ["all_gather", "a2a+p2p"]:
        pytest.skip("No support for THD format with cp_comm_type={all_gather, a2a+p2p}!")

    if (config.window_size[0] != -1 or config.window_size[1] not in [-1, 0]) and cp_comm_type in [
        "p2p",
        "a2a+p2p",
    ]:
        pytest.skip("No support for SWA with cp_comm_type={p2p, a2a+p2p}!")

    if cp_comm_type in ["a2a", "a2a+p2p"] and (
        config.num_heads % 2 != 0 or config.num_gqa_groups % 2 != 0
    ):
        pytest.skip(
            f"cp_comm_type=a2a requires num_heads ({config.num_heads}) and"
            f" num_gqa_groups ({config.num_gqa_groups}) divisible by 2!"
        )

    if config.softmax_type != "vanilla" and cp_comm_type != "a2a":
        pytest.skip(f"No support for non-vanilla softmax with cp_comm_type={cp_comm_type}!")
    if (
        config.softmax_type != "vanilla"
        and qkv_format == "thd"
        and get_cudnn_version() < (9, 18, 0)
    ):
        pytest.skip("No support for non-vanilla softmax with THD format and cuDNN < 9.18.0!")

    if dtype == "fp8" and scaling_mode is None:
        pytest.skip("dtype=fp8 requires scaling_mode != None!")
    if dtype != "fp8" and scaling_mode is not None:
        pytest.skip("dtype!=fp8 requires scaling_mode = None!")
    if dtype != "fp8" and not f16_O:
        pytest.skip("dtype!=fp8 requires f16_O=True!")
    if scaling_mode == "delayed" and f16_O:
        pytest.skip("scaling_mode=delayed requires f16_O=False!")
    if scaling_mode == "mxfp8" and not f16_O:
        pytest.skip("scaling_mode=mxfp8 requires f16_O=True!")
    if scaling_mode == "mxfp8" and fp8_mha:
        pytest.skip("No support for scaling_mode=mxfp8 with fp8_mha=True!")

    dtypes = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp8": torch.bfloat16}

    if qkv_format == "thd":
        config = copy.deepcopy(config)
        if "causal" in config.attn_mask_type:
            config.attn_mask_type = "padding_causal"
        else:
            config.attn_mask_type = "padding"

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
    if fp8 and scaling_mode == "mxfp8":
        fp8_meta["recipe"] = MXFP8BlockScaling(fp8_format=Format.E4M3, fp8_dpa=True)
        fp8_meta["local_recipes"] = [
            MXFP8BlockScaling(fp8_format=Format.E4M3, fp8_dpa=True),
        ]

    # For 111s, dbias calculation is not supported as of cuDNN 9.18, hence, test fwd only for 111s.
    is_training = False if config.bias_shape == "111s" else True
    available_backends, _, fused_attn_backends = get_available_attention_backends(
        config,
        qkv_dtype=dtypes[dtype] if dtype != "fp8" else torch.float8_e4m3fn,
        qkv_layout="_".join([qkv_format] * 3),
        fp8=fp8,
        fp8_meta=fp8_meta,
        is_training=is_training,
        deterministic=_deterministic,
    )
    _, fused_attn_supported, _ = available_backends
    if fused_attn_supported and config.attn_mask_type in ["causal", "padding_causal"]:
        config_copy = copy.deepcopy(config)
        config_copy.context_parallel = False
        config_copy.attn_mask_type = config.attn_mask_type + "_bottom_right"
        available_backends, _, fused_attn_backends = get_available_attention_backends(
            config_copy,
            qkv_dtype=dtypes[dtype] if dtype != "fp8" else torch.float8_e4m3fn,
            qkv_layout="_".join([qkv_format] * 3),
            fp8=fp8,
            fp8_meta=fp8_meta,
            is_training=is_training,
            deterministic=_deterministic,
        )
        _, fused_attn_supported, _ = available_backends
    if not fused_attn_supported:
        pytest.skip("No attention backend available.")

    run_distributed(
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
            is_training=is_training,
            deterministic=_deterministic,
            log_level=pytest_logging_level,
        ),
    )
