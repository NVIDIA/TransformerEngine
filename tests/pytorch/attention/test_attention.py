# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
import logging
import math
import os
import sys
import pathlib
from typing import Any, Dict, Tuple, Union

import pytest
import torch

from transformer_engine.common import recipe
from transformer_engine.pytorch import TransformerLayer, fp8_autocast, fp8_model_init
from transformer_engine.pytorch.attention.dot_product_attention import (
    DotProductAttention,
    _attention_backends,
)
from transformer_engine.pytorch.attention.multi_head_attention import MultiheadAttention
from transformer_engine.pytorch.attention.dot_product_attention.utils import (
    FlashAttentionUtils,
    check_set_window_size,
)
from transformer_engine.pytorch.attention import RotaryPositionEmbedding
import transformer_engine.pytorch.cpp_extensions as ext
from transformer_engine.pytorch.cpp_extensions.fused_attn import (
    FusedAttnBackend,
    fused_attn_bwd,
    fused_attn_fwd,
)
from transformer_engine.pytorch.distributed import CudaRNGStatesTracker
import transformer_engine.pytorch.fp8 as fp8
from transformer_engine.pytorch.module.base import TransformerEngineBaseModule
from transformer_engine.pytorch.utils import (
    get_device_compute_capability,
    init_method_normal,
    scaled_init_method_normal,
    is_bf16_compatible,
)
from transformer_engine.pytorch.utils import get_cudnn_version
import transformer_engine_torch as tex
from transformer_engine.pytorch.tensor.quantized_tensor import (
    Quantizer,
    prepare_for_saving,
    restore_from_saved,
)

_current_file = pathlib.Path(__file__).resolve()
sys.path.append(str(_current_file.parent.parent))
from utils import (
    reset_rng_states,
    ModelConfig,
    dtype_tols,
    get_available_attention_backends,
)

# Only run FP8 tests on H100
fp8_available, reason_for_no_fp8 = fp8.FP8GlobalStateManager.is_fp8_available()

seed = 1234
# Reset RNG states
reset_rng_states()


@pytest.fixture(autouse=True)
def reset_global_fp8_state():
    yield
    fp8.FP8GlobalStateManager.reset()


model_configs_base = {
    #     test:             b,  h, hg,  d,  sq, skv,   p,      mask,      bias
    "base_1_0": ModelConfig(8, 128, 16, 64),
    "base_1_1": ModelConfig(4, 128, 16, 64, max_seqlen_kv=256),
    "base_2_0": ModelConfig(2, 2048, 24, 128),
    "base_2_1": ModelConfig(1, 2048, 24, 128, max_seqlen_kv=4096),
    "base_3_0": ModelConfig(8, 1, 16, 128, max_seqlen_kv=2048),
    "base_3_1": ModelConfig(8, 1, 16, 256, max_seqlen_kv=2048),
    "base_4_0": ModelConfig(8, 1, 16, 192, max_seqlen_kv=2048),
    "base_4_1": ModelConfig(8, 128, 16, 192, max_seqlen_kv=2048),
    "base_5_0": ModelConfig(8, 1, 16, 512, max_seqlen_kv=2048),
    "base_5_1": ModelConfig(8, 128, 16, 512, max_seqlen_kv=2048),
    "base_6_0": ModelConfig(8, 1, 16, 1024, max_seqlen_kv=2048),
    "base_6_1": ModelConfig(8, 128, 16, 1024, max_seqlen_kv=2048),
}


param_types = [torch.float16]
if is_bf16_compatible():  # bf16 requires sm_80 or higher
    param_types.append(torch.bfloat16)
param_types_lean = [torch.bfloat16]


@pytest.mark.skipif(get_cudnn_version() < (8, 9, 1), reason="cuDNN 8.9.1+ is required.")
@pytest.mark.parametrize("dtype", param_types)
@pytest.mark.parametrize("model_configs", [model_configs_base])
@pytest.mark.parametrize("model", model_configs_base.keys())
@pytest.mark.parametrize("ckpt_attn", [False])
@pytest.mark.parametrize("workspace_opt", [True, False])
@pytest.mark.parametrize("qkv_layout", [None])
@pytest.mark.parametrize("swa", [False])
@pytest.mark.parametrize("pad_between_seqs", [False])
def test_dot_product_attention(
    dtype, model_configs, model, ckpt_attn, workspace_opt, qkv_layout, swa, pad_between_seqs
):
    """Test DotProductAttention module"""

    # Get configs
    tols = dict(atol=1e-3, rtol=1e-3)
    if dtype == torch.bfloat16:
        tols = dict(atol=1.5e-2, rtol=1.5e-2)
    config = model_configs[model]
    is_mla = config.head_dim_qk != config.head_dim_v
    is_mqa_gqa = config.num_heads != config.num_gqa_groups
    if qkv_layout is None:
        if config.attn_type == "self":
            qkv_layout = "sb3hd" if not is_mla and not is_mqa_gqa else "sbhd_sbhd_sbhd"
        else:
            qkv_layout = "bshd_bs2hd" if not is_mla and not is_mqa_gqa else "bshd_bshd_bshd"
    if "3" in qkv_layout and config.attn_type == "cross":
        pytest.skip("No need to test this layout for cross attention")

    if config.window_size == (-1, -1) and swa:
        config.window_size = [2, 2]
    config.window_size = check_set_window_size(config.attn_mask_type, config.window_size)

    is_training = True
    available_backends, _, fused_attn_backends = get_available_attention_backends(
        config,
        qkv_dtype=dtype,
        qkv_layout=qkv_layout,
        window_size=config.window_size,
        pad_between_seqs=pad_between_seqs,
        is_training=is_training,
    )
    flash_attn_supported, fused_attn_supported, unfused_attn_supported = available_backends
    if not fused_attn_supported:
        is_training = False
        available_backends, _, fused_attn_backends = get_available_attention_backends(
            config,
            qkv_dtype=dtype,
            qkv_layout=qkv_layout,
            window_size=config.window_size,
            pad_between_seqs=pad_between_seqs,
            is_training=is_training,
        )
        flash_attn_supported, fused_attn_supported, unfused_attn_supported = available_backends

    # FlashAttention does not support pad_between_seqs, but _run_dot_product_attention
    # mannually pads and unpads the input and output of FlashAttention for testing purposes
    if (
        pad_between_seqs
        and FlashAttentionUtils.is_installed
        and not (
            config.max_seqlen_q != config.max_seqlen_kv
            and config.attn_mask_type in ["causal", "padding_causal"]
        )
        and (config.window_size[0] == -1 or FlashAttentionUtils.v2_3_plus)
    ):
        flash_attn_supported = True

    # Skip if only unfused backend is supported
    if (len(fused_attn_backends) + flash_attn_supported + unfused_attn_supported) < 2:
        pytest.skip("Less than two backends to compare.")

    # UnfusedDotProductAttention backend
    if unfused_attn_supported:
        unfused_attn_fwd, unfused_attn_bwd = _run_dot_product_attention(
            dtype,
            config,
            "UnfusedDotProductAttention",
            ckpt_attn,
            qkv_layout,
            workspace_opt,
            pad_between_seqs,
            is_training,
        )

    # FusedAttention backend
    if fused_attn_supported:
        if len(fused_attn_backends) == 1:
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
        if len(fused_attn_backends) == 2:
            os.environ["NVTE_FUSED_ATTN_BACKEND"] = "0"
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
            os.environ["NVTE_FUSED_ATTN_BACKEND"] = "1"
            fused_attn_fwd_1, fused_attn_bwd_1 = _run_dot_product_attention(
                dtype,
                config,
                "FusedAttention",
                ckpt_attn,
                qkv_layout,
                workspace_opt,
                pad_between_seqs,
                is_training,
            )

    # FlashAttention backend
    if flash_attn_supported:
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

    logging.info(f"[test_dot_product_attention]: is_training = {is_training}")
    if unfused_attn_supported and flash_attn_supported:
        logging.info("[test_dot_product_attention]: unfused attn vs flash attn")
        torch.testing.assert_close(flash_attn_fwd, unfused_attn_fwd, **tols)
        for i, _ in enumerate(flash_attn_bwd):
            torch.testing.assert_close(unfused_attn_bwd[i], flash_attn_bwd[i], **tols)
    if unfused_attn_supported and fused_attn_supported:
        logging.info("[test_dot_product_attention]: unfused attn vs fused attn")
        torch.testing.assert_close(fused_attn_fwd, unfused_attn_fwd, **tols)
        for i, _ in enumerate(unfused_attn_bwd):
            torch.testing.assert_close(fused_attn_bwd[i], unfused_attn_bwd[i], **tols)
    if fused_attn_supported and flash_attn_supported:
        logging.info("[test_dot_product_attention]: fused attn vs flash attn")
        torch.testing.assert_close(fused_attn_fwd, flash_attn_fwd, **tols)
        for i, _ in enumerate(flash_attn_bwd):
            torch.testing.assert_close(fused_attn_bwd[i], flash_attn_bwd[i], **tols)
    if fused_attn_supported and len(fused_attn_backends) == 2:
        logging.info("[test_dot_product_attention]: fused attn backend 0 vs 1")
        torch.testing.assert_close(fused_attn_fwd, fused_attn_fwd_1, **tols)
        for i, _ in enumerate(fused_attn_bwd):
            torch.testing.assert_close(fused_attn_bwd[i], fused_attn_bwd_1[i], **tols)


@pytest.mark.skipif(get_cudnn_version() < (8, 9, 1), reason="cuDNN 8.9.1+ is required.")
@pytest.mark.parametrize("dtype", param_types)
@pytest.mark.parametrize("model_configs", [model_configs_base])
@pytest.mark.parametrize("model", ["base_1_1", "base_2_1"])
def test_dpa_checkpoint(dtype, model_configs, model):
    """Test DotProductAttention module with checkpointing"""
    test_dot_product_attention(dtype, model_configs, model, True, True, None, False, False)


model_configs_mla = {
    #    test:             b,  h, hg, dqk, sq, skv,   p,      mask,      bias   # attn , backend
    "mla_1_0": ModelConfig(8, 128, 16, 64, head_dim_v=128),  # self , 0
    "mla_1_1": ModelConfig(4, 128, 16, 64, max_seqlen_kv=256, head_dim_v=128),  # cross, 0
    "mla_1_2": ModelConfig(4, 128, 16, 192, max_seqlen_kv=256, head_dim_v=128),  # cross, 0
    "mla_2_0": ModelConfig(2, 2048, 24, 128, attn_mask_type="causal", head_dim_v=64),  # self , 1
    "mla_2_1": ModelConfig(
        1, 2048, 24, 128, max_seqlen_kv=4096, attn_mask_type="causal", head_dim_v=64
    ),  # cross, 1
    "mla_2_2": ModelConfig(
        1, 2048, 24, 192, max_seqlen_kv=4096, attn_mask_type="causal", head_dim_v=128
    ),  # cross, 1
    "mla_3_0": ModelConfig(8, 1, 16, 128, max_seqlen_kv=2048, head_dim_v=64),  # inference
    "mla_3_1": ModelConfig(8, 1, 16, 256, max_seqlen_kv=2048, head_dim_v=128),  # inference
    "mla_3_2": ModelConfig(8, 1, 16, 192, max_seqlen_kv=2048, head_dim_v=128),  # inference
}


@pytest.mark.skipif(get_cudnn_version() < (8, 9, 1), reason="cuDNN 8.9.1+ is required.")
@pytest.mark.parametrize("dtype", param_types)
@pytest.mark.parametrize("model_configs", [model_configs_mla])
@pytest.mark.parametrize("model", model_configs_mla.keys())
def test_dpa_mla(dtype, model_configs, model):
    """Test DotProductAttention module with Multi-Latent Attention (MLA)"""
    test_dot_product_attention(dtype, model_configs, model, True, True, None, False, False)


model_configs_mask = {
    #     test:             b,  h, hg,   d,   sq,  skv,   p,             mask,      bias
    "mask_1_0": ModelConfig(2, 2048, 16, 64, attn_mask_type="causal"),
    "mask_1_1": ModelConfig(2, 2048, 24, 128, num_gqa_groups=1, attn_mask_type="causal"),
    "mask_1_2": ModelConfig(2, 2048, 24, 128, max_seqlen_kv=4096, attn_mask_type="causal"),
    "mask_2_0": ModelConfig(2, 2048, 16, 64, attn_mask_type="causal_bottom_right"),
    "mask_2_1": ModelConfig(
        2, 2048, 24, 128, num_gqa_groups=1, attn_mask_type="causal_bottom_right"
    ),
    "mask_2_2": ModelConfig(
        2, 2048, 24, 128, max_seqlen_kv=4096, attn_mask_type="causal_bottom_right"
    ),
    "mask_3_0": ModelConfig(2, 2048, 16, 64, attn_mask_type="padding"),
    "mask_3_1": ModelConfig(2, 2048, 24, 128, num_gqa_groups=1, attn_mask_type="padding"),
    "mask_3_2": ModelConfig(2, 2048, 24, 128, max_seqlen_kv=4096, attn_mask_type="padding"),
    "mask_4_0": ModelConfig(2, 2048, 16, 64, attn_mask_type="padding_causal"),
    "mask_4_1": ModelConfig(2, 2048, 24, 128, num_gqa_groups=1, attn_mask_type="padding_causal"),
    "mask_4_2": ModelConfig(2, 2048, 24, 128, max_seqlen_kv=4096, attn_mask_type="padding_causal"),
    "mask_5_0": ModelConfig(2, 2048, 16, 64, attn_mask_type="padding_causal_bottom_right"),
    "mask_5_1": ModelConfig(
        2, 2048, 24, 128, num_gqa_groups=1, attn_mask_type="padding_causal_bottom_right"
    ),
    "mask_5_2": ModelConfig(
        2, 2048, 24, 128, max_seqlen_kv=4096, attn_mask_type="padding_causal_bottom_right"
    ),
    "mask_6_0": ModelConfig(2, 1, 16, 128, max_seqlen_kv=2048, attn_mask_type="causal"),
    "mask_6_1": ModelConfig(2, 1, 16, 256, max_seqlen_kv=2048, attn_mask_type="causal"),
    "mask_7_0": ModelConfig(
        2, 1, 16, 128, max_seqlen_kv=2048, attn_mask_type="causal_bottom_right"
    ),
    "mask_7_1": ModelConfig(
        2, 1, 16, 256, max_seqlen_kv=2048, attn_mask_type="causal_bottom_right"
    ),
    "mask_8_0": ModelConfig(2, 1, 24, 128, max_seqlen_kv=2048, attn_mask_type="padding"),
    "mask_8_1": ModelConfig(2, 1, 16, 256, max_seqlen_kv=2048, attn_mask_type="padding"),
    "mask_9_0": ModelConfig(2, 1, 24, 128, max_seqlen_kv=2048, attn_mask_type="padding_causal"),
    "mask_9_1": ModelConfig(2, 1, 16, 256, max_seqlen_kv=2048, attn_mask_type="padding_causal"),
    "mask_10_0": ModelConfig(
        2, 1, 24, 128, max_seqlen_kv=2048, attn_mask_type="padding_causal_bottom_right"
    ),
    "mask_10_1": ModelConfig(
        2, 1, 16, 256, max_seqlen_kv=2048, attn_mask_type="padding_causal_bottom_right"
    ),
}


@pytest.mark.skipif(get_cudnn_version() < (8, 9, 1), reason="cuDNN 8.9.1+ is required.")
@pytest.mark.parametrize("dtype", param_types_lean)
@pytest.mark.parametrize("model_configs", [model_configs_mask])
@pytest.mark.parametrize("model", model_configs_mask.keys())
def test_dpa_mask(dtype, model_configs, model):
    """Test DotProductAttention module with different mask types"""
    test_dot_product_attention(dtype, model_configs, model, False, True, None, False, False)


model_configs_bias = {
    #     test:             b,  h, hg,   d,   sq,  skv,   p,             mask,             bias
    "bias_1_0": ModelConfig(4, 128, 16, 64, attn_bias_type="post_scale_bias"),
    "bias_1_1": ModelConfig(2, 128, 16, 64, max_seqlen_kv=256, attn_bias_type="post_scale_bias"),
    "bias_1_2": ModelConfig(4, 2048, 24, 128, attn_bias_type="post_scale_bias"),
    "bias_1_3": ModelConfig(2, 2048, 24, 128, max_seqlen_kv=4096, attn_bias_type="post_scale_bias"),
    "bias_1_4": ModelConfig(4, 2048, 24, 128, attn_bias_type="alibi"),  # skipped
    "bias_1_5": ModelConfig(
        2, 2048, 24, 128, max_seqlen_kv=4096, attn_bias_type="alibi"
    ),  # skipped
    "bias_2_0": ModelConfig(
        4, 128, 16, 64, attn_mask_type="padding", attn_bias_type="post_scale_bias"
    ),  # skipped
    "bias_2_1": ModelConfig(
        2,
        128,
        16,
        64,
        max_seqlen_kv=256,
        attn_mask_type="padding",
        attn_bias_type="post_scale_bias",
    ),  # skipped
    "bias_2_2": ModelConfig(
        4, 2048, 24, 128, attn_mask_type="padding", attn_bias_type="post_scale_bias"
    ),  # skipped
    "bias_2_3": ModelConfig(
        2,
        2048,
        24,
        128,
        max_seqlen_kv=4096,
        attn_mask_type="padding",
        attn_bias_type="post_scale_bias",
    ),  # skipped
    "bias_2_4": ModelConfig(
        4, 2048, 24, 128, attn_mask_type="padding", attn_bias_type="alibi"
    ),  # skipped
    "bias_2_5": ModelConfig(
        2, 2048, 24, 128, max_seqlen_kv=4096, attn_mask_type="padding", attn_bias_type="alibi"
    ),  # skipped
    "bias_3_0": ModelConfig(
        4, 128, 16, 64, attn_mask_type="causal", attn_bias_type="post_scale_bias"
    ),
    "bias_3_1": ModelConfig(
        2, 128, 16, 64, max_seqlen_kv=256, attn_mask_type="causal", attn_bias_type="post_scale_bias"
    ),
    "bias_3_2": ModelConfig(
        4, 2048, 24, 128, attn_mask_type="causal", attn_bias_type="post_scale_bias"
    ),
    "bias_3_3": ModelConfig(
        2,
        2048,
        24,
        128,
        max_seqlen_kv=4096,
        attn_mask_type="causal",
        attn_bias_type="post_scale_bias",
    ),  # skipped
    "bias_3_4": ModelConfig(4, 2048, 24, 128, attn_mask_type="causal", attn_bias_type="alibi"),
    "bias_3_5": ModelConfig(
        2, 2048, 24, 128, max_seqlen_kv=4096, attn_mask_type="causal", attn_bias_type="alibi"
    ),  # skipped
    "bias_4_0": ModelConfig(
        4, 128, 16, 64, attn_mask_type="padding_causal", attn_bias_type="post_scale_bias"
    ),  # skipped
    "bias_4_1": ModelConfig(
        2,
        128,
        16,
        64,
        max_seqlen_kv=256,
        attn_mask_type="padding_causal",
        attn_bias_type="post_scale_bias",
    ),  # skipped
    "bias_4_2": ModelConfig(
        4, 2048, 24, 128, attn_mask_type="padding_causal", attn_bias_type="post_scale_bias"
    ),  # skipped
    "bias_4_3": ModelConfig(
        2,
        2048,
        24,
        128,
        max_seqlen_kv=4096,
        attn_mask_type="padding_causal",
        attn_bias_type="post_scale_bias",
    ),  # skipped
    "bias_4_4": ModelConfig(
        4, 2048, 24, 128, attn_mask_type="padding_causal", attn_bias_type="alibi"
    ),  # skipped
    "bias_4_5": ModelConfig(
        2,
        2048,
        24,
        128,
        max_seqlen_kv=4096,
        attn_mask_type="padding_causal",
        attn_bias_type="alibi",
    ),  # skipped
}


@pytest.mark.skipif(get_cudnn_version() < (8, 9, 1), reason="cuDNN 8.9.1+ is required.")
@pytest.mark.parametrize("dtype", param_types_lean)
@pytest.mark.parametrize("model_configs", [model_configs_bias])
@pytest.mark.parametrize("model", model_configs_bias.keys())
def test_dpa_bias(dtype, model_configs, model):
    """Test DotProductAttention module with different bias types"""
    test_dot_product_attention(dtype, model_configs, model, False, True, None, False, False)


model_configs_bias_shapes = {
    #     test:             b,  h, hg,   d,   sq,  skv,   p,
    "bias_1_0": ModelConfig(4, 128, 16, 64, attn_bias_type="post_scale_bias", bias_shape="11ss"),
    "bias_1_1": ModelConfig(2, 128, 16, 64, attn_bias_type="post_scale_bias", bias_shape="1hss"),
    "bias_1_2": ModelConfig(4, 2048, 24, 128, attn_bias_type="post_scale_bias", bias_shape="b1ss"),
    "bias_1_3": ModelConfig(2, 2048, 24, 128, attn_bias_type="post_scale_bias", bias_shape="bhss"),
    "bias_1_4": ModelConfig(
        4,
        2048,
        24,
        128,
        attn_mask_type="causal",
        attn_bias_type="alibi",
        bias_shape="1hss",
        alibi_type="custom",
    ),
    "bias_1_5": ModelConfig(
        2,
        2048,
        24,
        128,
        attn_mask_type="causal",
        attn_bias_type="alibi",
        bias_shape="bhss",
        alibi_type="custom",
    ),
}


@pytest.mark.skipif(get_cudnn_version() < (8, 9, 1), reason="cuDNN 8.9.1+ is required.")
@pytest.mark.parametrize("dtype", param_types_lean)
@pytest.mark.parametrize("model_configs", [model_configs_bias_shapes])
@pytest.mark.parametrize("model", model_configs_bias_shapes.keys())
def test_dpa_bias_shapes(dtype, model_configs, model):
    """Test DotProductAttention module with different bias types and shapes"""
    test_dot_product_attention(dtype, model_configs, model, False, True, None, False, False)


model_configs_swa = {
    #    test:             b,  h, hg,   d,   sq,  skv,   p,             mask,             bias
    "swa_1_1": ModelConfig(2, 2048, 16, 64),
    "swa_1_2": ModelConfig(2, 2048, 24, 128, num_gqa_groups=4),
    "swa_1_3": ModelConfig(2, 2048, 24, 128, max_seqlen_kv=4096),
    "swa_2_1": ModelConfig(2, 2048, 16, 64, attn_mask_type="causal"),
    "swa_2_2": ModelConfig(2, 2048, 24, 128, num_gqa_groups=4, attn_mask_type="causal"),
    "swa_2_3": ModelConfig(2, 2048, 24, 128, max_seqlen_kv=4096, attn_mask_type="causal"),
    "swa_3_1": ModelConfig(2, 2048, 16, 64, attn_mask_type="causal_bottom_right"),
    "swa_3_2": ModelConfig(
        2, 2048, 24, 128, num_gqa_groups=4, attn_mask_type="causal_bottom_right"
    ),
    "swa_3_3": ModelConfig(
        2, 2048, 24, 128, max_seqlen_kv=4096, attn_mask_type="causal_bottom_right"
    ),
    "swa_4_1": ModelConfig(2, 2048, 16, 64, attn_mask_type="padding"),
    "swa_4_2": ModelConfig(2, 2048, 24, 128, num_gqa_groups=4, attn_mask_type="padding"),
    "swa_4_3": ModelConfig(2, 2048, 24, 128, max_seqlen_kv=4096, attn_mask_type="padding"),
    "swa_5_1": ModelConfig(2, 2048, 16, 64, attn_mask_type="padding_causal"),
    "swa_5_2": ModelConfig(2, 2048, 24, 128, num_gqa_groups=4, attn_mask_type="padding_causal"),
    "swa_5_3": ModelConfig(2, 2048, 24, 128, max_seqlen_kv=4096, attn_mask_type="padding_causal"),
    "swa_6_1": ModelConfig(2, 2048, 16, 64, attn_mask_type="padding_causal_bottom_right"),
    "swa_6_2": ModelConfig(
        2, 2048, 24, 128, num_gqa_groups=4, attn_mask_type="padding_causal_bottom_right"
    ),
    "swa_6_3": ModelConfig(
        2, 2048, 24, 128, max_seqlen_kv=4096, attn_mask_type="padding_causal_bottom_right"
    ),
}


@pytest.mark.skipif(not FlashAttentionUtils.v2_3_plus, reason="Flash-attn 2.3+ is required.")
@pytest.mark.parametrize("dtype", param_types_lean)
@pytest.mark.parametrize("model_configs", [model_configs_swa])
@pytest.mark.parametrize("model", model_configs_swa.keys())
def test_dpa_sliding_window(dtype, model_configs, model):
    """Test DotProductAttention module with sliding window attention"""
    test_dot_product_attention(dtype, model_configs, model, False, True, None, True, False)


model_configs_alibi_slopes = {
    #     test:             b,  h, hg,   d,   sq,  skv,   p,      mask,    bias, alibi_type
    "alibi_1_0": ModelConfig(
        2, 128, 16, 64, attn_mask_type="causal", attn_bias_type="alibi", alibi_type="vanilla"
    ),
    "alibi_1_1": ModelConfig(
        1,
        128,
        16,
        64,
        max_seqlen_kv=256,
        attn_mask_type="causal",
        attn_bias_type="alibi",
        alibi_type="vanilla",
    ),
    "alibi_2_0": ModelConfig(
        2, 1024, 24, 128, attn_mask_type="causal", attn_bias_type="alibi", alibi_type="custom"
    ),
    "alibi_2_1": ModelConfig(
        1,
        1024,
        24,
        128,
        max_seqlen_kv=2048,
        attn_mask_type="causal",
        attn_bias_type="alibi",
        alibi_type="custom",
    ),
}


@pytest.mark.skipif(not FlashAttentionUtils.v2_3_plus, reason="Flash-attn 2.3+ is required.")
@pytest.mark.parametrize("dtype", param_types_lean)
@pytest.mark.parametrize("model_configs", [model_configs_alibi_slopes])
@pytest.mark.parametrize("model", model_configs_alibi_slopes.keys())
def test_dpa_alibi_slopes(dtype, model_configs, model):
    """Test DotProductAttention module with ALiBi slopes"""
    test_dot_product_attention(dtype, model_configs, model, False, True, None, False, False)


qkv_layouts = [
    "sb3hd",
    "sbh3d",
    "sbhd_sb2hd",
    "sbhd_sbh2d",
    "sbhd_sbhd_sbhd",
    "bs3hd",
    "bsh3d",
    "bshd_bs2hd",
    "bshd_bsh2d",
    "bshd_bshd_bshd",
]


model_configs_layout = {
    #       test:             b,  h, hg,   d,   sq,  skv,   p,             mask,             bias
    "layout_0_0": ModelConfig(2, 128, 16, 64),
    "layout_0_1": ModelConfig(
        2, 128, 16, 64, attn_mask_type="causal", attn_bias_type="post_scale_bias"
    ),
    "layout_0_2": ModelConfig(1, 128, 16, 64, max_seqlen_kv=256, attn_mask_type="padding"),
    "layout_0_3": ModelConfig(
        1,
        128,
        16,
        64,
        max_seqlen_kv=256,
        attn_mask_type="padding_causal",
        attn_bias_type="post_scale_bias",
    ),
    "layout_1_0": ModelConfig(2, 2048, 24, 128),
    "layout_1_1": ModelConfig(
        2, 2048, 24, 128, attn_mask_type="causal", attn_bias_type="post_scale_bias"
    ),
    "layout_1_2": ModelConfig(1, 2048, 24, 128, max_seqlen_kv=4096, attn_mask_type="padding"),
    "layout_1_3": ModelConfig(
        1,
        2048,
        24,
        128,
        max_seqlen_kv=4096,
        attn_mask_type="padding_causal",
        attn_bias_type="post_scale_bias",
    ),
    "layout_2_0": ModelConfig(2, 1, 16, 256, max_seqlen_kv=2048),
    "layout_2_1": ModelConfig(
        2, 2048, 24, 256, attn_mask_type="causal", attn_bias_type="post_scale_bias"
    ),
}


@pytest.mark.skipif(get_cudnn_version() < (8, 9, 5), reason="cuDNN 8.9.5+ is required.")
@pytest.mark.parametrize("dtype", param_types_lean)
@pytest.mark.parametrize("model_configs", [model_configs_layout])
@pytest.mark.parametrize("model", model_configs_layout.keys())
@pytest.mark.parametrize("qkv_layout", qkv_layouts)
def test_dpa_qkv_layout(dtype, model_configs, model, qkv_layout):
    """Test DotProductAttention module with different QKV layouts"""
    test_dot_product_attention(dtype, model_configs, model, False, True, qkv_layout, False, False)


qkv_layouts_thd = ["t3hd", "th3d", "thd_t2hd", "thd_th2d", "thd_thd_thd"]
model_configs_layout_thd = {
    #       test:             b,  h, hg,   d,   sq,  skv,   p,             mask,             bias
    "layout_0_0": ModelConfig(2, 2048, 16, 64, attn_mask_type="padding"),
    "layout_0_1": ModelConfig(2, 2048, 24, 128, num_gqa_groups=1, attn_mask_type="padding"),
    "layout_0_2": ModelConfig(2, 2048, 24, 128, max_seqlen_kv=4096, attn_mask_type="padding"),
    "layout_1_0": ModelConfig(2, 2048, 16, 64, attn_mask_type="padding_causal"),
    "layout_1_1": ModelConfig(2, 2048, 24, 128, num_gqa_groups=1, attn_mask_type="padding_causal"),
    "layout_1_2": ModelConfig(
        2, 2048, 24, 128, max_seqlen_kv=4096, attn_mask_type="padding_causal"
    ),
    "layout_2_0": ModelConfig(2, 2048, 16, 64, attn_mask_type="padding_causal_bottom_right"),
    "layout_2_1": ModelConfig(
        2, 2048, 24, 128, num_gqa_groups=1, attn_mask_type="padding_causal_bottom_right"
    ),
    "layout_2_2": ModelConfig(
        2, 2048, 24, 128, max_seqlen_kv=4096, attn_mask_type="padding_causal_bottom_right"
    ),
    "layout_3_0": ModelConfig(2, 2048, 16, 64, attn_mask_type="padding", window_size=(4, 4)),
    "layout_3_1": ModelConfig(
        2, 2048, 24, 128, num_gqa_groups=1, attn_mask_type="padding", window_size=(4, 4)
    ),
    "layout_3_2": ModelConfig(
        2, 2048, 24, 128, max_seqlen_kv=4096, attn_mask_type="padding", window_size=(4, 4)
    ),
    "layout_4_0": ModelConfig(2, 2048, 16, 64, attn_mask_type="padding_causal", window_size=(4, 0)),
    "layout_4_1": ModelConfig(
        2, 2048, 24, 128, num_gqa_groups=1, attn_mask_type="padding_causal", window_size=(4, 0)
    ),
    "layout_4_2": ModelConfig(
        2, 2048, 24, 128, max_seqlen_kv=4096, attn_mask_type="padding_causal", window_size=(4, 0)
    ),
    "layout_5_0": ModelConfig(
        2, 2048, 16, 64, attn_mask_type="padding_causal_bottom_right", window_size=(4, 0)
    ),
    "layout_5_1": ModelConfig(
        2,
        2048,
        24,
        128,
        num_gqa_groups=1,
        attn_mask_type="padding_causal_bottom_right",
        window_size=(4, 0),
    ),
    "layout_5_2": ModelConfig(
        2,
        2048,
        24,
        128,
        max_seqlen_kv=4096,
        attn_mask_type="padding_causal_bottom_right",
        window_size=(4, 0),
    ),
}


@pytest.mark.skipif(get_cudnn_version() < (9, 0, 0), reason="cuDNN 9.0.0+ is required.")
@pytest.mark.skipif(
    get_device_compute_capability() < (9, 0), reason="THD is only supported on Hopper+."
)
@pytest.mark.parametrize("dtype", param_types_lean)
@pytest.mark.parametrize("model_configs", [model_configs_layout_thd])
@pytest.mark.parametrize("model", model_configs_layout_thd.keys())
@pytest.mark.parametrize("qkv_layout", qkv_layouts_thd)
def test_dpa_qkv_layout_thd(dtype, model_configs, model, qkv_layout):
    """Test DotProductAttention module with different QKV layouts"""
    config = model_configs[model]
    if config.num_heads != config.num_gqa_groups and "3" in qkv_layout:
        pytest.skip("qkv_layout not applicable for MQA/GQA")
    logging.info("[test_dpa_qkv_layout_thd]: pad_between_seqs = True")
    pad_between_seqs = True
    test_dot_product_attention(
        dtype, model_configs, model, False, True, qkv_layout, False, pad_between_seqs
    )
    if get_cudnn_version() >= (9, 3, 0):
        logging.info("[test_dpa_qkv_layout_thd]: pad_between_seqs = False")
        # cuDNN 9.3.0+ is required to run pad_between_seqs = False/True in the same run
        pad_between_seqs = False
        test_dot_product_attention(
            dtype, model_configs, model, False, True, qkv_layout, False, pad_between_seqs
        )


def _run_dot_product_attention(
    dtype: torch.dtype,
    config: ModelConfig,
    backend: str,
    ckpt_attn: bool,
    qkv_layout: str,
    workspace_opt: bool,
    pad_between_seqs: bool,
    is_training: bool,
) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Run DotProductAttention module with one forward pass and one backward pass"""

    # Set RNG and environment varables
    reset_rng_states()
    os.environ["NVTE_FLASH_ATTN"] = "0"
    os.environ["NVTE_FUSED_ATTN"] = "0"
    if backend == "FlashAttention":
        os.environ["NVTE_FLASH_ATTN"] = "1"
    if backend == "FusedAttention":
        os.environ["NVTE_FUSED_ATTN"] = "1"
        os.environ["NVTE_FUSED_ATTN_FORCE_WORKSPACE_OPT"] = "1" if workspace_opt else "0"
    _attention_backends["backend_selection_requires_update"] = True

    # Create seqlens
    qkv_format = "".join([i for i in qkv_layout.split("_")[0] if i.isalpha()])
    if "padding" in config.attn_mask_type or qkv_format == "thd":
        if config.attn_type == "self":
            seqlens_q = torch.randint(
                1, config.max_seqlen_q, [config.batch_size], dtype=torch.int32, device="cuda"
            )
            seqlens_kv = seqlens_q
        if config.attn_type == "cross":
            if config.max_seqlen_q > 1:
                seqlens_q = torch.randint(
                    1, config.max_seqlen_q, [config.batch_size], dtype=torch.int32, device="cuda"
                )
            else:
                seqlens_q = torch.ones([config.batch_size], dtype=torch.int32, device="cuda")
            seqlens_kv = torch.randint(
                1, config.max_seqlen_kv, [config.batch_size], dtype=torch.int32, device="cuda"
            )
    else:
        seqlens_q = torch.full(
            [config.batch_size], config.max_seqlen_q, dtype=torch.int32, device="cuda"
        )
        seqlens_kv = torch.full(
            [config.batch_size], config.max_seqlen_kv, dtype=torch.int32, device="cuda"
        )
    cu_seqlens_q = torch.zeros(config.batch_size + 1, dtype=torch.int32, device="cuda")
    cu_seqlens_kv = torch.zeros(config.batch_size + 1, dtype=torch.int32, device="cuda")
    cu_seqlens_q[1:] = torch.cumsum(seqlens_q, dim=0)
    cu_seqlens_kv[1:] = torch.cumsum(seqlens_kv, dim=0)

    seqlens_q_after_pad = seqlens_q.clone()
    seqlens_kv_after_pad = seqlens_kv.clone()
    cu_seqlens_q_after_pad = cu_seqlens_q.clone()
    cu_seqlens_kv_after_pad = cu_seqlens_kv.clone()
    pad_len = [0] * config.batch_size
    if pad_between_seqs:
        max_pad_len = 3
        pad_len = torch.randint(0, max_pad_len + 1, [config.batch_size], device="cuda")  # 3
        seqlens_q_after_pad = seqlens_q + pad_len
        seqlens_kv_after_pad = seqlens_kv + pad_len
        cu_seqlens_q_after_pad[1:] = torch.cumsum(seqlens_q_after_pad, dim=0)
        cu_seqlens_kv_after_pad[1:] = torch.cumsum(seqlens_kv_after_pad, dim=0)

    # Create attention mask if padding
    attention_mask = None
    if "padding" in config.attn_mask_type:
        if config.attn_type == "self":
            attention_mask_q = torch.Tensor([]).to(dtype=torch.bool)
            for i in range(config.batch_size):
                attention_mask_q = torch.cat(
                    [
                        attention_mask_q,
                        torch.Tensor(
                            [False] * seqlens_q[i] + [True] * (config.max_seqlen_q - seqlens_q[i])
                        )
                        .to(dtype=torch.bool)
                        .unsqueeze(0)
                        .unsqueeze(0)
                        .unsqueeze(0),
                    ],
                    dim=0,
                )
            attention_mask = attention_mask_q.to(device="cuda")
        if config.attn_type == "cross":
            attention_mask_q = torch.Tensor([]).to(dtype=torch.bool)
            attention_mask_kv = torch.Tensor([]).to(dtype=torch.bool)
            for i in range(config.batch_size):
                attention_mask_q = torch.cat(
                    [
                        attention_mask_q,
                        torch.Tensor(
                            [False] * seqlens_q[i] + [True] * (config.max_seqlen_q - seqlens_q[i])
                        )
                        .to(dtype=torch.bool)
                        .unsqueeze(0)
                        .unsqueeze(0)
                        .unsqueeze(0),
                    ],
                    dim=0,
                )
                attention_mask_kv = torch.cat(
                    [
                        attention_mask_kv,
                        torch.Tensor(
                            [False] * seqlens_kv[i]
                            + [True] * (config.max_seqlen_kv - seqlens_kv[i])
                        )
                        .to(dtype=torch.bool)
                        .unsqueeze(0)
                        .unsqueeze(0)
                        .unsqueeze(0),
                    ],
                    dim=0,
                )
            attention_mask = (
                attention_mask_q.to(device="cuda"),
                attention_mask_kv.to(device="cuda"),
            )

    alibi_slopes = None
    if config.attn_bias_type == "alibi" and config.alibi_type == "custom":
        if config.bias_shape == "1hss":
            alibi_slopes = (
                torch.randn(config.num_heads).abs().to(dtype=torch.float32, device="cuda")
            )
        if config.bias_shape == "bhss":
            alibi_slopes = (
                torch.randn(config.batch_size, config.num_heads)
                .abs()
                .to(dtype=torch.float32, device="cuda")
            )

    # Create input tensors
    dim_to_num = {
        "b": config.batch_size,
        "sq": config.max_seqlen_q,
        "skv": config.max_seqlen_kv,
        "h": config.num_heads,
        "hg": config.num_gqa_groups,
        "dqk": config.head_dim_qk,
        "dv": config.head_dim_v,
        "t": cu_seqlens_q_after_pad[-1],
        "tg": cu_seqlens_kv_after_pad[-1],
        "3": 3,
        "2": 2,
        "1": 1,
    }
    inp = []
    inp_orig = []
    for i, layout in enumerate(qkv_layout.split("_")):
        layout = "_".join(layout)
        if i == 0:
            layout = layout.replace("s", "sq")
        else:
            layout = layout.replace("s", "skv")
            layout = layout.replace("h", "hg")
            layout = layout.replace("t", "tg")
        if i == 2:
            layout = layout.replace("d", "dv")
        else:
            layout = layout.replace("d", "dqk")
        tensor_shape = [dim_to_num[j] for j in layout.split("_")]
        tensor = 0.1 * torch.randn(tensor_shape, dtype=dtype, device="cuda")
        tensor_orig = tensor
        if qkv_format == "thd" and pad_between_seqs:
            tensor_orig = torch.Tensor([]).to(device="cuda", dtype=dtype)
            if layout in ["t_h_dqk", "t_3_h_dqk", "t_h_3_dqk"]:
                for i in range(1, config.batch_size + 1):
                    valid_range = (
                        cu_seqlens_q_after_pad[i - 1],
                        cu_seqlens_q_after_pad[i] - pad_len[i - 1],
                    )
                    pad_range = (
                        cu_seqlens_q_after_pad[i] - pad_len[i - 1],
                        cu_seqlens_q_after_pad[i],
                    )
                    tensor[pad_range[0] : pad_range[1]] = 0.0
                    tensor_orig = torch.cat(
                        [tensor_orig, tensor[valid_range[0] : valid_range[1]]], dim=0
                    )
            if layout in ["tg_hg_dqk", "tg_2_hg_dqk", "tg_hg_2_dqk", "tg_hg_dv"]:
                for i in range(1, config.batch_size + 1):
                    valid_range = (
                        cu_seqlens_kv_after_pad[i - 1],
                        cu_seqlens_kv_after_pad[i] - pad_len[i - 1],
                    )
                    pad_range = (
                        cu_seqlens_kv_after_pad[i] - pad_len[i - 1],
                        cu_seqlens_kv_after_pad[i],
                    )
                    tensor[pad_range[0] : pad_range[1]] = 0.0
                    tensor_orig = torch.cat(
                        [tensor_orig, tensor[valid_range[0] : valid_range[1]]], dim=0
                    )
        tensor_count = 1
        split_dim = 0
        for dim, l in enumerate(layout.split("_")):
            if l.isdigit():
                tensor_count = int(l)
                split_dim = dim
                break
        tensors = torch.split(tensor, 1, dim=split_dim) if split_dim != 0 else [tensor]
        tensors_orig = (
            torch.split(tensor_orig, 1, dim=split_dim) if split_dim != 0 else [tensor_orig]
        )
        for j in range(tensor_count):
            if split_dim != 0:
                inp.append(tensors[j].squeeze(split_dim))
                inp_orig.append(tensors_orig[j].squeeze(split_dim))
            else:
                inp.append(tensors[j])
                inp_orig.append(tensors_orig[j])
    for i in range(3):
        inp[i].requires_grad = True
        inp_orig[i].requires_grad = True

    # Create output gradient
    qkv_format_kv = "_".join(qkv_format)
    qkv_format_kv = qkv_format_kv.replace("s", "sq")
    qkv_format_kv = qkv_format_kv.replace("d", "dv")
    out_grad_shape = [dim_to_num[i] for i in qkv_format_kv.split("_")]
    out_grad_shape_new = [*out_grad_shape[:-2], out_grad_shape[-2] * out_grad_shape[-1]]
    out_grad = 0.001 * torch.randint(0, 200, out_grad_shape_new, dtype=dtype, device="cuda")
    out_grad_orig = out_grad
    if qkv_format == "thd" and pad_between_seqs:
        out_grad_orig = torch.Tensor([]).to(device="cuda", dtype=dtype)
        if qkv_format_kv == "t_h_dv":
            for i in range(1, config.batch_size + 1):
                valid_range = (
                    cu_seqlens_q_after_pad[i - 1],
                    cu_seqlens_q_after_pad[i] - pad_len[i - 1],
                )
                pad_range = (cu_seqlens_q_after_pad[i] - pad_len[i - 1], cu_seqlens_q_after_pad[i])
                out_grad[pad_range[0] : pad_range[1]] = 0.0
                out_grad_orig = torch.cat(
                    [out_grad_orig, out_grad[valid_range[0] : valid_range[1]]], dim=0
                )

    # Create bias
    if config.attn_bias_type in ["no_bias", "alibi"]:
        bias = None
    if config.attn_bias_type == "post_scale_bias":
        shape = "_".join(config.bias_shape)
        shape = shape.replace("_s_s", "_sq_skv")
        tensor_shape = [dim_to_num[j] for j in shape.split("_")]
        bias = torch.randn(tensor_shape, dtype=dtype, device="cuda")
        if config.bias_shape != "1hss":
            bias.requires_grad = False

    # Create RNG
    _DUMMY_CUDA_RNG_STATE_TRACKER = CudaRNGStatesTracker()
    _DUMMY_CUDA_RNG_STATE_TRACKER.add("model-parallel-rng", seed)

    def get_dummy_cuda_rng_tracker() -> CudaRNGStatesTracker:
        """Get cuda rng tracker."""
        return _DUMMY_CUDA_RNG_STATE_TRACKER

    # Set up model
    block = DotProductAttention(
        config.num_heads,
        (config.head_dim_qk, config.head_dim_v),
        num_gqa_groups=config.num_gqa_groups,
        attention_dropout=config.dropout_p,
        qkv_format=qkv_format,
        attn_mask_type=config.attn_mask_type,
        sequence_parallel=False,
        tp_size=1,
        get_rng_state_tracker=get_dummy_cuda_rng_tracker,
        tp_group=None,
        layer_number=1,
        attention_type=config.attn_type,
    ).to(dtype=dtype, device="cuda")
    if not is_training:
        block = block.eval()

    # Run a forward and backward pass
    if backend in ["FlashAttention", "UnfusedDotProductAttention"]:
        q = inp_orig[0]
        k = inp_orig[1]
        v = inp_orig[2]
        d_out = out_grad_orig
    if backend == "FusedAttention":
        q = inp[0]
        k = inp[1]
        v = inp[2]
        d_out = out_grad
    out = block(
        q,
        k,
        v,
        window_size=config.window_size,
        attention_mask=attention_mask,
        qkv_format=qkv_format,
        max_seqlen_q=config.max_seqlen_q,
        max_seqlen_kv=config.max_seqlen_kv,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_kv=cu_seqlens_kv,
        cu_seqlens_q_padded=cu_seqlens_q_after_pad if backend == "FusedAttention" else None,
        cu_seqlens_kv_padded=cu_seqlens_kv_after_pad if backend == "FusedAttention" else None,
        attn_mask_type=config.attn_mask_type,
        checkpoint_core_attention=ckpt_attn,
        core_attention_bias_type=config.attn_bias_type,
        core_attention_bias=bias,
        alibi_slopes=alibi_slopes,
        fast_zero_fill=True,
    )
    if is_training:
        out.backward(d_out)

    if backend in ["FlashAttention", "UnfusedDotProductAttention"]:
        if is_training:
            return out, (q.grad, k.grad, v.grad)
        else:
            return out, (None, None, None)
    if backend == "FusedAttention":
        if qkv_format == "thd" and pad_between_seqs:
            out_orig = torch.Tensor([]).to(device="cuda", dtype=dtype)
            if is_training:
                q_grad_orig = torch.Tensor([]).to(device="cuda", dtype=dtype)
                k_grad_orig = torch.Tensor([]).to(device="cuda", dtype=dtype)
                v_grad_orig = torch.Tensor([]).to(device="cuda", dtype=dtype)
            for i in range(1, config.batch_size + 1):
                valid_range_q = (
                    cu_seqlens_q_after_pad[i - 1],
                    cu_seqlens_q_after_pad[i] - pad_len[i - 1],
                )
                valid_range_kv = (
                    cu_seqlens_kv_after_pad[i - 1],
                    cu_seqlens_kv_after_pad[i] - pad_len[i - 1],
                )
                out_orig = torch.cat([out_orig, out[valid_range_q[0] : valid_range_q[1]]], dim=0)
                if is_training:
                    q_grad_orig = torch.cat(
                        [q_grad_orig, q.grad[valid_range_q[0] : valid_range_q[1]]], dim=0
                    )
                    k_grad_orig = torch.cat(
                        [k_grad_orig, k.grad[valid_range_kv[0] : valid_range_kv[1]]], dim=0
                    )
                    v_grad_orig = torch.cat(
                        [v_grad_orig, v.grad[valid_range_kv[0] : valid_range_kv[1]]], dim=0
                    )
            if is_training:
                return out_orig, (q_grad_orig, k_grad_orig, v_grad_orig)
            else:
                return out_orig, (None, None, None)
        else:
            if is_training:
                return out, (q.grad, k.grad, v.grad)
            else:
                return out, (None, None, None)


model_configs_te_layer = {
    #   test:             b,  h, hg,   d,   sq,  skv,   p,      mask,             bias
    "te_1_0": ModelConfig(2, 128, 16, 64, attn_bias_type="post_scale_bias"),
    "te_1_1": ModelConfig(
        4, 128, 16, 64, attn_mask_type="causal", attn_bias_type="post_scale_bias"
    ),
    "te_1_2": ModelConfig(
        2, 128, 16, 64, attn_mask_type="padding", attn_bias_type="post_scale_bias"
    ),
    "te_1_3": ModelConfig(2, 128, 16, 64, max_seqlen_kv=256, attn_mask_type="padding"),
    "te_2_0": ModelConfig(1, 2048, 16, 64, attn_mask_type="causal"),
    "te_2_1": ModelConfig(2, 2048, 16, 64),
    "te_2_2": ModelConfig(1, 2048, 16, 64, attn_mask_type="padding"),
    "te_2_3": ModelConfig(
        1, 2048, 16, 64, max_seqlen_kv=4096, attn_mask_type="padding_causal_bottom_right"
    ),
    "te_3_0": ModelConfig(4, 128, 16, 64, attn_mask_type="causal", attn_bias_type="alibi"),
    "te_3_1": ModelConfig(4, 2048, 16, 64, attn_mask_type="causal", attn_bias_type="alibi"),
}


@pytest.mark.skipif(get_cudnn_version() < (8, 9, 1), reason="cuDNN 8.9.1+ is required.")
@pytest.mark.parametrize("dtype", param_types)
@pytest.mark.parametrize("model_configs", [model_configs_te_layer])
@pytest.mark.parametrize("model", model_configs_te_layer.keys())
@pytest.mark.parametrize("ckpt_attn", [False])
@pytest.mark.parametrize("qkv_format", ["sbhd", "bshd", "thd"])
@pytest.mark.parametrize("fused_qkv_params", [False])
@pytest.mark.parametrize("RoPE", [False])
def test_transformer_layer(
    dtype, model_configs, model, ckpt_attn, qkv_format, fused_qkv_params, RoPE
):
    """Test TransformerLayer module"""

    # Get configs
    config = model_configs[model]
    tols = dict(atol=5e-2, rtol=5e-2)
    workspace_opt = True

    # Test backend availability
    is_training = True
    available_backends, _, fused_attn_backends = get_available_attention_backends(
        config,
        qkv_dtype=dtype,
        qkv_layout=(
            qkv_format.replace("hd", "h3d") if fused_qkv_params else qkv_format.replace("hd", "3hd")
        ),
        is_training=is_training,
    )
    flash_attn_supported, fused_attn_supported, unfused_attn_supported = available_backends
    if not fused_attn_supported:
        is_training = False
        available_backends, _, fused_attn_backends = get_available_attention_backends(
            config,
            qkv_dtype=dtype,
            qkv_layout=(
                qkv_format.replace("hd", "h3d")
                if fused_qkv_params
                else qkv_format.replace("hd", "3hd")
            ),
            is_training=is_training,
        )
        flash_attn_supported, fused_attn_supported, unfused_attn_supported = available_backends

    # Skip if only unfused backend is supported
    if (len(fused_attn_backends) + flash_attn_supported + unfused_attn_supported) < 2:
        pytest.skip("Less than two backends to compare.")
    # Skip if qkv_format = thd and "padding" not in attn_mask_type
    if qkv_format == "thd" and "padding" not in config.attn_mask_type:
        pytest.skip("THD requires padding mask.")

    # UnfusedDotProductAttention backend
    if unfused_attn_supported:
        unfused_attn_fwd, unfused_attn_bwd = _run_transformer_layer(
            dtype,
            config,
            "UnfusedDotProductAttention",
            ckpt_attn,
            qkv_format,
            workspace_opt,
            fused_qkv_params,
            RoPE,
            is_training,
        )

    # FusedAttention backend
    if fused_attn_supported:
        fused_attn_fwd, fused_attn_bwd = _run_transformer_layer(
            dtype,
            config,
            "FusedAttention",
            ckpt_attn,
            qkv_format,
            workspace_opt,
            fused_qkv_params,
            RoPE,
            is_training,
        )

    # FlashAttention backend
    if flash_attn_supported:
        flash_attn_fwd, flash_attn_bwd = _run_transformer_layer(
            dtype,
            config,
            "FlashAttention",
            ckpt_attn,
            qkv_format,
            workspace_opt,
            fused_qkv_params,
            RoPE,
            is_training,
        )

    logging.info(f"[test_transformer_layer]: is_training = {is_training}")
    if unfused_attn_supported and fused_attn_supported:
        logging.info("[test_transformer_layer]: unfused attn vs fused attn")
        torch.testing.assert_close(fused_attn_fwd, unfused_attn_fwd, **tols)
        torch.testing.assert_close(fused_attn_bwd, unfused_attn_bwd, **tols)
    if unfused_attn_supported and flash_attn_supported:
        logging.info("[test_transformer_layer]: unfused attn vs flash attn")
        torch.testing.assert_close(flash_attn_fwd, unfused_attn_fwd, **tols)
        torch.testing.assert_close(flash_attn_bwd, unfused_attn_bwd, **tols)
    if fused_attn_supported and flash_attn_supported:
        logging.info("[test_transformer_layer]: fused attn vs flash attn")
        torch.testing.assert_close(fused_attn_fwd, flash_attn_fwd, **tols)
        torch.testing.assert_close(fused_attn_bwd, flash_attn_bwd, **tols)


@pytest.mark.skipif(get_cudnn_version() < (8, 9, 1), reason="cuDNN 8.9.1+ is required.")
@pytest.mark.parametrize("dtype", param_types_lean)
@pytest.mark.parametrize("model_configs", [model_configs_te_layer])
@pytest.mark.parametrize("model", ["te_1_2", "te_2_0"])
@pytest.mark.parametrize("qkv_format", ["bshd", "sbhd"])
def test_te_layer_misc(dtype, model_configs, model, qkv_format):
    """Test TransformerLayer module with miscellaneous settings"""
    ckpt_attn = True
    fused_qkv_params = True
    RoPE = True
    test_transformer_layer(
        dtype, model_configs, model, ckpt_attn, qkv_format, fused_qkv_params, RoPE
    )


@pytest.mark.skipif(get_cudnn_version() < (8, 9, 1), reason="cuDNN 8.9.1+ is required.")
@pytest.mark.parametrize("dtype", param_types_lean)
@pytest.mark.parametrize("model_configs", [model_configs_te_layer])
@pytest.mark.parametrize("model", ["te_2_0", "te_2_1", "te_2_2"])
def test_te_layer_mqa_gqa(dtype, model_configs, model):
    """Test TransformerLayer module with MQA/GQA"""

    def find_factors(x):
        f = []
        for i in range(2, x + 1):
            if x % i == 0:
                f.append(i)
        return f

    ckpt_attn = True
    qkv_format = "bshd"
    fused_qkv_params = True
    RoPE = True
    config = model_configs[model]
    num_querys_per_gqa_group = find_factors(config.num_heads)

    for num_q_per_gqa_group in num_querys_per_gqa_group:
        config.num_gqa_groups = config.num_heads // num_q_per_gqa_group
        test_transformer_layer(
            dtype, model_configs, model, ckpt_attn, qkv_format, fused_qkv_params, RoPE
        )


def _run_transformer_layer(
    dtype: torch.dtype,
    config: ModelConfig,
    backend: str,
    ckpt_attn: bool,
    qkv_format: str,
    workspace_opt: bool,
    fused_qkv_params: bool,
    RoPE: bool,
    is_training: bool,
) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Run TransformerLayer module with one forward pass and one backward pass"""

    # Set RNG and environment variables
    reset_rng_states()
    os.environ["NVTE_FLASH_ATTN"] = "0"
    os.environ["NVTE_FUSED_ATTN"] = "0"
    if backend == "FlashAttention":
        os.environ["NVTE_FLASH_ATTN"] = "1"
    if backend == "FusedAttention":
        os.environ["NVTE_FUSED_ATTN"] = "1"
    _attention_backends["backend_selection_requires_update"] = True

    # Create input tensor
    if qkv_format == "sbhd":
        inp = torch.randn(
            config.max_seqlen_q,
            config.batch_size,
            config.hidden_size,
            dtype=dtype,
            device="cuda",
            requires_grad=True,
        )
        inp_enc = torch.randn(
            config.max_seqlen_kv,
            config.batch_size,
            config.hidden_size,
            dtype=dtype,
            device="cuda",
            requires_grad=True,
        )
    if qkv_format == "bshd":
        inp = torch.randn(
            config.batch_size,
            config.max_seqlen_q,
            config.hidden_size,
            dtype=dtype,
            device="cuda",
            requires_grad=True,
        )
        inp_enc = torch.randn(
            config.batch_size,
            config.max_seqlen_kv,
            config.hidden_size,
            dtype=dtype,
            device="cuda",
            requires_grad=True,
        )

    # Create seqlens
    if "padding" in config.attn_mask_type or qkv_format == "thd":
        if config.attn_type == "self":
            seqlens_q = torch.randint(
                1, config.max_seqlen_q, [config.batch_size], dtype=torch.int32, device="cuda"
            )
            seqlens_kv = seqlens_q
        if config.attn_type == "cross":
            if config.max_seqlen_q > 1:
                seqlens_q = torch.randint(
                    1, config.max_seqlen_q, [config.batch_size], dtype=torch.int32, device="cuda"
                )
            else:
                seqlens_q = torch.ones([config.batch_size], dtype=torch.int32, device="cuda")
            seqlens_kv = torch.randint(
                1, config.max_seqlen_kv, [config.batch_size], dtype=torch.int32, device="cuda"
            )
    else:
        seqlens_q = torch.full(
            [config.batch_size], config.max_seqlen_q, dtype=torch.int32, device="cuda"
        )
        seqlens_kv = torch.full(
            [config.batch_size], config.max_seqlen_kv, dtype=torch.int32, device="cuda"
        )
    cu_seqlens_q = torch.zeros(config.batch_size + 1, dtype=torch.int32, device="cuda")
    cu_seqlens_kv = torch.zeros(config.batch_size + 1, dtype=torch.int32, device="cuda")
    cu_seqlens_q[1:] = torch.cumsum(seqlens_q, dim=0)
    cu_seqlens_kv[1:] = torch.cumsum(seqlens_kv, dim=0)
    if qkv_format == "thd":
        inp = torch.randn(
            cu_seqlens_q[-1],
            config.hidden_size,
            dtype=dtype,
            device="cuda",
            requires_grad=True,
        )
        inp_enc = torch.randn(
            cu_seqlens_kv[-1],
            config.hidden_size,
            dtype=dtype,
            device="cuda",
            requires_grad=True,
        )

    sigma = 0.02
    init_method = init_method_normal(sigma)
    output_layer_init_method = scaled_init_method_normal(sigma, config.num_layers)

    layer_number = 1
    drop_path_rate = 0.0
    drop_path_rates = [rate.item() for rate in torch.linspace(0, drop_path_rate, config.num_layers)]

    # Create bias
    bias = None
    if config.attn_bias_type == "post_scale_bias":
        bias = torch.randn(
            1,
            config.num_heads,
            config.max_seqlen_q,
            config.max_seqlen_kv,
            dtype=dtype,
            device="cuda",
        )

    # Create RoPE
    rotary_pos_emb = None
    if RoPE:
        PE = RotaryPositionEmbedding(dim=config.head_dim_qk)
        rotary_pos_emb = PE(config.max_seqlen_q).to(device="cuda")

    # Set up model
    block = TransformerLayer(
        config.hidden_size,
        4 * config.hidden_size,
        config.num_heads,
        num_gqa_groups=config.num_gqa_groups,
        layernorm_epsilon=1e-5,
        hidden_dropout=0.0,
        attention_dropout=config.dropout_p,
        init_method=init_method,
        output_layer_init_method=output_layer_init_method,
        layer_number=layer_number,
        kv_channels=config.head_dim_qk,
        self_attn_mask_type=config.attn_mask_type,
        tp_group=None,
        tp_size=1,
        params_dtype=dtype,
        get_rng_state_tracker=None,
        fuse_wgrad_accumulation=False,
        seq_length=config.max_seqlen_q,
        micro_batch_size=config.batch_size,
        sequence_parallel=False,
        apply_residual_connection_post_layernorm=False,
        output_layernorm=False,
        layer_type="encoder" if config.attn_type == "self" else "decoder",
        drop_path_rate=drop_path_rates[layer_number - 1],
        set_parallel_mode=True,
        fuse_qkv_params=fused_qkv_params,
        zero_centered_gamma=False,
        qkv_weight_interleaved=False,
        ub_tp_comm_overlap=False,
        bias=True,
        attn_input_format=qkv_format,
    ).to(dtype=dtype, device="cuda")
    if not is_training:
        block = block.eval()

    # Create ALiBi slopes
    alibi_slopes = None
    if config.attn_bias_type == "alibi" and config.alibi_type == "custom":
        alibi_slopes = torch.randn(config.num_heads).abs().to(dtype=torch.float32, device="cuda")

    # Run a forward and backward pass
    out = block(
        inp,
        self_attn_mask_type=config.attn_mask_type,
        encoder_output=inp_enc if config.attn_type == "cross" else None,
        enc_dec_attn_mask_type=config.attn_mask_type if config.attn_type == "cross" else None,
        checkpoint_core_attention=False,
        rotary_pos_emb=rotary_pos_emb,
        core_attention_bias_type=config.attn_bias_type,
        core_attention_bias=bias,
        alibi_slopes=alibi_slopes,
        max_seqlen_q=config.max_seqlen_q,
        max_seqlen_kv=config.max_seqlen_kv,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_kv=cu_seqlens_kv,
    )
    if is_training:
        loss = out.sum()
        loss.backward()

    return out, inp.grad


model_configs_fp8_extra_state = {
    "large": ModelConfig(2, 128, 4, 128, num_layers=1),
}


@pytest.mark.skipif(not fp8_available, reason=reason_for_no_fp8)
@pytest.mark.skipif(get_device_compute_capability() < (9, 0), reason="FP8 tests require Hopper.")
@pytest.mark.skipif(get_cudnn_version() < (9, 3, 0), reason="cuDNN 9.3.0+ is required.")
@pytest.mark.parametrize("model", ["large"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_sanity_attention_extra_state(model, dtype):
    config = model_configs_fp8_extra_state[model]
    # Test backend availability
    is_training = True
    available_backends, _, fused_attn_backends = get_available_attention_backends(
        config,
        qkv_dtype=torch.float8_e4m3fn,
        qkv_layout="sb3hd",
        is_training=is_training,
    )
    flash_attn_supported, fused_attn_supported, unfused_attn_supported = available_backends
    if not fused_attn_supported and not flash_attn_supported:
        pytest.skip("No attention backend available.")

    outputs = _run_attention_extra_state(dtype, config, checkpoint=False)
    outputs_checkpoint = _run_attention_extra_state(dtype, config, checkpoint=True)
    outputs_checkpoint_v1_6 = _run_attention_extra_state(
        dtype, config, mimic_v1_6=True, checkpoint=True
    )

    # Check that results match
    tols = dtype_tols(dtype)
    if dtype in (torch.float16, torch.bfloat16):
        tols.update(dict(rtol=2e-2, atol=2e-3))
    for i, (ref, test) in enumerate(zip(outputs, outputs_checkpoint)):
        torch.testing.assert_close(
            test,
            ref,
            **tols,
        )
    for i, (ref, test) in enumerate(zip(outputs, outputs_checkpoint_v1_6)):
        torch.testing.assert_close(
            test,
            ref,
            **tols,
        )


def _run_attention_extra_state(dtype, config, checkpoint=False, mimic_v1_6=False):
    steps = 10
    path = "checkpoint.pt"
    fp8_enabled = True
    fp8_recipe = recipe.DelayedScaling(
        margin=0,
        fp8_format=recipe.Format.HYBRID,
        amax_history_len=1,
        amax_compute_algo="most_recent",
        fp8_dpa=fp8_enabled,
        fp8_mha=False,
    )

    reset_rng_states()
    hidden_states = torch.randn(
        (config.max_seqlen_q, config.batch_size, config.hidden_size),
        dtype=dtype,
        device="cuda",
        requires_grad=True,
    )

    def get_model(dtype, config):
        sigma = 0.023
        init_method = init_method_normal(sigma)
        output_layer_init_method = scaled_init_method_normal(sigma, config.num_layers)

        with fp8_model_init(enabled=fp8_enabled, recipe=fp8_recipe):
            block = TransformerLayer(
                config.hidden_size,
                4 * config.hidden_size,
                config.num_heads,
                init_method=init_method,
                output_layer_init_method=output_layer_init_method,
                hidden_dropout=0.0,
                attention_dropout=0.0,
                fuse_qkv_params=True,
                params_dtype=dtype,
                device="cuda",
            )
        return block

    block = get_model(dtype, config)
    for i in range(steps // 2):
        with fp8_autocast(enabled=fp8_enabled, fp8_recipe=fp8_recipe):
            output = block(hidden_states, None)
            loss = output.sum()
            loss.backward()

    if checkpoint:
        sd = block.state_dict()
        if mimic_v1_6:
            sd["self_attention.core_attention.fused_attention._extra_state"] = sd[
                "self_attention.core_attention._extra_state"
            ]
            del sd["self_attention.core_attention._extra_state"]
        torch.save(sd, path)

        param_grads = []
        for p in block.parameters():
            if p.requires_grad:
                param_grads.append(p.grad.clone())

        _cpu_rng_state_new = torch.get_rng_state()
        _cuda_rng_state_new = torch.cuda.get_rng_state()

        del block
        block = get_model(dtype, config)
        block.load_state_dict(torch.load(path, weights_only=False))
        torch.set_rng_state(_cpu_rng_state_new)
        torch.cuda.set_rng_state(_cuda_rng_state_new)

        for p in block.parameters():
            if p.requires_grad:
                p.grad = param_grads.pop(0)

        assert not param_grads, "Oops!"

    for i in range((steps + 1) // 2):
        with fp8_autocast(enabled=fp8_enabled, fp8_recipe=fp8_recipe):
            output = block(hidden_states, None)
            loss = output.sum()
            loss.backward()

    torch.cuda.synchronize()

    if os.path.exists(path):
        os.remove(path)

    outputs = [output, hidden_states.grad]
    for p in block.parameters():
        if p.requires_grad:
            outputs.append(p.grad)

    return outputs


model_configs_fp8_vs_f16 = {
    #  test:             b,  h, hg,   d,   sq,  skv,   p,      mask,      bias
    "fp8_9": ModelConfig(2, 2048, 16, 128),
    "fp8_10": ModelConfig(2, 2048, 24, 128, num_gqa_groups=12),
    "fp8_11": ModelConfig(1, 8192, 32, 128, num_gqa_groups=4),
    "fp8_12": ModelConfig(2, 2048, 16, 128, attn_mask_type="causal"),
    "fp8_13": ModelConfig(2, 2048, 24, 128, num_gqa_groups=12, attn_mask_type="causal"),
    "fp8_14": ModelConfig(1, 8192, 32, 128, num_gqa_groups=4, attn_mask_type="causal"),
    "fp8_15": ModelConfig(2, 2048, 16, 128, attn_mask_type="padding"),
    "fp8_16": ModelConfig(2, 2048, 24, 128, num_gqa_groups=12, attn_mask_type="padding"),
    "fp8_17": ModelConfig(1, 8192, 32, 128, num_gqa_groups=4, attn_mask_type="padding"),
    "fp8_18": ModelConfig(2, 2048, 16, 128, attn_mask_type="padding_causal"),
    "fp8_19": ModelConfig(2, 2048, 24, 128, num_gqa_groups=12, attn_mask_type="padding_causal"),
    "fp8_20": ModelConfig(1, 8192, 32, 128, num_gqa_groups=4, attn_mask_type="padding_causal"),
}

param_types_fp8_vs_f16 = [torch.float16, torch.bfloat16]
qkv_layout_fp8_vs_f16 = ["sbh3d", "bshd_bshd_bshd", "sbhd_sbhd_sbhd"]
qkv_format_fp8_vs_f16 = ["bshd", "sbhd"]


def _rmse(a, b):
    return math.sqrt((torch.pow((a - b), 2) / a.numel()).sum())


def _error(a, b, name_a, name_b, atol, rtol, rmse_tol):
    logging.debug(name_a + " min {:.6f} max {:.6f}".format(a.min().item(), a.max().item()))
    logging.debug(name_b + " min {:.6f} max {:.6f}".format(b.min().item(), b.max().item()))
    try:
        if a.dtype != b.dtype:
            a = a.to(b.dtype)
        torch.testing.assert_close(a, b, atol=atol, rtol=rtol)
    except Exception as e:
        logging.debug(e)

    rmse = _rmse(a, b)
    logging.debug(name_a + " vs " + name_b + " RMSE: {:.6f}".format(rmse))
    rmse_range = max(a.max().item(), b.max().item()) - min(a.min().item(), b.min().item())
    assert rmse < rmse_tol * rmse_range, (
        name_a
        + " vs "
        + name_b
        + " RMSE {:.5f} is over tolerance {:.5f} ({:.5f} * {:.5f})".format(
            rmse, rmse_tol * rmse_range, rmse_tol, rmse_range
        )
    )


@pytest.mark.skipif(get_cudnn_version() < (9, 2, 1), reason="cuDNN 9.2.1+ is required.")
@pytest.mark.skipif(not fp8_available, reason=reason_for_no_fp8)
@pytest.mark.skipif(get_device_compute_capability() < (9, 0), reason="FP8 tests require Hopper+.")
@pytest.mark.parametrize("dtype", param_types_fp8_vs_f16)
@pytest.mark.parametrize("model", model_configs_fp8_vs_f16.keys())
@pytest.mark.parametrize("qkv_format", qkv_format_fp8_vs_f16)
@pytest.mark.parametrize("input_layernorm", [True, False])
@pytest.mark.parametrize("fp8_dpa_bwd", [True, False])
@pytest.mark.parametrize("RoPE", [True, False])
@pytest.mark.parametrize("is_training", [True, False])
def test_mha_fp8_vs_f16(dtype, model, qkv_format, input_layernorm, fp8_dpa_bwd, RoPE, is_training):
    os.environ["NVTE_ALLOW_NONDETERMINISTIC_ALGO"] = "1"
    os.environ["NVTE_FP8_DPA_BWD"] = "1" if fp8_dpa_bwd else "0"
    config = model_configs_fp8_vs_f16[model]

    # Test backend availability
    available_backends, _, fused_attn_backends = get_available_attention_backends(
        config,
        qkv_dtype=torch.float8_e4m3fn,
        qkv_layout=qkv_format.replace("hd", "h3d"),
        is_training=is_training,
    )
    flash_attn_supported, fused_attn_supported, unfused_attn_supported = available_backends
    # Skip if only unfused backend is supported
    if (len(fused_attn_backends) + flash_attn_supported + unfused_attn_supported) < 2:
        pytest.skip("Less than two backends to compare.")
    if not fp8_dpa_bwd:
        available_backends, _, fused_attn_backends = get_available_attention_backends(
            config,
            qkv_dtype=dtype,
            qkv_layout=qkv_format.replace("hd", "h3d"),
            is_training=is_training,
        )
        _, fused_attn_supported, _ = available_backends
        if not fused_attn_supported:
            pytest.skip("No attention backend available.")

    if flash_attn_supported:
        os.environ["NVTE_FLASH_ATTN"] = "1"
        os.environ["NVTE_FUSED_ATTN"] = "0"
        _attention_backends["backend_selection_requires_update"] = True
        logging.info("[test_mha_fp8_vs_f16]: run with fp8_mha = True")
        flash_attn_fwd_fp8, param_names, flash_attn_bwd_fp8 = _run_mha_fp8_vs_f16(
            dtype, config, True, qkv_format, input_layernorm, RoPE, is_training
        )

    os.environ["NVTE_FLASH_ATTN"] = "0"
    os.environ["NVTE_FUSED_ATTN"] = "1"
    _attention_backends["backend_selection_requires_update"] = True
    logging.info("[test_mha_fp8_vs_f16]: run with fp8_mha = True")
    fused_attn_fwd_fp8, param_names, fused_attn_bwd_fp8 = _run_mha_fp8_vs_f16(
        dtype, config, True, qkv_format, input_layernorm, RoPE, is_training
    )

    logging.info("[test_mha_fp8_vs_f16]: run with fp8_mha = False")
    fused_attn_fwd_f16, param_names, fused_attn_bwd_f16 = _run_mha_fp8_vs_f16(
        dtype, config, False, qkv_format, input_layernorm, RoPE, is_training
    )

    atol = 5e-1
    rtol = 5e-1
    rmse_tol = 0.15
    logging.debug("========== {:^25s} ==========".format("forward output"))
    if flash_attn_supported:
        _error(
            flash_attn_fwd_fp8,
            fused_attn_fwd_f16,
            "flash_attn_fwd_fp8",
            "fused_attn_fwd_f16",
            atol,
            rtol,
            rmse_tol,
        )
    _error(
        fused_attn_fwd_fp8,
        fused_attn_fwd_f16,
        "fused_attn_fwd_fp8",
        "fused_attn_fwd_f16",
        atol,
        rtol,
        rmse_tol,
    )

    if is_training:
        for i in range(len(param_names[:1])):
            logging.debug("========== {:^25s} ==========".format(param_names[i]))
            _error(
                fused_attn_bwd_fp8[i],
                fused_attn_bwd_f16[i],
                f"fused_attn_bwd_fp8[{i}]",
                f"fused_attn_bwd_f16[{i}]",
                atol,
                rtol,
                rmse_tol,
            )


def _run_mha_fp8_vs_f16(dtype, config, fp8_mha, qkv_format, input_layernorm, RoPE, is_training):
    reset_rng_states()
    _DUMMY_CUDA_RNG_STATE_TRACKER = CudaRNGStatesTracker()
    _DUMMY_CUDA_RNG_STATE_TRACKER.add("model-parallel-rng", seed)

    def get_dummy_cuda_rng_tracker() -> CudaRNGStatesTracker:
        """Get cuda rng tracker."""
        return _DUMMY_CUDA_RNG_STATE_TRACKER

    fp8_recipe = recipe.DelayedScaling(
        margin=0,
        fp8_format=recipe.Format.HYBRID,
        amax_history_len=1,
        amax_compute_algo="most_recent",
        fp8_dpa=fp8_mha,
        fp8_mha=fp8_mha,
    )

    with fp8_model_init(enabled=fp8_mha, recipe=fp8_recipe):
        rotary_pos_emb = None
        if RoPE:
            PE = RotaryPositionEmbedding(dim=config.head_dim_qk)
            rotary_pos_emb = PE(config.max_seqlen_q).to(device="cuda")
        mha = MultiheadAttention(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_heads,
            kv_channels=config.head_dim_qk,
            num_gqa_groups=config.num_gqa_groups,
            attention_dropout=config.dropout_p,
            layer_number=1,
            bias=True,
            get_rng_state_tracker=get_dummy_cuda_rng_tracker,
            params_dtype=dtype,
            input_layernorm=input_layernorm,
            fuse_qkv_params=True,
            attention_type="self",
            qkv_weight_interleaved=True,
            qkv_format=qkv_format,
        ).to(dtype=dtype, device="cuda")
        if not is_training:
            mha = mha.eval()

    if "padding" in config.attn_mask_type or qkv_format == "thd":
        if config.attn_type == "self":
            seqlens_q = torch.randint(
                1, config.max_seqlen_q, [config.batch_size], dtype=torch.int32, device="cuda"
            )
            seqlens_kv = seqlens_q
        if config.attn_type == "cross":
            seqlens_q = torch.randint(
                1, config.max_seqlen_q, [config.batch_size], dtype=torch.int32, device="cuda"
            )
            seqlens_kv = torch.randint(
                1, config.max_seqlen_kv, [config.batch_size], dtype=torch.int32, device="cuda"
            )
    else:
        seqlens_q = torch.full(
            [config.batch_size], config.max_seqlen_q, dtype=torch.int32, device="cuda"
        )
        seqlens_kv = torch.full(
            [config.batch_size], config.max_seqlen_kv, dtype=torch.int32, device="cuda"
        )
    cu_seqlens_q = torch.zeros(config.batch_size + 1, dtype=torch.int32, device="cuda")
    cu_seqlens_kv = torch.zeros(config.batch_size + 1, dtype=torch.int32, device="cuda")
    cu_seqlens_q[1:] = torch.cumsum(seqlens_q, dim=0)
    cu_seqlens_kv[1:] = torch.cumsum(seqlens_kv, dim=0)

    dim_to_num = {
        "b": config.batch_size,
        "sq": config.max_seqlen_q,
        "skv": config.max_seqlen_kv,
        "h": config.num_heads,
        "hg": config.num_gqa_groups,
        "d": config.head_dim_qk,
        "t": cu_seqlens_q[-1],
        "tg": cu_seqlens_kv[-1],
        "3": 3,
        "2": 2,
        "1": 1,
    }
    layout = "_".join(qkv_format)
    layout = layout.replace("s", "sq")
    tensor_shape = [dim_to_num[j] for j in layout.split("_")]
    tensor = 0.01 * torch.randint(-100, 100, tensor_shape, dtype=dtype, device="cuda")
    hidden_states = tensor.view(*tensor.shape[:-2], -1)
    if is_training:
        hidden_states.requires_grad = True
    tensor = 0.01 * torch.randn(tensor_shape, dtype=dtype, device="cuda")
    out_grad = tensor.view(*tensor.shape[:-2], -1)

    with fp8_autocast(enabled=fp8_mha, fp8_recipe=fp8_recipe):
        out = mha(
            hidden_states,
            attn_mask_type=config.attn_mask_type,
            checkpoint_core_attention=False,
            core_attention_bias_type=config.attn_bias_type,
            is_first_microbatch=None,
            rotary_pos_emb=rotary_pos_emb,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_kv=cu_seqlens_kv,
        )
    if is_training:
        out.backward(out_grad)

    param_names = []
    param_names.append("hidden_states.grad")
    params = []
    params.append(hidden_states)
    for name, param in mha.named_parameters():
        if param.requires_grad:
            param_names.append(name + ".grad")
            params.append(param)

    if is_training:
        return out, param_names, tuple(x.grad for x in params)
    return out, param_names, tuple(None for x in params)


@pytest.mark.skipif(get_cudnn_version() < (9, 2, 1), reason="cuDNN 9.2.1+ is required.")
@pytest.mark.skipif(not fp8_available, reason=reason_for_no_fp8)
@pytest.mark.skipif(get_device_compute_capability() < (9, 0), reason="FP8 tests require Hopper+.")
@pytest.mark.parametrize("dtype", param_types_fp8_vs_f16)
@pytest.mark.parametrize("model", model_configs_fp8_vs_f16.keys())
@pytest.mark.parametrize("qkv_layout", qkv_layout_fp8_vs_f16)
@pytest.mark.parametrize("fp8_dpa_bwd", [True, False])
@pytest.mark.parametrize("is_training", [True, False])
def test_dpa_fp8_vs_f16(dtype, model, qkv_layout, fp8_dpa_bwd, is_training):
    config = model_configs_fp8_vs_f16[model]

    # TODO(cyang): think of another way to verify dropout results
    # test cuDNN FP8 dropout
    # 1. we modify the config here to not affect mha_fp8_vs_f16 tests
    # 2. there is no other backend that implements dropout the same way as cuDNN FP8, and as an
    #    indirect verification method, we create Q/K/V as all 1s and check if O is all 1s
    # 3. we avoid running FP16/BF16 kernels as they do not have dropout support on Blackwell
    # if "padding" not in config.attn_mask_type and "causal" not in config.attn_mask_type:
    #    if get_device_compute_capability() >= (10, 0):
    #        config.dropout_p = 0.1

    os.environ["NVTE_FP8_DPA_BWD"] = "1" if fp8_dpa_bwd else "0"
    os.environ["NVTE_ALLOW_NONDETERMINISTIC_ALGO"] = "1"

    # Test backend availability
    available_backends, _, fused_attn_backends = get_available_attention_backends(
        config,
        qkv_dtype=torch.float8_e4m3fn,
        qkv_layout=qkv_layout,
        is_training=is_training,
    )
    flash_attn_supported, fused_attn_supported, unfused_attn_supported = available_backends
    # Skip if only unfused backend is supported
    if flash_attn_supported + fused_attn_supported < 1:
        pytest.skip("No FP8 attention backend available.")
    if not fp8_dpa_bwd:
        available_backends, _, fused_attn_backends = get_available_attention_backends(
            config,
            qkv_dtype=dtype,
            qkv_layout=qkv_layout,
            is_training=is_training,
        )
        _, fused_attn_supported, _ = available_backends
        if not fused_attn_supported:
            pytest.skip("No attention backend available.")
    if config.num_heads != config.num_gqa_groups and "3" in qkv_layout:
        pytest.skip("qkv_layout not applicable for MQA/GQA")

    if flash_attn_supported:
        os.environ["NVTE_FLASH_ATTN"] = "1"
        os.environ["NVTE_FUSED_ATTN"] = "0"
        _attention_backends["backend_selection_requires_update"] = True
        logging.info("[test_dpa_fp8_vs_f16]: run with fp8_dpa = True")
        flash_attn_fwd_fp8, flash_attn_bwd_fp8 = _run_dpa_fp8_vs_f16(
            dtype, config, True, qkv_layout, is_training
        )

    os.environ["NVTE_FLASH_ATTN"] = "0"
    os.environ["NVTE_FUSED_ATTN"] = "1"
    _attention_backends["backend_selection_requires_update"] = True
    logging.info("[test_dpa_fp8_vs_f16]: run with fp8_dpa = True")
    fused_attn_fwd_fp8, fused_attn_bwd_fp8 = _run_dpa_fp8_vs_f16(
        dtype, config, True, qkv_layout, is_training
    )

    if config.dropout_p == 0.0:
        # test cuDNN FP8 dropout: need a FP16/BF16 reference on Blackwell
        logging.info("[test_dpa_fp8_vs_f16]: run with fp8_dpa = False")
        fused_attn_fwd_f16, fused_attn_bwd_f16 = _run_dpa_fp8_vs_f16(
            dtype, config, False, qkv_layout, is_training
        )

    atol = 5e-1
    rtol = 5e-2
    rmse_tol = 0.11
    bwd_names = ["dq", "dk", "dv"]
    logging.debug("========== {:^25s} ==========".format("forward output"))
    if flash_attn_supported:
        _error(
            flash_attn_fwd_fp8,
            fused_attn_fwd_f16,
            "flash_attn_fwd_fp8",
            "fused_attn_fwd_f16",
            atol,
            rtol,
            rmse_tol,
        )
    if config.dropout_p != 0.0:
        # test cuDNN FP8 dropout
        assert torch.all(
            fused_attn_fwd_fp8 == 1
        ), "fused_attn_fwd_fp8 must be all 1s when Q/K/V are all 1s."
    else:
        _error(
            fused_attn_fwd_fp8,
            fused_attn_fwd_f16,
            "fused_attn_fwd_fp8",
            "fused_attn_fwd_f16",
            atol,
            rtol,
            rmse_tol,
        )
        if is_training:
            for i, _ in enumerate(fused_attn_bwd_f16):
                logging.debug("========== {:^25s} ==========".format(bwd_names[i]))
                _error(
                    fused_attn_bwd_fp8[i],
                    fused_attn_bwd_f16[i],
                    f"fused_attn_bwd_fp8[{i}]",
                    f"fused_attn_bwd_f16[{i}]",
                    atol,
                    rtol,
                    rmse_tol,
                )


def _run_dpa_fp8_vs_f16(dtype, config, fp8_dpa, qkv_layout, is_training):

    reset_rng_states()
    _DUMMY_CUDA_RNG_STATE_TRACKER = CudaRNGStatesTracker()
    _DUMMY_CUDA_RNG_STATE_TRACKER.add("model-parallel-rng", seed)

    def get_dummy_cuda_rng_tracker() -> CudaRNGStatesTracker:
        """Get cuda rng tracker."""
        return _DUMMY_CUDA_RNG_STATE_TRACKER

    fp8_recipe = recipe.DelayedScaling(
        margin=0,
        fp8_format=recipe.Format.HYBRID,
        amax_history_len=1,
        amax_compute_algo="most_recent",
        fp8_dpa=fp8_dpa,
    )

    qkv_format = "".join([i for i in qkv_layout.split("_")[0] if i.isalpha()])
    with fp8_model_init(enabled=fp8_dpa):
        dpa = DotProductAttention(
            config.num_heads,
            config.head_dim_qk,
            num_gqa_groups=config.num_gqa_groups,
            attention_dropout=config.dropout_p,
            sequence_parallel=False,
            tp_size=1,
            get_rng_state_tracker=get_dummy_cuda_rng_tracker,
            tp_group=None,
            layer_number=1,
            attention_type="self",
            qkv_format=qkv_format,
        ).to(dtype=dtype, device="cuda")
        if not is_training:
            dpa = dpa.eval()

    if "padding" in config.attn_mask_type or qkv_format == "thd":
        if config.attn_type == "self":
            seqlens_q = torch.randint(
                1, config.max_seqlen_q, [config.batch_size], dtype=torch.int32, device="cuda"
            )
            seqlens_kv = seqlens_q
        if config.attn_type == "cross":
            seqlens_q = torch.randint(
                1, config.max_seqlen_q, [config.batch_size], dtype=torch.int32, device="cuda"
            )
            seqlens_kv = torch.randint(
                1, config.max_seqlen_kv, [config.batch_size], dtype=torch.int32, device="cuda"
            )
    else:
        seqlens_q = torch.full(
            [config.batch_size], config.max_seqlen_q, dtype=torch.int32, device="cuda"
        )
        seqlens_kv = torch.full(
            [config.batch_size], config.max_seqlen_kv, dtype=torch.int32, device="cuda"
        )
    cu_seqlens_q = torch.zeros(config.batch_size + 1, dtype=torch.int32, device="cuda")
    cu_seqlens_kv = torch.zeros(config.batch_size + 1, dtype=torch.int32, device="cuda")
    cu_seqlens_q[1:] = torch.cumsum(seqlens_q, dim=0)
    cu_seqlens_kv[1:] = torch.cumsum(seqlens_kv, dim=0)

    dim_to_num = {
        "b": config.batch_size,
        "sq": config.max_seqlen_q,
        "skv": config.max_seqlen_kv,
        "h": config.num_heads,
        "hg": config.num_gqa_groups,
        "d": config.head_dim_qk,
        "t": cu_seqlens_q[-1],
        "tg": cu_seqlens_kv[-1],
        "3": 3,
        "2": 2,
        "1": 1,
    }
    inp = []
    for i, layout in enumerate(qkv_layout.split("_")):
        layout = "_".join(layout)
        if i == 0:
            layout = layout.replace("s", "sq")
        else:
            layout = layout.replace("s", "skv")
            layout = layout.replace("h", "hg")
            layout = layout.replace("t", "tg")
        tensor_shape = [dim_to_num[j] for j in layout.split("_")]
        if config.dropout_p == 0.0:
            tensor = torch.randn(tensor_shape, dtype=dtype, device="cuda")
        else:
            # test cuDNN FP8 dropout
            tensor = torch.ones(tensor_shape, dtype=dtype, device="cuda")
        tensor_count = 1
        split_dim = 0
        for dim, l in enumerate(layout.split("_")):
            if l.isdigit():
                tensor_count = int(l)
                split_dim = dim
                break
        tensors = torch.split(tensor, 1, dim=split_dim) if split_dim != 0 else [tensor]
        for j in range(tensor_count):
            if split_dim != 0:
                inp.append(tensors[j].squeeze(split_dim))
            else:
                inp.append(tensors[j])
    for i in range(3):
        inp[i].requires_grad = True

    qkv_format_kv = "_".join(qkv_format)
    qkv_format_kv = qkv_format_kv.replace("s", "sq")
    out_grad_shape = [dim_to_num[i] for i in qkv_format_kv.split("_")]
    out_grad_shape_new = [*out_grad_shape[:-2], out_grad_shape[-2] * out_grad_shape[-1]]
    out_grad = torch.randn(out_grad_shape_new, dtype=dtype, device="cuda")

    with fp8_autocast(enabled=fp8_dpa, fp8_recipe=fp8_recipe):
        out = dpa(
            inp[0],
            inp[1],
            inp[2],
            qkv_format=qkv_format,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_kv=cu_seqlens_kv,
            max_seqlen_q=config.max_seqlen_q,
            max_seqlen_kv=config.max_seqlen_kv,
            attn_mask_type=config.attn_mask_type,
            checkpoint_core_attention=False,
            core_attention_bias_type=config.attn_bias_type,
        )
    if is_training:
        out.backward(out_grad)

    if is_training:
        return out, (inp[0].grad, inp[1].grad, inp[2].grad)
    return out, (None, None, None)


model_configs_fp8 = {
    #  test:             b,  h, hg,   d,   sq,  skv,   p,      mask,      bias
    "fp8_1": ModelConfig(1, 512, 1, 64),
    "fp8_2": ModelConfig(4, 512, 16, 64),
    "fp8_3": ModelConfig(1, 2048, 1, 128),
    "fp8_4": ModelConfig(2, 2048, 24, 128),
    "fp8_5": ModelConfig(1, 512, 1, 64, attn_mask_type="causal"),
    "fp8_6": ModelConfig(4, 512, 16, 64, attn_mask_type="causal"),
    "fp8_7": ModelConfig(1, 2048, 1, 128, attn_mask_type="causal"),
    "fp8_8": ModelConfig(2, 2048, 24, 128, attn_mask_type="causal"),
}
param_types_fp8 = [torch.float16, torch.bfloat16]
cudnn_frontend_version = int(os.getenv("NVTE_FUSED_ATTN_FE_VER", "1"))
models_v0 = ["fp8_1", "fp8_2", "fp8_5", "fp8_6"]
models_v1 = ["fp8_3", "fp8_4", "fp8_7", "fp8_8"]


@pytest.mark.skipif(
    (
        get_cudnn_version() < (8, 9, 3)
        if cudnn_frontend_version == 0
        else get_cudnn_version() < (9, 2, 1)
    ),
    reason=f"""cuDNN {"8.9.3" if cudnn_frontend_version == 0 else "9.2.1"}+ is required.""",
)
@pytest.mark.skipif(not fp8_available, reason=reason_for_no_fp8)
@pytest.mark.skipif(get_device_compute_capability() < (9, 0), reason="FP8 tests require Hopper+.")
@pytest.mark.parametrize("dtype", param_types_fp8)
@pytest.mark.parametrize("model", models_v1 if cudnn_frontend_version == 1 else models_v0)
def test_custom_mha_fp8_vs_f16(dtype, model):
    """Test FP8 dot product attention implementations based on cuDNN frontend
    v0.9 and v1.0+. Each test compares results from a custom implementation of
    an FP8 MHA module, i.e. Custom_MHA_FP8(), to results from an F16 MHA
    implementation, i.e. transformer_engine.pytorch.attention.MultiHeadAttention.
    Both paths take F16 input and output. QKV layout is t3hd or bs3hd"""

    config = model_configs_fp8[model]

    # Test backend availability
    is_training = True
    available_backends, _, fused_attn_backends = get_available_attention_backends(
        config,
        qkv_dtype=torch.float8_e4m3fn,
        qkv_layout="t3hd" if cudnn_frontend_version == 0 else "bs3hd",
        is_training=is_training,
    )
    flash_attn_supported, fused_attn_supported, unfused_attn_supported = available_backends
    if not (fused_attn_backends and unfused_attn_supported):
        pytest.skip("Not enough backends to run this test with.")

    fused_attn_fwd_fp8, fused_attn_bwd_fp8 = _run_custom_mha_fp8(dtype, config, "FusedAttention")
    unfused_attn_fwd_f16, unfused_attn_bwd_f16 = _run_ref_mha_f16(dtype, config, "UnfusedAttention")

    atol = 5e-1
    rtol = 5e-1
    rmse_tol = 0.13
    _error(
        fused_attn_fwd_fp8,
        unfused_attn_fwd_f16,
        "fused_attn_fwd_fp8",
        "unfused_attn_fwd_f16",
        atol,
        rtol,
        rmse_tol,
    )
    _error(
        fused_attn_bwd_fp8,
        unfused_attn_bwd_f16,
        "fused_attn_bwd_fp8",
        "unfused_attn_bwd_f16",
        atol,
        rtol,
        rmse_tol,
    )


def _run_custom_mha_fp8(dtype, config, backend):
    """Run Custom_MHA_FP8 with FP8 FusedAttention backend. Both input and output
    are in F16. QKV GEMM, DPA, and projection GEMM are calculated in FP8."""
    reset_rng_states()
    os.environ["NVTE_FLASH_ATTN"] = "0"
    os.environ["NVTE_FUSED_ATTN"] = "0"
    if backend == "FlashAttention":
        os.environ["NVTE_FLASH_ATTN"] = "1"
    if backend == "FusedAttention":
        os.environ["NVTE_FUSED_ATTN"] = "1"
    _attention_backends["backend_selection_requires_update"] = True

    inp = 0.0001 * torch.randint(
        -100,
        100,
        (config.batch_size * config.max_seqlen_q, config.num_heads * config.head_dim_qk),
        dtype=dtype,
        device="cuda",
        requires_grad=True,
    )
    seqlens = torch.full([config.batch_size], config.max_seqlen_q, dtype=torch.int32, device="cuda")
    cu_seqlens = torch.zeros(config.batch_size + 1, device="cuda", dtype=torch.int32)
    cu_seqlens[1:] = torch.cumsum(seqlens, dim=0)

    out_grad = 0.01 * torch.randn(
        config.batch_size * config.max_seqlen_q,
        config.num_heads * config.head_dim_qk,
        dtype=dtype,
        device="cuda",
    )
    torch.save(out_grad, "out_grad.pt")

    fp8_recipe = recipe.DelayedScaling(
        margin=0,
        fp8_format=recipe.Format.HYBRID,
        amax_history_len=1,
        amax_compute_algo="most_recent",
    )

    mha = Custom_MHA_FP8(config).to(dtype=dtype, device="cuda")
    with fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
        out = mha(inp, cu_seqlens, config.max_seqlen_q)
    out.backward(out_grad)

    out = torch.load("out.pt")
    dqkv = torch.load("dqkv.pt")
    return (
        out.view(config.batch_size, config.max_seqlen_q, -1),
        dqkv.view(
            config.batch_size, config.max_seqlen_q, 3, config.num_heads, config.head_dim_qk
        ).contiguous(),
    )


def _run_ref_mha_f16(dtype, config, backend):
    """Run reference F16 FusedAttention. Both input and output
    are in F16. QKV GEMM, DPA, and projection GEMM are also in F16."""

    os.environ["NVTE_FLASH_ATTN"] = "0"
    os.environ["NVTE_FUSED_ATTN"] = "0"
    if backend == "FlashAttention":
        os.environ["NVTE_FLASH_ATTN"] = "1"
    if backend == "FusedAttention":
        os.environ["NVTE_FUSED_ATTN"] = "1"
    _attention_backends["backend_selection_requires_update"] = True

    inp = torch.load("qkv.pt").to(device="cuda")
    inp.requires_grad = True
    seqlens = torch.full([config.batch_size], config.max_seqlen_q, dtype=torch.int32, device="cuda")
    cu_seqlens = torch.zeros(config.batch_size + 1, device="cuda", dtype=torch.int32)
    cu_seqlens[1:] = torch.cumsum(seqlens, dim=0)
    out_grad = (
        torch.load("out_grad.pt").to(device="cuda").view(config.batch_size, config.max_seqlen_q, -1)
    )

    _DUMMY_CUDA_RNG_STATE_TRACKER = CudaRNGStatesTracker()
    _DUMMY_CUDA_RNG_STATE_TRACKER.add("model-parallel-rng", seed)

    def get_dummy_cuda_rng_tracker() -> CudaRNGStatesTracker:
        """Get cuda rng tracker."""
        return _DUMMY_CUDA_RNG_STATE_TRACKER

    block = DotProductAttention(
        config.num_heads,
        config.head_dim_qk,
        attention_dropout=config.dropout_p,
        sequence_parallel=False,
        tp_size=1,
        get_rng_state_tracker=get_dummy_cuda_rng_tracker,
        tp_group=None,
        layer_number=1,
        attention_type="self",
        qkv_format="bshd",
    ).to(dtype=dtype, device="cuda")

    q = inp[:, :, 0, :, :]
    k = inp[:, :, 1, :, :]
    v = inp[:, :, 2, :, :]
    out = block(q, k, v, attn_mask_type=config.attn_mask_type)
    out.backward(out_grad)

    return out, inp.grad


_CUBLASLT_WORKSPACE_SIZE_BYTES = 33_554_432  # 32MiB
_2X_ACC_FPROP = False
_2X_ACC_DGRAD = False
_2X_ACC_WGRAD = False

META_QKV = tex.FP8FwdTensors.GEMM1_OUTPUT
META_DQKV = tex.FP8BwdTensors.GRAD_OUTPUT1
META_O = tex.FP8FwdTensors.GEMM2_INPUT
META_DO = tex.FP8BwdTensors.GRAD_INPUT2
META_S = tex.FP8FwdTensors.GEMM3_OUTPUT
META_DP = tex.FP8BwdTensors.GRAD_INPUT3


class _custom_mha_fp8(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        inp: torch.Tensor,
        qkv_weight: torch.Tensor,
        qkv_bias: torch.Tensor,
        cu_seqlens: torch.Tensor,
        num_heads: int,
        p_dropout: float,
        max_s: int,
        fast_zero_fill: bool,
        fp8_meta: Dict[str, Any],
        workspace: torch.Tensor,
        is_training: bool,
        mask_type: str,
        quantizers: list[Quantizer],
    ) -> torch.Tensor:
        qkv_dtype = inp.dtype

        assert inp.dim() == 2
        in_features = qkv_weight.shape[-1]
        h = num_heads
        d = in_features // h
        b = cu_seqlens.numel() - 1

        input_quantizer = quantizers["scaling_fwd"][tex.FP8FwdTensors.GEMM1_INPUT]
        qkv_quantizer = quantizers["scaling_fwd"][tex.FP8FwdTensors.GEMM2_INPUT]
        qkv_weight_quantizer = quantizers["scaling_fwd"][tex.FP8FwdTensors.GEMM1_WEIGHT]
        o_quantizer = quantizers["scaling_fwd"][tex.FP8FwdTensors.GEMM1_OUTPUT]
        dO_quantizer = quantizers["scaling_bwd"][tex.FP8BwdTensors.GRAD_OUTPUT1]
        dQKV_quantizer = quantizers["scaling_bwd"][tex.FP8BwdTensors.GRAD_INPUT1]
        s_quantizer = quantizers["scaling_bwd"][tex.FP8BwdTensors.GRAD_OUTPUT2]
        dP_quantizer = quantizers["scaling_bwd"][tex.FP8BwdTensors.GRAD_OUTPUT3]

        inp_fp8 = input_quantizer(inp)

        qkv_weight_fp8 = qkv_weight_quantizer(qkv_weight)

        qkv, *_ = ext.general_gemm(
            qkv_weight_fp8,
            inp_fp8,
            workspace,
            bias=qkv_bias,
            out_dtype=qkv_weight_fp8.dtype,
            quantization_params=qkv_quantizer,
            use_split_accumulator=_2X_ACC_FPROP,
        )
        qkv = qkv.view(-1, 3, h, d)
        qkv_fp16 = qkv.dequantize().view(b, max_s, 3, h, d).contiguous()
        torch.save(qkv_fp16, "qkv.pt")
        if cudnn_frontend_version == 1:
            qkv = qkv.view(b, max_s, 3, h, d)  # bs3hd

        # FMHA
        q_data = qkv._data[:, :, 0, :, :] if cudnn_frontend_version == 1 else qkv._data[:, 0, :, :]
        k_data = qkv._data[:, :, 1, :, :] if cudnn_frontend_version == 1 else qkv._data[:, 1, :, :]
        v_data = qkv._data[:, :, 2, :, :] if cudnn_frontend_version == 1 else qkv._data[:, 2, :, :]
        q = qkv.make_like(tensor=qkv, data=q_data, shape=q_data.shape)
        k = qkv.make_like(tensor=qkv, data=k_data, shape=k_data.shape)
        v = qkv.make_like(tensor=qkv, data=v_data, shape=v_data.shape)

        out, aux_ctx_tensors = fused_attn_fwd(
            is_training,
            max_s,
            max_s,
            cu_seqlens,
            cu_seqlens,
            q,
            k,
            v,
            qkv_dtype,
            FusedAttnBackend["FP8"],
            attn_scale=None,
            dropout=p_dropout,
            fast_zero_fill=fast_zero_fill,
            qkv_layout="bs3hd" if cudnn_frontend_version == 1 else "t3hd",
            attn_bias_type="no_bias",
            attn_mask_type=mask_type if cudnn_frontend_version == 1 else "padding",
            rng_gen=None,
            o_quantizer=o_quantizer,
            s_quantizer=s_quantizer,
        )

        tensors_to_save, tensor_objects = prepare_for_saving(
            q, k, v, inp_fp8, qkv_weight_fp8, workspace, out
        )

        ctx.save_for_backward(*tensors_to_save)
        ctx.tensor_objects = tensor_objects
        ctx.aux_ctx_tensors = aux_ctx_tensors
        ctx.qkv_dtype = qkv_dtype
        ctx.fp8_meta = fp8_meta
        ctx.cu_seqlens = cu_seqlens
        ctx.p_dropout = p_dropout
        ctx.max_s = max_s
        ctx.fast_zero_fill = fast_zero_fill
        ctx.hidden_size = in_features
        ctx.num_heads = num_heads
        ctx.mask_type = mask_type
        ctx.dtype = inp.dtype

        ctx.dQKV_quantizer = dQKV_quantizer
        ctx.dO_quantizer = dO_quantizer
        ctx.dP_quantizer = dP_quantizer
        ctx.S_quantizer = s_quantizer

        out = out.view(-1, in_features)  # (bs)(hd)
        out_fp16 = out.dequantize()
        torch.save(out_fp16, "out.pt")  # (bs)(hd)
        return out_fp16

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[Union[torch.Tensor, None], ...]:
        with torch.cuda.nvtx.range("_DPA"):
            saved_tensors = ctx.saved_tensors
            (q, k, v, inp_fp8, qkv_weight_fp8, workspace, out) = restore_from_saved(
                ctx.tensor_objects, saved_tensors
            )

            proj_dgrad = ctx.dO_quantizer(grad_output)
            fp8_dtype_backward = fp8.get_fp8_te_dtype(ctx.fp8_meta["recipe"], fprop_tensor=False)

            dq, dk, dv, *rest = fused_attn_bwd(
                ctx.max_s,
                ctx.max_s,
                ctx.cu_seqlens,
                ctx.cu_seqlens,
                q,
                k,
                v,
                out,
                proj_dgrad.view_as(out),
                ctx.qkv_dtype,
                fp8_dtype_backward,
                ctx.aux_ctx_tensors,
                FusedAttnBackend["FP8"],
                None,
                None,
                ctx.S_quantizer,
                ctx.dP_quantizer,
                ctx.dQKV_quantizer,
                attn_scale=None,
                dropout=ctx.p_dropout,
                fast_zero_fill=ctx.fast_zero_fill,
                qkv_layout="bs3hd" if cudnn_frontend_version == 1 else "t3hd",
                attn_bias_type="no_bias",
                attn_mask_type=ctx.mask_type if cudnn_frontend_version == 1 else "padding",
            )
            dim = 2 if cudnn_frontend_version == 1 else 1
            dqkv = torch.Tensor().to(device=dq._data.device, dtype=dq._data.dtype)
            dqkv_shape = list(dq._data.shape)
            dqkv_shape.insert(dim, 3)
            dqkv_stride = list(dq._data.stride())
            dqkv_stride.insert(dim, int(dqkv_stride[-3] / 3))
            dqkv.set_(
                dq._data.untyped_storage(), dq._data.storage_offset(), dqkv_shape, dqkv_stride
            )  # bs3hd

            dqkv_c = dqkv.view(-1, 3 * ctx.hidden_size)
            dqkv_c = dq.make_like(tensor=dq, data=dqkv_c, shape=dqkv_c.shape)
            dqkv_c_fp16 = dqkv_c.dequantize()
            torch.save(dqkv_c_fp16, "dqkv.pt")

            qkv_bgrad, dqkv = ext.bgrad_quantize(dqkv_c_fp16, ctx.dQKV_quantizer)
            dqkv_c._transpose = None
            dqkv_c._create_transpose()

            # QKV DGRAD
            qkv_dgrad, *_ = ext.general_gemm(
                qkv_weight_fp8,
                dqkv_c,
                workspace,
                ctx.dtype,
                use_split_accumulator=_2X_ACC_DGRAD,
                layout="NN",
            )

            # QKV WGRAD
            qkv_wgrad, *_ = ext.general_gemm(
                inp_fp8,
                dqkv,
                workspace,
                ctx.dtype,
                use_split_accumulator=_2X_ACC_WGRAD,
                layout="NT",
            )

        return (
            qkv_dgrad,
            qkv_wgrad,
            qkv_bgrad,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class Custom_MHA_FP8(TransformerEngineBaseModule):
    def __init__(self, config, params_dtype: torch.dtype = torch.float32):
        super().__init__()
        self.p_dropout = config.dropout_p
        self.h = config.num_heads
        self.hidden_size = config.hidden_size
        self.head_dim = config.head_dim_qk
        self.fast_zero_fill = True
        self.mask_type = config.attn_mask_type

        self.qkv_weight = torch.nn.Parameter(
            torch.empty(
                self.hidden_size * 3,
                self.hidden_size,
                device=torch.cuda.current_device(),
                dtype=params_dtype,
            )
        )
        self.qkv_bias = torch.nn.Parameter(
            torch.empty(
                self.hidden_size * 3,
                device=torch.cuda.current_device(),
                dtype=params_dtype,
            )
        )
        with torch.no_grad():
            self.qkv_bias.zero_()
            self.qkv_weight.fill_(1.0)
        self.workspace = torch.empty(
            _CUBLASLT_WORKSPACE_SIZE_BYTES, dtype=torch.int8, device="cuda"
        )

    def forward(
        self,
        inp: torch.Tensor,
        cu_seqlens,
        max_s,
    ) -> torch.Tensor:
        with self.prepare_forward(inp, num_gemms=3) as inp:
            out = _custom_mha_fp8.apply(
                inp,
                self.qkv_weight,
                self.qkv_bias,
                cu_seqlens,
                self.h,
                self.p_dropout,
                max_s,
                self.fast_zero_fill,
                self.fp8_meta,
                self.workspace,
                self.training,
                self.mask_type,
                self.quantizers,
            )
        return out
