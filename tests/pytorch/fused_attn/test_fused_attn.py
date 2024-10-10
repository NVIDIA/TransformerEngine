# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import functools
import logging
import math
import os
from importlib.metadata import version
from typing import Any, Dict, List, Tuple, Union, Optional
from contextlib import contextmanager

import pytest
import torch

from transformer_engine.common import recipe
from transformer_engine.pytorch import TransformerLayer, fp8_autocast, fp8_model_init
from transformer_engine.pytorch.attention import (
    DotProductAttention,
    MultiheadAttention,
    RotaryPositionEmbedding,
    get_attention_backend,
    _flash_attn_2_3_plus,
    _flash_attn_3_is_installed,
    check_set_window_size,
    AttentionParams,
    _attention_backends,
)
from transformer_engine.pytorch.constants import TE_DType
import transformer_engine.pytorch.cpp_extensions as ext
from transformer_engine.pytorch.cpp_extensions.fused_attn import (
    AttnBiasType,
    AttnMaskType,
    FusedAttnBackend,
    QKVLayout,
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
from transformer_engine_torch import NVTE_Fused_Attn_Backend

# Only run FP8 tests on H100
fp8_available, reason_for_no_fp8 = fp8.FP8GlobalStateManager.is_fp8_available()

# Initialize RNG state
seed = 1234
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
_cpu_rng_state = torch.get_rng_state()
_cuda_rng_state = torch.cuda.get_rng_state()


def reset_rng_states() -> None:
    """Revert back to initial RNG state"""
    torch.set_rng_state(_cpu_rng_state)
    torch.cuda.set_rng_state(_cuda_rng_state)


@pytest.fixture(autouse=True)
def reset_global_fp8_state():
    yield
    fp8.FP8GlobalStateManager.reset()


class ModelConfig:
    def __init__(
        self,
        batch_size: int,
        num_heads: int,
        num_gqa_groups: int,
        head_dim_qk: int,
        max_seqlen_q: int,
        max_seqlen_kv: int,
        dropout_p: float,
        attn_mask_type: str,
        attn_bias_type: str,
        head_dim_v: int = None,
        alibi_type: str = "none",
        num_layers: int = 1,
        bias_shape: str = "1hss",
        window_size: Tuple[int, int] = (-1, -1),
    ):
        self.batch_size = batch_size
        self.num_heads = num_heads
        self.num_gqa_groups = num_gqa_groups
        self.head_dim_qk = head_dim_qk
        self.head_dim_v = head_dim_qk if head_dim_v is None else head_dim_v
        self.hidden_size = num_heads * head_dim_qk
        self.hidden_size_kv = num_gqa_groups * self.head_dim_v
        self.max_seqlen_q = max_seqlen_q
        self.max_seqlen_kv = max_seqlen_kv
        self.dropout_p = dropout_p
        self.attn_mask_type = attn_mask_type
        self.attn_bias_type = attn_bias_type
        self.alibi_type = alibi_type
        self.attn_type = "self" if (max_seqlen_q == max_seqlen_kv) else "cross"
        self.num_layers = num_layers
        self.bias_shape = bias_shape
        self.window_size = window_size


@contextmanager
def logging_context(highest_level=logging.WARNING):
    previous_level = logging.root.manager.disable
    logging.disable(highest_level)
    try:
        yield
    finally:
        logging.disable(previous_level)


def _get_attention_backends(
    config: ModelConfig,
    qkv_dtype: torch.dtype,
    qkv_layout: str,
    window_size: Tuple[int, int] = (-1, -1),
    pad_between_seqs: bool = False,
    context_parallel: bool = False,
    deterministic: bool = False,
    fp8: bool = False,
    fp8_meta: Optional[Dict[str, Any]] = None,
) -> Tuple[List, List]:
    """Check if what attention backends support a model configuration"""

    os.environ["NVTE_FLASH_ATTN"] = "1"
    os.environ["NVTE_FUSED_ATTN"] = "1"
    os.environ["NVTE_UNFUSED_ATTN"] = "1"
    _attention_backends["backend_selection_requires_update"] = True

    alibi_slopes_shape = None
    if config.attn_bias_type == "alibi" and config.alibi_type == "custom":
        if config.bias_shape == "1hss":
            alibi_slopes_shape = [config.num_heads]
        if config.bias_shape == "bhss":
            alibi_slopes_shape = [config.batch_size, config.num_heads]

    core_attention_bias_shape = (
        config.bias_shape if config.attn_bias_type == "post_scale_bias" else None
    )
    core_attention_bias_requires_grad = False
    # d=256 is supported by cuDNN 9.0+ for inference but not training
    if (
        config.attn_bias_type == "post_scale_bias"
        and config.head_dim_qk <= 128
        and config.head_dim_v <= 128
    ):
        core_attention_bias_requires_grad = True

    fused_attn_backends = []
    available_backends = None
    fused_attention_backend = None

    def test():
        attention_params = AttentionParams(
            qkv_dtype=qkv_dtype,
            qkv_layout=qkv_layout,
            batch_size=config.batch_size,
            num_heads=config.num_heads,
            num_gqa_groups=config.num_gqa_groups,
            max_seqlen_q=config.max_seqlen_q,
            max_seqlen_kv=config.max_seqlen_kv,
            head_dim_qk=config.head_dim_qk,
            head_dim_v=config.head_dim_v,
            attn_mask_type=config.attn_mask_type,
            window_size=window_size,
            alibi_slopes_shape=alibi_slopes_shape,
            core_attention_bias_type=config.attn_bias_type,
            core_attention_bias_shape=core_attention_bias_shape,
            core_attention_bias_requires_grad=core_attention_bias_requires_grad,
            pad_between_seqs=pad_between_seqs,
            attention_dropout=config.dropout_p,
            context_parallel=context_parallel,
            deterministic=deterministic,
            fp8=fp8,
            fp8_meta=fp8_meta,
        )
        _, _, fused_attention_backend, _, available_backends = get_attention_backend(
            attention_params
        )
        return available_backends, fused_attention_backend

    backends = {0: "F16_max512_seqlen", 1: "F16_arbitrary_seqlen", 2: "FP8"}
    with logging_context():
        for i in range(3):
            os.environ["NVTE_FUSED_ATTN_BACKEND"] = str(i)
            _attention_backends["backend_selection_requires_update"] = True
            available_backends, fused_attention_backend = test()
            if fused_attention_backend == FusedAttnBackend[backends[i]]:
                fused_attn_backends.append(fused_attention_backend)
    return available_backends, fused_attn_backends


model_configs_base = {
    #     test:             b,  h, hg,   d,   sq,  skv,   p,      mask,      bias   # attn , backend
    "base_1_0": ModelConfig(8, 16, 16, 64, 128, 128, 0.0, "no_mask", "no_bias"),  # self , 0
    "base_1_1": ModelConfig(4, 16, 16, 64, 128, 256, 0.0, "no_mask", "no_bias"),  # cross, 0
    "base_2_0": ModelConfig(2, 24, 24, 128, 2048, 2048, 0.0, "no_mask", "no_bias"),  # self , 1
    "base_2_1": ModelConfig(1, 24, 24, 128, 2048, 4096, 0.0, "no_mask", "no_bias"),  # cross, 1
    "base_3_0": ModelConfig(8, 16, 16, 128, 1, 2048, 0.0, "no_mask", "no_bias"),  # inference
    "base_3_1": ModelConfig(8, 16, 16, 256, 1, 2048, 0.0, "no_mask", "no_bias"),  # inference
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
    if qkv_layout is None:
        if config.attn_type == "self":
            qkv_layout = "sb3hd" if not is_mla else "sbhd_sbhd_sbhd"
        else:
            qkv_layout = "bshd_bs2hd" if not is_mla else "bshd_bshd_bshd"
    if "3" in qkv_layout and config.attn_type == "cross":
        pytest.skip("No need to test this layout for cross attention")

    # Test backend availability
    window_size = (-1, -1)
    if swa:
        window_size = [2, 2]
    config.window_size = check_set_window_size(config.attn_mask_type, window_size)
    available_backends, fused_attn_backends = _get_attention_backends(
        config,
        qkv_dtype=dtype,
        qkv_layout=qkv_layout,
        window_size=config.window_size,
        pad_between_seqs=pad_between_seqs,
    )
    flash_attn_supported, fused_attn_supported, unfused_attn_supported = available_backends
    # FlashAttention does not support pad_between_seqs, but _run_dot_product_attention
    # mannually pads and unpads the input and output of FlashAttention for testing purposes
    if pad_between_seqs and not (
        config.max_seqlen_q != config.max_seqlen_kv
        and config.attn_mask_type in ["causal", "padding_causal"]
    ):
        flash_attn_supported = True

    # Skip if only unfused backend is supported
    if (len(fused_attn_backends) + flash_attn_supported + unfused_attn_supported) < 2:
        pytest.skip("Less than two backends to compare.")

    is_training = config.head_dim_qk <= 128 and config.head_dim_v <= 128
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

    if unfused_attn_supported and fused_attn_supported:
        logging.info("[test_dot_product_attention]: unfused attn vs fused attn")
        torch.testing.assert_close(fused_attn_fwd, unfused_attn_fwd, **tols)
        for i, _ in enumerate(unfused_attn_bwd):
            torch.testing.assert_close(fused_attn_bwd[i], unfused_attn_bwd[i], **tols)
    if unfused_attn_supported and flash_attn_supported:
        logging.info("[test_dot_product_attention]: unfused attn vs flash attn")
        torch.testing.assert_close(flash_attn_fwd, unfused_attn_fwd, **tols)
        for i, _ in enumerate(flash_attn_bwd):
            torch.testing.assert_close(unfused_attn_bwd[i], flash_attn_bwd[i], **tols)
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
    "mla_1_0": ModelConfig(
        8, 16, 16, 64, 128, 128, 0.0, "no_mask", "no_bias", head_dim_v=128
    ),  # self , 0
    "mla_1_1": ModelConfig(
        4, 16, 16, 64, 128, 256, 0.0, "no_mask", "no_bias", head_dim_v=128
    ),  # cross, 0
    "mla_2_0": ModelConfig(
        2, 24, 24, 128, 2048, 2048, 0.0, "causal", "no_bias", head_dim_v=64
    ),  # self , 1
    "mla_2_1": ModelConfig(
        1, 24, 24, 128, 2048, 4096, 0.0, "causal", "no_bias", head_dim_v=64
    ),  # cross, 1
    "mla_3_0": ModelConfig(
        8, 16, 16, 128, 1, 2048, 0.0, "no_mask", "no_bias", head_dim_v=64
    ),  # inference
    "mla_3_1": ModelConfig(
        8, 16, 16, 256, 1, 2048, 0.0, "no_mask", "no_bias", head_dim_v=128
    ),  # inference
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
    "mask_1_0": ModelConfig(8, 16, 16, 64, 128, 128, 0.0, "causal", "no_bias"),
    "mask_1_1": ModelConfig(4, 16, 16, 64, 128, 256, 0.0, "causal", "no_bias"),
    "mask_2_0": ModelConfig(2, 24, 24, 128, 2048, 2048, 0.0, "causal", "no_bias"),
    "mask_2_1": ModelConfig(1, 24, 24, 128, 2048, 4096, 0.0, "causal", "no_bias"),
    "mask_3_0": ModelConfig(8, 16, 16, 64, 128, 128, 0.0, "padding", "no_bias"),
    "mask_3_1": ModelConfig(4, 16, 16, 64, 128, 256, 0.0, "padding", "no_bias"),
    "mask_4_0": ModelConfig(2, 24, 24, 128, 2048, 2048, 0.0, "padding", "no_bias"),
    "mask_4_1": ModelConfig(1, 24, 24, 128, 2048, 4096, 0.0, "padding", "no_bias"),
    "mask_5_0": ModelConfig(8, 16, 16, 64, 128, 128, 0.0, "padding_causal", "no_bias"),
    "mask_5_1": ModelConfig(4, 16, 16, 64, 128, 256, 0.0, "padding_causal", "no_bias"),
    "mask_6_0": ModelConfig(2, 24, 24, 128, 2048, 2048, 0.0, "padding_causal", "no_bias"),
    "mask_6_1": ModelConfig(1, 24, 24, 128, 2048, 4096, 0.0, "padding_causal", "no_bias"),
    "mask_7_0": ModelConfig(2, 24, 24, 128, 2048, 2048, 0.0, "causal_bottom_right", "no_bias"),
    "mask_7_1": ModelConfig(1, 24, 24, 128, 2048, 4096, 0.0, "causal_bottom_right", "no_bias"),
    "mask_8_0": ModelConfig(
        2, 24, 24, 128, 2048, 2048, 0.0, "padding_causal_bottom_right", "no_bias"
    ),
    "mask_8_1": ModelConfig(
        1, 24, 24, 128, 2048, 4096, 0.0, "padding_causal_bottom_right", "no_bias"
    ),
    "mask_9_0": ModelConfig(2, 16, 16, 128, 1, 2048, 0.0, "causal", "no_bias"),
    "mask_9_1": ModelConfig(2, 16, 16, 256, 1, 2048, 0.0, "causal", "no_bias"),
    "mask_10_0": ModelConfig(2, 16, 16, 128, 1, 2048, 0.0, "causal_bottom_right", "no_bias"),
    "mask_10_1": ModelConfig(2, 16, 16, 256, 1, 2048, 0.0, "causal_bottom_right", "no_bias"),
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
    "bias_1_0": ModelConfig(4, 16, 16, 64, 128, 128, 0.0, "no_mask", "post_scale_bias"),
    "bias_1_1": ModelConfig(2, 16, 16, 64, 128, 256, 0.0, "no_mask", "post_scale_bias"),
    "bias_1_2": ModelConfig(4, 24, 24, 128, 2048, 2048, 0.0, "no_mask", "post_scale_bias"),
    "bias_1_3": ModelConfig(2, 24, 24, 128, 2048, 4096, 0.0, "no_mask", "post_scale_bias"),
    "bias_1_4": ModelConfig(4, 24, 24, 128, 2048, 2048, 0.0, "no_mask", "alibi"),  # skipped
    "bias_1_5": ModelConfig(2, 24, 24, 128, 2048, 4096, 0.0, "no_mask", "alibi"),  # skipped
    "bias_2_0": ModelConfig(4, 16, 16, 64, 128, 128, 0.0, "padding", "post_scale_bias"),  # skipped
    "bias_2_1": ModelConfig(2, 16, 16, 64, 128, 256, 0.0, "padding", "post_scale_bias"),  # skipped
    "bias_2_2": ModelConfig(
        4, 24, 24, 128, 2048, 2048, 0.0, "padding", "post_scale_bias"
    ),  # skipped
    "bias_2_3": ModelConfig(
        2, 24, 24, 128, 2048, 4096, 0.0, "padding", "post_scale_bias"
    ),  # skipped
    "bias_2_4": ModelConfig(4, 24, 24, 128, 2048, 2048, 0.0, "padding", "alibi"),  # skipped
    "bias_2_5": ModelConfig(2, 24, 24, 128, 2048, 4096, 0.0, "padding", "alibi"),  # skipped
    "bias_3_0": ModelConfig(4, 16, 16, 64, 128, 128, 0.0, "causal", "post_scale_bias"),
    "bias_3_1": ModelConfig(2, 16, 16, 64, 128, 256, 0.0, "causal", "post_scale_bias"),
    "bias_3_2": ModelConfig(4, 24, 24, 128, 2048, 2048, 0.0, "causal", "post_scale_bias"),
    "bias_3_3": ModelConfig(
        2, 24, 24, 128, 2048, 4096, 0.0, "causal", "post_scale_bias"
    ),  # skipped
    "bias_3_4": ModelConfig(4, 24, 24, 128, 2048, 2048, 0.0, "causal", "alibi"),
    "bias_3_5": ModelConfig(2, 24, 24, 128, 2048, 4096, 0.0, "causal", "alibi"),  # skipped
    "bias_4_0": ModelConfig(
        4, 16, 16, 64, 128, 128, 0.0, "padding_causal", "post_scale_bias"
    ),  # skipped
    "bias_4_1": ModelConfig(
        2, 16, 16, 64, 128, 256, 0.0, "padding_causal", "post_scale_bias"
    ),  # skipped
    "bias_4_2": ModelConfig(
        4, 24, 24, 128, 2048, 2048, 0.0, "padding_causal", "post_scale_bias"
    ),  # skipped
    "bias_4_3": ModelConfig(
        2, 24, 24, 128, 2048, 4096, 0.0, "padding_causal", "post_scale_bias"
    ),  # skipped
    "bias_4_4": ModelConfig(4, 24, 24, 128, 2048, 2048, 0.0, "padding_causal", "alibi"),  # skipped
    "bias_4_5": ModelConfig(2, 24, 24, 128, 2048, 4096, 0.0, "padding_causal", "alibi"),  # skipped
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
    "bias_1_0": ModelConfig(
        4,
        16,
        16,
        64,
        128,
        128,
        0.0,
        #        mask,                     bias,       bias_shape,
        "no_mask",
        "post_scale_bias",
        bias_shape="11ss",
    ),
    "bias_1_1": ModelConfig(
        2, 16, 16, 64, 128, 128, 0.0, "no_mask", "post_scale_bias", bias_shape="1hss"
    ),
    "bias_1_2": ModelConfig(
        4, 24, 24, 128, 2048, 2048, 0.0, "no_mask", "post_scale_bias", bias_shape="b1ss"
    ),
    "bias_1_3": ModelConfig(
        2, 24, 24, 128, 2048, 2048, 0.0, "no_mask", "post_scale_bias", bias_shape="bhss"
    ),
    "bias_1_4": ModelConfig(
        4, 24, 24, 128, 2048, 2048, 0.0, "causal", "alibi", bias_shape="1hss", alibi_type="custom"
    ),
    "bias_1_5": ModelConfig(
        2, 24, 24, 128, 2048, 2048, 0.0, "causal", "alibi", bias_shape="bhss", alibi_type="custom"
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
    "swa_1_0": ModelConfig(4, 16, 16, 64, 128, 128, 0.0, "no_mask", "no_bias"),
    "swa_1_1": ModelConfig(2, 16, 16, 64, 128, 256, 0.0, "no_mask", "no_bias"),
    "swa_1_2": ModelConfig(4, 24, 24, 128, 2048, 2048, 0.0, "no_mask", "no_bias"),
    "swa_1_3": ModelConfig(2, 24, 24, 128, 2048, 4096, 0.0, "no_mask", "no_bias"),
    "swa_2_0": ModelConfig(4, 16, 16, 64, 128, 128, 0.0, "causal", "no_bias"),
    "swa_2_1": ModelConfig(2, 16, 16, 64, 128, 256, 0.0, "causal", "no_bias"),
    "swa_2_2": ModelConfig(4, 24, 24, 128, 2048, 2048, 0.0, "causal", "no_bias"),
    "swa_2_3": ModelConfig(2, 24, 24, 128, 2048, 4096, 0.0, "causal", "no_bias"),
    "swa_3_0": ModelConfig(4, 16, 16, 64, 128, 128, 0.0, "causal_bottom_right", "no_bias"),
    "swa_3_1": ModelConfig(2, 16, 16, 64, 128, 256, 0.0, "causal_bottom_right", "no_bias"),
    "swa_3_2": ModelConfig(4, 24, 24, 128, 2048, 2048, 0.0, "causal_bottom_right", "no_bias"),
    "swa_3_3": ModelConfig(2, 24, 24, 128, 2048, 4096, 0.0, "causal_bottom_right", "no_bias"),
}


@pytest.mark.skipif(not _flash_attn_2_3_plus, reason="Flash-attn 2.3+ is required.")
@pytest.mark.parametrize("dtype", param_types_lean)
@pytest.mark.parametrize("model_configs", [model_configs_swa])
@pytest.mark.parametrize("model", model_configs_swa.keys())
def test_dpa_sliding_window(dtype, model_configs, model):
    """Test DotProductAttention module with sliding window attention"""
    test_dot_product_attention(dtype, model_configs, model, False, True, None, True, False)


model_configs_alibi_slopes = {
    #     test:             b,  h, hg,   d,   sq,  skv,   p,      mask,    bias, alibi_type
    "alibi_1_0": ModelConfig(2, 16, 16, 64, 128, 128, 0.0, "causal", "alibi", alibi_type="vanilla"),
    "alibi_1_1": ModelConfig(1, 16, 16, 64, 128, 256, 0.0, "causal", "alibi", alibi_type="vanilla"),
    "alibi_2_0": ModelConfig(
        2, 24, 24, 128, 1024, 1024, 0.0, "causal", "alibi", alibi_type="custom"
    ),
    "alibi_2_1": ModelConfig(
        1, 24, 24, 128, 1024, 2048, 0.0, "causal", "alibi", alibi_type="custom"
    ),
}


@pytest.mark.skipif(not _flash_attn_2_3_plus, reason="Flash-attn 2.3+ is required.")
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
    "layout_0_0": ModelConfig(2, 16, 16, 64, 128, 128, 0.0, "no_mask", "no_bias"),
    "layout_0_1": ModelConfig(2, 16, 16, 64, 128, 128, 0.0, "causal", "post_scale_bias"),
    "layout_0_2": ModelConfig(1, 16, 16, 64, 128, 256, 0.0, "padding", "no_bias"),
    "layout_0_3": ModelConfig(1, 16, 16, 64, 128, 256, 0.0, "padding_causal", "post_scale_bias"),
    "layout_1_0": ModelConfig(2, 24, 24, 128, 2048, 2048, 0.0, "no_mask", "no_bias"),
    "layout_1_1": ModelConfig(2, 24, 24, 128, 2048, 2048, 0.0, "causal", "post_scale_bias"),
    "layout_1_2": ModelConfig(1, 24, 24, 128, 2048, 4096, 0.0, "padding", "no_bias"),
    "layout_1_3": ModelConfig(1, 24, 24, 128, 2048, 4096, 0.0, "padding_causal", "post_scale_bias"),
    "layout_2_0": ModelConfig(2, 16, 16, 256, 1, 2048, 0.0, "no_mask", "no_bias"),
    "layout_2_1": ModelConfig(2, 24, 24, 256, 2048, 2048, 0.0, "causal", "post_scale_bias"),
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
    "layout_0_1": ModelConfig(1, 16, 16, 64, 128, 128, 0.0, "padding", "no_bias"),
    "layout_0_2": ModelConfig(8, 16, 16, 64, 128, 128, 0.0, "padding", "no_bias"),
    "layout_0_3": ModelConfig(1, 16, 16, 64, 128, 128, 0.0, "padding_causal", "no_bias"),
    "layout_0_4": ModelConfig(8, 16, 16, 64, 128, 128, 0.0, "padding_causal", "no_bias"),
    "layout_1_1": ModelConfig(1, 16, 16, 64, 2048, 2048, 0.0, "padding", "no_bias"),
    "layout_1_2": ModelConfig(8, 16, 16, 64, 2048, 2048, 0.0, "padding", "no_bias"),
    "layout_1_3": ModelConfig(1, 16, 16, 64, 2048, 2048, 0.0, "padding_causal", "no_bias"),
    "layout_1_4": ModelConfig(8, 16, 16, 64, 2048, 2048, 0.0, "padding_causal", "no_bias"),
    "layout_2_1": ModelConfig(1, 16, 16, 128, 128, 128, 0.0, "padding", "no_bias"),
    "layout_2_2": ModelConfig(1, 16, 16, 64, 128, 256, 0.0, "padding", "no_bias"),
    "layout_2_3": ModelConfig(1, 16, 16, 128, 2048, 2048, 0.0, "padding_causal", "no_bias"),
    "layout_2_4": ModelConfig(8, 16, 16, 64, 2048, 4096, 0.0, "padding_causal", "no_bias"),
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
    pad_between_seqs = True
    test_dot_product_attention(
        dtype, model_configs, model, False, True, qkv_layout, False, pad_between_seqs
    )
    if get_cudnn_version() >= (9, 3, 0):
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
    "te_1_0": ModelConfig(2, 16, 16, 64, 128, 128, 0.0, "no_mask", "post_scale_bias"),
    "te_1_1": ModelConfig(4, 16, 16, 64, 128, 128, 0.0, "causal", "post_scale_bias"),
    "te_1_2": ModelConfig(2, 16, 16, 64, 128, 128, 0.0, "padding", "post_scale_bias"),
    "te_2_0": ModelConfig(1, 16, 16, 64, 2048, 2048, 0.0, "causal", "no_bias"),
    "te_2_1": ModelConfig(2, 16, 16, 64, 2048, 2048, 0.0, "no_mask", "no_bias"),
    "te_2_2": ModelConfig(1, 16, 16, 64, 2048, 2048, 0.0, "padding", "no_bias"),
    "te_3_0": ModelConfig(4, 16, 16, 64, 128, 128, 0.0, "causal", "alibi"),
    "te_3_1": ModelConfig(4, 16, 16, 64, 2048, 2048, 0.0, "causal", "alibi"),
}


@pytest.mark.skipif(get_cudnn_version() < (8, 9, 1), reason="cuDNN 8.9.1+ is required.")
@pytest.mark.parametrize("dtype", param_types)
@pytest.mark.parametrize("model_configs", [model_configs_te_layer])
@pytest.mark.parametrize("model", model_configs_te_layer.keys())
@pytest.mark.parametrize("ckpt_attn", [False])
@pytest.mark.parametrize("qkv_format", ["sbhd"])
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
    available_backends, fused_attn_backends = _get_attention_backends(
        config,
        qkv_dtype=dtype,
        qkv_layout="sbh3d" if fused_qkv_params else "sb3hd",
    )
    flash_attn_supported, fused_attn_supported, unfused_attn_supported = available_backends

    # Skip if only unfused backend is supported
    if (len(fused_attn_backends) + flash_attn_supported + unfused_attn_supported) < 2:
        pytest.skip("Less than two backends to compare.")

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
        )

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
    inp = torch.randn(
        config.max_seqlen_q,
        config.batch_size,
        config.hidden_size,
        dtype=dtype,
        device="cuda",
        requires_grad=True,
    )
    # In case the format to be tested is batch-first, need to transpose the
    # input tensor.
    if qkv_format == "bshd":
        inp = inp.transpose(0, 1)

    # Create seqlens
    if "padding" in config.attn_mask_type:
        seqlens_q = torch.randint(
            1, config.max_seqlen_q, [config.batch_size], dtype=torch.int32, device="cuda"
        )
    else:
        seqlens_q = torch.full(
            [config.batch_size], config.max_seqlen_q, dtype=torch.int32, device="cuda"
        )

    # Create attention mask if padding
    attention_mask = None
    if "padding" in config.attn_mask_type:
        attention_mask_q = torch.Tensor([]).to(dtype=torch.bool)
        for i in range(config.batch_size):
            attention_mask_q = torch.cat(
                [
                    attention_mask_q,
                    torch.Tensor(
                        [False] * seqlens_q[i] + [True] * (config.max_seqlen_q - seqlens_q[i])
                    )
                    .to(torch.bool)
                    .unsqueeze(0)
                    .unsqueeze(0)
                    .unsqueeze(0),
                ],
                dim=0,
            )
        attention_mask = attention_mask_q.to(device="cuda")

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
        layer_type="encoder",
        drop_path_rate=drop_path_rates[layer_number - 1],
        set_parallel_mode=True,
        fuse_qkv_params=fused_qkv_params,
        zero_centered_gamma=False,
        qkv_weight_interleaved=False,
        ub_tp_comm_overlap=False,
        bias=True,
        attn_input_format=qkv_format,
    ).to(dtype=dtype, device="cuda")

    # Create ALiBi slopes
    alibi_slopes = None
    if config.attn_bias_type == "alibi" and config.alibi_type == "custom":
        alibi_slopes = torch.randn(config.num_heads).abs().to(dtype=torch.float32, device="cuda")

    # Run a forward and backward pass
    out = block(
        inp,
        attention_mask=attention_mask,
        self_attn_mask_type=config.attn_mask_type,
        checkpoint_core_attention=False,
        rotary_pos_emb=rotary_pos_emb,
        core_attention_bias_type=config.attn_bias_type,
        core_attention_bias=bias,
        alibi_slopes=alibi_slopes,
    )
    loss = out.sum()
    loss.backward()

    return out, inp.grad


model_configs_fp8_vs_f16 = {
    #  test:             b,  h, hg,   d,   sq,  skv,   p,      mask,      bias
    "fp8_9": ModelConfig(2, 24, 24, 128, 2048, 2048, 0.0, "no_mask", "no_bias"),
    "fp8_10": ModelConfig(2, 24, 24, 128, 2048, 2048, 0.0, "causal", "no_bias"),
    "fp8_11": ModelConfig(2, 24, 12, 128, 2048, 2048, 0.0, "no_mask", "no_bias"),
    "fp8_12": ModelConfig(2, 24, 12, 128, 2048, 2048, 0.0, "causal", "no_bias"),
    "fp8_13": ModelConfig(1, 32, 4, 128, 8192, 8192, 0.0, "no_mask", "no_bias"),
    "fp8_14": ModelConfig(1, 32, 4, 128, 8192, 8192, 0.0, "causal", "no_bias"),
    "fp8_15": ModelConfig(1, 16, 16, 128, 2048, 2048, 0.0, "no_mask", "no_bias"),
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

    if _flash_attn_3_is_installed and not is_training:
        if RoPE:
            pytest.skip("Flash Attention doesn't support FP8 MHA with RoPE.")
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
    if _flash_attn_3_is_installed and not is_training:
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

    with fp8_model_init(enabled=fp8_mha):
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

    if config.num_heads != config.num_gqa_groups and "3" in qkv_layout:
        pytest.skip("qkv_layout not applicable for MQA/GQA")

    os.environ["NVTE_FP8_DPA_BWD"] = "1" if fp8_dpa_bwd else "0"
    os.environ["NVTE_ALLOW_NONDETERMINISTIC_ALGO"] = "1"

    if _flash_attn_3_is_installed and not is_training:
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

    logging.info("[test_dpa_fp8_vs_f16]: run with fp8_dpa = False")
    fused_attn_fwd_f16, fused_attn_bwd_f16 = _run_dpa_fp8_vs_f16(
        dtype, config, False, qkv_layout, is_training
    )

    atol = 5e-1
    rtol = 5e-2
    rmse_tol = 0.1
    bwd_names = ["dq", "dk", "dv"]
    logging.debug("========== {:^25s} ==========".format("forward output"))
    if _flash_attn_3_is_installed and not is_training:
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
        tensor = torch.randn(tensor_shape, dtype=dtype, device="cuda")
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
            is_first_microbatch=True,
        )
        if is_training:
            out.backward(out_grad)

    if is_training:
        return out, (inp[0].grad, inp[1].grad, inp[2].grad)
    return out, (None, None, None)


model_configs_fp8 = {
    #  test:             b,  h, hg,   d,   sq,  skv,   p,      mask,      bias
    "fp8_1": ModelConfig(1, 1, 1, 64, 512, 512, 0.0, "no_mask", "no_bias"),
    "fp8_2": ModelConfig(4, 16, 16, 64, 512, 512, 0.0, "no_mask", "no_bias"),
    "fp8_3": ModelConfig(1, 1, 1, 128, 2048, 2048, 0.0, "no_mask", "no_bias"),
    "fp8_4": ModelConfig(2, 24, 24, 128, 2048, 2048, 0.0, "no_mask", "no_bias"),
    "fp8_5": ModelConfig(1, 1, 1, 64, 512, 512, 0.0, "causal", "no_bias"),
    "fp8_6": ModelConfig(4, 16, 16, 64, 512, 512, 0.0, "causal", "no_bias"),
    "fp8_7": ModelConfig(1, 1, 1, 128, 2048, 2048, 0.0, "causal", "no_bias"),
    "fp8_8": ModelConfig(2, 24, 24, 128, 2048, 2048, 0.0, "causal", "no_bias"),
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

    fused_attn_fwd_fp8, fused_attn_bwd_fp8 = _run_custom_mha_fp8(dtype, config, "FusedAttention")
    unfused_attn_fwd_f16, unfused_attn_bwd_f16 = _run_ref_mha_f16(dtype, config, "UnfusedAttention")

    atol = 5e-1
    rtol = 5e-1
    rmse_tol = 0.1
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
    ) -> torch.Tensor:

        assert inp.dim() == 2
        in_features = qkv_weight.shape[-1]
        h = num_heads
        d = in_features // h
        b = cu_seqlens.numel() - 1

        fp8_dtype_forward = fp8.get_fp8_te_dtype(fp8_meta["recipe"], fprop_tensor=True)

        inp_fp8, inp_t_fp8 = ext.fp8_cast_transpose_fused(
            inp,
            fp8_meta["scaling_fwd"],
            tex.FP8FwdTensors.GEMM1_INPUT,
            fp8_dtype_forward,
        )

        qkv_weight_fp8, qkv_weight_t_fp8 = ext.fp8_cast_transpose_fused(
            qkv_weight,
            fp8_meta["scaling_fwd"],
            tex.FP8FwdTensors.GEMM1_WEIGHT,
            fp8_dtype_forward,
        )

        M = None
        ZInv = None
        philox_unpacked = None

        qkv, _ = ext.fp8_gemm(
            qkv_weight_fp8,
            fp8_meta["scaling_fwd"].scale_inv,
            tex.FP8FwdTensors.GEMM1_WEIGHT,
            fp8_dtype_forward,
            inp_fp8,
            fp8_meta["scaling_fwd"].scale_inv,
            tex.FP8FwdTensors.GEMM1_INPUT,
            fp8_dtype_forward,
            torch.uint8,
            workspace,
            bias=qkv_bias,
            use_bias=True,
            out_index=META_QKV,
            fp8_meta_tensor=fp8_meta["scaling_fwd"],
            use_split_accumulator=_2X_ACC_FPROP,
            D_dtype=fp8_dtype_forward,
        )
        qkv = qkv.view(-1, 3, h, d)
        qkv_fp16 = (
            ext.cast_from_fp8(
                qkv, fp8_meta["scaling_fwd"], META_QKV, fp8_dtype_forward, tex.DType.kFloat16
            )
            .view(b, max_s, 3, h, d)
            .contiguous()
        )
        torch.save(qkv_fp16, "qkv.pt")
        if cudnn_frontend_version == 1:
            qkv = qkv.view(b, max_s, 3, h, d)  # bs3hd

        # FMHA
        out, aux_ctx_tensors, *rest = fused_attn_fwd(
            is_training,
            max_s,
            max_s,
            cu_seqlens,
            cu_seqlens,
            qkv[:, :, 0, :, :] if cudnn_frontend_version == 1 else qkv[:, 0, :, :],
            qkv[:, :, 1, :, :] if cudnn_frontend_version == 1 else qkv[:, 1, :, :],
            qkv[:, :, 2, :, :] if cudnn_frontend_version == 1 else qkv[:, 2, :, :],
            fp8_dtype_forward,
            FusedAttnBackend["FP8"],
            None,
            None,
            None,
            fp8_meta["scaling_fwd"].scale_inv,  # d_scale_qkv
            META_QKV,  # d_scale_qkv_offset
            fp8_meta["scaling_fwd"].scale_inv,  # d_scale_s
            META_S,  # d_scale_s_offset
            fp8_meta["scaling_fwd"].scale,  # q_scale_s
            META_S,  # q_scale_s_offset
            fp8_meta["scaling_fwd"].scale,  # q_scale_o
            META_O,  # q_scale_o_offset
            fp8_meta["scaling_fwd"].amax_history,  # amax_s
            META_S,  # amax_s_offset
            fp8_meta["scaling_fwd"].amax_history,  # amax_o
            META_O,  # amax_o_offset
            attn_scale=None,
            dropout=p_dropout,
            fast_zero_fill=fast_zero_fill,
            qkv_layout="bs3hd" if cudnn_frontend_version == 1 else "t3hd",
            attn_bias_type="no_bias",
            attn_mask_type=mask_type if cudnn_frontend_version == 1 else "padding",
            rng_gen=None,
        )

        M, ZInv, philox_unpacked = aux_ctx_tensors

        ctx.save_for_backward(
            inp_t_fp8,
            qkv_weight_t_fp8,
            workspace,
            qkv,
            out,
            fp8_meta["scaling_fwd"].scale,
            fp8_meta["scaling_fwd"].scale_inv,
        )
        ctx.aux_ctx_tensors = aux_ctx_tensors
        ctx.fp8_meta = fp8_meta
        ctx.cu_seqlens = cu_seqlens
        ctx.p_dropout = p_dropout
        ctx.max_s = max_s
        ctx.fast_zero_fill = fast_zero_fill
        ctx.hidden_size = in_features
        ctx.num_heads = num_heads
        ctx.mask_type = mask_type
        ctx.dtype = inp.dtype

        out = out.view(-1, in_features)  # (bs)(hd)
        out_fp16 = ext.cast_from_fp8(
            out, fp8_meta["scaling_fwd"], META_O, fp8_dtype_forward, tex.DType.kFloat16
        )
        torch.save(out_fp16, "out.pt")  # (bs)(hd)
        return out_fp16

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[Union[torch.Tensor, None], ...]:
        with torch.cuda.nvtx.range("_DPA"):
            (
                inp_t_fp8,
                qkv_weight_t_fp8,
                workspace,
                qkv,
                out,
                fwd_scales,
                fwd_scale_inverses,
            ) = ctx.saved_tensors
            fp8_dtype_forward = fp8.get_fp8_te_dtype(ctx.fp8_meta["recipe"], fprop_tensor=True)
            fp8_dtype_backward = fp8.get_fp8_te_dtype(ctx.fp8_meta["recipe"], fprop_tensor=False)

            proj_dgrad = ext.cast_to_fp8(
                grad_output, ctx.fp8_meta["scaling_bwd"], META_DO, fp8_dtype_backward
            )  # (bs)(hd)

            dq, dk, dv, *rest = fused_attn_bwd(
                ctx.max_s,
                ctx.max_s,
                ctx.cu_seqlens,
                ctx.cu_seqlens,
                qkv[:, :, 0, :, :] if cudnn_frontend_version == 1 else qkv[:, 0, :, :],
                qkv[:, :, 1, :, :] if cudnn_frontend_version == 1 else qkv[:, 1, :, :],
                qkv[:, :, 2, :, :] if cudnn_frontend_version == 1 else qkv[:, 2, :, :],
                out,
                proj_dgrad.view_as(out),
                fp8_dtype_forward,
                fp8_dtype_backward,
                ctx.aux_ctx_tensors,
                FusedAttnBackend["FP8"],
                None,
                None,
                fwd_scale_inverses[META_QKV],  # d_scale_qkv,
                fwd_scale_inverses[META_S],  # d_scale_s,
                fwd_scale_inverses[META_O],  # d_scale_o,
                ctx.fp8_meta["scaling_bwd"].scale_inv[META_DO],  # d_scale_do
                ctx.fp8_meta["scaling_bwd"].scale_inv[META_DP],  # d_scale_dp
                fwd_scales[META_S],  # q_scale_s
                ctx.fp8_meta["scaling_bwd"].scale[META_DP],  # q_scale_dp
                ctx.fp8_meta["scaling_bwd"].scale[META_DQKV],  # q_scale_dqkv
                ctx.fp8_meta["scaling_bwd"].amax_history[0][META_DP],  # amax_dp
                ctx.fp8_meta["scaling_bwd"].amax_history[0][META_DQKV],  # amax_dqkv
                attn_scale=None,
                dropout=ctx.p_dropout,
                fast_zero_fill=ctx.fast_zero_fill,
                qkv_layout="bs3hd" if cudnn_frontend_version == 1 else "t3hd",
                attn_bias_type="no_bias",
                attn_mask_type=ctx.mask_type if cudnn_frontend_version == 1 else "padding",
            )
            dim = 2 if cudnn_frontend_version == 1 else 1
            dqkv = torch.Tensor().to(device=dq.device, dtype=dq.dtype)
            dqkv_shape = list(dq.shape)
            dqkv_shape.insert(dim, 3)
            dqkv_stride = list(dq.stride())
            dqkv_stride.insert(dim, int(dqkv_stride[-3] / 3))
            dqkv.set_(dq.untyped_storage(), dq.storage_offset(), dqkv_shape, dqkv_stride)  # bs3hd

            dqkv_c = dqkv.view(-1, 3 * ctx.hidden_size)
            dqkv_c_fp16 = ext.cast_from_fp8(
                dqkv_c,
                ctx.fp8_meta["scaling_bwd"],
                META_DQKV,
                fp8_dtype_backward,
                tex.DType.kFloat16,
            )
            torch.save(dqkv_c_fp16, "dqkv.pt")

            qkv_bgrad, dqkv_t = ext.fp8_transpose_bgrad_fused(
                dqkv_c,
                ctx.fp8_meta["scaling_bwd"],
                META_DQKV,
                fp8_dtype_backward,
                ctx.dtype,
            )

            # QKV DGRAD
            qkv_dgrad, _ = ext.fp8_gemm(
                qkv_weight_t_fp8,
                fwd_scale_inverses,
                tex.FP8FwdTensors.GEMM1_WEIGHT,
                fp8_dtype_forward,
                dqkv_c,
                ctx.fp8_meta["scaling_bwd"].scale_inv,
                META_DQKV,
                fp8_dtype_backward,
                ctx.dtype,
                workspace,
                use_split_accumulator=_2X_ACC_DGRAD,
            )
            # QKV WGRAD
            qkv_wgrad, _ = ext.fp8_gemm(
                inp_t_fp8,
                fwd_scale_inverses,
                tex.FP8FwdTensors.GEMM1_INPUT,
                fp8_dtype_forward,
                dqkv_t,
                ctx.fp8_meta["scaling_bwd"].scale_inv,
                META_DQKV,
                fp8_dtype_backward,
                ctx.dtype,
                workspace,
                use_split_accumulator=_2X_ACC_WGRAD,
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
        with self.prepare_forward(inp, None, num_gemms=3) as inp:
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
            )
        return out
