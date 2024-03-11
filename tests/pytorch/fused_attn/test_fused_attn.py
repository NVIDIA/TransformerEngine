# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import functools
from importlib.metadata import version
import os
import math
from typing import Any, Dict, List, Tuple, Union

from pkg_resources import packaging
import pytest
import torch

from transformer_engine.common import recipe
from transformer_engine.pytorch import TransformerLayer, fp8_autocast
from transformer_engine.pytorch.attention import (
    DotProductAttention,
    RotaryPositionEmbedding,
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
from transformer_engine.pytorch.distributed import (
    _set_cuda_rng_state,
    CudaRNGStatesTracker,
)
import transformer_engine.pytorch.fp8 as fp8
from transformer_engine.pytorch.module.base import (
    TransformerEngineBaseModule,
    _prepare_backward,
)
from transformer_engine.pytorch.utils import (
    get_device_compute_capability,
    init_method_normal,
    scaled_init_method_normal,
    is_bf16_compatible,
)
import transformer_engine_extensions as tex
from transformer_engine_extensions import NVTE_Fused_Attn_Backend

# Only run FP8 tests on H100
fp8_available, reason_for_no_fp8 = fp8.FP8GlobalStateManager.is_fp8_available()

# Initialize RNG state
seed = 1234
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
_cpu_rng_state = torch.get_rng_state()
_cuda_rng_state = torch.cuda.get_rng_state()

_NVTE_DEBUG = int(os.getenv("NVTE_DEBUG", "0"))

def reset_rng_states() -> None:
    """Revert back to initial RNG state"""
    torch.set_rng_state(_cpu_rng_state)
    _set_cuda_rng_state(_cuda_rng_state)

@functools.cache
def _cudnn_version() -> Tuple[int, int, int]:
    """Runtime cuDNN version (major, minor, patch)"""
    encoded_version = ext.get_cudnn_version()
    major, encoded_version = divmod(encoded_version, 1000)
    minor, patch = divmod(encoded_version, 100)
    return (major, minor, patch)

class ModelConfig:
    def __init__(
        self,
        batch_size: int,
        num_heads: int,
        num_gqa_groups: int,
        head_dim: int,
        max_seqlen_q: int,
        max_seqlen_kv: int,
        dropout_p: float,
        attn_mask_type: str,
        attn_bias_type: str,
        alibi_type: str = "none",
        num_layers: int = 1,
        bias_shape: str = "1hss",
    ):
        self.batch_size = batch_size
        self.num_heads = num_heads
        self.num_gqa_groups = num_gqa_groups
        self.head_dim = head_dim
        self.hidden_size = num_heads * head_dim
        self.hidden_size_kv = num_gqa_groups * head_dim
        self.max_seqlen_q = max_seqlen_q
        self.max_seqlen_kv = max_seqlen_kv
        self.dropout_p = dropout_p
        self.attn_mask_type  = attn_mask_type
        self.attn_bias_type  = attn_bias_type
        self.alibi_type  = alibi_type
        self.attn_type  = "self" if (max_seqlen_q == max_seqlen_kv) else "cross"
        self.num_layers = num_layers
        self.bias_shape = bias_shape

def _is_fused_attention_supported(
    config: ModelConfig,
    dtype: torch.dtype,
    qkv_layout: str = "sbh3d",
) -> Tuple[bool, NVTE_Fused_Attn_Backend]:
    """Check if FusedAttention supports a model configuration"""
    backends = []
    os.environ["NVTE_FUSED_ATTN_BACKEND"] = "0"
    backend = tex.get_fused_attn_backend(
        TE_DType[dtype],
        TE_DType[dtype],
        QKVLayout[qkv_layout],
        AttnBiasType[config.attn_bias_type],
        AttnMaskType[config.attn_mask_type],
        config.dropout_p,
        config.num_heads,
        config.num_gqa_groups,
        config.max_seqlen_q,
        config.max_seqlen_kv,
        config.head_dim,
    )
    if backend == FusedAttnBackend["FP8"]:
        backends.append(backend)
        return True, backends
    if backend == FusedAttnBackend["F16_arbitrary_seqlen"]:
        backends.append(backend)
        return True, backends
    if backend == FusedAttnBackend["F16_max512_seqlen"]:
        backends.append(backend)
        os.environ["NVTE_FUSED_ATTN_BACKEND"] = "1"
        backend = tex.get_fused_attn_backend(
            TE_DType[dtype],
            TE_DType[dtype],
            QKVLayout[qkv_layout],
            AttnBiasType[config.attn_bias_type],
            AttnMaskType[config.attn_mask_type],
            config.dropout_p,
            config.num_heads,
            config.num_gqa_groups,
            config.max_seqlen_q,
            config.max_seqlen_kv,
            config.head_dim,
        )
        if backend == FusedAttnBackend["F16_arbitrary_seqlen"]:
            backends.append(backend)
        return True, backends
    return False, backends

@functools.cache
def _is_flash_attention_2_available() -> bool:
    """Check if flash-attn 2.0+ is available"""
    Version = packaging.version.Version
    return Version(version("flash-attn")) >= Version("2")

@functools.cache
def _is_flash_attention_2_1() -> bool:
    """Check if flash-attn 2.1+ is available"""
    Version = packaging.version.Version
    return Version(version("flash-attn")) >= Version("2.1")

@functools.cache
def _is_flash_attention_2_3() -> bool:
    """Check if flash-attn 2.3+ is available"""
    Version = packaging.version.Version
    return Version(version("flash-attn")) >= Version("2.3")

def _is_flash_attention_supported(config: ModelConfig) -> bool:
    """Check if FlashAttention supports a model configuration"""
    if get_device_compute_capability() < (8, 0):
        return False
    if config.attn_bias_type not in ["no_bias", "alibi"]:
        return False
    if config.num_heads != config.num_gqa_groups and not _is_flash_attention_2_available():
        return False
    if "causal" in config.attn_mask_type and config.attn_type == "cross":
        if _is_flash_attention_2_1():
            # FAv2.1 implements causal mask for cross attention differently
            # https://github.com/Dao-AILab/flash-attention#21-change-behavior-of-causal-flag
            return False
    return True

def _is_unfused_attention_supported(config: ModelConfig) -> bool:
    """Check if UnfusedDotProductAttention supports a model configuration"""
    if ("padding" in config.attn_mask_type):
        return False
    if ("causal" in config.attn_mask_type and config.attn_type == 'cross'):
        return False
    return True

model_configs_base = {
    #     test:             b,  h, hg,   d,   sq,  skv,   p,      mask,      bias   # attn , backend
    "base_1_0": ModelConfig(8, 16, 16,  64,  128,  128, 0.0, "no_mask", "no_bias"), # self , 0
    "base_1_1": ModelConfig(4, 16, 16,  64,  128,  256, 0.0, "no_mask", "no_bias"), # cross, 0
    "base_2_0": ModelConfig(2, 24, 24, 128, 2048, 2048, 0.0, "no_mask", "no_bias"), # self , 1
    "base_2_1": ModelConfig(1, 24, 24, 128, 2048, 4096, 0.0, "no_mask", "no_bias"), # cross, 1
}

param_types = [torch.float16]
if is_bf16_compatible():  # bf16 requires sm_80 or higher
    param_types.append(torch.bfloat16)
param_types_lean = [torch.bfloat16]

def get_swa(seq_q, seq_kv, w=None):
    """Generate a random sliding window size (left, right) if w is None,
    and create its equivalent attention mask in [seq_q, seq_kv] shape"""
    if w is None:
        w = torch.randint(0, seq_kv, [2], dtype=torch.int32, device="cuda")
    m = torch.ones(seq_q, seq_kv, dtype=torch.bool, device="cuda")
    mu = torch.triu(m, diagonal=seq_kv-seq_q-w[0])
    ml = torch.tril(mu, diagonal=seq_kv-seq_q+w[1])
    ml = ~ ml
    return w, ml

@pytest.mark.skipif(_cudnn_version() < (8,9,1), reason="cuDNN 8.9.1+ is required.")
@pytest.mark.parametrize("dtype", param_types)
@pytest.mark.parametrize("model_configs", [model_configs_base])
@pytest.mark.parametrize("model", model_configs_base.keys())
@pytest.mark.parametrize("ckpt_attn", [False])
@pytest.mark.parametrize("workspace_opt", [True, False])
@pytest.mark.parametrize("qkv_layout", [None])
@pytest.mark.parametrize("swa", [False])
def test_dot_product_attention(dtype, model_configs, model, ckpt_attn, workspace_opt, qkv_layout, swa):
    """Test DotProductAttention module"""

    # Get configs
    tols = dict(atol=5e-3, rtol=5e-3)
    if dtype == torch.bfloat16:
        tols = dict(atol=2.5e-2, rtol=2.5e-2)
    config = model_configs[model]
    if qkv_layout is None:
        if config.attn_type == "self":
            qkv_layout = "sb3hd"
        else:
            qkv_layout = "sbhd_sb2hd"
    if "3" in qkv_layout and config.attn_type == "cross":
        pytest.skip(
            "No need to test this layout for cross attention"
        )

    # Skip if only unfused backend is supported
    unfused_attn_supported = _is_unfused_attention_supported(config)
    if config.max_seqlen_q <= 512 and config.max_seqlen_kv <= 512:
        os.environ["NVTE_FUSED_ATTN_BACKEND"] = "0"
    fused_attn_supported, fused_attn_backend = _is_fused_attention_supported(
        config, dtype, qkv_layout=qkv_layout,
    )
    if swa:
        fused_attn_supported = False
    flash_attn_supported = _is_flash_attention_supported(config)
    if (len(fused_attn_backend) + flash_attn_supported + unfused_attn_supported) < 2:
        pytest.skip("Less than two backends to compare.")

    # UnfusedDotProductAttention backend
    if unfused_attn_supported:
        if swa:
            attn_mask_type = config.attn_mask_type
            config.attn_mask_type = "arbitrary"
        unfused_attn_fwd, unfused_attn_bwd = _run_dot_product_attention(
            dtype, config, "UnfusedDotProductAttention", ckpt_attn, qkv_layout, workspace_opt, swa,
        )
        if swa:
            config.attn_mask_type = attn_mask_type

    # FusedAttention backend
    if fused_attn_supported:
        if len(fused_attn_backend) == 1:
            fused_attn_fwd, fused_attn_bwd = _run_dot_product_attention(
                dtype, config, "FusedAttention", ckpt_attn, qkv_layout, workspace_opt, swa,
            )
        if len(fused_attn_backend) == 2:
            os.environ["NVTE_FUSED_ATTN_BACKEND"] = "0"
            fused_attn_fwd, fused_attn_bwd = _run_dot_product_attention(
                dtype, config, "FusedAttention", ckpt_attn, qkv_layout, workspace_opt, swa,
            )
            os.environ["NVTE_FUSED_ATTN_BACKEND"] = "1"
            fused_attn_fwd_1, fused_attn_bwd_1 = _run_dot_product_attention(
                dtype, config, "FusedAttention", ckpt_attn, qkv_layout, workspace_opt, swa,
            )

    # FlashAttention backend
    if flash_attn_supported:
        flash_attn_fwd, flash_attn_bwd = _run_dot_product_attention(
            dtype, config, "FlashAttention", ckpt_attn, qkv_layout, workspace_opt, swa,
        )

    if unfused_attn_supported and fused_attn_supported:
        if _NVTE_DEBUG:
            print("[test_dot_product_attention]: unfused attn vs fused attn")
        torch.testing.assert_close(fused_attn_fwd, unfused_attn_fwd, **tols)
        for i,_ in enumerate(unfused_attn_bwd):
            torch.testing.assert_close(fused_attn_bwd[i], unfused_attn_bwd[i], **tols)
    if unfused_attn_supported and flash_attn_supported:
        if _NVTE_DEBUG:
            print("[test_dot_product_attention]: unfused attn vs flash attn")
        torch.testing.assert_close(flash_attn_fwd, unfused_attn_fwd, **tols)
        for i,_ in enumerate(flash_attn_bwd):
            torch.testing.assert_close(unfused_attn_bwd[i], flash_attn_bwd[i], **tols)
    if fused_attn_supported and flash_attn_supported:
        if _NVTE_DEBUG:
            print("[test_dot_product_attention]: fused attn vs flash attn")
        torch.testing.assert_close(fused_attn_fwd, flash_attn_fwd, **tols)
        for i,_ in enumerate(flash_attn_bwd):
            torch.testing.assert_close(fused_attn_bwd[i], flash_attn_bwd[i], **tols)
    if fused_attn_supported and len(fused_attn_backend) == 2:
        if _NVTE_DEBUG:
            print("[test_dot_product_attention]: fused attn backend 0 vs 1")
        torch.testing.assert_close(fused_attn_fwd, fused_attn_fwd_1, **tols)
        for i,_ in enumerate(fused_attn_bwd):
            torch.testing.assert_close(fused_attn_bwd[i], fused_attn_bwd_1[i], **tols)

@pytest.mark.skipif(_cudnn_version() < (8,9,1), reason="cuDNN 8.9.1+ is required.")
@pytest.mark.parametrize("dtype", param_types)
@pytest.mark.parametrize("model_configs", [model_configs_base])
@pytest.mark.parametrize("model", ["base_1_1", "base_2_1"])
def test_dpa_checkpoint(dtype, model_configs, model):
    """Test DotProductAttention module with checkpointing"""
    test_dot_product_attention(dtype, model_configs, model, True, True, None, False)

model_configs_mask = {
    #     test:             b,  h, hg,   d,   sq,  skv,   p,             mask,      bias
    "mask_1_0": ModelConfig(8, 16, 16,  64,  128,  128, 0.0,         "causal", "no_bias"),
    "mask_1_1": ModelConfig(4, 16, 16,  64,  128,  256, 0.0,         "causal", "no_bias"),
    "mask_2_0": ModelConfig(2, 24, 24, 128, 2048, 2048, 0.0,         "causal", "no_bias"),
    "mask_2_1": ModelConfig(1, 24, 24, 128, 2048, 4096, 0.0,         "causal", "no_bias"),
    "mask_3_0": ModelConfig(8, 16, 16,  64,  128,  128, 0.0,        "padding", "no_bias"),
    "mask_3_1": ModelConfig(4, 16, 16,  64,  128,  256, 0.0,        "padding", "no_bias"),
    "mask_4_0": ModelConfig(2, 24, 24, 128, 2048, 2048, 0.0,        "padding", "no_bias"),
    "mask_4_1": ModelConfig(1, 24, 24, 128, 2048, 4096, 0.0,        "padding", "no_bias"),
    "mask_5_0": ModelConfig(8, 16, 16,  64,  128,  128, 0.0, "padding_causal", "no_bias"),
    "mask_5_1": ModelConfig(4, 16, 16,  64,  128,  256, 0.0, "padding_causal", "no_bias"),
    "mask_6_0": ModelConfig(2, 24, 24, 128, 2048, 2048, 0.0, "padding_causal", "no_bias"),
    "mask_6_1": ModelConfig(1, 24, 24, 128, 2048, 4096, 0.0, "padding_causal", "no_bias"),
}

@pytest.mark.skipif(_cudnn_version() < (8,9,1), reason="cuDNN 8.9.1+ is required.")
@pytest.mark.parametrize("dtype", param_types_lean)
@pytest.mark.parametrize("model_configs", [model_configs_mask])
@pytest.mark.parametrize("model", model_configs_mask.keys())
def test_dpa_mask(dtype, model_configs, model):
    """Test DotProductAttention module with different mask types"""
    test_dot_product_attention(dtype, model_configs, model, False, True, None, False)

model_configs_bias = {
    #     test:             b,  h, hg,   d,   sq,  skv,   p,             mask,             bias
    "bias_1_0": ModelConfig(4, 16, 16,  64,  128,  128, 0.0,        "no_mask", "post_scale_bias"),
    "bias_1_1": ModelConfig(2, 16, 16,  64,  128,  256, 0.0,        "no_mask", "post_scale_bias"),
    "bias_1_2": ModelConfig(4, 24, 24, 128, 2048, 2048, 0.0,        "no_mask", "post_scale_bias"),
    "bias_1_3": ModelConfig(2, 24, 24, 128, 2048, 4096, 0.0,        "no_mask", "post_scale_bias"),
    "bias_1_4": ModelConfig(4, 24, 24, 128, 2048, 2048, 0.0,        "no_mask",           "alibi"), # skipped
    "bias_1_5": ModelConfig(2, 24, 24, 128, 2048, 4096, 0.0,        "no_mask",           "alibi"), # skipped
    "bias_2_0": ModelConfig(4, 16, 16,  64,  128,  128, 0.0,        "padding", "post_scale_bias"), # skipped
    "bias_2_1": ModelConfig(2, 16, 16,  64,  128,  256, 0.0,        "padding", "post_scale_bias"), # skipped
    "bias_2_2": ModelConfig(4, 24, 24, 128, 2048, 2048, 0.0,        "padding", "post_scale_bias"), # skipped
    "bias_2_3": ModelConfig(2, 24, 24, 128, 2048, 4096, 0.0,        "padding", "post_scale_bias"), # skipped
    "bias_2_4": ModelConfig(4, 24, 24, 128, 2048, 2048, 0.0,        "padding",           "alibi"), # skipped
    "bias_2_5": ModelConfig(2, 24, 24, 128, 2048, 4096, 0.0,        "padding",           "alibi"), # skipped
    "bias_3_0": ModelConfig(4, 16, 16,  64,  128,  128, 0.0,         "causal", "post_scale_bias"),
    "bias_3_1": ModelConfig(2, 16, 16,  64,  128,  256, 0.0,         "causal", "post_scale_bias"),
    "bias_3_2": ModelConfig(4, 24, 24, 128, 2048, 2048, 0.0,         "causal", "post_scale_bias"),
    "bias_3_3": ModelConfig(2, 24, 24, 128, 2048, 4096, 0.0,         "causal", "post_scale_bias"), # skipped
    "bias_3_4": ModelConfig(4, 24, 24, 128, 2048, 2048, 0.0,         "causal",           "alibi"),
    "bias_3_5": ModelConfig(2, 24, 24, 128, 2048, 4096, 0.0,         "causal",           "alibi"), # skipped
    "bias_4_0": ModelConfig(4, 16, 16,  64,  128,  128, 0.0, "padding_causal", "post_scale_bias"), # skipped
    "bias_4_1": ModelConfig(2, 16, 16,  64,  128,  256, 0.0, "padding_causal", "post_scale_bias"), # skipped
    "bias_4_2": ModelConfig(4, 24, 24, 128, 2048, 2048, 0.0, "padding_causal", "post_scale_bias"), # skipped
    "bias_4_3": ModelConfig(2, 24, 24, 128, 2048, 4096, 0.0, "padding_causal", "post_scale_bias"), # skipped
    "bias_4_4": ModelConfig(4, 24, 24, 128, 2048, 2048, 0.0, "padding_causal",           "alibi"), # skipped
    "bias_4_5": ModelConfig(2, 24, 24, 128, 2048, 4096, 0.0, "padding_causal",           "alibi"), # skipped
}

@pytest.mark.skipif(_cudnn_version() < (8,9,1), reason="cuDNN 8.9.1+ is required.")
@pytest.mark.parametrize("dtype", param_types_lean)
@pytest.mark.parametrize("model_configs", [model_configs_bias])
@pytest.mark.parametrize("model", model_configs_bias.keys())
def test_dpa_bias(dtype, model_configs, model):
    """Test DotProductAttention module with different bias types"""
    test_dot_product_attention(dtype, model_configs, model, False, True, None, False)

model_configs_bias_shapes = {
    #     test:             b,  h, hg,   d,   sq,  skv,   p,
    "bias_1_0": ModelConfig(4, 16, 16,  64,  128,  128, 0.0,
    #        mask,                     bias,       bias_shape,
        "no_mask",        "post_scale_bias", bias_shape='11ss'),
    "bias_1_1": ModelConfig(2, 16, 16,  64,  128,  128, 0.0,
        "no_mask",        "post_scale_bias", bias_shape='1hss'),
    "bias_1_2": ModelConfig(4, 24, 24, 128, 2048, 2048, 0.0,
        "no_mask",         "post_scale_bias", bias_shape='b1ss'),
    "bias_1_3": ModelConfig(2, 24, 24, 128, 2048, 2048, 0.0,
        "no_mask",         "post_scale_bias", bias_shape='bhss'),
    "bias_1_4": ModelConfig(4, 24, 24, 128, 2048, 2048, 0.0,
        "causal",                   "alibi", bias_shape='1hss', alibi_type='custom'),
    "bias_1_5": ModelConfig(2, 24, 24, 128, 2048, 2048, 0.0,
        "causal",                   "alibi", bias_shape='bhss', alibi_type='custom'),
}

@pytest.mark.skipif(_cudnn_version() < (8,9,1), reason="cuDNN 8.9.1+ is required.")
@pytest.mark.parametrize("dtype", param_types_lean)
@pytest.mark.parametrize("model_configs", [model_configs_bias_shapes])
@pytest.mark.parametrize("model", model_configs_bias_shapes.keys())
def test_dpa_bias_shapes(dtype, model_configs, model):
    """Test DotProductAttention module with different bias types and shapes"""
    test_dot_product_attention(dtype, model_configs, model, False, True, None, False)

model_configs_swa = {
    #     test:             b,  h, hg,   d,   sq,  skv,   p,             mask,             bias
    "swa_1_0": ModelConfig(4, 16, 16,  64,  128,  128, 0.0,        "no_mask",          "no_bias"),
    "swa_1_1": ModelConfig(2, 16, 16,  64,  128,  256, 0.0,        "no_mask",          "no_bias"),
    "swa_1_2": ModelConfig(4, 24, 24, 128, 2048, 2048, 0.0,        "no_mask",          "no_bias"),
    "swa_1_3": ModelConfig(2, 24, 24, 128, 2048, 4096, 0.0,        "no_mask",          "no_bias"),
}
@pytest.mark.skipif(not _is_flash_attention_2_3(), reason="Flash-attn 2.3+ is required.")
@pytest.mark.parametrize("dtype", param_types_lean)
@pytest.mark.parametrize("model_configs", [model_configs_swa])
@pytest.mark.parametrize("model", model_configs_swa.keys())
def test_dpa_sliding_window(dtype, model_configs, model):
    """Test DotProductAttention module with sliding window attention"""
    test_dot_product_attention(dtype, model_configs, model, False, True, None, True)

model_configs_alibi_slopes = {
    #     test:             b,  h, hg,   d,   sq,  skv,   p,      mask,    bias, alibi_type
    "alibi_1_0": ModelConfig(2, 16, 16,  64,  128,  128, 0.0, "causal", "alibi", alibi_type="vanilla"),
    "alibi_1_1": ModelConfig(1, 16, 16,  64,  128,  256, 0.0, "causal", "alibi", alibi_type="vanilla"),
    "alibi_2_0": ModelConfig(2, 24, 24, 128, 1024, 1024, 0.0, "causal", "alibi", alibi_type= "custom"),
    "alibi_2_1": ModelConfig(1, 24, 24, 128, 1024, 2048, 0.0, "causal", "alibi", alibi_type= "custom"),
}
@pytest.mark.skipif(not _is_flash_attention_2_3(), reason="Flash-attn 2.3+ is required.")
@pytest.mark.parametrize("dtype", param_types_lean)
@pytest.mark.parametrize("model_configs", [model_configs_alibi_slopes])
@pytest.mark.parametrize("model", model_configs_alibi_slopes.keys())
def test_dpa_alibi_slopes(dtype, model_configs, model):
    """Test DotProductAttention module with ALiBi slopes"""
    test_dot_product_attention(dtype, model_configs, model, False, True, None, False)

qkv_layouts = [
    'sb3hd', 'sbh3d', 'sbhd_sb2hd', 'sbhd_sbh2d', 'sbhd_sbhd_sbhd',
    'bs3hd', 'bsh3d', 'bshd_bs2hd', 'bshd_bsh2d', 'bshd_bshd_bshd',
    # will add tests for thd layouts later when the support is available in fused attention
    #'t3hd', 'th3d', 'thd_t2hd', 'thd_th2d', 'thd_thd_thd',
    ]

model_configs_layout = {
    #       test:             b,  h, hg,   d,   sq,  skv,   p,             mask,             bias
    "layout_0_0": ModelConfig(2, 16, 16,  64,  128,  128, 0.0,        "no_mask",         "no_bias"),
    "layout_0_1": ModelConfig(2, 16, 16,  64,  128,  128, 0.0,         "causal", "post_scale_bias"),
    "layout_0_2": ModelConfig(1, 16, 16,  64,  128,  256, 0.0,        "padding",         "no_bias"),
    "layout_0_3": ModelConfig(1, 16, 16,  64,  128,  256, 0.0, "padding_causal", "post_scale_bias"),
    "layout_1_0": ModelConfig(2, 24, 24, 128, 2048, 2048, 0.0,        "no_mask",         "no_bias"),
    "layout_1_1": ModelConfig(2, 24, 24, 128, 2048, 2048, 0.0,         "causal", "post_scale_bias"),
    "layout_1_2": ModelConfig(1, 24, 24, 128, 2048, 4096, 0.0,        "padding",         "no_bias"),
    "layout_1_3": ModelConfig(1, 24, 24, 128, 2048, 4096, 0.0, "padding_causal", "post_scale_bias"),
}

@pytest.mark.skipif(_cudnn_version() < (8,9,5), reason="cuDNN 8.9.5+ is required.")
@pytest.mark.parametrize("dtype", param_types_lean)
@pytest.mark.parametrize("model_configs", [model_configs_layout])
@pytest.mark.parametrize("model", model_configs_layout.keys())
@pytest.mark.parametrize("qkv_layout", qkv_layouts)
def test_dpa_qkv_layout(dtype, model_configs, model, qkv_layout):
    """Test DotProductAttention module with different QKV layouts"""
    test_dot_product_attention(dtype, model_configs, model, False, True, qkv_layout, False)

def _run_dot_product_attention(
        dtype: torch.dtype,
        config: ModelConfig,
        backend: str,
        ckpt_attn: bool,
        qkv_layout: str,
        workspace_opt: bool,
        swa: bool,
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

    # Create seqlens
    qkv_format = ''.join([i for i in qkv_layout.split('_')[0] if i.isalpha()])
    if "padding" in config.attn_mask_type or qkv_format == 'thd':
        if config.attn_type == 'self':
            seqlens_q = torch.randint(1, config.max_seqlen_q, [config.batch_size],
                    dtype=torch.int32, device="cuda")
            seqlens_kv = seqlens_q
        if config.attn_type == 'cross':
            seqlens_q = torch.randint(1, config.max_seqlen_q, [config.batch_size],
                    dtype=torch.int32, device="cuda")
            seqlens_kv = torch.randint(1, config.max_seqlen_kv, [config.batch_size],
                    dtype=torch.int32, device="cuda")
    else:
        seqlens_q = torch.full([config.batch_size], config.max_seqlen_q,
                dtype=torch.int32, device="cuda")
        seqlens_kv = torch.full([config.batch_size], config.max_seqlen_kv,
                dtype=torch.int32, device="cuda")
    cu_seqlens_q = torch.zeros(config.batch_size + 1, dtype=torch.int32, device="cuda")
    cu_seqlens_kv = torch.zeros(config.batch_size + 1, dtype=torch.int32, device="cuda")
    cu_seqlens_q[1:] = torch.cumsum(seqlens_q, dim=0)
    cu_seqlens_kv[1:] = torch.cumsum(seqlens_kv, dim=0)

    # Create attention mask if padding
    attention_mask = None
    if "padding" in config.attn_mask_type:
        if config.attn_type == 'self':
            attention_mask_q = torch.Tensor([]).to(dtype=torch.bool)
            for i in range(config.batch_size):
                attention_mask_q = torch.cat([attention_mask_q,
                    torch.Tensor([True]*seqlens_q[i] + [False]*(config.max_seqlen_q-seqlens_q[i]))
                    .to(dtype=torch.bool).unsqueeze(0).unsqueeze(0).unsqueeze(0)], dim=0)
            attention_mask = attention_mask_q.to(device="cuda")
        if config.attn_type == 'cross':
            attention_mask_q = torch.Tensor([]).to(dtype=torch.bool)
            attention_mask_kv = torch.Tensor([]).to(dtype=torch.bool)
            for i in range(config.batch_size):
                attention_mask_q = torch.cat([attention_mask_q,
                    torch.Tensor([True]*seqlens_q[i] + [False]*(config.max_seqlen_q-seqlens_q[i]))
                    .to(dtype=torch.bool).unsqueeze(0).unsqueeze(0).unsqueeze(0)], dim=0)
                attention_mask_kv = torch.cat([attention_mask_kv, torch.Tensor(
                    [True]*seqlens_kv[i] + [False]*(config.max_seqlen_kv-seqlens_kv[i]))
                    .to(dtype=torch.bool).unsqueeze(0).unsqueeze(0).unsqueeze(0)], dim=0)
            attention_mask = (
                    attention_mask_q.to(device="cuda"), attention_mask_kv.to(device="cuda"))
    if swa:
        window_size, attention_mask = get_swa(config.max_seqlen_q, config.max_seqlen_kv)
    elif "causal" in config.attn_mask_type:
        window_size, attention_mask = (-1, 0), None
    else:
        window_size, attention_mask = None, None

    alibi_slopes = None
    if config.attn_bias_type == "alibi" and config.alibi_type == "custom":
        if config.bias_shape == "1hss":
            alibi_slopes = torch.randn(
                config.num_heads).abs().to(dtype=torch.float32, device="cuda")
        if config.bias_shape == "bhss":
            alibi_slopes = torch.randn(
                config.batch_size, config.num_heads).abs().to(dtype=torch.float32, device="cuda")

    # Create input tensors
    dim_to_num = {
        'b'  : config.batch_size,
        'sq' : config.max_seqlen_q,
        'skv': config.max_seqlen_kv,
        'h'  : config.num_heads,
        'hg' : config.num_gqa_groups,
        'd'  : config.head_dim,
        't'  : cu_seqlens_q[-1],
        'tg' : cu_seqlens_kv[-1],
        '3'  : 3,
        '2'  : 2,
        '1'  : 1,
        }
    inp = []
    for i,layout in enumerate(qkv_layout.split('_')):
        layout = '_'.join(layout)
        if i == 0:
            layout = layout.replace('s', 'sq')
        else:
            layout = layout.replace('s', 'skv')
            layout = layout.replace('h', 'hg')
            layout = layout.replace('t', 'tg')
        tensor_shape = [dim_to_num[j] for j in layout.split('_')]
        tensor = 0.1 * torch.randn(tensor_shape, dtype=dtype, device="cuda")
        tensor_count = 1
        split_dim = 0
        for dim, l in enumerate(layout.split('_')):
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

    # Create output gradient
    qkv_format_kv = '_'.join(qkv_format)
    qkv_format_kv = qkv_format_kv.replace('s', 'sq')
    out_grad_shape = [dim_to_num[i] for i in qkv_format_kv.split('_')]
    out_grad_shape_new = [*out_grad_shape[:-2], out_grad_shape[-2] * out_grad_shape[-1]]
    out_grad = 0.001 * torch.randint(0, 200, out_grad_shape_new, dtype=dtype, device="cuda")

    # Create bias
    if config.attn_bias_type in ['no_bias', 'alibi']:
        bias = None
    if config.attn_bias_type == 'post_scale_bias':
        shape = '_'.join(config.bias_shape)
        shape = shape.replace('_s_s', '_sq_skv')
        tensor_shape = [dim_to_num[j] for j in shape.split('_')]
        bias = torch.randn(tensor_shape, dtype=dtype, device="cuda")
        if config.bias_shape != '1hss':
            bias.requires_grad = False

    # Create RNG
    _DUMMY_CUDA_RNG_STATE_TRACKER = CudaRNGStatesTracker()
    _DUMMY_CUDA_RNG_STATE_TRACKER.add("model-parallel-rng", seed)
    def get_dummy_cuda_rng_tracker() -> CudaRNGStatesTracker:
        """Get cuda rng tracker."""
        return _DUMMY_CUDA_RNG_STATE_TRACKER

    # Set up model
    block = (
         DotProductAttention(
                config.num_heads,
                config.head_dim,
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
    )

    # Run a forward and backward pass
    out = block(inp[0], inp[1], inp[2],
            window_size=window_size,
            attention_mask=attention_mask,
            qkv_format=qkv_format,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_kv=cu_seqlens_kv,
            attn_mask_type=config.attn_mask_type,
            checkpoint_core_attention=ckpt_attn,
            core_attention_bias_type=config.attn_bias_type,
            core_attention_bias=bias,
            alibi_slopes=alibi_slopes,
            fast_zero_fill=True)
    out.backward(out_grad)

    return out, (inp[0].grad, inp[1].grad, inp[2].grad)

model_configs_te_layer = {
    #   test:             b,  h, hg,   d,   sq,  skv,   p,      mask,             bias
    "te_1_0": ModelConfig(2, 16, 16,  64,  128,  128, 0.0, "no_mask", "post_scale_bias"),
    "te_1_1": ModelConfig(4, 16, 16,  64,  128,  128, 0.0,  "causal", "post_scale_bias"),
    "te_1_2": ModelConfig(2, 16, 16,  64,  128,  128, 0.0, "padding", "post_scale_bias"),
    "te_2_0": ModelConfig(1, 16, 16,  64, 2048, 2048, 0.0,  "causal",         "no_bias"),
    "te_2_1": ModelConfig(2, 16, 16,  64, 2048, 2048, 0.0, "no_mask",         "no_bias"),
    "te_2_2": ModelConfig(1, 16, 16,  64, 2048, 2048, 0.0, "padding",         "no_bias"),
    "te_3_0": ModelConfig(4, 16, 16,  64,  128,  128, 0.0,  "causal",           "alibi"),
    "te_3_1": ModelConfig(4, 16, 16,  64, 2048, 2048, 0.0,  "causal",           "alibi"),
}

@pytest.mark.skipif(_cudnn_version() < (8,9,1), reason="cuDNN 8.9.1+ is required.")
@pytest.mark.parametrize("dtype", param_types)
@pytest.mark.parametrize("model_configs", [model_configs_te_layer])
@pytest.mark.parametrize("model", model_configs_te_layer.keys())
@pytest.mark.parametrize("ckpt_attn", [False])
@pytest.mark.parametrize("qkv_format", ["sbhd"])
@pytest.mark.parametrize("fused_qkv_params", [False])
@pytest.mark.parametrize("RoPE", [False])
def test_transformer_layer(dtype, model_configs, model, ckpt_attn, qkv_format, fused_qkv_params, RoPE):
    """Test TransformerLayer module"""

    # Get configs
    config = model_configs[model]
    tols = dict(atol=5e-1, rtol=5e-2)
    workspace_opt = True

    # Skip if only unfused backend is supported
    if config.max_seqlen_q <= 512 and config.max_seqlen_kv <= 512:
        os.environ["NVTE_FUSED_ATTN_BACKEND"] = "0"
    fused_attn_supported, fused_attn_backend = _is_fused_attention_supported(
        config,
        dtype,
        qkv_layout="sbh3d" if fused_qkv_params else "sb3hd",
    )
    flash_attn_supported = _is_flash_attention_supported(config)
    unfused_attn_supported = _is_unfused_attention_supported(config)
    if (len(fused_attn_backend) + flash_attn_supported + unfused_attn_supported) < 2:
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
        if _NVTE_DEBUG:
            print("[test_transformer_layer]: unfused attn vs fused attn")
        torch.testing.assert_close(fused_attn_fwd, unfused_attn_fwd, **tols)
        torch.testing.assert_close(fused_attn_bwd, unfused_attn_bwd, **tols)
    if unfused_attn_supported and flash_attn_supported:
        if _NVTE_DEBUG:
            print("[test_transformer_layer]: unfused attn vs flash attn")
        torch.testing.assert_close(flash_attn_fwd, unfused_attn_fwd, **tols)
        torch.testing.assert_close(flash_attn_bwd, unfused_attn_bwd, **tols)
    if fused_attn_supported and flash_attn_supported:
        if _NVTE_DEBUG:
            print("[test_transformer_layer]: fused attn vs flash attn")
        torch.testing.assert_close(fused_attn_fwd, flash_attn_fwd, **tols)
        torch.testing.assert_close(fused_attn_bwd, flash_attn_bwd, **tols)

@pytest.mark.skipif(_cudnn_version() < (8,9,1), reason="cuDNN 8.9.1+ is required.")
@pytest.mark.parametrize("dtype", param_types_lean)
@pytest.mark.parametrize("model_configs", [model_configs_te_layer])
@pytest.mark.parametrize("model", ["te_1_2", "te_2_0"])
@pytest.mark.parametrize("qkv_format", ["bshd", "sbhd"])
def test_te_layer_misc(dtype, model_configs, model, qkv_format):
    """Test TransformerLayer module with miscellaneous settings"""
    ckpt_attn = True
    fused_qkv_params = True
    RoPE = True
    test_transformer_layer(dtype, model_configs, model,
            ckpt_attn, qkv_format, fused_qkv_params, RoPE)

@pytest.mark.skipif(_cudnn_version() < (8,9,1), reason="cuDNN 8.9.1+ is required.")
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
        config.num_gqa_groups=config.num_heads // num_q_per_gqa_group
        test_transformer_layer(dtype, model_configs, model,
                ckpt_attn, qkv_format, fused_qkv_params, RoPE)

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

    # Create input tensor
    inp = torch.randn(config.max_seqlen_q, config.batch_size, config.hidden_size,
            dtype=dtype, device="cuda", requires_grad = True)
    # In case the format to be tested is batch-first, need to transpose the
    # input tensor.
    if qkv_format == "bshd":
            inp = inp.transpose(0,1)

    # Create seqlens
    if "padding" in config.attn_mask_type:
        seqlens_q = torch.randint(1, config.max_seqlen_q, [config.batch_size],
                dtype=torch.int32, device="cuda")
    else:
        seqlens_q = torch.full([config.batch_size], config.max_seqlen_q,
                dtype=torch.int32, device="cuda")

    # Create attention mask if padding
    attention_mask = None
    if "padding" in config.attn_mask_type:
        attention_mask_q = torch.Tensor([]).to(dtype=torch.bool)
        for i in range(config.batch_size):
            attention_mask_q = torch.cat([attention_mask_q,
                torch.Tensor([True]*seqlens_q[i] + [False]*(config.max_seqlen_q-seqlens_q[i]))
                .to(torch.bool).unsqueeze(0).unsqueeze(0).unsqueeze(0)], dim=0)
        attention_mask = attention_mask_q.to(device="cuda")

    sigma = 0.02
    init_method = init_method_normal(sigma)
    output_layer_init_method = scaled_init_method_normal(sigma, config.num_layers)

    layer_number = 1
    drop_path_rate = 0.0
    drop_path_rates = [
            rate.item() for rate in torch.linspace(0, drop_path_rate, config.num_layers)]

    # Create bias
    bias = None
    if config.attn_bias_type == 'post_scale_bias':
        bias = torch.randn(1, config.num_heads, config.max_seqlen_q, config.max_seqlen_kv,
                dtype=dtype, device="cuda")

    # Create RoPE
    rotary_pos_emb = None
    if RoPE:
        PE = RotaryPositionEmbedding(dim=config.head_dim)
        rotary_pos_emb = PE(config.max_seqlen_q).to(device="cuda")

    # Set up model
    block = (
        TransformerLayer(
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
            kv_channels=config.head_dim,
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
        )
        .to(dtype=dtype, device="cuda")
    )

    # Create ALiBi slopes
    alibi_slopes = None
    if config.attn_bias_type == "alibi" and config.alibi_type == "custom":
        alibi_slopes = torch.randn(
            config.num_heads).abs().to(dtype=torch.float32, device="cuda")

    # Run a forward and backward pass
    out = block(inp,
        attention_mask=attention_mask,
        self_attn_mask_type=config.attn_mask_type,
        checkpoint_core_attention=False,
        rotary_pos_emb=rotary_pos_emb,
        core_attention_bias_type=config.attn_bias_type,
        core_attention_bias=bias,
        alibi_slopes=alibi_slopes)
    loss = out.sum()
    loss.backward()

    return out, inp.grad


model_configs_fp8 = {
    #  test:             b,  h, hg,   d,   sq,  skv,   p,      mask,      bias
    "fp8_1": ModelConfig(1, 16, 16,  64,  512,  512, 0.0, "no_mask", "no_bias"),
    "fp8_2": ModelConfig(4, 16, 16,  64,  512,  512, 0.0, "no_mask", "no_bias"),
}
param_types_fp8 = [torch.float16]

@pytest.mark.skipif(_cudnn_version() < (8,9,3), reason="cuDNN 8.9.3+ is required.")
@pytest.mark.skipif(not fp8_available, reason=reason_for_no_fp8)
@pytest.mark.skipif(get_device_compute_capability() != (9, 0), reason="FP8 tests require Hopper.")
@pytest.mark.parametrize("dtype", param_types_fp8)
@pytest.mark.parametrize("model", model_configs_fp8.keys())
def test_dpa_fp8(dtype, model):
    """Test FP8 dot product attention

    FusedAttention uses fused_attn_fwd/bwd_qkvpacked from cpp_extensions,
    and UnfusedDotProductAttention uses plain PyTorch operations in FP16
    and converts inputs/outputs from/to FP8.

    """

    config = model_configs_fp8[model]

    # Skip if not supported
    fused_attn_supported, fused_attn_backend = _is_fused_attention_supported(
        config, dtype)
    if not fused_attn_supported:
        pytest.skip("FusedAttention does not support this model config")

    # Run dot-product attention with different backends
    fused_attn_fwd, fused_attn_bwd = _run_dpa_fp8(
        dtype, config, "FusedAttention")
    unfused_attn_fwd, unfused_attn_bwd = _run_dpa_fp8_ref(
        dtype, config, "UnfusedDotProductAttention")

    tols = dict(atol=2.5e-2, rtol=2.5e-2)
    torch.testing.assert_close(fused_attn_fwd, unfused_attn_fwd, **tols)
    torch.testing.assert_close(fused_attn_bwd, unfused_attn_bwd, **tols)

def _run_dpa_fp8(dtype, config, backend):
    """Run FusedAttention FP8 backend, i.e.
    fused_attn_fwd/bwd_qkvpacked from cpp_extensions"""

    reset_rng_states()
    os.environ["NVTE_FLASH_ATTN"] = "0"
    os.environ["NVTE_FUSED_ATTN"] = "0"
    if backend == "FlashAttention":
        os.environ["NVTE_FLASH_ATTN"] = "1"
    if backend == "FusedAttention":
        os.environ["NVTE_FUSED_ATTN"] = "1"

    inp = 0.01 * torch.randn(
            config.batch_size * config.max_seqlen_q, config.num_heads * config.head_dim,
            dtype=dtype, device="cuda", requires_grad=True)
    seqlens = torch.full([config.batch_size], config.max_seqlen_q,
            dtype=torch.int32, device="cuda")
    cu_seqlens = torch.zeros(config.batch_size + 1, device="cuda", dtype=torch.int32)
    cu_seqlens[1:] = torch.cumsum(seqlens, dim=0)
    out_grad = 0.01 * torch.randn(
            config.batch_size * config.max_seqlen_q, config.num_heads * config.head_dim,
            dtype=dtype, device="cuda")
    torch.save(out_grad, 'out_grad.pt')

    fp8_recipe = recipe.DelayedScaling(
        margin=0,
        interval=1,
        fp8_format=recipe.Format.HYBRID,
        amax_history_len=1,
        amax_compute_algo="most_recent",
    )

    dpa = DPA_FP8(config).to(dtype=torch.float16, device="cuda")
    with fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
        out = dpa(inp, cu_seqlens, config.max_seqlen_q)
        out.backward(out_grad)

    context = torch.load("ctx.pt")
    dqkv = torch.load('dqkv.pt')
    return (context.view(config.batch_size, config.max_seqlen_q, -1).transpose(0,1),
            dqkv.view(config.batch_size, config.max_seqlen_q, 3,
            config.num_heads, config.head_dim).transpose(0,1).contiguous())

def _run_dpa_fp8_ref(dtype, config, backend):
    """Run UnfusedDotProductAttention as a reference, i.e.
    plain PyTorch implementation in FP16 and inputs/outputs
    are converted from/to FP8"""

    os.environ["NVTE_FLASH_ATTN"] = "0"
    os.environ["NVTE_FUSED_ATTN"] = "0"
    if backend == "FlashAttention":
        os.environ["NVTE_FLASH_ATTN"] = "1"
    if backend == "FusedAttention":
        os.environ["NVTE_FUSED_ATTN"] = "1"

    inp = torch.load('qkv.pt').to(device="cuda")
    inp.requires_grad = True
    seqlens = torch.full([config.batch_size], config.max_seqlen_q, dtype=torch.int32, device="cuda")
    cu_seqlens = torch.zeros(config.batch_size + 1, device="cuda", dtype=torch.int32)
    cu_seqlens[1:] = torch.cumsum(seqlens, dim=0)
    out_grad = torch.load('out_grad.pt').to(device="cuda").view(
            config.batch_size, config.max_seqlen_q, -1).transpose(0,1)

    _DUMMY_CUDA_RNG_STATE_TRACKER = CudaRNGStatesTracker()
    _DUMMY_CUDA_RNG_STATE_TRACKER.add("model-parallel-rng", seed)

    def get_dummy_cuda_rng_tracker() -> CudaRNGStatesTracker:
        """Get cuda rng tracker."""
        return _DUMMY_CUDA_RNG_STATE_TRACKER

    _DUMMY_CUDA_RNG_STATE_TRACKER = CudaRNGStatesTracker()
    _DUMMY_CUDA_RNG_STATE_TRACKER.add("model-parallel-rng", seed)

    def get_dummy_cuda_rng_tracker():
        """Get cuda rng tracker."""
        return _DUMMY_CUDA_RNG_STATE_TRACKER

    block = (
         DotProductAttention(
                config.num_heads,
                config.head_dim,
                attention_dropout=config.dropout_p,
                sequence_parallel=False,
                tp_size=1,
                get_rng_state_tracker=get_dummy_cuda_rng_tracker,
                tp_group=None,
                layer_number=1,
                attention_type="self"
        ).to(dtype=dtype, device="cuda")
    )

    q = inp[:, :,0,:,:]
    k = inp[:, :,1,:,:]
    v = inp[:, :,2,:,:]
    out = block(q, k, v, attn_mask_type=config.attn_mask_type)
    out.backward(out_grad)

    return out, inp.grad


_CUBLASLT_WORKSPACE_SIZE_BYTES = 33_554_432  # 32MiB
_2X_ACC_FPROP = False
_2X_ACC_DGRAD = False
_2X_ACC_WGRAD = False

META_QKV  = tex.FP8FwdTensors.GEMM1_OUTPUT
META_O    = tex.FP8FwdTensors.GEMM2_INPUT
META_DO   = tex.FP8BwdTensors.GRAD_INPUT2
META_DQKV = tex.FP8BwdTensors.GRAD_OUTPUT1

META_S    = tex.FP8FwdTensors.GEMM3_WEIGHT
META_DS   = tex.FP8BwdTensors.GRAD_INPUT3

class _dpa_fp8(torch.autograd.Function):
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
    ) -> torch.Tensor:

        assert inp.dim() == 2
        in_features = qkv_weight.shape[-1]
        h = num_heads
        d = in_features // h
        b = cu_seqlens.numel() - 1
        is_nl = False
        if b < 4 and b > 1:
            max_s = 512
            is_nl = True

        fp8_dtype_forward = fp8.get_fp8_te_dtype(fp8_meta["recipe"], fprop_tensor=True)

        inputmat, inputmat_t = ext.fp8_cast_transpose_fused(
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

        qkv_out, _ = ext.fp8_gemm(
            qkv_weight_fp8,
            fp8_meta["scaling_fwd"].scale_inv,
            tex.FP8FwdTensors.GEMM1_WEIGHT,
            fp8_dtype_forward,
            inputmat,
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
        qkv_out = qkv_out.view(-1, 3, h, d)
        qkv_out_fp16 = ext.cast_from_fp8(qkv_out, fp8_meta["scaling_fwd"],
                META_QKV, fp8_dtype_forward,
                tex.DType.kFloat16).view(b, max_s, 3, h, d).transpose(0,1).contiguous()
        torch.save(qkv_out_fp16, 'qkv.pt')

        # FMHA
        context_, aux_ctx_tensors, *rest = fused_attn_fwd(
                is_training,
                max_s,
                max_s,
                cu_seqlens,
                cu_seqlens,
                qkv_out[:,0,:,:],
                qkv_out[:,1,:,:],
                qkv_out[:,2,:,:],
                fp8_dtype_forward,
                FusedAttnBackend["FP8"],
                None,
                fp8_meta["scaling_fwd"].scale_inv[META_QKV],
                fp8_meta["scaling_fwd"].scale[META_S],
                fp8_meta["scaling_fwd"].scale[META_O],
                fp8_meta["scaling_fwd"].amax_history[0][META_S],
                fp8_meta["scaling_fwd"].amax_history[0][META_O],
                attn_scale=None,
                dropout=p_dropout,
                fast_zero_fill=fast_zero_fill,
                qkv_layout="t3hd",
                attn_bias_type="no_bias",
                attn_mask_type="padding",
                rng_gen=None,
                )
        M, ZInv, philox_unpacked = aux_ctx_tensors

        context = context_.view(-1, in_features)
        context_t = tex.fp8_transpose(context, fp8_dtype_forward)

        ctx.save_for_backward(
            inputmat_t, qkv_weight_t_fp8, workspace,
            qkv_out,
            context_, context_t,
            fp8_meta["scaling_fwd"].scale,
            fp8_meta["scaling_fwd"].scale_inv,
        )
        ctx.aux_ctx_tensors = aux_ctx_tensors
        ctx.fp8_meta = fp8_meta
        ctx.cu_seqlens = cu_seqlens
        ctx.p_dropout = p_dropout
        ctx.max_s = max_s
        ctx.fast_zero_fill = fast_zero_fill
        ctx.is_nl = is_nl
        ctx.hidden_size = in_features
        ctx.num_heads = num_heads

        context_fp16 = ext.cast_from_fp8(context, fp8_meta["scaling_fwd"],
                META_O, fp8_dtype_forward, tex.DType.kFloat16)
        torch.save(context_fp16, 'ctx.pt')
        return context_fp16


    @staticmethod
    def backward(
        ctx, grad_output: torch.Tensor
    ) -> Tuple[Union[torch.Tensor, None], ...]:

        with _prepare_backward(True, ctx.fp8_meta, None, 1, name="_DPA"):
            (
                inputmat_t,
                qkv_weight_t_fp8,
                workspace,
                qkv_out,
                context, context_t,
                fwd_scales,
                fwd_scale_inverses,
            ) = ctx.saved_tensors
            fp8_dtype_forward = fp8.get_fp8_te_dtype(
                ctx.fp8_meta["recipe"], fprop_tensor=True
            )
            fp8_dtype_backward = fp8.get_fp8_te_dtype(
                ctx.fp8_meta["recipe"], fprop_tensor=False
            )

            proj_dgrad = ext.cast_to_fp8(
                grad_output, ctx.fp8_meta["scaling_bwd"], META_DO, fp8_dtype_backward
            )

            dq, dk, dv, *rest = fused_attn_bwd(
                    ctx.max_s,
                    ctx.max_s,
                    ctx.cu_seqlens,
                    ctx.cu_seqlens,
                    qkv_out[:,0,:,:],
                    qkv_out[:,1,:,:],
                    qkv_out[:,2,:,:],
                    context,
                    proj_dgrad.view_as(context),
                    fp8_dtype_forward,
                    ctx.aux_ctx_tensors,
                    FusedAttnBackend["FP8"],
                    fwd_scale_inverses[META_QKV], # d_scale_qkv,
                    fwd_scale_inverses[META_S], # d_scale_s,
                    fwd_scale_inverses[META_O], # d_scale_o,
                    ctx.fp8_meta['scaling_bwd'].scale_inv[META_DO], # d_scale_do
                    fwd_scales[META_S], # q_scale_s
                    ctx.fp8_meta['scaling_bwd'].scale[META_DS], # q_scale_ds
                    ctx.fp8_meta['scaling_bwd'].scale[META_DQKV], # q_scale_dqkv
                    ctx.fp8_meta['scaling_bwd'].amax_history[0][META_DS], # amax_ds
                    ctx.fp8_meta['scaling_bwd'].amax_history[0][META_DQKV], # amax_dqkv
                    None,
                    ctx.p_dropout,
                    ctx.fast_zero_fill,
                    "t3hd",
                    "no_bias",
                    "padding",
                    )
            dqkv = torch.cat([dq.unsqueeze(1), dk.unsqueeze(1), dv.unsqueeze(1)], dim=1)

            dqkv_grad_output_c = dqkv.view(-1, 3*ctx.hidden_size)
            dqkv_grad_output_c_fp16 = ext.cast_from_fp8(dqkv_grad_output_c,
                ctx.fp8_meta["scaling_bwd"], META_DQKV,
                fp8_dtype_backward, tex.DType.kFloat16)
            torch.save(dqkv_grad_output_c_fp16, 'dqkv.pt')

            qkv_bgrad, dqkv_grad_output_t = ext.fp8_transpose_bgrad_fused(
                dqkv_grad_output_c,
                ctx.fp8_meta["scaling_bwd"],
                META_DQKV,
                fp8_dtype_backward,
                torch.float16,
            )

            # QKV DGRAD
            qkv_dgrad, _ = ext.fp8_gemm(
                qkv_weight_t_fp8,
                fwd_scale_inverses,
                tex.FP8FwdTensors.GEMM1_WEIGHT,
                fp8_dtype_forward,
                dqkv_grad_output_c,
                ctx.fp8_meta["scaling_bwd"].scale_inv,
                META_DQKV,
                fp8_dtype_backward,
                torch.float16,
                workspace,
                use_split_accumulator=_2X_ACC_DGRAD,
            )
            # QKV WGRAD
            qkv_wgrad, _ = ext.fp8_gemm(
                inputmat_t,
                fwd_scale_inverses,
                tex.FP8FwdTensors.GEMM1_INPUT,
                fp8_dtype_forward,
                dqkv_grad_output_t,
                ctx.fp8_meta["scaling_bwd"].scale_inv,
                META_DQKV,
                fp8_dtype_backward,
                torch.float16,
                workspace,
                use_split_accumulator=_2X_ACC_WGRAD,
            )

        return (qkv_dgrad,
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
            None)

class DPA_FP8(TransformerEngineBaseModule):
    def __init__(
        self,
        config,
        params_dtype: torch.dtype = torch.float32):
        super().__init__()
        self.p_dropout = config.dropout_p
        self.h = config.num_heads
        self.hidden_size = config.hidden_size
        self.head_dim = config.head_dim
        self.fast_zero_fill = True

        self.qkv_weight = torch.nn.Parameter(
            torch.empty(
                self.hidden_size * 3,
                self.hidden_size,
                device=torch.cuda.current_device(),
                dtype=params_dtype,
            )
        )
        self.fp8_weight_shapes.append(self.qkv_weight.shape)
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
        self, inp: torch.Tensor,
        cu_seqlens, max_s,
    ) -> torch.Tensor:
        with self.prepare_forward(inp, None, num_gemms=3) as inp:
            out = _dpa_fp8.apply(
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
                self.training)
        return out

    def get_fp8_weights_scratchpad(
        self,
        is_first_microbatch: Union[bool, None],
    ) -> List[torch.Tensor]:
        """Needs override."""
