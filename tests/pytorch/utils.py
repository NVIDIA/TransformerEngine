# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

from __future__ import annotations

import logging
import os
from contextlib import contextmanager

import pytest
import torch

import transformer_engine
import transformer_engine.common.recipe
import transformer_engine.pytorch as te
import transformer_engine_torch as tex
from transformer_engine.pytorch.attention.dot_product_attention import _attention_backends
from transformer_engine.pytorch.attention.dot_product_attention.utils import (
    get_attention_backend,
    AttentionParams,
    AttentionLogging,
)
from transformer_engine.pytorch.cpp_extensions.fused_attn import FusedAttnBackend


def str_to_dtype(dtype: str | torch.dtype) -> torch.dtype:
    """Convert type name to PyTorch dtype"""
    if isinstance(dtype, torch.dtype):
        return dtype
    name = str(dtype).strip().lower()
    if name.startswith("torch."):
        name = name.replace("torch.", "", 1)
    if name.startswith("fp"):
        name = name.replace("fp", "float", 1)
    dtype = dict(
        float32=torch.float32,
        float=torch.float32,
        float64=torch.float64,
        double=torch.float64,
        float16=torch.float16,
        half=torch.float16,
        bfloat16=torch.bfloat16,
        bf16=torch.bfloat16,
        float8_e4m3fn=torch.float8_e4m3fn,
        float8_e4m3=torch.float8_e4m3fn,
        float8e4m3=torch.float8_e4m3fn,
        float8=torch.float8_e4m3fn,
        float8_e5m2=torch.float8_e5m2,
        float8e5m2=torch.float8_e5m2,
        uint8=torch.uint8,
        byte=torch.uint8,
        int8=torch.int8,
        char=torch.int8,
        int16=torch.int16,
        short=torch.int16,
        int32=torch.int32,
        int=torch.int32,
        int64=torch.int64,
        long=torch.int64,
        bool=torch.bool,
    )[name]
    return dtype


def dtype_tols(dtype: torch.dtype | tex.DType) -> dict[str, float]:
    """Estimated numerical error for a datatype

    Based on tolerances for torch.testing.assert_close.

    """

    # Transformer Engine dtypes
    if isinstance(dtype, tex.DType):
        dtype = {
            tex.DType.kByte: torch.uint8,
            tex.DType.kInt32: torch.int32,
            tex.DType.kFloat32: torch.float32,
            tex.DType.kFloat16: torch.half,
            tex.DType.kBFloat16: torch.bfloat16,
            tex.DType.kFloat8E4M3: torch.float8_e4m3fn,
            tex.DType.kFloat8E5M2: torch.float8_e5m2,
        }[dtype]

    # PyTorch dtypes
    if dtype == torch.float16:
        return dict(rtol=1e-3, atol=1e-5)
    if dtype == torch.bfloat16:
        return dict(rtol=1.6e-2, atol=1e-5)
    if dtype == torch.float32:
        return dict(rtol=1.3e-6, atol=1e-5)
    if dtype == torch.float64:
        return dict(rtol=1e-7, atol=1e-7)
    if dtype == torch.float8_e4m3fn:
        return dict(rtol=0.125, atol=0.0675)  # epsilon = 0.0625
    if dtype == torch.float8_e5m2:
        return dict(rtol=0.25, atol=0.125)  # epsilon = 0.152
    raise ValueError(f"Unsupported dtype ({dtype})")


def make_recipe(name: Optional[str]) -> Optional[Recipe]:
    """Make recipe for quantization scheme"""
    if name is None:
        return None
    if name in ("fp8", "fp8_delayed_scaling"):
        return transformer_engine.common.recipe.DelayedScaling(
            fp8_format=transformer_engine.common.recipe.Format.E4M3,
            amax_history_len=8,
        )
    if name == "fp8_current_scaling":
        return transformer_engine.common.recipe.Float8CurrentScaling(
            fp8_format=transformer_engine.common.recipe.Format.E4M3,
        )
    if name == "mxfp8":
        return transformer_engine.common.recipe.MXFP8BlockScaling(
            fp8_format=transformer_engine.common.recipe.Format.E4M3,
        )
    if name == "fp8_block_scaling":
        return transformer_engine.common.recipe.Float8BlockScaling()
    raise ValueError(f"Unsupported quantization scheme ({name})")


# Cached RNG state
_rng_states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None


def reset_rng_states() -> None:
    """Revert to deterministic RNG state"""
    global _rng_states
    if _rng_states is None:
        torch.manual_seed(1234)
        torch.cuda.manual_seed(1234)
        _rng_states = (torch.get_rng_state(), torch.cuda.get_rng_state())
    else:
        cpu_rng_state, cuda_rng_state = _rng_states
        torch.set_rng_state(cpu_rng_state)
        torch.cuda.set_rng_state(cuda_rng_state)


class ModelConfig:
    def __init__(
        self,
        batch_size: int,
        max_seqlen_q: int,
        num_heads: int,
        head_dim_qk: int,
        max_seqlen_kv: int = None,
        num_gqa_groups: int = None,
        head_dim_v: int = None,
        dropout_p: float = 0.0,
        attn_mask_type: str = "no_mask",
        attn_bias_type: str = "no_bias",
        alibi_type: str = "none",
        bias_shape: str = "1hss",
        window_size: Tuple[int, int] = (-1, -1),
        total_requests: int = None,
        max_ctx_len: int = None,
        num_layers: int = 1,
        eps: float = 1e-5,
    ):
        self.batch_size = batch_size
        self.max_seqlen_q = max_seqlen_q
        self.max_seqlen_kv = max_seqlen_q if max_seqlen_kv is None else max_seqlen_kv
        self.num_heads = num_heads
        self.num_gqa_groups = num_heads if num_gqa_groups is None else num_gqa_groups
        self.head_dim_qk = head_dim_qk
        self.head_dim_v = head_dim_qk if head_dim_v is None else head_dim_v
        if self.head_dim_qk == self.head_dim_v:
            self.kv_channels = self.head_dim_qk
        else:
            self.kv_channels = (self.head_dim_qk, self.head_dim_v)
        self.hidden_size = self.num_heads * self.head_dim_qk
        self.hidden_size_kv = self.num_gqa_groups * self.head_dim_v
        self.dropout_p = dropout_p
        self.attn_mask_type = attn_mask_type
        self.attn_bias_type = attn_bias_type
        self.alibi_type = alibi_type
        self.attn_type = "self" if (self.max_seqlen_q == self.max_seqlen_kv) else "cross"
        self.bias_shape = bias_shape
        self.window_size = window_size
        self.total_requests = total_requests
        self.max_ctx_len = max_ctx_len
        self.num_layers = num_layers
        self.eps = eps


@contextmanager
def logging_context(highest_level=logging.WARNING):
    previous_level = logging.root.manager.disable
    logging.disable(highest_level)
    try:
        yield
    finally:
        logging.disable(previous_level)


def get_available_attention_backends(
    config: ModelConfig,
    qkv_dtype: torch.dtype,
    qkv_layout: str,
    window_size: Tuple[int, int] = (-1, -1),
    pad_between_seqs: bool = False,
    context_parallel: bool = False,
    deterministic: bool = False,
    fp8: bool = False,
    fp8_meta: Optional[Dict[str, Any]] = None,
    is_training: bool = True,
    inference_params: Optional[InferenceParams] = None,
) -> Tuple[List, List]:
    """Check for all available attention backends that support a model configuration"""

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
    flash_attention_backend = None
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
            is_training=is_training,
            inference_params=inference_params,
        )
        (
            use_flash_attention,
            flash_attention_backend,
            use_fused_attention,
            fused_attention_backend,
            use_unfused_attention,
            available_backends,
        ) = get_attention_backend(attention_params)
        # Set attention.py _attention_backends var using return value
        # from get_attention_backend()
        _attention_backends["use_flash_attention"] = use_flash_attention
        _attention_backends["use_fused_attention"] = use_fused_attention
        _attention_backends["flash_attention_backend"] = flash_attention_backend
        _attention_backends["fused_attention_backend"] = fused_attention_backend
        _attention_backends["use_unfused_attention"] = use_unfused_attention
        _attention_backends["backend_selection_requires_update"] = False
        return available_backends, flash_attention_backend, fused_attention_backend

    backends = {0: "F16_max512_seqlen", 1: "F16_arbitrary_seqlen", 2: "FP8"}
    if AttentionLogging._is_logging_setup is False:
        AttentionLogging.setup_logging()
    with logging_context(highest_level=AttentionLogging._log_level):
        for i in range(3):
            os.environ["NVTE_FUSED_ATTN_BACKEND"] = str(i)
            _attention_backends["backend_selection_requires_update"] = True
            available_backends, flash_attention_backend, fused_attention_backend = test()
            if fused_attention_backend == FusedAttnBackend[backends[i]]:
                fused_attn_backends.append(fused_attention_backend)
    return available_backends, flash_attention_backend, fused_attn_backends
