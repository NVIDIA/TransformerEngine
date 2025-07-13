# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

from __future__ import annotations

import logging
import os
from contextlib import contextmanager

import torch

from transformer_engine.pytorch.attention.dot_product_attention import _attention_backends
from transformer_engine.pytorch.attention.dot_product_attention.utils import (
    get_attention_backend,
    AttentionParams,
)
from transformer_engine.pytorch.attention.dot_product_attention.utils import (
    AttentionLogging as attn_log,
)
from transformer_engine.pytorch.cpp_extensions.fused_attn import FusedAttnBackend


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
    is_training: bool = True,
    inference_params: Optional[InferenceParams] = None,
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
            use_fused_attention,
            flash_attention_backend,
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
    if attn_log._is_logging_setup is False:
        attn_log.setup_logging()
    with logging_context(highest_level=attn_log._log_level):
        for i in range(3):
            os.environ["NVTE_FUSED_ATTN_BACKEND"] = str(i)
            _attention_backends["backend_selection_requires_update"] = True
            available_backends, flash_attention_backend, fused_attention_backend = test()
            if fused_attention_backend == FusedAttnBackend[backends[i]]:
                fused_attn_backends.append(fused_attention_backend)
    return available_backends, flash_attention_backend, fused_attn_backends
