# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Adapt JAX MXFP8 quantization buffers to the shared attention graph contract."""

import jax.numpy as jnp

from transformer_engine.common.cudnn.mxfp8 import quantization_layouts, scale_reordering
from ..quantize import QuantizerFactory, QuantizeLayout, ScalingMode, ScaledTensor2x
from .quantization import quantize


def reorder_scale(scale, columnwise=False):
    """Pad and reorder each logical BHSD scale matrix for cuDNN."""
    scale = scale.view(jnp.uint8)
    padded, tiled, permutation = scale_reordering(scale.shape, columnwise)
    scale = jnp.pad(scale, tuple((0, target - size) for size, target in zip(scale.shape, padded)))
    if columnwise:
        scale = scale.swapaxes(-2, -1)
    return scale.reshape(tiled).transpose(permutation).reshape(padded)


def quantize_operand(value, rowwise=True, columnwise=False):
    """Quantize a logical BHSD operand along its attention contraction axes."""
    # The public colwise-only wrapper uses a pure-JAX fallback. Its division
    # by the minimum E8M0 scale can turn constant zero blocks into NaNs during
    # XLA constant folding. Request the fused common kernel and discard its
    # unused rowwise result when only columnwise data is needed.
    layout = QuantizeLayout.ROWWISE_COLWISE if columnwise else QuantizeLayout.ROWWISE
    quantizer = QuantizerFactory.create(
        scaling_mode=ScalingMode.MXFP8_1D_SCALING,
        q_dtype=jnp.float8_e4m3fn,
        q_layout=layout,
    )
    result = quantize(value, quantizer)
    if isinstance(result, ScaledTensor2x):
        return result.rowwise_tensor if rowwise else None, result.colwise_tensor
    return result, None


def prepare_qkv(qkv, backward=False):
    """Return data and reordered scales named by the common graph builder."""
    values = {}
    for name, value, (rowwise, columnwise) in zip(
        ("q", "k", "v"), qkv, quantization_layouts(backward)
    ):
        value = value.reshape(-1, *value.shape[-3:]).transpose(0, 2, 1, 3)
        row, col = quantize_operand(value, rowwise, columnwise)
        primary = row if rowwise else col
        values[name] = primary.data
        values["descale_" + name] = reorder_scale(primary.scale_inv, not rowwise)
        if rowwise and columnwise:
            values[name + "_t"] = col.data
            values["descale_" + name + "_t"] = reorder_scale(col.scale_inv, True)
    return values


def prepare_doutput(value):
    """Keep the original gradient and prepare both MXFP8 representations."""
    value = value.reshape(-1, *value.shape[-3:])
    row, col = quantize_operand(value.transpose(0, 2, 1, 3), True, True)
    return {
        "do": row.data,
        "do_t": col.data,
        "do_f16": value,
        "descale_do": reorder_scale(row.scale_inv),
        "descale_do_t": reorder_scale(col.scale_inv, True),
    }
