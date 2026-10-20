# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Adapt JAX MXFP8 quantization buffers to the shared attention graph contract."""

from functools import lru_cache, partial

import jax
import jax.numpy as jnp
from jax.experimental.custom_partitioning import custom_partitioning
from jax.sharding import NamedSharding, PartitionSpec
from jax.custom_batching import sequential_vmap

from transformer_engine.common.cudnn.mxfp8 import quantization_layouts, scale_reordering
from ..quantize import QuantizerFactory, QuantizeLayout, ScalingMode, ScaledTensor2x
from ..quantize.tensor import ScaledTensorFactory
from .quantization import quantize
from .base import _JAX_GSPMD_SUPPORTED


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


def quantize_qkv(qkv):
    """Prepare separate BSHD operands with independent BHSD row/column quantizations."""
    buffers = _quantize_qkv(tuple(jax.lax.stop_gradient(x) for x in qkv))
    results = []
    for value, parts in zip(qkv, buffers):
        tensors = []
        for i, columnwise in enumerate((False, True)):
            data, scale, amax = parts[3 * i : 3 * i + 3]
            tensors.append(
                ScaledTensorFactory.create_1x(
                    data,
                    scale,
                    amax,
                    scaling_mode=ScalingMode.MXFP8_1D_SCALING,
                    dq_dtype=value.dtype,
                    is_colwise=columnwise,
                )
            )
        results.append(ScaledTensor2x(tensors[0], tensors[1]))
    return tuple(results)


@sequential_vmap
def _quantize_qkv(qkv):
    results = []
    for value in qkv:
        pair = quantize_operand(value.transpose(0, 2, 1, 3), True, True)
        results.append(
            tuple(x for tensor in pair for x in (tensor.data, tensor.scale_inv, tensor.amax))
        )
    return tuple(results)


def prequantized_buffers(qkv, quantized_qkv, layout):
    """Validate and extract unswizzled BHSD MXFP8 operands without dequantizing them."""
    if layout.is_thd():
        raise ValueError("MXFP8 attention does not support THD format.")
    expected = 1 if layout.is_qkvpacked() else 2 if layout.is_kvpacked() else 3
    if len(qkv) != expected:
        raise ValueError(f"Expected {expected} model operands for {layout}, received {len(qkv)}.")
    if layout.is_qkvpacked():
        operands = tuple(qkv[0][..., i, :, :] for i in range(3))
    elif layout.is_kvpacked():
        operands = qkv[0], qkv[1][..., 0, :, :], qkv[1][..., 1, :, :]
    else:
        operands = qkv
    if len(quantized_qkv) != 3:
        raise ValueError("quantized_qkv must contain separate Q, K, and V ScaledTensor2x operands.")
    buffers = []
    for value, tensor in zip(operands, quantized_qkv):
        if value.ndim != 4 or not isinstance(tensor, ScaledTensor2x):
            raise ValueError("Prequantized attention requires BSHD arrays and BHSD ScaledTensor2x.")
        shape = (value.shape[0], value.shape[2], value.shape[1], value.shape[3])
        row, col = tensor.rowwise_tensor, tensor.colwise_tensor
        for part, columnwise in ((row, False), (col, True)):
            valid = (
                part.scaling_mode == ScalingMode.MXFP8_1D_SCALING,
                part.data.dtype == jnp.float8_e4m3fn,
                part.data.shape == shape,
                part.scale_inv.dtype == jnp.float8_e8m0fnu,
                part.flatten_axis == 3,
                part.is_colwise == columnwise,
                part.data_layout == "N",
                not part.has_rht_applied,
            )
            if not all(valid):
                raise ValueError(
                    "quantized_qkv requires unswizzled E4M3 MXFP8 BHSD data with independent "
                    "rowwise/columnwise scales and no Hadamard transform."
                )
        buffers.extend((row.data, col.data, row.scale_inv, col.scale_inv))
    return tuple(jax.lax.stop_gradient(x) for x in buffers)


def context_parallel_qkv(qkv, buffers, layout):
    """Unpack model storage and decode supplied inputs before CP communication.

    MXFP8 CP communicates model-precision values and quantizes the local Q/K/V
    slices independently for each contraction. As in PyTorch, prequantized
    inputs are decoded before communication. Gradients target the original
    model arrays, treating the quantization as a straight-through operation.
    """
    if layout.is_qkvpacked():
        qkv = tuple(qkv[0][..., i, :, :] for i in range(3))
    elif layout.is_kvpacked():
        qkv = qkv[0], qkv[1][..., 0, :, :], qkv[1][..., 1, :, :]
    if buffers is None:
        return qkv
    decoded = []
    for i, source in enumerate(qkv):
        data, _, scale, _ = buffers[4 * i : 4 * i + 4]
        value = data.astype(jnp.float32) * jnp.repeat(scale.astype(jnp.float32), 32, axis=-1)
        value = value.transpose(0, 2, 1, 3).astype(source.dtype)
        decoded.append(jax.lax.stop_gradient(value) + (source - jax.lax.stop_gradient(source)))
    return tuple(decoded)


def _prepared_values(buffers, backward):
    values = {}
    for i, name in enumerate(("q", "k", "v")):
        row, col, row_scale, col_scale = buffers[4 * i : 4 * i + 4]
        columnwise = name == "v" and not backward
        values[name] = col if columnwise else row
        values["descale_" + name] = reorder_scale(
            col_scale if columnwise else row_scale, columnwise
        )
        if backward and name != "v":
            values[name + "_t"] = col
            values["descale_" + name + "_t"] = reorder_scale(col_scale, True)
    return values


def _execute_prequantized(args, buffers, config, backward):
    # Local imports keep the quantization adapter independent of attention's
    # primitive registration order.
    from .attention import FusedAttnFwdPrimitive, FusedAttnBwdPrimitive, generate_cu_seqlen
    from . import cudnn_attention
    from ..attention import SequenceDescriptor

    base = FusedAttnBwdPrimitive if backward else FusedAttnFwdPrimitive
    result_avals = base.abstract(
        *(jax.core.ShapedArray(x.shape, x.dtype) for x in args), config=config
    )
    start = 9 if backward else 6
    descriptor = SequenceDescriptor(
        seqlens=args[start : start + 2],
        seq_offsets=args[start + 2 : start + 4],
        segment_ids=args[start + 4 : start + 6],
        segment_pos=args[start + 6 : start + 8],
    )
    lengths, _ = descriptor._get_seqlens_and_offsets(
        config.attn_mask_type,
        config.qkv_layout,
        config.window_size,
        config.max_segments_per_seq,
        allow_fast_causal_path=config.allow_fast_causal_path,
    )
    args = (*args[:start], *(generate_cu_seqlen(x.flatten()) for x in lengths), *args[start + 2 :])
    results = cudnn_attention.execute(
        "bwd" if backward else "fwd",
        args,
        config,
        result_avals,
        prepared_qkv=_prepared_values(buffers, backward),
    )
    if backward:
        return results[:5]
    output, stats, max_logit, rng, _ = results
    return output, stats, rng, max_logit


def execute_prequantized(args, buffers, config, backward):
    """Execute supplied quantizations with the standard attention gradient contract."""
    return _batched_executor(config, backward)(args, buffers)


@lru_cache(maxsize=256)
def _batched_executor(config, backward):
    # Independent vmap elements retain their per-head block boundaries. Use a
    # loop rather than the generic quantizer's flattening of mapped dimensions.
    @sequential_vmap
    def execute(args, buffers):
        return _partitioned_prequantized(args, buffers, config, backward)

    return execute


@partial(custom_partitioning, static_argnums=(2, 3))
def _partitioned_prequantized(args, buffers, config, backward):
    return _execute_prequantized(args, buffers, config, backward)


def _prequantized_partition(config, backward, mesh, arg_infos, result_infos):
    from .attention import FusedAttnFwdPrimitive, FusedAttnBwdPrimitive
    from ..sharding import get_mesh_axis_size
    from .misc import get_padded_spec

    if get_mesh_axis_size(config.cp_axis, mesh) > 1:
        raise ValueError("Prequantized MXFP8 context parallelism requires local requantization.")
    base = FusedAttnBwdPrimitive if backward else FusedAttnFwdPrimitive
    _, _, out_shardings, arg_shardings = base.partition(config, mesh, arg_infos[0], result_infos)
    data_shardings = []
    for index in range(3):
        source = (
            0
            if config.qkv_layout.is_qkvpacked()
            else min(index, 1) if config.qkv_layout.is_kvpacked() else index
        )
        spec = get_padded_spec(arg_infos[0][source])
        # Model inputs are BSHD (possibly packed); supplied buffers are BHSD.
        prepared = NamedSharding(mesh, PartitionSpec(spec[0], spec[-2], spec[1], spec[-1]))
        data_shardings.extend((prepared,) * 4)

    def impl(local_args, local_buffers):
        result = _execute_prequantized(local_args, local_buffers, config, backward)
        if backward and config.softmax_type.name == "LEARNABLE_SOFTMAX":
            from ..sharding import all_reduce_sum_along_dp_fsdp

            result = (*result[:4], all_reduce_sum_along_dp_fsdp(result[4], mesh))
        return result

    return mesh, impl, out_shardings, (arg_shardings, tuple(data_shardings))


def _prequantized_rule(config, backward, mesh, value_types, result_types):
    from .attention import FusedAttnFwdPrimitive, FusedAttnBwdPrimitive
    from jax.experimental.custom_partitioning import SdyShardingRule

    base = FusedAttnBwdPrimitive if backward else FusedAttnFwdPrimitive
    count = 17 if backward else 14
    rule = base.shardy_sharding_rule(config, mesh, value_types[:count], result_types)
    inputs = (*rule.operand_mappings, *((f"…{100 + i}",) for i in range(12)))
    return SdyShardingRule(inputs, rule.result_mappings)


def _prequantized_infer(config, backward, mesh, arg_infos, result_infos):
    from .attention import FusedAttnFwdPrimitive, FusedAttnBwdPrimitive

    base = FusedAttnBwdPrimitive if backward else FusedAttnFwdPrimitive
    return base.infer_sharding_from_operands(config, mesh, arg_infos[0], result_infos)


_partitioned_prequantized.def_partition(
    partition=_prequantized_partition,
    sharding_rule=_prequantized_rule,
    **({"infer_sharding_from_operands": _prequantized_infer} if _JAX_GSPMD_SUPPORTED else {}),
)
