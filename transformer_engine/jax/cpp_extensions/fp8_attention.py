# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""JAX execution adapter for common cuDNN FP8 attention graph construction."""

from __future__ import annotations

import copy
import operator
import os
from dataclasses import dataclass
from functools import reduce
from typing import Any

import jax
import jax.numpy as jnp
from jax import ffi

from transformer_engine.common.attention.cudnn import (
    FusedAttentionConfig,
    check_fp8_fused_attention_support,
)
from transformer_engine.common.attention.fp8 import (
    FP8AttentionGraphConfig,
    attention_format_stride,
    build_fp8_backward_operation,
    build_fp8_forward_operation,
    mxfp8_padded_sizes,
)

from ..quantize import ScalingMode, TensorUsage, swizzle_mxfp8_scale
from .attention import _FusedAttnRNGStateChecker
from .cudnn_attention import (
    _UID_DK,
    _UID_DO,
    _UID_DQ,
    _UID_DROPOUT_OFFSET,
    _UID_DROPOUT_SEED,
    _UID_DV,
    _UID_K,
    _UID_O,
    _UID_Q,
    _UID_SEQ_KV,
    _UID_SEQ_Q,
    _UID_STATS,
    _UID_V,
    _device_arch,
    _is_dropout,
    _is_padding,
    _layout_info,
    _mask_options,
    _matrix_stride,
    _policy_layout,
    _policy_mask_name,
    _qkv_bindings,
)
from .cudnn_graph import (
    GraphBinding,
    SerializedGraph,
    cudnn_data_type,
    dtype_name,
    finalize_graph,
    import_cudnn,
    make_graph,
    serialized_graph,
)
from .misc import get_cudnn_version
from .quantization import quantize

__all__ = ["FP8AttentionConfig", "fused_attn_fp8_bwd", "fused_attn_fp8_fwd"]


@dataclass(frozen=True)
class FP8AttentionConfig:
    """Static configuration for JAX FP8 dot-product attention."""

    attn_bias_type: Any
    attn_mask_type: Any
    softmax_type: Any
    qkv_layout: Any
    scaling_factor: float
    dropout_probability: float
    is_training: bool
    window_size: tuple[int, int]
    bottom_right_diagonal: bool = False


_UID_DESCALE_Q = 101
_UID_DESCALE_K = 102
_UID_DESCALE_V = 103
_UID_DESCALE_S = 104
_UID_SCALE_S = 105
_UID_SCALE_O = 106
_UID_AMAX_S = 107
_UID_AMAX_O = 108
_UID_DESCALE_O = 109
_UID_DESCALE_DO = 110
_UID_DESCALE_DP = 111
_UID_SCALE_DQ = 112
_UID_SCALE_DK = 113
_UID_SCALE_DV = 114
_UID_SCALE_DP = 115
_UID_AMAX_DQ = 116
_UID_AMAX_DK = 117
_UID_AMAX_DV = 118
_UID_AMAX_DP = 119
_UID_Q_T = 120
_UID_K_T = 121
_UID_DO_T = 122
_UID_DESCALE_Q_T = 123
_UID_DESCALE_K_T = 124
_UID_DESCALE_DO_T = 125
_UID_DO_F16 = 126


@dataclass(frozen=True)
class FP8AttentionGraphInfo:
    """Serialized graph and result metadata for an FP8 attention call."""

    graph: SerializedGraph
    output_shape: tuple[int, ...]
    q_shape: tuple[int, ...]
    k_shape: tuple[int, ...]
    v_shape: tuple[int, ...]


_graph_cache: dict[tuple[Any, ...], FP8AttentionGraphInfo] = {}


def _cache_key(direction, mode, avals, config, output_dtype):
    return (
        direction,
        mode,
        config,
        dtype_name(output_dtype),
        tuple((tuple(aval.shape), dtype_name(aval.dtype)) for aval in avals),
        get_cudnn_version(),
        _device_arch(),
    )


def _tensor(graph, *, name, dim, stride, dtype, uid):
    return graph.tensor(
        name=name,
        dim=tuple(int(value) for value in dim),
        stride=tuple(int(value) for value in stride),
        data_type=dtype,
        uid=uid,
    )


def _scalar(graph, cudnn, name, uid):
    return _tensor(
        graph,
        name=name,
        dim=(1, 1, 1, 1),
        stride=(1, 1, 1, 1),
        dtype=cudnn.data_type.FLOAT,
        uid=uid,
    )


def _mx_scale(
    graph,
    cudnn,
    *,
    name,
    uid,
    batch,
    heads,
    seqlen,
    dim,
):
    # _mxfp8_scale_inv transposes every compact scale buffer to contiguous BHSD
    # before applying cuDNN's F8_128x4 physical reordering.
    return _tensor(
        graph,
        name=name,
        dim=(batch, heads, seqlen, dim),
        stride=attention_format_stride(batch, heads, seqlen, dim, "bhsd"),
        dtype=cudnn.data_type.FP8_E8M0,
        uid=uid,
    ).set_reordering_type(cudnn.tensor_reordering.F8_128x4)


def _logical_shapes(info):
    q = (*info.batch_shape, info.q_max_seqlen, info.q_heads, info.qk_dim)
    k = (*info.batch_shape, info.kv_max_seqlen, info.kv_heads, info.qk_dim)
    v = (*info.batch_shape, info.kv_max_seqlen, info.kv_heads, info.v_dim)
    o = (*info.batch_shape, info.q_max_seqlen, info.q_heads, info.v_dim)
    return q, k, v, o


def _graph_io_tensors(graph, cudnn, q_aval, k_aval, v_aval, config):
    info = _layout_info(q_aval, k_aval, v_aval, config.qkv_layout)
    io_dtype = cudnn_data_type(cudnn, q_aval.dtype)
    q = _tensor(
        graph,
        name="Q",
        dim=(info.input_batch, info.q_heads, info.q_max_seqlen, info.qk_dim),
        stride=_matrix_stride(
            info, config.qkv_layout, "q", info.q_max_seqlen, info.kv_max_seqlen
        ),
        dtype=io_dtype,
        uid=_UID_Q,
    )
    k = _tensor(
        graph,
        name="K",
        dim=(info.input_batch, info.kv_heads, info.kv_max_seqlen, info.qk_dim),
        stride=_matrix_stride(
            info, config.qkv_layout, "k", info.q_max_seqlen, info.kv_max_seqlen
        ),
        dtype=io_dtype,
        uid=_UID_K,
    )
    v = _tensor(
        graph,
        name="V",
        dim=(info.input_batch, info.kv_heads, info.kv_max_seqlen, info.v_dim),
        stride=_matrix_stride(
            info, config.qkv_layout, "v", info.q_max_seqlen, info.kv_max_seqlen
        ),
        dtype=io_dtype,
        uid=_UID_V,
    )
    return info, io_dtype, q, k, v


def _fp8_options(cudnn, info, config):
    options = _mask_options(cudnn, info, config)
    is_padding = options.pop("is_padding", _is_padding(config))
    if "diagonal_band_left_bound" in options:
        options["left_bound"] = options.pop("diagonal_band_left_bound")
    if "diagonal_band_right_bound" in options:
        options["right_bound"] = options.pop("diagonal_band_right_bound")
    options.update(
        attn_scale=float(config.scaling_factor),
        use_padding_mask=is_padding,
    )
    return options, is_padding


def build_fp8_fwd_graph(
    q_aval,
    k_aval,
    v_aval,
    q_scale_aval,
    k_scale_aval,
    v_scale_aval,
    config,
    mode: str,
    output_dtype,
) -> FP8AttentionGraphInfo:
    """Build or retrieve a dense JAX FP8 attention forward graph."""

    avals = (q_aval, k_aval, v_aval, q_scale_aval, k_scale_aval, v_scale_aval)
    key = _cache_key("fwd", mode, avals, config, output_dtype)
    if key in _graph_cache:
        return _graph_cache[key]

    cudnn = import_cudnn()
    graph = make_graph(cudnn, cudnn_data_type(cudnn, q_aval.dtype))
    info, _, q, k, v = _graph_io_tensors(graph, cudnn, q_aval, k_aval, v_aval, config)
    q_shape, k_shape, v_shape, o_shape = _logical_shapes(info)
    input_bindings = list(
        _qkv_bindings(info, config.qkv_layout, jnp.dtype(q_aval.dtype).itemsize)
    )
    tensors = {"q": q, "k": k, "v": v}
    options, is_padding = _fp8_options(cudnn, info, config)
    options["generate_stats"] = True

    if mode == "mxfp8":
        if is_padding:
            raise ValueError("JAX MXFP8 attention does not support padding masks.")
        options.pop("use_padding_mask", None)
        # The current cuDNN MXFP8 forward binding uses diagonal_band_* names.
        if "left_bound" in options:
            options["diagonal_band_left_bound"] = options.pop("left_bound")
        if "right_bound" in options:
            options["diagonal_band_right_bound"] = options.pop("right_bound")
        padded = mxfp8_padded_sizes(
            info.q_max_seqlen, info.kv_max_seqlen, info.qk_dim, info.v_dim
        )
        scale_specs = (
            (
                "descale_q",
                _UID_DESCALE_Q,
                info.q_heads,
                "s_q_padded",
                "d_qk_scale_padded",
                3,
            ),
            (
                "descale_k",
                _UID_DESCALE_K,
                info.kv_heads,
                "s_kv_padded",
                "d_qk_scale_padded",
                4,
            ),
            (
                "descale_v",
                _UID_DESCALE_V,
                info.kv_heads,
                "s_kv_scale_padded",
                "d_v_padded",
                6,
            ),
        )
        for name, uid, heads, s_key, d_key, buffer_index in scale_specs:
            tensors[name] = _mx_scale(
                graph,
                cudnn,
                name=name,
                uid=uid,
                batch=info.input_batch,
                heads=heads,
                seqlen=padded[s_key],
                dim=padded[d_key],
            )
            input_bindings.append(GraphBinding(uid, buffer_index))
    else:
        for name, uid, index in (
            ("descale_q", _UID_DESCALE_Q, 3),
            ("descale_k", _UID_DESCALE_K, 4),
            ("descale_v", _UID_DESCALE_V, 6),
            ("descale_s", _UID_DESCALE_S, 7),
            ("scale_s", _UID_SCALE_S, 8),
            ("scale_o", _UID_SCALE_O, 9),
        ):
            tensors[name] = _scalar(graph, cudnn, name, uid)
            input_bindings.append(GraphBinding(uid, index))

    if is_padding:
        seq_q = _tensor(
            graph,
            name="seq_len_q",
            dim=(info.input_batch, 1, 1, 1),
            stride=(1, 1, 1, 1),
            dtype=cudnn.data_type.INT32,
            uid=_UID_SEQ_Q,
        )
        seq_kv = _tensor(
            graph,
            name="seq_len_kv",
            dim=(info.input_batch, 1, 1, 1),
            stride=(1, 1, 1, 1),
            dtype=cudnn.data_type.INT32,
            uid=_UID_SEQ_KV,
        )
        options.update(seq_len_q=seq_q, seq_len_kv=seq_kv)
        input_bindings.extend(
            (GraphBinding(_UID_SEQ_Q, 10), GraphBinding(_UID_SEQ_KV, 11))
        )

    output_bindings = [GraphBinding(_UID_O, 0), GraphBinding(_UID_STATS, 1)]
    if _is_dropout(config):
        seed = _tensor(
            graph,
            name="dropout_seed",
            dim=(1, 1, 1, 1),
            stride=(1, 1, 1, 1),
            dtype=cudnn.data_type.INT64,
            uid=_UID_DROPOUT_SEED,
        )
        offset = _tensor(
            graph,
            name="dropout_offset",
            dim=(1, 1, 1, 1),
            stride=(1, 1, 1, 1),
            dtype=cudnn.data_type.INT64,
            uid=_UID_DROPOUT_OFFSET,
        )
        options["dropout"] = (float(config.dropout_probability), seed, offset)
        output_bindings.extend(
            (
                GraphBinding(_UID_DROPOUT_SEED, 3),
                GraphBinding(_UID_DROPOUT_OFFSET, 3, 8),
            )
        )

    op = build_fp8_forward_operation(
        graph,
        tensors,
        options,
        FP8AttentionGraphConfig(mode, "te_jax_fp8_sdpa_forward"),
    )
    output = op["output"]
    output.set_output(True).set_uid(_UID_O).set_data_type(
        cudnn_data_type(cudnn, output_dtype)
    ).set_dim(
        (info.input_batch, info.q_heads, info.q_max_seqlen, info.v_dim)
    ).set_stride(
        attention_format_stride(
            info.input_batch, info.q_heads, info.q_max_seqlen, info.v_dim, "bshd"
        )
    )
    stats = op["stats"]
    stats.set_output(True).set_uid(_UID_STATS).set_data_type(
        cudnn.data_type.FLOAT
    ).set_dim((info.input_batch, info.q_heads, info.q_max_seqlen, 1)).set_stride(
        (info.q_heads * info.q_max_seqlen, info.q_max_seqlen, 1, 1)
    )
    if mode != "mxfp8":
        for name, uid, offset in (
            ("amax_s", _UID_AMAX_S, 0),
            ("amax_o", _UID_AMAX_O, 4),
        ):
            op[name].set_output(True).set_uid(uid).set_data_type(
                cudnn.data_type.FLOAT
            ).set_dim((1, 1, 1, 1)).set_stride((1, 1, 1, 1))
            output_bindings.append(GraphBinding(uid, 2, offset))
    else:
        op["amax_o"].set_output(False).set_data_type(cudnn.data_type.FLOAT).set_dim(
            (1, 1, 1, 1)
        ).set_stride((1, 1, 1, 1))

    workspace, data, version = finalize_graph(
        cudnn, graph, description=f"JAX {mode} FP8 attention forward"
    )
    result = serialized_graph(
        serialized_graph_data=data,
        cudnn_frontend_version=version,
        workspace_size=workspace,
        input_bindings=input_bindings,
        output_bindings=output_bindings,
    )
    info_result = FP8AttentionGraphInfo(result, o_shape, q_shape, k_shape, v_shape)
    _graph_cache[key] = info_result
    return info_result


def _mx_bwd_scales(graph, cudnn, info, padded, input_bindings):
    specs = (
        (
            "descale_q",
            _UID_DESCALE_Q,
            info.q_heads,
            "s_q_padded",
            "d_qk_scale_padded",
            11,
        ),
        (
            "descale_q_t",
            _UID_DESCALE_Q_T,
            info.q_heads,
            "s_q_scale_padded",
            "d_qk_padded",
            25,
        ),
        (
            "descale_k",
            _UID_DESCALE_K,
            info.kv_heads,
            "s_kv_padded",
            "d_qk_scale_padded",
            12,
        ),
        (
            "descale_k_t",
            _UID_DESCALE_K_T,
            info.kv_heads,
            "s_kv_scale_padded",
            "d_qk_padded",
            26,
        ),
        (
            "descale_v",
            _UID_DESCALE_V,
            info.kv_heads,
            "s_kv_padded",
            "d_v_scale_padded",
            13,
        ),
        (
            "descale_do",
            _UID_DESCALE_DO,
            info.q_heads,
            "s_q_padded",
            "d_v_scale_padded",
            15,
        ),
        (
            "descale_do_t",
            _UID_DESCALE_DO_T,
            info.q_heads,
            "s_q_scale_padded",
            "d_v_padded",
            24,
        ),
    )
    tensors = {}
    for name, uid, heads, s_key, d_key, index in specs:
        tensors[name] = _mx_scale(
            graph,
            cudnn,
            name=name,
            uid=uid,
            batch=info.input_batch,
            heads=heads,
            seqlen=padded[s_key],
            dim=padded[d_key],
        )
        input_bindings.append(GraphBinding(uid, index))
    return tensors


def build_fp8_bwd_graph(
    q_aval,
    k_aval,
    v_aval,
    stats_aval,
    output_aval,
    doutput_aval,
    config,
    mode: str,
    grad_dtype,
) -> FP8AttentionGraphInfo:
    """Build or retrieve a dense JAX FP8 attention backward graph."""

    avals = (q_aval, k_aval, v_aval, stats_aval, output_aval, doutput_aval)
    key = _cache_key("bwd", mode, avals, config, grad_dtype)
    if key in _graph_cache:
        return _graph_cache[key]

    cudnn = import_cudnn()
    graph = make_graph(cudnn, cudnn_data_type(cudnn, q_aval.dtype))
    info, io_dtype, q, k, v = _graph_io_tensors(
        graph, cudnn, q_aval, k_aval, v_aval, config
    )
    q_shape, k_shape, v_shape, o_shape = _logical_shapes(info)
    input_bindings = list(
        _qkv_bindings(info, config.qkv_layout, jnp.dtype(q_aval.dtype).itemsize)
    )
    o = _tensor(
        graph,
        name="O",
        dim=(info.input_batch, info.q_heads, info.q_max_seqlen, info.v_dim),
        stride=attention_format_stride(
            info.input_batch, info.q_heads, info.q_max_seqlen, info.v_dim, "bshd"
        ),
        dtype=cudnn_data_type(cudnn, output_aval.dtype),
        uid=_UID_O,
    )
    do = _tensor(
        graph,
        name="dO",
        dim=(info.input_batch, info.q_heads, info.q_max_seqlen, info.v_dim),
        stride=attention_format_stride(
            info.input_batch, info.q_heads, info.q_max_seqlen, info.v_dim, "bshd"
        ),
        dtype=cudnn_data_type(cudnn, doutput_aval.dtype),
        uid=_UID_DO,
    )
    stats = _tensor(
        graph,
        name="Stats",
        dim=(info.input_batch, info.q_heads, info.q_max_seqlen, 1),
        stride=(info.q_heads * info.q_max_seqlen, info.q_max_seqlen, 1, 1),
        dtype=cudnn.data_type.FLOAT,
        uid=_UID_STATS,
    )
    input_bindings.extend(
        (GraphBinding(_UID_STATS, 5), GraphBinding(_UID_O, 7), GraphBinding(_UID_DO, 8))
    )
    tensors = {"q": q, "k": k, "v": v, "o": o, "do": do, "stats": stats}
    options, is_padding = _fp8_options(cudnn, info, config)
    options["use_deterministic_algorithm"] = not bool(
        int(os.getenv("NVTE_ALLOW_NONDETERMINISTIC_ALGO", "1"))
    )
    if is_padding:
        seq_q = _tensor(
            graph,
            name="seq_len_q",
            dim=(info.input_batch, 1, 1, 1),
            stride=(1, 1, 1, 1),
            dtype=cudnn.data_type.INT32,
            uid=_UID_SEQ_Q,
        )
        seq_kv = _tensor(
            graph,
            name="seq_len_kv",
            dim=(info.input_batch, 1, 1, 1),
            stride=(1, 1, 1, 1),
            dtype=cudnn.data_type.INT32,
            uid=_UID_SEQ_KV,
        )
        options.update(seq_len_q=seq_q, seq_len_kv=seq_kv)
        input_bindings.extend(
            (GraphBinding(_UID_SEQ_Q, 9), GraphBinding(_UID_SEQ_KV, 10))
        )

    if _is_dropout(config):
        seed = _tensor(
            graph,
            name="dropout_seed",
            dim=(1, 1, 1, 1),
            stride=(1, 1, 1, 1),
            dtype=cudnn.data_type.INT64,
            uid=_UID_DROPOUT_SEED,
        )
        offset = _tensor(
            graph,
            name="dropout_offset",
            dim=(1, 1, 1, 1),
            stride=(1, 1, 1, 1),
            dtype=cudnn.data_type.INT64,
            uid=_UID_DROPOUT_OFFSET,
        )
        options["dropout"] = (float(config.dropout_probability), seed, offset)
        input_bindings.extend(
            (
                GraphBinding(_UID_DROPOUT_SEED, 6),
                GraphBinding(_UID_DROPOUT_OFFSET, 6, 8),
            )
        )

    if mode == "mxfp8":
        if is_padding:
            raise ValueError("JAX MXFP8 attention does not support padding masks.")
        q_t = _tensor(
            graph,
            name="Q_T",
            dim=(info.input_batch, info.q_heads, info.q_max_seqlen, info.qk_dim),
            stride=attention_format_stride(
                info.input_batch, info.q_heads, info.q_max_seqlen, info.qk_dim, "bshd"
            ),
            dtype=io_dtype,
            uid=_UID_Q_T,
        )
        k_t = _tensor(
            graph,
            name="K_T",
            dim=(info.input_batch, info.kv_heads, info.kv_max_seqlen, info.qk_dim),
            stride=attention_format_stride(
                info.input_batch, info.kv_heads, info.kv_max_seqlen, info.qk_dim, "bshd"
            ),
            dtype=io_dtype,
            uid=_UID_K_T,
        )
        do_t = _tensor(
            graph,
            name="dO_T",
            dim=(info.input_batch, info.q_heads, info.q_max_seqlen, info.v_dim),
            stride=attention_format_stride(
                info.input_batch, info.q_heads, info.q_max_seqlen, info.v_dim, "bshd"
            ),
            dtype=cudnn_data_type(cudnn, doutput_aval.dtype),
            uid=_UID_DO_T,
        )
        do_f16 = _tensor(
            graph,
            name="dO_f16",
            dim=(info.input_batch, info.q_heads, info.q_max_seqlen, info.v_dim),
            stride=attention_format_stride(
                info.input_batch, info.q_heads, info.q_max_seqlen, info.v_dim, "bshd"
            ),
            dtype=cudnn_data_type(cudnn, output_aval.dtype),
            uid=_UID_DO_F16,
        )
        tensors.update(q_t=q_t, k_t=k_t, do_t=do_t, do_f16=do_f16)
        input_bindings.extend(
            (
                GraphBinding(_UID_Q_T, 3),
                GraphBinding(_UID_K_T, 4),
                GraphBinding(_UID_DO_T, 23),
                GraphBinding(_UID_DO_F16, 27),
            )
        )
        padded = mxfp8_padded_sizes(
            info.q_max_seqlen, info.kv_max_seqlen, info.qk_dim, info.v_dim
        )
        tensors.update(_mx_bwd_scales(graph, cudnn, info, padded, input_bindings))
    else:
        for name, uid, index in (
            ("descale_q", _UID_DESCALE_Q, 11),
            ("descale_k", _UID_DESCALE_K, 12),
            ("descale_v", _UID_DESCALE_V, 13),
            ("descale_o", _UID_DESCALE_O, 14),
            ("descale_do", _UID_DESCALE_DO, 15),
            ("descale_s", _UID_DESCALE_S, 16),
            ("descale_dp", _UID_DESCALE_DP, 17),
            ("scale_s", _UID_SCALE_S, 18),
            ("scale_dq", _UID_SCALE_DQ, 19),
            ("scale_dk", _UID_SCALE_DK, 20),
            ("scale_dv", _UID_SCALE_DV, 21),
            ("scale_dp", _UID_SCALE_DP, 22),
        ):
            tensors[name] = _scalar(graph, cudnn, name, uid)
            input_bindings.append(GraphBinding(uid, index))

    op = build_fp8_backward_operation(
        graph,
        tensors,
        options,
        FP8AttentionGraphConfig(mode, "te_jax_fp8_sdpa_backward"),
    )
    output_bindings = []
    for name, uid, index, shape in (
        ("dq", _UID_DQ, 0, q_shape),
        ("dk", _UID_DK, 1, k_shape),
        ("dv", _UID_DV, 2, v_shape),
    ):
        batch = reduce(operator.mul, shape[:-3], 1)
        seqlen, heads, dim = shape[-3:]
        op[name].set_output(True).set_uid(uid).set_data_type(
            cudnn_data_type(cudnn, grad_dtype)
        ).set_dim((batch, heads, seqlen, dim)).set_stride(
            attention_format_stride(batch, heads, seqlen, dim, "bshd")
        )
        output_bindings.append(GraphBinding(uid, index))

    if mode != "mxfp8":
        for name, uid, offset in (
            ("amax_dq", _UID_AMAX_DQ, 0),
            ("amax_dk", _UID_AMAX_DK, 4),
            ("amax_dv", _UID_AMAX_DV, 8),
            ("amax_dp", _UID_AMAX_DP, 12),
        ):
            op[name].set_output(True).set_uid(uid).set_data_type(
                cudnn.data_type.FLOAT
            ).set_dim((1, 1, 1, 1)).set_stride((1, 1, 1, 1))
            output_bindings.append(GraphBinding(uid, 3, offset))
    else:
        for amax in op["amax"]:
            amax.set_output(False).set_data_type(cudnn.data_type.FLOAT).set_dim(
                (1, 1, 1, 1)
            ).set_stride((1, 1, 1, 1))

    workspace, data, version = finalize_graph(
        cudnn, graph, description=f"JAX {mode} FP8 attention backward"
    )
    result = serialized_graph(
        serialized_graph_data=data,
        cudnn_frontend_version=version,
        workspace_size=workspace,
        input_bindings=input_bindings,
        output_bindings=output_bindings,
    )
    graph_info = FP8AttentionGraphInfo(result, o_shape, q_shape, k_shape, v_shape)
    _graph_cache[key] = graph_info
    return graph_info


def execute_fp8_fwd(
    q,
    k,
    v,
    q_scale_inv,
    k_scale_inv,
    v_scale_inv,
    s_scale_inv,
    s_scale,
    o_scale,
    q_seqlen,
    kv_seqlen,
    seed,
    *,
    config,
    mode,
    output_dtype,
):
    """Execute a serialized dense FP8 attention forward graph."""

    graph_info = build_fp8_fwd_graph(
        jax.ShapeDtypeStruct(q.shape, q.dtype),
        jax.ShapeDtypeStruct(k.shape, k.dtype),
        jax.ShapeDtypeStruct(v.shape, v.dtype),
        jax.ShapeDtypeStruct(q_scale_inv.shape, q_scale_inv.dtype),
        jax.ShapeDtypeStruct(k_scale_inv.shape, k_scale_inv.dtype),
        jax.ShapeDtypeStruct(v_scale_inv.shape, v_scale_inv.dtype),
        config,
        mode,
        output_dtype,
    )
    result_specs = (
        jax.ShapeDtypeStruct(graph_info.output_shape, output_dtype),
        jax.ShapeDtypeStruct(
            (
                *graph_info.output_shape[:-3],
                graph_info.output_shape[-2],
                graph_info.output_shape[-3],
                1,
            ),
            jnp.float32,
        ),
        jax.ShapeDtypeStruct((2,), jnp.float32),
        jax.ShapeDtypeStruct((seed.shape[0], 4), jnp.uint32),
        jax.ShapeDtypeStruct((graph_info.graph.workspace_size,), jnp.uint8),
    )
    return ffi.ffi_call("te_fused_attn_forward_ffi", result_specs)(
        q,
        k,
        v,
        q_scale_inv,
        k_scale_inv,
        seed,
        v_scale_inv,
        s_scale_inv,
        s_scale,
        o_scale,
        q_seqlen,
        kv_seqlen,
        is_ragged=False,
        rng_offset_increment=16,
        **graph_info.graph.ffi_attrs(),
    )


def execute_fp8_bwd(
    q,
    k,
    v,
    q_t,
    k_t,
    stats,
    rng_state,
    output,
    doutput,
    q_seqlen,
    kv_seqlen,
    q_scale_inv,
    k_scale_inv,
    v_scale_inv,
    o_scale_inv,
    do_scale_inv,
    s_scale_inv,
    dp_scale_inv,
    s_scale,
    dq_scale,
    dk_scale,
    dv_scale,
    dp_scale,
    do_t,
    do_scale_inv_t,
    q_scale_inv_t,
    k_scale_inv_t,
    doutput_f16,
    *,
    config,
    mode,
    grad_dtype,
):
    """Execute a serialized dense FP8 attention backward graph."""

    graph_info = build_fp8_bwd_graph(
        jax.ShapeDtypeStruct(q.shape, q.dtype),
        jax.ShapeDtypeStruct(k.shape, k.dtype),
        jax.ShapeDtypeStruct(v.shape, v.dtype),
        jax.ShapeDtypeStruct(stats.shape, stats.dtype),
        jax.ShapeDtypeStruct(output.shape, output.dtype),
        jax.ShapeDtypeStruct(doutput.shape, doutput.dtype),
        config,
        mode,
        grad_dtype,
    )
    result_specs = (
        jax.ShapeDtypeStruct(graph_info.q_shape, grad_dtype),
        jax.ShapeDtypeStruct(graph_info.k_shape, grad_dtype),
        jax.ShapeDtypeStruct(graph_info.v_shape, grad_dtype),
        jax.ShapeDtypeStruct((4,), jnp.float32),
        jax.ShapeDtypeStruct((0,), output.dtype),
        jax.ShapeDtypeStruct((graph_info.graph.workspace_size,), jnp.uint8),
    )
    return ffi.ffi_call("te_fused_attn_backward_ffi", result_specs)(
        q,
        k,
        v,
        q_t,
        k_t,
        stats,
        rng_state,
        output,
        doutput,
        q_seqlen,
        kv_seqlen,
        q_scale_inv,
        k_scale_inv,
        v_scale_inv,
        o_scale_inv,
        do_scale_inv,
        s_scale_inv,
        dp_scale_inv,
        s_scale,
        dq_scale,
        dk_scale,
        dv_scale,
        dp_scale,
        do_t,
        do_scale_inv_t,
        q_scale_inv_t,
        k_scale_inv_t,
        doutput_f16,
        is_ragged=False,
        **graph_info.graph.ffi_attrs(),
    )


def _scaling_mode(quantizer) -> str:
    mode = quantizer.scaling_mode
    if mode == ScalingMode.DELAYED_TENSOR_SCALING:
        return "delayed"
    if mode == ScalingMode.CURRENT_TENSOR_SCALING:
        return "current"
    if mode == ScalingMode.MXFP8_1D_SCALING:
        return "mxfp8"
    raise ValueError(f"FP8 attention does not support scaling mode {mode}.")


def _rowwise(tensor):
    return tensor.get_tensor(TensorUsage.LHS)


def _colwise(tensor):
    return tensor.get_tensor(TensorUsage.RHS)


def _quantize_many(values, quantizer, *, both=False):
    tensors = []
    amaxes = []
    for value in values:
        local_quantizer = copy.copy(quantizer)
        tensor = quantize(value, quantizer=local_quantizer, flatten_axis=-2)
        tensors.append(tensor)
        rowwise_amax = _rowwise(tensor).amax
        if rowwise_amax is not None:
            amaxes.append(rowwise_amax)
    if quantizer.scaling_mode == ScalingMode.DELAYED_TENSOR_SCALING and amaxes:
        quantizer.update(jnp.max(jnp.stack(amaxes)))
    if both:
        # Access both layouts here so an invalid recipe/layout fails before graph construction.
        for tensor in tensors:
            _rowwise(tensor)
            _colwise(tensor)
    return tuple(tensors)


def _quantized_operands(qkv, layout, quantizer, *, both=False):
    quantized = _quantize_many(qkv, quantizer, both=both)
    empty = jnp.zeros((0,), dtype=quantizer.q_dtype)
    if layout.is_qkvpacked():
        tensor = quantized[0]
        return (tensor, tensor, tensor), (_rowwise(tensor).data, empty, empty)
    if layout.is_kvpacked():
        q_tensor, kv_tensor = quantized
        return (
            q_tensor,
            kv_tensor,
            kv_tensor,
        ), (_rowwise(q_tensor).data, _rowwise(kv_tensor).data, empty)
    if layout.is_separate():
        return quantized, tuple(_rowwise(tensor).data for tensor in quantized)
    raise ValueError(f"FP8 attention does not support layout {layout}.")


def _scale_inv(tensor, *, colwise=False):
    return (_colwise(tensor) if colwise else _rowwise(tensor)).scale_inv


def _mxfp8_scale_inv(tensor, *, colwise=False):
    """Prepare a compact BSHD MXFP8 scale tensor for cuDNN's F8_128x4 layout."""

    component = _colwise(tensor) if colwise else _rowwise(tensor)
    *batch_shape, seqlen, heads, dim = component.data.shape
    batch = reduce(operator.mul, batch_shape, 1)
    if colwise:
        scale = component.scale_inv.reshape(batch, seqlen // 32, heads, dim)
        target_seqlen = ((seqlen + 127) // 128) * 4
        target_dim = ((dim + 127) // 128) * 128
    else:
        scale = component.scale_inv.reshape(batch, seqlen, heads, dim // 32)
        target_seqlen = ((seqlen + 127) // 128) * 128
        target_dim = ((dim + 127) // 128) * 4
    scale = jnp.pad(
        scale,
        (
            (0, 0),
            (0, target_seqlen - scale.shape[1]),
            (0, 0),
            (0, target_dim - scale.shape[3]),
        ),
        mode="constant",
        constant_values=2**-127,
    )
    scale = jnp.transpose(scale, (0, 2, 1, 3))
    return swizzle_mxfp8_scale(scale, -1, colwise)


def _tensor_scale(quantizer):
    if quantizer.scaling_mode == ScalingMode.DELAYED_TENSOR_SCALING:
        return quantizer.scale
    return jnp.ones((1,), dtype=jnp.float32)


def _tensor_scale_inv(quantizer):
    return jnp.reciprocal(_tensor_scale(quantizer))


def _graph_scale_inv(tensor, mode, *, colwise=False):
    if mode == "mxfp8":
        return _mxfp8_scale_inv(tensor, colwise=colwise)
    return _scale_inv(tensor, colwise=colwise)


def _sequence_lengths(sequence_descriptor, config):
    (q_seqlen, kv_seqlen), _ = sequence_descriptor.get_seqlens_and_offsets(
        config.attn_mask_type,
        config.qkv_layout,
        config.window_size,
        1,
    )
    return q_seqlen.flatten(), kv_seqlen.flatten()


def _validate_fp8_support(qkv, quantizers, config, mode):
    if config.qkv_layout.is_qkvpacked():
        q = k = v = qkv[0]
    elif config.qkv_layout.is_kvpacked():
        q, k = qkv
        v = k
    else:
        q, k, v = qkv
    info = _layout_info(q, k, v, config.qkv_layout)
    q_dtype = jnp.dtype(quantizers.qkv.q_dtype)
    dtype_name_ = {
        jnp.dtype(jnp.float8_e4m3fn): "float8_e4m3",
        jnp.dtype(jnp.float8_e5m2): "float8_e5m2",
    }.get(q_dtype, str(q_dtype))
    support = check_fp8_fused_attention_support(
        FusedAttentionConfig(
            is_training=bool(config.is_training),
            q_dtype=dtype_name_,
            kv_dtype=dtype_name_,
            layout=_policy_layout(config.qkv_layout),
            bias_type=config.attn_bias_type.name.lower(),
            mask_type=_policy_mask_name(config.attn_mask_type),
            softmax_type=config.softmax_type.name.lower().removesuffix("_softmax"),
            dropout=float(config.dropout_probability),
            num_attn_heads=info.q_heads,
            num_gqa_groups=info.kv_heads,
            max_seqlen_q=info.q_max_seqlen,
            max_seqlen_kv=info.kv_max_seqlen,
            head_dim_qk=info.qk_dim,
            head_dim_v=info.v_dim,
            window_size=tuple(int(value) for value in config.window_size),
            return_max_logit=False,
            cuda_graph=False,
            deterministic=not bool(
                int(os.getenv("NVTE_ALLOW_NONDETERMINISTIC_ALGO", "1"))
            ),
            cudnn_version=get_cudnn_version(),
            sm_arch=_device_arch(),
        )
    )
    if not support.supported:
        raise ValueError(
            f"Unsupported JAX FP8 attention configuration: {support.reason}."
        )
    if mode == "mxfp8" and (get_cudnn_version() < (9, 21, 0) or _device_arch() < 100):
        raise ValueError("MXFP8 attention requires cuDNN 9.21 and SM100 or newer.")


def fused_attn_fp8_fwd(qkv, sequence_descriptor, seed, quantizers, config):
    """Quantize high-precision inputs and execute dense FP8 attention forward."""

    mode = _scaling_mode(quantizers.qkv)
    if any(
        _scaling_mode(q) != mode
        for q in (
            quantizers.s,
            quantizers.o,
            quantizers.do,
            quantizers.dp,
            quantizers.dqkv,
        )
    ):
        raise ValueError("All FP8 attention quantizers must use the same scaling mode.")
    if config.qkv_layout.is_thd():
        raise NotImplementedError("FP8 attention does not support THD layouts in JAX.")
    if mode == "mxfp8" and not config.qkv_layout.is_separate():
        raise NotImplementedError(
            "JAX MXFP8 attention currently requires separate BSHD Q/K/V."
        )
    if getattr(config.attn_bias_type, "name", "") != "NO_BIAS":
        raise NotImplementedError("FP8 attention does not support attention bias.")
    if getattr(config.softmax_type, "name", "") != "VANILLA_SOFTMAX":
        raise NotImplementedError(
            "JAX FP8 attention currently supports vanilla softmax only."
        )
    _validate_fp8_support(qkv, quantizers, config, mode)
    if mode == "mxfp8" and _is_padding(config):
        raise NotImplementedError("JAX MXFP8 attention does not support padding masks.")

    quantized, data = _quantized_operands(
        qkv, config.qkv_layout, quantizers.qkv, both=mode == "mxfp8"
    )
    q_tensor, k_tensor, v_tensor = quantized
    q_data, k_data, v_data = data
    # MXFP8 forward consumes V in the columnwise orientation.
    if mode == "mxfp8":
        v_data = _colwise(v_tensor).data
    q_seqlen, kv_seqlen = _sequence_lengths(sequence_descriptor, config)
    seed = _FusedAttnRNGStateChecker().check_seed(
        seed, config.dropout_probability, config.is_training
    )
    output_dtype = quantizers.o.q_dtype if mode == "delayed" else qkv[0].dtype
    s_scale = _tensor_scale(quantizers.s)
    s_scale_inv = jnp.reciprocal(s_scale)
    output_scale_inv = _tensor_scale_inv(quantizers.o)
    raw_output, stats, amax, rng_state, _ = execute_fp8_fwd(
        q_data,
        k_data,
        v_data,
        _graph_scale_inv(q_tensor, mode),
        _graph_scale_inv(k_tensor, mode),
        _graph_scale_inv(v_tensor, mode, colwise=mode == "mxfp8"),
        s_scale_inv,
        s_scale,
        _tensor_scale(quantizers.o),
        q_seqlen,
        kv_seqlen,
        seed,
        config=config,
        mode=mode,
        output_dtype=output_dtype,
    )
    if mode == "delayed":
        quantizers.s.update(amax[0:1])
        quantizers.o.update(amax[1:2])
        output = (raw_output.astype(qkv[0].dtype) * output_scale_inv).astype(
            qkv[0].dtype
        )
    else:
        output = raw_output
    return output, (
        quantized,
        raw_output,
        stats,
        rng_state,
        q_seqlen,
        kv_seqlen,
        quantizers,
        s_scale,
        s_scale_inv,
        output_scale_inv,
    )


def _split_gradient_outputs(dq, dk, dv, layout):
    if layout.is_qkvpacked():
        return (jnp.stack((dq, dk, dv), axis=-3),)
    if layout.is_kvpacked():
        return dq, jnp.stack((dk, dv), axis=-3)
    return dq, dk, dv


def fused_attn_fp8_bwd(ctx, doutput, config):
    """Execute FP8 attention backward and update delayed-scaling quantizers."""

    (
        quantized,
        raw_output,
        stats,
        rng_state,
        q_seqlen,
        kv_seqlen,
        quantizers,
        s_scale,
        s_scale_inv,
        output_scale_inv,
    ) = ctx
    q_tensor, k_tensor, v_tensor = quantized
    mode = _scaling_mode(quantizers.qkv)
    input_dtype = _rowwise(q_tensor).dq_dtype
    (do_tensor,) = _quantize_many((doutput,), quantizers.do, both=mode == "mxfp8")
    q_data, k_data, v_data = (
        _rowwise(q_tensor).data,
        _rowwise(k_tensor).data,
        _rowwise(v_tensor).data,
    )
    empty_data = jnp.zeros((0,), dtype=q_data.dtype)
    empty_scale = jnp.ones((1,), dtype=jnp.float32)
    if config.qkv_layout.is_qkvpacked():
        k_data = v_data = empty_data
    elif config.qkv_layout.is_kvpacked():
        v_data = empty_data
    if mode == "mxfp8":
        q_t, k_t = _colwise(q_tensor).data, _colwise(k_tensor).data
        do_t = _colwise(do_tensor).data
        q_scale_t, k_scale_t = (
            _graph_scale_inv(q_tensor, mode, colwise=True),
            _graph_scale_inv(k_tensor, mode, colwise=True),
        )
        do_scale_t = _graph_scale_inv(do_tensor, mode, colwise=True)
    else:
        q_t = k_t = do_t = empty_data
        q_scale_t = k_scale_t = do_scale_t = empty_scale

    grad_dtype = quantizers.dqkv.q_dtype if mode == "delayed" else input_dtype
    grad_scale_inv = _tensor_scale_inv(quantizers.dqkv)
    dq, dk, dv, amax, _, _ = execute_fp8_bwd(
        q_data,
        k_data,
        v_data,
        q_t,
        k_t,
        stats,
        rng_state,
        raw_output,
        _rowwise(do_tensor).data,
        q_seqlen,
        kv_seqlen,
        _graph_scale_inv(q_tensor, mode),
        _graph_scale_inv(k_tensor, mode),
        _graph_scale_inv(v_tensor, mode),
        output_scale_inv,
        _graph_scale_inv(do_tensor, mode),
        s_scale_inv,
        _tensor_scale_inv(quantizers.dp),
        s_scale,
        _tensor_scale(quantizers.dqkv),
        _tensor_scale(quantizers.dqkv),
        _tensor_scale(quantizers.dqkv),
        _tensor_scale(quantizers.dp),
        do_t,
        do_scale_t,
        q_scale_t,
        k_scale_t,
        doutput,
        config=config,
        mode=mode,
        grad_dtype=grad_dtype,
    )
    if mode == "delayed":
        quantizers.dqkv.update(jnp.max(amax[:3]))
        quantizers.dp.update(amax[3:4])
        dq, dk, dv = (
            (tensor.astype(input_dtype) * grad_scale_inv).astype(input_dtype)
            for tensor in (dq, dk, dv)
        )
    return _split_gradient_outputs(dq, dk, dv, config.qkv_layout), quantizers
