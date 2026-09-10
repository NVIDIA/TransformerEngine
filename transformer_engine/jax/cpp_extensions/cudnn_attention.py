# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Python cuDNN frontend graphs for the standard JAX fused-attention path."""

from __future__ import annotations

import operator
import os
from dataclasses import dataclass
from functools import reduce
from typing import Any

import jax.numpy as jnp
import numpy as np

from .cudnn_graph import (
    GraphBinding,
    SerializedGraph,
    cudnn_data_type,
    dtype_name,
    finalize_graph,
    import_cudnn,
    serialized_graph,
)
from .misc import get_all_device_compute_capability, get_cudnn_version

# Stable graph tensor UIDs.  They are deliberately shared by forward and backward
# so serialized graphs and variant packs remain easy to inspect.
_UID_Q = 1
_UID_K = 2
_UID_V = 3
_UID_O = 4
_UID_STATS = 5
_UID_MAX = 6
_UID_BIAS = 7
_UID_SINK = 8
_UID_SEQ_Q = 9
_UID_SEQ_KV = 10
_UID_OFFSET_Q = 11
_UID_OFFSET_K = 12
_UID_OFFSET_V = 13
_UID_OFFSET_O = 14
_UID_OFFSET_STATS = 15
_UID_DROPOUT_SEED = 16
_UID_DROPOUT_OFFSET = 17
_UID_DO = 18
_UID_DQ = 19
_UID_DK = 20
_UID_DV = 21
_UID_DBIAS = 22
_UID_DSINK = 23
_UID_ATTN_SCALE = 24


@dataclass(frozen=True)
class AttentionGraphInfo:
    """Serialized graph plus abstract-output information used by JAX lowering."""

    graph: SerializedGraph
    stats_shape: tuple[int, ...]
    max_shape: tuple[int, ...]


@dataclass(frozen=True)
class _LayoutInfo:
    batch_shape: tuple[int, ...]
    input_batch: int
    q_max_seqlen: int
    kv_max_seqlen: int
    q_heads: int
    kv_heads: int
    qk_dim: int
    v_dim: int


_graph_cache: dict[tuple[Any, ...], AttentionGraphInfo] = {}


def _layout_info(q_aval, k_aval, v_aval, layout) -> _LayoutInfo:
    """Parse TE's three supported JAX QKV layout groups."""
    if layout.is_qkvpacked():
        *batch_shape, q_seqlen, packed, q_heads, qk_dim = q_aval.shape
        if packed != 3:
            raise ValueError(
                f"QKV-packed fused attention expects dimension 3, got {q_aval.shape}."
            )
        kv_seqlen = q_seqlen
        kv_heads = q_heads
        v_dim = qk_dim
    elif layout.is_kvpacked():
        *batch_shape, q_seqlen, q_heads, qk_dim = q_aval.shape
        *kv_batch_shape, kv_seqlen, packed, kv_heads, v_dim = k_aval.shape
        if tuple(batch_shape) != tuple(kv_batch_shape) or packed != 2:
            raise ValueError(
                f"Invalid KV-packed fused-attention shapes: {q_aval}, {k_aval}."
            )
        if qk_dim != v_dim:
            raise ValueError(
                "KV-packed fused attention requires equal QK and V head dimensions."
            )
    elif layout.is_separate():
        *batch_shape, q_seqlen, q_heads, qk_dim = q_aval.shape
        *k_batch_shape, kv_seqlen, kv_heads, k_dim = k_aval.shape
        *v_batch_shape, v_seqlen, v_heads, v_dim = v_aval.shape
        if tuple(batch_shape) != tuple(k_batch_shape) or tuple(batch_shape) != tuple(
            v_batch_shape
        ):
            raise ValueError(
                "Separate Q, K and V tensors must have matching batch shapes."
            )
        if qk_dim != k_dim or kv_seqlen != v_seqlen or kv_heads != v_heads:
            raise ValueError(
                "Separate fused-attention K and V shapes are inconsistent."
            )
    else:
        raise ValueError(f"Unsupported JAX fused-attention layout: {layout}.")
    return _LayoutInfo(
        batch_shape=tuple(int(dim) for dim in batch_shape),
        input_batch=reduce(operator.mul, batch_shape, 1),
        q_max_seqlen=int(q_seqlen),
        kv_max_seqlen=int(kv_seqlen),
        q_heads=int(q_heads),
        kv_heads=int(kv_heads),
        qk_dim=int(qk_dim),
        v_dim=int(v_dim),
    )


def _matrix_stride(
    info: _LayoutInfo, layout, matrix: str, graph_sq: int, graph_skv: int
):
    """Port generateMatrixStrides for JAX's BSHD/THD layout subset."""
    if matrix in ("q", "o"):
        heads = info.q_heads
        dim = info.qk_dim if matrix == "q" else info.v_dim
        seqlen = graph_sq
    else:
        heads = info.kv_heads
        dim = info.qk_dim if matrix == "k" else info.v_dim
        seqlen = graph_skv

    if matrix in ("q", "k", "v") and layout.is_qkvpacked():
        return (
            graph_sq * 3 * info.q_heads * info.qk_dim,
            dim,
            3 * info.q_heads * info.qk_dim,
            1,
        )
    if matrix in ("k", "v") and layout.is_kvpacked():
        return (
            graph_skv * 2 * info.kv_heads * info.qk_dim,
            dim,
            2 * info.kv_heads * info.qk_dim,
            1,
        )
    return (seqlen * heads * dim, dim, heads * dim, 1)


def _qkv_bindings(info: _LayoutInfo, layout, itemsize: int, *, outputs: bool = False):
    """Return UID bindings for separate or physically packed QKV buffers."""
    if outputs:
        uids = (_UID_DQ, _UID_DK, _UID_DV)
    else:
        uids = (_UID_Q, _UID_K, _UID_V)
    if layout.is_qkvpacked():
        stride = info.q_heads * info.qk_dim * itemsize
        return (
            GraphBinding(uids[0], 0, 0),
            GraphBinding(uids[1], 0, stride),
            GraphBinding(uids[2], 0, 2 * stride),
        )
    if layout.is_kvpacked():
        stride = info.kv_heads * info.qk_dim * itemsize
        return (
            GraphBinding(uids[0], 0, 0),
            GraphBinding(uids[1], 1, 0),
            GraphBinding(uids[2], 1, stride),
        )
    return tuple(GraphBinding(uid, index, 0) for index, uid in enumerate(uids))


def _is_bias(config) -> bool:
    return getattr(config.attn_bias_type, "name", "") == "POST_SCALE_BIAS"


def _is_padding(config) -> bool:
    return bool(config.attn_mask_type.is_padding())


def _is_causal(config) -> bool:
    name = getattr(config.attn_mask_type, "name", "")
    return name in ("CAUSAL_MASK", "PADDING_CAUSAL_MASK")


def _is_bottom_right(config) -> bool:
    return bool(config.attn_mask_type.is_bottom_right())


def _has_sink(config) -> bool:
    return getattr(config.softmax_type, "name", "") != "VANILLA_SOFTMAX"


def _is_dropout(config) -> bool:
    return bool(config.is_training and config.dropout_probability != 0.0)


def _device_arch() -> int:
    capabilities = get_all_device_compute_capability()
    return int(capabilities[0]) if capabilities else 0


def ragged_graph_batch_size(input_batch: int, max_segments_per_seq: int) -> int:
    """Preserve the legacy cuDNN graph batch-size bucket for ragged attention."""
    batch = int(input_batch) * int(max_segments_per_seq)
    # Bucketing is part of cuDNN's ragged-stats layout, introduced in 9.6.
    # Older versions use dense stats and require the physical metadata extent.
    if get_cudnn_version() < (9, 6, 0) or _device_arch() == 120:
        return batch
    if batch <= 32:
        return 32
    if batch <= 512:
        return 1 << (batch - 1).bit_length()
    return ((batch + 511) // 512) * 512


def _ragged_graph_token_count(tokens: int) -> int:
    """Preserve the legacy cuDNN graph token-count bucket for ragged attention."""
    tokens = int(tokens)
    if tokens <= 1024:
        return 1024
    if tokens <= 32768:
        return 1 << (tokens - 1).bit_length()
    return ((tokens + 32767) // 32768) * 32768


def _graph_dimensions(info: _LayoutInfo, config):
    """Return logical cuDNN B/H/S dimensions and physical auxiliary shapes."""
    is_ragged = config.qkv_layout.is_thd()
    cudnn_version = get_cudnn_version()
    arch = _device_arch()
    use_ragged_stats = is_ragged and cudnn_version >= (9, 6, 0) and arch != 120
    if is_ragged:
        graph_batch = ragged_graph_batch_size(
            info.input_batch, config.max_segments_per_seq
        )
        if cudnn_version < (9, 6, 0) or arch == 120:
            graph_sq = info.q_max_seqlen
            graph_skv = info.kv_max_seqlen
        else:
            graph_sq = _ragged_graph_token_count(info.input_batch * info.q_max_seqlen)
            graph_skv = _ragged_graph_token_count(
                info.input_batch * info.kv_max_seqlen
            )
    else:
        graph_batch = info.input_batch
        graph_sq = info.q_max_seqlen
        graph_skv = info.kv_max_seqlen

    if is_ragged and cudnn_version >= (9, 6, 0):
        stats_shape = (*info.batch_shape, info.q_max_seqlen, info.q_heads, 1)
    elif cudnn_version >= (9, 6, 0):
        stats_shape = (*info.batch_shape, info.q_heads, info.q_max_seqlen, 1)
    else:
        stats_shape = (
            *info.batch_shape,
            info.q_heads,
            info.q_max_seqlen,
            int(config.max_segments_per_seq),
        )
    if config.return_max_logit:
        max_shape = (
            (*info.batch_shape, info.q_max_seqlen, info.q_heads, 1)
            if use_ragged_stats
            else (*info.batch_shape, info.q_heads, info.q_max_seqlen, 1)
        )
    else:
        max_shape = (0,)
    return graph_batch, graph_sq, graph_skv, use_ragged_stats, stats_shape, max_shape


def _tensor(graph, cudnn, *, name, dim, stride, dtype, uid):
    return graph.tensor(
        name=name,
        dim=tuple(int(x) for x in dim),
        stride=tuple(int(x) for x in stride),
        data_type=dtype,
        uid=uid,
    )


def _ragged_offset(graph, cudnn, name: str, uid: int, graph_batch: int, dtype):
    return _tensor(
        graph,
        cudnn,
        name=name,
        dim=(graph_batch + 1, 1, 1, 1),
        stride=(1, 1, 1, 1),
        dtype=dtype,
        uid=uid,
    )


def _ragged_offset_spec(cudnn):
    """Return the cuDNN datatype and byte size for external element offsets."""
    use_int64 = get_cudnn_version() >= (9, 5, 0)
    dtype = cudnn.data_type.INT64 if use_int64 else cudnn.data_type.INT32
    itemsize = np.dtype(np.int64 if use_int64 else np.int32).itemsize
    return dtype, itemsize


def _mask_options(cudnn, info: _LayoutInfo, config):
    is_padding = _is_padding(config)
    causal = _is_causal(config)
    bottom_right = _is_bottom_right(config)
    bottom_right_diagonal = bool(config.bottom_right_diagonal)
    if bottom_right and info.q_max_seqlen == info.kv_max_seqlen and not is_padding:
        causal = True
        bottom_right = False
        bottom_right_diagonal = False
    window_left, window_right = (
        config.cp_striped_window_size
        if config.cp_striped_window_size is not None
        else config.window_size
    )
    cudnn_version = get_cudnn_version()
    options = {
        "diagonal_alignment": (
            cudnn.diagonal_alignment.BOTTOM_RIGHT
            if bottom_right_diagonal or bottom_right
            else cudnn.diagonal_alignment.TOP_LEFT
        ),
    }
    # Before cuDNN 9.6 the preferred right-band API was unavailable, so preserve
    # the legacy causal flags used by the C++ frontend graph.
    if cudnn_version < (9, 6, 0):
        options["use_causal_mask"] = causal
        options["use_causal_mask_bottom_right"] = bottom_right
    if cudnn_version >= (9, 2, 0) and window_left != -1:
        options["diagonal_band_left_bound"] = int(window_left) + 1
    if cudnn_version >= (9, 6, 0):
        if window_right != -1:
            options["diagonal_band_right_bound"] = int(window_right)
        elif causal or bottom_right:
            options["diagonal_band_right_bound"] = 0
    return options


def _scalar_tensor(graph, cudnn, name: str, uid: int, dtype):
    return graph.tensor(
        name=name,
        dim=(1, 1, 1, 1),
        stride=(1, 1, 1, 1),
        data_type=dtype,
        is_pass_by_value=True,
        uid=uid,
    )


def _cache_key(direction: str, q_aval, k_aval, v_aval, bias_aval, config, *extra_avals):
    avals = (q_aval, k_aval, v_aval, bias_aval, *extra_avals)
    return (
        direction,
        config,
        tuple((tuple(aval.shape), dtype_name(aval.dtype)) for aval in avals),
        get_cudnn_version(),
        _device_arch(),
    )


def build_fwd_graph(q_aval, k_aval, v_aval, bias_aval, config) -> AttentionGraphInfo:
    """Build or retrieve the standard fused-attention forward graph."""
    key = _cache_key("fwd", q_aval, k_aval, v_aval, bias_aval, config)
    if key not in _graph_cache:
        _graph_cache[key] = _build_fwd_graph(q_aval, k_aval, v_aval, bias_aval, config)
    return _graph_cache[key]


def _build_fwd_graph(q_aval, k_aval, v_aval, bias_aval, config) -> AttentionGraphInfo:
    cudnn = import_cudnn()
    info = _layout_info(q_aval, k_aval, v_aval, config.qkv_layout)
    graph_batch, graph_sq, graph_skv, ragged_stats, stats_shape, max_shape = (
        _graph_dimensions(info, config)
    )
    io_dtype = cudnn_data_type(cudnn, q_aval.dtype)
    graph = cudnn.pygraph(
        io_data_type=io_dtype,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )

    q = _tensor(
        graph,
        cudnn,
        name="q",
        dim=(graph_batch, info.q_heads, graph_sq, info.qk_dim),
        stride=_matrix_stride(info, config.qkv_layout, "q", graph_sq, graph_skv),
        dtype=io_dtype,
        uid=_UID_Q,
    )
    k = _tensor(
        graph,
        cudnn,
        name="k",
        dim=(graph_batch, info.kv_heads, graph_skv, info.qk_dim),
        stride=_matrix_stride(info, config.qkv_layout, "k", graph_sq, graph_skv),
        dtype=io_dtype,
        uid=_UID_K,
    )
    v = _tensor(
        graph,
        cudnn,
        name="v",
        dim=(graph_batch, info.kv_heads, graph_skv, info.v_dim),
        stride=_matrix_stride(info, config.qkv_layout, "v", graph_sq, graph_skv),
        dtype=io_dtype,
        uid=_UID_V,
    )

    input_bindings = list(
        _qkv_bindings(info, config.qkv_layout, jnp.dtype(q_aval.dtype).itemsize)
    )
    output_bindings = [GraphBinding(_UID_O, 0), GraphBinding(_UID_STATS, 1)]
    scale = _scalar_tensor(
        graph, cudnn, "attn_scale", _UID_ATTN_SCALE, cudnn.data_type.FLOAT
    )
    scalar_uids = [_UID_ATTN_SCALE]
    scalar_values = [np.asarray(config.scaling_factor, dtype=np.float32).tobytes()]

    kwargs = {
        "name": "te_fused_attention",
        "q": q,
        "k": k,
        "v": v,
        "generate_stats": True,
        "attn_scale": scale,
        **_mask_options(cudnn, info, config),
    }

    if _is_bias(config):
        *bias_batch_shape, bias_heads, bias_sq, bias_skv = bias_aval.shape
        bias_batch = reduce(operator.mul, bias_batch_shape, 1)
        bias = _tensor(
            graph,
            cudnn,
            name="bias",
            dim=(bias_batch, bias_heads, bias_sq, bias_skv),
            stride=(bias_heads * bias_sq * bias_skv, bias_sq * bias_skv, bias_skv, 1),
            dtype=io_dtype,
            uid=_UID_BIAS,
        )
        kwargs["bias"] = bias
        input_bindings.append(GraphBinding(_UID_BIAS, 3))

    if _has_sink(config):
        sink = _tensor(
            graph,
            cudnn,
            name="softmax_offset",
            dim=(1, info.q_heads, 1, 1),
            stride=(info.q_heads, 1, 1, 1),
            dtype=cudnn.data_type.FLOAT,
            uid=_UID_SINK,
        )
        kwargs["sink_token"] = sink
        input_bindings.append(GraphBinding(_UID_SINK, 4))

    if _is_padding(config):
        seq_q = _tensor(
            graph,
            cudnn,
            name="seq_len_q",
            dim=(graph_batch, 1, 1, 1),
            stride=(1, 1, 1, 1),
            dtype=cudnn.data_type.INT32,
            uid=_UID_SEQ_Q,
        )
        seq_kv = _tensor(
            graph,
            cudnn,
            name="seq_len_kv",
            dim=(graph_batch, 1, 1, 1),
            stride=(1, 1, 1, 1),
            dtype=cudnn.data_type.INT32,
            uid=_UID_SEQ_KV,
        )
        kwargs.update(use_padding_mask=True, seq_len_q=seq_q, seq_len_kv=seq_kv)
        input_bindings.extend(
            (GraphBinding(_UID_SEQ_Q, 6), GraphBinding(_UID_SEQ_KV, 7))
        )

    offset_q = offset_k = offset_v = offset_o = offset_stats = None
    if config.qkv_layout.is_thd():
        offset_dtype, offset_itemsize = _ragged_offset_spec(cudnn)
        offset_bytes = (graph_batch + 1) * offset_itemsize
        offset_q = _ragged_offset(
            graph, cudnn, "offset_q", _UID_OFFSET_Q, graph_batch, offset_dtype
        )
        offset_k = _ragged_offset(
            graph, cudnn, "offset_k", _UID_OFFSET_K, graph_batch, offset_dtype
        )
        offset_v = _ragged_offset(
            graph, cudnn, "offset_v", _UID_OFFSET_V, graph_batch, offset_dtype
        )
        offset_o = _ragged_offset(
            graph, cudnn, "offset_o", _UID_OFFSET_O, graph_batch, offset_dtype
        )
        q.set_ragged_offset(offset_q)
        k.set_ragged_offset(offset_k)
        v.set_ragged_offset(offset_v)
        input_bindings.extend(
            (
                GraphBinding(_UID_OFFSET_Q, 10, 0),
                GraphBinding(_UID_OFFSET_K, 10, offset_bytes),
                GraphBinding(_UID_OFFSET_V, 10, 2 * offset_bytes),
                GraphBinding(_UID_OFFSET_O, 10, 3 * offset_bytes),
            )
        )

    if _is_dropout(config):
        seed = _tensor(
            graph,
            cudnn,
            name="dropout_seed",
            dim=(1, 1, 1, 1),
            stride=(1, 1, 1, 1),
            dtype=cudnn.data_type.INT64,
            uid=_UID_DROPOUT_SEED,
        )
        offset = _tensor(
            graph,
            cudnn,
            name="dropout_offset",
            dim=(1, 1, 1, 1),
            stride=(1, 1, 1, 1),
            dtype=cudnn.data_type.INT64,
            uid=_UID_DROPOUT_OFFSET,
        )
        kwargs["dropout"] = (float(config.dropout_probability), seed, offset)
        # rng_state is forward result #3: two int64 values stored as four uint32 values.
        output_bindings.extend(
            (
                GraphBinding(_UID_DROPOUT_SEED, 3, 0),
                GraphBinding(_UID_DROPOUT_OFFSET, 3, 8),
            )
        )

    if config.return_max_logit:
        max_tensor = _tensor(
            graph,
            cudnn,
            name="max_logit",
            dim=(graph_batch, info.q_heads, graph_sq, 1),
            stride=(
                (info.q_heads * graph_sq, 1, info.q_heads, 1)
                if ragged_stats
                else (info.q_heads * graph_sq, graph_sq, 1, 1)
            ),
            dtype=cudnn.data_type.FLOAT,
            uid=_UID_MAX,
        )
        if ragged_stats:
            offset_stats = _ragged_offset(
                graph,
                cudnn,
                "offset_stats",
                _UID_OFFSET_STATS,
                graph_batch,
                dtype=offset_dtype,
            )
            max_tensor.set_ragged_offset(offset_stats)
            input_bindings.append(GraphBinding(_UID_OFFSET_STATS, 10, 4 * offset_bytes))
        max_tensor.set_output(True)
        kwargs["score_max"] = max_tensor
        output_bindings.append(GraphBinding(_UID_MAX, 2))

    output, stats = graph.sdpa(**kwargs)
    output.set_output(True).set_uid(_UID_O).set_dim(
        (graph_batch, info.q_heads, graph_sq, info.v_dim)
    ).set_stride(_matrix_stride(info, config.qkv_layout, "o", graph_sq, graph_skv))
    output.set_data_type(io_dtype)
    if config.qkv_layout.is_thd():
        output.set_ragged_offset(offset_o)

    stats.set_output(True).set_uid(_UID_STATS).set_data_type(cudnn.data_type.FLOAT)
    stats.set_dim((graph_batch, info.q_heads, graph_sq, 1))
    if ragged_stats:
        if offset_stats is None:
            offset_stats = _ragged_offset(
                graph,
                cudnn,
                "offset_stats",
                _UID_OFFSET_STATS,
                graph_batch,
                dtype=offset_dtype,
            )
            input_bindings.append(GraphBinding(_UID_OFFSET_STATS, 10, 4 * offset_bytes))
        stats.set_stride((info.q_heads * graph_sq, 1, info.q_heads, 1))
        stats.set_ragged_offset(offset_stats)
    else:
        stats.set_stride((info.q_heads * graph_sq, graph_sq, 1, 1))

    workspace, data, version = finalize_graph(
        cudnn, graph, description="fused-attention forward"
    )
    result = serialized_graph(
        serialized_graph_data=data,
        cudnn_frontend_version=version,
        workspace_size=workspace,
        input_bindings=input_bindings,
        output_bindings=output_bindings,
        scalar_uids=scalar_uids,
        scalar_values=scalar_values,
    )
    return AttentionGraphInfo(result, stats_shape, max_shape)


def build_bwd_graph(
    q_aval,
    k_aval,
    v_aval,
    bias_aval,
    stats_aval,
    output_aval,
    doutput_aval,
    config,
) -> AttentionGraphInfo:
    """Build or retrieve the standard fused-attention backward graph."""
    key = _cache_key(
        "bwd",
        q_aval,
        k_aval,
        v_aval,
        bias_aval,
        config,
        stats_aval,
        output_aval,
        doutput_aval,
    )
    if key not in _graph_cache:
        _graph_cache[key] = _build_bwd_graph(
            q_aval,
            k_aval,
            v_aval,
            bias_aval,
            stats_aval,
            output_aval,
            doutput_aval,
            config,
        )
    return _graph_cache[key]


def _build_bwd_graph(
    q_aval,
    k_aval,
    v_aval,
    bias_aval,
    stats_aval,
    output_aval,
    doutput_aval,
    config,
) -> AttentionGraphInfo:
    cudnn = import_cudnn()
    info = _layout_info(q_aval, k_aval, v_aval, config.qkv_layout)
    graph_batch, graph_sq, graph_skv, ragged_stats, stats_shape, max_shape = (
        _graph_dimensions(info, config)
    )
    io_dtype = cudnn_data_type(cudnn, q_aval.dtype)
    graph = cudnn.pygraph(
        io_data_type=io_dtype,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )

    def io_tensor(name, dim, stride, uid, dtype=io_dtype):
        return _tensor(
            graph, cudnn, name=name, dim=dim, stride=stride, dtype=dtype, uid=uid
        )

    q = io_tensor(
        "q",
        (graph_batch, info.q_heads, graph_sq, info.qk_dim),
        _matrix_stride(info, config.qkv_layout, "q", graph_sq, graph_skv),
        _UID_Q,
    )
    k = io_tensor(
        "k",
        (graph_batch, info.kv_heads, graph_skv, info.qk_dim),
        _matrix_stride(info, config.qkv_layout, "k", graph_sq, graph_skv),
        _UID_K,
    )
    v = io_tensor(
        "v",
        (graph_batch, info.kv_heads, graph_skv, info.v_dim),
        _matrix_stride(info, config.qkv_layout, "v", graph_sq, graph_skv),
        _UID_V,
    )
    output = io_tensor(
        "o",
        (graph_batch, info.q_heads, graph_sq, info.v_dim),
        _matrix_stride(info, config.qkv_layout, "o", graph_sq, graph_skv),
        _UID_O,
    )
    doutput = io_tensor(
        "dO",
        (graph_batch, info.q_heads, graph_sq, info.v_dim),
        _matrix_stride(info, config.qkv_layout, "o", graph_sq, graph_skv),
        _UID_DO,
    )
    stats = io_tensor(
        "stats",
        (graph_batch, info.q_heads, graph_sq, 1),
        (
            (info.q_heads * graph_sq, 1, info.q_heads, 1)
            if ragged_stats
            else (info.q_heads * graph_sq, graph_sq, 1, 1)
        ),
        _UID_STATS,
        cudnn.data_type.FLOAT,
    )

    itemsize = jnp.dtype(q_aval.dtype).itemsize
    input_bindings = list(_qkv_bindings(info, config.qkv_layout, itemsize))
    input_bindings.extend(
        (
            GraphBinding(_UID_STATS, 5),
            GraphBinding(_UID_O, 7),
            GraphBinding(_UID_DO, 8),
        )
    )
    output_bindings = list(
        _qkv_bindings(info, config.qkv_layout, itemsize, outputs=True)
    )
    scale = _scalar_tensor(
        graph, cudnn, "attn_scale", _UID_ATTN_SCALE, cudnn.data_type.FLOAT
    )
    scalar_uids = [_UID_ATTN_SCALE]
    scalar_values = [np.asarray(config.scaling_factor, dtype=np.float32).tobytes()]

    kwargs = {
        "name": "te_fused_attention_backward",
        "q": q,
        "k": k,
        "v": v,
        "o": output,
        "dO": doutput,
        "stats": stats,
        "attn_scale": scale,
        **_mask_options(cudnn, info, config),
    }
    if get_cudnn_version() >= (9, 0, 0):
        kwargs["use_deterministic_algorithm"] = not bool(
            int(os.getenv("NVTE_ALLOW_NONDETERMINISTIC_ALGO", "1"))
        )
    if ragged_stats:
        kwargs["max_total_seq_len_q"] = graph_sq
    if (
        config.qkv_layout.is_thd()
        and get_cudnn_version() >= (9, 6, 0)
        and _device_arch() != 120
    ):
        kwargs["max_total_seq_len_kv"] = graph_skv

    if _is_bias(config):
        *bias_batch_shape, bias_heads, bias_sq, bias_skv = bias_aval.shape
        bias_batch = reduce(operator.mul, bias_batch_shape, 1)
        bias = io_tensor(
            "bias",
            (bias_batch, bias_heads, bias_sq, bias_skv),
            (bias_heads * bias_sq * bias_skv, bias_sq * bias_skv, bias_skv, 1),
            _UID_BIAS,
        )
        kwargs["bias"] = bias
        input_bindings.append(GraphBinding(_UID_BIAS, 3))
        if not (bias_batch == 1 and bias_heads == 1 and bias_sq == 1):
            dbias = io_tensor(
                "dBias",
                (bias_batch, bias_heads, bias_sq, bias_skv),
                (bias_heads * bias_sq * bias_skv, bias_sq * bias_skv, bias_skv, 1),
                _UID_DBIAS,
            )
            dbias.set_output(True)
            kwargs["dBias"] = dbias
            output_bindings.append(GraphBinding(_UID_DBIAS, 3))

    if _has_sink(config):
        sink = io_tensor(
            "softmax_offset",
            (1, info.q_heads, 1, 1),
            (info.q_heads, 1, 1, 1),
            _UID_SINK,
            cudnn.data_type.FLOAT,
        )
        dsink = io_tensor(
            "dsoftmax_offset",
            (1, info.q_heads, 1, 1),
            (info.q_heads, 1, 1, 1),
            _UID_DSINK,
            cudnn.data_type.FLOAT,
        )
        dsink.set_output(True)
        kwargs.update(sink_token=sink, dSink_token=dsink)
        input_bindings.append(GraphBinding(_UID_SINK, 4))
        output_bindings.append(GraphBinding(_UID_DSINK, 4))

    if _is_padding(config):
        seq_q = io_tensor(
            "seq_len_q",
            (graph_batch, 1, 1, 1),
            (1, 1, 1, 1),
            _UID_SEQ_Q,
            cudnn.data_type.INT32,
        )
        seq_kv = io_tensor(
            "seq_len_kv",
            (graph_batch, 1, 1, 1),
            (1, 1, 1, 1),
            _UID_SEQ_KV,
            cudnn.data_type.INT32,
        )
        kwargs.update(use_padding_mask=True, seq_len_q=seq_q, seq_len_kv=seq_kv)
        input_bindings.extend(
            (GraphBinding(_UID_SEQ_Q, 9), GraphBinding(_UID_SEQ_KV, 10))
        )

    if config.qkv_layout.is_thd():
        offset_dtype, offset_itemsize = _ragged_offset_spec(cudnn)
        offset_bytes = (graph_batch + 1) * offset_itemsize
        offset_q = _ragged_offset(
            graph, cudnn, "offset_q", _UID_OFFSET_Q, graph_batch, offset_dtype
        )
        offset_k = _ragged_offset(
            graph, cudnn, "offset_k", _UID_OFFSET_K, graph_batch, offset_dtype
        )
        offset_v = _ragged_offset(
            graph, cudnn, "offset_v", _UID_OFFSET_V, graph_batch, offset_dtype
        )
        offset_o = _ragged_offset(
            graph, cudnn, "offset_o", _UID_OFFSET_O, graph_batch, offset_dtype
        )
        q.set_ragged_offset(offset_q)
        k.set_ragged_offset(offset_k)
        v.set_ragged_offset(offset_v)
        output.set_ragged_offset(offset_o)
        doutput.set_ragged_offset(offset_o)
        input_bindings.extend(
            (
                GraphBinding(_UID_OFFSET_Q, 13, 0),
                GraphBinding(_UID_OFFSET_K, 13, offset_bytes),
                GraphBinding(_UID_OFFSET_V, 13, 2 * offset_bytes),
                GraphBinding(_UID_OFFSET_O, 13, 3 * offset_bytes),
            )
        )
        if ragged_stats:
            offset_stats = _ragged_offset(
                graph=graph,
                cudnn=cudnn,
                name="offset_stats",
                uid=_UID_OFFSET_STATS,
                graph_batch=graph_batch,
                dtype=offset_dtype,
            )
            stats.set_ragged_offset(offset_stats)
            input_bindings.append(GraphBinding(_UID_OFFSET_STATS, 13, 4 * offset_bytes))

    if _is_dropout(config):
        seed = io_tensor(
            "dropout_seed",
            (1, 1, 1, 1),
            (1, 1, 1, 1),
            _UID_DROPOUT_SEED,
            cudnn.data_type.INT64,
        )
        offset = io_tensor(
            "dropout_offset",
            (1, 1, 1, 1),
            (1, 1, 1, 1),
            _UID_DROPOUT_OFFSET,
            cudnn.data_type.INT64,
        )
        kwargs["dropout"] = (float(config.dropout_probability), seed, offset)
        input_bindings.extend(
            (
                GraphBinding(_UID_DROPOUT_SEED, 6, 0),
                GraphBinding(_UID_DROPOUT_OFFSET, 6, 8),
            )
        )

    dq, dk, dv = graph.sdpa_backward(**kwargs)
    q_stride = _matrix_stride(info, config.qkv_layout, "q", graph_sq, graph_skv)
    k_stride = _matrix_stride(info, config.qkv_layout, "k", graph_sq, graph_skv)
    v_stride = _matrix_stride(info, config.qkv_layout, "v", graph_sq, graph_skv)
    dq.set_output(True).set_uid(_UID_DQ).set_dim(
        (graph_batch, info.q_heads, graph_sq, info.qk_dim)
    ).set_stride(q_stride)
    dk.set_output(True).set_uid(_UID_DK).set_dim(
        (graph_batch, info.kv_heads, graph_skv, info.qk_dim)
    ).set_stride(k_stride)
    dv.set_output(True).set_uid(_UID_DV).set_dim(
        (graph_batch, info.kv_heads, graph_skv, info.v_dim)
    ).set_stride(v_stride)
    if config.qkv_layout.is_thd():
        dq.set_ragged_offset(offset_q)
        dk.set_ragged_offset(offset_k)
        dv.set_ragged_offset(offset_v)

    workspace, data, version = finalize_graph(
        cudnn, graph, description="fused-attention backward"
    )
    result = serialized_graph(
        serialized_graph_data=data,
        cudnn_frontend_version=version,
        workspace_size=workspace,
        input_bindings=input_bindings,
        output_bindings=output_bindings,
        scalar_uids=scalar_uids,
        scalar_values=scalar_values,
    )
    return AttentionGraphInfo(result, stats_shape, max_shape)


def clear_graph_cache():
    """Clear the process-local serialized graph cache (primarily for tests)."""
    _graph_cache.clear()


def _encoded_cudnn_version() -> int:
    major, minor, patch = get_cudnn_version()
    magnitude = 1000 if major < 9 else 10000
    return major * magnitude + minor * 100 + patch


def is_fused_attn_supported(helper) -> bool:
    """JAX-local port of the F16/BF16 cuDNN attention compatibility policy.

    cuDNN frontend ``check_support`` remains authoritative when a concrete graph is
    built.  This early policy preserves the public fallback behavior for callers that
    ask about availability before Q/K/V abstract values exist.
    """
    if jnp.dtype(helper.q_dtype) not in (
        jnp.dtype(jnp.float16),
        jnp.dtype(jnp.bfloat16),
    ):
        return False
    if jnp.dtype(helper.q_dtype) != jnp.dtype(helper.kv_dtype):
        return False

    version = _encoded_cudnn_version()
    arch = _device_arch()
    is_thd = helper.qkv_layout.is_thd()
    is_training = bool(helper.is_training)
    sq, skv = int(helper.q_max_seqlen), int(helper.kv_max_seqlen)
    h, hg = int(helper.q_num_heads), int(helper.kv_num_heads)
    dqk, dv = int(helper.head_dim_qk), int(helper.head_dim_v)
    dropout = float(helper.dropout_probability)
    bias_name = helper.attn_bias_type.name
    mask_name = helper.attn_mask_type.name
    softmax_name = helper.softmax_type.name
    left, right = helper.window_size
    deterministic = not bool(int(os.getenv("NVTE_ALLOW_NONDETERMINISTIC_ALGO", "1")))

    architecture_ok = (
        (version < 8903 and arch in (80, 90))
        or (version >= 8903 and 80 <= arch < 100)
        or (version >= 90700 and arch >= 100)
    )
    if version < 8900 or not architecture_ok:
        return False
    if version < 90000 and (sq % 64 or skv % 64):
        return False
    if version < 8907 and h != hg:
        return False
    if dqk % 8 or dv % 8:
        return False

    standard_dim = dqk <= 128 and dv <= 128
    hopper_large_dim = (
        dqk <= 256
        and dv <= 256
        and (
            (not is_training and arch == 90 and version >= 90100)
            or (is_training and arch == 90 and version >= 90500)
        )
    )
    blackwell_fwd_any_dim = (
        not is_training and arch >= 100 and version >= 90900 and sq > 1
    )
    generic_fwd_any_dim = (
        not is_training
        and version >= 91002
        and (
            sq > 1
            or (sq == 1 and mask_name not in ("CAUSAL_MASK", "PADDING_CAUSAL_MASK"))
        )
    )
    blackwell_mla_bwd = (
        dqk == 192 and dv == 128 and is_training and arch >= 100 and version >= 91100
    )
    blackwell_d256_bwd = (
        dqk == 256
        and dv == 256
        and is_training
        and 100 <= arch < 110
        and version >= (92500 if is_thd else 92300)
        and bias_name == "NO_BIAS"
        and dropout == 0.0
        and softmax_name == "VANILLA_SOFTMAX"
        and (
            (left == -1 and right == -1)
            or (
                mask_name
                in (
                    "CAUSAL_MASK",
                    "PADDING_CAUSAL_MASK",
                    "CAUSAL_BOTTOM_RIGHT_MASK",
                    "PADDING_CAUSAL_BOTTOM_RIGHT_MASK",
                )
                and right in (-1, 0)
            )
        )
    )
    if not (
        standard_dim
        or hopper_large_dim
        or blackwell_fwd_any_dim
        or generic_fwd_any_dim
        or blackwell_mla_bwd
        or blackwell_d256_bwd
    ):
        return False
    if (
        version >= 91100
        and is_training
        and arch == 90
        and dqk >= 128
        and dv >= 128
        and (dqk, dv) != (192, 128)
        and dqk != dv
    ):
        return False

    post_scale_bias_supported = bias_name == "POST_SCALE_BIAS" and (
        (version >= 8906 and arch >= 90) or (version >= 90000 and arch >= 80)
    )
    if bias_name != "NO_BIAS" and not post_scale_bias_supported:
        return False

    dense_basic_masks = mask_name in (
        "NO_MASK",
        "CAUSAL_MASK",
        "PADDING_MASK",
        "PADDING_CAUSAL_MASK",
    )
    thd_basic_masks = mask_name in ("PADDING_MASK", "PADDING_CAUSAL_MASK")
    br_mask = mask_name == "CAUSAL_BOTTOM_RIGHT_MASK"
    padding_br_mask = mask_name == "PADDING_CAUSAL_BOTTOM_RIGHT_MASK"
    mask_ok = False
    if version < 8906:
        mask_ok = mask_name == "CAUSAL_MASK" and not is_thd
    elif not is_thd and dense_basic_masks:
        mask_ok = True
    if version >= 90100 and is_thd and thd_basic_masks:
        mask_ok = True
    if (
        version >= 90300
        and not is_thd
        and br_mask
        and sq % 64 == 0
        and skv % 64 == 0
        and sq <= skv
        and bias_name == "NO_BIAS"
        and dropout == 0.0
    ):
        mask_ok = True
    if (
        version >= 90600
        and padding_br_mask
        and sq % 64 == 0
        and skv % 64 == 0
        and sq <= skv
        and bias_name == "NO_BIAS"
        and dropout == 0.0
    ):
        mask_ok = True
    if version >= 90700:
        mask_ok = (
            mask_name in ("NO_MASK", "CAUSAL_MASK")
            or (
                mask_name
                in (
                    "PADDING_MASK",
                    "PADDING_CAUSAL_MASK",
                    "PADDING_CAUSAL_BOTTOM_RIGHT_MASK",
                )
                and bias_name == "NO_BIAS"
                and dropout == 0.0
            )
            or (
                mask_name
                in ("CAUSAL_BOTTOM_RIGHT_MASK", "PADDING_CAUSAL_BOTTOM_RIGHT_MASK")
                and sq <= skv
            )
        )
    if not mask_ok:
        return False
    if (
        mask_name in ("PADDING_MASK", "PADDING_CAUSAL_MASK")
        and bias_name == "POST_SCALE_BIAS"
    ):
        return False

    if is_thd and not (
        arch >= 90 and ((version >= 90100 and h == hg) or version >= 90600)
    ):
        return False

    full_window = left == -1 and right == -1
    if version < 90200:
        window_ok = left == -1 and right in (-1, 0)
    elif version < 90600:
        window_ok = (full_window and mask_name == "NO_MASK") or (
            left >= -1
            and right == 0
            and mask_name in ("NO_MASK", "CAUSAL_MASK", "CAUSAL_BOTTOM_RIGHT_MASK")
            and (mask_name != "CAUSAL_BOTTOM_RIGHT_MASK" or sq == skv)
            and sq <= skv
            and dropout == 0.0
            and bias_name == "NO_BIAS"
            and not is_thd
        )
    else:
        bottom_right_swa_supported = (
            mask_name
            not in ("CAUSAL_BOTTOM_RIGHT_MASK", "PADDING_CAUSAL_BOTTOM_RIGHT_MASK")
            or arch < 100
            or sq == skv
            or version > 90700
        )
        window_ok = (
            left == -1
            and right in (-1, 0)
            or (
                left >= -1
                and right >= -1
                and mask_name
                in (
                    "NO_MASK",
                    "CAUSAL_MASK",
                    "PADDING_MASK",
                    "PADDING_CAUSAL_MASK",
                    "CAUSAL_BOTTOM_RIGHT_MASK",
                    "PADDING_CAUSAL_BOTTOM_RIGHT_MASK",
                )
                and sq <= skv
                and bias_name == "NO_BIAS"
                and dropout == 0.0
                and bottom_right_swa_supported
            )
        )
    if not window_ok:
        return False

    if is_thd:
        if helper.qkv_layout.is_qkvpacked():
            max_offset = 3 * h * dqk * sq
        elif helper.qkv_layout.is_kvpacked():
            max_offset = max(h * dqk * sq, 2 * hg * dqk * skv)
        else:
            max_offset = max(h * dqk * sq, hg * dqk * skv, hg * dv * skv)
        if max_offset > np.iinfo(np.int32).max and version < 90500:
            return False

    if version in (91000, 91001):
        return False
    if version < 91301 and softmax_name != "VANILLA_SOFTMAX":
        return False
    if helper.return_max_logit and version < 92100:
        return False
    if arch >= 100 and is_training:
        if deterministic:
            if version < 91801 or dropout != 0.0 or bias_name != "NO_BIAS":
                return False
        elif dropout != 0.0 and bias_name != "NO_BIAS":
            return False
    if arch == 120 and (
        version < 91801
        or (deterministic and is_training)
        or (is_thd and helper.qkv_layout.is_qkvpacked())
    ):
        return False
    return not (
        version == 91400
        and skv > 1024
        and left != -1
        and mask_name not in ("CAUSAL_MASK", "CAUSAL_BOTTOM_RIGHT_MASK")
    )
