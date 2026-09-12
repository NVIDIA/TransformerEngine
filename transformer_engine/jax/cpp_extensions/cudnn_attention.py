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

from transformer_engine.common.attention.cudnn import (
    AttentionLayout,
    FusedAttentionConfig,
    FusedAttentionSupport,
    check_f16_fused_attention_support,
    cudnn_mask_options,
    ragged_batch_bucket,
    ragged_token_bucket,
)

from .cudnn_graph import (
    GraphBinding,
    SerializedGraph,
    cudnn_data_type,
    dtype_name,
    finalize_graph,
    import_cudnn,
    make_graph,
    record_cache_event,
    record_cache_lookup,
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
            raise ValueError(f"QKV-packed fused attention expects dimension 3, got {q_aval.shape}.")
        kv_seqlen = q_seqlen
        kv_heads = q_heads
        v_dim = qk_dim
    elif layout.is_kvpacked():
        *batch_shape, q_seqlen, q_heads, qk_dim = q_aval.shape
        *kv_batch_shape, kv_seqlen, packed, kv_heads, v_dim = k_aval.shape
        if tuple(batch_shape) != tuple(kv_batch_shape) or packed != 2:
            raise ValueError(f"Invalid KV-packed fused-attention shapes: {q_aval}, {k_aval}.")
        if qk_dim != v_dim:
            raise ValueError("KV-packed fused attention requires equal QK and V head dimensions.")
    elif layout.is_separate():
        *batch_shape, q_seqlen, q_heads, qk_dim = q_aval.shape
        *k_batch_shape, kv_seqlen, kv_heads, k_dim = k_aval.shape
        *v_batch_shape, v_seqlen, v_heads, v_dim = v_aval.shape
        if tuple(batch_shape) != tuple(k_batch_shape) or tuple(batch_shape) != tuple(v_batch_shape):
            raise ValueError("Separate Q, K and V tensors must have matching batch shapes.")
        if qk_dim != k_dim or kv_seqlen != v_seqlen or kv_heads != v_heads:
            raise ValueError("Separate fused-attention K and V shapes are inconsistent.")
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


def _matrix_stride(info: _LayoutInfo, layout, matrix: str, graph_sq: int, graph_skv: int):
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
    return ragged_batch_bucket(batch)


def _ragged_graph_token_count(tokens: int) -> int:
    """Preserve the legacy cuDNN graph token-count bucket for ragged attention."""
    return ragged_token_bucket(tokens)


def _graph_dimensions(info: _LayoutInfo, config):
    """Return logical cuDNN B/H/S dimensions and physical auxiliary shapes."""
    is_ragged = config.qkv_layout.is_thd()
    cudnn_version = get_cudnn_version()
    arch = _device_arch()
    use_ragged_stats = is_ragged and cudnn_version >= (9, 6, 0) and arch != 120
    if is_ragged:
        graph_batch = ragged_graph_batch_size(info.input_batch, config.max_segments_per_seq)
        if cudnn_version < (9, 6, 0) or arch == 120:
            graph_sq = info.q_max_seqlen
            graph_skv = info.kv_max_seqlen
        else:
            graph_sq = _ragged_graph_token_count(info.input_batch * info.q_max_seqlen)
            graph_skv = _ragged_graph_token_count(info.input_batch * info.kv_max_seqlen)
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
    cp_striped_window_size = getattr(config, "cp_striped_window_size", None)
    window_left, window_right = (
        cp_striped_window_size if cp_striped_window_size is not None else config.window_size
    )
    options = cudnn_mask_options(
        causal=_is_causal(config),
        bottom_right=_is_bottom_right(config),
        padding=_is_padding(config),
        bottom_right_diagonal=bool(config.bottom_right_diagonal),
        window_size=(window_left, window_right),
        max_seqlen_q=info.q_max_seqlen,
        max_seqlen_kv=info.kv_max_seqlen,
        cudnn_version=get_cudnn_version(),
    )
    options.pop("is_padding")
    options["diagonal_alignment"] = (
        cudnn.diagonal_alignment.BOTTOM_RIGHT
        if options["diagonal_alignment"] == "bottom_right"
        else cudnn.diagonal_alignment.TOP_LEFT
    )
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
    graph_info = _graph_cache.get(key)
    record_cache_lookup(("f16", "fwd"), hit=graph_info is not None, key=key)
    if graph_info is None:
        _graph_cache[key] = _build_fwd_graph(q_aval, k_aval, v_aval, bias_aval, config)
        record_cache_event(("f16", "fwd"), "cache_graph")
        graph_info = _graph_cache[key]
    return graph_info


def _build_fwd_graph(q_aval, k_aval, v_aval, bias_aval, config) -> AttentionGraphInfo:
    cudnn = import_cudnn()
    info = _layout_info(q_aval, k_aval, v_aval, config.qkv_layout)
    graph_batch, graph_sq, graph_skv, ragged_stats, stats_shape, max_shape = _graph_dimensions(
        info, config
    )
    io_dtype = cudnn_data_type(cudnn, q_aval.dtype)
    graph = make_graph(cudnn, io_dtype)

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

    input_bindings = list(_qkv_bindings(info, config.qkv_layout, jnp.dtype(q_aval.dtype).itemsize))
    output_bindings = [GraphBinding(_UID_O, 0), GraphBinding(_UID_STATS, 1)]
    scale = _scalar_tensor(graph, cudnn, "attn_scale", _UID_ATTN_SCALE, cudnn.data_type.FLOAT)
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
    if getattr(config.attn_bias_type, "name", "") == "ALIBI":
        kwargs["use_alibi_mask"] = True

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
        input_bindings.extend((GraphBinding(_UID_SEQ_Q, 6), GraphBinding(_UID_SEQ_KV, 7)))

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

    cache_site = ("f16", "fwd")
    workspace, data, version = finalize_graph(
        cudnn,
        graph,
        description="fused-attention forward",
        cache_site=cache_site,
    )
    result = serialized_graph(
        serialized_graph_data=data,
        cudnn_frontend_version=version,
        workspace_size=workspace,
        cache_site=cache_site,
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
    graph_info = _graph_cache.get(key)
    record_cache_lookup(("f16", "bwd"), hit=graph_info is not None, key=key)
    if graph_info is None:
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
        record_cache_event(("f16", "bwd"), "cache_graph")
        graph_info = _graph_cache[key]
    return graph_info


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
    graph_batch, graph_sq, graph_skv, ragged_stats, stats_shape, max_shape = _graph_dimensions(
        info, config
    )
    io_dtype = cudnn_data_type(cudnn, q_aval.dtype)
    graph = make_graph(cudnn, io_dtype)

    def io_tensor(name, dim, stride, uid, dtype=io_dtype):
        return _tensor(graph, cudnn, name=name, dim=dim, stride=stride, dtype=dtype, uid=uid)

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
    output_bindings = list(_qkv_bindings(info, config.qkv_layout, itemsize, outputs=True))
    scale = _scalar_tensor(graph, cudnn, "attn_scale", _UID_ATTN_SCALE, cudnn.data_type.FLOAT)
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
    if getattr(config.attn_bias_type, "name", "") == "ALIBI":
        kwargs["use_alibi_mask"] = True
    if get_cudnn_version() >= (9, 0, 0):
        kwargs["use_deterministic_algorithm"] = not bool(
            int(os.getenv("NVTE_ALLOW_NONDETERMINISTIC_ALGO", "1"))
        )
    if ragged_stats:
        kwargs["max_total_seq_len_q"] = graph_sq
    if config.qkv_layout.is_thd() and get_cudnn_version() >= (9, 6, 0) and _device_arch() != 120:
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
        input_bindings.extend((GraphBinding(_UID_SEQ_Q, 9), GraphBinding(_UID_SEQ_KV, 10)))

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

    cache_site = ("f16", "bwd")
    workspace, data, version = finalize_graph(
        cudnn,
        graph,
        description="fused-attention backward",
        cache_site=cache_site,
    )
    result = serialized_graph(
        serialized_graph_data=data,
        cudnn_frontend_version=version,
        workspace_size=workspace,
        cache_site=cache_site,
        input_bindings=input_bindings,
        output_bindings=output_bindings,
        scalar_uids=scalar_uids,
        scalar_values=scalar_values,
    )
    return AttentionGraphInfo(result, stats_shape, max_shape)


def clear_graph_cache():
    """Clear the process-local serialized graph cache (primarily for tests)."""
    _graph_cache.clear()


def _policy_layout(layout) -> AttentionLayout:
    qkv_format = layout.get_qkv_format().name.lower()
    if layout.is_qkvpacked():
        layout_group = "qkv_packed"
    elif layout.is_kvpacked():
        layout_group = "kv_packed"
    else:
        layout_group = "separate"
    return AttentionLayout(
        qkv_format=qkv_format,
        q_format=qkv_format,
        kv_format=qkv_format,
        layout_group=layout_group,
        is_qkvpacked=layout.is_qkvpacked(),
    )


def _policy_mask_name(mask) -> str:
    return {
        "NO_MASK": "no_mask",
        "CAUSAL_MASK": "causal",
        "PADDING_MASK": "padding",
        "PADDING_CAUSAL_MASK": "padding_causal",
        "CAUSAL_BOTTOM_RIGHT_MASK": "causal_bottom_right",
        "PADDING_CAUSAL_BOTTOM_RIGHT_MASK": "padding_causal_bottom_right",
    }[mask.name]


def get_fused_attn_support(helper) -> FusedAttentionSupport:
    """Return the shared F16/BF16 cuDNN attention compatibility result."""

    return check_f16_fused_attention_support(
        FusedAttentionConfig(
            is_training=bool(helper.is_training),
            q_dtype=str(jnp.dtype(helper.q_dtype)),
            kv_dtype=str(jnp.dtype(helper.kv_dtype)),
            layout=_policy_layout(helper.qkv_layout),
            bias_type=helper.attn_bias_type.name.lower(),
            mask_type=_policy_mask_name(helper.attn_mask_type),
            softmax_type=helper.softmax_type.name.lower().removesuffix("_softmax"),
            dropout=float(helper.dropout_probability),
            num_attn_heads=int(helper.q_num_heads),
            num_gqa_groups=int(helper.kv_num_heads),
            max_seqlen_q=int(helper.q_max_seqlen),
            max_seqlen_kv=int(helper.kv_max_seqlen),
            head_dim_qk=int(helper.head_dim_qk),
            head_dim_v=int(helper.head_dim_v),
            window_size=tuple(int(value) for value in helper.window_size),
            return_max_logit=bool(helper.return_max_logit),
            cuda_graph=False,
            deterministic=not bool(int(os.getenv("NVTE_ALLOW_NONDETERMINISTIC_ALGO", "1"))),
            cudnn_version=get_cudnn_version(),
            sm_arch=_device_arch(),
        )
    )


def is_fused_attn_supported(helper) -> bool:
    """Apply the shared F16/BF16 cuDNN attention compatibility policy."""

    return get_fused_attn_support(helper).supported
