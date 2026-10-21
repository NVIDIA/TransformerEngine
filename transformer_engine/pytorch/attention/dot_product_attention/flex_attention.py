# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""cuDNN-backed Flex Attention helpers."""

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import torch

from transformer_engine.common.cudnn.attention import (
    AttentionOptions,
    AttentionGraphSpec,
    build_attention_graph,
)
from transformer_engine.common.cudnn.graph import TensorSpec

from transformer_engine.common.cudnn.score_mod import (
    callback_cache_key,
    explicit_cache_key,
    freeze_cache_key,
    wrap_score_mod as _wrap_score_mod,
)

from transformer_engine.pytorch.cpp_extensions.cudnn import (
    _import_cudnn_frontend,
    _build_cudnn_pygraph,
    _finalize_cudnn_graph,
    _execute_cudnn_graph,
)

_cudnn_score_mod_graph_cache: Dict[Tuple[Any, ...], Any] = {}
_SCORE_MOD_UNCACHEABLE = object()


def _bhsd_dim_stride(
    tensor: torch.Tensor, tensor_format: str
) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """Describe an SBHD/BSHD tensor as cuDNN frontend's logical BHSD format."""
    if tensor_format == "sbhd":
        return (
            (tensor.shape[1], tensor.shape[2], tensor.shape[0], tensor.shape[3]),
            (tensor.stride(1), tensor.stride(2), tensor.stride(0), tensor.stride(3)),
        )
    if tensor_format == "bshd":
        return (
            (tensor.shape[0], tensor.shape[2], tensor.shape[1], tensor.shape[3]),
            (tensor.stride(0), tensor.stride(2), tensor.stride(1), tensor.stride(3)),
        )
    raise ValueError(f"Flex Attention only supports SBHD/BSHD tensor formats, got {tensor_format}.")


def _graph_tensor_spec(tensor, tensor_format=None, output=False):
    """Translate framework metadata into the shared graph descriptor."""
    dim, stride = (
        _bhsd_dim_stride(tensor, tensor_format)
        if tensor_format is not None
        else (tuple(tensor.shape), tuple(tensor.stride()))
    )
    dtype = {torch.float16: "HALF", torch.bfloat16: "BFLOAT16", torch.float32: "FLOAT"}[
        tensor.dtype
    ]
    return TensorSpec(dim, stride, dtype, output=output)


# score_mod graph cache helpers.
def _freeze_score_mod_cache_key(value: Any) -> Any:
    """Freeze a structural callback key using this framework's tensor predicate."""
    return freeze_cache_key(value, lambda value: isinstance(value, torch.Tensor))


def _score_mod_explicit_cache_key(callback_owner: Any) -> Optional[Any]:
    """Read a callback owner's optional structural key."""
    return explicit_cache_key(callback_owner, lambda value: isinstance(value, torch.Tensor))


def _score_mod_callback_cache_key(callback: Optional[Callable]) -> Any:
    """Share stable callbacks and distinguish module-level anonymous functions."""
    return callback_cache_key(
        callback, lambda value: isinstance(value, torch.Tensor), lambda: _SCORE_MOD_UNCACHEABLE
    )


def _score_mod_device_key(device: torch.device) -> Tuple[Any, ...]:
    """Normalize a tensor device for graph cache keys."""
    if device.type == "cuda":
        index = device.index
        if index is None:
            index = torch.cuda.current_device()
        return (device.type, index)
    return (device.type, device.index)


def _score_mod_tensor_metadata(tensor: torch.Tensor) -> Tuple[Any, ...]:
    """Describe tensor metadata that can affect cuDNN graph construction."""
    return (
        tuple(tensor.size()),
        tuple(tensor.stride()),
        tensor.dtype,
        _score_mod_device_key(tensor.device),
    )


def _score_mod_tensor_dict_metadata(
    tensors: Optional[Dict[str, torch.Tensor]],
) -> Tuple[Tuple[str, Tuple[Any, ...]], ...]:
    """Describe score_mod tensor parameters without including their values."""
    if tensors is None:
        return ()
    return tuple((name, _score_mod_tensor_metadata(tensor)) for name, tensor in tensors.items())


def _score_mod_bhsd_tensor_metadata(tensor: torch.Tensor, tensor_format: str) -> Tuple[Any, ...]:
    """Describe an SBHD/BSHD runtime tensor as a cuDNN BHSD graph tensor."""
    dim, stride = _bhsd_dim_stride(tensor, tensor_format)
    return (dim, stride, tensor.dtype, _score_mod_device_key(tensor.device))


def _make_cudnn_graph_tensor_dict(graph, tensors: Optional[Dict[str, torch.Tensor]]):
    """Create cuDNN graph tensors matching runtime tensors."""
    if tensors is None:
        return {}
    return {name: graph.tensor_like(tensor) for name, tensor in tensors.items()}


# cuDNN frontend score_mod graph helpers.
@dataclass
class _CudnnScoreModFwdGraphEntry:
    """Cached cuDNN frontend graph and graph tensor handles for score_mod fprop."""

    graph: Any
    q: Any
    k: Any
    v: Any
    output: Any
    stats: Optional[Any]
    score_mod_graph_tensors: Dict[str, Any]
    workspace_size: int


@dataclass
class _CudnnScoreModBwdGraphEntry:
    """Cached cuDNN frontend graph and graph tensor handles for score_mod bprop."""

    graph: Any
    q: Any
    k: Any
    v: Any
    output: Any
    d_output: Any
    stats: Any
    dq: Any
    dk: Any
    dv: Any
    score_mod_graph_tensors: Dict[str, Any]
    score_mod_bprop_graph_tensors: Dict[str, Any]
    workspace_size: int


def _cudnn_score_mod_fwd_cache_key(
    is_training: bool,
    query_layer: torch.Tensor,
    key_layer: torch.Tensor,
    value_layer: torch.Tensor,
    q_format: str,
    kv_format: str,
    attn_scale: float,
    score_mod: Callable,
    score_mod_tensors: Optional[Dict[str, torch.Tensor]],
    output_layer: torch.Tensor,
    stats: Optional[torch.Tensor],
) -> Optional[Tuple[Any, ...]]:
    """Pre-build cache key for score_mod fprop execution plans.

    cuDNN exposes graph.key(), but only after graph construction has run the user callback.
    This key avoids rebuilding the Python graph on cache hits.
    """
    score_mod_key = _score_mod_callback_cache_key(score_mod)
    if score_mod_key is _SCORE_MOD_UNCACHEABLE:
        return None
    return (
        "fwd",
        is_training,
        q_format,
        kv_format,
        attn_scale,
        score_mod_key,
        _score_mod_bhsd_tensor_metadata(query_layer, q_format),
        _score_mod_bhsd_tensor_metadata(key_layer, kv_format),
        _score_mod_bhsd_tensor_metadata(value_layer, kv_format),
        _score_mod_bhsd_tensor_metadata(output_layer, q_format),
        _score_mod_tensor_metadata(stats) if stats is not None else None,
        _score_mod_tensor_dict_metadata(score_mod_tensors),
    )


def _cudnn_score_mod_bwd_cache_key(
    query_layer: torch.Tensor,
    key_layer: torch.Tensor,
    value_layer: torch.Tensor,
    output_layer: torch.Tensor,
    d_out: torch.Tensor,
    stats: torch.Tensor,
    q_format: str,
    kv_format: str,
    attn_scale: float,
    score_mod: Callable,
    score_mod_bprop: Optional[Callable],
    score_mod_tensors: Optional[Dict[str, torch.Tensor]],
    score_mod_bprop_tensors: Optional[Dict[str, torch.Tensor]],
    deterministic: bool,
) -> Optional[Tuple[Any, ...]]:
    """Pre-build cache key for score_mod bprop execution plans."""
    score_mod_key = _score_mod_callback_cache_key(score_mod)
    score_mod_bprop_key = _score_mod_callback_cache_key(score_mod_bprop)
    if score_mod_key is _SCORE_MOD_UNCACHEABLE or score_mod_bprop_key is _SCORE_MOD_UNCACHEABLE:
        return None
    return (
        "bwd",
        q_format,
        kv_format,
        attn_scale,
        deterministic,
        score_mod_key,
        score_mod_bprop_key,
        _score_mod_bhsd_tensor_metadata(query_layer, q_format),
        _score_mod_bhsd_tensor_metadata(key_layer, kv_format),
        _score_mod_bhsd_tensor_metadata(value_layer, kv_format),
        _score_mod_bhsd_tensor_metadata(output_layer, q_format),
        _score_mod_bhsd_tensor_metadata(d_out, q_format),
        _score_mod_tensor_metadata(stats),
        _score_mod_tensor_dict_metadata(score_mod_tensors),
        _score_mod_tensor_dict_metadata(score_mod_bprop_tensors),
    )


def _build_cudnn_score_mod_fwd_graph(
    is_training: bool,
    query_layer: torch.Tensor,
    key_layer: torch.Tensor,
    value_layer: torch.Tensor,
    q_format: str,
    kv_format: str,
    attn_scale: float,
    score_mod: Callable,
    score_mod_tensors: Optional[Dict[str, torch.Tensor]],
    output_layer: torch.Tensor,
    stats: Optional[torch.Tensor],
) -> _CudnnScoreModFwdGraphEntry:
    """Build a cached cuDNN frontend graph for score_mod fprop."""
    cudnn = _import_cudnn_frontend()

    graph = _build_cudnn_pygraph(query_layer.dtype, query_layer.device)
    extra = _make_cudnn_graph_tensor_dict(graph, score_mod_tensors)
    specs = {
        "q": _graph_tensor_spec(query_layer, q_format),
        "k": _graph_tensor_spec(key_layer, kv_format),
        "v": _graph_tensor_spec(value_layer, kv_format),
        "o": _graph_tensor_spec(output_layer, q_format, output=True),
    }
    if is_training:
        assert stats is not None
        specs["stats"] = _graph_tensor_spec(stats, output=True)
    built = build_attention_graph(
        cudnn,
        graph,
        AttentionGraphSpec.from_dict(
            AttentionOptions(attn_scale=attn_scale, generate_stats=is_training), specs
        ),
        score_mod=_wrap_score_mod(score_mod, extra),
    )
    workspace_size = _finalize_cudnn_graph(graph)
    return _CudnnScoreModFwdGraphEntry(
        graph=graph,
        q=built.tensors["q"],
        k=built.tensors["k"],
        v=built.tensors["v"],
        output=built.tensors["o"],
        stats=built.tensors.get("stats"),
        score_mod_graph_tensors=extra,
        workspace_size=workspace_size,
    )


def _get_cudnn_score_mod_fwd_graph(
    is_training: bool,
    query_layer: torch.Tensor,
    key_layer: torch.Tensor,
    value_layer: torch.Tensor,
    q_format: str,
    kv_format: str,
    attn_scale: float,
    score_mod: Callable,
    score_mod_tensors: Optional[Dict[str, torch.Tensor]],
    output_layer: torch.Tensor,
    stats: Optional[torch.Tensor],
) -> _CudnnScoreModFwdGraphEntry:
    """Return a cached cuDNN frontend graph for score_mod fprop."""
    build_args = (
        is_training,
        query_layer,
        key_layer,
        value_layer,
        q_format,
        kv_format,
        attn_scale,
        score_mod,
        score_mod_tensors,
        output_layer,
        stats,
    )
    key = _cudnn_score_mod_fwd_cache_key(*build_args)
    if key is None:
        return _build_cudnn_score_mod_fwd_graph(*build_args)
    entry = _cudnn_score_mod_graph_cache.get(key)
    if entry is None:
        entry = _build_cudnn_score_mod_fwd_graph(*build_args)
        _cudnn_score_mod_graph_cache[key] = entry
    return entry


def _build_cudnn_score_mod_bwd_graph(
    query_layer: torch.Tensor,
    key_layer: torch.Tensor,
    value_layer: torch.Tensor,
    output_layer: torch.Tensor,
    d_out: torch.Tensor,
    stats: torch.Tensor,
    q_format: str,
    kv_format: str,
    attn_scale: float,
    score_mod: Callable,
    score_mod_bprop: Optional[Callable],
    score_mod_tensors: Optional[Dict[str, torch.Tensor]],
    score_mod_bprop_tensors: Optional[Dict[str, torch.Tensor]],
    deterministic: bool,
) -> _CudnnScoreModBwdGraphEntry:
    """Build a cached cuDNN frontend graph for score_mod bprop."""
    cudnn = _import_cudnn_frontend()
    graph = _build_cudnn_pygraph(query_layer.dtype, query_layer.device)
    extra = _make_cudnn_graph_tensor_dict(graph, score_mod_tensors)
    bprop_extra = (
        _make_cudnn_graph_tensor_dict(graph, score_mod_bprop_tensors)
        if score_mod_bprop is not None
        else {}
    )
    specs = {
        "q": _graph_tensor_spec(query_layer, q_format),
        "k": _graph_tensor_spec(key_layer, kv_format),
        "v": _graph_tensor_spec(value_layer, kv_format),
        "o": _graph_tensor_spec(output_layer, q_format),
        "do": _graph_tensor_spec(d_out, q_format),
        "stats": _graph_tensor_spec(stats),
        "dq": _graph_tensor_spec(torch.empty_like(query_layer), q_format, output=True),
        "dk": _graph_tensor_spec(torch.empty_like(key_layer), kv_format, output=True),
        "dv": _graph_tensor_spec(torch.empty_like(value_layer), kv_format, output=True),
    }
    built = build_attention_graph(
        cudnn,
        graph,
        AttentionGraphSpec.from_dict(
            AttentionOptions(backward=True, attn_scale=attn_scale, deterministic=deterministic),
            specs,
        ),
        score_mod=_wrap_score_mod(score_mod, extra),
        score_mod_bprop=_wrap_score_mod(score_mod_bprop, bprop_extra),
    )
    workspace_size = _finalize_cudnn_graph(graph)
    return _CudnnScoreModBwdGraphEntry(
        graph=graph,
        q=built.tensors["q"],
        k=built.tensors["k"],
        v=built.tensors["v"],
        output=built.tensors["o"],
        d_output=built.tensors["do"],
        stats=built.tensors["stats"],
        dq=built.tensors["dq"],
        dk=built.tensors["dk"],
        dv=built.tensors["dv"],
        score_mod_graph_tensors=extra,
        score_mod_bprop_graph_tensors=bprop_extra,
        workspace_size=workspace_size,
    )


def _get_cudnn_score_mod_bwd_graph(
    query_layer: torch.Tensor,
    key_layer: torch.Tensor,
    value_layer: torch.Tensor,
    output_layer: torch.Tensor,
    d_out: torch.Tensor,
    stats: torch.Tensor,
    q_format: str,
    kv_format: str,
    attn_scale: float,
    score_mod: Callable,
    score_mod_bprop: Optional[Callable],
    score_mod_tensors: Optional[Dict[str, torch.Tensor]],
    score_mod_bprop_tensors: Optional[Dict[str, torch.Tensor]],
    deterministic: bool,
) -> _CudnnScoreModBwdGraphEntry:
    """Return a cached cuDNN frontend graph for score_mod bprop."""
    build_args = (
        query_layer,
        key_layer,
        value_layer,
        output_layer,
        d_out,
        stats,
        q_format,
        kv_format,
        attn_scale,
        score_mod,
        score_mod_bprop,
        score_mod_tensors,
        score_mod_bprop_tensors,
        deterministic,
    )
    key = _cudnn_score_mod_bwd_cache_key(*build_args)
    if key is None:
        return _build_cudnn_score_mod_bwd_graph(*build_args)
    entry = _cudnn_score_mod_graph_cache.get(key)
    if entry is None:
        entry = _build_cudnn_score_mod_bwd_graph(*build_args)
        _cudnn_score_mod_graph_cache[key] = entry
    return entry


class FusedAttentionWithScoreModFunc(torch.autograd.Function):
    """cuDNN frontend Python SDPA path with Flex Attention score_mod support."""

    @staticmethod
    def forward(
        ctx,
        is_training: bool,
        query_layer: torch.Tensor,
        key_layer: torch.Tensor,
        value_layer: torch.Tensor,
        q_format: str,
        kv_format: str,
        attn_scale: float,
        score_mod: Callable,
        score_mod_bprop: Optional[Callable],
        score_mod_tensors: Optional[Dict[str, torch.Tensor]],
        score_mod_bprop_tensors: Optional[Dict[str, torch.Tensor]],
        deterministic: bool,
    ) -> torch.Tensor:
        # pylint: disable=missing-function-docstring
        q_bhsd_dim, _ = _bhsd_dim_stride(query_layer, q_format)
        score_mod_tensors = dict(score_mod_tensors or {})
        score_mod_bprop_tensors = dict(score_mod_bprop_tensors or {})
        output_shape = (*query_layer.shape[:-1], value_layer.shape[-1])
        output_layer = torch.empty(output_shape, device=query_layer.device, dtype=query_layer.dtype)
        if is_training:
            stats = torch.empty(
                (*q_bhsd_dim[:-1], 1),
                device=query_layer.device,
                dtype=torch.float32,
            )
        else:
            stats = None

        entry = _get_cudnn_score_mod_fwd_graph(
            is_training,
            query_layer,
            key_layer,
            value_layer,
            q_format,
            kv_format,
            attn_scale,
            score_mod,
            score_mod_tensors,
            output_layer,
            stats,
        )
        variant_pack = {
            entry.q: query_layer,
            entry.k: key_layer,
            entry.v: value_layer,
            entry.output: output_layer,
        }
        if is_training:
            variant_pack[entry.stats] = stats
        for name, graph_tensor in entry.score_mod_graph_tensors.items():
            variant_pack[graph_tensor] = score_mod_tensors[name]

        _execute_cudnn_graph(
            entry.graph,
            variant_pack,
            entry.workspace_size,
            query_layer.device,
        )

        ctx.is_training = is_training
        ctx.q_format = q_format
        ctx.kv_format = kv_format
        ctx.attn_scale = attn_scale
        ctx.score_mod = score_mod
        ctx.score_mod_bprop = score_mod_bprop
        ctx.score_mod_tensor_names = tuple(score_mod_tensors.keys())
        ctx.score_mod_bprop_tensor_names = tuple(score_mod_bprop_tensors.keys())
        ctx.deterministic = deterministic
        if is_training:
            # save_for_backward records version counters without copying tensor data.
            # This catches in-place score_mod tensor updates before backward.
            ctx.save_for_backward(
                query_layer,
                key_layer,
                value_layer,
                output_layer,
                stats,
                *score_mod_tensors.values(),
                *score_mod_bprop_tensors.values(),
            )
        else:
            ctx.save_for_backward(query_layer, key_layer, value_layer, output_layer)

        return output_layer

    @staticmethod
    def backward(ctx, d_out: torch.Tensor):
        # pylint: disable=missing-function-docstring
        if not ctx.is_training:
            raise RuntimeError(
                "score_mod backward requires DotProductAttention to be in training mode."
            )

        saved_tensors = ctx.saved_tensors
        query_layer, key_layer, value_layer, output_layer, stats = saved_tensors[:5]
        score_mod_tensors_end = 5 + len(ctx.score_mod_tensor_names)
        score_mod_tensors = dict(
            zip(ctx.score_mod_tensor_names, saved_tensors[5:score_mod_tensors_end])
        )
        score_mod_bprop_tensors = dict(
            zip(ctx.score_mod_bprop_tensor_names, saved_tensors[score_mod_tensors_end:])
        )
        d_out = d_out.contiguous()

        dq_layer = torch.empty_like(query_layer)
        dk_layer = torch.empty_like(key_layer)
        dv_layer = torch.empty_like(value_layer)
        entry = _get_cudnn_score_mod_bwd_graph(
            query_layer,
            key_layer,
            value_layer,
            output_layer,
            d_out,
            stats,
            ctx.q_format,
            ctx.kv_format,
            ctx.attn_scale,
            ctx.score_mod,
            ctx.score_mod_bprop,
            score_mod_tensors,
            score_mod_bprop_tensors,
            ctx.deterministic,
        )
        variant_pack = {
            entry.q: query_layer,
            entry.k: key_layer,
            entry.v: value_layer,
            entry.output: output_layer,
            entry.d_output: d_out,
            entry.stats: stats,
            entry.dq: dq_layer,
            entry.dk: dk_layer,
            entry.dv: dv_layer,
        }
        for name, graph_tensor in entry.score_mod_graph_tensors.items():
            variant_pack[graph_tensor] = score_mod_tensors[name]
        for name, graph_tensor in entry.score_mod_bprop_graph_tensors.items():
            variant_pack[graph_tensor] = score_mod_bprop_tensors[name]

        _execute_cudnn_graph(
            entry.graph,
            variant_pack,
            entry.workspace_size,
            query_layer.device,
        )

        return (
            None,
            dq_layer,
            dk_layer,
            dv_layer,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )
