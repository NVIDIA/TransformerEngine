# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""cuDNN-backed Flex Attention helpers."""

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import torch

from transformer_engine.common.attention.score_mod import (
    UNCACHEABLE_SCORE_MOD,
    score_mod_callback_cache_key,
)

from ._cudnn_graph import (
    current_stream_handle,
    finalize_graph,
    import_cudnn_frontend,
    make_graph,
    torch_to_cudnn_dtype,
)

_cudnn_score_mod_graph_cache: Dict[Tuple[Any, ...], Any] = {}
_SCORE_MOD_UNCACHEABLE = UNCACHEABLE_SCORE_MOD


def _import_cudnn_frontend():
    """Compatibility wrapper around the shared attention graph runtime."""
    return import_cudnn_frontend()


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
    raise ValueError(
        f"Flex Attention only supports SBHD/BSHD tensor formats, got {tensor_format}."
    )


def _bhsd_graph_tensor(graph, tensor: torch.Tensor, tensor_format: str):
    """Create a cuDNN graph tensor with BHSD dims and TE-layout strides."""
    dim, stride = _bhsd_dim_stride(tensor, tensor_format)
    return graph.tensor(dim=dim, stride=stride, data_type=tensor.dtype)


# score_mod graph cache helpers.
def _score_mod_callback_cache_key(callback: Optional[Callable]) -> Any:
    """Compatibility wrapper around the shared score-modification key policy."""

    return score_mod_callback_cache_key(
        callback,
        is_array=lambda item: isinstance(item, torch.Tensor),
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
    return tuple(
        (name, _score_mod_tensor_metadata(tensor)) for name, tensor in tensors.items()
    )


def _score_mod_bhsd_tensor_metadata(
    tensor: torch.Tensor, tensor_format: str
) -> Tuple[Any, ...]:
    """Describe an SBHD/BSHD runtime tensor as a cuDNN BHSD graph tensor."""
    dim, stride = _bhsd_dim_stride(tensor, tensor_format)
    return (dim, stride, tensor.dtype, _score_mod_device_key(tensor.device))


def _make_cudnn_graph_tensor_dict(graph, tensors: Optional[Dict[str, torch.Tensor]]):
    """Create cuDNN graph tensors matching runtime tensors."""
    if tensors is None:
        return {}
    return {name: graph.tensor_like(tensor) for name, tensor in tensors.items()}


# cuDNN frontend score_mod graph helpers.
def _wrap_score_mod(score_mod: Optional[Callable], graph_tensors: Dict[str, Any]):
    """Adapt TE's score_mod signature to cuDNN frontend's two-argument callback."""
    if score_mod is None:
        return None

    def _wrapped_score_mod(sdpa_graph, score_tensor):
        return score_mod(sdpa_graph, score_tensor, graph_tensors)

    return _wrapped_score_mod


def _get_cudnn_current_stream_handle(cudnn, device: torch.device):
    """Compatibility wrapper around the shared current-stream handle."""
    del cudnn
    return current_stream_handle(device)


def _build_cudnn_pygraph(dtype: torch.dtype, device: torch.device):
    """Create a cuDNN frontend Python graph for F16/BF16 SDPA."""
    if dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(
            f"Flex Attention only supports FP16/BF16 tensors, got {dtype}."
        )
    return make_graph(torch_to_cudnn_dtype(dtype), device, name="te_flex_attention")


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


def _finalize_cudnn_graph(graph) -> int:
    """Compatibility wrapper around shared graph finalization."""
    return finalize_graph(graph)


def _execute_cudnn_graph(
    graph,
    variant_pack: Dict[Any, torch.Tensor],
    workspace_size: int,
    device: torch.device,
):
    """Execute a built cuDNN frontend Python graph."""
    cudnn = _import_cudnn_frontend()

    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    workspace = torch.empty(
        workspace_size,
        device=device,
        dtype=torch.uint8,
    )
    graph.execute(
        variant_pack,
        workspace,
        handle=_get_cudnn_current_stream_handle(cudnn, device),
    )


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

    cuDNN exposes graph.key(), but only after graph construction has run the user
    callback. This key avoids rebuilding the Python graph on cache hits.
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
    if (
        score_mod_key is _SCORE_MOD_UNCACHEABLE
        or score_mod_bprop_key is _SCORE_MOD_UNCACHEABLE
    ):
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
    q = _bhsd_graph_tensor(graph, query_layer, q_format)
    k = _bhsd_graph_tensor(graph, key_layer, kv_format)
    v = _bhsd_graph_tensor(graph, value_layer, kv_format)

    score_mod_graph_tensors = _make_cudnn_graph_tensor_dict(graph, score_mod_tensors)
    wrapped_score_mod = _wrap_score_mod(score_mod, score_mod_graph_tensors)

    output_dim, output_stride = _bhsd_dim_stride(output_layer, q_format)
    output, stats_tensor = graph.sdpa(
        name="te_score_mod_sdpa",
        q=q,
        k=k,
        v=v,
        generate_stats=is_training,
        attn_scale=attn_scale,
        use_causal_mask=False,
        score_mod=wrapped_score_mod,
    )
    output.set_output(True).set_dim(output_dim).set_stride(output_stride)

    if is_training:
        assert stats is not None
        stats_tensor.set_output(True).set_dim(stats.size()).set_stride(
            stats.stride()
        ).set_data_type(cudnn.data_type.FLOAT)
    else:
        stats_tensor = None

    workspace_size = _finalize_cudnn_graph(graph)
    return _CudnnScoreModFwdGraphEntry(
        graph=graph,
        q=q,
        k=k,
        v=v,
        output=output,
        stats=stats_tensor,
        score_mod_graph_tensors=score_mod_graph_tensors,
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
    graph = _build_cudnn_pygraph(query_layer.dtype, query_layer.device)
    q = _bhsd_graph_tensor(graph, query_layer, q_format)
    k = _bhsd_graph_tensor(graph, key_layer, kv_format)
    v = _bhsd_graph_tensor(graph, value_layer, kv_format)
    output = _bhsd_graph_tensor(graph, output_layer, q_format)
    d_output = _bhsd_graph_tensor(graph, d_out, q_format)
    stats_tensor = graph.tensor_like(stats)

    score_mod_graph_tensors = _make_cudnn_graph_tensor_dict(graph, score_mod_tensors)
    score_mod_bprop_graph_tensors = (
        _make_cudnn_graph_tensor_dict(graph, score_mod_bprop_tensors)
        if score_mod_bprop is not None
        else {}
    )
    wrapped_score_mod = _wrap_score_mod(score_mod, score_mod_graph_tensors)
    wrapped_score_mod_bprop = _wrap_score_mod(
        score_mod_bprop, score_mod_bprop_graph_tensors
    )

    dq_layer = torch.empty_like(query_layer)
    dk_layer = torch.empty_like(key_layer)
    dv_layer = torch.empty_like(value_layer)
    dq_dim, dq_stride = _bhsd_dim_stride(dq_layer, q_format)
    dk_dim, dk_stride = _bhsd_dim_stride(dk_layer, kv_format)
    dv_dim, dv_stride = _bhsd_dim_stride(dv_layer, kv_format)
    dq, dk, dv = graph.sdpa_backward(
        name="te_score_mod_sdpa_backward",
        q=q,
        k=k,
        v=v,
        o=output,
        dO=d_output,
        stats=stats_tensor,
        attn_scale=attn_scale,
        use_causal_mask=False,
        score_mod=wrapped_score_mod,
        score_mod_bprop=wrapped_score_mod_bprop,
        use_deterministic_algorithm=deterministic,
    )
    dq.set_output(True).set_dim(dq_dim).set_stride(dq_stride)
    dk.set_output(True).set_dim(dk_dim).set_stride(dk_stride)
    dv.set_output(True).set_dim(dv_dim).set_stride(dv_stride)

    workspace_size = _finalize_cudnn_graph(graph)
    return _CudnnScoreModBwdGraphEntry(
        graph=graph,
        q=q,
        k=k,
        v=v,
        output=output,
        d_output=d_output,
        stats=stats_tensor,
        dq=dq,
        dk=dk,
        dv=dv,
        score_mod_graph_tensors=score_mod_graph_tensors,
        score_mod_bprop_graph_tensors=score_mod_bprop_graph_tensors,
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
        output_layer = torch.empty(
            output_shape, device=query_layer.device, dtype=query_layer.dtype
        )
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
                "score_mod backward requires DotProductAttention to be in "
                "training mode."
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
