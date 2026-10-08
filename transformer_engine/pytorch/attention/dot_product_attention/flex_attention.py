# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""cuDNN-backed Flex Attention helpers."""

from dataclasses import dataclass
import inspect
from typing import Any, Callable, Dict, Optional, Tuple

import torch

from . import cudnn_pygraph

_BACKEND_NAME = "Flex Attention"
_cudnn_score_mod_graph_cache: Dict[Tuple[Any, ...], Any] = {}
_SCORE_MOD_UNCACHEABLE = object()


def _import_cudnn_frontend():
    """Import the cuDNN frontend Python package.

    Never enables the FROST engines: that switch is process-wide, so a backend that does not want
    them must not ask. See ``cudnn_pygraph.import_cudnn_frontend``.
    """
    return cudnn_pygraph.import_cudnn_frontend()


def _bhsd_dim_stride(
    tensor: torch.Tensor, tensor_format: str
) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """Describe an SBHD/BSHD tensor as cuDNN frontend's logical BHSD format."""
    return cudnn_pygraph.bhsd_dim_stride(tensor, tensor_format, backend_name=_BACKEND_NAME)


# score_mod graph cache helpers.
def _freeze_score_mod_cache_key(value: Any) -> Any:
    """Convert a user-provided score_mod graph key into a hashable structure."""
    if isinstance(value, torch.Tensor):
        raise TypeError(
            "score_mod_graph_cache_key() must not include tensors. Pass runtime tensors "
            "through score_mod_tensors or score_mod_bprop_tensors instead."
        )
    if isinstance(value, dict):
        items = (
            (
                _freeze_score_mod_cache_key(key),
                _freeze_score_mod_cache_key(val),
            )
            for key, val in value.items()
        )
        return tuple(sorted(items, key=repr))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_score_mod_cache_key(item) for item in value)
    if isinstance(value, (set, frozenset)):
        items = (_freeze_score_mod_cache_key(item) for item in value)
        return tuple(sorted(items, key=repr))
    try:
        hash(value)
    except TypeError as exc:
        raise TypeError(
            "score_mod_graph_cache_key() must return a hashable value or a nested "
            "combination of dict/list/tuple/set values."
        ) from exc
    return value


def _score_mod_explicit_cache_key(callback_owner: Any) -> Optional[Any]:
    """Return a user-provided structural graph key for a score_mod callback."""
    explicit_key = getattr(callback_owner, "score_mod_graph_cache_key", None)
    if explicit_key is None:
        return None
    explicit_key = explicit_key() if callable(explicit_key) else explicit_key
    return _freeze_score_mod_cache_key(explicit_key)


def _score_mod_callback_cache_key(callback: Optional[Callable]) -> Any:
    """Create a stable graph cache key for a score_mod callable.

    Module-level named functions are assumed to have stable topology. Anonymous functions
    are keyed by code object because lambdas in the same module can share the same
    qualname. Stateful bound methods and callable instances need an explicit
    score_mod_graph_cache_key(); otherwise their graphs are left uncached to avoid reusing
    stale graphs after Python object address reuse.
    """
    if callback is None:
        return None
    self_obj = getattr(callback, "__self__", None)
    func_obj = getattr(callback, "__func__", None)
    if self_obj is not None and func_obj is not None:
        explicit_key = _score_mod_explicit_cache_key(self_obj)
        if explicit_key is None:
            return _SCORE_MOD_UNCACHEABLE
        return (
            "bound_method",
            type(self_obj),
            func_obj.__module__,
            func_obj.__qualname__,
            explicit_key,
        )

    explicit_key = _score_mod_explicit_cache_key(callback)
    if explicit_key is not None:
        return (
            "callable",
            type(callback),
            getattr(callback, "__module__", None),
            getattr(callback, "__qualname__", None),
            explicit_key,
        )

    if (
        inspect.isfunction(callback)
        and callback.__closure__ is None
        and "<locals>" not in callback.__qualname__
    ):
        if callback.__name__ == "<lambda>" or not callback.__qualname__:
            return ("function", callback.__module__, callback.__code__)
        return ("function", callback.__module__, callback.__qualname__)

    return _SCORE_MOD_UNCACHEABLE


def _score_mod_device_key(device: torch.device) -> Tuple[Any, ...]:
    """Normalize a tensor device for graph cache keys."""
    return cudnn_pygraph.device_key(device)


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
    return cudnn_pygraph.tensor_key(tensor, tensor_format, backend_name=_BACKEND_NAME) + (
        cudnn_pygraph.device_key(tensor.device),
    )


# cuDNN frontend score_mod graph helpers.
def _wrap_score_mod(score_mod: Optional[Callable], graph_tensors: Dict[str, Any]):
    """Adapt TE's score_mod signature to cuDNN frontend's two-argument callback."""
    if score_mod is None:
        return None

    def _wrapped_score_mod(sdpa_graph, score_tensor):
        return score_mod(sdpa_graph, score_tensor, graph_tensors)

    return _wrapped_score_mod


def _finalize_cudnn_graph(graph) -> int:
    """Build a cuDNN frontend Python graph and return its workspace size."""
    workspace_size, _ = cudnn_pygraph.finalize_plans(graph, backend_name=_BACKEND_NAME)
    return workspace_size


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


def _execute_cudnn_graph(
    graph,
    variant_pack: Dict[Any, torch.Tensor],
    workspace_size: int,
    device: torch.device,
):
    """Execute a built cuDNN frontend Python graph."""
    cudnn_pygraph.execute_graph(
        graph, variant_pack, workspace_size, device, backend_name=_BACKEND_NAME
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


def _bhsd_tensor_spec(tensor: torch.Tensor, tensor_format: str):
    """Describe a tensor for ``cudnn_pygraph``: BHSD dims with TE-layout strides."""
    dim, stride = _bhsd_dim_stride(tensor, tensor_format)
    return (dim, stride, tensor.dtype)


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
    if is_training:
        assert stats is not None

    entry = cudnn_pygraph.build_fwd(
        dtype=query_layer.dtype,
        device=query_layer.device,
        backend_name=_BACKEND_NAME,
        name="te_score_mod_sdpa",
        q=_bhsd_tensor_spec(query_layer, q_format),
        k=_bhsd_tensor_spec(key_layer, kv_format),
        v=_bhsd_tensor_spec(value_layer, kv_format),
        out=_bhsd_dim_stride(output_layer, q_format),
        stats=((stats.size(), stats.stride(), cudnn.data_type.FLOAT) if is_training else None),
        attn_scale=attn_scale,
        aux_tensors={"score_mod": score_mod_tensors or {}},
        sdpa_kwargs=lambda aux: {
            "use_causal_mask": False,
            "score_mod": _wrap_score_mod(score_mod, aux["score_mod"]),
        },
    )
    return _CudnnScoreModFwdGraphEntry(
        graph=entry["graph"],
        q=entry["q"],
        k=entry["k"],
        v=entry["v"],
        output=entry["out"],
        stats=entry["stats"],
        score_mod_graph_tensors=entry["aux"]["score_mod"],
        workspace_size=entry["workspace"],
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
    return cudnn_pygraph.cached_graph(
        _cudnn_score_mod_graph_cache,
        _cudnn_score_mod_fwd_cache_key(*build_args),
        lambda: _build_cudnn_score_mod_fwd_graph(*build_args),
        device=query_layer.device,
    )


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
    dq_layer = torch.empty_like(query_layer)
    dk_layer = torch.empty_like(key_layer)
    dv_layer = torch.empty_like(value_layer)

    entry = cudnn_pygraph.build_bwd(
        dtype=query_layer.dtype,
        device=query_layer.device,
        backend_name=_BACKEND_NAME,
        name="te_score_mod_sdpa_backward",
        q=_bhsd_tensor_spec(query_layer, q_format),
        k=_bhsd_tensor_spec(key_layer, kv_format),
        v=_bhsd_tensor_spec(value_layer, kv_format),
        o=_bhsd_tensor_spec(output_layer, q_format),
        do=_bhsd_tensor_spec(d_out, q_format),
        stats=stats,
        dq=_bhsd_dim_stride(dq_layer, q_format),
        dk=_bhsd_dim_stride(dk_layer, kv_format),
        dv=_bhsd_dim_stride(dv_layer, kv_format),
        attn_scale=attn_scale,
        deterministic=deterministic,
        aux_tensors={
            "score_mod": score_mod_tensors or {},
            "score_mod_bprop": (
                score_mod_bprop_tensors or {} if score_mod_bprop is not None else {}
            ),
        },
        sdpa_kwargs=lambda aux: {
            "use_causal_mask": False,
            "score_mod": _wrap_score_mod(score_mod, aux["score_mod"]),
            "score_mod_bprop": _wrap_score_mod(score_mod_bprop, aux["score_mod_bprop"]),
        },
    )
    return _CudnnScoreModBwdGraphEntry(
        graph=entry["graph"],
        q=entry["q"],
        k=entry["k"],
        v=entry["v"],
        output=entry["o"],
        d_output=entry["do"],
        stats=entry["stats"],
        dq=entry["dq"],
        dk=entry["dk"],
        dv=entry["dv"],
        score_mod_graph_tensors=entry["aux"]["score_mod"],
        score_mod_bprop_graph_tensors=entry["aux"]["score_mod_bprop"],
        workspace_size=entry["workspace"],
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
    return cudnn_pygraph.cached_graph(
        _cudnn_score_mod_graph_cache,
        _cudnn_score_mod_bwd_cache_key(*build_args),
        lambda: _build_cudnn_score_mod_bwd_graph(*build_args),
        device=query_layer.device,
    )


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
