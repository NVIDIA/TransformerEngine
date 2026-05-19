# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""cuDNN-backed flex attention helpers."""

from dataclasses import dataclass
import importlib
import importlib.util
import inspect
from pathlib import Path
import sys
from typing import Any, Callable, Dict, Optional, Tuple

import torch

_cudnn_score_mod_handles: Dict[torch.device, Any] = {}
_cudnn_score_mod_graph_cache: Dict[Tuple[Any, ...], Any] = {}
_SCORE_MOD_UNCACHEABLE = object()
_CUDNN_FRONTEND_PYTHON_PATH = (
    Path(__file__).resolve().parents[4] / "3rdparty" / "cudnn-frontend" / "python"
)


def _import_cudnn_frontend():
    """Import the vendored cuDNN frontend if built, otherwise use the installed package."""
    cudnn_frontend_path = str(_CUDNN_FRONTEND_PYTHON_PATH)
    cudnn_frontend_package = _CUDNN_FRONTEND_PYTHON_PATH / "cudnn"
    if any(cudnn_frontend_package.glob("_compiled_module*")):
        if cudnn_frontend_path not in sys.path:
            sys.path.insert(0, cudnn_frontend_path)
        return importlib.import_module("cudnn")

    if importlib.util.find_spec("cudnn") is not None:
        return importlib.import_module("cudnn")

    raise ImportError(
        "cuDNN Frontend Python package not found. "
        "Install it with: pip install nvidia-cudnn-frontend"
    )


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
    raise ValueError(f"score_mod only supports SBHD/BSHD tensor formats, got {tensor_format}.")


def _bhsd_graph_tensor(graph, tensor: torch.Tensor, tensor_format: str):
    """Create a cuDNN graph tensor with BHSD dims and TE-layout strides."""
    dim, stride = _bhsd_dim_stride(tensor, tensor_format)
    return graph.tensor(dim=dim, stride=stride, data_type=tensor.dtype)


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


# score_mod cuDNN frontend graph helpers.
def _wrap_score_mod(score_mod: Optional[Callable], graph_tensors: Dict[str, Any]):
    """Adapt TE's score_mod signature to cuDNN frontend's two-argument callback."""
    if score_mod is None:
        return None

    def _wrapped_score_mod(sdpa_graph, score_tensor):
        return score_mod(sdpa_graph, score_tensor, graph_tensors)

    return _wrapped_score_mod


def _get_cudnn_current_stream_handle(cudnn, device: torch.device):
    """Return a cuDNN handle for device, bound to PyTorch's current stream."""
    if device.type != "cuda":
        raise ValueError(f"score_mod only supports CUDA tensors, got device {device}.")
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())

    handle = _cudnn_score_mod_handles.get(device)
    with torch.cuda.device(device):
        if handle is None:
            handle = cudnn.create_handle()
            _cudnn_score_mod_handles[device] = handle

        stream = torch.cuda.current_stream(device).cuda_stream
        cudnn.set_stream(handle=handle, stream=stream)
    return handle


def _build_cudnn_pygraph(dtype: torch.dtype, device: torch.device):
    """Create a cuDNN frontend Python graph for F16/BF16 SDPA."""
    cudnn = _import_cudnn_frontend()

    if dtype == torch.float16:
        io_data_type = cudnn.data_type.HALF
    elif dtype == torch.bfloat16:
        io_data_type = cudnn.data_type.BFLOAT16
    else:
        raise ValueError(f"score_mod only supports FP16/BF16 tensors, got {dtype}.")

    graph = cudnn.pygraph(
        io_data_type=io_data_type,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
        handle=_get_cudnn_current_stream_handle(cudnn, device),
    )
    return cudnn, graph


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
    """Build a cuDNN frontend Python graph and return its workspace size."""
    cudnn = _import_cudnn_frontend()

    graph.validate()
    graph.build_operation_graph()
    try:
        graph.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
        graph.check_support()
    except cudnn.cudnnGraphNotSupportedError as exc:
        raise RuntimeError(f"cuDNN score_mod SDPA graph is not supported: {exc}") from exc
    graph.build_plans(cudnn.build_plan_policy.HEURISTICS_CHOICE)
    return max(graph.get_workspace_size(), 1)


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
    output_layer: torch.Tensor,
    stats: Optional[torch.Tensor],
    q_format: str,
    kv_format: str,
    attn_scale: float,
    score_mod: Callable,
    score_mod_tensors: Optional[Dict[str, torch.Tensor]],
) -> Optional[Tuple[Any, ...]]:
    """Cache key for score_mod fprop execution plans."""
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
    """Cache key for score_mod bprop execution plans."""
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

    _, graph = _build_cudnn_pygraph(query_layer.dtype, query_layer.device)
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
    key = _cudnn_score_mod_fwd_cache_key(
        is_training,
        query_layer,
        key_layer,
        value_layer,
        output_layer,
        stats,
        q_format,
        kv_format,
        attn_scale,
        score_mod,
        score_mod_tensors,
    )
    if key is None:
        return _build_cudnn_score_mod_fwd_graph(
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
    entry = _cudnn_score_mod_graph_cache.get(key)
    if entry is None:
        entry = _build_cudnn_score_mod_fwd_graph(
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
    _, graph = _build_cudnn_pygraph(query_layer.dtype, query_layer.device)
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
    wrapped_score_mod_bprop = _wrap_score_mod(score_mod_bprop, score_mod_bprop_graph_tensors)

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
    key = _cudnn_score_mod_bwd_cache_key(
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
    if key is None:
        return _build_cudnn_score_mod_bwd_graph(
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
    entry = _cudnn_score_mod_graph_cache.get(key)
    if entry is None:
        entry = _build_cudnn_score_mod_bwd_graph(
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
        _cudnn_score_mod_graph_cache[key] = entry
    return entry


class FusedAttentionWithScoreModFunc(torch.autograd.Function):
    """cuDNN frontend Python SDPA path with score_mod callback support."""

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
