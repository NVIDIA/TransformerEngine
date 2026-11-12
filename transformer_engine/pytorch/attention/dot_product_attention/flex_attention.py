# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""cuDNN-backed Flex Attention helpers."""

from typing import Any, Callable, Dict, Optional, Tuple

import torch
from transformer_engine.pytorch.device import te_device_type

from transformer_engine.common.cudnn.attention import (
    AttentionOptions,
    AttentionGraphSpec,
    build_attention_graph,
)
from transformer_engine.common.cudnn.graph import (
    TensorSpec,
    GraphCacheEntry,
    check_graph_support,
    compile_graph,
)

from transformer_engine.common.cudnn.score_mod import (
    callback_cache_key,
    explicit_cache_key,
    freeze_cache_key,
    wrap_score_mod as _wrap_score_mod,
)

from transformer_engine.pytorch.cpp_extensions.cudnn import (
    _import_cudnn_frontend,
    _build_cudnn_pygraph,
    _graph_cache,
    _execute_cudnn_graph,
)

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
    if device.type == te_device_type():
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


def _make_cudnn_graph_tensor_dict(graph, tensors: Optional[Dict[str, torch.Tensor]]):
    """Create cuDNN graph tensors matching runtime tensors."""
    if tensors is None:
        return {}
    return {name: graph.tensor_like(tensor) for name, tensor in tensors.items()}


def _score_mod_bindings(tensors, extra, bprop_extra):
    """Keep callback operand names separate from SDPA and from each other."""
    return {
        **tensors,
        **{f"score_mod:{name}": tensor for name, tensor in extra.items()},
        **{f"score_mod_bprop:{name}": tensor for name, tensor in bprop_extra.items()},
    }


def _cudnn_score_mod_cache_key(spec, device, score_mod, score_mod_bprop, extra, bprop_extra):
    """Combine the shared SDPA descriptor with callback topology and operand metadata."""
    keys = tuple(
        _score_mod_callback_cache_key(callback) for callback in (score_mod, score_mod_bprop)
    )
    if any(key is _SCORE_MOD_UNCACHEABLE for key in keys):
        return None
    return (
        "score_mod",
        _score_mod_device_key(device),
        spec,
        keys,
        _score_mod_tensor_dict_metadata(extra),
        _score_mod_tensor_dict_metadata(bprop_extra),
    )


def _build_cudnn_score_mod_graph(spec, device, score_mod, score_mod_bprop, extra, bprop_extra):
    """Build and check either direction without retaining runtime tensors."""
    cudnn = _import_cudnn_frontend()
    graph = _build_cudnn_pygraph(getattr(cudnn.data_type, dict(spec.tensors)["q"].dtype), device)
    extra = _make_cudnn_graph_tensor_dict(graph, extra)
    bprop_extra = _make_cudnn_graph_tensor_dict(graph, bprop_extra)
    built = build_attention_graph(
        cudnn,
        graph,
        spec,
        score_mod=_wrap_score_mod(score_mod, extra),
        score_mod_bprop=_wrap_score_mod(score_mod_bprop, bprop_extra),
    )
    built.tensors = _score_mod_bindings(built.tensors, extra, bprop_extra)
    check_graph_support(cudnn, graph, has_score_mod=True)
    return GraphCacheEntry(built)


def _get_cudnn_score_mod_graph(
    options,
    tensors,
    q_format,
    kv_format,
    score_mod,
    score_mod_bprop=None,
    score_mod_tensors=None,
    score_mod_bprop_tensors=None,
):
    """Use the ordinary plan cache and return bindings for this invocation."""
    extra = score_mod_tensors or {}
    bprop_extra = (score_mod_bprop_tensors or {}) if score_mod_bprop is not None else {}
    outputs = ("dq", "dk", "dv") if options.backward else ("o", "stats")
    specs = {
        name: _graph_tensor_spec(
            tensor,
            (
                None
                if name == "stats"
                else (kv_format if name in ("k", "v", "dk", "dv") else q_format)
            ),
            output=name in outputs,
        )
        for name, tensor in tensors.items()
    }
    spec = AttentionGraphSpec.from_dict(options, specs)
    device = tensors["q"].device
    key = _cudnn_score_mod_cache_key(spec, device, score_mod, score_mod_bprop, extra, bprop_extra)
    entry = _graph_cache.get(
        key,
        lambda: _build_cudnn_score_mod_graph(
            spec, device, score_mod, score_mod_bprop, extra, bprop_extra
        ),
    )
    _graph_cache.compile(
        entry,
        lambda: compile_graph(_import_cudnn_frontend(), entry.built.graph, has_score_mod=True),
    )
    bindings = _score_mod_bindings(tensors, extra, bprop_extra)
    return entry, {entry.built.tensors[name]: tensor for name, tensor in bindings.items()}


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

        tensors = {"q": query_layer, "k": key_layer, "v": value_layer, "o": output_layer}
        if stats is not None:
            tensors["stats"] = stats
        entry, variant_pack = _get_cudnn_score_mod_graph(
            AttentionOptions(attn_scale=attn_scale, generate_stats=is_training),
            tensors,
            q_format,
            kv_format,
            score_mod,
            score_mod_tensors=score_mod_tensors,
        )

        _execute_cudnn_graph(
            entry.built.graph,
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
        entry, variant_pack = _get_cudnn_score_mod_graph(
            AttentionOptions(
                backward=True, attn_scale=ctx.attn_scale, deterministic=ctx.deterministic
            ),
            {
                "q": query_layer,
                "k": key_layer,
                "v": value_layer,
                "o": output_layer,
                "do": d_out,
                "stats": stats,
                "dq": dq_layer,
                "dk": dk_layer,
                "dv": dv_layer,
            },
            ctx.q_format,
            ctx.kv_format,
            ctx.score_mod,
            ctx.score_mod_bprop,
            score_mod_tensors,
            score_mod_bprop_tensors,
        )

        _execute_cudnn_graph(
            entry.built.graph,
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
