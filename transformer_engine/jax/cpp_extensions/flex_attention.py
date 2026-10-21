# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""cuDNN frontend score_mod fused attention helpers."""

import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import ffi

from transformer_engine.common.cudnn.attention import (
    AttentionOptions,
    AttentionGraphSpec,
    build_attention_graph,
)
from transformer_engine.common.cudnn.graph import (
    TensorSpec,
    create_graph,
    attention_matrix_spec,
    row_major_stride as _row_major_stride,
)

from transformer_engine.common.cudnn.score_mod import (
    UncacheableKey,
    callback_cache_key,
    explicit_cache_key,
    freeze_cache_key,
    wrap_score_mod as _wrap_score_mod,
)

from .cudnn_graph import (
    SerializedGraph as _SerializedScoreModGraph,
    encode_frontend_version as _encode_cudnn_frontend_version,
    check_frontend_version as _check_cudnn_frontend_version_match,
    graph_hash as _score_mod_graph_hash,
    pack_scalar_values as _pack_score_mod_scalar_values,
    serialize_graph as _serialized_score_mod_graph,
    finalize_graph as _finalize_score_mod_graph,
    import_cudnn as _import_cudnn_for_score_mod,
)

__all__ = [
    "FusedAttnScoreModHelper",
    "make_fused_attn_score_mod_config",
    "validate_fused_attn_score_mod",
    "fused_attn_score_mod_fwd",
    "fused_attn_score_mod_bwd",
]


def _is_non_deterministic_allowed():
    """Check if non-deterministic kernels are allowed."""
    return bool(int(os.getenv("NVTE_ALLOW_NONDETERMINISTIC_ALGO", "1")))


def _enum_name(value: Any) -> str:
    return getattr(value, "name", str(value))


def validate_fused_attn_score_mod(
    qkv: Tuple[jnp.ndarray, ...],
    bias: Optional[jnp.ndarray],
    sequence_descriptor: Optional[Any],
    seed: Optional[jnp.ndarray],
    attn_bias_type: Any,
    attn_mask_type: Any,
    qkv_layout: Any,
    softmax_type: Any,
    dropout_probability: float,
    max_segments_per_seq: int,
    window_size: Optional[Tuple[int, int]],
    context_parallel_strategy: Any,
    context_parallel_causal_load_balanced: bool,
    context_parallel_axis: str,
    softmax_offset: Optional[jnp.ndarray],
    stripe_size: int | None,
):
    """Validate arguments for the cuDNN frontend score_mod path."""
    header = "score_mod fused_attn"
    if _enum_name(qkv_layout) != "BSHD_BSHD_BSHD":
        raise ValueError(f"{header} currently only supports QKVLayout.BSHD_BSHD_BSHD.")
    if len(qkv) != 3:
        raise ValueError(f"{header} requires separate query, key and value tensors.")
    if any(tensor.ndim != 4 for tensor in qkv):
        raise ValueError(f"{header} requires rank-4 BSHD query/key/value tensors.")
    q, k, v = qkv
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise ValueError(f"{header} requires query, key and value to have the same dtype.")
    if q.dtype not in (jnp.float16, jnp.bfloat16):
        raise ValueError(f"{header} only supports FP16/BF16 query, key and value tensors.")
    if q.shape[0] != k.shape[0] or q.shape[0] != v.shape[0]:
        raise ValueError(f"{header} requires matching batch dimensions.")
    if k.shape[1] != v.shape[1]:
        raise ValueError(f"{header} requires key and value sequence lengths to match.")
    if k.shape[2] != v.shape[2]:
        raise ValueError(f"{header} requires key and value head counts to match.")
    if q.shape[3] != k.shape[3]:
        raise ValueError(f"{header} requires query/key head dimensions to match.")

    if bias is not None or _enum_name(attn_bias_type) != "NO_BIAS":
        raise ValueError(f"{header} is mutually exclusive with attention bias.")
    if sequence_descriptor is not None:
        raise ValueError(f"{header} is mutually exclusive with padding/sequence descriptors.")
    if seed is not None:
        raise ValueError(f"{header} is mutually exclusive with dropout seed.")
    if _enum_name(attn_mask_type) != "NO_MASK":
        raise ValueError(f"{header} is mutually exclusive with attention masks.")
    if _enum_name(softmax_type) != "VANILLA_SOFTMAX" or softmax_offset is not None:
        raise ValueError(f"{header} only supports vanilla softmax without softmax_offset.")
    if dropout_probability != 0.0:
        raise ValueError(f"{header} is mutually exclusive with dropout.")
    if max_segments_per_seq != 1:
        raise ValueError(f"{header} is mutually exclusive with packed/ragged sequence metadata.")
    if window_size not in (None, (-1, -1)):
        raise ValueError(f"{header} is mutually exclusive with sliding-window attention.")
    if _enum_name(context_parallel_strategy) != "DEFAULT":
        raise ValueError(f"{header} is mutually exclusive with context parallelism.")
    if context_parallel_causal_load_balanced or context_parallel_axis:
        raise ValueError(f"{header} is mutually exclusive with context parallelism.")
    if stripe_size is not None:
        raise ValueError(f"{header} is mutually exclusive with striped context parallelism.")


@dataclass(frozen=True)
class _ScoreModScalarSpec:
    """Static pass-by-value scalar used when building a cuDNN frontend graph."""

    name: str
    dtype: str
    value: bytes
    dim: Tuple[int, ...] = (1, 1, 1, 1)
    stride: Tuple[int, ...] = (1, 1, 1, 1)


_UncacheableScoreModKey = UncacheableKey


def _score_mod_key_is_uncacheable(key: Any) -> bool:
    return isinstance(key, _UncacheableScoreModKey)


def _freeze_score_mod_cache_key(value: Any) -> Any:
    """Freeze a structural callback key using this framework's tensor predicate."""
    return freeze_cache_key(value, _is_array_operand)


def _score_mod_explicit_cache_key(callback_owner: Any) -> Optional[Any]:
    """Read a callback owner's optional structural key."""
    return explicit_cache_key(callback_owner, _is_array_operand)


def _score_mod_callback_cache_key(callback: Optional[Callable]) -> Any:
    """Share stable callbacks and distinguish module-level anonymous functions."""
    return callback_cache_key(callback, _is_array_operand, _UncacheableScoreModKey)


@dataclass(frozen=True)
class _FusedAttnScoreModConfig:
    """Static configuration for cuDNN frontend score_mod SDPA graphs."""

    score_mod: Callable
    score_mod_bprop: Optional[Callable]
    score_mod_key: Any
    score_mod_bprop_key: Any
    score_mod_tensor_names: Tuple[str, ...]
    score_mod_bprop_tensor_names: Tuple[str, ...]
    score_mod_scalars: Tuple[_ScoreModScalarSpec, ...]
    score_mod_bprop_scalars: Tuple[_ScoreModScalarSpec, ...]
    scaling_factor: float
    is_training: bool
    deterministic: bool

    def __hash__(self):
        return hash(
            (
                self.score_mod_key,
                self.score_mod_bprop_key,
                self.score_mod_tensor_names,
                self.score_mod_bprop_tensor_names,
                self.score_mod_scalars,
                self.score_mod_bprop_scalars,
                self.scaling_factor,
                self.is_training,
                self.deterministic,
            )
        )

    def __eq__(self, other):
        if not isinstance(other, _FusedAttnScoreModConfig):
            return False
        return (
            self.score_mod_key == other.score_mod_key
            and self.score_mod_bprop_key == other.score_mod_bprop_key
            and self.score_mod_tensor_names == other.score_mod_tensor_names
            and self.score_mod_bprop_tensor_names == other.score_mod_bprop_tensor_names
            and self.score_mod_scalars == other.score_mod_scalars
            and self.score_mod_bprop_scalars == other.score_mod_bprop_scalars
            and self.scaling_factor == other.scaling_factor
            and self.is_training == other.is_training
            and self.deterministic == other.deterministic
        )


# Callback operands use a separate UID range from the shared SDPA builder.
_SCORE_MOD_FWD_TENSOR_UID_BASE = 1000
_SCORE_MOD_BPROP_TENSOR_UID_BASE = 2000
_SCORE_MOD_FWD_SCALAR_UID_BASE = 3000
_SCORE_MOD_BPROP_SCALAR_UID_BASE = 4000

_score_mod_graph_cache: Dict[Tuple[Any, ...], _SerializedScoreModGraph] = {}


def _matrix_spec(cudnn, aval, output=False):
    """Translate JAX's contiguous BSHD storage into a shared logical descriptor."""
    batch, seqlen, heads, dim = aval.shape
    return attention_matrix_spec(
        (batch, heads, seqlen, dim),
        "bshd",
        _cudnn_data_type(cudnn, aval.dtype).name,
        output=output,
    )


def _dtype_name(dtype) -> str:
    return str(jnp.dtype(dtype))


def _is_array_operand(value: Any) -> bool:
    return (
        hasattr(value, "shape")
        and hasattr(value, "dtype")
        and not isinstance(value, (bool, int, float, complex, np.generic))
    )


def _scalar_to_spec(name: str, value: Any) -> _ScoreModScalarSpec:
    if isinstance(value, bool):
        dtype = np.bool_
    elif isinstance(value, int):
        dtype = np.int32
    elif isinstance(value, float):
        dtype = np.float32
    elif isinstance(value, np.generic):
        dtype = value.dtype
    else:
        scalar = np.asarray(value)
        if scalar.shape != ():
            raise ValueError(
                f"score_mod tensor '{name}' is neither a JAX array nor a scalar pass-by-value."
            )
        dtype = scalar.dtype

    scalar = np.full((1, 1, 1, 1), value, dtype=dtype)
    return _ScoreModScalarSpec(name=name, dtype=str(scalar.dtype), value=scalar.tobytes())


def _split_score_mod_tensors(
    tensors: Optional[Mapping[str, Any]], *, argument_name: str
) -> Tuple[Tuple[str, ...], Tuple[jnp.ndarray, ...], Tuple[_ScoreModScalarSpec, ...]]:
    if tensors is None:
        return (), (), ()
    if not isinstance(tensors, Mapping):
        raise TypeError(f"{argument_name} must be a mapping from string names to tensors/scalars.")

    names = []
    operands = []
    scalars = []
    for name, value in tensors.items():
        if not isinstance(name, str):
            raise TypeError(f"{argument_name} keys must be strings, got {type(name).__name__}.")
        if _is_array_operand(value):
            if len(value.shape) == 0:
                raise ValueError(
                    f"{argument_name}['{name}'] is a rank-0 array. Use a Python/NumPy scalar "
                    "for cuDNN pass-by-value scalars, or reshape it to a tensor."
                )
            names.append(name)
            operands.append(jnp.asarray(value))
        else:
            scalars.append(_scalar_to_spec(name, value))
    return tuple(names), tuple(operands), tuple(scalars)


def _make_fused_attn_score_mod_config(
    score_mod: Callable,
    score_mod_bprop: Optional[Callable],
    score_mod_tensors: Optional[Mapping[str, Any]],
    score_mod_bprop_tensors: Optional[Mapping[str, Any]],
    scaling_factor: float,
    is_training: bool,
) -> Tuple[_FusedAttnScoreModConfig, Tuple[jnp.ndarray, ...], Tuple[jnp.ndarray, ...]]:
    """Normalize score_mod operands and create a static graph-build config."""
    if not callable(score_mod):
        raise TypeError("score_mod must be callable.")
    if score_mod_bprop is not None and not callable(score_mod_bprop):
        raise TypeError("score_mod_bprop must be callable when provided.")
    if score_mod_bprop is None and score_mod_bprop_tensors:
        raise ValueError("score_mod_bprop_tensors requires score_mod_bprop to be provided.")

    tensor_names, tensor_operands, scalars = _split_score_mod_tensors(
        score_mod_tensors, argument_name="score_mod_tensors"
    )
    bprop_tensor_names, bprop_tensor_operands, bprop_scalars = _split_score_mod_tensors(
        score_mod_bprop_tensors, argument_name="score_mod_bprop_tensors"
    )
    config = _FusedAttnScoreModConfig(
        score_mod=score_mod,
        score_mod_bprop=score_mod_bprop,
        score_mod_key=_score_mod_callback_cache_key(score_mod),
        score_mod_bprop_key=_score_mod_callback_cache_key(score_mod_bprop),
        score_mod_tensor_names=tensor_names,
        score_mod_bprop_tensor_names=bprop_tensor_names,
        score_mod_scalars=scalars,
        score_mod_bprop_scalars=bprop_scalars,
        scaling_factor=float(scaling_factor),
        is_training=bool(is_training),
        deterministic=not _is_non_deterministic_allowed(),
    )
    return config, tensor_operands, bprop_tensor_operands


def _cudnn_data_type(cudnn, dtype):
    dtype = jnp.dtype(dtype)
    if dtype == jnp.float16:
        return cudnn.data_type.HALF
    if dtype == jnp.bfloat16:
        return cudnn.data_type.BFLOAT16
    if dtype == jnp.float32:
        return cudnn.data_type.FLOAT
    if dtype == jnp.float64:
        return cudnn.data_type.DOUBLE
    if dtype == jnp.int32:
        return cudnn.data_type.INT32
    if dtype == jnp.int64:
        return cudnn.data_type.INT64
    if dtype == jnp.uint8:
        return cudnn.data_type.UINT8
    if dtype == jnp.bool_:
        return cudnn.data_type.BOOLEAN
    raise ValueError(f"Unsupported score_mod tensor dtype: {dtype}.")


def _cudnn_data_type_from_name(cudnn, dtype_name: str):
    if dtype_name == "bfloat16":
        return cudnn.data_type.BFLOAT16
    return _cudnn_data_type(cudnn, np.dtype(dtype_name))


def _graph_tensor_from_aval(cudnn, graph, name: str, aval, uid: int):
    shape = tuple(int(dim) for dim in aval.shape)
    return graph.tensor(
        name=name,
        dim=shape,
        stride=_row_major_stride(shape),
        data_type=_cudnn_data_type(cudnn, aval.dtype),
        uid=uid,
    )


def _score_mod_graph_tensors(
    cudnn,
    graph,
    names: Tuple[str, ...],
    avals: Sequence[Any],
    scalars: Tuple[_ScoreModScalarSpec, ...],
    tensor_uid_base: int,
    scalar_uid_base: int,
):
    graph_tensors = {}
    tensor_uids = []
    for index, (name, aval) in enumerate(zip(names, avals)):
        uid = tensor_uid_base + index
        graph_tensors[name] = _graph_tensor_from_aval(cudnn, graph, name, aval, uid)
        tensor_uids.append(uid)

    scalar_uids = []
    scalar_values = []
    for index, scalar in enumerate(scalars):
        uid = scalar_uid_base + index
        graph_tensors[scalar.name] = graph.tensor(
            name=scalar.name,
            dim=scalar.dim,
            stride=scalar.stride,
            is_pass_by_value=True,
            data_type=_cudnn_data_type_from_name(cudnn, scalar.dtype),
            uid=uid,
        )
        scalar_uids.append(uid)
        scalar_values.append(scalar.value)

    return graph_tensors, tuple(tensor_uids), tuple(scalar_uids), tuple(scalar_values)


def _graph_cache_key(
    direction: str,
    config: _FusedAttnScoreModConfig,
    avals: Sequence[Any],
) -> Optional[Tuple[Any, ...]]:
    if _score_mod_key_is_uncacheable(config.score_mod_key) or _score_mod_key_is_uncacheable(
        config.score_mod_bprop_key
    ):
        return None
    return (
        direction,
        config,
        tuple((tuple(aval.shape), _dtype_name(aval.dtype)) for aval in avals),
    )


def _shape_dtype(value) -> jax.ShapeDtypeStruct:
    return jax.ShapeDtypeStruct(tuple(value.shape), value.dtype)


def _build_score_mod_fwd_graph(q_aval, k_aval, v_aval, score_mod_avals, config):
    cudnn = _import_cudnn_for_score_mod()
    graph = create_graph(cudnn, _cudnn_data_type(cudnn, q_aval.dtype))
    extra, tensor_uids, scalar_uids, scalar_values = _score_mod_graph_tensors(
        cudnn,
        graph,
        config.score_mod_tensor_names,
        score_mod_avals,
        config.score_mod_scalars,
        _SCORE_MOD_FWD_TENSOR_UID_BASE,
        _SCORE_MOD_FWD_SCALAR_UID_BASE,
    )
    output_shape = (*q_aval.shape[:-1], v_aval.shape[-1])
    specs = {
        "q": _matrix_spec(cudnn, q_aval),
        "k": _matrix_spec(cudnn, k_aval),
        "v": _matrix_spec(cudnn, v_aval),
        "o": _matrix_spec(cudnn, jax.ShapeDtypeStruct(output_shape, q_aval.dtype), output=True),
    }
    output_names = ["o"]
    if config.is_training:
        batch, seqlen, heads, _ = q_aval.shape
        shape = (batch, heads, seqlen, 1)
        specs["stats"] = TensorSpec(shape, _row_major_stride(shape), "FLOAT", output=True)
        output_names.append("stats")
    built = build_attention_graph(
        cudnn,
        graph,
        AttentionGraphSpec.from_dict(
            AttentionOptions(attn_scale=config.scaling_factor, generate_stats=config.is_training),
            specs,
        ),
        score_mod=_wrap_score_mod(config.score_mod, extra),
    )
    workspace_size, data, frontend_version = _finalize_score_mod_graph(cudnn, graph)
    return _serialized_score_mod_graph(
        serialized_graph=data,
        cudnn_frontend_version=frontend_version,
        workspace_size=workspace_size,
        input_uids=[built.uids[name] for name in ("q", "k", "v")] + list(tensor_uids),
        output_uids=[built.uids[name] for name in output_names],
        scalar_uids=scalar_uids,
        scalar_values=scalar_values,
    )


def _build_score_mod_bwd_graph(
    q_aval,
    k_aval,
    v_aval,
    output_aval,
    doutput_aval,
    stats_aval,
    score_mod_avals,
    score_mod_bprop_avals,
    config,
):
    cudnn = _import_cudnn_for_score_mod()
    graph = create_graph(cudnn, _cudnn_data_type(cudnn, q_aval.dtype))
    extra, tensor_uids, scalar_uids, scalar_values = _score_mod_graph_tensors(
        cudnn,
        graph,
        config.score_mod_tensor_names,
        score_mod_avals,
        config.score_mod_scalars,
        _SCORE_MOD_FWD_TENSOR_UID_BASE,
        _SCORE_MOD_FWD_SCALAR_UID_BASE,
    )
    bprop_extra, bprop_tensor_uids, bprop_scalar_uids, bprop_scalar_values = (
        _score_mod_graph_tensors(
            cudnn,
            graph,
            config.score_mod_bprop_tensor_names,
            score_mod_bprop_avals,
            config.score_mod_bprop_scalars,
            _SCORE_MOD_BPROP_TENSOR_UID_BASE,
            _SCORE_MOD_BPROP_SCALAR_UID_BASE,
        )
    )
    specs = {
        "q": _matrix_spec(cudnn, q_aval),
        "k": _matrix_spec(cudnn, k_aval),
        "v": _matrix_spec(cudnn, v_aval),
        "o": _matrix_spec(cudnn, output_aval),
        "do": _matrix_spec(cudnn, doutput_aval),
        "stats": TensorSpec(tuple(stats_aval.shape), _row_major_stride(stats_aval.shape), "FLOAT"),
        "dq": _matrix_spec(cudnn, q_aval, output=True),
        "dk": _matrix_spec(cudnn, k_aval, output=True),
        "dv": _matrix_spec(cudnn, v_aval, output=True),
    }
    built = build_attention_graph(
        cudnn,
        graph,
        AttentionGraphSpec.from_dict(
            AttentionOptions(
                backward=True, attn_scale=config.scaling_factor, deterministic=config.deterministic
            ),
            specs,
        ),
        score_mod=_wrap_score_mod(config.score_mod, extra),
        score_mod_bprop=_wrap_score_mod(config.score_mod_bprop, bprop_extra),
    )
    workspace_size, data, frontend_version = _finalize_score_mod_graph(cudnn, graph)
    return _serialized_score_mod_graph(
        serialized_graph=data,
        cudnn_frontend_version=frontend_version,
        workspace_size=workspace_size,
        input_uids=[built.uids[name] for name in ("q", "k", "v", "o", "do", "stats")]
        + list(tensor_uids)
        + list(bprop_tensor_uids),
        output_uids=[built.uids[name] for name in ("dq", "dk", "dv")],
        scalar_uids=[*scalar_uids, *bprop_scalar_uids],
        scalar_values=[*scalar_values, *bprop_scalar_values],
    )


def _fused_attn_score_mod_fwd(
    qkv: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    score_mod_tensors: Tuple[jnp.ndarray, ...],
    config: _FusedAttnScoreModConfig,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Run cuDNN frontend SDPA forward with a score_mod callback."""
    q, k, v = qkv
    q_aval, k_aval, v_aval = map(_shape_dtype, (q, k, v))
    score_mod_avals = tuple(_shape_dtype(arg) for arg in score_mod_tensors)
    key = _graph_cache_key("fwd", config, (q_aval, k_aval, v_aval, *score_mod_avals))
    if key is None:
        graph = _build_score_mod_fwd_graph(q_aval, k_aval, v_aval, score_mod_avals, config)
    else:
        if key not in _score_mod_graph_cache:
            _score_mod_graph_cache[key] = _build_score_mod_fwd_graph(
                q_aval, k_aval, v_aval, score_mod_avals, config
            )
        graph = _score_mod_graph_cache[key]

    batch, q_seqlen, q_heads, _ = q.shape
    _, _, _, v_head_dim = v.shape
    output_shape = jax.ShapeDtypeStruct((batch, q_seqlen, q_heads, v_head_dim), q.dtype)
    stats_shape = (batch, q_heads, q_seqlen, 1) if config.is_training else (0,)
    stats = jax.ShapeDtypeStruct(stats_shape, jnp.float32)
    workspace = jax.ShapeDtypeStruct((graph.workspace_size,), jnp.uint8)
    output, softmax_stats, _ = ffi.ffi_call(
        "te_cudnn_graph_ffi",
        (output_shape, stats, workspace),
    )(
        q,
        k,
        v,
        *score_mod_tensors,
        serialized_graph=graph.serialized_graph,
        graph_hash0=graph.graph_hash[0],
        graph_hash1=graph.graph_hash[1],
        cudnn_frontend_version=graph.cudnn_frontend_version,
        input_uids=graph.input_uids,
        output_uids=graph.output_uids,
        scalar_uids=graph.scalar_uids,
        scalar_sizes=graph.scalar_sizes,
        scalar_values=graph.scalar_values,
    )
    return output, softmax_stats


def _fused_attn_score_mod_bwd(
    qkv: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    output: jnp.ndarray,
    doutput: jnp.ndarray,
    softmax_stats: jnp.ndarray,
    score_mod_tensors: Tuple[jnp.ndarray, ...],
    score_mod_bprop_tensors: Tuple[jnp.ndarray, ...],
    config: _FusedAttnScoreModConfig,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Run cuDNN frontend SDPA backward with score_mod callbacks."""
    if not config.is_training:
        raise RuntimeError("score_mod backward requires fused_attn(..., is_training=True).")

    q, k, v = qkv
    all_inputs = (q, k, v, output, doutput, softmax_stats, *score_mod_tensors)
    all_inputs = (*all_inputs, *score_mod_bprop_tensors)
    avals = tuple(_shape_dtype(arg) for arg in all_inputs)
    key = _graph_cache_key("bwd", config, avals)
    if key is None:
        graph = _build_score_mod_bwd_graph(
            *avals[:6],
            avals[6 : 6 + len(score_mod_tensors)],
            avals[6 + len(score_mod_tensors) :],
            config,
        )
    else:
        if key not in _score_mod_graph_cache:
            _score_mod_graph_cache[key] = _build_score_mod_bwd_graph(
                *avals[:6],
                avals[6 : 6 + len(score_mod_tensors)],
                avals[6 + len(score_mod_tensors) :],
                config,
            )
        graph = _score_mod_graph_cache[key]

    dq = jax.ShapeDtypeStruct(q.shape, q.dtype)
    dk = jax.ShapeDtypeStruct(k.shape, k.dtype)
    dv = jax.ShapeDtypeStruct(v.shape, v.dtype)
    workspace = jax.ShapeDtypeStruct((graph.workspace_size,), jnp.uint8)
    dq, dk, dv, _ = ffi.ffi_call(
        "te_cudnn_graph_ffi",
        (dq, dk, dv, workspace),
    )(
        q,
        k,
        v,
        output,
        doutput,
        softmax_stats,
        *score_mod_tensors,
        *score_mod_bprop_tensors,
        serialized_graph=graph.serialized_graph,
        graph_hash0=graph.graph_hash[0],
        graph_hash1=graph.graph_hash[1],
        cudnn_frontend_version=graph.cudnn_frontend_version,
        input_uids=graph.input_uids,
        output_uids=graph.output_uids,
        scalar_uids=graph.scalar_uids,
        scalar_sizes=graph.scalar_sizes,
        scalar_values=graph.scalar_values,
    )
    return dq, dk, dv


class FusedAttnScoreModHelper:
    """Helper for cuDNN frontend score_mod fused attention graphs."""

    make_config = staticmethod(_make_fused_attn_score_mod_config)
    forward = staticmethod(_fused_attn_score_mod_fwd)
    backward = staticmethod(_fused_attn_score_mod_bwd)


def make_fused_attn_score_mod_config(
    score_mod: Callable,
    score_mod_bprop: Optional[Callable],
    score_mod_tensors: Optional[Mapping[str, Any]],
    score_mod_bprop_tensors: Optional[Mapping[str, Any]],
    scaling_factor: float,
    is_training: bool,
) -> Tuple[_FusedAttnScoreModConfig, Tuple[jnp.ndarray, ...], Tuple[jnp.ndarray, ...]]:
    """Normalize score_mod operands and create a static graph-build config."""
    return FusedAttnScoreModHelper.make_config(
        score_mod,
        score_mod_bprop,
        score_mod_tensors,
        score_mod_bprop_tensors,
        scaling_factor,
        is_training,
    )


def fused_attn_score_mod_fwd(
    qkv: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    score_mod_tensors: Tuple[jnp.ndarray, ...],
    config: _FusedAttnScoreModConfig,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Run cuDNN frontend SDPA forward with a score_mod callback."""
    return FusedAttnScoreModHelper.forward(qkv, score_mod_tensors, config)


def fused_attn_score_mod_bwd(
    qkv: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    output: jnp.ndarray,
    doutput: jnp.ndarray,
    softmax_stats: jnp.ndarray,
    score_mod_tensors: Tuple[jnp.ndarray, ...],
    score_mod_bprop_tensors: Tuple[jnp.ndarray, ...],
    config: _FusedAttnScoreModConfig,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Run cuDNN frontend SDPA backward with score_mod callbacks."""
    return FusedAttnScoreModHelper.backward(
        qkv,
        output,
        doutput,
        softmax_stats,
        score_mod_tensors,
        score_mod_bprop_tensors,
        config,
    )
