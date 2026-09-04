# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""cuDNN frontend score_mod fused attention helpers."""

import hashlib
import importlib
import inspect
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import ffi

import transformer_engine_jax

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


class _UncacheableScoreModKey:
    """Unique static key for callbacks that must not share compiled score_mod graphs."""

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return self is other


def _score_mod_key_is_uncacheable(key: Any) -> bool:
    return isinstance(key, _UncacheableScoreModKey)


def _freeze_score_mod_cache_key(value: Any) -> Any:
    """Convert a user-provided score_mod graph key into a hashable structure."""
    if _is_array_operand(value):
        raise TypeError(
            "score_mod_graph_cache_key() must not include tensors. Pass runtime tensors "
            "through score_mod_tensors or score_mod_bprop_tensors instead."
        )
    if isinstance(value, Mapping):
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
            "combination of mapping/list/tuple/set values."
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

    Module-level functions are assumed to have stable topology. Stateful bound methods and
    callable instances need an explicit score_mod_graph_cache_key(); otherwise their graphs
    are left uncached to avoid reusing stale graphs after Python object address reuse.
    """
    if callback is None:
        return None
    self_obj = getattr(callback, "__self__", None)
    func_obj = getattr(callback, "__func__", None)
    if self_obj is not None and func_obj is not None:
        explicit_key = _score_mod_explicit_cache_key(self_obj)
        if explicit_key is None:
            return _UncacheableScoreModKey()
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
        return ("function", callback.__module__, callback.__qualname__)

    return _UncacheableScoreModKey()


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


@dataclass(frozen=True)
class _SerializedScoreModGraph:
    """Serialized cuDNN frontend graph and static metadata for C++ execution."""

    serialized_graph: bytes
    graph_hash: Tuple[int, int]
    cudnn_frontend_version: int
    workspace_size: int
    input_uids: np.ndarray
    output_uids: np.ndarray
    scalar_uids: np.ndarray
    scalar_sizes: np.ndarray
    scalar_values: np.ndarray


# cuDNN frontend tensor UIDs are arbitrary, but assigning stable values makes serialized
# graphs deterministic and simplifies debugging across the Python graph builder and C++ executor.
_SCORE_MOD_UID_Q = 1
_SCORE_MOD_UID_K = 2
_SCORE_MOD_UID_V = 3
_SCORE_MOD_UID_O = 4
_SCORE_MOD_UID_STATS = 5
_SCORE_MOD_UID_DO = 6
_SCORE_MOD_UID_DQ = 7
_SCORE_MOD_UID_DK = 8
_SCORE_MOD_UID_DV = 9
_SCORE_MOD_FWD_TENSOR_UID_BASE = 1000
_SCORE_MOD_BPROP_TENSOR_UID_BASE = 2000
_SCORE_MOD_FWD_SCALAR_UID_BASE = 3000
_SCORE_MOD_BPROP_SCALAR_UID_BASE = 4000

_score_mod_graph_cache: Dict[Tuple[Any, ...], _SerializedScoreModGraph] = {}


def _row_major_stride(shape: Sequence[int]) -> Tuple[int, ...]:
    stride = []
    running = 1
    for dim in reversed(tuple(shape)):
        stride.append(running)
        running *= dim
    return tuple(reversed(stride))


def _bshd_as_bhsd_dim_stride(shape: Sequence[int]) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    if len(shape) != 4:
        raise ValueError(f"score_mod requires rank-4 BSHD tensors, got shape={shape}.")
    batch, seqlen, heads, head_dim = tuple(shape)
    return (
        (batch, heads, seqlen, head_dim),
        (seqlen * heads * head_dim, head_dim, heads * head_dim, 1),
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


def _encode_cudnn_frontend_version(version: str) -> int:
    public_version = version.split("+", 1)[0].split("-", 1)[0]
    parts = public_version.split(".")
    if len(parts) < 3:
        raise RuntimeError(f"Could not parse cuDNN frontend Python version: {version!r}.")
    major, minor, patch = (int(part) for part in parts[:3])
    return major * 10000 + minor * 100 + patch


def _check_cudnn_frontend_version_match(cudnn) -> int:
    python_version_string = getattr(cudnn, "__version__", None)
    if python_version_string is None:
        raise RuntimeError("cuDNN frontend Python package does not expose __version__.")
    python_version = _encode_cudnn_frontend_version(python_version_string)
    cpp_version = int(transformer_engine_jax.get_cudnn_frontend_version())
    if python_version != cpp_version:
        raise RuntimeError(
            "cuDNN frontend Python/C++ version mismatch for score_mod graph serialization: "
            f"Python cudnn.__version__={python_version_string!r} encodes to {python_version}, "
            f"but Transformer Engine C++ was built with CUDNN_FRONTEND_VERSION={cpp_version}. "
            "Use matching cuDNN frontend Python package and C++ headers."
        )
    return python_version


def _score_mod_graph_hash(serialized_graph: bytes) -> Tuple[int, int]:
    digest = hashlib.sha256(serialized_graph).digest()
    return (
        int.from_bytes(digest[0:8], byteorder="little", signed=True),
        int.from_bytes(digest[8:16], byteorder="little", signed=True),
    )


def _pack_score_mod_scalar_values(
    scalar_values: Sequence[bytes],
) -> Tuple[np.ndarray, np.ndarray]:
    scalar_sizes = np.asarray([len(value) for value in scalar_values], dtype=np.int64)
    packed_values = np.zeros((len(scalar_values), 16), dtype=np.uint8)
    for index, value in enumerate(scalar_values):
        if len(value) > 16:
            raise ValueError("score_mod pass-by-value scalars must be at most 16 bytes.")
        packed_values[index, : len(value)] = np.frombuffer(value, dtype=np.uint8)
    return scalar_sizes, packed_values.reshape(-1)


def _serialized_score_mod_graph(
    *,
    serialized_graph: bytes,
    cudnn_frontend_version: int,
    workspace_size: int,
    input_uids: Sequence[int],
    output_uids: Sequence[int],
    scalar_uids: Sequence[int],
    scalar_values: Sequence[bytes],
) -> _SerializedScoreModGraph:
    scalar_sizes, packed_scalar_values = _pack_score_mod_scalar_values(scalar_values)
    return _SerializedScoreModGraph(
        serialized_graph=serialized_graph,
        graph_hash=_score_mod_graph_hash(serialized_graph),
        cudnn_frontend_version=int(cudnn_frontend_version),
        workspace_size=int(workspace_size),
        input_uids=np.asarray(input_uids, dtype=np.int64),
        output_uids=np.asarray(output_uids, dtype=np.int64),
        scalar_uids=np.asarray(scalar_uids, dtype=np.int64),
        scalar_sizes=scalar_sizes,
        scalar_values=packed_scalar_values,
    )


def _wrap_score_mod(score_mod: Optional[Callable], graph_tensors: Dict[str, Any]):
    if score_mod is None:
        return None

    def wrapped_score_mod(sdpa_graph, score_tensor):
        return score_mod(sdpa_graph, score_tensor, graph_tensors)

    return wrapped_score_mod


def _finalize_score_mod_graph(cudnn, graph) -> Tuple[int, bytes, int]:
    graph.validate()
    graph.build_operation_graph()
    try:
        graph.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
        graph.check_support()
    except cudnn.cudnnGraphNotSupportedError as exc:
        raise RuntimeError(f"cuDNN score_mod SDPA graph is not supported: {exc}") from exc
    graph.build_plans(cudnn.build_plan_policy.HEURISTICS_CHOICE)
    serialized_graph = bytes(graph.serialize())
    return (
        max(int(graph.get_workspace_size()), 1),
        serialized_graph,
        _check_cudnn_frontend_version_match(cudnn),
    )


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


def _import_cudnn_for_score_mod():
    try:
        cudnn = importlib.import_module("cudnn")
    except ImportError as exc:
        raise ImportError(
            "score_mod fused_attn requires the cuDNN frontend Python package (`cudnn`)."
        ) from exc
    _check_cudnn_frontend_version_match(cudnn)
    return cudnn


def _build_score_mod_fwd_graph(q_aval, k_aval, v_aval, score_mod_avals, config):
    cudnn = _import_cudnn_for_score_mod()

    io_data_type = _cudnn_data_type(cudnn, q_aval.dtype)
    graph = cudnn.pygraph(
        io_data_type=io_data_type,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )

    q_dim, q_stride = _bshd_as_bhsd_dim_stride(q_aval.shape)
    k_dim, k_stride = _bshd_as_bhsd_dim_stride(k_aval.shape)
    v_dim, v_stride = _bshd_as_bhsd_dim_stride(v_aval.shape)
    q = graph.tensor(
        name="q", dim=q_dim, stride=q_stride, data_type=io_data_type, uid=_SCORE_MOD_UID_Q
    )
    k = graph.tensor(
        name="k", dim=k_dim, stride=k_stride, data_type=io_data_type, uid=_SCORE_MOD_UID_K
    )
    v = graph.tensor(
        name="v", dim=v_dim, stride=v_stride, data_type=io_data_type, uid=_SCORE_MOD_UID_V
    )

    score_mod_graph_tensors, tensor_uids, scalar_uids, scalar_values = _score_mod_graph_tensors(
        cudnn,
        graph,
        config.score_mod_tensor_names,
        score_mod_avals,
        config.score_mod_scalars,
        _SCORE_MOD_FWD_TENSOR_UID_BASE,
        _SCORE_MOD_FWD_SCALAR_UID_BASE,
    )

    output, stats = graph.sdpa(
        name="te_score_mod_sdpa",
        q=q,
        k=k,
        v=v,
        generate_stats=config.is_training,
        attn_scale=config.scaling_factor,
        use_causal_mask=False,
        score_mod=_wrap_score_mod(config.score_mod, score_mod_graph_tensors),
    )

    batch, q_seqlen, q_heads, _ = q_aval.shape
    _, _, _, v_head_dim = v_aval.shape
    output_dim, output_stride = _bshd_as_bhsd_dim_stride((batch, q_seqlen, q_heads, v_head_dim))
    output.set_output(True).set_uid(_SCORE_MOD_UID_O).set_dim(output_dim).set_stride(output_stride)
    output.set_data_type(io_data_type)

    output_uids = [_SCORE_MOD_UID_O]
    if config.is_training:
        stats_shape = (batch, q_heads, q_seqlen, 1)
        stats.set_output(True).set_uid(_SCORE_MOD_UID_STATS).set_dim(stats_shape).set_stride(
            _row_major_stride(stats_shape)
        )
        stats.set_data_type(cudnn.data_type.FLOAT)
        output_uids.append(_SCORE_MOD_UID_STATS)

    workspace_size, serialized_graph, frontend_version = _finalize_score_mod_graph(cudnn, graph)
    return _serialized_score_mod_graph(
        serialized_graph=serialized_graph,
        cudnn_frontend_version=frontend_version,
        workspace_size=workspace_size,
        input_uids=[_SCORE_MOD_UID_Q, _SCORE_MOD_UID_K, _SCORE_MOD_UID_V, *tensor_uids],
        output_uids=output_uids,
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

    io_data_type = _cudnn_data_type(cudnn, q_aval.dtype)
    graph = cudnn.pygraph(
        io_data_type=io_data_type,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )

    q_dim, q_stride = _bshd_as_bhsd_dim_stride(q_aval.shape)
    k_dim, k_stride = _bshd_as_bhsd_dim_stride(k_aval.shape)
    v_dim, v_stride = _bshd_as_bhsd_dim_stride(v_aval.shape)
    o_dim, o_stride = _bshd_as_bhsd_dim_stride(output_aval.shape)
    do_dim, do_stride = _bshd_as_bhsd_dim_stride(doutput_aval.shape)
    q = graph.tensor(
        name="q", dim=q_dim, stride=q_stride, data_type=io_data_type, uid=_SCORE_MOD_UID_Q
    )
    k = graph.tensor(
        name="k", dim=k_dim, stride=k_stride, data_type=io_data_type, uid=_SCORE_MOD_UID_K
    )
    v = graph.tensor(
        name="v", dim=v_dim, stride=v_stride, data_type=io_data_type, uid=_SCORE_MOD_UID_V
    )
    output = graph.tensor(
        name="o", dim=o_dim, stride=o_stride, data_type=io_data_type, uid=_SCORE_MOD_UID_O
    )
    doutput = graph.tensor(
        name="dO", dim=do_dim, stride=do_stride, data_type=io_data_type, uid=_SCORE_MOD_UID_DO
    )
    stats = graph.tensor(
        name="stats",
        dim=tuple(int(dim) for dim in stats_aval.shape),
        stride=_row_major_stride(stats_aval.shape),
        data_type=cudnn.data_type.FLOAT,
        uid=_SCORE_MOD_UID_STATS,
    )

    score_mod_graph_tensors, tensor_uids, scalar_uids, scalar_values = _score_mod_graph_tensors(
        cudnn,
        graph,
        config.score_mod_tensor_names,
        score_mod_avals,
        config.score_mod_scalars,
        _SCORE_MOD_FWD_TENSOR_UID_BASE,
        _SCORE_MOD_FWD_SCALAR_UID_BASE,
    )
    (
        score_mod_bprop_graph_tensors,
        bprop_tensor_uids,
        bprop_scalar_uids,
        bprop_scalar_values,
    ) = _score_mod_graph_tensors(
        cudnn,
        graph,
        config.score_mod_bprop_tensor_names,
        score_mod_bprop_avals,
        config.score_mod_bprop_scalars,
        _SCORE_MOD_BPROP_TENSOR_UID_BASE,
        _SCORE_MOD_BPROP_SCALAR_UID_BASE,
    )

    dq, dk, dv = graph.sdpa_backward(
        name="te_score_mod_sdpa_backward",
        q=q,
        k=k,
        v=v,
        o=output,
        dO=doutput,
        stats=stats,
        attn_scale=config.scaling_factor,
        use_causal_mask=False,
        score_mod=_wrap_score_mod(config.score_mod, score_mod_graph_tensors),
        score_mod_bprop=_wrap_score_mod(config.score_mod_bprop, score_mod_bprop_graph_tensors),
        use_deterministic_algorithm=config.deterministic,
    )

    dq.set_output(True).set_uid(_SCORE_MOD_UID_DQ).set_dim(q_dim).set_stride(q_stride)
    dk.set_output(True).set_uid(_SCORE_MOD_UID_DK).set_dim(k_dim).set_stride(k_stride)
    dv.set_output(True).set_uid(_SCORE_MOD_UID_DV).set_dim(v_dim).set_stride(v_stride)

    workspace_size, serialized_graph, frontend_version = _finalize_score_mod_graph(cudnn, graph)
    return _serialized_score_mod_graph(
        serialized_graph=serialized_graph,
        cudnn_frontend_version=frontend_version,
        workspace_size=workspace_size,
        input_uids=[
            _SCORE_MOD_UID_Q,
            _SCORE_MOD_UID_K,
            _SCORE_MOD_UID_V,
            _SCORE_MOD_UID_O,
            _SCORE_MOD_UID_DO,
            _SCORE_MOD_UID_STATS,
            *tensor_uids,
            *bprop_tensor_uids,
        ],
        output_uids=[_SCORE_MOD_UID_DQ, _SCORE_MOD_UID_DK, _SCORE_MOD_UID_DV],
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
        "te_fused_attn_score_mod_forward_ffi",
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
        "te_fused_attn_score_mod_backward_ffi",
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
