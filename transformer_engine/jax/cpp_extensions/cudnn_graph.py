# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Shared cuDNN frontend graph serialization support for JAX custom calls.

cuDNN frontend graphs are constructed and planned in Python while XLA executes a
serialized graph through a small, graph-agnostic FFI runtime.  A binding maps a
cuDNN tensor UID to an XLA operand/result and an optional byte offset.  Offsets
are required for TE's packed QKV layouts, where several logical cuDNN tensors
share one JAX buffer.
"""

from __future__ import annotations

import hashlib
import importlib
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import transformer_engine_jax


@dataclass(frozen=True)
class GraphBinding:
    """Bind a cuDNN tensor UID to a JAX buffer and byte offset."""

    uid: int
    buffer_index: int
    byte_offset: int = 0


@dataclass(frozen=True)
class SerializedGraph:
    """Serialized cuDNN graph and static metadata for the generic FFI executor."""

    serialized_graph: bytes
    graph_hash: tuple[int, int]
    cudnn_frontend_version: int
    workspace_size: int
    input_uids: np.ndarray
    input_buffer_indices: np.ndarray
    input_byte_offsets: np.ndarray
    output_uids: np.ndarray
    output_buffer_indices: np.ndarray
    output_byte_offsets: np.ndarray
    scalar_uids: np.ndarray
    scalar_sizes: np.ndarray
    scalar_values: np.ndarray

    def ffi_attrs(self) -> dict[str, Any]:
        """Return the static attributes consumed by the generic C++ executor."""
        return {
            "serialized_graph": self.serialized_graph,
            "graph_hash0": self.graph_hash[0],
            "graph_hash1": self.graph_hash[1],
            "cudnn_frontend_version": self.cudnn_frontend_version,
            "input_uids": self.input_uids,
            "input_buffer_indices": self.input_buffer_indices,
            "input_byte_offsets": self.input_byte_offsets,
            "output_uids": self.output_uids,
            "output_buffer_indices": self.output_buffer_indices,
            "output_byte_offsets": self.output_byte_offsets,
            "scalar_uids": self.scalar_uids,
            "scalar_sizes": self.scalar_sizes,
            "scalar_values": self.scalar_values,
        }


def row_major_stride(shape: Sequence[int]) -> tuple[int, ...]:
    """Return element strides for a contiguous row-major tensor."""
    stride = []
    running = 1
    for dim in reversed(tuple(shape)):
        stride.append(running)
        running *= int(dim)
    return tuple(reversed(stride))


def bshd_as_bhsd_dim_stride(
    shape: Sequence[int],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Describe a contiguous BSHD buffer as cuDNN's logical BHSD tensor."""
    if len(shape) != 4:
        raise ValueError(f"Expected a rank-4 BSHD tensor, got shape={shape}.")
    batch, seqlen, heads, head_dim = (int(dim) for dim in shape)
    return (
        (batch, heads, seqlen, head_dim),
        (seqlen * heads * head_dim, head_dim, heads * head_dim, 1),
    )


def dtype_name(dtype) -> str:
    """Stable dtype name for graph cache keys."""
    return str(jnp.dtype(dtype))


def cudnn_data_type(cudnn, dtype):
    """Convert a NumPy/JAX dtype to a cuDNN frontend data type."""
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
    raise ValueError(f"Unsupported cuDNN graph tensor dtype: {dtype}.")


def cudnn_data_type_from_name(cudnn, dtype_name_: str):
    """Convert a serialized NumPy dtype name to a cuDNN frontend dtype."""
    if dtype_name_ == "bfloat16":
        return cudnn.data_type.BFLOAT16
    return cudnn_data_type(cudnn, np.dtype(dtype_name_))


def graph_tensor_from_aval(cudnn, graph, name: str, aval, uid: int):
    """Create a contiguous graph tensor from a JAX abstract value."""
    shape = tuple(int(dim) for dim in aval.shape)
    return graph.tensor(
        name=name,
        dim=shape,
        stride=row_major_stride(shape),
        data_type=cudnn_data_type(cudnn, aval.dtype),
        uid=uid,
    )


def encode_cudnn_frontend_version(version: str) -> int:
    """Encode a PEP-440 cuDNN frontend version as MMmmpp."""
    public_version = version.split("+", 1)[0].split("-", 1)[0]
    parts = public_version.split(".")
    if len(parts) < 3:
        raise RuntimeError(
            f"Could not parse cuDNN frontend Python version: {version!r}."
        )
    major, minor, patch = (int(part) for part in parts[:3])
    return major * 10000 + minor * 100 + patch


def check_cudnn_frontend_version_match(cudnn) -> int:
    """Ensure Python and C++ frontend versions use a compatible wire format."""
    python_version_string = getattr(cudnn, "__version__", None)
    if python_version_string is None:
        raise RuntimeError("cuDNN frontend Python package does not expose __version__.")
    python_version = encode_cudnn_frontend_version(python_version_string)
    cpp_version = int(transformer_engine_jax.get_cudnn_frontend_version())
    if python_version != cpp_version:
        raise RuntimeError(
            "cuDNN frontend Python/C++ version mismatch for graph serialization: "
            f"Python cudnn.__version__={python_version_string!r} encodes to {python_version}, "
            f"but Transformer Engine C++ was built with CUDNN_FRONTEND_VERSION={cpp_version}. "
            "Use matching cuDNN frontend Python package and C++ headers."
        )
    return python_version


def import_cudnn():
    """Import and validate the cuDNN frontend Python binding."""
    try:
        cudnn = importlib.import_module("cudnn")
    except ImportError as exc:
        raise ImportError(
            "JAX fused_attn requires the cuDNN frontend Python package (`cudnn`)."
        ) from exc
    check_cudnn_frontend_version_match(cudnn)
    return cudnn


def graph_hash(serialized_graph: bytes) -> tuple[int, int]:
    """Return two signed int64 values used as the C++ graph-cache key."""
    digest = hashlib.sha256(serialized_graph).digest()
    return (
        int.from_bytes(digest[0:8], byteorder="little", signed=True),
        int.from_bytes(digest[8:16], byteorder="little", signed=True),
    )


def pack_scalar_values(scalar_values: Sequence[bytes]) -> tuple[np.ndarray, np.ndarray]:
    """Pack pass-by-value scalars into fixed, aligned 16-byte records."""
    scalar_sizes = np.asarray([len(value) for value in scalar_values], dtype=np.int64)
    packed_values = np.zeros((len(scalar_values), 16), dtype=np.uint8)
    for index, value in enumerate(scalar_values):
        if len(value) > 16:
            raise ValueError("cuDNN pass-by-value scalars must be at most 16 bytes.")
        packed_values[index, : len(value)] = np.frombuffer(value, dtype=np.uint8)
    return scalar_sizes, packed_values.reshape(-1)


def serialized_graph(
    *,
    serialized_graph_data: bytes,
    cudnn_frontend_version: int,
    workspace_size: int,
    input_bindings: Sequence[GraphBinding],
    output_bindings: Sequence[GraphBinding],
    scalar_uids: Sequence[int] = (),
    scalar_values: Sequence[bytes] = (),
) -> SerializedGraph:
    """Construct normalized, NumPy-backed metadata for an FFI graph call."""
    scalar_sizes, packed_scalar_values = pack_scalar_values(scalar_values)

    def binding_array(bindings, field):
        return np.asarray(
            [getattr(binding, field) for binding in bindings], dtype=np.int64
        )

    return SerializedGraph(
        serialized_graph=serialized_graph_data,
        graph_hash=graph_hash(serialized_graph_data),
        cudnn_frontend_version=int(cudnn_frontend_version),
        workspace_size=max(int(workspace_size), 1),
        input_uids=binding_array(input_bindings, "uid"),
        input_buffer_indices=binding_array(input_bindings, "buffer_index"),
        input_byte_offsets=binding_array(input_bindings, "byte_offset"),
        output_uids=binding_array(output_bindings, "uid"),
        output_buffer_indices=binding_array(output_bindings, "buffer_index"),
        output_byte_offsets=binding_array(output_bindings, "byte_offset"),
        scalar_uids=np.asarray(scalar_uids, dtype=np.int64),
        scalar_sizes=scalar_sizes,
        scalar_values=packed_scalar_values,
    )


def finalize_graph(cudnn, graph, *, description: str) -> tuple[int, bytes, int]:
    """Validate, plan and serialize a cuDNN frontend graph."""
    graph.validate()
    graph.build_operation_graph()
    try:
        graph.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
        graph.check_support()
    except cudnn.cudnnGraphNotSupportedError as exc:
        raise RuntimeError(
            f"cuDNN {description} graph is not supported: {exc}"
        ) from exc
    graph.build_plans(cudnn.build_plan_policy.HEURISTICS_CHOICE)
    return (
        max(int(graph.get_workspace_size()), 1),
        bytes(graph.serialize()),
        check_cudnn_frontend_version_match(cudnn),
    )


def shape_dtype(value) -> jax.ShapeDtypeStruct:
    """Return a hashable-enough static shape/dtype descriptor for graph construction."""
    return jax.ShapeDtypeStruct(tuple(value.shape), value.dtype)
