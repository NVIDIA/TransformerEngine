# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Serialize Python-built cuDNN graphs for the JAX-owned FFI executor."""

import hashlib
import importlib
from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np
import transformer_engine_jax


@dataclass(frozen=True)
class SerializedGraph:
    """A compiled graph and its buffer bindings."""

    serialized_graph: bytes
    graph_hash: Tuple[int, int]
    cudnn_frontend_version: int
    workspace_size: int
    input_uids: np.ndarray
    output_uids: np.ndarray
    scalar_uids: np.ndarray
    scalar_sizes: np.ndarray
    scalar_values: np.ndarray


def encode_frontend_version(version: str) -> int:
    """Encode the public version as CUDNN_FRONTEND_VERSION."""
    parts = version.split("+", 1)[0].split("-", 1)[0].split(".")
    if len(parts) < 3:
        raise RuntimeError(f"Could not parse cuDNN frontend Python version: {version!r}.")
    major, minor, patch = (int(part) for part in parts[:3])
    return major * 10000 + minor * 100 + patch


def check_frontend_version(cudnn) -> int:
    """Require compatible Python and C++ serialization formats."""
    version = getattr(cudnn, "__version__", None)
    if version is None:
        raise RuntimeError("cuDNN frontend Python package does not expose __version__.")
    python_version = encode_frontend_version(version)
    cpp_version = int(transformer_engine_jax.get_cudnn_frontend_version())
    if python_version != cpp_version:
        raise RuntimeError(
            "cuDNN frontend Python/C++ version mismatch for graph serialization: "
            f"Python cudnn.__version__={version!r} encodes to {python_version}, "
            f"but Transformer Engine C++ was built with CUDNN_FRONTEND_VERSION={cpp_version}. "
            "Use matching cuDNN frontend Python package and C++ headers."
        )
    return python_version


def import_cudnn():
    """Load the required Python frontend lazily."""
    try:
        cudnn = importlib.import_module("cudnn")
    except ImportError as exc:
        raise ImportError(
            "JAX fused_attn requires the cuDNN frontend Python package (`cudnn`)."
        ) from exc
    check_frontend_version(cudnn)
    return cudnn


def graph_hash(serialized_graph: bytes) -> Tuple[int, int]:
    """Identify a serialized plan independently of Python object lifetimes."""
    digest = hashlib.sha256(serialized_graph).digest()
    return tuple(int.from_bytes(digest[i : i + 8], "little", signed=True) for i in (0, 8))


def pack_scalar_values(values: Sequence[bytes]) -> Tuple[np.ndarray, np.ndarray]:
    """Pack pass-by-value constants into aligned slots."""
    sizes = np.asarray([len(value) for value in values], dtype=np.int64)
    packed = np.zeros((len(values), 16), dtype=np.uint8)
    for index, value in enumerate(values):
        if len(value) > 16:
            raise ValueError("cuDNN pass-by-value scalars must be at most 16 bytes.")
        packed[index, : len(value)] = np.frombuffer(value, dtype=np.uint8)
    return sizes, packed.reshape(-1)


def serialize_graph(
    *,
    serialized_graph,
    cudnn_frontend_version,
    workspace_size,
    input_uids,
    output_uids,
    scalar_uids=(),
    scalar_values=(),
) -> SerializedGraph:
    """Package a graph and its stable tensor UIDs."""
    sizes, values = pack_scalar_values(scalar_values)
    return SerializedGraph(
        serialized_graph,
        graph_hash(serialized_graph),
        int(cudnn_frontend_version),
        int(workspace_size),
        np.asarray(input_uids, dtype=np.int64),
        np.asarray(output_uids, dtype=np.int64),
        np.asarray(scalar_uids, dtype=np.int64),
        sizes,
        values,
    )


def finalize_graph(cudnn, graph):
    """Validate, select and compile a supported execution plan."""
    try:
        graph.validate()
        graph.build_operation_graph()
        graph.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
        graph.check_support()
        graph.build_plans(cudnn.build_plan_policy.HEURISTICS_CHOICE)
    except cudnn.cudnnGraphNotSupportedError as exc:
        raise RuntimeError(f"cuDNN SDPA graph is not supported: {exc}") from exc
    return (
        max(int(graph.get_workspace_size()), 1),
        bytes(graph.serialize()),
        check_frontend_version(cudnn),
    )


def graph_attributes(graph):
    """Attributes shared by ordinary and score-modified attention."""
    return {
        "serialized_graph": graph.serialized_graph,
        "graph_hash0": graph.graph_hash[0],
        "graph_hash1": graph.graph_hash[1],
        "cudnn_frontend_version": graph.cudnn_frontend_version,
        "input_uids": graph.input_uids,
        "output_uids": graph.output_uids,
        "scalar_uids": graph.scalar_uids,
        "scalar_sizes": graph.scalar_sizes,
        "scalar_values": graph.scalar_values,
    }
