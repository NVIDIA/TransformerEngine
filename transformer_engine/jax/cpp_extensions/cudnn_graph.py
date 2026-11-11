# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Serialize Python-built cuDNN graphs for the JAX-owned FFI executor."""

import hashlib
from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np
import transformer_engine_jax

from transformer_engine.common.cudnn.graph import (
    import_cudnn as _import_cudnn,
    check_graph_support as _check_graph_support,
    compile_graph as _compile_graph,
)


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
    cudnn = _import_cudnn()
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


def _serializable_plan_indices(graph, *, has_score_mod=False):
    """Return plan indices eligible for the JAX C++ serialization path."""
    # cuDNN cannot currently serialize FROST SDPA plans for C++ execution.
    # Remove this workaround once cuDNN fixes its serialization path.
    plan_names = [
        graph.get_plan_name_at_index(index) for index in range(graph.get_execution_plan_count())
    ]
    prefixes = ("sdpa_fwd", "sdpa_bwd") if has_score_mod else ("sdpa_fwd_", "sdpa_bwd_")
    return tuple(index for index, name in enumerate(plan_names) if not name.startswith(prefixes))


def check_graph_support(cudnn, graph, run_stage=None, *, has_score_mod=False):
    """Require support from a plan that the JAX executor can deserialize."""

    def filter_plans(candidate):
        candidate.deselect_engines(["sdpa_fwd_", "sdpa_bwd_"])
        indices = _serializable_plan_indices(candidate, has_score_mod=has_score_mod)
        if not indices:
            raise cudnn.cudnnGraphNotSupportedError("No serializable cuDNN SDPA execution plan.")
        # Frontend 1.29/1.31 check_support can accept an excluded Python plan,
        # or suppress a backend rejection if Python plans remain. Pin an eligible
        # plan so support comes from the same executor used at runtime.
        candidate.select_plan(indices[0])

    _check_graph_support(
        cudnn, graph, run_stage, has_score_mod=has_score_mod, filter_plans=filter_plans
    )


def compile_graph(cudnn, graph, run_stage=None, *, has_score_mod=False):
    """Compile an eligible plan, retaining fallback after support-query pinning."""
    last_error = None
    for index in _serializable_plan_indices(graph, has_score_mod=has_score_mod):
        graph.select_plan(index)
        try:
            return _compile_graph(cudnn, graph, run_stage, has_score_mod=has_score_mod)
        except cudnn.cudnnGraphNotSupportedError as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    raise cudnn.cudnnGraphNotSupportedError("No serializable cuDNN SDPA execution plan.")


def finalize_graph(cudnn, graph, *, has_score_mod=False):
    """Validate, select and compile a supported execution plan."""
    check_graph_support(cudnn, graph, has_score_mod=has_score_mod)
    workspace_size = compile_graph(cudnn, graph, has_score_mod=has_score_mod)
    return workspace_size, bytes(graph.serialize()), check_frontend_version(cudnn)


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
