# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Shared cuDNN Frontend Python graph runtime for PyTorch attention."""

from __future__ import annotations

from dataclasses import dataclass, field
import importlib
import threading
from typing import Any, Dict, Hashable, Optional, Tuple

import torch

_thread_state = threading.local()


def import_cudnn_frontend():
    """Import the cuDNN Frontend Python package lazily.

    PyTorch FusedAttention is optional at import time, so importing Transformer
    Engine must not eagerly initialize cuDNN or fail on CPU-only processes.
    """

    try:
        return importlib.import_module("cudnn")
    except ImportError as exc:
        raise ImportError(
            "cuDNN Frontend Python package not found. Install "
            "nvidia-cudnn-frontend>=1.27.0."
        ) from exc


def _device_key(device: torch.device) -> Tuple[str, Optional[int]]:
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        return (device.type, torch.cuda.current_device())
    return (device.type, device.index)


def current_stream_handle(device: torch.device):
    """Return this thread's cuDNN handle, bound to PyTorch's current stream."""

    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError(f"cuDNN attention only supports CUDA tensors, got {device}.")
    device = torch.device("cuda", _device_key(device)[1])
    cudnn = import_cudnn_frontend()

    handles = getattr(_thread_state, "handles", None)
    if handles is None:
        handles = {}
        _thread_state.handles = handles
    handle = handles.get(device)
    with torch.cuda.device(device):
        if handle is None:
            handle = cudnn.create_handle()
            handles[device] = handle
        cudnn.set_stream(
            handle=handle,
            stream=torch.cuda.current_stream(device).cuda_stream,
        )
    return handle


def torch_to_cudnn_dtype(dtype: torch.dtype):
    """Map a PyTorch scalar dtype to a cuDNN Frontend dtype."""

    cudnn = import_cudnn_frontend()
    mapping = {
        torch.float16: cudnn.data_type.HALF,
        torch.bfloat16: cudnn.data_type.BFLOAT16,
        torch.float32: cudnn.data_type.FLOAT,
        torch.int32: cudnn.data_type.INT32,
        torch.int64: cudnn.data_type.INT64,
        torch.uint8: cudnn.data_type.UINT8,
        torch.float8_e4m3fn: cudnn.data_type.FP8_E4M3,
        torch.float8_e5m2: cudnn.data_type.FP8_E5M2,
    }
    try:
        return mapping[dtype]
    except KeyError as exc:
        raise ValueError(f"Unsupported cuDNN graph tensor dtype {dtype}.") from exc


def make_graph(io_dtype: Any, device: torch.device, *, name: str):
    """Create an SDPA graph using FP32 intermediate and compute types."""

    cudnn = import_cudnn_frontend()
    return cudnn.pygraph(
        name=name,
        io_data_type=io_dtype,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
        handle=current_stream_handle(device),
    )


def finalize_graph(graph) -> int:
    """Build a cuDNN graph and return its required workspace size."""

    cudnn = import_cudnn_frontend()
    graph.validate()
    graph.build_operation_graph()
    try:
        graph.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
        graph.check_support()
    except cudnn.cudnnGraphNotSupportedError as exc:
        raise RuntimeError(f"cuDNN attention graph is not supported: {exc}") from exc
    graph.build_plans(cudnn.build_plan_policy.HEURISTICS_CHOICE)
    return max(int(graph.get_workspace_size()), 1)


@dataclass
class GraphEntry:
    """Built graph plus named graph tensors and stream-local workspaces."""

    graph: Any
    tensors: Dict[str, Any]
    workspace_size: int
    _workspaces: Dict[int, torch.Tensor] = field(default_factory=dict, repr=False)

    def workspace(self, device: torch.device) -> torch.Tensor:
        """Get a stable workspace for the current stream.

        A workspace may be reused by asynchronous launches on one stream, but
        must not be shared by independent streams. CUDA graph capture also
        keeps the allocation alive for CUDA graph replay. PyTorch's caching
        allocator is capture-aware, so a stream-specific workspace may be
        created on the first captured invocation after the graph itself has
        been warmed and cached.
        """

        device = torch.device(device)
        stream = torch.cuda.current_stream(device)
        stream_key = int(stream.cuda_stream)
        workspace = self._workspaces.get(stream_key)
        if workspace is None:
            workspace = torch.empty(
                self.workspace_size,
                dtype=torch.uint8,
                device=device,
            )
            self._workspaces[stream_key] = workspace
        return workspace

    def execute(self, variant_pack: Dict[Any, Any], device: torch.device) -> None:
        """Execute the graph on PyTorch's current stream."""

        self.graph.execute(
            variant_pack,
            self.workspace(device),
            handle=current_stream_handle(device),
        )


def graph_cache() -> Dict[Hashable, GraphEntry]:
    """Return a thread-local graph cache."""

    cache = getattr(_thread_state, "graph_cache", None)
    if cache is None:
        cache = {}
        _thread_state.graph_cache = cache
    return cache


def get_graph_entry(key: Hashable) -> Optional[GraphEntry]:
    """Look up a graph in this thread's cache."""

    return graph_cache().get(key)


def put_graph_entry(key: Hashable, entry: GraphEntry) -> GraphEntry:
    """Insert and return a graph cache entry."""

    graph_cache()[key] = entry
    return entry


def clear_graph_cache() -> None:
    """Clear thread-local handles and graphs. Intended for tests."""

    _thread_state.handles = {}
    _thread_state.graph_cache = {}
