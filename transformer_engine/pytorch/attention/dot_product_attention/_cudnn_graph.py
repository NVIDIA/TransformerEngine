# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Shared cuDNN Frontend Python graph runtime for PyTorch attention."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Dict, Hashable, Optional, Tuple

import torch

from transformer_engine.common.attention.cache_debug import (
    build_recorder,
    record_event,
    record_lookup,
)
from transformer_engine.common.cudnn_frontend import (
    build_cudnn_graph,
    make_cudnn_graph,
)
from transformer_engine.common.cudnn_frontend import (
    import_cudnn_frontend as _import_cudnn_frontend,
)

_thread_state = threading.local()


def import_cudnn_frontend():
    """Import the cuDNN Frontend Python package lazily.

    PyTorch FusedAttention is optional at import time, so importing Transformer
    Engine must not eagerly initialize cuDNN or fail on CPU-only processes.
    """

    return _import_cudnn_frontend(
        feature="PyTorch fused attention",
        requirement="nvidia-cudnn-frontend>=1.28.0",
    )


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
    return make_cudnn_graph(
        cudnn,
        io_dtype,
        name=name,
        handle=current_stream_handle(device),
    )


def finalize_graph(graph, *, cache_site: Tuple[str, str]) -> int:
    """Build a cuDNN graph and return its required workspace size."""

    cudnn = import_cudnn_frontend()
    return build_cudnn_graph(
        cudnn,
        graph,
        description="attention",
        debug_callback=build_recorder(*cache_site),
    )


@dataclass
class GraphEntry:
    """Built graph plus named graph tensors."""

    graph: Any
    tensors: Dict[str, Any]
    workspace_size: int
    cache_site: Optional[Tuple[str, str]] = None

    def execute(self, variant_pack: Dict[Any, Any], device: torch.device) -> None:
        """Execute the graph on PyTorch's current stream."""

        if self.cache_site is not None:
            record_event(*self.cache_site, "execute", device=_device_key(device)[1])
        # Workspaces are execution scratch. Keeping them in graph cache entries
        # retains one potentially large allocation for every cached configuration.
        workspace = torch.empty(
            self.workspace_size,
            dtype=torch.uint8,
            device=device,
        )
        self.graph.execute(
            variant_pack,
            workspace,
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

    entry = graph_cache().get(key)
    cache_site = _cache_site(key)
    if cache_site is not None:
        record_lookup(*cache_site, hit=entry is not None, key=key)
    return entry


def put_graph_entry(key: Hashable, entry: GraphEntry) -> GraphEntry:
    """Insert and return a graph cache entry."""

    graph_cache()[key] = entry
    cache_site = _cache_site(key)
    if cache_site is not None:
        entry.cache_site = cache_site
        record_event(*cache_site, "cache_graph")
    return entry


def _cache_site(key: Hashable) -> Optional[Tuple[str, str]]:
    """Extract a diagnostic build site from an attention graph cache key."""

    if not isinstance(key, tuple) or not key or not isinstance(key[0], str):
        return None
    try:
        backend, direction = key[0].split("_", 1)
    except ValueError:
        return None
    if backend not in ("f16", "fp8") or direction not in ("fwd", "bwd"):
        return None
    return backend, direction


def clear_graph_cache() -> None:
    """Clear thread-local handles and graphs. Intended for tests."""

    _thread_state.handles = {}
    _thread_state.graph_cache = {}
