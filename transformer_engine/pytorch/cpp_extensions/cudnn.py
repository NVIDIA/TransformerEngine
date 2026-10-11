# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""cuDNN Python frontend graph lifecycle shared by attention implementations."""

import threading

import torch

from transformer_engine.common.cudnn.graph import (
    import_cudnn as _import_cudnn_frontend,
    create_graph,
    check_graph_support,
    compile_graph,
)

_handles = threading.local()


def _get_cudnn_current_stream_handle(cudnn, device):
    """Get a thread-local, device-specific handle on PyTorch's current stream."""
    if device.type != "cuda":
        raise ValueError(f"cuDNN attention requires CUDA tensors, got {device}.")
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    if not hasattr(_handles, "by_device"):
        _handles.by_device = {}
    with torch.cuda.device(device):
        handle = _handles.by_device.get(device)
        if handle is None:
            handle = cudnn.create_handle()
            _handles.by_device[device] = handle
        cudnn.set_stream(handle=handle, stream=torch.cuda.current_stream(device).cuda_stream)
    return handle


def _build_cudnn_pygraph(dtype, device):
    """Create an attention graph with FP32 accumulation."""
    cudnn = _import_cudnn_frontend()
    dtype = {torch.float16: cudnn.data_type.HALF, torch.bfloat16: cudnn.data_type.BFLOAT16}.get(
        dtype, dtype
    )
    return create_graph(cudnn, dtype, _get_cudnn_current_stream_handle(cudnn, device))


def _finalize_cudnn_graph(graph):
    """Check support, build an execution plan, and return workspace bytes."""
    cudnn = _import_cudnn_frontend()
    check_graph_support(cudnn, graph)
    return compile_graph(cudnn, graph)


def _execute_cudnn_graph(graph, variant_pack, workspace_size, device):
    """Execute on the current stream using framework-owned workspace."""
    cudnn = _import_cudnn_frontend()
    workspace = torch.empty(workspace_size, device=device, dtype=torch.uint8)
    graph.execute(
        variant_pack,
        workspace,
        handle=_get_cudnn_current_stream_handle(cudnn, device),
    )
