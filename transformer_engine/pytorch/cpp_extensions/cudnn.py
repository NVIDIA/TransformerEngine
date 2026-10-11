# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""cuDNN Python frontend graph lifecycle shared by attention implementations."""

import importlib
import threading

import torch

_handles = threading.local()


def _import_cudnn_frontend():
    """Import the cuDNN frontend Python package."""
    try:
        return importlib.import_module("cudnn")
    except ImportError as exc:
        raise ImportError(
            "cuDNN frontend Python package not found. "
            "Install it with: pip install nvidia-cudnn-frontend"
        ) from exc


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
    return cudnn.pygraph(
        io_data_type=dtype,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
        handle=_get_cudnn_current_stream_handle(cudnn, device),
    )


def _finalize_cudnn_graph(graph):
    """Check support, build an execution plan, and return workspace bytes."""
    cudnn = _import_cudnn_frontend()
    graph.validate()
    graph.build_operation_graph()
    graph.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    graph.check_support()
    graph.build_plans(cudnn.build_plan_policy.HEURISTICS_CHOICE)
    return max(graph.get_workspace_size(), 1)


def _execute_cudnn_graph(graph, variant_pack, workspace_size, device):
    """Execute on the current stream using framework-owned workspace."""
    cudnn = _import_cudnn_frontend()
    workspace = torch.empty(workspace_size, device=device, dtype=torch.uint8)
    graph.execute(
        variant_pack,
        workspace,
        handle=_get_cudnn_current_stream_handle(cudnn, device),
    )
