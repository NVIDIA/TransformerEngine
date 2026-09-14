# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Tests for the PyTorch cuDNN graph runtime."""

import weakref

import torch

from transformer_engine.pytorch.attention.dot_product_attention import (
    _cudnn_graph,
    cudnn_attention,
)


def test_page_table_uses_cudnn_logical_layout():
    """cuDNN paged attention requires a four-dimensional table descriptor."""

    class Graph:
        @staticmethod
        def tensor(**kwargs):
            return kwargs

    page_table = torch.empty_strided((3, 4), (6, 1), dtype=torch.int32)
    graph_tensor = cudnn_attention._make_page_table_graph_tensor(
        Graph(), page_table, batch=2, name="page_table_k"
    )

    assert graph_tensor == {
        "name": "page_table_k",
        "dim": (2, 1, 4, 1),
        "stride": (6, 6, 1, 1),
        "data_type": torch.int32,
    }


def test_graph_entry_does_not_retain_execution_workspaces(monkeypatch):
    """Cached graph entries should not own per-execution scratch allocations."""

    active_devices = []
    entered_devices = []

    class DeviceGuard:
        def __init__(self, device):
            self.device = device

        def __enter__(self):
            active_devices.append(self.device)
            entered_devices.append(self.device)

        def __exit__(self, *_args):
            active_devices.pop()

    class Workspace:
        pass

    class Graph:
        def __init__(self):
            self.workspace_refs = []

        def execute(self, variant_pack, workspace, handle):
            assert variant_pack == {"q": "tensor"}
            assert handle == "handle"
            assert active_devices == [torch.device("cpu")]
            self.workspace_refs.append(weakref.ref(workspace))

    allocated_workspace_refs = []

    def allocate_workspace(size, *, dtype, device):
        assert size == 123
        assert dtype == torch.uint8
        assert device == torch.device("cpu")
        workspace = Workspace()
        allocated_workspace_refs.append(weakref.ref(workspace))
        return workspace

    monkeypatch.setattr(_cudnn_graph.torch, "empty", allocate_workspace)
    monkeypatch.setattr(_cudnn_graph.torch.cuda, "device", DeviceGuard)
    monkeypatch.setattr(_cudnn_graph, "current_stream_handle", lambda _device: "handle")

    graph = Graph()
    entry = _cudnn_graph.GraphEntry(graph=graph, tensors={}, workspace_size=123)
    entry.execute({"q": "tensor"}, torch.device("cpu"))
    entry.execute({"q": "tensor"}, torch.device("cpu"))

    assert entered_devices == [torch.device("cpu"), torch.device("cpu")]
    assert not active_devices
    assert len(allocated_workspace_refs) == 2
    assert all(ref() is None for ref in allocated_workspace_refs)
    assert all(ref() is None for ref in graph.workspace_refs)
