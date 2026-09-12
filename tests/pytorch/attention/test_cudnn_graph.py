# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Tests for the PyTorch cuDNN graph runtime."""

import weakref

import torch

from transformer_engine.pytorch.attention.dot_product_attention import _cudnn_graph


def test_graph_entry_does_not_retain_execution_workspaces(monkeypatch):
    """Cached graph entries should not own per-execution scratch allocations."""

    class Workspace:
        pass

    class Graph:
        def __init__(self):
            self.workspace_refs = []

        def execute(self, variant_pack, workspace, handle):
            assert variant_pack == {"q": "tensor"}
            assert handle == "handle"
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
    monkeypatch.setattr(_cudnn_graph, "current_stream_handle", lambda _device: "handle")

    graph = Graph()
    entry = _cudnn_graph.GraphEntry(graph=graph, tensors={}, workspace_size=123)
    entry.execute({"q": "tensor"}, torch.device("cpu"))
    entry.execute({"q": "tensor"}, torch.device("cpu"))

    assert len(allocated_workspace_refs) == 2
    assert all(ref() is None for ref in allocated_workspace_refs)
    assert all(ref() is None for ref in graph.workspace_refs)
