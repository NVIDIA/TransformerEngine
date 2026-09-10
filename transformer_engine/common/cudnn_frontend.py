# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Framework-independent helpers for constructing cuDNN frontend graphs."""

from __future__ import annotations

import importlib
from typing import Any


def import_cudnn_frontend(*, feature: str, requirement: str):
    """Import the cuDNN frontend Python package lazily."""

    try:
        return importlib.import_module("cudnn")
    except ImportError as exc:
        raise ImportError(
            f"{feature} requires the cuDNN frontend Python package. Install {requirement}."
        ) from exc


def make_cudnn_graph(
    cudnn,
    io_dtype: Any,
    *,
    name: str | None = None,
    handle: Any = None,
):
    """Create a cuDNN graph with TE's standard compute and intermediate types."""

    kwargs = {
        "io_data_type": io_dtype,
        "intermediate_data_type": cudnn.data_type.FLOAT,
        "compute_data_type": cudnn.data_type.FLOAT,
    }
    if name is not None:
        kwargs["name"] = name
    if handle is not None:
        kwargs["handle"] = handle
    return cudnn.pygraph(**kwargs)


def build_cudnn_graph(cudnn, graph, *, description: str) -> int:
    """Validate and plan a graph, returning a nonzero workspace size."""

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
    return max(int(graph.get_workspace_size()), 1)
