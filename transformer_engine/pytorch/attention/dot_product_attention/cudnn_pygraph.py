# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Mechanics of driving cuDNN Frontend's Python graph API from PyTorch.

Importing the frontend, holding one stream-current handle per device, describing TE tensors in
cuDNN's logical BHSD form, and creating, selecting and building plans. No attention semantics, and
no knowledge of any backend's cache-key layout.

``flex_attention.py`` and ``frost_attention.py`` both drive cuDNN through this API. They share
this module for ownership rather than for line count: the state below is process-global -- one
``cudnn`` module, one ``CUDNN_FRONTEND_ENABLE_FROST_ENGINES`` switch, one engine ranking, one
handle per device -- and giving it two owners is how this code has produced bugs before.

``backend_name`` is threaded through purely so a failure still says which backend was driving.
"""

from __future__ import annotations

import contextlib
import importlib
import os
from typing import Any, Dict, Optional, Sequence, Tuple

import torch

_cudnn = None
_frost_engines_enabled = False
_HANDLES: Dict[Tuple[str, torch.device], Any] = {}


def import_cudnn_frontend(enable_frost_engines: bool = False):
    """Import cuDNN Frontend, enabling the FROST engines if this caller needs them.

    ``enable_frost_engines`` is not merely additive: the switch also ranks FROST ahead of the
    backend engines everywhere, so a caller that does not want FROST must not ask for it. Hence
    the default is off, and FROST asks explicitly.

    The enabling is deliberately outside the import memo. Both backends call this, and whichever
    one reaches it first would otherwise decide for the process: with the flag inside the memo, a
    flex call would cache the module with FROST off and every later FROST call would get a cuDNN
    that offers no FROST engine, which surfaces much later as "no cuDNN engine matching ... was
    offered". Enabling late is sound because the switch is read per graph rather than at import:
    in cuDNN Frontend 1.29.0 ``engines/manifest.py`` consults the environment inside
    ``offered_ids()``, reached from ``engines_for(graph)`` on every ``create_execution_plans``.

    Note the switch is process-wide and never unset, so enabling it for FROST also reorders the
    candidates a concurrent score_mod graph sees. Callers that require a particular engine should
    verify by plan name rather than rely on the switch, which is what
    ``finalize_plans(require_plan_token=...)`` does.
    """
    global _cudnn, _frost_engines_enabled  # pylint: disable=global-statement
    if _cudnn is None:
        try:
            _cudnn = importlib.import_module("cudnn")
        except ImportError as exc:
            raise ImportError(
                "cuDNN frontend Python package not found. "
                "Install it with: pip install nvidia-cudnn-frontend"
            ) from exc

    if enable_frost_engines and not _frost_engines_enabled:
        os.environ.setdefault("CUDNN_FRONTEND_ENABLE_FROST_ENGINES", "1")
        importlib.import_module("cudnn.sdpa")
        _frost_engines_enabled = True

    return _cudnn


def cudnn_module():
    """The imported frontend, or None if nothing has imported it yet.

    For callers that want to inspect the module without triggering an import, such as a version
    probe that must not enable anything as a side effect.
    """
    return _cudnn


def frost_engines_enabled() -> bool:
    """Whether this process has switched the FROST engines on."""
    return _frost_engines_enabled


def handle_for(device: torch.device, *, backend_name: str = "cuDNN attention"):
    """A cuDNN handle for ``device``, rebound to PyTorch's current stream on every call.

    Without the rebinding, cuDNN runs on its handle's own stream while the tensors and workspace
    are allocated on PyTorch's current stream, and nothing orders the two. That is not
    hypothetical: the p2p context-parallel ring issues attention inside
    ``with torch.cuda.stream(cp_stream)``, so on alternating ring steps the kernel and its buffers
    would otherwise be on different streams. The same cached plan is executed from different
    streams across steps, so this has to happen per call rather than once per handle.

    Keyed on ``(backend_name, device)`` rather than on the device alone. A cuDNN handle is not
    thread-safe and ``set_stream`` mutates it, so one handle shared by two backends widens an
    existing within-backend race into a cross-backend one for no benefit.
    """
    if device.type != "cuda":
        raise ValueError(f"{backend_name} only supports CUDA tensors, got device {device}.")
    cudnn = _cudnn if _cudnn is not None else import_cudnn_frontend()
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    key = (backend_name, device)
    with torch.cuda.device(device):
        handle = _HANDLES.get(key)
        if handle is None:
            handle = cudnn.create_handle()
            _HANDLES[key] = handle
        cudnn.set_stream(handle=handle, stream=torch.cuda.current_stream(device).cuda_stream)
    return handle


def io_data_type(cudnn, dtype: torch.dtype, *, backend_name: str = "cuDNN attention"):
    """Map a torch dtype to the cuDNN enum these SDPA graphs are declared with.

    Takes ``cudnn`` rather than importing it, so a dtype lookup cannot import the frontend or
    flip the FROST switch as a side effect.
    """
    if dtype == torch.float16:
        return cudnn.data_type.HALF
    if dtype == torch.bfloat16:
        return cudnn.data_type.BFLOAT16
    raise ValueError(f"{backend_name} only supports FP16/BF16 tensors, got {dtype}.")


def build_pygraph(
    dtype: torch.dtype, device: torch.device, *, backend_name: str = "cuDNN attention"
):
    """A cuDNN frontend graph for F16/BF16 SDPA, bound to this device's stream-current handle."""
    cudnn = _cudnn if _cudnn is not None else import_cudnn_frontend()
    return cudnn.pygraph(
        io_data_type=io_data_type(cudnn, dtype, backend_name=backend_name),
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
        handle=handle_for(device, backend_name=backend_name),
    )


def bhsd_dim_stride(
    tensor: torch.Tensor, tensor_format: str, *, backend_name: str = "cuDNN attention"
) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """Describe an SBHD/BSHD tensor as cuDNN frontend's logical BHSD format.

    The tensor is never permuted. cuDNN takes dims and strides, so reordering the descriptors
    says the same thing as permuting the tensor and costs nothing.
    """
    if tensor_format == "sbhd":
        return (
            (tensor.shape[1], tensor.shape[2], tensor.shape[0], tensor.shape[3]),
            (tensor.stride(1), tensor.stride(2), tensor.stride(0), tensor.stride(3)),
        )
    if tensor_format == "bshd":
        return (
            (tensor.shape[0], tensor.shape[2], tensor.shape[1], tensor.shape[3]),
            (tensor.stride(0), tensor.stride(2), tensor.stride(1), tensor.stride(3)),
        )
    raise ValueError(f"{backend_name} only supports SBHD/BSHD tensor formats, got {tensor_format}.")


def bhsd_graph_tensor(
    graph, tensor: torch.Tensor, tensor_format: str, *, backend_name: str = "cuDNN attention"
):
    """Create a cuDNN graph tensor with BHSD dims and TE-layout strides."""
    dim, stride = bhsd_dim_stride(tensor, tensor_format, backend_name=backend_name)
    return graph.tensor(dim=dim, stride=stride, data_type=tensor.dtype)


def device_key(device: torch.device) -> Tuple[Any, ...]:
    """Normalize a device for a cache key.

    ``index is None`` is resolved to the current device, so ``cuda`` and ``cuda:0`` cannot key
    two entries for one physical device. The type is part of the key too, so CPU cannot alias it.
    """
    if device.type == "cuda" and device.index is None:
        return ("cuda", torch.cuda.current_device())
    return (device.type, device.index)


def tensor_key(
    tensor: torch.Tensor, tensor_format: str, *, backend_name: str = "cuDNN attention"
) -> Tuple[Any, ...]:
    """A tensor as the graph will see it: BHSD dims, BHSD strides, dtype.

    Strides belong in the key because the graph is built for this exact layout -- that is what
    lets bshd and sbhd run without a transpose -- and the dtype because every node is declared
    with one. Two formats that produce the same description are the same graph and should share
    a plan, which is why the format itself is not keyed.
    """
    dim, stride = bhsd_dim_stride(tensor, tensor_format, backend_name=backend_name)
    return (tuple(dim), tuple(stride), tensor.dtype)


def cached_graph(cache: Dict[Any, Any], key: Optional[Any], build, *, device=None):
    """Memoize a built graph. ``key=None`` means uncacheable: build and return without storing.

    The build runs under ``device`` when one is given, not merely with its handle: the plans are
    JIT-compiled, and a compile path may read the ambient CUDA context rather than the handle.
    That applies to the uncacheable path too, which is why there is one build site rather than
    one per branch -- a second would be free to forget the scope.
    """
    entry = None if key is None else cache.get(key)
    if entry is None:
        scope = (
            torch.cuda.device(device)
            if device is not None and device.type == "cuda"
            else contextlib.nullcontext()
        )
        with scope:
            entry = build()
        if key is not None:
            cache[key] = entry
    return entry


def finalize_plans(
    graph,
    *,
    backend_name: str = "cuDNN attention",
    heuristics: Optional[Sequence[Any]] = None,
    build_policy: Any = None,
    require_plan_token: Optional[str] = None,
    not_found_hint: Any = "",
) -> Tuple[int, Optional[str]]:
    """Create plans, optionally pin one by name, build, and return (workspace size, plan name).

    ``require_plan_token`` makes the choice strict: only a plan whose name contains the token is
    acceptable, and anything else raises. That is not a stylistic preference. Without a pin,
    ``build_plans`` walks the ranked list from index 0 and finalizes the first plan that builds,
    logging each decline at INFO, so a graph that the intended engine declines runs on whatever
    cuDNN ranked next with nothing in the return value to say so. At head_dim 512 that matters in
    the forward, where an ordinary engine may well build and compute a different function from the
    FROST kernel. The backward is self-limiting, since no non-FROST d512 backward exists, so an
    unpinned backward would fail loudly on its own.

    The token is matched as a substring rather than by equality on purpose: cuDNN has already
    collapsed per-head-dim engine names (``..._d512`` and friends) into a single row once, and the
    substring test survived that.

    Pinning also changes what ``check_support`` means. Selecting a plan sets cuDNN's internal
    ``_plan_pinned``, and only then is a decline fatal; unpinned, cuDNN records the decline and
    keeps walking. So the pin has to come first both because the check is scoped to the selected
    plan and because it is what makes the check binding at all.
    """
    cudnn = _cudnn if _cudnn is not None else import_cudnn_frontend()

    graph.validate()
    graph.build_operation_graph()

    if heuristics is None:
        heuristics = [cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK]

    if require_plan_token is None:
        try:
            graph.create_execution_plans(list(heuristics))
            graph.check_support()
        except cudnn.cudnnGraphNotSupportedError as exc:
            raise RuntimeError(f"cuDNN {backend_name} SDPA graph is not supported: {exc}") from exc
        if build_policy is None:
            build_policy = cudnn.build_plan_policy.HEURISTICS_CHOICE
        graph.build_plans(build_policy)
        return max(graph.get_workspace_size(), 1), None

    graph.create_execution_plans(list(heuristics))
    names = [graph.get_plan_name_at_index(i) for i in range(graph.get_execution_plan_count())]
    hits = [i for i, n in enumerate(names) if require_plan_token in n]
    if not hits:
        # Callable hints are resolved only here: a caller may want to look up package versions to
        # explain the failure, and that work should not happen on the success path.
        hint = not_found_hint() if callable(not_found_hint) else not_found_hint
        raise RuntimeError(
            f"no cuDNN engine matching {require_plan_token!r} was offered."
            f" Candidate plans: {names[:6]}.{(' ' + hint) if hint else ''}"
        )
    graph.select_plan(hits[0])
    # The engine is pinned, so a decline here is its own verdict and cuDNN puts the reason in the
    # exception. Surface it: a plan offered and then refused is the harder failure to read.
    try:
        graph.check_support()
        graph.build_plans()
    except cudnn.cudnnGraphNotSupportedError as exc:
        hint = not_found_hint() if callable(not_found_hint) else not_found_hint
        raise RuntimeError(
            f"cuDNN engine {names[hits[0]]!r} was offered but declined this graph:"
            f" {exc}{(' ' + hint) if hint else ''}"
        ) from exc
    return max(graph.get_workspace_size(), 1), names[hits[0]]
