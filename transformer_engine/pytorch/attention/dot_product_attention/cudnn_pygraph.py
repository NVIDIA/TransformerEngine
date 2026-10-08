# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Helpers for driving cuDNN frontend's Python graph API from PyTorch.

Shared by flex_attention.py and frost_attention.py: importing the frontend, holding a handle per
device, describing tensors in cuDNN's BHSD form, and building and planning the SDPA forward and
backward graphs. Anything specific to one backend, such as a mask, a score_mod or which engine
to pin, is passed in by the caller.
"""

from __future__ import annotations

import contextlib
import importlib
import os
from typing import Any, Dict, Optional, Sequence, Tuple

import torch

# The cuDNN FROST SDPA engines, named once so the two opposite instructions about them cannot
# drift: frost pins one of these by name, flex bars both. cuDNN has already renamed this family
# once (collapsing the per-head-dim ``..._d512`` rows), and if these strings stopped matching,
# frost would fail loudly while flex failed silently.
FROST_FWD_PLAN_TOKEN = "sdpa_fwd_prefill_sm100"
FROST_BWD_PLAN_TOKEN = "sdpa_bwd_sm100"
FROST_PLAN_TOKENS = (FROST_FWD_PLAN_TOKEN, FROST_BWD_PLAN_TOKEN)

# Distinguishes "the caller said nothing" from "the caller asked for no exclusions at all".
_BAR_FROST_BY_DEFAULT = object()

_cudnn = None
_frost_engines_enabled = False
_HANDLES: Dict[Tuple[str, torch.device], Any] = {}


def import_cudnn_frontend(enable_frost_engines: bool = False):
    """Import cuDNN Frontend, enabling the FROST engines if this caller needs them.

    The switch ranks FROST ahead of the backend engines process-wide, so it defaults off and
    FROST asks explicitly. A caller needing a particular engine should verify by plan name rather
    than trust the switch, which is what ``finalize_plans(require_plan_token=...)`` does.

    The enabling sits outside the import memo deliberately. Inside it, whichever backend imported
    cuDNN first would decide for the process, and a flex-first import would leave every later
    FROST call with no engine on offer. Enabling late works because cuDNN re-reads the environment
    per graph, inside ``offered_ids()`` on every ``create_execution_plans``.
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

    For a caller that must inspect it without triggering an import, such as a version probe.
    """
    return _cudnn


def frost_engines_enabled() -> bool:
    """Whether this process has switched the FROST engines on."""
    return _frost_engines_enabled


def handle_for(device: torch.device, *, backend_name: str = "cuDNN attention"):
    """A cuDNN handle for ``device``, rebound to PyTorch's current stream on every call.

    Without the rebinding cuDNN runs on its handle's own stream while the tensors and workspace
    sit on PyTorch's, with nothing ordering the two. The p2p context-parallel ring issues
    attention inside ``with torch.cuda.stream(cp_stream)``, and executes one cached plan from
    different streams across ring steps, so this is per call rather than per handle.

    Keyed on ``(backend_name, device)``: a cuDNN handle is not thread-safe and ``set_stream``
    mutates it, so one handle shared by two backends widens an existing race for no benefit.
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

    Takes ``cudnn`` rather than importing it, so a dtype lookup cannot flip the FROST switch.
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

    The tensor is never permuted: cuDNN takes dims and strides, so reordering the descriptors
    says the same thing and costs nothing.
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
    """Normalize a device for a cache key, so ``cuda`` and ``cuda:0`` cannot key two entries."""
    if device.type == "cuda" and device.index is None:
        return ("cuda", torch.cuda.current_device())
    return (device.type, device.index)


def tensor_key(
    tensor: torch.Tensor, tensor_format: str, *, backend_name: str = "cuDNN attention"
) -> Tuple[Any, ...]:
    """A tensor as the graph will see it: BHSD dims, BHSD strides, dtype.

    Strides are keyed because the graph is built for this exact layout, which is what lets bshd
    and sbhd run without a transpose. The format itself is not: two formats giving the same
    description are the same graph.
    """
    dim, stride = bhsd_dim_stride(tensor, tensor_format, backend_name=backend_name)
    return (tuple(dim), tuple(stride), tensor.dtype)


def cached_graph(cache: Dict[Any, Any], key: Optional[Any], build, *, device=None):
    """Memoize a built graph. ``key=None`` means uncacheable: build, return, do not store.

    The build runs under ``device`` rather than merely with its handle, because the plans are
    JIT-compiled and a compile path may read the ambient CUDA context. One build site, not one
    per branch: a second is free to forget that.
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
    exclude_plan_tokens: Any = _BAR_FROST_BY_DEFAULT,
) -> Tuple[int, Optional[str]]:
    """Create plans, optionally pin one by name, build, and return (workspace size, plan name).

    ``require_plan_token`` makes the choice strict: a plan whose name lacks the token raises.
    Unpinned, ``build_plans`` walks the ranked list and finalizes the first that builds, so a
    graph the intended engine declines runs on whatever cuDNN ranked next with nothing in the
    return value saying so. The token is matched as a substring because cuDNN has already
    collapsed per-head-dim engine names into one row once. The pin has to precede
    ``check_support``: selecting a plan sets cuDNN's ``_plan_pinned``, and only then is a decline
    fatal rather than recorded.

    ``exclude_plan_tokens`` is the opposite instruction, defaulting to barring the FROST engines,
    which accept a score_mod graph and then compute without it. The switch offering them is
    process-wide, so declining to ask is not enough. Forgetting to exclude is silently wrong while
    excluding wrongly costs a slower plan or a loud decline, so the default favours the caller
    that does not want them; it is skipped when a plan is pinned, which has already named one.

    Remove the default once cuDNN is fixed: the gate that should decline these graphs reads a key
    ``sdpa()`` never writes, so it never fires. The match is on a substring of the engine name, so
    a rename would break it; what catches that is the end-to-end test comparing flex's output
    across the switch, not anything here. Measured: deselect marks rather than removes, so the
    plan list still names a barred engine afterwards and cannot be asserted on.
    """
    cudnn = _cudnn if _cudnn is not None else import_cudnn_frontend()

    graph.validate()
    graph.build_operation_graph()

    if heuristics is None:
        heuristics = [cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK]

    if require_plan_token is None:
        if exclude_plan_tokens is _BAR_FROST_BY_DEFAULT:
            exclude_plan_tokens = FROST_PLAN_TOKENS
        try:
            graph.create_execution_plans(list(heuristics))
            # Bar the named engines before the walk, so build_plans falls through to the first
            # entry that is both unbarred and buildable. Inert where they are not on offer, which
            # is every process that has not enabled them. getattr so a frontend predating
            # deselect_engines degrades rather than raising.
            deselect = getattr(graph, "deselect_engines", None)
            if exclude_plan_tokens and deselect is not None:
                deselect(list(exclude_plan_tokens))
            graph.check_support()
        except cudnn.cudnnGraphNotSupportedError as exc:
            # Name the bar in the message: if it removed the only viable plan, the graph is not
            # what was unsupported.
            barred = (
                f" (barred engines: {list(exclude_plan_tokens)})" if exclude_plan_tokens else ""
            )
            raise RuntimeError(
                f"cuDNN {backend_name} SDPA graph is not supported: {exc}{barred}"
            ) from exc
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


def _declare_tensor(graph, name: str, spec: Any):
    """Declare one graph input.

    ``spec`` is a ``torch.Tensor``, described with ``tensor_like``, or a ``(dim, stride)`` /
    ``(dim, stride, data_type)`` descriptor. Omitting ``data_type`` lets the tensor inherit the
    graph's ``io_data_type``.
    """
    if isinstance(spec, torch.Tensor):
        return graph.tensor_like(spec)
    kwargs: Dict[str, Any] = {"name": name, "dim": list(spec[0]), "stride": list(spec[1])}
    if len(spec) > 2 and spec[2] is not None:
        kwargs["data_type"] = spec[2]
    return graph.tensor(**kwargs)


def _mark_output(tensor, spec: Any):
    """Mark a graph output and apply whichever of dim/stride/data_type ``spec`` supplies.

    Each field is optional because the backends specify different subsets, and setting one a
    caller left out would be describing the tensor for it rather than from it.
    """
    tensor.set_output(True)
    if spec[0] is not None:
        tensor.set_dim(list(spec[0]))
    if spec[1] is not None:
        tensor.set_stride(list(spec[1]))
    if len(spec) > 2 and spec[2] is not None:
        tensor.set_data_type(spec[2])
    return tensor


def _declare_aux(graph, aux_tensors):
    """Declare the auxiliary runtime tensors an sdpa callback reads, grouped by role."""
    return {
        group: {name: graph.tensor_like(t) for name, t in tensors.items()}
        for group, tensors in (aux_tensors or {}).items()
    }


def _resolve_sdpa_kwargs(sdpa_kwargs, aux):
    """Extra ``sdpa``/``sdpa_backward`` arguments, as a dict or a callable taking ``aux``.

    The callable form exists because a score_mod closes over graph tensors that cannot be built
    until the graph is, so the caller gets them handed back here rather than building its own.
    """
    return sdpa_kwargs(aux) if callable(sdpa_kwargs) else dict(sdpa_kwargs or {})


def build_fwd(
    *,
    dtype: torch.dtype,
    device: torch.device,
    backend_name: str,
    name: str,
    q: Any,
    k: Any,
    v: Any,
    out: Any,
    attn_scale: float,
    stats: Any = None,
    aux_tensors: Optional[Dict[str, Dict[str, torch.Tensor]]] = None,
    sdpa_kwargs: Any = None,
    heuristics: Optional[Sequence[Any]] = None,
    require_plan_token: Optional[str] = None,
    not_found_hint: Any = "",
    exclude_plan_tokens: Any = _BAR_FROST_BY_DEFAULT,
) -> Dict[str, Any]:
    """Build and plan an SDPA forward graph.

    ``stats`` is the LSE output descriptor, or ``None`` to skip generating it. Everything
    backend-specific -- a mask, a score_mod, which engine to pin or bar -- arrives through
    ``sdpa_kwargs`` and the plan arguments, which are passed to :func:`finalize_plans`.
    """
    graph = build_pygraph(dtype, device, backend_name=backend_name)
    tq = _declare_tensor(graph, "q", q)
    tk = _declare_tensor(graph, "k", k)
    tv = _declare_tensor(graph, "v", v)
    aux = _declare_aux(graph, aux_tensors)
    tout, tstats = graph.sdpa(
        name=name,
        q=tq,
        k=tk,
        v=tv,
        generate_stats=stats is not None,
        attn_scale=attn_scale,
        **_resolve_sdpa_kwargs(sdpa_kwargs, aux),
    )
    _mark_output(tout, out)
    if stats is None:
        tstats = None
    else:
        _mark_output(tstats, stats)
    workspace, plan = finalize_plans(
        graph,
        backend_name=backend_name,
        heuristics=heuristics,
        require_plan_token=require_plan_token,
        not_found_hint=not_found_hint,
        exclude_plan_tokens=exclude_plan_tokens,
    )
    return {
        "graph": graph,
        "q": tq,
        "k": tk,
        "v": tv,
        "out": tout,
        "stats": tstats,
        "aux": aux,
        "workspace": workspace,
        "plan": plan,
    }


def build_bwd(
    *,
    dtype: torch.dtype,
    device: torch.device,
    backend_name: str,
    name: str,
    q: Any,
    k: Any,
    v: Any,
    o: Any,
    do: Any,
    stats: Any,
    dq: Any,
    dk: Any,
    dv: Any,
    attn_scale: float,
    deterministic: bool = False,
    aux_tensors: Optional[Dict[str, Dict[str, torch.Tensor]]] = None,
    sdpa_kwargs: Any = None,
    heuristics: Optional[Sequence[Any]] = None,
    require_plan_token: Optional[str] = None,
    not_found_hint: Any = "",
    exclude_plan_tokens: Any = _BAR_FROST_BY_DEFAULT,
) -> Dict[str, Any]:
    """Build and plan an SDPA backward graph. The counterpart of :func:`build_fwd`."""
    graph = build_pygraph(dtype, device, backend_name=backend_name)
    handles = {
        n: _declare_tensor(graph, n, spec)
        for n, spec in (("q", q), ("k", k), ("v", v), ("o", o), ("do", do), ("stats", stats))
    }
    aux = _declare_aux(graph, aux_tensors)
    tdq, tdk, tdv = graph.sdpa_backward(
        name=name,
        q=handles["q"],
        k=handles["k"],
        v=handles["v"],
        o=handles["o"],
        dO=handles["do"],
        stats=handles["stats"],
        attn_scale=attn_scale,
        use_deterministic_algorithm=deterministic,
        **_resolve_sdpa_kwargs(sdpa_kwargs, aux),
    )
    for handle, spec in ((tdq, dq), (tdk, dk), (tdv, dv)):
        _mark_output(handle, spec)
    workspace, plan = finalize_plans(
        graph,
        backend_name=backend_name,
        heuristics=heuristics,
        require_plan_token=require_plan_token,
        not_found_hint=not_found_hint,
        exclude_plan_tokens=exclude_plan_tokens,
    )
    handles.update({"dq": tdq, "dk": tdk, "dv": tdv})
    return {"graph": graph, "aux": aux, "workspace": workspace, "plan": plan, **handles}


def execute_graph(
    graph,
    variant_pack: Dict[Any, torch.Tensor],
    workspace_size: int,
    device: torch.device,
    *,
    backend_name: str = "cuDNN attention",
):
    """Allocate a built graph's workspace and run it on the device's current stream.

    The handle is resolved here rather than by the caller so it is always rebound immediately
    before ``execute``, which is what keeps cuDNN on the same stream as the tensors.
    """
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    workspace = torch.empty(workspace_size, device=device, dtype=torch.uint8)
    graph.execute(variant_pack, workspace, handle=handle_for(device, backend_name=backend_name))
