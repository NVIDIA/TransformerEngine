# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Shared cuDNN Frontend Python-graph plumbing.

Two attention backends build cuDNN graphs from Python: flex_attention.py, for score_mod, and
frost_attention.py, for the CuTe-DSL SDPA kernels at head_dim in (256, 512]. They differ in the
SDPA node they build, which cannot be shared because cuDNN treats a score_mod and a diagonal band
as mutually exclusive, but everything around that node is the same work: importing the frontend,
holding one handle per device on PyTorch's current stream, describing an SBHD/BSHD tensor in the
BHSD form cuDNN wants, finalizing plans, and executing.

This module is that common part: the plumbing, plus the one piece of shared attention
vocabulary, translating a TE mask type and window into cuDNN's diagonal band.
"""

from typing import Any, Dict, Optional, Sequence, Tuple

import os

import torch


_cudnn = None
_handles: Dict[torch.device, Any] = {}


def import_cudnn_frontend(enable_frost_engines: bool = False):
    """Import cuDNN Frontend once, optionally with the FROST engines registered.

    ``enable_frost_engines`` is not merely additive: the switch also ranks FROST ahead of the
    backend engines everywhere, so a caller that does not want FROST must not ask for it.

    The switch is set before the import because the engines are documented as registering at
    import time. Measured on B200 with cuDNN Frontend 1.29.0 the ordering turns out not to
    matter, but setting it first is what the documentation asks for and costs nothing. Callers
    that require a FROST engine should verify by plan name rather than rely on the switch, which
    is what ``finalize_plans(require_plan_token=...)`` does.
    """
    global _cudnn  # pylint: disable=global-statement
    if _cudnn is None:
        if enable_frost_engines:
            os.environ.setdefault("CUDNN_FRONTEND_ENABLE_FROST_ENGINES", "1")
        try:
            import cudnn  # pylint: disable=import-outside-toplevel

            if enable_frost_engines:
                # pylint: disable=import-outside-toplevel,unused-import
                import cudnn.sdpa  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "cuDNN frontend Python package not found. "
                "Install it with: pip install nvidia-cudnn-frontend"
            ) from exc

        _cudnn = cudnn
    return _cudnn


def handle_for(device: torch.device, *, backend_name: str = "cuDNN attention"):
    """A cuDNN handle for ``device``, rebound to PyTorch's current stream on every call.

    Without the rebinding, cuDNN runs on its handle's own stream while the tensors and workspace
    are allocated on PyTorch's current stream, and nothing orders the two. That is not
    hypothetical: the p2p context-parallel ring issues attention inside
    ``with torch.cuda.stream(cp_stream)``, so on alternating ring steps the kernel and its buffers
    would otherwise be on different streams. The same cached plan is executed from different
    streams across steps, so this has to happen per call rather than once per handle.
    """
    if device.type != "cuda":
        raise ValueError(f"{backend_name} requires CUDA tensors; got device {device}")
    cudnn = _cudnn if _cudnn is not None else import_cudnn_frontend()
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    with torch.cuda.device(device):
        handle = _handles.get(device)
        if handle is None:
            handle = cudnn.create_handle()
            _handles[device] = handle
        cudnn.set_stream(handle=handle, stream=torch.cuda.current_stream(device).cuda_stream)
    return handle


def io_data_type(cudnn, dtype: torch.dtype, *, backend_name: str = "cuDNN attention"):
    """Map a torch dtype to the cuDNN frontend enum, for the dtypes these backends accept."""
    if dtype == torch.float16:
        return cudnn.data_type.HALF
    if dtype == torch.bfloat16:
        return cudnn.data_type.BFLOAT16
    raise ValueError(f"{backend_name} only supports FP16/BF16 tensors, got {dtype}")


def build_pygraph(dtype: torch.dtype, device: torch.device, *,
                  backend_name: str = "cuDNN attention"):
    """A cuDNN frontend graph for F16/BF16 SDPA, bound to this device's stream-current handle."""
    cudnn = _cudnn if _cudnn is not None else import_cudnn_frontend()
    return cudnn.pygraph(
        io_data_type=io_data_type(cudnn, dtype, backend_name=backend_name),
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
        handle=handle_for(device, backend_name=backend_name),
    )


def bhsd_dim_stride(
    tensor: torch.Tensor, tensor_format: str
) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """Describe an SBHD/BSHD tensor as cuDNN frontend's logical BHSD form.

    No copy and no permute: the strides are handed to cuDNN as they are, which is what lets both
    layouts be served directly. sbhd matters because that is what Megatron uses internally.
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
    raise ValueError(f"Only SBHD/BSHD tensor formats are supported, got {tensor_format}.")


def bhsd_graph_tensor(graph, tensor: torch.Tensor, tensor_format: str):
    """Create a cuDNN graph tensor with BHSD dims and the tensor's own strides."""
    dim, stride = bhsd_dim_stride(tensor, tensor_format)
    return graph.tensor(dim=dim, stride=stride, data_type=tensor.dtype)


def diagonal_band_kwargs(cudnn, attn_mask_type: str, window: Tuple[int, int]) -> Dict[str, Any]:
    """cuDNN sdpa kwargs for a TE (mask type, window): a diagonal alignment plus a band.

    Note the off-by-one. cuDNN's left bound counts the diagonal itself and TE's window_size does
    not, so a window of w becomes a left bound of w + 1. Passing it through unconverted silently
    drops one token of context per layer, which no shape-level test would catch.

    These kwargs are mutually exclusive with score_mod: cuDNN rejects a graph carrying both with
    "Attention score mod enabled and hence other subgraphs are disabled".
    """
    left, right = window
    opts: Dict[str, Any] = {}
    if attn_mask_type in ("causal", "causal_bottom_right") or right == 0:
        opts["diagonal_alignment"] = (
            cudnn.diagonal_alignment.BOTTOM_RIGHT
            if attn_mask_type == "causal_bottom_right"
            else cudnn.diagonal_alignment.TOP_LEFT
        )
        opts["diagonal_band_right_bound"] = 0
    if left != -1:
        opts["diagonal_band_left_bound"] = left + 1
    return opts


def finalize_plans(
    graph,
    *,
    heuristics: Optional[Sequence[Any]] = None,
    build_policy: Any = None,
    require_plan_token: Optional[str] = None,
    not_found_hint: Any = "",
) -> Tuple[int, Optional[str]]:
    """Create plans, optionally pin one by name, build, and return (workspace size, plan name).

    ``require_plan_token`` makes the choice strict: only a plan whose name contains the token is
    acceptable, and anything else raises. That is not a stylistic preference. Without a pin,
    ``build_plans`` walks the ranked list and finalizes the first plan that builds, so a graph
    that a specialised engine declines would quietly run on a fallback instead, which for the
    FROST head-dim range is the wrong kernel rather than a slower one.

    The pin must precede ``check_support``: that call is scoped to the *selected* plan, so running
    it first would answer for whichever plan the heuristic happened to rank at index 0.
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
            raise RuntimeError(f"cuDNN SDPA graph is not supported: {exc}") from exc
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
    graph.check_support()
    graph.build_plans()
    return max(graph.get_workspace_size(), 1), names[hits[0]]


def selected_plan_name(graph, index: int = 0) -> str:
    """Name of the plan at ``index``, for logging and for asserting which engine answered."""
    return graph.get_plan_name_at_index(index)


def execute_graph(
    graph,
    variant_pack: Dict[Any, torch.Tensor],
    workspace_size: int,
    device: torch.device,
    *,
    backend_name: str = "cuDNN attention",
):
    """Execute a built graph on this device's stream-current handle."""
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    workspace = torch.empty(workspace_size, device=device, dtype=torch.uint8)
    graph.execute(
        variant_pack,
        workspace,
        handle=handle_for(device, backend_name=backend_name),
    )
