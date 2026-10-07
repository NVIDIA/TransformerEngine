# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""cuDNN FROST attention backend for head_dim in (256, 512] on SM100/SM103.

**Experimental and subject to change.** The engines this wraps are themselves experimental in
cuDNN Frontend, and if the fused path gains these shapes this backend may be folded into it.

Why a separate Python backend rather than teaching the existing C++ fused path: FROST engines are
registered at Python import time behind CUDNN_FRONTEND_ENABLE_FROST_ENGINES and require the
nvidia-cutlass-dsl Python package, while TE's C++ builds against cuDNN Frontend headers only.
Reaching them requires a Python graph, which is what this module is.
"""

from __future__ import annotations

import os
from importlib.metadata import PackageNotFoundError, version as get_pkg_version
from typing import Any, Dict, Optional, Sequence, Tuple

import torch
from packaging.version import InvalidVersion, Version as PkgVersion

from . import cudnn_pygraph

__all__ = [
    "is_frost_attention_available",
    "is_frost_attention_supported",
    "fused_attn_fwd",
    "fused_attn_bwd",
]


# cudnn-frontend declares cutlass-dsl >= 4.6.2 but FROST enforces >= 4.7.0 at plan-build time.
# Below that floor every FROST engine declines silently and backend plans come back instead, so
# the selected plan is checked by NAME rather than trusting that the engine was used.
_FROST_FWD_PLAN_TOKEN = "sdpa_fwd_prefill_sm100"
_FROST_BWD_PLAN_TOKEN = "sdpa_bwd_sm100"
_MIN_CUTLASS_DSL = PkgVersion("4.7.0")

# 1.29.0 is the first release carrying the head_dim=512 backward. 1.28.0 ships the forward only,
# so without this check training would build a forward plan and raise on the first backward.
_MIN_CUDNN_FRONTEND = PkgVersion("1.29.0")

_SUPPORTED_ARCHS = ((10, 0), (10, 3))
_MAX_HEAD_DIM = 512
_MIN_HEAD_DIM = 257  # below this the existing cuDNN/flash backends already serve the shape
# The engine pads head_dim to a multiple of 8, so 260 is in range but not servable. Declined
# here rather than failing later at plan selection.
_HEAD_DIM_MULTIPLE = 8

_BACKEND_NAME = "FrostAttention"
_availability: Optional[Tuple[bool, str]] = None
_PLAN_CACHE: dict = {}


def _import_cudnn_frontend(enable_frost_engines: bool = True):
    """Import cuDNN Frontend with the FROST engines on, which is what this backend needs.

    The default differs from the shared module's, where it is off. Every use site here wants the
    engines; a caller that does not must not ask for them, because the switch is process-wide.
    See ``cudnn_pygraph.import_cudnn_frontend`` for why the enabling sits outside the import memo.
    """
    return cudnn_pygraph.import_cudnn_frontend(enable_frost_engines=enable_frost_engines)


def _diagonal_band_kwargs(cudnn, attn_mask_type: str, window: Tuple[int, int]) -> Dict[str, Any]:
    """cuDNN sdpa kwargs for a TE (mask type, window): a diagonal alignment plus a band.

    Note the off-by-one. cuDNN's left bound counts the diagonal itself and TE's window_size does
    not, so a window of w becomes a left bound of w + 1. Passing it through unconverted silently
    drops one token of context per layer, which no shape-level test would catch.

    These kwargs are mutually exclusive with score_mod. cuDNN enforces that in the backward node
    only ("Attention score mod enabled and hence other subgraphs are disabled"); its forward node
    composes the two without complaint. Callers must still refuse the pair on both sides, because
    forward and backward have to carry the same mask or the gradients belong to a different
    attention than the output does.
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


def _pkg_version(name: str, module=None) -> Tuple[Optional[PkgVersion], Optional[str]]:
    """(parsed version, raw string) for a package. Either element is None if undeterminable.

    Distribution metadata first, matching the sibling check in fused_mla_q_uproj.py, with the
    module attribute as a fallback so a source or vendored install is not misreported as absent.
    The raw string is returned separately so callers can tell "not installed" from "installed but
    unparseable"; those warrant different answers, and conflating them declines valid installs.
    """
    raw = None
    for candidate in (lambda: get_pkg_version(name), lambda: getattr(module, "__version__", None)):
        try:
            raw = candidate()
        except PackageNotFoundError:
            raw = None
        if isinstance(raw, str):
            break
        raw = None
    if raw is None:
        return None, None
    try:
        return PkgVersion(raw), raw
    except InvalidVersion:
        return None, raw


def is_frost_attention_available() -> Tuple[bool, str]:
    """Whether the FROST kernels can be used at all, with a reason when they cannot.

    Cached, because this is consulted on every backend-selection call.
    """
    global _availability
    if _availability is not None:
        return _availability

    def _no(reason):
        global _availability
        _availability = (False, reason)
        return _availability

    if not torch.cuda.is_available():
        return _no("no CUDA device")
    if os.environ.get("CUDNN_FRONTEND_ENABLE_FROST_ENGINES", "1") == "0":
        # Explicitly switched off. Declining here is the difference between falling back cleanly
        # and raising from _select_frost_plan once a plan is built.
        return _no("CUDNN_FRONTEND_ENABLE_FROST_ENGINES=0 disables the FROST engines")
    if torch.cuda.get_device_capability() not in _SUPPORTED_ARCHS:
        major, minor = torch.cuda.get_device_capability()
        return _no(f"cuDNN FROST head_dim>256 kernels are SM100/SM103 only; found sm{major}{minor}")
    try:
        # Without the engines: this only reads a version, and enabling reorders plan selection
        # process-wide even when the checks below decline. The use sites enable it.
        _import_cudnn_frontend(enable_frost_engines=False)
    except ImportError as exc:
        return _no(f"nvidia-cudnn-frontend not importable: {exc}")

    # Decline only on positive evidence: a version below a floor, or a package absent outright.
    # An unparseable version defers to _select_frost_plan, which checks the plan by name.
    frontend, frontend_raw = _pkg_version("nvidia-cudnn-frontend", cudnn_pygraph.cudnn_module())
    if frontend is not None and frontend < _MIN_CUDNN_FRONTEND:
        return _no(
            f"nvidia-cudnn-frontend {frontend_raw} registers no sm100 backward engine; >="
            f" {_MIN_CUDNN_FRONTEND} is required (1.28.0 ships the d512 forward only, so this would"
            " otherwise raise on the first backward rather than here)"
        )

    cutlass, cutlass_raw = _pkg_version("nvidia-cutlass-dsl")
    if cutlass_raw is None:
        return _no(f"nvidia-cutlass-dsl not installed (FROST requires >= {_MIN_CUTLASS_DSL})")
    if cutlass is not None and cutlass < _MIN_CUTLASS_DSL:
        # Worth being loud: this combination fails by silently declining, not by raising.
        return _no(
            f"nvidia-cutlass-dsl {cutlass_raw} is below the FROST floor {_MIN_CUTLASS_DSL}; FROST"
            " engines would be silently skipped in favour of ordinary cuDNN backend plans"
        )

    _availability = (True, "")
    return _availability


# cuDNN expresses causal, bottom-right and sliding-window masking as one mechanism, a diagonal
# alignment plus a two-sided band, which is what _mask_options builds. Both alignments are needed:
# the p2p ring produces square tiles where they coincide, while all_gather trims KV so they differ.
_SUPPORTED_MASKS = ("no_mask", "causal", "causal_bottom_right")

# Sliding window as TE spells it: (left, right), -1 meaning unbounded on that side.
_NO_WINDOW = (-1, -1)


def _mask_spec(attn_mask_type: str, window_size=None):
    """Validate a TE mask type and window, returning the hashable spec the plan is keyed on."""
    if attn_mask_type not in _SUPPORTED_MASKS:
        raise NotImplementedError(
            f"FROST attention supports attn_mask_type in {str(_SUPPORTED_MASKS)}; got"
            f" {attn_mask_type!r}"
        )
    try:
        window = _NO_WINDOW if window_size is None else tuple(window_size)
    except TypeError:
        # Raised as NotImplementedError so the selector declines instead of propagating out of
        # backend selection, which is the only thing is_frost_attention_supported catches.
        raise NotImplementedError(
            f"window_size must be a (left, right) pair; got {window_size!r}"
        ) from None
    if len(window) != 2:
        raise NotImplementedError(f"window_size must be a (left, right) pair; got {window!r}")
    if window[0] < -1:
        # cuDNN's left bound must be >= 1, so a left of -2 would build diagonal_band_left_bound=-1
        # and fail at plan build rather than declining here.
        raise NotImplementedError(f"window_size left must be -1 or >= 0; got {window!r}")
    if window[1] not in (-1, 0):
        # A right bound past the diagonal is future context. cuDNN can express it, but no TE mask
        # type asks for it, so decline rather than guess the intent.
        raise NotImplementedError(f"FROST attention does not support a right window {window!r}")
    return attn_mask_type, window


def _mask_options(cudnn, spec):
    """cuDNN sdpa kwargs for a (mask type, window) spec: a diagonal alignment plus a band."""
    attn_mask_type, window = spec
    return _diagonal_band_kwargs(cudnn, attn_mask_type, window)


_SUPPORTED_QKV_FORMATS = ("bshd", "sbhd")


def _qkv_format_from_layout(qkv_layout: str) -> str:
    """The single qkv_format a TE qkv_layout names, e.g. 'bshd_bshd_bshd' -> 'bshd'."""
    formats = {
        "".join(c for c in part if c.isalpha())
        for part in qkv_layout.replace("paged_kv_", "").split("_")
    }
    if len(formats) != 1:
        raise NotImplementedError(
            f"FROST attention needs q, k and v in one format; got qkv_layout {qkv_layout!r}"
        )
    qkv_format = formats.pop()
    # Carried over from the permute helper this replaced, so the thd decline keeps its reason.
    if qkv_format not in _SUPPORTED_QKV_FORMATS:
        raise NotImplementedError(
            f"FROST attention supports qkv_format in {_SUPPORTED_QKV_FORMATS}; got"
            f" {qkv_format!r}. thd needs varlen support that is not implemented here."
        )
    return qkv_format


def _te_mask_spec(attn_mask_type: str, window_size, bottom_right_diagonal: bool):
    """Fold TE's (mask type, window, diagonal anchor) into the spec the plan is keyed on.

    TE carries the anchor in its own flag, so normalise it into the mask type before building the
    band: diagonal_band_kwargs reads the anchor off the name, and taking it from the name alone
    would quietly give a top-left band where the caller asked for bottom-right.
    """
    if "padding" in attn_mask_type:
        raise NotImplementedError(
            f"FROST attention does not support a padding mask; got {attn_mask_type!r}"
        )
    left, right = _NO_WINDOW if window_size is None else tuple(window_size)
    if "causal" in attn_mask_type and right == -1:
        right = 0
    if right == 0:
        attn_mask_type = "causal_bottom_right" if bottom_right_diagonal else "causal"
    else:
        attn_mask_type = "no_mask"
    return _mask_spec(attn_mask_type, (left, right))


def _bottom_right_diagonal(attn_mask_type: str, bottom_right_diagonal) -> bool:
    """Resolve the anchor flag the same way cpp_extensions.fused_attn does.

    ``None`` means "read it off the mask name". The dispatcher resolves it before calling in,
    so this only matters for a direct call, where ``bool(None)`` would quietly give a top-left
    band to a caller that asked for bottom-right.
    """
    if bottom_right_diagonal is None:
        return attn_mask_type in {"causal_bottom_right", "padding_causal_bottom_right"}
    return bool(bottom_right_diagonal)


def _name_for(table, value, default=None):
    """Reverse a cpp_extensions str-to-enum table."""
    for name, enum_value in table.items():
        if enum_value == value:
            return name
    return default


def is_frost_attention_supported(params) -> Tuple[int, str]:
    """Whether this fused-attention config should run on the FROST sub-backend.

    Takes a FusedAttentionParams and returns (sub-backend value, reject message), the same shape
    as tex.get_fused_attn_backend, so get_attention_backend can fall through to it when the C++
    backends decline.

    Deliberately does not probe availability. That imports cuDNN Frontend with the FROST engines
    enabled, which changes the engine pool for every cuDNN consumer in the process, and this runs
    for every attention config on the machine. get_attention_backend checks availability once at
    the end, the way it checks flash-attn versions.
    """
    # pylint: disable-next=import-outside-toplevel
    from ...cpp_extensions.fused_attn import (
        AttnBiasType,
        AttnMaskType,
        FusedAttnBackend,
        QKVFormat,
        QKVLayout,
        SoftmaxType,
        TORCH_DType,
    )

    no_backend = int(FusedAttnBackend.No_Backend)

    if int(os.environ.get("NVTE_FROST_ATTN", "1")) == 0:
        return no_backend, "FROST is disabled by NVTE_FROST_ATTN=0"

    # Each head_dim is checked on its own: q/k and v get separate graph nodes, so an
    # asymmetric pair is served as long as both dims land in the range.
    for name, head_dim in (
        ("head_dim_qk", params.head_dim_qk),
        ("head_dim_v", params.head_dim_v),
    ):
        if not _MIN_HEAD_DIM <= head_dim <= _MAX_HEAD_DIM:
            return no_backend, f"FROST covers head_dim in (256, 512]; got {name}={head_dim}"
        if head_dim % _HEAD_DIM_MULTIPLE != 0:
            return (
                no_backend,
                f"FROST needs {name} to be a multiple of {_HEAD_DIM_MULTIPLE}; got {head_dim}",
            )

    qkv_dtype = TORCH_DType.get(params.qkv_dtype)
    if qkv_dtype not in (torch.bfloat16, torch.float16):
        return no_backend, f"FROST supports bf16/fp16; got {params.qkv_dtype}"
    if params.dropout != 0.0:
        return no_backend, "FROST does not support dropout"
    if _name_for(AttnBiasType, params.bias_type) != "no_bias":
        return no_backend, "FROST does not support attention bias"
    if _name_for(SoftmaxType, params.softmax_type) != "vanilla":
        return no_backend, "FROST only supports vanilla softmax"
    if params.num_pages_k != 0 or params.num_pages_v != 0:
        return no_backend, "FROST does not support paged KV"
    if params.return_max_logit:
        return no_backend, "FROST does not return max_logit"
    if params.cuda_graph:
        return no_backend, "FROST graphs are built lazily and cannot be captured"
    if params.deterministic and params.is_training:
        # The backward uses an atomic dQ accumulation whose order is not fixed, so repeat runs
        # differ in the last bits. Nothing selects a deterministic variant, so decline instead.
        return no_backend, "FROST does not have a deterministic backward"

    qkv_layout = _name_for(QKVLayout, params.qkv_layout)
    if qkv_layout is None:
        return no_backend, f"FROST got an unrecognised qkv_layout {params.qkv_layout}"
    try:
        qkv_format = _qkv_format_from_layout(qkv_layout)
    except NotImplementedError as exc:
        return no_backend, str(exc)
    if qkv_format not in _SUPPORTED_QKV_FORMATS:
        return (
            no_backend,
            f"FROST supports qkv_format in {_SUPPORTED_QKV_FORMATS}; got {qkv_format}",
        )
    # The kernels write O and dQKV with q's strides, so any format that differs from the input
    # would need a copy the fused path does not make. Nothing asks for one today.
    for name, value in (
        ("o_format", _name_for(QKVFormat, params.o_format)),
        ("do_format", _name_for(QKVFormat, params.do_format)),
        ("dqkv_layout", _name_for(QKVLayout, params.dqkv_layout)),
    ):
        if value is None:
            continue
        value = _qkv_format_from_layout(value) if name == "dqkv_layout" else value
        if value != qkv_format:
            return no_backend, f"FROST needs {name} to match qkv_format; got {value}/{qkv_format}"

    attn_mask_type = _name_for(AttnMaskType, params.attn_mask_type)
    if attn_mask_type is None:
        return no_backend, f"FROST got an unrecognised attn_mask_type {params.attn_mask_type}"
    try:
        mask_for_band, _ = _te_mask_spec(
            attn_mask_type,
            (params.window_size_left, params.window_size_right),
            params.bottom_right_diagonal,
        )
    except NotImplementedError as exc:
        return no_backend, str(exc)
    if (
        mask_for_band == "causal"
        and "causal" not in attn_mask_type
        and params.max_seqlen_q != params.max_seqlen_kv
    ):
        # Such a window takes its anchor only from bottom_right_diagonal, which defaults to
        # top-left, while the all-gather ring measures its window bottom-right. Decline rather
        # than guess which was meant.
        return (
            no_backend,
            (
                "FROST declines a right-bounded window on a non-causal mask with max_seqlen_q !="
                " max_seqlen_kv, where the diagonal anchor is ambiguous"
            ),
        )

    return int(FusedAttnBackend.FROST), ""


def _check_layout(name: str, t: torch.Tensor) -> None:
    """Validate a [b, h, s, d] view.

    The graphs are built from each tensor's ACTUAL strides rather than one fixed layout, so bshd
    and sbhd are both served without a transpose. The only hard requirement is that the head
    dimension is contiguous, which the kernels assume.
    """
    if t.dim() != 4:
        raise ValueError(f"{name} must be 4D [b, h, s, d]; got {tuple(t.shape)}")
    if t.stride(3) != 1:
        raise ValueError(
            f"{name} must have a contiguous head dimension; got shape {tuple(t.shape)} stride"
            f" {tuple(t.stride())}"
        )


def _check_dtype(name: str, t: torch.Tensor, expected: torch.dtype) -> None:
    """Require a tensor to carry the dtype its graph node was declared with.

    Every node but `stats` is declared from q's dtype, and execute() binds raw pointers, so a
    tensor of another dtype would have its bits reinterpreted with no error at all. `dout`
    matters most: it arrives from autograd and is not this module's to control.
    """
    if t.dtype != expected:
        raise ValueError(f"{name} must be {expected} to match q; got {t.dtype}")


def _check_kv_match(k: torch.Tensor, v: torch.Tensor) -> None:
    """Require v to agree with k on batch, heads and sequence length.

    head_dim is free: v has its own graph node and its own cache-key entry, so an asymmetric
    pair builds its own plan. The other three index the same KV positions as k by definition,
    and a mismatch would bind a differently shaped buffer with no error at all.
    """
    if tuple(k.shape[:3]) != tuple(v.shape[:3]):
        raise ValueError(
            f"k and v must agree on batch, heads and seqlen; got {k.shape} and {v.shape}"
        )


def _head_dim_strides(shape: Sequence[int], ref_strides: Sequence[int]) -> list:
    """Dense strides for ``shape`` in the memory order ``ref_strides`` describes.

    O, dO and the O-shaped grads follow q's layout but carry v's head_dim, so when the two head
    dims differ they cannot reuse q's strides. The graph node and the allocation both go through
    here so they cannot drift apart.
    """
    order = sorted(range(len(shape)), key=lambda i: ref_strides[i], reverse=True)
    strides = [0] * len(shape)
    acc = 1
    for i in reversed(order):
        strides[i] = acc
        acc *= shape[i]
    return strides


def _o_shape_stride(shape, d_v, ref_strides):
    """Shape and strides for an O-shaped tensor: ``shape``'s layout carrying v's head_dim.

    Works in either space. The head dim is last in both TE's bshd/sbhd and cuDNN's BHSD, and the
    rule only reorders by stride magnitude, so the graph node and the allocation can each apply it
    in their own space and still agree. Equal head dims keep the reference strides untouched,
    which preserves a caller's non-dense view.
    """
    out = list(shape[:3]) + [d_v]
    return out, (list(ref_strides) if d_v == shape[3] else _head_dim_strides(out, ref_strides))


def _select_frost_plan(graph, token: str, what: str):
    """Select a plan whose name proves a FROST engine was chosen.

    Falling back to whatever plan happens to be first would defeat the purpose. A too-old
    nvidia-cutlass-dsl makes the FROST engines decline silently, and in the forward an ordinary
    engine may then build and compute something else; the pin turns that into a named error at
    the first forward rather than a wrong number or a backward that fails later for no visible
    reason.
    """

    # Both versions, because either floor can cause this and blaming one misdirects. Looked up
    # defensively: this explains a failure, so it must not raise itself.
    def hint():
        return (
            f"Wanted the FROST {what} engine."
            " nvidia-cudnn-frontend="
            f"{_pkg_version('nvidia-cudnn-frontend', cudnn_pygraph.cudnn_module())[1] or 'unknown'}"
            f" (floor {_MIN_CUDNN_FRONTEND}),"
            f" nvidia-cutlass-dsl={_pkg_version('nvidia-cutlass-dsl')[1] or 'unknown'}"
            f" (floor {_MIN_CUTLASS_DSL})."
        )

    cudnn = _import_cudnn_frontend()
    _, name = cudnn_pygraph.finalize_plans(
        graph,
        backend_name=_BACKEND_NAME,
        heuristics=[cudnn.heur_mode.A],
        require_plan_token=token,
        not_found_hint=hint,
    )
    return name


def _build_fwd(key, device) -> dict:
    """Build (and JIT-compile) a forward graph. Expensive; always reached through the cache."""
    cudnn = _import_cudnn_frontend()
    # deterministic is unused here: it selects a backward algorithm. Callers pass False for the
    # forward so the two never split the forward cache.
    _dev, (shq, qs, dtype), (shk, ks, _), (shv, vs, _), mask, scale, _deterministic = key
    b, hq, sq = shq[0], shq[1], shq[2]
    sho, o_stride = _o_shape_stride(shq, shv[3], qs)

    graph = cudnn_pygraph.build_pygraph(dtype, device, backend_name=_BACKEND_NAME)
    tq = graph.tensor(name="q", dim=list(shq), stride=list(qs))
    tk = graph.tensor(name="k", dim=list(shk), stride=list(ks))
    tv = graph.tensor(name="v", dim=list(shv), stride=list(vs))
    tout, tlse = graph.sdpa(
        name="frost_fwd",
        q=tq,
        k=tk,
        v=tv,
        generate_stats=True,  # the CP ring needs the LSE, and it is cheap
        attn_scale=scale,
        **_mask_options(cudnn, mask),
    )
    tout.set_output(True).set_dim(sho).set_stride(list(o_stride))  # out: q's layout, v's head_dim
    tlse.set_output(True).set_dim([b, hq, sq, 1]).set_stride([hq * sq, sq, 1, 1]).set_data_type(
        cudnn.data_type.FLOAT
    )
    plan = _select_frost_plan(graph, _FROST_FWD_PLAN_TOKEN, "forward")
    return {
        "graph": graph,
        "handles": (tq, tk, tv, tout, tlse),
        "workspace": max(graph.get_workspace_size(), 1),
        "plan": plan,
    }


def _build_bwd(key, device) -> dict:
    """Build (and JIT-compile) a backward graph. Expensive; always reached through the cache."""
    cudnn = _import_cudnn_frontend()
    _dev, (shq, qs, dtype), (shk, ks, _), (shv, vs, _), mask, scale, deterministic = key
    io_dt = cudnn_pygraph.io_data_type(cudnn, dtype, backend_name=_BACKEND_NAME)
    b, hq, sq = shq[0], shq[1], shq[2]
    sho, o_stride = _o_shape_stride(shq, shv[3], qs)

    graph = cudnn_pygraph.build_pygraph(dtype, device, backend_name=_BACKEND_NAME)
    handles = {}
    # Each grad is declared with the layout of the tensor it differentiates.
    for name, shape, stride in (
        ("q", shq, qs),
        ("k", shk, ks),
        ("v", shv, vs),
        ("o", sho, o_stride),
        ("do", sho, o_stride),
    ):
        handles[name] = graph.tensor(name=name, dim=list(shape), stride=list(stride))
    handles["stats"] = graph.tensor(
        name="stats",
        dim=[b, hq, sq, 1],
        stride=[hq * sq, sq, 1, 1],
        data_type=cudnn.data_type.FLOAT,
    )
    tdq, tdk, tdv = graph.sdpa_backward(
        name="frost_bwd",
        q=handles["q"],
        k=handles["k"],
        v=handles["v"],
        o=handles["o"],
        dO=handles["do"],
        stats=handles["stats"],
        attn_scale=scale,
        use_deterministic_algorithm=deterministic,
        **_mask_options(cudnn, mask),
    )
    for tensor, stride in ((tdq, qs), (tdk, ks), (tdv, vs)):
        tensor.set_output(True).set_data_type(io_dt).set_stride(list(stride))
    plan = _select_frost_plan(graph, _FROST_BWD_PLAN_TOKEN, "backward")
    handles["dq"], handles["dk"], handles["dv"] = tdq, tdk, tdv
    return {
        "graph": graph,
        "handles": handles,
        "workspace": max(graph.get_workspace_size(), 1),
        "plan": plan,
    }


def _cached(kind: str, key, device):
    """Plan cache. See module docstring: building dominates executing even once the JIT is
    cached, so this is required rather than an optimisation."""
    build = _build_fwd if kind == "fwd" else _build_bwd
    return cudnn_pygraph.cached_graph(
        _PLAN_CACHE, (kind,) + key, lambda: build(key, device), device=device
    )


def _validate_qkv(q, k, v, qkv_format):
    """Check the tensors the graph will bind, and return their BHSD descriptions.

    These are not stylistic guards. ``execute`` binds raw pointers, so a tensor whose shape,
    dtype or layout disagrees with the node it is bound to is reinterpreted rather than rejected.
    The context-parallel ring calls the backward outside autograd, so neither direction may
    assume the other ran first.
    """
    for name, tensor in (("q", q), ("k", k), ("v", v)):
        _check_layout(name, tensor)
        _check_dtype(name, tensor, q.dtype)
    _check_kv_match(k, v)
    qd, _ = _bhsd(q, qkv_format)
    kd, _ = _bhsd(k, qkv_format)
    vd, _ = _bhsd(v, qkv_format)
    if kd[0] != qd[0] or kd[3] != qd[3]:
        # The graph declares k and v with q's batch and head_dim, so a mismatch would bind a
        # differently shaped buffer to that node and read the wrong elements silently.
        raise ValueError(f"k must match q in batch and head_dim; got q {qd} and k {kd} in BHSD")
    if qd[1] % kd[1] != 0:
        raise ValueError(f"num_heads must be divisible by num_gqa_groups; got {qd[1]} and {kd[1]}")
    return qd, kd, vd


def _bhsd(t: torch.Tensor, qkv_format: str):
    """``t`` described in cuDNN's logical BHSD, without permuting it."""
    return cudnn_pygraph.bhsd_dim_stride(t, qkv_format, backend_name=_BACKEND_NAME)


def _key(q, k, v, qkv_format, mask, scale, deterministic=False):
    """The plan cache key.

    Structured per tensor rather than flattened, so the builders destructure it by name instead
    of by position, and so the per-tensor fragment is the same one flex_attention keys on.
    """

    def described(t):
        return cudnn_pygraph.tensor_key(t, qkv_format, backend_name=_BACKEND_NAME)

    return (
        # Built under whichever device was current, so it must not be reused on another. Matches
        # the C++ fused-attn cache, which keys on device_id.
        cudnn_pygraph.device_key(q.device),
        described(q),
        described(k),
        described(v),
        mask,
        float(scale),
        # The deterministic backward is a different algorithm, not a flag on the same one, so a
        # plan built either way must not be handed to a call that asked for the other.
        bool(deterministic),
    )


def _frost_only(**unsupported):
    """Raise if any feature the selector should have declined reached the kernels anyway."""
    for name, value in unsupported.items():
        if value:
            raise NotImplementedError(f"FROST attention does not support {name}")


def fused_attn_fwd(
    is_training,
    max_seqlen_q,
    max_seqlen_kv,
    cu_seqlens_q,
    cu_seqlens_kv,
    q,
    k,
    v,
    fake_dtype,
    fused_attention_backend,
    attn_bias=None,
    cu_seqlens_q_padded=None,
    cu_seqlens_kv_padded=None,
    page_table_k=None,
    page_table_v=None,
    s_quantizer=None,
    o_quantizer=None,
    attn_scale=None,
    dropout=0.0,
    fast_zero_fill=True,
    qkv_layout="sbh3d",
    o_format="sbhd",
    qkv_scale_inv_format=None,
    attn_bias_type="no_bias",
    attn_mask_type="padding",
    softmax_type="vanilla",
    window_size=(-1, -1),
    bottom_right_diagonal=None,
    rng_gen=None,
    softmax_offset=None,
    return_max_logit=False,
    cuda_graph=False,
):  # pylint: disable=unused-argument
    """FROST forward behind the cpp_extensions.fused_attn_fwd signature.

    Mirrors that signature so FusedAttnFunc and the context-parallel ring reach these kernels
    without knowing which sub-backend they got. Returns (out, aux_ctx_tensors) with
    aux_ctx_tensors = [softmax_lse, rng_state]; softmax_lse is [b, h, s] fp32 natural-log
    logsumexp, which is what the ring correction consumes.

    cu_seqlens and the padded variants are ignored: they carry thd offsets, and thd is declined
    at selection.
    """
    _frost_only(
        dropout=dropout != 0.0,
        attention_bias=attn_bias_type != "no_bias",
        paged_kv=page_table_k is not None or page_table_v is not None,
        fp8=s_quantizer is not None or o_quantizer is not None,
        sink_attention=softmax_type != "vanilla",
        max_logit=return_max_logit,
        cuda_graph_capture=cuda_graph,
    )
    qkv_format = _qkv_format_from_layout(qkv_layout)
    if o_format != qkv_format:
        raise NotImplementedError(
            f"FROST attention needs o_format to match qkv_format; got {o_format}/{qkv_format}"
        )
    # _te_mask_spec validates as it normalises, so what it returns is the spec the plan keys on.
    mask = _te_mask_spec(
        attn_mask_type, window_size, _bottom_right_diagonal(attn_mask_type, bottom_right_diagonal)
    )

    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    qd, _, vd = _validate_qkv(q, k, v, qkv_format)
    scale = attn_scale if attn_scale is not None else qd[3] ** -0.5
    entry = _cached("fwd", _key(q, k, v, qkv_format, mask, scale), q.device)
    tq, tk, tv, tout, tlse = entry["handles"]

    # Allocated per call so concurrent uses cannot alias; the cache holds only the plan.
    # empty_strided, not empty_like: the latter does not preserve an arbitrary permuted stride.
    # The output takes the caller's format with v's head_dim, so nothing is converted on the way
    # out; the LSE is BHSD whatever the inputs were, which is what the ring correction expects.
    out_shape, out_stride = _o_shape_stride(q.shape, vd[3], q.stride())
    out = torch.empty_strided(out_shape, out_stride, device=q.device, dtype=q.dtype)
    lse = torch.empty(qd[0], qd[1], qd[2], 1, device=q.device, dtype=torch.float32)
    workspace = torch.empty(entry["workspace"], device=q.device, dtype=torch.uint8)
    entry["graph"].execute(
        {tq: q, tk: k, tv: v, tout: out, tlse: lse},
        workspace,
        handle=cudnn_pygraph.handle_for(q.device, backend_name=_BACKEND_NAME),
    )
    # A real tensor, not None: it is saved for backward and handed to the activation offload
    # hooks, neither of which accepts None. FROST has no dropout, so nothing reads it.
    rng_state = torch.empty(2, dtype=torch.int64, device=q.device)
    return out, [lse.squeeze(-1), rng_state]


def fused_attn_bwd(
    max_seqlen_q,
    max_seqlen_kv,
    cu_seqlens_q,
    cu_seqlens_kv,
    q,
    k,
    v,
    o,
    d_o,
    fake_dtype,
    aux_ctx_tensors,
    fused_attention_backend,
    cu_seqlens_q_padded=None,
    cu_seqlens_kv_padded=None,
    s_quantizer=None,
    dp_quantizer=None,
    dqkv_quantizer=None,
    attn_scale=None,
    dropout=0.0,
    fast_zero_fill=True,
    qkv_layout="sbh3d",
    o_format="sbhd",
    do_format="sbhd",
    dqkv_layout="sbh3d",
    qkv_scale_inv_format=None,
    do_scale_inv_format=None,
    attn_bias_type="no_bias",
    attn_mask_type="padding",
    softmax_type="vanilla",
    window_size=(-1, -1),
    bottom_right_diagonal=None,
    deterministic=False,
    cuda_graph=False,
):  # pylint: disable=unused-argument
    """FROST backward behind the cpp_extensions.fused_attn_bwd signature.

    Returns (dq, dk, dv, dbias) with dbias always None, matching what the fused path returns for
    a no_bias config.
    """
    _frost_only(
        dropout=dropout != 0.0,
        attention_bias=attn_bias_type != "no_bias",
        fp8=s_quantizer is not None or dqkv_quantizer is not None,
        sink_attention=softmax_type != "vanilla",
        cuda_graph_capture=cuda_graph,
    )
    qkv_format = _qkv_format_from_layout(qkv_layout)
    # o and dO used to carry their own format into the permute; they now share qkv_format, so a
    # divergence would silently describe them with the wrong strides. The selector already
    # declines it, but this is reached directly too.
    for name, fmt in (("o_format", o_format), ("do_format", do_format)):
        if fmt != qkv_format:
            raise NotImplementedError(
                f"FROST attention needs {name} to match qkv_format; got {fmt}/{qkv_format}"
            )
    mask = _te_mask_spec(
        attn_mask_type, window_size, _bottom_right_diagonal(attn_mask_type, bottom_right_diagonal)
    )
    softmax_lse = aux_ctx_tensors[0]

    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    o, d_o = o.contiguous(), d_o.contiguous()
    qd, _, vd = _validate_qkv(q, k, v, qkv_format)
    for name, tensor in (("o", o), ("d_o", d_o)):
        _check_layout(name, tensor)
        _check_dtype(name, tensor, q.dtype)
    o_shape, o_stride = _o_shape_stride(q.shape, vd[3], q.stride())
    for name, tensor in (("o", o), ("d_o", d_o)):
        if list(tensor.shape) != o_shape:
            raise ValueError(f"{name} must be shaped {o_shape}; got {list(tensor.shape)}")
    if softmax_lse.dtype != torch.float32:
        raise ValueError(f"softmax_lse must be fp32; got {softmax_lse.dtype}")
    # Compared against the BHSD description, not q's own shape: the LSE is always [b, h, s]
    # whatever format the tensors arrived in.
    if tuple(softmax_lse.shape[:3]) != tuple(qd[:3]):
        raise ValueError(
            f"softmax_lse must be [b, h, s] matching q; got {tuple(softmax_lse.shape)} and"
            f" {tuple(qd[:3])}"
        )

    scale = attn_scale if attn_scale is not None else qd[3] ** -0.5
    entry = _cached("bwd", _key(q, k, v, qkv_format, mask, scale, deterministic), q.device)
    h = entry["handles"]

    if softmax_lse.dim() == 3:
        softmax_lse = softmax_lse.unsqueeze(-1)
    softmax_lse = softmax_lse.contiguous()

    # The graph expects o and dO in the layout the forward wrote, and dO comes from autograd
    # with strides we do not control, so restride rather than silently reading the wrong elements.
    def _as(t, stride):
        if list(t.stride()) == list(stride):
            return t
        buf = torch.empty_strided(t.shape, stride, device=t.device, dtype=t.dtype)
        buf.copy_(t)
        return buf

    o, d_o = _as(o, o_stride), _as(d_o, o_stride)
    dq = torch.empty_strided(q.shape, q.stride(), device=q.device, dtype=q.dtype)
    dk = torch.empty_strided(k.shape, k.stride(), device=k.device, dtype=k.dtype)
    dv = torch.empty_strided(v.shape, v.stride(), device=v.device, dtype=v.dtype)
    workspace = torch.empty(entry["workspace"], device=q.device, dtype=torch.uint8)
    entry["graph"].execute(
        {
            h["q"]: q,
            h["k"]: k,
            h["v"]: v,
            h["o"]: o,
            h["do"]: d_o,
            h["stats"]: softmax_lse,
            h["dq"]: dq,
            h["dk"]: dk,
            h["dv"]: dv,
        },
        workspace,
        handle=cudnn_pygraph.handle_for(q.device, backend_name=_BACKEND_NAME),
    )
    return dq, dk, dv, None
