# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""cuDNN FROST attention. This feature is **experimental and subject to change**.

FROST runs through cudnn-frontend's Python API and is registered behind the
CUDNN_FRONTEND_ENABLE_FROST_ENGINES=1 flag. Different from FusedAttnBackend.F16_arbitrary_seqlen
and FusedAttnBackend.FP8, FusedAttnBackend.FROST is Python only and requires the Python
installation of cudnn-frontend, not only its C++ header files.
"""

from __future__ import annotations

import os
from typing import Optional, Sequence, Tuple

import torch
from packaging.version import Version as PkgVersion

from . import cudnn_pygraph

__all__ = [
    "is_frost_attention_available",
    "is_frost_attention_supported",
    "fused_attn_fwd",
    "fused_attn_bwd",
]


# cudnn-frontend declares cutlass-dsl >= 4.6.2 but FROST enforces >= 4.7.0 at plan-build time.
# Below that floor every FROST engine declines silently and backend plans come back instead, so
# the selected plan is checked by name rather than trusted.
_FROST_FWD_PLAN_TOKEN, _FROST_BWD_PLAN_TOKEN = cudnn_pygraph.FROST_PLAN_TOKENS
_MIN_CUTLASS_DSL = PkgVersion("4.7.0")

# 1.29.0 is the first release carrying the head_dim=512 backward. 1.28.0 ships the forward only,
# so without this check training would build a forward plan and raise on the first backward.
_MIN_CUDNN_FRONTEND = PkgVersion("1.29.0")

_SUPPORTED_ARCHS = ((10, 0), (10, 3))
# These mirror sdpa_bwd_sm100's declared envelope: d_envelope_floor=256, d={512},
# d_pad_multiple=8.
_MAX_HEAD_DIM = 512
# Below this the existing cuDNN and flash backends already serve the shape.
_MIN_HEAD_DIM = 257
# The engine pads head_dim to a multiple of 8, so 260 is in range but not servable. Declined
# here rather than failing later at plan selection.
_HEAD_DIM_MULTIPLE = 8

_BACKEND_NAME = "FrostAttention"
_availability: Optional[Tuple[bool, str]] = None


def is_frost_attention_available() -> Tuple[bool, str]:
    """Whether the FROST kernels can be used at all, with a reason when they cannot.

    Cached, because selection consults it for every config that resolves to FROST.
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
        # Decline here, or finalize_plans raises later with no FROST plan on offer.
        return _no("CUDNN_FRONTEND_ENABLE_FROST_ENGINES=0 disables the FROST engines")
    if torch.cuda.get_device_capability() not in _SUPPORTED_ARCHS:
        major, minor = torch.cuda.get_device_capability()
        return _no(f"cuDNN FROST head_dim>256 kernels are SM100/SM103 only; found sm{major}{minor}")
    try:
        # Without the engines: this only reads a version, and enabling reorders plan selection
        # process-wide even when the checks below decline. The use sites enable it.
        frontend_module = cudnn_pygraph.import_cudnn_frontend(enable_frost_engines=False)
    except ImportError as exc:
        return _no(f"nvidia-cudnn-frontend not importable: {exc}")

    # Decline only on positive evidence: a version below a floor, or a package absent outright.
    # An unparseable version defers to the plan pin, which checks the engine by name.
    frontend, frontend_raw = cudnn_pygraph.pkg_version("nvidia-cudnn-frontend", frontend_module)
    if frontend is not None and frontend < _MIN_CUDNN_FRONTEND:
        return _no(
            f"nvidia-cudnn-frontend {frontend_raw} registers no sm100 backward engine; >="
            f" {_MIN_CUDNN_FRONTEND} is required (1.28.0 ships the d512 forward only, so this would"
            " otherwise raise on the first backward rather than here)"
        )

    cutlass, cutlass_raw = cudnn_pygraph.pkg_version("nvidia-cutlass-dsl")
    if cutlass_raw is None:
        return _no(f"nvidia-cutlass-dsl not installed (FROST requires >= {_MIN_CUTLASS_DSL})")
    if cutlass is not None and cutlass < _MIN_CUTLASS_DSL:
        return _no(
            f"nvidia-cutlass-dsl {cutlass_raw} is below the FROST floor {_MIN_CUTLASS_DSL}; FROST"
            " engines would be silently skipped in favour of ordinary cuDNN backend plans"
        )

    _availability = (True, "")
    return _availability


# Both alignments are needed: the p2p ring produces square tiles where top-left and bottom-right
# coincide, while all_gather trims KV so they differ.
_SUPPORTED_MASKS = ("no_mask", "causal", "causal_bottom_right")

# Sliding window as TE spells it: (left, right), -1 meaning unbounded on that side.
_NO_WINDOW = (-1, -1)


def _window_pair(window_size) -> Tuple[int, int]:
    """Normalise window_size to a (left, right) pair, declining anything that is not one.

    Guarded here and not in _mask_spec, since _te_mask_spec unpacks the window first. Raised as
    NotImplementedError because that is the only exception is_frost_attention_supported catches;
    a TypeError would escape backend selection.
    """
    if window_size is None:
        return _NO_WINDOW
    try:
        window = tuple(window_size)
    except TypeError:
        raise NotImplementedError(
            f"window_size must be a (left, right) pair; got {window_size!r}"
        ) from None
    if len(window) != 2 or not all(isinstance(bound, int) for bound in window):
        raise NotImplementedError(
            f"window_size must be a pair of ints (left, right); got {window!r}"
        )
    return window


def _mask_spec(attn_mask_type: str, window_size=None):
    """Validate a TE mask type and window, returning the hashable spec the plan is keyed on."""
    if attn_mask_type not in _SUPPORTED_MASKS:
        raise NotImplementedError(
            f"FROST attention supports attn_mask_type in {str(_SUPPORTED_MASKS)}; got"
            f" {attn_mask_type!r}"
        )
    window = _window_pair(window_size)
    if window[0] < -1:
        # cuDNN's left bound must be >= 1, so a left of -2 would build diagonal_band_left_bound=-1
        # and fail at plan build rather than declining here.
        raise NotImplementedError(f"window_size left must be -1 or >= 0; got {window!r}")
    if window[1] not in (-1, 0):
        # A right bound past the diagonal is future context. cuDNN can express it, but no TE mask
        # type asks for it, so decline rather than guess the intent.
        raise NotImplementedError(f"FROST attention does not support a right window {window!r}")
    return attn_mask_type, window


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
    if qkv_format not in _SUPPORTED_QKV_FORMATS:
        raise NotImplementedError(
            f"FROST attention supports qkv_format in {_SUPPORTED_QKV_FORMATS}; got"
            f" {qkv_format!r}. thd needs varlen support that is not implemented here."
        )
    return qkv_format


def _te_mask_spec(attn_mask_type: str, window_size, bottom_right_diagonal: bool):
    """Fold TE's (mask type, window, diagonal anchor) into the spec the plan is keyed on.

    TE carries the anchor in its own flag, so normalise it into the mask type before building the
    band: the band reads the anchor off the name, and taking it from the name alone would quietly
    give a top-left band where the caller asked for bottom-right.
    """
    if "padding" in attn_mask_type:
        raise NotImplementedError(
            f"FROST attention does not support a padding mask; got {attn_mask_type!r}"
        )
    left, right = _window_pair(window_size)
    if "causal" in attn_mask_type and right == -1:
        right = 0
    if right == 0:
        attn_mask_type = "causal_bottom_right" if bottom_right_diagonal else "causal"
    else:
        attn_mask_type = "no_mask"
    return _mask_spec(attn_mask_type, (left, right))


def _bottom_right_diagonal(attn_mask_type: str, bottom_right_diagonal) -> bool:
    """Resolve the anchor flag the way cpp_extensions.fused_attn does.

    ``None`` means read it off the mask name. Only a direct call sees it unresolved.
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

    Returns (sub-backend value, reject message) like tex.get_fused_attn_backend, so
    get_attention_backend can fall through to it when the C++ backends decline. Does not probe
    availability, which would enable the FROST engines process-wide; get_attention_backend does
    that once at the end.
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

    # Per dimension: v has its own graph node and its own cache-key entry.
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
    # The rule is the backward engine's, not this range's. sdpa_bwd_sm100 is the only f16 FROST
    # backward on SM100/SM103 and leaves dqk_ge_dv unset, which cuDNN reads as requiring
    # d_qk == d_v at every head dim. The forward engine would serve 192/128, so only it binds.
    if params.head_dim_qk != params.head_dim_v:
        # Declined for both directions, since is_training follows module.training and eval()
        # leaves autograd on, so an eval-mode call is no promise that no backward follows.
        return (
            no_backend,
            (
                "FROST requires symmetric head_dim in (256, 512]; got"
                f" {params.head_dim_qk}/{params.head_dim_v}"
            ),
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
        return (
            no_backend,
            (
                "FROST has no deterministic backward; its dQ accumulation is atomic and unordered,"
                " so repeat runs differ in the last bits"
            ),
        )

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
    for name, value in (
        ("o_format", _name_for(QKVFormat, params.o_format)),
        ("do_format", _name_for(QKVFormat, params.do_format)),
        ("dqkv_layout", _name_for(QKVLayout, params.dqkv_layout)),
    ):
        if value is None:
            continue
        value = _qkv_format_from_layout(value) if name == "dqkv_layout" else value
        if value != qkv_format:
            return (
                no_backend,
                (
                    f"FROST needs {name} to match qkv_format, since the kernels write O and dQKV"
                    f" with q's strides; got {value}/{qkv_format}"
                ),
            )

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
        # top-left, while the all-gather ring measures its window bottom-right.
        return (
            no_backend,
            (
                "FROST declines a right-bounded window on a non-causal mask with max_seqlen_q !="
                " max_seqlen_kv, where the diagonal anchor is ambiguous"
            ),
        )

    return int(FusedAttnBackend.FROST), ""


def _check_dtype(name: str, t: torch.Tensor, expected: torch.dtype) -> None:
    """Require a tensor to carry the dtype its graph node was declared with.

    ``d_o`` matters most: it arrives from autograd and is not this module's to control.
    """
    if t.dtype != expected:
        raise ValueError(f"{name} must be {expected} to match q; got {t.dtype}")


def _check_kv_match(k: torch.Tensor, v: torch.Tensor) -> None:
    """Require v to agree with k on batch, heads and sequence length.

    head_dim is free: v has its own graph node and its own cache-key entry.
    """
    if tuple(k.shape[:3]) != tuple(v.shape[:3]):
        raise ValueError(
            f"k and v must agree on batch, heads and seqlen; got {k.shape} and {v.shape}"
        )


def _head_dim_strides(shape: Sequence[int], ref_strides: Sequence[int]) -> list:
    """Dense strides for ``shape`` in the memory order ``ref_strides`` describes."""
    order = sorted(range(len(shape)), key=lambda i: ref_strides[i], reverse=True)
    strides = [0] * len(shape)
    acc = 1
    for i in reversed(order):
        strides[i] = acc
        acc *= shape[i]
    return strides


def _o_shape_stride(shape, d_v, ref_strides):
    """Shape and strides for an O-shaped tensor: ``shape``'s layout carrying v's head_dim.

    Works in either space, since the head dim is last in both TE's bshd/sbhd and cuDNN's BHSD and
    the rule only reorders by stride magnitude. Equal head dims keep the reference strides, which
    preserves a caller's non-dense view.
    """
    out = list(shape[:3]) + [d_v]
    return out, (list(ref_strides) if d_v == shape[3] else _head_dim_strides(out, ref_strides))


def _frost_plan_args(token: str, what: str) -> dict:
    """Plan arguments that pin an engine whose name proves FROST was chosen.

    Without the pin a too-old nvidia-cutlass-dsl lets an ordinary forward engine build and
    compute something else silently, instead of raising at the first forward.
    """

    cudnn = cudnn_pygraph.import_cudnn_frontend(enable_frost_engines=True)

    # Both versions, because either floor can cause this and blaming one misdirects. Looked up
    # defensively: this explains a failure, so it must not raise itself.
    def hint():
        return (
            f"Wanted the FROST {what} engine."
            f" nvidia-cudnn-frontend={cudnn_pygraph.pkg_version('nvidia-cudnn-frontend', cudnn)[1] or 'unknown'} (floor"
            f" {_MIN_CUDNN_FRONTEND}),"
            f" nvidia-cutlass-dsl={cudnn_pygraph.pkg_version('nvidia-cutlass-dsl')[1] or 'unknown'} (floor"
            f" {_MIN_CUTLASS_DSL})."
        )

    return {
        "heuristics": [cudnn.heur_mode.A],
        "require_plan_token": token,
        "not_found_hint": hint,
    }


def _build_fwd(key, device) -> dict:
    """Build (and JIT-compile) a forward graph. Expensive; always reached through the cache."""
    cudnn = cudnn_pygraph.import_cudnn_frontend(enable_frost_engines=True)
    # deterministic selects a backward algorithm; the forward is always keyed False.
    _, _dev, ((shq, qs, dtype), (shk, ks, _), (shv, vs, _)), scale, _deterministic, mask = key
    b, hq, sq = shq[0], shq[1], shq[2]
    sho, o_stride = _o_shape_stride(shq, shv[3], qs)
    return cudnn_pygraph.build_fwd(
        dtype=dtype,
        device=device,
        backend_name=_BACKEND_NAME,
        name="frost_fwd",
        q=(shq, qs),
        k=(shk, ks),
        v=(shv, vs),
        out=(sho, o_stride),  # out: q's layout, v's head_dim
        # the CP ring needs the LSE, and it is cheap
        stats=([b, hq, sq, 1], [hq * sq, sq, 1, 1], cudnn.data_type.FLOAT),
        attn_scale=scale,
        mask=mask,
        **_frost_plan_args(_FROST_FWD_PLAN_TOKEN, "forward"),
    )


def _build_bwd(key, device) -> dict:
    """Build (and JIT-compile) a backward graph. Expensive; always reached through the cache."""
    cudnn = cudnn_pygraph.import_cudnn_frontend(enable_frost_engines=True)
    _, _dev, ((shq, qs, dtype), (shk, ks, _), (shv, vs, _)), scale, deterministic, mask = key
    io_dt = cudnn_pygraph.io_data_type(cudnn, dtype, backend_name=_BACKEND_NAME)
    b, hq, sq = shq[0], shq[1], shq[2]
    sho, o_stride = _o_shape_stride(shq, shv[3], qs)
    return cudnn_pygraph.build_bwd(
        dtype=dtype,
        device=device,
        backend_name=_BACKEND_NAME,
        name="frost_bwd",
        q=(shq, qs),
        k=(shk, ks),
        v=(shv, vs),
        o=(sho, o_stride),
        do=(sho, o_stride),
        stats=([b, hq, sq, 1], [hq * sq, sq, 1, 1], cudnn.data_type.FLOAT),
        # Each grad is declared with the layout of the tensor it differentiates.
        dq=(None, qs, io_dt),
        dk=(None, ks, io_dt),
        dv=(None, vs, io_dt),
        attn_scale=scale,
        deterministic=deterministic,
        mask=mask,
        **_frost_plan_args(_FROST_BWD_PLAN_TOKEN, "backward"),
    )


def _cached(key, device):
    """Plan cache. Building dominates executing even once the JIT is cached, so this is
    required rather than an optimisation."""
    build = _build_fwd if key[0] == "fwd" else _build_bwd
    return cudnn_pygraph.cached_graph(_BACKEND_NAME, key, lambda: build(key, device), device=device)


def _validate_qkv(q, k, v, qkv_format):
    """Check the tensors the graph will bind, and return their BHSD descriptions.

    ``execute`` binds raw pointers, so a tensor disagreeing with the node it is bound to is
    reinterpreted rather than rejected. Both directions validate: the context-parallel ring
    calls the backward outside autograd, so neither may assume the other ran first.

    Rank is checked here because cuDNN never sees the tensor, only the descriptor this module
    builds from it. bhsd_dim_stride reads dims 0-3, so a rank-5 tensor is described as 4D, and
    the cache key carries strides but not rank, so it can reuse a valid 4D plan. A non-unit
    head-dim stride does reach the descriptor, so graph.validate() catches that one.
    """
    for name, tensor in (("q", q), ("k", k), ("v", v)):
        if tensor.dim() != 4:
            raise ValueError(f"{name} must be 4D; got shape {tuple(tensor.shape)}")
        _check_dtype(name, tensor, q.dtype)
    _check_kv_match(k, v)
    qd, _ = cudnn_pygraph.bhsd_dim_stride(q, qkv_format, backend_name=_BACKEND_NAME)
    kd, _ = cudnn_pygraph.bhsd_dim_stride(k, qkv_format, backend_name=_BACKEND_NAME)
    vd, _ = cudnn_pygraph.bhsd_dim_stride(v, qkv_format, backend_name=_BACKEND_NAME)
    if kd[0] != qd[0] or kd[3] != qd[3]:
        # The graph declares k and v with q's batch and head_dim, so a mismatch would bind a
        # differently shaped buffer to that node and read the wrong elements silently.
        raise ValueError(f"k must match q in batch and head_dim; got q {qd} and k {kd} in BHSD")
    if qd[1] % kd[1] != 0:
        raise ValueError(f"num_heads must be divisible by num_gqa_groups; got {qd[1]} and {kd[1]}")
    return qd, kd, vd


def _key(direction, q, k, v, qkv_format, mask, scale, deterministic=False):
    """The plan cache key, which is also the build recipe: the builders destructure it and hold
    no tensors of their own."""
    return cudnn_pygraph.sdpa_cache_key(
        direction,
        q.device,
        [
            cudnn_pygraph.bhsd_tensor_spec(t, qkv_format, backend_name=_BACKEND_NAME)
            for t in (q, k, v)
        ],
        attn_scale=scale,
        deterministic=deterministic,
        extras=(mask,),
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

    Returns (out, aux_ctx_tensors) with aux_ctx_tensors = [softmax_lse, rng_state]. softmax_lse
    is [b, h, s, 1] fp32 natural-log logsumexp, matching the C++ fused_attn_fwd; the ring
    correction squeezes it itself.

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
    mask = _te_mask_spec(
        attn_mask_type, window_size, _bottom_right_diagonal(attn_mask_type, bottom_right_diagonal)
    )

    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    qd, _, vd = _validate_qkv(q, k, v, qkv_format)
    scale = attn_scale if attn_scale is not None else qd[3] ** -0.5
    entry = _cached(_key("fwd", q, k, v, qkv_format, mask, scale), q.device)
    tq, tk, tv, tout, tlse = (entry[n] for n in ("q", "k", "v", "out", "stats"))

    # empty_strided, not empty_like: the output carries v's head_dim, so it is not q's shape.
    # The LSE is BHSD whatever the inputs were, which is what the ring correction expects.
    out_shape, out_stride = _o_shape_stride(q.shape, vd[3], q.stride())
    out = torch.empty_strided(out_shape, out_stride, device=q.device, dtype=q.dtype)
    lse = torch.empty(qd[0], qd[1], qd[2], 1, device=q.device, dtype=torch.float32)
    cudnn_pygraph.execute_graph(
        entry["graph"],
        {tq: q, tk: k, tv: v, tout: out, tlse: lse},
        entry["workspace"],
        q.device,
        backend_name=_BACKEND_NAME,
    )
    # A real tensor, not None: it is saved for backward and handed to the activation offload
    # hooks, neither of which accepts None. FROST has no dropout, so nothing reads it.
    rng_state = torch.empty(2, dtype=torch.int64, device=q.device)
    return out, [lse, rng_state]


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

    Returns (dq, dk, dv, dbias, dsoftmax_offset) with the last two always None, matching the
    fused path for a no_bias vanilla-softmax config.
    """
    _frost_only(
        dropout=dropout != 0.0,
        attention_bias=attn_bias_type != "no_bias",
        fp8=s_quantizer is not None or dqkv_quantizer is not None,
        sink_attention=softmax_type != "vanilla",
        cuda_graph_capture=cuda_graph,
    )
    qkv_format = _qkv_format_from_layout(qkv_layout)
    # The selector declines this too, but the CP ring calls this entry point directly.
    for name, fmt in (
        ("o_format", o_format),
        ("do_format", do_format),
        ("dqkv_layout", _qkv_format_from_layout(dqkv_layout)),
    ):
        if fmt != qkv_format:
            raise NotImplementedError(
                f"FROST attention needs {name} to match qkv_format, since o and dO are read"
                " with q's strides and the grads returned in its layout; got"
                f" {fmt}/{qkv_format}"
            )
    mask = _te_mask_spec(
        attn_mask_type, window_size, _bottom_right_diagonal(attn_mask_type, bottom_right_diagonal)
    )
    softmax_lse = aux_ctx_tensors[0]

    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    o, d_o = o.contiguous(), d_o.contiguous()
    qd, _, vd = _validate_qkv(q, k, v, qkv_format)
    for name, tensor in (("o", o), ("d_o", d_o)):
        _check_dtype(name, tensor, q.dtype)
    o_shape, o_stride = _o_shape_stride(q.shape, vd[3], q.stride())
    for name, tensor in (("o", o), ("d_o", d_o)):
        if list(tensor.shape) != o_shape:
            raise ValueError(f"{name} must be shaped {o_shape}; got {list(tensor.shape)}")
    if softmax_lse.dtype != torch.float32:
        raise ValueError(f"softmax_lse must be fp32; got {softmax_lse.dtype}")
    # Compared against the BHSD description, not q's own shape: the LSE is always [b, h, s]
    # whatever format the tensors arrived in. The whole shape too: the stats node is declared
    # [b, h, s, 1], so any other trailing dim is bound by pointer and read with wrong strides.
    lse_shape = tuple(softmax_lse.shape)
    if lse_shape not in (tuple(qd[:3]), tuple(qd[:3]) + (1,)):
        raise ValueError(
            f"softmax_lse must be [b, h, s] or [b, h, s, 1] matching q; got {lse_shape}"
        )

    scale = attn_scale if attn_scale is not None else qd[3] ** -0.5
    entry = _cached(_key("bwd", q, k, v, qkv_format, mask, scale, deterministic), q.device)
    h = entry

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
    cudnn_pygraph.execute_graph(
        entry["graph"],
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
        entry["workspace"],
        q.device,
        backend_name=_BACKEND_NAME,
    )
    return dq, dk, dv, None, None
