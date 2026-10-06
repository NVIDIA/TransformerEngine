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

Three properties of these kernels were verified on Blackwell, and each constrains the code:

1. cuDNN's causal masking is TOP_LEFT aligned unless bottom-right is requested. The two coincide
   when SQ == SKV, so the distinction is invisible in square tests and decisive for all_gather,
   which trims KV. Masking is built as a diagonal band so causal, bottom-right and sliding window
   come from one mechanism.

2. Plan building must be cached. It dominates an execute even after cuDNN has cached the JIT, so a
   per-call build would leave training build-bound. Hence `_PLAN_CACHE`.

3. The forward LSE is natural-log logsumexp in fp32, shaped [b, h, s, 1]. Squeezed to [b, h, s] it
   is what the CP ring correction in context_parallel.py consumes, which is what makes ring
   attention over these kernels valid at all.
"""

from __future__ import annotations

import contextlib
import os
from importlib.metadata import PackageNotFoundError, version as get_pkg_version
from typing import Optional, Tuple

import torch
from packaging.version import InvalidVersion, Version as PkgVersion

from transformer_engine.pytorch.attention.dot_product_attention import cudnn_pygraph

__all__ = [
    "is_frost_attention_available",
    "is_frost_attention_supported",
    "fused_attn_fwd",
    "fused_attn_bwd",
    "frost_attn_fwd",
    "frost_attn_bwd",
    "to_frost_layout",
    "from_frost_layout",
]


# FROST engines are opt-in inside cuDNN Frontend, and they additionally require a newer
# nvidia-cutlass-dsl than cudnn-frontend itself declares. cudnn-frontend requires >= 4.6.2 while
# FROST enforces >= 4.7.0 at plan-build time; with 4.6.2 installed every FROST engine silently
# declines and ordinary cuDNN backend plans are returned with no error at all. We therefore check
# the selected plan by NAME rather than trusting that the engine was used.
_FROST_FWD_PLAN_TOKEN = "sdpa_fwd_prefill_sm100"
_FROST_BWD_PLAN_TOKEN = "sdpa_bwd_sm100"
_MIN_CUTLASS_DSL = PkgVersion("4.7.0")

# 1.29.0 is the first release carrying the head_dim=512 BACKWARD (bprop_d512_f16_sm100). 1.28.0
# ships the forward only, and the repo's own pin allows it, so without this check training would
# build a forward plan and then raise on the first backward.
_MIN_CUDNN_FRONTEND = PkgVersion("1.29.0")

_SUPPORTED_ARCHS = ((10, 0), (10, 3))
_MAX_HEAD_DIM = 512
_MIN_HEAD_DIM = 257  # below this the existing cuDNN/flash backends already serve the shape
# The engine pads head_dim to a multiple of 8, so 260 is not servable even though it is in range.
# Without this it passes the gate and then fails at plan selection with a message about missing
# engines, instead of declining cleanly here.
_HEAD_DIM_MULTIPLE = 8

_cudnn = None
_availability: Optional[Tuple[bool, str]] = None
_PLAN_CACHE: dict = {}
_HANDLES = cudnn_pygraph._handles  # pylint: disable=protected-access


def _import_cudnn(enable_frost_engines: bool = True):
    """Import cuDNN Frontend, registering the FROST engines unless told not to.

    The switch is process-wide and ranks FROST ahead of the backend engines for every cuDNN Python
    graph afterwards, including other backends’ graphs, so it is set only where FROST is actually
    used. _select_frost_plan verifies the engine by plan name regardless, rather than trusting the
    flag.
    """
    global _cudnn  # pylint: disable=global-statement
    # Kept bound: _pkg_version falls back to the module's __version__ when distribution metadata
    # is unavailable, which is how a source or vendored install avoids being misreported.
    _cudnn = cudnn_pygraph.import_cudnn_frontend(enable_frost_engines=enable_frost_engines)
    return _cudnn


def _handle_for(device: torch.device):
    """A cuDNN handle for `device`, bound to PyTorch's current stream on every call."""
    return cudnn_pygraph.handle_for(device, backend_name="FrostAttention")


def _device_from_key(device_key) -> torch.device:
    """Rebuild the torch.device that _key recorded, for building under the right device."""
    kind, index = device_key
    return torch.device(kind) if index is None else torch.device(kind, index)


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
        # Without the engines: this only needs the module to read a version off it, and enabling
        # here would reorder plan selection for the whole process even when the checks below go on
        # to decline FROST, which is all cost and no benefit. The use sites enable it.
        _import_cudnn(enable_frost_engines=False)
    except ImportError as exc:
        return _no(f"nvidia-cudnn-frontend not importable: {exc}")

    # Decline on positive evidence that FROST cannot work: a version below a floor, or a package
    # that is absent outright. A version that is present but unparseable is NOT evidence, so it
    # defers to _select_frost_plan, which checks the plan by name and reports both versions.
    frontend, frontend_raw = _pkg_version("nvidia-cudnn-frontend", _cudnn)
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


# TE mask types this backend serves. cuDNN expresses causal, bottom-right and sliding-window
# masking as ONE mechanism -- a diagonal alignment plus a two-sided band -- rather than three
# separate flags, so that is what _mask_options builds. The legacy spellings desugar into exactly
# that: pygraph/sdpa.cpp maps use_causal_mask to (TOP_LEFT, right_bound=0) and
# use_causal_mask_bottom_right to (BOTTOM_RIGHT, right_bound=0), and refuses to combine either
# with an explicit right bound. Building the band directly is equivalent for those two and
# additionally expresses a left bound, which is what a sliding window is.
#
# Both alignments are needed. The p2p ring produces square diagonal tiles, where top-left and
# bottom-right coincide, while all_gather trims KV and relies on bottom-right alignment, where
# the two differ completely.
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
    return cudnn_pygraph.diagonal_band_kwargs(cudnn, attn_mask_type, window)


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
    return formats.pop()


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

    head_dim_qk, head_dim_v = params.head_dim_qk, params.head_dim_v
    if head_dim_qk != head_dim_v:
        return no_backend, f"FROST requires symmetric head_dim; got {head_dim_qk}/{head_dim_v}"
    if not _MIN_HEAD_DIM <= head_dim_qk <= _MAX_HEAD_DIM:
        return no_backend, f"FROST covers head_dim in (256, 512]; got {head_dim_qk}"
    if head_dim_qk % _HEAD_DIM_MULTIPLE != 0:
        return (
            no_backend,
            f"FROST needs head_dim to be a multiple of {_HEAD_DIM_MULTIPLE}; got {head_dim_qk}",
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
        # A right-bounded window on a non-causal mask takes its anchor only from
        # bottom_right_diagonal, which defaults to top-left, while the all-gather ring trims KV
        # and measures its window against the bottom-right diagonal. Those differ exactly when
        # the q and kv lengths do, so decline rather than guess which one was meant.
        return no_backend, (
            "FROST declines a right-bounded window on a non-causal mask with max_seqlen_q !="
            " max_seqlen_kv, where the diagonal anchor is ambiguous"
        )

    return int(FusedAttnBackend.FROST), ""


def to_frost_layout(t: torch.Tensor, qkv_format: str) -> torch.Tensor:
    """View a tensor in TE's qkv_format as [b, h, s, d].

    No copy: the cuDNN graphs are built from each tensor's actual strides, so both bshd and
    sbhd are served directly. sbhd matters because that is what Megatron uses internally, and
    transposing into bshd on every call would copy the whole tensor.
    """
    if qkv_format == "bshd":  # [b, s, h, d] -> [b, h, s, d]
        return t.permute(0, 2, 1, 3)
    if qkv_format == "sbhd":  # [s, b, h, d] -> [b, h, s, d]
        return t.permute(1, 2, 0, 3)
    raise NotImplementedError(
        f"FROST attention supports qkv_format 'bshd' and 'sbhd'; got {qkv_format!r}. thd needs"
        " varlen support that is not implemented here."
    )


def from_frost_layout(t: torch.Tensor, qkv_format: str) -> torch.Tensor:
    """Inverse of to_frost_layout."""
    if qkv_format == "bshd":  # [b, h, s, d] -> [b, s, h, d]
        return t.permute(0, 2, 1, 3)
    if qkv_format == "sbhd":  # [b, h, s, d] -> [s, b, h, d]
        return t.permute(2, 0, 1, 3)
    raise NotImplementedError(
        f"FROST attention supports qkv_format 'bshd' and 'sbhd'; got {qkv_format!r}."
    )


def _cudnn_dtype(dtype: torch.dtype):
    cudnn = _import_cudnn()
    return {
        torch.bfloat16: cudnn.data_type.BFLOAT16,
        torch.float16: cudnn.data_type.HALF,
    }[dtype]


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
    """Require v to match k in both shape and layout.

    Both graphs declare v with k's shape and stride, and _key records only q's and k's, so a v
    that differs would hit a cached plan built for k's layout and read the wrong elements with no
    error at all. Callers in TE always split k and v from one QKV tensor, so this costs nothing
    and is purely a guard against a silent wrong answer.
    """
    if k.shape != v.shape:
        raise ValueError(f"k and v must have the same shape; got {k.shape} and {v.shape}")
    if k.stride() != v.stride():
        raise ValueError(
            f"k and v must have the same layout; got strides {tuple(k.stride())} and"
            f" {tuple(v.stride())}"
        )


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
            f"{_pkg_version('nvidia-cudnn-frontend', _cudnn)[1] or 'unknown'}"
            f" (floor {_MIN_CUDNN_FRONTEND}),"
            f" nvidia-cutlass-dsl={_pkg_version('nvidia-cutlass-dsl')[1] or 'unknown'}"
            f" (floor {_MIN_CUTLASS_DSL})."
        )

    cudnn = _import_cudnn()
    _, name = cudnn_pygraph.finalize_plans(
        graph,
        heuristics=[cudnn.heur_mode.A],
        require_plan_token=token,
        not_found_hint=hint,
    )
    return name


def _build_fwd(key) -> dict:
    """Build (and JIT-compile) a forward graph. Expensive; always reached through the cache."""
    cudnn = _import_cudnn()
    # deterministic is unused here: it selects a backward algorithm. Callers pass False for the
    # forward so the two never split the forward cache.
    *_device, b, hq, hkv, sq, skv, d, dtype, mask, scale, qs, ks, _deterministic = key
    shq, shkv = [b, hq, sq, d], [b, hkv, skv, d]

    graph = cudnn_pygraph.build_pygraph(
        dtype, _device_from_key(_device), backend_name="FrostAttention"
    )
    tq = graph.tensor(name="q", dim=shq, stride=list(qs))
    tk = graph.tensor(name="k", dim=shkv, stride=list(ks))
    tv = graph.tensor(name="v", dim=shkv, stride=list(ks))
    tout, tlse = graph.sdpa(
        name="frost_fwd",
        q=tq,
        k=tk,
        v=tv,
        generate_stats=True,  # the CP ring needs the LSE, and it is cheap
        attn_scale=scale,
        **_mask_options(cudnn, mask),
    )
    tout.set_output(True).set_dim(shq).set_stride(list(qs))  # out mirrors q
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


def _build_bwd(key) -> dict:
    """Build (and JIT-compile) a backward graph. Expensive; always reached through the cache."""
    cudnn = _import_cudnn()
    *_device, b, hq, hkv, sq, skv, d, dtype, mask, scale, qs, ks, deterministic = key
    io_dt = _cudnn_dtype(dtype)
    shq, shkv = [b, hq, sq, d], [b, hkv, skv, d]

    graph = cudnn_pygraph.build_pygraph(
        dtype, _device_from_key(_device), backend_name="FrostAttention"
    )
    handles = {}
    # o and dO share q's layout; k, v and their grads share k's.
    for name, shape, stride in (
        ("q", shq, qs),
        ("k", shkv, ks),
        ("v", shkv, ks),
        ("o", shq, qs),
        ("do", shq, qs),
    ):
        handles[name] = graph.tensor(name=name, dim=shape, stride=list(stride))
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
    for tensor, stride in ((tdq, qs), (tdk, ks), (tdv, ks)):
        tensor.set_output(True).set_data_type(io_dt).set_stride(list(stride))
    plan = _select_frost_plan(graph, _FROST_BWD_PLAN_TOKEN, "backward")
    handles["dq"], handles["dk"], handles["dv"] = tdq, tdk, tdv
    return {
        "graph": graph,
        "handles": handles,
        "workspace": max(graph.get_workspace_size(), 1),
        "plan": plan,
    }


def _cached(kind: str, key):
    """Plan cache. See module docstring: building dominates executing even once the JIT is
    cached, so this is required rather than an optimisation."""
    cache_key = (kind,) + key
    entry = _PLAN_CACHE.get(cache_key)
    if entry is None:
        # Build under the device the key names, not merely with that device's handle: the plans
        # are CuTe-DSL JIT-compiled, and a compile path is far more likely to read the ambient
        # CUDA context than the handle. Free to do, and removes the question entirely.
        device = _device_from_key(key[:2])
        with torch.cuda.device(device) if device.type == "cuda" else contextlib.nullcontext():
            entry = _build_fwd(key) if kind == "fwd" else _build_bwd(key)
        _PLAN_CACHE[cache_key] = entry
    return entry


def _key(q, k, mask, scale, deterministic=False):
    return (
        # The graph is built under whichever device was current, so it must not be reused on
        # another one. Matches the C++ fused-attn cache, which keys on device_id for the same
        # reason. Type is included too, so a CPU tensor cannot alias cuda:0.
        q.device.type,
        q.device.index,
        q.shape[0],
        q.shape[1],
        k.shape[1],
        q.shape[2],
        k.shape[2],
        q.shape[3],
        q.dtype,
        mask,
        float(scale),
        # Strides are part of the plan: the graph is built for this exact layout, which is what
        # lets bshd and sbhd both run without a transpose.
        tuple(q.stride()),
        tuple(k.stride()),
        # The deterministic backward is a different algorithm, not a flag on the same one, so a
        # plan built either way must not be handed to a call that asked for the other.
        bool(deterministic),
    )


def frost_attn_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_scale: Optional[float] = None,
    attn_mask_type: str = "causal",
    window_size: Optional[Tuple[int, int]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Forward attention via cuDNN FROST.

    q, k, v are [b, h, s, d] views; bshd and sbhd are both served, since the graph is built from
    each tensor's actual strides. GQA is supported directly (h_kv may differ from h_q) and SQ
    need not equal SKV, which is what lets a CP ring step use this. Returns (out, softmax_lse)
    with softmax_lse as [b, h, s] fp32 natural-log logsumexp, the layout and convention the CP
    ring correction expects.
    """
    for name, tensor in (("q", q), ("k", k), ("v", v)):
        _check_layout(name, tensor)
        _check_dtype(name, tensor, q.dtype)
    _check_kv_match(k, v)
    if k.shape[0] != q.shape[0] or k.shape[3] != q.shape[3]:
        # The graph declares k and v with q's batch and head_dim, so a mismatch would bind a
        # differently shaped buffer to that node and read the wrong elements silently.
        raise ValueError(f"k must match q in batch and head_dim; got q {q.shape} and k {k.shape}")
    if q.shape[1] % k.shape[1] != 0:
        raise ValueError(
            f"num_heads must be divisible by num_gqa_groups; got {q.shape[1]} and {k.shape[1]}"
        )

    mask = _mask_spec(attn_mask_type, window_size)
    scale = attn_scale if attn_scale is not None else q.shape[-1] ** -0.5
    entry = _cached("fwd", _key(q, k, mask, scale))
    tq, tk, tv, tout, tlse = entry["handles"]

    b, hq, sq, _ = q.shape
    # Allocate per call: the cache holds only the compiled plan, never output buffers, so that
    # concurrent or nested uses cannot alias each other. empty_strided rather than empty_like:
    # the latter does not preserve an arbitrary permuted stride, and the graph was built for
    # q's exact strides.
    out = torch.empty_strided(q.shape, q.stride(), device=q.device, dtype=q.dtype)
    lse = torch.empty(b, hq, sq, 1, device=q.device, dtype=torch.float32)
    workspace = torch.empty(entry["workspace"], device=q.device, dtype=torch.uint8)
    entry["graph"].execute(
        {tq: q, tk: k, tv: v, tout: out, tlse: lse}, workspace, handle=_handle_for(q.device)
    )
    return out, lse.squeeze(-1)


def frost_attn_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    softmax_lse: torch.Tensor,
    dout: torch.Tensor,
    attn_scale: Optional[float] = None,
    attn_mask_type: str = "causal",
    deterministic: bool = False,
    window_size: Optional[Tuple[int, int]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Backward attention via cuDNN FROST. `softmax_lse` is [b, h, s] as returned by the forward."""
    for name, tensor in (("q", q), ("k", k), ("v", v), ("out", out), ("dout", dout)):
        _check_layout(name, tensor)
        _check_dtype(name, tensor, q.dtype)
    _check_kv_match(k, v)
    # The same shape assumptions the forward makes, plus o/dO, which the graph declares with q's
    # shape. The forward runs first in autograd, but the CP ring calls this directly.
    if k.shape[0] != q.shape[0] or k.shape[3] != q.shape[3]:
        raise ValueError(f"k must match q in batch and head_dim; got q {q.shape} and k {k.shape}")
    if q.shape[1] % k.shape[1] != 0:
        raise ValueError(
            f"num_heads must be divisible by num_gqa_groups; got {q.shape[1]} and {k.shape[1]}"
        )
    for name, tensor in (("out", out), ("dout", dout)):
        if tensor.shape != q.shape:
            raise ValueError(f"{name} must have q's shape; got {tensor.shape} and {q.shape}")
    if softmax_lse.dtype != torch.float32:
        raise ValueError(f"softmax_lse must be fp32; got {softmax_lse.dtype}")
    if tuple(softmax_lse.shape[:3]) != tuple(q.shape[:3]):
        raise ValueError(
            f"softmax_lse must be [b, h, s] matching q; got {tuple(softmax_lse.shape)} and"
            f" {tuple(q.shape)}"
        )

    mask = _mask_spec(attn_mask_type, window_size)
    scale = attn_scale if attn_scale is not None else q.shape[-1] ** -0.5
    entry = _cached("bwd", _key(q, k, mask, scale, deterministic))
    h = entry["handles"]

    if softmax_lse.dim() == 3:
        softmax_lse = softmax_lse.unsqueeze(-1)
    softmax_lse = softmax_lse.contiguous()

    # The graph expects o and dO in q's layout. A caller may hand us either with different
    # strides (dO in particular comes from autograd), so restride rather than silently reading
    # the wrong elements.
    def _as(t, ref):
        if tuple(t.stride()) == tuple(ref.stride()):
            return t
        buf = torch.empty_strided(t.shape, ref.stride(), device=t.device, dtype=t.dtype)
        buf.copy_(t)
        return buf

    out = _as(out, q)
    dout = _as(dout, q)

    dq = torch.empty_strided(q.shape, q.stride(), device=q.device, dtype=q.dtype)
    dk = torch.empty_strided(k.shape, k.stride(), device=k.device, dtype=k.dtype)
    dv = torch.empty_strided(v.shape, v.stride(), device=v.device, dtype=v.dtype)
    workspace = torch.empty(entry["workspace"], device=q.device, dtype=torch.uint8)
    entry["graph"].execute(
        {
            h["q"]: q,
            h["k"]: k,
            h["v"]: v,
            h["o"]: out,
            h["do"]: dout,
            h["stats"]: softmax_lse,
            h["dq"]: dq,
            h["dk"]: dk,
            h["dv"]: dv,
        },
        workspace,
        handle=_handle_for(q.device),
    )
    return dq, dk, dv


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
    mask_type, window = _te_mask_spec(attn_mask_type, window_size, bool(bottom_right_diagonal))

    out, softmax_lse = frost_attn_fwd(
        to_frost_layout(q.contiguous(), qkv_format),
        to_frost_layout(k.contiguous(), qkv_format),
        to_frost_layout(v.contiguous(), qkv_format),
        attn_scale=attn_scale,
        attn_mask_type=mask_type,
        window_size=window,
    )
    # A real tensor rather than None: it is saved for backward and handed to the activation
    # offload hooks alongside softmax_lse, neither of which accepts None. FROST has no dropout,
    # so nothing reads it.
    rng_state = torch.empty(2, dtype=torch.int64, device=q.device)
    return from_frost_layout(out, qkv_format), [softmax_lse, rng_state]


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
    mask_type, window = _te_mask_spec(attn_mask_type, window_size, bool(bottom_right_diagonal))
    softmax_lse = aux_ctx_tensors[0]

    dq, dk, dv = frost_attn_bwd(
        to_frost_layout(q.contiguous(), qkv_format),
        to_frost_layout(k.contiguous(), qkv_format),
        to_frost_layout(v.contiguous(), qkv_format),
        to_frost_layout(o.contiguous(), o_format),
        softmax_lse,
        to_frost_layout(d_o.contiguous(), do_format),
        attn_scale=attn_scale,
        attn_mask_type=mask_type,
        deterministic=deterministic,
        window_size=window,
    )
    return (
        from_frost_layout(dq, qkv_format),
        from_frost_layout(dk, qkv_format),
        from_frost_layout(dv, qkv_format),
        None,
    )
