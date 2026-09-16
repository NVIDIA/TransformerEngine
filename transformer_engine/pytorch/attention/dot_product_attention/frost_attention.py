# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""cuDNN FROST attention backend for head_dim in (256, 512] on SM100/SM103.

Why this exists. Gemma-4 global layers use symmetric head_dim=512, and no backend TE can select
today serves both that head dim and context parallelism: FlashAttention 2/3 cap at 256, FA4 is
gated off at symmetric 512, the C++ cuDNN fused path is refused a graph by cuDNN above 256, and
the unfused path supports 512 but cannot do CP. cuDNN Frontend 1.29.0 ships CuTe-DSL ("FROST")
SDPA kernels that do serve symmetric 512 forward and backward on Blackwell.

Why a separate Python backend rather than teaching the existing C++ fused path. The 256 ceiling
there is not a TE check -- the f16 dispatch applies no head-dim test and simply asks cuDNN to
build a graph -- so the natural question is why the new engines cannot just be picked up. They
cannot: FROST engines are registered at Python import time behind
CUDNN_FRONTEND_ENABLE_FROST_ENGINES and require the nvidia-cutlass-dsl Python package, while
TE's C++ builds against cuDNN Frontend headers only. Reaching them therefore requires a Python
graph, which is what this module is.

Three properties of these kernels were verified on Blackwell before this was written, and each
one constrains the code:

1. cuDNN's `use_causal_mask` is TOP-LEFT aligned and `use_causal_mask_bottom_right` is
   bottom-right. They coincide when SQ == SKV, so the distinction is invisible in square tests
   and decisive for all_gather, which trims KV. Both alignments were checked against a
   reference rather than assumed, and masking is built as a diagonal band so causal,
   bottom-right and sliding window come from one mechanism instead of three spellings.

2. Plan building must be cached. Building a plan is by far the most expensive cuDNN frontend
   call here, and dominates an execute even after cuDNN has cached the JIT and made rebuilds
   cheap, so a per-call build would leave training build-bound. Hence `_PLAN_CACHE`.

3. The forward LSE is natural-log logsumexp in fp32, shaped [b, h, s, 1]. Squeezed to [b, h, s]
   it is exactly what the CP ring correction in context_parallel.py consumes, which is what
   makes ring attention over these kernels valid at all.

Numerics were validated against the criterion FlashAttention applies to itself, namely that the
kernel error must stay within 2x the error bf16 inputs alone produce, across square and
rectangular, causal and non-causal, windowed and unwindowed shapes.
"""

from __future__ import annotations

import contextlib
import os
from importlib.metadata import PackageNotFoundError, version as get_pkg_version
from typing import Optional, Tuple

import torch
from packaging.version import InvalidVersion, Version as PkgVersion

__all__ = [
    "is_frost_attention_available",
    "is_frost_attention_supported",
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
_HANDLES: dict = {}


def _import_cudnn():
    """Import cuDNN Frontend with FROST engines enabled, once.

    The switch is set before the import because the documentation describes the engines as
    registering at import time. Measured on B200 with cuDNN Frontend 1.29.0, the ordering turns
    out not to matter: importing cudnn and cudnn.sdpa first with the switch unset, then setting
    it and building a plan, still selects a FROST engine. Setting it first is kept because it is
    what the documentation asks for and costs nothing, but nothing here depends on winning that
    race, and _select_frost_plan verifies the engine by plan name regardless.
    """
    global _cudnn
    if _cudnn is None:
        # Must be set before the import: the engines are registered at import time.
        os.environ.setdefault("CUDNN_FRONTEND_ENABLE_FROST_ENGINES", "1")
        import cudnn  # pylint: disable=import-outside-toplevel
        import cudnn.sdpa  # noqa: F401  pylint: disable=import-outside-toplevel,unused-import

        _cudnn = cudnn
    return _cudnn


def _handle_for(device: torch.device):
    """A cuDNN handle for `device`, bound to PyTorch's current stream on it.

    Without this, cuDNN runs on its default handle's stream while the tensors and workspace are
    allocated on PyTorch's current stream, and nothing orders the two. That is not hypothetical
    here: the p2p CP ring issues attention inside `with torch.cuda.stream(cp_stream)`, so on
    alternating ring steps the kernel and its buffers would be on different streams. Re-binding
    on every call is what flex_attention.py does, and is required because the same cached plan is
    executed from different streams across ring steps.
    """
    if device.type != "cuda":
        raise ValueError("FrostAttention requires CUDA tensors; got device %s" % device)
    cudnn = _import_cudnn()
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    with torch.cuda.device(device):
        handle = _HANDLES.get(device)
        if handle is None:
            handle = cudnn.create_handle()
            _HANDLES[device] = handle
        cudnn.set_stream(handle=handle, stream=torch.cuda.current_stream(device).cuda_stream)
    return handle


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
        return _no(
            "cuDNN FROST head_dim>256 kernels are SM100/SM103 only; found sm%d%d"
            % torch.cuda.get_device_capability()
        )
    try:
        _import_cudnn()
    except ImportError as exc:
        return _no("nvidia-cudnn-frontend not importable: %s" % exc)

    # Decline on positive evidence that FROST cannot work: a version below a floor, or a package
    # that is absent outright. A version that is present but unparseable is NOT evidence, so it
    # defers to _select_frost_plan, which checks the plan by name and reports both versions.
    frontend, frontend_raw = _pkg_version("nvidia-cudnn-frontend", _cudnn)
    if frontend is not None and frontend < _MIN_CUDNN_FRONTEND:
        return _no(
            "nvidia-cudnn-frontend %s registers no sm100 backward engine; >= %s is required"
            " (1.28.0 ships the d512 forward only, so this would otherwise raise on the first"
            " backward rather than here)" % (frontend_raw, _MIN_CUDNN_FRONTEND)
        )

    cutlass, cutlass_raw = _pkg_version("nvidia-cutlass-dsl")
    if cutlass_raw is None:
        return _no("nvidia-cutlass-dsl not installed (FROST requires >= %s)" % _MIN_CUTLASS_DSL)
    if cutlass is not None and cutlass < _MIN_CUTLASS_DSL:
        # Worth being loud: this combination fails by silently declining, not by raising.
        return _no(
            "nvidia-cutlass-dsl %s is below the FROST floor %s; FROST engines would be"
            " silently skipped in favour of ordinary cuDNN backend plans"
            % (cutlass_raw, _MIN_CUTLASS_DSL)
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
            "FROST attention supports attn_mask_type in %s; got %r"
            % (str(_SUPPORTED_MASKS), attn_mask_type)
        )
    try:
        window = _NO_WINDOW if window_size is None else tuple(window_size)
    except TypeError:
        # Raised as NotImplementedError so the selector declines instead of propagating out of
        # backend selection, which is the only thing is_frost_attention_supported catches.
        raise NotImplementedError(
            "window_size must be a (left, right) pair; got %r" % (window_size,)
        ) from None
    if len(window) != 2:
        raise NotImplementedError("window_size must be a (left, right) pair; got %r" % (window,))
    if window[0] < -1:
        # cuDNN's left bound must be >= 1, so a left of -2 would build diagonal_band_left_bound=-1
        # and fail at plan build rather than declining here.
        raise NotImplementedError("window_size left must be -1 or >= 0; got %r" % (window,))
    if window[1] not in (-1, 0):
        # A right bound past the diagonal is future context. cuDNN can express it, but no TE mask
        # type asks for it, so decline rather than guess the intent.
        raise NotImplementedError("FROST attention does not support a right window %r" % (window,))
    return attn_mask_type, window


def _mask_options(cudnn, spec):
    """cuDNN sdpa kwargs for a (mask type, window) spec: a diagonal alignment plus a band."""
    attn_mask_type, window = spec
    left, right = window
    options = {}
    if attn_mask_type in ("causal", "causal_bottom_right") or right == 0:
        options["diagonal_alignment"] = (
            cudnn.diagonal_alignment.BOTTOM_RIGHT
            if attn_mask_type == "causal_bottom_right"
            else cudnn.diagonal_alignment.TOP_LEFT
        )
        options["diagonal_band_right_bound"] = 0
    if left != -1:
        # cuDNN counts the diagonal itself, TE does not, hence the +1 -- the same convention the
        # C++ fused path and the Python port both use.
        options["diagonal_band_left_bound"] = left + 1
    return options


def is_frost_attention_supported(
    head_dim_qk: int,
    head_dim_v: int,
    qkv_dtype: torch.dtype,
    attn_mask_type: str,
    dropout: float = 0.0,
    attn_bias_type: str = "no_bias",
    window_size: Optional[Tuple[int, int]] = None,
) -> Tuple[bool, str]:
    """Whether this specific attention configuration should route to FROST.

    Shape and dtype are checked before availability, and the ordering is deliberate rather than
    stylistic. Probing availability imports cuDNN Frontend and sets
    CUDNN_FRONTEND_ENABLE_FROST_ENGINES, which registers extra engines process-wide and so is
    visible to every other cuDNN consumer in the process. This function runs for every attention
    config on the machine, the vast majority of which are nowhere near head_dim 512, and none of
    them should pay that cost or have their engine pool changed underneath them.
    """
    if head_dim_qk != head_dim_v:
        return False, "FROST path requires symmetric head_dim; got %d/%d" % (
            head_dim_qk,
            head_dim_v,
        )
    if not _MIN_HEAD_DIM <= head_dim_qk <= _MAX_HEAD_DIM:
        return False, "FROST path covers head_dim in (256, 512]; got %d" % head_dim_qk
    if head_dim_qk % _HEAD_DIM_MULTIPLE != 0:
        return False, "FROST path needs head_dim to be a multiple of %d; got %d" % (
            _HEAD_DIM_MULTIPLE,
            head_dim_qk,
        )
    if qkv_dtype not in (torch.bfloat16, torch.float16):
        return False, "FROST path supports bf16/fp16; got %s" % qkv_dtype
    if dropout != 0.0:
        return False, "FROST path does not support dropout"
    if attn_bias_type != "no_bias":
        return False, "FROST path does not support attention bias"
    try:
        _mask_spec(attn_mask_type, window_size)
    except NotImplementedError as exc:
        return False, str(exc)
    ok, reason = is_frost_attention_available()
    if not ok:
        return False, reason
    return True, ""


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
        "FROST attention supports qkv_format 'bshd' and 'sbhd'; got %r."
        " thd needs varlen support that is not implemented here." % qkv_format
    )


def from_frost_layout(t: torch.Tensor, qkv_format: str) -> torch.Tensor:
    """Inverse of to_frost_layout."""
    if qkv_format == "bshd":  # [b, h, s, d] -> [b, s, h, d]
        return t.permute(0, 2, 1, 3)
    if qkv_format == "sbhd":  # [b, h, s, d] -> [s, b, h, d]
        return t.permute(2, 0, 1, 3)
    raise NotImplementedError(
        "FROST attention supports qkv_format 'bshd' and 'sbhd'; got %r." % qkv_format
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
        raise ValueError("%s must be 4D [b, h, s, d]; got %s" % (name, tuple(t.shape)))
    if t.stride(3) != 1:
        raise ValueError(
            "%s must have a contiguous head dimension; got shape %s stride %s"
            % (name, tuple(t.shape), tuple(t.stride()))
        )


def _check_dtype(name: str, t: torch.Tensor, expected: torch.dtype) -> None:
    """Require a tensor to carry the dtype its graph node was declared with.

    Every node but `stats` is declared from q's dtype, and execute() binds raw pointers, so a
    tensor of another dtype would have its bits reinterpreted with no error at all. `dout`
    matters most: it arrives from autograd and is not this module's to control.
    """
    if t.dtype != expected:
        raise ValueError("%s must be %s to match q; got %s" % (name, expected, t.dtype))


def _check_kv_match(k: torch.Tensor, v: torch.Tensor) -> None:
    """Require v to match k in both shape and layout.

    Both graphs declare v with k's shape and stride, and _key records only q's and k's, so a v
    that differs would hit a cached plan built for k's layout and read the wrong elements with no
    error at all. Callers in TE always split k and v from one QKV tensor, so this costs nothing
    and is purely a guard against a silent wrong answer.
    """
    if k.shape != v.shape:
        raise ValueError("k and v must have the same shape; got %s and %s" % (k.shape, v.shape))
    if k.stride() != v.stride():
        raise ValueError(
            "k and v must have the same layout; got strides %s and %s"
            % (tuple(k.stride()), tuple(v.stride()))
        )


def _select_frost_plan(graph, token: str, what: str):
    """Select a plan whose name proves a FROST engine was chosen.

    Falling back to whatever plan happens to be first would defeat the purpose: at these head
    dims the non-FROST plans do not exist, so an unnoticed fallback would either fail obscurely
    or quietly serve a different shape.
    """
    cudnn = _import_cudnn()
    graph.create_execution_plans([cudnn.heur_mode.A])
    names = [graph.get_plan_name_at_index(i) for i in range(graph.get_execution_plan_count())]
    hits = [i for i, n in enumerate(names) if token in n]
    if not hits:
        # Both versions, because either floor can cause this and blaming one misdirects. Looked
        # up defensively: this is the message explaining a failure, so it must not raise itself.
        raise RuntimeError(
            "no cuDNN FROST %s engine was offered (looked for %r). Candidate plans: %s."
            " nvidia-cudnn-frontend=%s (floor %s), nvidia-cutlass-dsl=%s (floor %s)."
            % (
                what,
                token,
                names[:6],
                _pkg_version("nvidia-cudnn-frontend", _cudnn)[1] or "unknown",
                _MIN_CUDNN_FRONTEND,
                _pkg_version("nvidia-cutlass-dsl")[1] or "unknown",
                _MIN_CUTLASS_DSL,
            )
        )
    graph.select_plan(hits[0])
    graph.check_support()
    graph.build_plans()
    return names[hits[0]]


def _build_fwd(key) -> dict:
    """Build (and JIT-compile) a forward graph. Expensive; always reached through the cache."""
    cudnn = _import_cudnn()
    # deterministic is unused here: it selects a backward algorithm. Callers pass False for the
    # forward so the two never split the forward cache.
    *_device, b, hq, hkv, sq, skv, d, dtype, mask, scale, qs, ks, _deterministic = key
    io_dt = _cudnn_dtype(dtype)
    shq, shkv = [b, hq, sq, d], [b, hkv, skv, d]

    graph = cudnn.pygraph(
        io_data_type=io_dt,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
        handle=_handle_for(_device_from_key(_device)),
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
    graph.validate()
    graph.build_operation_graph()
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

    graph = cudnn.pygraph(
        io_data_type=io_dt,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
        handle=_handle_for(_device_from_key(_device)),
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
    graph.validate()
    graph.build_operation_graph()
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
        raise ValueError(
            "k must match q in batch and head_dim; got q %s and k %s" % (q.shape, k.shape)
        )
    if q.shape[1] % k.shape[1] != 0:
        raise ValueError(
            "num_heads must be divisible by num_gqa_groups; got %d and %d"
            % (q.shape[1], k.shape[1])
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
        raise ValueError(
            "k must match q in batch and head_dim; got q %s and k %s" % (q.shape, k.shape)
        )
    if q.shape[1] % k.shape[1] != 0:
        raise ValueError(
            "num_heads must be divisible by num_gqa_groups; got %d and %d"
            % (q.shape[1], k.shape[1])
        )
    for name, tensor in (("out", out), ("dout", dout)):
        if tensor.shape != q.shape:
            raise ValueError(
                "%s must have q's shape; got %s and %s" % (name, tensor.shape, q.shape)
            )
    if softmax_lse.dtype != torch.float32:
        raise ValueError("softmax_lse must be fp32; got %s" % softmax_lse.dtype)
    if tuple(softmax_lse.shape[:3]) != tuple(q.shape[:3]):
        raise ValueError(
            "softmax_lse must be [b, h, s] matching q; got %s and %s"
            % (tuple(softmax_lse.shape), tuple(q.shape))
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
