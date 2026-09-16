# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""cuDNN FROST attention backend for head_dim in (256, 512] on SM100/SM103.

Why this exists. Gemma-4 global layers use symmetric head_dim=512, and no backend TE can select
today serves both that head dim and context parallelism: FlashAttention 2/3 cap at 256, FA4 is
gated off at symmetric 512, the C++ cuDNN fused path caps at 256, and the unfused path supports
512 but cannot do CP. cuDNN Frontend 1.29.0 ships CuTe-DSL ("FROST") SDPA kernels that do serve
symmetric 512 forward and backward on Blackwell, reachable through the ordinary cuDNN graph API.
This module wraps them so TE, including its CP ring, can dispatch to them.

Three properties were measured on B200 before this was written, and each one constrains the code:

1. cuDNN's `use_causal_mask` is TOP-LEFT aligned and `use_causal_mask_bottom_right` is
   bottom-right; both were verified against references at SQ=1024/SKV=2048, where the two
   disagree by three orders of magnitude (1.6e-03 vs 3.5e+00). They coincide when SQ == SKV, so
   the distinction is invisible in square tests and decisive for all_gather, which trims KV.
   `_MASK_MODES` lists only spellings checked this way: sdpa() ignores unknown kwargs silently,
   so an unverified name would apply no mask at all and still run.

2. Plan building must be cached. Building a plan costs ~1972 ms the first time and ~12 ms once
   cuDNN has cached the JIT, against a ~0.129 ms execute. Even the cached rebuild is ~90x an
   execute, so a per-call build would make training build-bound. Hence `_PLAN_CACHE`.

3. The forward LSE is natural-log logsumexp in fp32, shaped [b, h, s, 1]. Squeezed to [b, h, s]
   it is exactly what the CP ring correction in context_parallel.py consumes (max err 1.8e-06 vs
   an fp64 reference), which is what makes ring attention over these kernels valid at all.

Numerics were validated against the criterion FlashAttention applies to itself, namely that the
kernel error must stay within 2x the error bf16 inputs alone produce: observed 0.21x to 0.62x
across square and rectangular, causal and non-causal shapes.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import torch

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
_MIN_CUTLASS_DSL = (4, 7, 0)

_SUPPORTED_ARCHS = ((10, 0), (10, 3))
_MAX_HEAD_DIM = 512
_MIN_HEAD_DIM = 257  # below this the existing cuDNN/flash backends already serve the shape

_cudnn = None
_availability: Optional[Tuple[bool, str]] = None
_PLAN_CACHE: dict = {}


def _import_cudnn():
    """Import cuDNN Frontend with FROST engines enabled, once."""
    global _cudnn
    if _cudnn is None:
        # Must be set before the import: the engines are registered at import time.
        os.environ.setdefault("CUDNN_FRONTEND_ENABLE_FROST_ENGINES", "1")
        import cudnn  # pylint: disable=import-outside-toplevel
        import cudnn.sdpa  # noqa: F401  pylint: disable=import-outside-toplevel,unused-import

        _cudnn = cudnn
    return _cudnn


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
    if torch.cuda.get_device_capability() not in _SUPPORTED_ARCHS:
        return _no(
            "cuDNN FROST head_dim>256 kernels are SM100/SM103 only; found sm%d%d"
            % torch.cuda.get_device_capability()
        )
    try:
        _import_cudnn()
    except ImportError as exc:
        return _no("nvidia-cudnn-frontend not importable: %s" % exc)

    from importlib.metadata import PackageNotFoundError, version

    try:
        raw = version("nvidia-cutlass-dsl")
    except PackageNotFoundError:
        return _no("nvidia-cutlass-dsl not installed (FROST requires >= 4.7.0)")
    try:
        parsed = tuple(int(p) for p in raw.split(".")[:3])
    except ValueError:
        parsed = (0, 0, 0)
    if parsed < _MIN_CUTLASS_DSL:
        # Worth being loud: this combination fails by silently declining, not by raising.
        return _no(
            "nvidia-cutlass-dsl %s is below the FROST floor 4.7.0; FROST engines would be"
            " silently skipped in favour of ordinary cuDNN backend plans" % raw
        )

    _availability = (True, "")
    return _availability


# cuDNN sdpa() kwargs per TE mask type.
#
# These exact spellings are behaviourally verified, which matters more than it sounds: sdpa()
# takes **kwargs and SILENTLY IGNORES names it does not recognise, so a typo here would apply no
# mask at all and still build and run. Do not add an entry without checking the output against a
# reference for that alignment.
#
# Both alignments are needed. The p2p ring produces square diagonal tiles (top-left and
# bottom-right coincide there), while all_gather trims KV and relies on bottom-right alignment,
# where the two differ completely.
_MASK_MODES = {
    "no_mask": {},
    "causal": {"use_causal_mask": True},
    "causal_bottom_right": {"use_causal_mask_bottom_right": True},
}


def _mask_mode(attn_mask_type: str) -> str:
    """Validate a TE mask type and return its key in _MASK_MODES.

    Anything not listed is rejected rather than approximated: the failure mode of guessing wrong
    is silent numerical corruption, not an exception.
    """
    if attn_mask_type in _MASK_MODES:
        return attn_mask_type
    raise NotImplementedError(
        "FROST attention supports attn_mask_type in %s; got %r. Padding variants need varlen"
        " support that is not implemented here." % (sorted(_MASK_MODES), attn_mask_type)
    )


def is_frost_attention_supported(
    head_dim_qk: int,
    head_dim_v: int,
    qkv_dtype: torch.dtype,
    attn_mask_type: str,
    dropout: float = 0.0,
    attn_bias_type: str = "no_bias",
) -> Tuple[bool, str]:
    """Whether this specific attention configuration should route to FROST."""
    ok, reason = is_frost_attention_available()
    if not ok:
        return False, reason
    if head_dim_qk != head_dim_v:
        return False, "FROST path requires symmetric head_dim; got %d/%d" % (
            head_dim_qk,
            head_dim_v,
        )
    if not _MIN_HEAD_DIM <= head_dim_qk <= _MAX_HEAD_DIM:
        return False, "FROST path covers head_dim in (256, 512]; got %d" % head_dim_qk
    if qkv_dtype not in (torch.bfloat16, torch.float16):
        return False, "FROST path supports bf16/fp16; got %s" % qkv_dtype
    if dropout != 0.0:
        return False, "FROST path does not support dropout"
    if attn_bias_type != "no_bias":
        return False, "FROST path does not support attention bias"
    try:
        _mask_mode(attn_mask_type)
    except NotImplementedError as exc:
        return False, str(exc)
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
        from importlib.metadata import version

        raise RuntimeError(
            "no cuDNN FROST %s engine was offered (looked for %r). Candidate plans: %s."
            " nvidia-cutlass-dsl=%s (FROST floor 4.7.0)."
            % (what, token, names[:6], version("nvidia-cutlass-dsl"))
        )
    graph.select_plan(hits[0])
    graph.check_support()
    graph.build_plans()
    return names[hits[0]]


def _build_fwd(key) -> dict:
    """Build (and JIT-compile) a forward graph. Expensive; always reached through the cache."""
    cudnn = _import_cudnn()
    b, hq, hkv, sq, skv, d, dtype, mask, scale, qs, ks = key
    io_dt = _cudnn_dtype(dtype)
    shq, shkv = [b, hq, sq, d], [b, hkv, skv, d]

    graph = cudnn.pygraph(
        io_data_type=io_dt,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
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
        **_MASK_MODES[mask],
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
    b, hq, hkv, sq, skv, d, dtype, mask, scale, qs, ks = key
    io_dt = _cudnn_dtype(dtype)
    shq, shkv = [b, hq, sq, d], [b, hkv, skv, d]

    graph = cudnn.pygraph(
        io_data_type=io_dt,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
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
        **_MASK_MODES[mask],
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
    """Plan cache. See module docstring: a build is ~15000x an execute, so this is required."""
    cache_key = (kind,) + key
    entry = _PLAN_CACHE.get(cache_key)
    if entry is None:
        entry = _build_fwd(key) if kind == "fwd" else _build_bwd(key)
        _PLAN_CACHE[cache_key] = entry
    return entry


def _key(q, k, mask, scale):
    return (
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
    )


def frost_attn_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_scale: Optional[float] = None,
    attn_mask_type: str = "causal",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Forward attention via cuDNN FROST.

    q, k, v are [b, h, s, d] views over BSHD-contiguous memory. GQA is supported directly
    (h_kv may differ from h_q) and SQ need not equal SKV, which is what lets a CP ring step
    use this. Returns (out, softmax_lse) with softmax_lse as [b, h, s] fp32 natural-log
    logsumexp, the layout and convention the CP ring correction expects.
    """
    for name, tensor in (("q", q), ("k", k), ("v", v)):
        _check_layout(name, tensor)
    if k.shape != v.shape:
        raise ValueError("k and v must have the same shape; got %s and %s" % (k.shape, v.shape))
    if q.shape[1] % k.shape[1] != 0:
        raise ValueError(
            "num_heads must be divisible by num_gqa_groups; got %d and %d"
            % (q.shape[1], k.shape[1])
        )

    mask = _mask_mode(attn_mask_type)
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
    entry["graph"].execute({tq: q, tk: k, tv: v, tout: out, tlse: lse}, workspace)
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
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Backward attention via cuDNN FROST. `softmax_lse` is [b, h, s] as returned by the forward."""
    for name, tensor in (("q", q), ("k", k), ("v", v), ("out", out), ("dout", dout)):
        _check_layout(name, tensor)

    mask = _mask_mode(attn_mask_type)
    scale = attn_scale if attn_scale is not None else q.shape[-1] ** -0.5
    entry = _cached("bwd", _key(q, k, mask, scale))
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
    )
    return dq, dk, dv
