# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""PyTorch Expert Parallelism (EP) API."""

from __future__ import annotations

import atexit
import warnings
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

import torch
import torch.distributed as dist

import transformer_engine_torch as tex

from .cpu_offload import mark_not_offload
from .distributed import symm_mem_alloc, release_symm_mem_pool
from .quantized_tensor import QuantizedTensor

# Type-hint-only import; keeps the ``Recipe`` annotation without a runtime import of
# common.recipe (the concrete recipe classes are imported lazily where used).
if TYPE_CHECKING:
    from ..common.recipe import Recipe

__all__ = [
    "EpBuffer",
    "ep_bootstrap",
    "is_ep_bootstrapped",
    "ep_finalize",
    "ep_dispatch",
    "ep_combine",
    "symm_mem_alloc",
    "release_symm_mem_pool",
    "is_symm_backed",
]


# ``symm_mem_alloc`` (imported from .distributed) allocates the symm-mem buffers
# used by the zero-copy IO path. Set ``ep_bootstrap(zero_copy=True)`` to opt in;
# the C++ backend then operates the EP group in zero-copy mode.


# Bootstrap


# NCCL EP requires NCCL >= 2.30.4 (matches the C++ backend's runtime check).
_MIN_NCCL_VERSION = (2, 30, 4)


def _check_nccl_runtime_version() -> None:
    """Raise with a clear message if the loaded libnccl is too old for NCCL EP."""
    import ctypes

    try:
        lib = ctypes.CDLL("libnccl.so.2", mode=ctypes.RTLD_GLOBAL)
        v = ctypes.c_int(0)
        if lib.ncclGetVersion(ctypes.byref(v)) != 0:
            warnings.warn("ncclGetVersion failed; skipping NCCL EP version check.")
            return
    except OSError:  # libnccl not findable; let the C++ side error
        return
    n = v.value
    # NCCL packs as (major*10000 + minor*100 + patch) up to ~2.x; newer
    # builds use the same scheme. Decode defensively.
    major, minor, patch = n // 10000, (n // 100) % 100, n % 100
    if (major, minor, patch) < _MIN_NCCL_VERSION:
        min_str = ".".join(str(x) for x in _MIN_NCCL_VERSION)
        raise RuntimeError(
            f"NCCL EP requires NCCL >= {min_str}, found {major}.{minor}.{patch} at runtime. "
            "Set LD_LIBRARY_PATH to a newer libnccl.so before launching."
        )


_BOOTSTRAPPED = False
_ATEXIT_REGISTERED = False
# EP group captured at bootstrap; used by the zero-copy symm-mem pool allocator.
_EP_GROUP: Optional[dist.ProcessGroup] = None
# Eager-mode toggle captured at bootstrap (set when recv_capacity_per_rank is
# omitted); ep_dispatch reads it to size the recv outputs from the per-step
# recv-token total instead of a fixed recv_capacity_per_rank.
_EAGER = False


def _atexit_finalize() -> None:
    """Best-effort teardown at interpreter shutdown; swallows errors."""
    global _BOOTSTRAPPED, _EP_GROUP, _EAGER
    if _BOOTSTRAPPED:
        try:
            tex.ep_finalize()
        except Exception:  # pylint: disable=broad-exception-caught
            import traceback

            traceback.print_exc()
        finally:
            _BOOTSTRAPPED = False
            _EP_GROUP = None
            _EAGER = False


def ep_bootstrap(
    ep_group: dist.ProcessGroup,
    num_experts: int,
    max_tokens_per_rank: int,
    hidden_dim: int,
    num_topk: int,
    recv_capacity_per_rank: Optional[int] = None,
    max_num_sms: int = 0,
    zero_copy: bool = False,
    drop_on_overflow: bool = False,
    max_token_dtype: torch.dtype = torch.bfloat16,
) -> None:
    """Initialize EP by borrowing ep_group's NCCL comm. Call once per process.

    max_token_dtype sets the widest token dtype this EP group will dispatch;
    it sizes NCCL EP staging buffers.

    ``recv_capacity_per_rank`` bounds the tokens one rank receives per step and
    sizes the recv outputs. Omit it (``None``) for eager mode, which sizes recv
    outputs from the per-step recv total instead; eager needs a host sync each
    step and is not CUDA-graph capturable.

    ``zero_copy`` opts the EP group into the symm-mem zero-copy IO path; pass
    ``True`` only when payload tensors are allocated via ``symm_mem_alloc``.
    Requires ``recv_capacity_per_rank``. To capture a CUDA graph, supply
    persistent recv_tokens / grad_out buffers to dispatch/combine; the pool-based
    auto-allocation used when they are omitted is not CUDA-graph capturable.

    ``num_topk`` is the per-token top-k; it sizes NCCL EP internal buffers.

    ``drop_on_overflow`` drops tokens exceeding ``recv_capacity_per_rank`` instead
    of trapping. Requires ``recv_capacity_per_rank``.
    """
    global _BOOTSTRAPPED, _ATEXIT_REGISTERED, _EP_GROUP, _EAGER
    eager = recv_capacity_per_rank is None
    if _BOOTSTRAPPED:
        raise RuntimeError("ep_bootstrap was already called in this process")
    if ep_group.size() < 2:
        raise ValueError(f"ep_bootstrap requires ep_group.size() >= 2 (got {ep_group.size()}).")
    if num_topk < 1:
        raise ValueError(f"ep_bootstrap requires num_topk >= 1 (got {num_topk}).")
    if zero_copy and eager:
        raise ValueError("ep_bootstrap: zero_copy requires recv_capacity_per_rank")
    if drop_on_overflow and eager:
        raise ValueError("ep_bootstrap: drop_on_overflow requires recv_capacity_per_rank")
    _check_nccl_runtime_version()
    if zero_copy:
        if not tex.ep_zero_copy_supported():
            raise RuntimeError(
                "ep_bootstrap: zero_copy=True requires the Transformer Engine torch extension "
                "built with NCCL symm-mem support (torch >= 2.11 with USE_NCCL)."
            )
        warnings.warn(
            "ep_bootstrap(zero_copy=True) is experimental; the symm-mem IO path "
            "and its alias contracts on EpBuffer slots are subject to change.",
            stacklevel=2,
        )

    # Materialize the PG's NCCL comm before borrowing its raw handle.
    dist.barrier(group=ep_group, device_ids=[torch.cuda.current_device()])
    comm_ptr = ep_group._get_backend(torch.device("cuda"))._comm_ptr()

    tex.ep_initialize(
        int(comm_ptr),
        str(ep_group.group_name),
        int(num_experts),
        int(max_tokens_per_rank),
        # Eager mode (recv_capacity_per_rank=None) sizes recv buffers per routing,
        # so the group uses the library-derived bound (0 = NCCL_EP_AUTO).
        int(recv_capacity_per_rank or 0),
        int(hidden_dim),
        int(max_num_sms),
        max_token_dtype,
        bool(zero_copy),
        int(num_topk),
        bool(drop_on_overflow),
    )
    _BOOTSTRAPPED = True
    _EP_GROUP = ep_group
    _EAGER = bool(eager)
    if not _ATEXIT_REGISTERED:
        atexit.register(_atexit_finalize)
        _ATEXIT_REGISTERED = True


def is_ep_bootstrapped() -> bool:
    """Whether EP has been initialized in this process."""
    return _BOOTSTRAPPED


def ep_finalize() -> None:
    """Optional explicit EP teardown; idempotent.

    An atexit handler covers normal interpreter shutdown, so most users do not
    need to call this. Call it explicitly only before
    ``dist.destroy_process_group()``, since the borrowed NCCL comm becomes
    invalid once the PG is destroyed. This also releases the symm-mem pool, so
    a caller that used ``symm_mem_alloc(use_pool=True)`` does not need a separate
    ``release_symm_mem_pool()`` before destroying the PG.
    """
    global _BOOTSTRAPPED, _EP_GROUP, _EAGER
    if not _BOOTSTRAPPED:
        return
    try:
        # Deregister pooled symm-mem windows while the group's comm is still valid.
        release_symm_mem_pool()
        tex.ep_finalize()
    finally:
        _BOOTSTRAPPED = False
        _EP_GROUP = None
        _EAGER = False


def is_symm_backed(t: torch.Tensor) -> bool:
    """Whether ``t`` is symm-mem-backed on the EP group. Prefer torch's local ``is_symm_mem_tensor``
    when the build provides it (no collective, no exception); otherwise fall back to the rendezvous
    probe the C++ ep kernel uses (``maybe_make_window``): cached for an already-registered tensor,
    raises for a plain one."""
    from torch.distributed import _symmetric_memory as _symm

    if hasattr(_symm, "is_symm_mem_tensor"):
        return bool(_symm.is_symm_mem_tensor(t))
    if _EP_GROUP is None:
        raise RuntimeError(
            "is_symm_backed called before ensure_nccl_ep_bootstrapped(); no EP group registered."
        )
    try:
        _symm.rendezvous(t, _EP_GROUP.group_name)
        return True
    except Exception:  # pylint: disable=broad-exception-caught
        return False


# Buffer


class EpBuffer:
    """Per-microbatch EP layer state: handle_mem, tokens_per_expert, and shape/dtype config.
    Use one EpBuffer per concurrently-in-flight call (e.g. per PP-1F1B microbatch).
    """

    __slots__ = (
        "handle_mem",
        "top_k",
        "alignment",
        "max_tokens_per_rank",
        "recv_capacity_per_rank",
        "hidden_dim",
        "num_local_experts",
        "payload_dtype",
        "device",
        "tokens_per_expert",
        "zero_copy",
        "eager",
        "total_recv_tokens",
        "dispatch_fwd_quant_recipe",
        "combine_bwd_quant_recipe",
    )

    def __init__(
        self,
        top_k: int,
        max_tokens_per_rank: int,
        hidden_dim: int,
        num_local_experts: int,
        recv_capacity_per_rank: Optional[int] = None,
        alignment: int = 0,
        payload_dtype: torch.dtype = torch.bfloat16,
        device: Optional[torch.device] = None,
        dispatch_fwd_quant_recipe: Optional["Recipe"] = None,
        combine_bwd_quant_recipe: Optional["Recipe"] = None,
    ) -> None:
        if not _BOOTSTRAPPED:
            raise RuntimeError("EpBuffer requires ep_bootstrap() to be called first.")
        if device is None:
            device = torch.device("cuda", torch.cuda.current_device())
        alignment = int(alignment)
        if alignment > 1 and (alignment & (alignment - 1)) != 0:
            raise ValueError(f"alignment must be 0, 1, or a power of two (got {alignment}).")
        self.eager = _EAGER
        if not self.eager and recv_capacity_per_rank is None:
            raise ValueError(
                "EpBuffer requires recv_capacity_per_rank unless the EP group was "
                "bootstrapped in eager mode (recv_capacity_per_rank omitted)."
            )
        self.top_k = int(top_k)
        self.alignment = alignment
        self.max_tokens_per_rank = int(max_tokens_per_rank)
        self.recv_capacity_per_rank = (
            None if recv_capacity_per_rank is None else int(recv_capacity_per_rank)
        )
        self.hidden_dim = int(hidden_dim)
        self.num_local_experts = int(num_local_experts)
        self.payload_dtype = payload_dtype
        self.device = device
        self.zero_copy = bool(tex.ep_get_zero_copy())
        self.dispatch_fwd_quant_recipe = dispatch_fwd_quant_recipe
        self.combine_bwd_quant_recipe = combine_bwd_quant_recipe

        size_bytes = tex.ep_handle_mem_size(self.top_k, self.alignment)
        self.handle_mem = torch.empty(int(size_bytes), dtype=torch.uint8, device=device)
        self.tokens_per_expert = torch.empty(
            self.num_local_experts, dtype=torch.int64, device=device
        )
        # Persistent tensor; keep resident if activation CPU offloading is on.
        mark_not_offload(self.handle_mem)
        # Per-step recv-token total (int64 [1]), written by ep_prepare. Eager reads it
        # host-side to size the recv outputs, so it lives in pinned host memory the prepare
        # kernel writes directly (UVA) — no D2H copy, just a stream sync. Graph mode keeps
        # it on device for the backend's post-replay overflow check. The eager tensor is host
        # memory, so activation CPU offloading (which targets device tensors) never touches it
        # and no mark_not_offload guard is needed.
        if self.eager:
            self.total_recv_tokens = torch.empty(1, dtype=torch.int64, pin_memory=True)
        else:
            self.total_recv_tokens = torch.empty(1, dtype=torch.int64, device=device)
            mark_not_offload(self.total_recv_tokens)


# torch.library custom ops (so they don't graph-break under torch.compile)

_LIB = "transformer_engine_ep"


@torch.library.custom_op(
    f"{_LIB}::prepare",
    mutates_args=("handle_mem", "tokens_per_expert", "total_recv_tokens"),
    device_types="cuda",
)
def _prepare_op(
    handle_mem: torch.Tensor,
    top_k: int,
    topk_idx: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    alignment: int,
    total_recv_tokens: torch.Tensor,
) -> None:
    tex.ep_prepare(handle_mem, topk_idx, tokens_per_expert, top_k, alignment, total_recv_tokens)


@_prepare_op.register_fake
def _(*_args, **_kw):
    return None


@torch.library.custom_op(
    f"{_LIB}::dispatch",
    mutates_args=("recv_tokens", "recv_topk_weights", "recv_scale_inv"),
    device_types="cuda",
)
def _dispatch_op(
    handle_mem: torch.Tensor,
    topk_idx: torch.Tensor,
    tokens: torch.Tensor,
    topk_weights: torch.Tensor,
    recv_tokens: torch.Tensor,
    recv_topk_weights: torch.Tensor,
    tokens_scale_inv: Optional[torch.Tensor] = None,
    recv_scale_inv: Optional[torch.Tensor] = None,
) -> None:
    tex.ep_dispatch(
        handle_mem,
        topk_idx,
        tokens,
        topk_weights,
        recv_tokens,
        recv_topk_weights,
        tokens_scale_inv,
        recv_scale_inv,
    )


@_dispatch_op.register_fake
def _(*_args, **_kw):
    return None


@torch.library.custom_op(
    f"{_LIB}::prepare_and_dispatch",
    mutates_args=(
        "handle_mem",
        "tokens_per_expert",
        "total_recv_tokens",
        "recv_tokens",
        "recv_topk_weights",
        "recv_scale_inv",
    ),
    device_types="cuda",
)
def _prepare_and_dispatch_op(
    handle_mem: torch.Tensor,
    topk_idx: torch.Tensor,
    tokens: torch.Tensor,
    topk_weights: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    total_recv_tokens: torch.Tensor,
    top_k: int,
    alignment: int,
    recv_tokens: torch.Tensor,
    recv_topk_weights: torch.Tensor,
    recv_scale_inv: Optional[torch.Tensor] = None,
    tokens_scale_inv: Optional[torch.Tensor] = None,
) -> None:
    tex.ep_prepare_and_dispatch(
        handle_mem,
        topk_idx,
        tokens,
        topk_weights,
        tokens_per_expert,
        total_recv_tokens,
        top_k,
        alignment,
        recv_tokens,
        recv_topk_weights,
        recv_scale_inv,
        tokens_scale_inv,
    )


@_prepare_and_dispatch_op.register_fake
def _(*_args, **_kw):
    return None


@torch.library.custom_op(
    f"{_LIB}::combine",
    mutates_args=("result",),
    device_types="cuda",
)
def _combine_op(
    handle_mem: torch.Tensor,
    expert_out: torch.Tensor,
    result: torch.Tensor,
) -> None:
    tex.ep_combine(handle_mem, expert_out, result)


@_combine_op.register_fake
def _(*_args, **_kw):
    return None


@torch.library.custom_op(
    f"{_LIB}::dispatch_bwd",
    mutates_args=("grad_tokens", "grad_topk_weights"),
    device_types="cuda",
)
def _dispatch_bwd_op(
    handle_mem: torch.Tensor,
    grad: torch.Tensor,
    g_recv_topk_weights: torch.Tensor,
    grad_tokens: torch.Tensor,
    grad_topk_weights: torch.Tensor,
) -> None:
    tex.ep_dispatch_bwd(handle_mem, grad, g_recv_topk_weights, grad_tokens, grad_topk_weights)


@_dispatch_bwd_op.register_fake
def _(*_args, **_kw):
    return None


@torch.library.custom_op(
    f"{_LIB}::combine_bwd",
    mutates_args=("grad_expert_out", "grad_expert_out_scale_inv"),
    device_types="cuda",
)
def _combine_bwd_op(
    handle_mem: torch.Tensor,
    grad: torch.Tensor,
    grad_expert_out: torch.Tensor,
    grad_scale_inv: Optional[torch.Tensor] = None,
    grad_expert_out_scale_inv: Optional[torch.Tensor] = None,
) -> None:
    tex.ep_combine_bwd(handle_mem, grad, grad_expert_out, grad_scale_inv, grad_expert_out_scale_inv)


@_combine_bwd_op.register_fake
def _(*_args, **_kw):
    return None


# Non-autograd primitives


def ep_prepare(buffer: "EpBuffer", topk_idx: torch.Tensor) -> torch.Tensor:
    """AllGather the routing map; fills ``buffer.handle_mem`` and returns
    ``buffer.tokens_per_expert`` (int64, shape [num_local_experts]). topk_idx must
    be int32 or int64.

    Also fills ``buffer.total_recv_tokens`` (int64 [1]; pinned host in eager mode,
    device otherwise) with the per-step recv total; graph mode reads it device-side
    to detect overflow.
    """
    torch.ops.transformer_engine_ep.prepare(
        buffer.handle_mem,
        buffer.top_k,
        topk_idx,
        buffer.tokens_per_expert,
        buffer.alignment,
        buffer.total_recv_tokens,
    )
    return buffer.tokens_per_expert


def _ep_dispatch_raw(
    buffer: "EpBuffer",
    topk_idx: torch.Tensor,
    tokens: torch.Tensor,
    topk_weights: torch.Tensor,
    recv_tokens: torch.Tensor,
    recv_topk_weights: torch.Tensor,
) -> None:
    """Raw dispatch; no autograd, no prepare. Caller must run ep_prepare first."""
    tex.ep_dispatch(
        buffer.handle_mem, topk_idx, tokens, topk_weights, recv_tokens, recv_topk_weights
    )


def _ep_combine_raw(buffer: "EpBuffer", expert_out: torch.Tensor, result: torch.Tensor) -> None:
    """Raw combine; no autograd. Caller pre-weights expert_out."""
    tex.ep_combine(buffer.handle_mem, expert_out, result)


# autograd.Function wrappers


@dataclass(slots=True)
class _DispatchState:
    """Backward state + output-shaping flags for the prepare-and-dispatch op. ``handle_mem`` is
    buffer-owned and mutated in place, so it rides here as a plain attribute rather than through
    save_for_backward (which would version-track it). The autograd wrapper stashes this on ctx;
    other callers keep it however they manage backward state."""

    handle_mem: torch.Tensor
    tokens_shape: torch.Size
    topk_weights_shape: torch.Size
    tokens_T_flat: int
    topk_T_flat: int
    top_k: int
    hidden_dim: int
    eager: bool
    is_scaled: bool


def _ep_prepare_and_dispatch_fwd(
    tokens: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_idx: torch.Tensor,
    buffer: "EpBuffer",
    recv_tokens: Optional[torch.Tensor],
    recv_topk_weights: Optional[torch.Tensor],
    tokens_scale_inv: Optional[torch.Tensor],
):
    """Validate inputs, size/allocate the recv outputs, and run prepare+dispatch. Eager sizes recv
    from the host count; otherwise recv uses the buffer static recv capacity. Returns the recv
    output (a per-expert GroupedTensor for MXFP8, else the raw recv tokens), the recv topk weights,
    and a _DispatchState for backward. No autograd; the caller owns context handling."""
    handle_mem = buffer.handle_mem
    tokens_per_expert = buffer.tokens_per_expert
    total_recv_tokens = buffer.total_recv_tokens
    top_k = buffer.top_k
    alignment = buffer.alignment
    eager = buffer.eager
    num_recv_tokens = buffer.recv_capacity_per_rank
    payload_dtype = buffer.payload_dtype
    is_scaled = tokens_scale_inv is not None
    tokens_data = tokens._rowwise_data if isinstance(tokens, QuantizedTensor) else tokens
    assert tokens_data.dim() == 2, "EP dispatch tokens must be 2D [num_tokens, hidden]"
    hidden = tokens_data.shape[-1]
    if is_scaled and tokens._fp8_dtype != tex.DType.kFloat8E4M3:
        raise NotImplementedError("EP dispatch supports only E4M3 MXFP8 tokens for now.")
    # Reinterpret byte-backed FP8 data as the fp8 dtype so the backend sees a scaled tensor.
    dispatch_tokens = tokens_data.view(torch.float8_e4m3fn) if is_scaled else tokens_data

    recv_scale_inv = None
    if eager:
        # The C++ op sizes and allocates the recv outputs from the host recv-count and returns
        # them (called directly, not via torch.library: eager forbids graph capture).
        outs = tex.ep_prepare_and_dispatch(
            handle_mem,
            topk_idx,
            dispatch_tokens,
            topk_weights,
            tokens_per_expert,
            total_recv_tokens,
            top_k,
            alignment,
            tokens_scale_inv=tokens_scale_inv,
        )
        recv_tokens, recv_topk_weights = outs[0], outs[1]
        recv_scale_inv = outs[2] if is_scaled else None
    else:
        device = tokens_data.device
        zero_copy = buffer.zero_copy
        if is_scaled:
            # recv data + scales share one buffer (data then scales); carve or allocate here.
            recv_tokens, recv_scale_inv = _scale_alloc_io(
                recv_tokens,
                num_recv_tokens,
                hidden,
                tokens_scale_inv.shape[-1],
                tokens_data.dtype,
                tokens_scale_inv.dtype,
                device,
                zero_copy,
            )
        elif recv_tokens is None:
            recv_tokens = _alloc_io((num_recv_tokens, hidden), payload_dtype, device, zero_copy)
        if recv_topk_weights is None:
            recv_topk_weights = _alloc_io((num_recv_tokens,), torch.float32, device, zero_copy)
        dispatch_recv = recv_tokens.view(torch.float8_e4m3fn) if is_scaled else recv_tokens
        torch.ops.transformer_engine_ep.prepare_and_dispatch(
            handle_mem,
            topk_idx,
            dispatch_tokens,
            topk_weights,
            tokens_per_expert,
            total_recv_tokens,
            top_k,
            alignment,
            dispatch_recv,
            recv_topk_weights,
            recv_scale_inv,
            tokens_scale_inv,
        )
    state = _DispatchState(
        handle_mem=handle_mem,
        tokens_shape=tokens.shape,
        topk_weights_shape=topk_weights.shape,
        tokens_T_flat=tokens_data.shape[0],
        topk_T_flat=topk_weights.numel() // topk_weights.shape[-1],
        top_k=topk_weights.shape[-1],
        hidden_dim=hidden,
        eager=eager,
        is_scaled=is_scaled,
    )
    # For scaled inputs the expert-major recv data + scales are wrapped into a per-expert
    # GroupedTensor so downstream grouped GEMM and autograd see a proper quantized grouped tensor.
    if is_scaled:
        recv_out = _make_grouped_mxfp8(
            recv_tokens.view(tokens._rowwise_data.dtype),
            recv_scale_inv,
            tokens_per_expert,
            tokens._fp8_dtype,
            tokens.dtype,
        )
        return recv_out, recv_topk_weights, state
    return recv_tokens, recv_topk_weights, state


def _ep_dispatch_bwd(
    state: "_DispatchState",
    g_recv_tokens: torch.Tensor,
    g_recv_topk_weights: torch.Tensor,
):
    """Run dispatch_bwd and reshape the grads to the forward input layout. Returns
    ``(grad_tokens, grad_topk_weights)``. No autograd; the caller owns context handling."""
    handle_mem = state.handle_mem
    device = handle_mem.device
    g_recv_tokens = g_recv_tokens.contiguous()
    g_recv_topk_weights = g_recv_topk_weights.contiguous()
    # Dispatch grad follows the recv grad's (high-precision) dtype; the quantizer's STE
    # owns the fp8 boundary for scaled inputs.
    grad_tokens = torch.empty(
        state.tokens_T_flat, state.hidden_dim, dtype=g_recv_tokens.dtype, device=device
    )
    grad_topk_weights = torch.empty(
        state.topk_T_flat, state.top_k, dtype=torch.float32, device=device
    )
    # Eager is not graph-capturable, so call the backend op directly and skip torch.library.
    if state.eager:
        tex.ep_dispatch_bwd(
            handle_mem, g_recv_tokens, g_recv_topk_weights, grad_tokens, grad_topk_weights
        )
    else:
        torch.ops.transformer_engine_ep.dispatch_bwd(
            handle_mem,
            g_recv_tokens,
            g_recv_topk_weights,
            grad_tokens,
            grad_topk_weights,
        )
    return (
        grad_tokens.view(state.tokens_shape),
        grad_topk_weights.view(state.topk_weights_shape),
    )


class _EpPrepareAndDispatch(torch.autograd.Function):
    """Autograd prepare and dispatch grouped into one C++ op (two kernel launches, not a fused
    kernel). In eager mode the op sizes and allocates the recv outputs from the per-step host
    recv-count, so no Python runs between the count read and the dispatch launch; caller-supplied
    buffers and zero-copy are then forbidden. Otherwise the recv outputs are allocated here to the
    static recv capacity (caller-supplied or symm-mem-backed under zero-copy) and passed in. When
    ``tokens_scale_inv`` is set (MXFP8 for now), ``tokens`` is the quantized tensor kept as the
    autograd operand so grad reaches the pre-quant input, and recv is returned as a per-expert
    GroupedTensor. The compute lives in ``_ep_prepare_and_dispatch_fwd`` / ``_ep_dispatch_bwd``;
    this wrapper only bridges autograd context handling."""

    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        tokens: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_idx: torch.Tensor,
        buffer: "EpBuffer",
        recv_tokens: Optional[torch.Tensor] = None,
        recv_topk_weights: Optional[torch.Tensor] = None,
        tokens_scale_inv: Optional[torch.Tensor] = None,
    ):
        """Only tokens and topk_weights are differentiable, so the non-diff buffer tensors ride on
        the buffer object to keep the autograd operand list short."""
        recv_out, recv_topk_weights, state = _ep_prepare_and_dispatch_fwd(
            tokens, topk_weights, topk_idx, buffer, recv_tokens, recv_topk_weights, tokens_scale_inv
        )
        ctx.state = state
        # Detach so the long-lived buffers aren't tracked as differentiable outputs; autograd
        # re-attaches grad_fn pointing back at this Function. Eager recv tensors are freshly built
        # in C++ (requires_grad=False, no input alias), so skip detach and its Python->C++ hop. The
        # scaled recv is a freshly built GroupedTensor, so only its topk weights need detaching.
        if state.is_scaled:
            return recv_out, recv_topk_weights if state.eager else recv_topk_weights.detach()
        if state.eager:
            return recv_out, recv_topk_weights
        return recv_out.detach(), recv_topk_weights.detach()

    @staticmethod
    def backward(ctx, g_recv_tokens, g_recv_topk_weights):  # type: ignore[override]
        """Dispatch bwd: run dispatch_bwd and reshape grads to the fwd input layout."""
        grad_tokens, grad_topk_weights = _ep_dispatch_bwd(
            ctx.state, g_recv_tokens, g_recv_topk_weights
        )
        return (
            grad_tokens,
            grad_topk_weights,
            None,  # topk_idx
            None,  # buffer
            None,  # recv_tokens
            None,  # recv_topk_weights
            None,  # tokens_scale_inv
        )


@dataclass(slots=True)
class _CombineState:
    """Backward state for the combine op. ``grad_out`` and ``handle_mem`` ride here as plain
    attributes rather than through save_for_backward, which would version-track tensors we mutate.
    The autograd wrapper stashes this on ctx; other callers keep it however they manage state."""

    handle_mem: torch.Tensor
    grad_out: Optional[torch.Tensor]
    bwd_quant_recipe: object
    token_counts: torch.Tensor
    expert_out_shape: torch.Size
    expert_out_dtype: torch.dtype
    device: torch.device
    eager: bool
    zero_copy: bool


def _ep_combine_fwd(
    expert_out: torch.Tensor,
    grad_out: Optional[torch.Tensor],
    buffer: "EpBuffer",
    num_local_tokens: int,
    bwd_quant_recipe,
):
    """Run combine and return ``(result, _CombineState)``. Eager mode is not graph-capturable, so
    it calls the backend op directly and skips the torch.library dispatch. No autograd; the caller
    owns context handling."""
    handle_mem = buffer.handle_mem
    eager = buffer.eager
    device = expert_out.device
    result = torch.empty(num_local_tokens, buffer.hidden_dim, dtype=expert_out.dtype, device=device)
    if eager:
        tex.ep_combine(handle_mem, expert_out, result)
    else:
        torch.ops.transformer_engine_ep.combine(handle_mem, expert_out, result)
    state = _CombineState(
        handle_mem=handle_mem,
        grad_out=grad_out,
        bwd_quant_recipe=bwd_quant_recipe,
        token_counts=buffer.tokens_per_expert,
        expert_out_shape=expert_out.shape,
        expert_out_dtype=expert_out.dtype,
        device=device,
        eager=eager,
        zero_copy=buffer.zero_copy,
    )
    return result, state


def _ep_combine_bwd(state: "_CombineState", g_result: torch.Tensor):
    """Scatter the result-grad to expert positions and return the expert_out grad. High-precision
    sends the grad as-is; a quantized recipe (MXFP8 today) quantizes it and returns a per-expert
    GroupedTensor. No autograd; the caller owns context handling."""
    if not g_result.is_contiguous():
        g_result = g_result.contiguous()
    handle_mem = state.handle_mem

    if state.bwd_quant_recipe is None:
        grad_expert_out = state.grad_out
        if grad_expert_out is None:
            grad_expert_out = _alloc_io(
                state.expert_out_shape, state.expert_out_dtype, state.device, state.zero_copy
            )
        if state.eager:
            tex.ep_combine_bwd(handle_mem, g_result, grad_expert_out, None, None)
        else:
            torch.ops.transformer_engine_ep.combine_bwd(handle_mem, g_result, grad_expert_out)
    else:
        mx, g_scale_inv = _quantize_mxfp8(g_result)
        g_data = mx._rowwise_data
        recv_pr, hidden = state.expert_out_shape[0], state.expert_out_shape[-1]
        ge_data, ge_scale_inv = _scale_alloc_io(
            state.grad_out,
            recv_pr,
            hidden,
            g_scale_inv.shape[-1],
            g_data.dtype,
            g_scale_inv.dtype,
            state.device,
            state.zero_copy,
        )
        # The backend keys on the fp8 scaling mode; reinterpret the byte-backed data as fp8.
        g_data_fp8 = g_data.view(torch.float8_e4m3fn)
        ge_data_fp8 = ge_data.view(torch.float8_e4m3fn)
        if state.eager:
            tex.ep_combine_bwd(handle_mem, g_data_fp8, ge_data_fp8, g_scale_inv, ge_scale_inv)
        else:
            torch.ops.transformer_engine_ep.combine_bwd(
                handle_mem, g_data_fp8, ge_data_fp8, g_scale_inv, ge_scale_inv
            )
        grad_expert_out = _make_grouped_mxfp8(
            ge_data, ge_scale_inv, state.token_counts, mx._fp8_dtype, state.expert_out_dtype
        )
    return grad_expert_out


class _EpCombine(torch.autograd.Function):
    """Autograd combine; bwd scatters the expert_out grad into ``grad_out``. When the caller
    supplies it that buffer is used as-is; otherwise it is allocated in the backward from the
    symm-mem pool in zero-copy mode (one-sided target) or a plain tensor in normal mode (keeps
    allocation torch.compile / CUDA-graph safe and lets autograd own the grad's lifetime).

    Only ``expert_out`` is differentiable; the non-diff buffer tensors ride on ``buffer`` to keep
    the operand list short. The compute lives in ``_ep_combine_fwd`` / ``_ep_combine_bwd``; this
    wrapper only bridges autograd context handling."""

    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        expert_out: torch.Tensor,
        grad_out: Optional[torch.Tensor],
        buffer: "EpBuffer",
        num_local_tokens: int,
        bwd_quant_recipe=None,
    ):
        """Combine fwd; stashes the backward state on ctx. When ``bwd_quant_recipe`` is set, the
        backward sends the result-grad as MXFP8."""
        result, ctx.state = _ep_combine_fwd(
            expert_out, grad_out, buffer, num_local_tokens, bwd_quant_recipe
        )
        return result

    @staticmethod
    def backward(ctx, g_result):  # type: ignore[override]
        """Combine bwd; scatters the result-grad to expert positions."""
        grad_expert_out = _ep_combine_bwd(ctx.state, g_result)
        return (
            grad_expert_out,
            None,  # grad_out
            None,  # buffer
            None,  # num_local_tokens
            None,  # bwd_quant_recipe
        )


# Public high-level wrappers


# NCCL EP inputs are bfloat16; MXFP8 is applied internally via the buffer's dispatch_fwd_quant_recipe.
def _require_bf16(name: str, t: torch.Tensor) -> None:
    if t.dtype is not torch.bfloat16:
        raise NotImplementedError(
            "NCCL EP currently supports only bfloat16 or MXFP8 payloads; got"
            f" {name}.dtype={t.dtype}."
        )


def _alloc_io(shape, dtype: torch.dtype, device, zero_copy: bool) -> torch.Tensor:
    """Allocate a dispatch/combine IO tensor the caller did not supply: from the symm-mem pool in
    zero-copy mode (auto-registered segment, lifecycle managed by torch refcount), else plain.

    The zero-copy pool path is not CUDA-graph capturable; supply persistent recv_tokens / grad_out
    buffers to capture a graph."""
    if zero_copy:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "EP zero-copy pool allocation is not CUDA-graph capturable; supply persistent "
                "recv_tokens / grad_out buffers (allocated once via symm_mem_alloc) before capture."
            )
        t = symm_mem_alloc(shape, dtype, _EP_GROUP, device=device, use_pool=True)
        # symm-mem storage is non-resizable; exempt it from CPU activation offloading (which
        # releases via storage.resize_(0)). Matters for bf16 recv_tokens (the saved activation).
        mark_not_offload(t)
        return t
    return torch.empty(*shape, dtype=dtype, device=device)


def _quantize_mxfp8(x: torch.Tensor):
    """Quantize a high-precision tensor to MXFP8 and return ``(quantized_tensor, scale_inv)`` where
    ``scale_inv`` is the compact ``[T, H/block]`` scale the EP backend routes. The quantized tensor
    is returned so callers can keep it as the autograd operand; its ``_rowwise_data`` is the fp8
    payload and ``scale_inv.shape[-1]`` the scale-column count. EP routes and returns E4M3 data in
    both directions, so quantize to E4M3 regardless of pass. Strips the GEMM scale row padding to
    the compact ``[T, H/block]`` layout; requires a 16-byte-aligned scale row."""
    from .constants import MXFP8_BLOCK_SCALING_SIZE
    from .tensor.mxfp8_tensor import MXFP8Quantizer

    mx = MXFP8Quantizer(tex.DType.kFloat8E4M3, rowwise=True, columnwise=False).quantize(x)
    if mx._with_gemm_swizzled_scales:
        raise RuntimeError(
            "internal MXFP8 quantization produced swizzled scales; EP dispatch needs compact."
        )
    data = mx._rowwise_data
    scale_inv = mx._rowwise_scale_inv
    if data is None or scale_inv is None:
        raise ValueError("MXFP8 tokens must carry rowwise data and scale_inv for EP dispatch.")
    t_flat = x.shape[0]
    hidden = x.shape[-1]
    cols = hidden // MXFP8_BLOCK_SCALING_SIZE
    # The backend forwards each token's scale row with a 16-byte-aligned store, so the row
    # (cols * dtype bytes) must be a multiple of 16.
    scale_row_bytes = cols * scale_inv.element_size()
    if scale_row_bytes % 16 != 0:
        raise ValueError(
            f"MXFP8 dispatch requires a 16-byte-aligned scale row; hidden={hidden} gives "
            f"{scale_row_bytes} bytes. Use a hidden size that is a multiple of "
            f"{16 * MXFP8_BLOCK_SCALING_SIZE}."
        )
    # scale_inv is 2D [round_up(T, 128), cols]; drop the row padding to the logical [T, H/block]
    # the backend expects. cols is a multiple of 4 (16-byte row), so no column padding and the
    # slice stays contiguous; assert rather than force a copy.
    scale_inv = scale_inv[:t_flat, :cols]
    if not scale_inv.is_contiguous():
        raise ValueError(
            "MXFP8 dispatch requires compact contiguous scales [T, H/block]; got a "
            f"non-contiguous [{t_flat}, {cols}] slice."
        )
    return mx, scale_inv


def _scale_alloc_io(buf, rows, data_cols, scale_cols, data_dtype, scale_dtype, device, zero_copy):
    """Block-scaled output data + scale buffers, each ``rows`` tall, laid out back-to-back
    (``[rows, data_cols]`` data of ``data_dtype`` then ``[rows, scale_cols]`` scales of
    ``scale_dtype``). Carve both from a single caller ``buf`` when it is large enough, so one
    symm-mem window backs both views; else allocate them (symm-mem pool under zero-copy, else
    plain). Recipe-agnostic: byte sizes come from the element sizes."""
    data_bytes = rows * data_cols * data_dtype.itemsize
    scale_bytes = rows * scale_cols * scale_dtype.itemsize
    if buf is not None:
        # Reinterpret in place; a non-contiguous buf would force a copy and leave the caller's
        # buffer unwritten, so require contiguous and view rather than reshape.
        if not buf.is_contiguous():
            raise ValueError("scaled output buffer must be contiguous.")
        flat = buf.view(-1).view(torch.uint8)
        if flat.numel() < data_bytes + scale_bytes:
            raise ValueError(
                f"scaled output buffer too small: need {data_bytes + scale_bytes} bytes "
                f"(data + scales), got {flat.numel()}."
            )
        data = flat[:data_bytes].view(data_dtype).reshape(rows, data_cols)
        scale_inv = (
            flat[data_bytes : data_bytes + scale_bytes].view(scale_dtype).reshape(rows, scale_cols)
        )
        return data, scale_inv
    data = _alloc_io((rows, data_cols), data_dtype, device, zero_copy)
    scale_inv = _alloc_io((rows, scale_cols), scale_dtype, device, zero_copy)
    return data, scale_inv


def _make_grouped_mxfp8(data, scale_inv, token_counts, fp8_dtype, fake_dtype):
    """Wrap expert-major MXFP8 recv data + compact e8m0 scales as a per-expert ``GroupedTensor``.

    ``token_counts`` (int64 [num_local_experts]) is the padded per-expert row counts (128-aligned),
    used as the group sizes. Grouping is device-side (first_dims/tensor_offsets), so the counts never
    sync to host; the outer shape is the static recv capacity, bounded per expert by first_dims.
    """
    from .tensor.grouped_tensor import GroupedTensor
    from .tensor.mxfp8_tensor import MXFP8Quantizer

    assert data.dim() == 2, "recv data must be 2D [capacity_rows, hidden]"
    capacity_rows, hidden = data.shape
    quantizer = MXFP8Quantizer(fp8_dtype, rowwise=True, columnwise=False)
    return GroupedTensor(
        shape=(capacity_rows, hidden),
        dtype=fake_dtype,
        num_tensors=token_counts.numel(),
        quantizer=quantizer,
        data=data.reshape(-1).detach(),
        scale_inv=scale_inv.reshape(-1).detach(),
        first_dims=token_counts,
        tensor_offsets=tex.splits_to_offsets(token_counts, hidden),
    )


def ep_dispatch(
    buffer: EpBuffer,
    tokens: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    recv_tokens: Optional[torch.Tensor] = None,
    recv_topk_weights: Optional[torch.Tensor] = None,
):
    """Prepare + dispatch with autograd. ``tokens`` is bfloat16; ``topk_idx`` is int32 or int64.

    When the buffer's ``dispatch_fwd_quant_recipe`` is set (``MXFP8BlockScaling`` only for now), tokens
    are quantized internally and recv is returned as a per-expert ``GroupedTensor``; otherwise recv
    stays bfloat16. A pre-quantized ``tokens`` is not accepted.

    ``recv_tokens`` / ``recv_topk_weights`` are the recv outputs: pass caller-owned buffers
    (symm-mem-backed under zero-copy) or leave them None to allocate. For MXFP8 the recv data and
    scales share ``recv_tokens`` (data then scales), so size it to at least
    ``recv_capacity_per_rank * (hidden + hidden/block)`` bytes. Eager mode sizes the recv outputs
    per step and forbids caller-supplied buffers. Under zero-copy, leaving them None allocates from
    the symm-mem pool, which is not CUDA-graph capturable; pass persistent buffers to capture a graph.

    Returns (recv_tokens, recv_topk_weights, tokens_per_expert); tokens_per_expert is non-diff. See
    ``buffer.total_recv_tokens`` for the per-step recv total.
    """
    if topk_weights.dtype is not torch.float32:
        raise TypeError(
            f"topk_weights must be float32; got dtype={topk_weights.dtype}. "
            "Cast with topk_weights.float() before calling."
        )
    if isinstance(tokens, QuantizedTensor):
        raise NotImplementedError(
            "NCCL EP dispatch takes a bfloat16 input and quantizes internally when the buffer's "
            "dispatch_fwd_quant_recipe is set; a pre-quantized tensor is not accepted."
        )
    _require_bf16("tokens", tokens)
    if buffer.eager and (recv_tokens is not None or recv_topk_weights is not None):
        raise ValueError(
            "eager mode sizes the recv outputs from the per-step recv-token total "
            "and cannot use caller-supplied recv_tokens / recv_topk_weights"
        )

    # Quantize up front (before prepare) so the quant kernels overlap the eager count sync and the
    # quantized tensor stays the autograd operand; grad reaches the pre-quant input.
    tokens_scale_inv = None
    if buffer.dispatch_fwd_quant_recipe is not None:
        from ..common.recipe import MXFP8BlockScaling

        if not isinstance(buffer.dispatch_fwd_quant_recipe, MXFP8BlockScaling):
            raise NotImplementedError(
                "EP block-scaled dispatch supports MXFP8BlockScaling only; got "
                f"{type(buffer.dispatch_fwd_quant_recipe).__name__}."
            )
        tokens, tokens_scale_inv = _quantize_mxfp8(tokens)

    # Fused prepare + dispatch in one C++ op. Eager sizes the recv outputs from the per-step host
    # recv-count (allocated in C++, no caller buffers); non-eager sizes them to the static recv
    # capacity here (caller-supplied or symm-mem-backed under zero-copy) and passes them in.
    recv_out, recv_topk_weights = _EpPrepareAndDispatch.apply(
        tokens,
        topk_weights,
        topk_idx,
        buffer,
        recv_tokens,
        recv_topk_weights,
        tokens_scale_inv,
    )
    return recv_out, recv_topk_weights, buffer.tokens_per_expert


def ep_combine(
    buffer: EpBuffer,
    expert_out: torch.Tensor,
    *,
    num_local_tokens: Optional[int] = None,
    grad_out: Optional[torch.Tensor] = None,
):
    """Combine with autograd; caller pre-applies topk weighting.

    ``expert_out`` is the combine input (symm-mem-backed under zero-copy). ``grad_out`` is the
    backward's grad target: pass a caller-owned buffer or leave it None to allocate. For MXFP8 the
    grad data and scales share ``grad_out`` (data then scales), so size it to at least
    ``recv_capacity_per_rank * (hidden + hidden/block)`` bytes (non-zero-copy only). Eager mode sizes
    the grad target per step and forbids a caller-supplied buffer. Under zero-copy, leaving it None
    allocates from the symm-mem pool, which is not CUDA-graph capturable; pass a persistent buffer to
    capture a graph. Result shape is (num_local_tokens, hidden_dim); num_local_tokens defaults to
    buffer.max_tokens_per_rank.
    """
    _require_bf16("expert_out", expert_out)
    if buffer.eager and grad_out is not None:
        raise ValueError(
            "eager mode sizes the combine grad target per step and cannot use a "
            "caller-supplied grad_out"
        )
    if num_local_tokens is None:
        num_local_tokens = buffer.max_tokens_per_rank
    # When combine_bwd_quant_recipe is set the combine backward sends the result-grad over the
    # wire as MXFP8 and returns the expert_out grad as a GroupedTensor.
    bwd_quant_recipe = None
    if buffer.combine_bwd_quant_recipe is not None:
        from ..common.recipe import MXFP8BlockScaling

        if not isinstance(buffer.combine_bwd_quant_recipe, MXFP8BlockScaling):
            raise NotImplementedError(
                "EP combine backward supports MXFP8BlockScaling only; got "
                f"{type(buffer.combine_bwd_quant_recipe).__name__}."
            )
        bwd_quant_recipe = buffer.combine_bwd_quant_recipe
    return _EpCombine.apply(
        expert_out,
        grad_out,
        buffer,
        num_local_tokens,
        bwd_quant_recipe,
    )
