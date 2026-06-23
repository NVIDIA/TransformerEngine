# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""PyTorch Expert Parallelism (EP) API."""

from __future__ import annotations

import atexit
import warnings
from typing import Optional

import torch
import torch.distributed as dist

import transformer_engine_torch as tex

from .cpu_offload import mark_not_offload


__all__ = [
    "EpBuffer",
    "ep_bootstrap",
    "ep_finalize",
    "ep_dispatch",
    "ep_combine",
    "symm_mem_alloc",
]


# Symmetric-memory buffer allocator
#
# Used for the symm-mem zero-copy IO path. Set ``ep_bootstrap(zero_copy=True)``
# to opt in; the C++ backend then operates the EP group in zero-copy mode.


def symm_mem_alloc(
    shape,
    dtype: torch.dtype,
    ep_group: dist.ProcessGroup,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Allocate and rendezvous a symm-mem buffer on ep_group. Collective on ep_group."""
    if device is None:
        device = torch.device("cuda", torch.cuda.current_device())
    try:
        from torch.distributed import _symmetric_memory as _symm_mem
    except ImportError as e:
        raise RuntimeError(
            "torch.distributed._symmetric_memory is unavailable; symm_mem_alloc "
            "requires PyTorch built with NCCL symm-mem support."
        ) from e
    if _symm_mem.get_backend(device) != "NCCL":
        _symm_mem.set_backend("NCCL")
    t = _symm_mem.empty(*shape, dtype=dtype, device=device)
    _symm_mem.rendezvous(t, group=ep_group.group_name)
    return t


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
            import warnings

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


def _atexit_finalize() -> None:
    """Best-effort teardown at interpreter shutdown; swallows errors."""
    global _BOOTSTRAPPED
    if _BOOTSTRAPPED:
        try:
            tex.ep_finalize()
        except Exception:  # pylint: disable=broad-exception-caught
            import traceback

            traceback.print_exc()
        finally:
            _BOOTSTRAPPED = False


def ep_bootstrap(
    ep_group: dist.ProcessGroup,
    num_experts: int,
    max_tokens_per_rank: int,
    recv_capacity_per_rank: int,
    hidden_dim: int,
    max_num_sms: int = 0,
    zero_copy: bool = False,
    max_token_dtype: torch.dtype = torch.bfloat16,
) -> None:
    """Initialize EP by borrowing ep_group's NCCL comm. Call once per process.

    max_token_dtype sets the widest token dtype this EP group will dispatch;
    it sizes NCCL EP staging buffers.

    ``zero_copy`` opts the EP group into the symm-mem zero-copy IO path; pass
    ``True`` only when payload tensors are allocated via ``symm_mem_alloc``.
    Defaults to ``False``.
    """
    global _BOOTSTRAPPED, _ATEXIT_REGISTERED
    if _BOOTSTRAPPED:
        raise RuntimeError("ep_bootstrap was already called in this process")
    if ep_group.size() < 2:
        raise ValueError(f"ep_bootstrap requires ep_group.size() >= 2 (got {ep_group.size()}).")
    _check_nccl_runtime_version()
    if zero_copy:
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
        int(recv_capacity_per_rank),
        int(hidden_dim),
        int(max_num_sms),
        max_token_dtype,
        bool(zero_copy),
    )
    _BOOTSTRAPPED = True
    if not _ATEXIT_REGISTERED:
        atexit.register(_atexit_finalize)
        _ATEXIT_REGISTERED = True


def ep_finalize() -> None:
    """Optional explicit EP teardown; idempotent.

    An atexit handler covers normal interpreter shutdown, so most users do not
    need to call this. Call it explicitly only before
    ``dist.destroy_process_group()``, since the borrowed NCCL comm becomes
    invalid once the PG is destroyed.
    """
    global _BOOTSTRAPPED
    if not _BOOTSTRAPPED:
        return
    try:
        tex.ep_finalize()
    finally:
        _BOOTSTRAPPED = False


# Buffer


class EpBuffer:
    """Per-microbatch EP layer state holding handle_mem and token_counts.
    Cross-rank payload buffers are caller-supplied to ep_dispatch and
    ep_combine; allocate via symm_mem_alloc in zero-copy mode.
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
        "token_counts",
        "zero_copy",
    )

    def __init__(
        self,
        top_k: int,
        max_tokens_per_rank: int,
        recv_capacity_per_rank: int,
        hidden_dim: int,
        num_local_experts: int,
        alignment: int = 0,
        payload_dtype: torch.dtype = torch.bfloat16,
        device: Optional[torch.device] = None,
    ) -> None:
        if device is None:
            device = torch.device("cuda", torch.cuda.current_device())
        alignment = int(alignment)
        if alignment > 1 and (alignment & (alignment - 1)) != 0:
            raise ValueError(f"alignment must be 0, 1, or a power of two (got {alignment}).")
        self.top_k = int(top_k)
        self.alignment = alignment
        self.max_tokens_per_rank = int(max_tokens_per_rank)
        self.recv_capacity_per_rank = int(recv_capacity_per_rank)
        self.hidden_dim = int(hidden_dim)
        self.num_local_experts = int(num_local_experts)
        self.payload_dtype = payload_dtype
        self.device = device
        self.zero_copy = bool(tex.ep_get_zero_copy())

        size_bytes = tex.ep_handle_mem_size(self.top_k, self.alignment)
        self.handle_mem = torch.empty(int(size_bytes), dtype=torch.uint8, device=device)
        self.token_counts = torch.empty(self.num_local_experts, dtype=torch.int32, device=device)
        # Persistent tensor; keep resident if activation CPU offloading is on.
        mark_not_offload(self.handle_mem)

    @classmethod
    def from_external(
        cls,
        top_k: int,
        max_tokens_per_rank: int,
        recv_capacity_per_rank: int,
        hidden_dim: int,
        num_local_experts: int,
        *,
        token_counts: Optional[torch.Tensor] = None,
        alignment: int = 0,
        payload_dtype: torch.dtype = torch.bfloat16,
        device: Optional[torch.device] = None,
    ) -> "EpBuffer":
        """Construct from a caller-allocated token_counts; handle_mem is always fresh."""
        if device is None:
            device = torch.device("cuda", torch.cuda.current_device())
        alignment = int(alignment)
        if alignment > 1 and (alignment & (alignment - 1)) != 0:
            raise ValueError(f"alignment must be 0, 1, or a power of two (got {alignment}).")
        counts_shape = (num_local_experts,)

        inst = cls.__new__(cls)
        inst.top_k = int(top_k)
        inst.alignment = alignment
        inst.max_tokens_per_rank = int(max_tokens_per_rank)
        inst.recv_capacity_per_rank = int(recv_capacity_per_rank)
        inst.hidden_dim = int(hidden_dim)
        inst.num_local_experts = int(num_local_experts)
        inst.payload_dtype = payload_dtype
        inst.device = device
        inst.zero_copy = bool(tex.ep_get_zero_copy())

        size_bytes = tex.ep_handle_mem_size(inst.top_k, inst.alignment)
        inst.handle_mem = torch.empty(int(size_bytes), dtype=torch.uint8, device=device)
        # Persistent tensor; keep resident if activation CPU offloading is on.
        mark_not_offload(inst.handle_mem)

        if token_counts is not None:
            if tuple(token_counts.shape) != counts_shape:
                raise ValueError(
                    f"token_counts shape {tuple(token_counts.shape)} != expected {counts_shape}"
                )
            if token_counts.dtype != torch.int32:
                raise ValueError(f"token_counts dtype {token_counts.dtype} != expected int32")
            inst.token_counts = token_counts
        else:
            inst.token_counts = torch.empty(counts_shape, dtype=torch.int32, device=device)
        return inst

    def record_stream(self, stream: torch.cuda.Stream) -> None:
        """Defer caching-allocator reclaim of owned tensors until stream catches up."""
        self.handle_mem.record_stream(stream)
        self.token_counts.record_stream(stream)


# torch.library custom ops (so they don't graph-break under torch.compile)

_LIB = "transformer_engine_ep"


@torch.library.custom_op(
    f"{_LIB}::prepare",
    mutates_args=("handle_mem", "token_counts"),
    device_types="cuda",
)
def _prepare_op(
    handle_mem: torch.Tensor,
    top_k: int,
    topk_idx: torch.Tensor,
    token_counts: torch.Tensor,
    alignment: int,
) -> None:
    tex.ep_prepare(handle_mem, topk_idx, token_counts, top_k, alignment)


@_prepare_op.register_fake
def _(*_args, **_kw):
    return None


@torch.library.custom_op(
    f"{_LIB}::dispatch",
    mutates_args=("recv_tokens", "recv_topk_weights"),
    device_types="cuda",
)
def _dispatch_op(
    handle_mem: torch.Tensor,
    topk_idx: torch.Tensor,
    tokens: torch.Tensor,
    topk_weights: torch.Tensor,
    recv_tokens: torch.Tensor,
    recv_topk_weights: torch.Tensor,
) -> None:
    tex.ep_dispatch(handle_mem, topk_idx, tokens, topk_weights, recv_tokens, recv_topk_weights)


@_dispatch_op.register_fake
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
    mutates_args=("grad_expert_out",),
    device_types="cuda",
)
def _combine_bwd_op(
    handle_mem: torch.Tensor,
    grad: torch.Tensor,
    grad_expert_out: torch.Tensor,
) -> None:
    tex.ep_combine_bwd(handle_mem, grad, grad_expert_out)


@_combine_bwd_op.register_fake
def _(*_args, **_kw):
    return None


# Non-autograd primitives


def ep_prepare(buffer: "EpBuffer", topk_idx: torch.Tensor) -> torch.Tensor:
    """AllGather the routing map; fills ``buffer.handle_mem`` and returns
    ``buffer.token_counts`` (int32, shape [num_local_experts]). topk_idx must
    be int64.
    """
    torch.ops.transformer_engine_ep.prepare(
        buffer.handle_mem, buffer.top_k, topk_idx, buffer.token_counts, buffer.alignment
    )
    return buffer.token_counts


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


class _EpDispatch(torch.autograd.Function):
    """Autograd prepare+dispatch; bwd uses user-supplied grad inputs as-is."""

    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        handle_mem: torch.Tensor,
        top_k: int,
        alignment: int,
        recv_tokens: torch.Tensor,
        recv_topk_weights: torch.Tensor,
        token_counts: torch.Tensor,
        topk_idx: torch.Tensor,
        tokens: torch.Tensor,
        topk_weights: torch.Tensor,
    ):
        """Prepare + dispatch fwd."""
        torch.ops.transformer_engine_ep.prepare(
            handle_mem, top_k, topk_idx, token_counts, alignment
        )
        torch.ops.transformer_engine_ep.dispatch(
            handle_mem,
            topk_idx,
            tokens,
            topk_weights,
            recv_tokens,
            recv_topk_weights,
        )
        ctx.save_for_backward(handle_mem)
        ctx.tokens_shape = tokens.shape
        ctx.tokens_dtype = tokens.dtype
        ctx.topk_weights_shape = topk_weights.shape
        ctx.tokens_T_flat = tokens.numel() // tokens.shape[-1]
        ctx.topk_T_flat = topk_weights.numel() // topk_weights.shape[-1]
        ctx.top_k = topk_weights.shape[-1]
        ctx.recv_capacity = recv_tokens.shape[0]
        ctx.hidden_dim = tokens.shape[-1]
        ctx.mark_non_differentiable(token_counts)
        # Detach so the long-lived buffers aren't tracked as differentiable outputs;
        # autograd re-attaches grad_fn pointing back at this Function.
        return recv_tokens.detach(), recv_topk_weights.detach(), token_counts

    @staticmethod
    def backward(ctx, g_recv_tokens, g_recv_topk_weights, _g_token_counts):  # type: ignore[override]
        """Dispatch bwd; normalizes grad-input layout, otherwise passes through."""
        (handle_mem,) = ctx.saved_tensors
        device = handle_mem.device
        g_recv_tokens = g_recv_tokens.contiguous()
        g_recv_topk_weights = g_recv_topk_weights.contiguous()
        grad_tokens = torch.empty(
            ctx.tokens_T_flat, ctx.hidden_dim, dtype=ctx.tokens_dtype, device=device
        )
        grad_topk_weights = torch.empty(
            ctx.topk_T_flat, ctx.top_k, dtype=torch.float32, device=device
        )
        torch.ops.transformer_engine_ep.dispatch_bwd(
            handle_mem,
            g_recv_tokens,
            g_recv_topk_weights,
            grad_tokens,
            grad_topk_weights,
        )
        return (
            None,  # handle_mem
            None,  # top_k
            None,  # alignment
            None,  # recv_tokens
            None,  # recv_topk_weights
            None,  # token_counts
            None,  # topk_idx
            grad_tokens.view(ctx.tokens_shape),
            grad_topk_weights.view(ctx.topk_weights_shape),
        )


class _EpCombine(torch.autograd.Function):
    """Autograd combine; bwd writes into grad_expert_out, or expert_out's storage if None."""

    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        handle_mem: torch.Tensor,
        num_local_tokens: int,
        hidden_dim: int,
        grad_expert_out: Optional[torch.Tensor],
        expert_out: torch.Tensor,
    ):
        """Combine fwd; stashes grad_expert_out (or expert_out) as the bwd output slot."""
        device = expert_out.device
        result = torch.empty(num_local_tokens, hidden_dim, dtype=expert_out.dtype, device=device)
        torch.ops.transformer_engine_ep.combine(handle_mem, expert_out, result)
        grad_combine_in = grad_expert_out if grad_expert_out is not None else expert_out
        # bwd write target reused across microbatches; keep resident under CPU offloading.
        mark_not_offload(grad_combine_in)
        ctx.save_for_backward(handle_mem, grad_combine_in)
        return result

    @staticmethod
    def backward(ctx, g_result):  # type: ignore[override]
        if not g_result.is_contiguous():
            g_result = g_result.contiguous()
        handle_mem, grad_combine_in = ctx.saved_tensors
        torch.ops.transformer_engine_ep.combine_bwd(handle_mem, g_result, grad_combine_in)
        return (
            None,  # handle_mem
            None,  # num_local_tokens
            None,  # hidden_dim
            None,  # grad_expert_out
            grad_combine_in,
        )


# Public high-level wrappers


# NCCL EP currently only supports bfloat16 payload tensors.
def _require_bf16(name: str, t: torch.Tensor) -> None:
    if t.dtype is not torch.bfloat16:
        raise NotImplementedError(
            f"NCCL EP currently supports only bfloat16 payloads; got {name}.dtype={t.dtype}."
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
    """Prepare + dispatch with autograd. topk_idx must be int64.

    recv_tokens / recv_topk_weights are used as-is if supplied, else allocated.
    Zero-copy mode requires both to be supplied and symm-mem-backed.
    Returns (recv_tokens, recv_topk_weights, token_counts); token_counts is non-diff.
    """
    _require_bf16("tokens", tokens)
    if topk_weights.dtype is not torch.float32:
        raise TypeError(
            f"topk_weights must be float32; got dtype={topk_weights.dtype}. "
            "Cast with topk_weights.float() before calling."
        )
    if buffer.zero_copy and (recv_tokens is None or recv_topk_weights is None):
        raise ValueError(
            "ep_dispatch: zero-copy mode requires caller-supplied recv_tokens and "
            "recv_topk_weights (allocate via symm_mem_alloc)."
        )
    if recv_tokens is None:
        recv_tokens = torch.empty(
            buffer.recv_capacity_per_rank,
            buffer.hidden_dim,
            dtype=buffer.payload_dtype,
            device=buffer.device,
        )
    if recv_topk_weights is None:
        recv_topk_weights = torch.empty(
            buffer.recv_capacity_per_rank, dtype=torch.float32, device=buffer.device
        )
    return _EpDispatch.apply(
        buffer.handle_mem,
        buffer.top_k,
        buffer.alignment,
        recv_tokens,
        recv_topk_weights,
        buffer.token_counts,
        topk_idx,
        tokens,
        topk_weights,
    )


def ep_combine(
    buffer: EpBuffer,
    expert_out: torch.Tensor,
    *,
    num_local_tokens: Optional[int] = None,
    grad_expert_out: Optional[torch.Tensor] = None,
):
    """Combine with autograd; caller pre-applies topk weighting.

    grad_expert_out is the slot the bwd writes into; if None, expert_out's storage is reused.
    Zero-copy mode requires both expert_out and grad_expert_out to be symm-mem-backed.
    Result shape is (num_local_tokens, buffer.hidden_dim); defaults to buffer.max_tokens_per_rank rows.
    """
    _require_bf16("expert_out", expert_out)
    if buffer.zero_copy and grad_expert_out is None:
        raise ValueError(
            "ep_combine: zero-copy mode requires caller-supplied grad_expert_out "
            "(allocate via symm_mem_alloc)."
        )
    if num_local_tokens is None:
        num_local_tokens = buffer.max_tokens_per_rank
    return _EpCombine.apply(
        buffer.handle_mem,
        num_local_tokens,
        buffer.hidden_dim,
        grad_expert_out,
        expert_out,
    )
