# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""PyTorch Expert Parallelism (EP) API."""

from __future__ import annotations

import atexit
from typing import Optional

import torch
import torch.distributed as dist

import transformer_engine_torch as tex


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
        except Exception:
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
    """Per-microbatch EP layer state: routing handle + persistent payload slots.

    Owns the per-call ``handle_mem`` routing scratch and the payload buffers
    consumed by :func:`ep_dispatch` / :func:`ep_combine`. Allocate one
    EpBuffer per concurrently-in-flight call on the layer (one per PP-1F1B
    microbatch); sharing across overlapping calls clobbers tensors the
    earlier bwd still reads. Call ``record_stream`` from streams other than
    the allocation stream.

    Cross-rank payload slots are symm-mem-backed when ``ep_bootstrap`` was
    called with ``zero_copy=True`` (requires ``ep_group``); otherwise plain
    HBM.
    """

    __slots__ = (
        # routing
        "handle_mem",
        "top_k",
        "alignment",
        # layer config
        "max_tokens_per_rank",
        "recv_capacity_per_rank",
        "hidden_dim",
        "num_local_experts",
        "payload_dtype",
        "device",
        # payload slots
        "recv_tokens",
        "combine_in",
        "recv_topk_weights",
        "token_counts",
        "grad_tokens",
        "grad_topk_weights",
    )

    def __init__(
        self,
        top_k: int,
        max_tokens_per_rank: int,
        recv_capacity_per_rank: int,
        hidden_dim: int,
        num_local_experts: int,
        alignment: int = 0,
        ep_group: Optional[dist.ProcessGroup] = None,
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

        size_bytes = tex.ep_handle_mem_size(self.top_k, self.alignment)
        self.handle_mem = torch.empty(int(size_bytes), dtype=torch.uint8, device=device)

        recv_shape = (self.recv_capacity_per_rank, self.hidden_dim)
        send_shape = (self.max_tokens_per_rank, self.hidden_dim)
        zero_copy = bool(tex.ep_get_zero_copy())
        if zero_copy:
            if ep_group is None:
                raise ValueError("EpBuffer requires ep_group when ep_bootstrap(zero_copy=True).")
            self.recv_tokens = symm_mem_alloc(recv_shape, payload_dtype, ep_group, device=device)
            self.combine_in = symm_mem_alloc(recv_shape, payload_dtype, ep_group, device=device)
            self.recv_topk_weights = symm_mem_alloc(
                (self.recv_capacity_per_rank,), torch.float32, ep_group, device=device
            )
            self.grad_tokens = symm_mem_alloc(send_shape, payload_dtype, ep_group, device=device)
        else:
            self.recv_tokens = torch.empty(recv_shape, dtype=payload_dtype, device=device)
            self.combine_in = torch.empty(recv_shape, dtype=payload_dtype, device=device)
            self.recv_topk_weights = torch.empty(
                self.recv_capacity_per_rank, dtype=torch.float32, device=device
            )
            self.grad_tokens = torch.empty(send_shape, dtype=payload_dtype, device=device)
        # Per-rank scratch; never cross-rank, plain HBM regardless of mode.
        self.token_counts = torch.empty(self.num_local_experts, dtype=torch.int32, device=device)
        self.grad_topk_weights = torch.empty(
            (self.max_tokens_per_rank, self.top_k), dtype=torch.float32, device=device
        )

    @classmethod
    def from_external(
        cls,
        top_k: int,
        max_tokens_per_rank: int,
        recv_capacity_per_rank: int,
        hidden_dim: int,
        num_local_experts: int,
        *,
        recv_tokens: torch.Tensor,
        combine_in: torch.Tensor,
        recv_topk_weights: Optional[torch.Tensor] = None,
        grad_tokens: Optional[torch.Tensor] = None,
        token_counts: Optional[torch.Tensor] = None,
        grad_topk_weights: Optional[torch.Tensor] = None,
        alignment: int = 0,
        payload_dtype: torch.dtype = torch.bfloat16,
        device: Optional[torch.device] = None,
    ) -> "EpBuffer":
        """Construct from caller-allocated payload buffers.

        Useful for sharing a pre-allocated pool across layers/microbatches and
        for plugging in symm-mem-backed tensors. Caller-supplied slots are
        validated against the expected shape and dtype; ``None`` slots get a
        fresh HBM allocation. ``handle_mem`` is always allocated fresh.
        """
        if device is None:
            device = torch.device("cuda", torch.cuda.current_device())
        alignment = int(alignment)
        if alignment > 1 and (alignment & (alignment - 1)) != 0:
            raise ValueError(f"alignment must be 0, 1, or a power of two (got {alignment}).")
        recv_shape = (recv_capacity_per_rank, hidden_dim)
        send_shape = (max_tokens_per_rank, hidden_dim)
        topk_shape = (max_tokens_per_rank, top_k)
        recv_w_shape = (recv_capacity_per_rank,)
        counts_shape = (num_local_experts,)

        def _check(t: torch.Tensor, name: str, shape: tuple, dtype: torch.dtype) -> torch.Tensor:
            if tuple(t.shape) != shape:
                raise ValueError(f"{name} shape {tuple(t.shape)} != expected {shape}")
            if t.dtype != dtype:
                raise ValueError(f"{name} dtype {t.dtype} != expected {dtype}")
            return t

        inst = cls.__new__(cls)
        inst.top_k = int(top_k)
        inst.alignment = alignment
        inst.max_tokens_per_rank = int(max_tokens_per_rank)
        inst.recv_capacity_per_rank = int(recv_capacity_per_rank)
        inst.hidden_dim = int(hidden_dim)
        inst.num_local_experts = int(num_local_experts)
        inst.payload_dtype = payload_dtype
        inst.device = device

        size_bytes = tex.ep_handle_mem_size(inst.top_k, inst.alignment)
        inst.handle_mem = torch.empty(int(size_bytes), dtype=torch.uint8, device=device)

        inst.recv_tokens = _check(recv_tokens, "recv_tokens", recv_shape, payload_dtype)
        inst.combine_in = _check(combine_in, "combine_in", recv_shape, payload_dtype)
        inst.recv_topk_weights = (
            _check(recv_topk_weights, "recv_topk_weights", recv_w_shape, torch.float32)
            if recv_topk_weights is not None
            else torch.empty(recv_w_shape, dtype=torch.float32, device=device)
        )
        inst.grad_tokens = (
            _check(grad_tokens, "grad_tokens", send_shape, payload_dtype)
            if grad_tokens is not None
            else torch.empty(send_shape, dtype=payload_dtype, device=device)
        )
        inst.token_counts = (
            _check(token_counts, "token_counts", counts_shape, torch.int32)
            if token_counts is not None
            else torch.empty(counts_shape, dtype=torch.int32, device=device)
        )
        inst.grad_topk_weights = (
            _check(grad_topk_weights, "grad_topk_weights", topk_shape, torch.float32)
            if grad_topk_weights is not None
            else torch.empty(topk_shape, dtype=torch.float32, device=device)
        )
        return inst

    def record_stream(self, stream: torch.cuda.Stream) -> None:
        """Record stream as a user of all owned tensors so the caching allocator
        defers reclaim until stream has caught up."""
        for t in (
            self.handle_mem,
            self.recv_tokens,
            self.combine_in,
            self.recv_topk_weights,
            self.grad_tokens,
            self.token_counts,
            self.grad_topk_weights,
        ):
            t.record_stream(stream)


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
def _(*args, **kw):
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
def _(*args, **kw):
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
def _(*args, **kw):
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
def _(*args, **kw):
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
def _(*args, **kw):
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
    """Autograd-aware prepare + dispatch. Fwd/bwd share handle_mem and the
    EpBuffer slots; do not re-run ep_prepare between them and do not share
    EpBuffer with another in-flight call (see EpBuffer).
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        handle_mem: torch.Tensor,
        top_k: int,
        alignment: int,
        recv_tokens: torch.Tensor,
        recv_topk_weights: torch.Tensor,
        token_counts: torch.Tensor,
        grad_tokens_buf: torch.Tensor,
        grad_topk_weights_buf: torch.Tensor,
        topk_idx: torch.Tensor,
        tokens: torch.Tensor,
        topk_weights: torch.Tensor,
    ):
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
        ctx.handle_mem = handle_mem
        ctx.grad_tokens_buf = grad_tokens_buf
        ctx.grad_topk_weights_buf = grad_topk_weights_buf
        ctx.tokens_shape = tokens.shape
        ctx.tokens_dtype = tokens.dtype
        ctx.topk_weights_shape = topk_weights.shape
        ctx.topk_weights_dtype = topk_weights.dtype
        ctx.tokens_T_flat = tokens.numel() // tokens.shape[-1]
        ctx.topk_T_flat = topk_weights.numel() // topk_weights.shape[-1]
        ctx.recv_capacity = recv_tokens.shape[0]
        ctx.hidden_dim = tokens.shape[-1]
        ctx.mark_non_differentiable(token_counts)
        # Detach so the long-lived buffers aren't tracked as differentiable outputs;
        # autograd re-attaches grad_fn pointing back at this Function.
        return recv_tokens.detach(), recv_topk_weights.detach(), token_counts

    @staticmethod
    def backward(ctx, g_recv_tokens, g_recv_topk_weights, _g_token_counts):  # type: ignore[override]
        device = ctx.handle_mem.device
        if g_recv_tokens is None:
            g_recv_tokens = torch.zeros(
                ctx.recv_capacity, ctx.hidden_dim, dtype=ctx.tokens_dtype, device=device
            )
        if g_recv_topk_weights is None:
            g_recv_topk_weights = torch.zeros(ctx.recv_capacity, dtype=torch.float32, device=device)
        if not g_recv_tokens.is_contiguous():
            g_recv_tokens = g_recv_tokens.contiguous()
        if not g_recv_topk_weights.is_contiguous():
            g_recv_topk_weights = g_recv_topk_weights.contiguous()
        # Narrow the persistent slots to this call's flattened leading dim.
        grad_tokens = ctx.grad_tokens_buf.narrow(0, 0, ctx.tokens_T_flat)
        grad_topk_weights = ctx.grad_topk_weights_buf.narrow(0, 0, ctx.topk_T_flat)
        torch.ops.transformer_engine_ep.dispatch_bwd(
            ctx.handle_mem,
            g_recv_tokens,
            g_recv_topk_weights,
            grad_tokens,
            grad_topk_weights,
        )
        # Reshape back to the original input shape so autograd's grad slot matches.
        grad_tokens_out = grad_tokens.view(ctx.tokens_shape)
        grad_topk_weights_out = grad_topk_weights.view(ctx.topk_weights_shape)
        return (
            None,  # handle_mem
            None,  # top_k
            None,  # alignment
            None,  # recv_tokens
            None,  # recv_topk_weights
            None,  # token_counts
            None,  # grad_tokens_buf
            None,  # grad_topk_weights_buf
            None,  # topk_idx
            grad_tokens_out,
            grad_topk_weights_out,
        )


class _EpCombine(torch.autograd.Function):
    """Autograd-aware combine. combine_in is reused as grad_combine_in in bwd;
    fwd/bwd share handle_mem so don't re-run ep_prepare between them. Caller
    must pre-apply the topk weighting to expert_out.
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        handle_mem: torch.Tensor,
        combine_in: torch.Tensor,
        num_local_tokens: int,
        hidden_dim: int,
        expert_out: torch.Tensor,
    ):
        device = expert_out.device
        # Stage expert_out into the persistent combine_in slot (symm-mem-backed
        # in the zero-copy path); its storage is reused as grad_combine_in in bwd.
        combine_in.copy_(expert_out)
        result = torch.empty(num_local_tokens, hidden_dim, dtype=expert_out.dtype, device=device)
        torch.ops.transformer_engine_ep.combine(handle_mem, combine_in, result)
        ctx.handle_mem = handle_mem
        ctx.combine_in = combine_in  # reused as grad_combine_in in bwd
        return result

    @staticmethod
    def backward(ctx, g_result):  # type: ignore[override]
        grad_combine_in = ctx.combine_in
        if not g_result.is_contiguous():
            g_result = g_result.contiguous()
        torch.ops.transformer_engine_ep.combine_bwd(ctx.handle_mem, g_result, grad_combine_in)
        return (
            None,  # handle_mem
            None,  # combine_in
            None,  # num_local_tokens
            None,  # hidden_dim
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
):
    """Run prepare + dispatch with autograd. topk_idx must be int64.

    Returns (recv_tokens, recv_topk_weights, token_counts); views into the
    buffer's persistent slots — consume them before the next ep_dispatch on
    the same buffer or they get overwritten. token_counts is non-differentiable.
    """
    _require_bf16("tokens", tokens)
    return _EpDispatch.apply(
        buffer.handle_mem,
        buffer.top_k,
        buffer.alignment,
        buffer.recv_tokens,
        buffer.recv_topk_weights,
        buffer.token_counts,
        buffer.grad_tokens,
        buffer.grad_topk_weights,
        topk_idx,
        tokens,
        topk_weights,
    )


def ep_combine(
    buffer: EpBuffer,
    expert_out: torch.Tensor,
    *,
    num_local_tokens: Optional[int] = None,
):
    """Combine expert outputs back to the source rank, with autograd. The
    caller must pre-apply the topk weighting to expert_out.

    Result shape is (num_local_tokens, buffer.hidden_dim); defaults to
    buffer.max_tokens_per_rank rows.
    """
    _require_bf16("expert_out", expert_out)
    if num_local_tokens is None:
        num_local_tokens = buffer.max_tokens_per_rank
    return _EpCombine.apply(
        buffer.handle_mem,
        buffer.combine_in,
        num_local_tokens,
        buffer.hidden_dim,
        expert_out,
    )
