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
from .distributed import symm_mem_alloc


__all__ = [
    "EpBuffer",
    "ep_bootstrap",
    "ep_finalize",
    "ep_dispatch",
    "ep_combine",
    "symm_mem_alloc",
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
# Eager-mode toggle captured at bootstrap; ep_dispatch reads it to size the recv
# outputs from the per-step recv-token total instead of recv_capacity_per_rank.
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
    recv_capacity_per_rank: int,
    hidden_dim: int,
    max_num_sms: int = 0,
    zero_copy: bool = False,
    eager: bool = False,
    max_num_topk: int = 0,
    drop_on_overflow: bool = False,
    max_token_dtype: torch.dtype = torch.bfloat16,
) -> None:
    """Initialize EP by borrowing ep_group's NCCL comm. Call once per process.

    max_token_dtype sets the widest token dtype this EP group will dispatch;
    it sizes NCCL EP staging buffers.

    ``zero_copy`` opts the EP group into the symm-mem zero-copy IO path; pass
    ``True`` only when payload tensors are allocated via ``symm_mem_alloc``.
    Defaults to ``False``.

    ``eager`` sizes the dispatch recv outputs from the per-step recv-token total
    reported by ep_prepare, instead of the fixed ``recv_capacity_per_rank`` upper
    bound. This requires a host sync each step, so it is not CUDA-graph
    capturable. Mutually exclusive with ``zero_copy``. Defaults to ``False``.

    ``max_num_topk`` is the upper bound on per-token top-k; it sizes NCCL EP
    internal buffers and is required (>= 1) for ``eager`` mode.

    ``drop_on_overflow`` drops tokens that exceed ``recv_capacity_per_rank``
    instead of trapping. Not supported in ``eager`` mode. Defaults to ``False``.
    """
    global _BOOTSTRAPPED, _ATEXIT_REGISTERED, _EP_GROUP, _EAGER
    if _BOOTSTRAPPED:
        raise RuntimeError("ep_bootstrap was already called in this process")
    if ep_group.size() < 2:
        raise ValueError(f"ep_bootstrap requires ep_group.size() >= 2 (got {ep_group.size()}).")
    if zero_copy and eager:
        raise ValueError("ep_bootstrap: zero_copy and eager modes are mutually exclusive")
    if eager and max_num_topk < 1:
        raise ValueError("ep_bootstrap: eager mode requires max_num_topk >= 1")
    if eager and drop_on_overflow:
        raise ValueError("ep_bootstrap: drop_on_overflow is not supported in eager mode")
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
        # Eager mode sizes recv buffers per routing, so the group uses the
        # library-derived bound (0 = NCCL_EP_AUTO) instead of a fixed budget.
        0 if eager else int(recv_capacity_per_rank),
        int(hidden_dim),
        int(max_num_sms),
        max_token_dtype,
        bool(zero_copy),
        int(max_num_topk),
        bool(drop_on_overflow),
    )
    _BOOTSTRAPPED = True
    _EP_GROUP = ep_group
    _EAGER = bool(eager)
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
    global _BOOTSTRAPPED, _EP_GROUP, _EAGER
    if not _BOOTSTRAPPED:
        return
    try:
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
        "_host_total_recv_tokens",
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
        self.eager = _EAGER

        size_bytes = tex.ep_handle_mem_size(self.top_k, self.alignment)
        self.handle_mem = torch.empty(int(size_bytes), dtype=torch.uint8, device=device)
        self.tokens_per_expert = torch.empty(
            self.num_local_experts, dtype=torch.int64, device=device
        )
        # Persistent tensor; keep resident if activation CPU offloading is on.
        mark_not_offload(self.handle_mem)
        # Per-step recv-token total (device int64 [1]), written by ep_prepare.
        # The true pre-drop count: under drop_on_overflow it includes dropped
        # tokens and may exceed recv_capacity_per_rank. Eager mode syncs it to
        # size the recv outputs; graph mode reads it after replay to check overflow.
        self.total_recv_tokens = torch.empty(1, dtype=torch.int64, device=device)
        mark_not_offload(self.total_recv_tokens)
        # Host mirror of total_recv_tokens; set by ep_prepare in eager mode (one
        # D2H) to size the recv outputs. None otherwise.
        self._host_total_recv_tokens: Optional[int] = None


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
    ``buffer.tokens_per_expert`` (int64, shape [num_local_experts]). topk_idx must
    be int32 or int64.

    Always fills ``buffer.total_recv_tokens`` (device int64 [1]) with the
    per-step recv-token total. In eager mode its host value is cached in
    ``buffer._host_total_recv_tokens`` (one D2H) to size the recv outputs;
    otherwise it stays device-side for the caller to read after a graph replay
    and compare against ``recv_capacity_per_rank`` to detect overflow.
    """
    torch.ops.transformer_engine_ep.prepare(
        buffer.handle_mem,
        buffer.top_k,
        topk_idx,
        buffer.tokens_per_expert,
        buffer.alignment,
        buffer.total_recv_tokens,
    )
    if buffer.eager:
        buffer._host_total_recv_tokens = int(buffer.total_recv_tokens.item())
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
        tokens_per_expert: torch.Tensor,
        total_recv_tokens: torch.Tensor,
        topk_idx: torch.Tensor,
        tokens: torch.Tensor,
        topk_weights: torch.Tensor,
        skip_prepare: bool = False,
    ):
        """Prepare + dispatch fwd. In eager mode the caller runs prepare first
        (to size recv outputs from the recv-token total), so ``skip_prepare``
        avoids re-running the routing AllGather here."""
        if not skip_prepare:
            torch.ops.transformer_engine_ep.prepare(
                handle_mem, top_k, topk_idx, tokens_per_expert, alignment, total_recv_tokens
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
        ctx.mark_non_differentiable(tokens_per_expert)
        # Detach so the long-lived buffers aren't tracked as differentiable outputs;
        # autograd re-attaches grad_fn pointing back at this Function.
        return recv_tokens.detach(), recv_topk_weights.detach(), tokens_per_expert

    @staticmethod
    def backward(ctx, g_recv_tokens, g_recv_topk_weights, _g_tokens_per_expert):  # type: ignore[override]
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
            None,  # tokens_per_expert
            None,  # total_recv_tokens
            None,  # topk_idx
            grad_tokens.view(ctx.tokens_shape),
            grad_topk_weights.view(ctx.topk_weights_shape),
            None,  # skip_prepare
        )


class _EpCombine(torch.autograd.Function):
    """Autograd combine.

    bwd scatters the expert_out grad into ``grad_out``. When the caller supplies it
    (mcore-managed mode) that buffer is used as-is; otherwise it is allocated on the
    fly here — from the symm-mem pool in zero-copy mode (one-sided target), or a plain
    tensor in normal mode (keeps allocation torch.compile / CUDA-graph safe and lets
    autograd own the grad's lifetime).

    ``grad_out`` is the backward's scatter target (an output it writes, never reads),
    so it is stashed as a plain ctx attribute rather than via save_for_backward, which
    would version-track a tensor we mutate.
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        handle_mem: torch.Tensor,
        num_local_tokens: int,
        hidden_dim: int,
        grad_out: Optional[torch.Tensor],
        expert_out: torch.Tensor,
    ):
        """Combine fwd; stashes the bwd grad target or expert_out shape to size it."""
        device = expert_out.device
        result = torch.empty(num_local_tokens, hidden_dim, dtype=expert_out.dtype, device=device)
        torch.ops.transformer_engine_ep.combine(handle_mem, expert_out, result)
        ctx.save_for_backward(handle_mem)
        ctx.grad_out = grad_out
        if grad_out is None:
            ctx.expert_out_shape = expert_out.shape
            ctx.expert_out_dtype = expert_out.dtype
            ctx.device = device
        return result

    @staticmethod
    def backward(ctx, g_result):  # type: ignore[override]
        """Combine bwd; scatters the result grad into the grad target."""
        if not g_result.is_contiguous():
            g_result = g_result.contiguous()
        (handle_mem,) = ctx.saved_tensors
        grad_expert_out = ctx.grad_out
        if grad_expert_out is None:
            grad_expert_out = _alloc_io(
                ctx.expert_out_shape, ctx.expert_out_dtype, ctx.device, tex.ep_get_zero_copy()
            )
        torch.ops.transformer_engine_ep.combine_bwd(handle_mem, g_result, grad_expert_out)
        return (
            None,  # handle_mem
            None,  # num_local_tokens
            None,  # hidden_dim
            None,  # grad_out
            grad_expert_out,
        )


# Public high-level wrappers


# NCCL EP currently only supports bfloat16 payload tensors.
def _require_bf16(name: str, t: torch.Tensor) -> None:
    if t.dtype is not torch.bfloat16:
        raise NotImplementedError(
            f"NCCL EP currently supports only bfloat16 payloads; got {name}.dtype={t.dtype}."
        )


def _alloc_io(shape, dtype: torch.dtype, device, zero_copy: bool) -> torch.Tensor:
    """Allocate a dispatch/combine IO tensor the caller did not supply: from the symm-mem pool in
    zero-copy mode (auto-registered segment, lifecycle managed by torch refcount), else plain."""
    if zero_copy:
        t = symm_mem_alloc(shape, dtype, _EP_GROUP, device=device, use_pool=True)
        # symm-mem storage is non-resizable; exempt it from CPU activation offloading (which
        # releases via storage.resize_(0)). Matters for bf16 recv_tokens (the saved activation).
        mark_not_offload(t)
        return t
    return torch.empty(*shape, dtype=dtype, device=device)


def ep_dispatch(
    buffer: EpBuffer,
    tokens: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    recv_tokens: Optional[torch.Tensor] = None,
    recv_topk_weights: Optional[torch.Tensor] = None,
):
    """Prepare + dispatch with autograd. topk_idx must be int32 or int64.

    ``recv_tokens`` / ``recv_topk_weights`` are the dispatch recv outputs: pass caller-owned buffers
    (mcore-managed mode; in zero-copy they must be symm-mem-backed) or leave them None to allocate on
    the fly (zero-copy: symm-mem pool; normal: plain). In eager mode the recv outputs are sized to
    this step's recv-token total and must not be caller-supplied. Returns (recv_tokens,
    recv_topk_weights, tokens_per_expert); tokens_per_expert is non-diff. See ``buffer.total_recv_tokens`` for
    the per-step recv total.
    """
    _require_bf16("tokens", tokens)
    if topk_weights.dtype is not torch.float32:
        raise TypeError(
            f"topk_weights must be float32; got dtype={topk_weights.dtype}. "
            "Cast with topk_weights.float() before calling."
        )
    skip_prepare = False
    if buffer.eager:
        if recv_tokens is not None or recv_topk_weights is not None:
            raise ValueError(
                "eager mode sizes the recv outputs from the per-step recv-token total "
                "and cannot use caller-supplied recv_tokens / recv_topk_weights"
            )
        # Prepare first to learn this step's recv total, then size the recv
        # outputs to it; skip re-running prepare inside the autograd forward.
        ep_prepare(buffer, topk_idx)
        rows = buffer._host_total_recv_tokens
        skip_prepare = True
    else:
        rows = buffer.recv_capacity_per_rank
    if recv_tokens is None:
        recv_tokens = _alloc_io(
            (rows, buffer.hidden_dim),
            buffer.payload_dtype,
            buffer.device,
            buffer.zero_copy,
        )
    if recv_topk_weights is None:
        recv_topk_weights = _alloc_io((rows,), torch.float32, buffer.device, buffer.zero_copy)
    return _EpDispatch.apply(
        buffer.handle_mem,
        buffer.top_k,
        buffer.alignment,
        recv_tokens,
        recv_topk_weights,
        buffer.tokens_per_expert,
        buffer.total_recv_tokens,
        topk_idx,
        tokens,
        topk_weights,
        skip_prepare,
    )


def ep_combine(
    buffer: EpBuffer,
    expert_out: torch.Tensor,
    *,
    num_local_tokens: Optional[int] = None,
    grad_out: Optional[torch.Tensor] = None,
):
    """Combine with autograd; caller pre-applies topk weighting.

    ``expert_out`` is the combine input (always caller-supplied; in zero-copy it must be symm-mem-
    backed). ``grad_out`` is the backward's grad target: pass a caller-owned buffer (mcore-managed
    mode) or leave it None to allocate on the fly in the backward (zero-copy: symm-mem pool; normal:
    plain). In eager mode the grad target is sized per step and must not be caller-supplied. Result
    shape is (num_local_tokens, buffer.hidden_dim); defaults to buffer.max_tokens_per_rank rows.
    """
    _require_bf16("expert_out", expert_out)
    if buffer.eager and grad_out is not None:
        raise ValueError(
            "eager mode sizes the combine grad target per step and cannot use a "
            "caller-supplied grad_out"
        )
    if num_local_tokens is None:
        num_local_tokens = buffer.max_tokens_per_rank
    return _EpCombine.apply(
        buffer.handle_mem,
        num_local_tokens,
        buffer.hidden_dim,
        grad_out,
        expert_out,
    )
