# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Opt-in cuDNN Frontend compact GQA backward dispatch."""
import os
import weakref
from threading import RLock
from typing import TYPE_CHECKING, Optional

import torch

from transformer_engine.pytorch.attention.packed_sequence import (
    _AttentionBackendWorkspace,
    _get_cu_seqlens,
)

if TYPE_CHECKING:
    from .backends import FusedAttnBwdArgs

_ENABLED = os.getenv("NVTE_CUDNN_COMPACT_GQA_BWD", "0") == "1"
_PLANS = {}
_BOUND_WORKSPACES = {}
_PLAN_LOCK = RLock()


def compact_gqa_enabled() -> bool:
    """Whether the optional backend was enabled before TE was imported."""
    return _ENABLED


def _packing(args: "FusedAttnBwdArgs"):
    logical = _get_cu_seqlens(args.cu_seqlens_q)
    if logical is None or logical != _get_cu_seqlens(args.cu_seqlens_kv):
        return None
    physical = [
        logical if tensor is None else _get_cu_seqlens(tensor)
        for tensor in (args.cu_seqlens_q_padded, args.cu_seqlens_kv_padded)
    ]
    offsets = physical[0]
    if (
        offsets is None
        or offsets != physical[1]
        or len(offsets) != len(logical)
        or offsets[-1] != args.q.shape[0]
    ):
        return None
    lengths = tuple(b - a for a, b in zip(logical, logical[1:]))
    if any(length > b - a for length, a, b in zip(lengths, offsets, offsets[1:])):
        return None
    if max(lengths) > min(args.max_seqlen_q, args.max_seqlen_kv):
        return None
    return offsets, lengths


def _eligible(args: "FusedAttnBwdArgs", d_out: torch.Tensor, lse: torch.Tensor) -> bool:
    q = args.q
    if args.fp8 or args.is_input_fp8 or args.use_FAv2_bwd or args.deterministic:
        return False
    if args.qkv_layout != "thd_thd_thd" or args.dqkv_layout != "thd_thd_thd":
        return False
    if args.o_format != "thd" or args.nominal_dtype != torch.bfloat16:
        return False
    if args.dropout_p != 0 or args.attn_bias_type != "no_bias" or args.softmax_type != "vanilla":
        return False
    if args.attn_mask_type not in ("causal", "padding_causal") or args.attn_scale not in (
        None,
        1 / 16,
    ):
        return False
    if args.window_size not in (None, (-1, -1), (-1, 0)):
        return False
    if not isinstance(q, torch.Tensor) or not q.is_cuda:
        return False
    if torch.cuda.get_device_capability(q.device) != (10, 7):
        return False
    if torch.cuda.is_current_stream_capturing() or torch.compiler.is_compiling():
        return False
    tokens = q.shape[0]
    if not tokens:
        return False
    for tensor, heads in ((q, 8), (args.k, 1), (args.v, 1), (args.out, 8), (d_out, 8)):
        if not isinstance(tensor, torch.Tensor):
            return False
        if (
            tensor.device != q.device
            or tensor.dtype != torch.bfloat16
            or tuple(tensor.shape) != (tokens, heads, 256)
            or not tensor.is_contiguous()
            or tensor.data_ptr() % 16
        ):
            return False
    return (
        isinstance(lse, torch.Tensor)
        and lse.device == q.device
        and lse.dtype == torch.float32
        and lse.is_contiguous()
        and tuple(lse.shape) in ((tokens, 8), (tokens, 8, 1))
    )


def try_compact_gqa_backward(
    args: "FusedAttnBwdArgs", d_out: torch.Tensor, lse: torch.Tensor
) -> Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Return gradients, or None when stock backward should handle this call."""
    if not _ENABLED or not _eligible(args, d_out, lse):
        return None
    packing = _packing(args)
    if packing is None:
        return None
    # Experimental dependency is loaded only for an eligible opt-in call.
    try:
        from cudnn.sdpa.bwd.compact_gqa import CompactGqaBackward, compact_gqa_backward
    except ImportError as error:
        raise RuntimeError(
            "NVTE_CUDNN_COMPACT_GQA_BWD requires the cuDNN Frontend compact GQA API."
        ) from error
    scope = args.backend_workspace
    if scope is None:
        scope = _AttentionBackendWorkspace()
    offsets, lengths = packing
    capacity = max(1, *lengths)
    stream = torch.cuda.current_stream(args.q.device).cuda_stream
    key = (args.q.device, stream, lse.ndim)
    # Forward captures the owner so autograd threads do not depend on ContextVar propagation.
    with scope.lock:
        if not scope.active:
            return None
        with _PLAN_LOCK:
            if key not in _PLANS:
                plan = CompactGqaBackward(
                    args.q,
                    args.k,
                    args.v,
                    args.out,
                    d_out,
                    lse,
                    max_seqlen=capacity,
                    query_rows=32768,
                    groups=4,
                    fast_store=True,
                )
                plan.compile(stream)
                _PLANS[key] = plan
            plan = _PLANS[key]
            buffer_key = ("compact_gqa", *key)
            buffer, allocated = scope.buffers.get(buffer_key, (None, 0))
            if allocated < capacity:
                buffer = torch.empty(
                    plan.scratch_workspace_bytes(capacity), dtype=torch.uint8, device=args.q.device
                )
                allocated = capacity
                scope.buffers[buffer_key] = (buffer, allocated)
            bound = _BOUND_WORKSPACES.get(key)
            if bound is None or bound() is not buffer:
                plan.initialize_workspace(buffer, stream, max_seqlen=allocated)
                _BOUND_WORKSPACES[key] = weakref.ref(buffer)
            result = compact_gqa_backward(
                args.q,
                args.k,
                args.v,
                args.out,
                d_out,
                lse,
                plan=plan,
                workspace=buffer,
                current_stream=stream,
                sequence_offsets=offsets,
                sequence_lengths=lengths,
            )
    return result["dq"], result["dk"], result["dv"]
