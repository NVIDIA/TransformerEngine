# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Fused ring forward with head-parallel FlashAttention backward."""

import torch
import torch.distributed as dist
import triton

from transformer_engine.common.triton.cp_packing import _pack_cp_tensors, _unpack_cp_tensors
from transformer_engine.pytorch.utils import get_device_compute_capability


def _is_supported(q, k, v, cp_size):
    """Check static geometry and optional backend availability before dispatch."""
    from .backends import _flash_attn_bwd_v4

    return (
        _flash_attn_bwd_v4 is not None
        and get_device_compute_capability() == (10, 0)
        and cp_size in (2, 4, 8)
        and all(t.ndim == 4 and t.is_cuda for t in (q, k, v))
        and q.shape == k.shape
        and q.shape[:3] == v.shape[:3]
        and q.shape[0] > 0
        and q.shape[0] % 2 == 0
        and q.shape[1] > 0
        and q.shape[2] > 0
        and q.shape[2] % cp_size == 0
        and (q.shape[-1], v.shape[-1]) in ((128, 128), (192, 128))
        and q.dtype in (torch.float16, torch.bfloat16)
        and all(t.dtype == q.dtype and t.device == q.device and t.stride(-1) == 1 for t in (k, v))
        and q.stride(-1) == 1
    )


def _backward(q, k, v, output, gradient, lse, group, softmax_scale, deterministic):
    """Exchange the saved ring tensors once in each direction around FA4 backward."""
    from .backends import _flash_attn_bwd_v4

    q, k, v, output, gradient, lse = _to_heads(q, k, v, output, gradient, lse, group)
    dq, dk, dv = _flash_attn_bwd_v4(
        q.transpose(0, 1),
        k.transpose(0, 1),
        v.transpose(0, 1),
        output.transpose(0, 1),
        gradient.transpose(0, 1),
        lse,
        softmax_scale=softmax_scale,
        causal=True,
        deterministic=deterministic,
    )
    return _to_sequence(dq.transpose(0, 1), dk.transpose(0, 1), dv.transpose(0, 1), group)


def _exchange(tensors: tuple, group: object, *, forward: bool, lse=None) -> tuple:
    size = dist.get_world_size(group)
    q, k, v = tensors[:3]
    sequence, batch, heads, _ = q.shape
    if (
        batch < 1
        or size not in (2, 4, 8)
        or any(
            t.dtype != q.dtype
            or t.dtype not in (torch.float16, torch.bfloat16)
            or t.stride(-1) != 1
            for t in tensors
        )
    ):
        raise ValueError("Fused packing requires FP16/BF16, CP2/4/8 and contiguous head widths")
    dq, dv = q.shape[-1], v.shape[-1]
    if (dq, dv) not in ((128, 128), (192, 128)) or k.shape != q.shape or v.shape[:3] != q.shape[:3]:
        raise ValueError("Unexpected MLA head geometry")
    local_s, local_h = (sequence, heads // size) if forward else (sequence // size, heads)
    if local_s % 2 or (forward and heads % size):
        raise ValueError("Invalid balanced CP partition")
    width = triton.cdiv(2 * dq + 3 * dv + 2, 64) * 64 if forward else 2 * dq + dv
    elements = size * local_s * batch * local_h * width
    # Strided inputs can require wide offsets even when the payload is small.
    index64 = elements >= 2**31 or any(
        sum((dim - 1) * stride for dim, stride in zip(t.shape, t.stride())) >= 2**31
        for t in (*tensors, lse)
        if t is not None
    )
    # NCCL byte transport and integer kernel loads preserve every FP32 LSE bit.
    wire = torch.empty(elements * 2, dtype=torch.uint8, device=q.device)
    received = torch.empty_like(wire)
    o, do = tensors[3:] if forward else (q, q)
    if forward and (lse.dtype != torch.float32 or lse.shape != (batch, heads, sequence)):
        raise ValueError("Unexpected native LSE geometry")
    _pack_cp_tensors[(triton.cdiv(wire.numel() // 2, 1024),)](
        q.view(torch.int16),
        k.view(torch.int16),
        v.view(torch.int16),
        o.view(torch.int16),
        do.view(torch.int16),
        lse.view(torch.int32) if forward else None,
        wire.view(torch.int16),
        local_s,
        batch,
        dq,
        dv,
        local_h,
        size,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        o.stride(0),
        o.stride(1),
        o.stride(2),
        do.stride(0),
        do.stride(1),
        do.stride(2),
        lse.stride(0) if forward else 0,
        lse.stride(1) if forward else 0,
        lse.stride(2) if forward else 0,
        forward,
        width,
        1024,
        index64,
    )
    dist.all_to_all_single(received, wire, group=group)
    # The sent buffer is dead after the collective and becomes the output slab.
    out_s, out_h = (sequence * size, local_h) if forward else (local_s, heads * size)
    count = out_s * batch * out_h
    widths = (dq, dq, dv, dv, dv) if forward else (dq, dq, dv)
    slab = wire.view(q.dtype)
    outputs = []
    offset = 0
    for head_width in widths:
        outputs.append(
            slab[offset : offset + count * head_width].view(out_s, batch, out_h, head_width)
        )
        offset += count * head_width
    global_lse = (
        slab[offset : offset + count * 2].view(torch.float32).view(batch, out_h, out_s)
        if forward
        else None
    )
    qo, ko, vo = outputs[0], outputs[1], outputs[2]
    oo, d_o = outputs[3:] if forward else (qo, qo)
    _unpack_cp_tensors[(triton.cdiv(wire.numel() // 2, 1024),)](
        received.view(torch.int16),
        qo.view(torch.int16),
        ko.view(torch.int16),
        vo.view(torch.int16),
        oo.view(torch.int16),
        d_o.view(torch.int16),
        global_lse.view(torch.int16) if forward else None,
        local_s,
        batch,
        dq,
        dv,
        local_h,
        size,
        forward,
        width,
        1024,
        index64,
    )
    return (*outputs, global_lse) if forward else tuple(outputs)


def _to_heads(query, key, value, output, gradient, lse, group) -> tuple:
    """Exchange native ring tensors, accounting for the final KV buffer owner."""
    result = _exchange((query, key, value, output, gradient), group, forward=True, lse=lse)
    return result[0], result[1], result[2], result[3], result[4], result[5]


def _to_sequence(dq, dk, dv, group) -> tuple:
    """Return all head partitions to their original balanced token owner."""
    result = _exchange((dq, dk, dv), group, forward=False)
    return result[0], result[1], result[2]
