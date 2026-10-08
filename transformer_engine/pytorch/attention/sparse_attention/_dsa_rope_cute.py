# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""CuTe DSL kernels for DSv4 trailing-channel interleaved RoPE.

The device bodies come from the retained DSv4 experiments:
forward SHA256 0355016c141365dfa436deda15a29cce49146a0815833ab7963d941ef67f7e5c
and backward SHA256 fdda9fb8bcf44e998d000bb8f78354fcf02f2c0b7cb4d9d3c9787a24abaaef82.
Only their host boundary is narrowed to TE's packed single-tensor contract.
"""

import torch
import cutlass
import cutlass.cute as cute
import tvm_ffi
from cutlass.cute.runtime import from_dlpack
from cuda.bindings import driver as cuda


@cute.kernel
def _mla_rope_fwd_inplace_kernel(
    q: cute.Tensor,
    cos: cute.Tensor,
    sin: cute.Tensor,
    cu_seqlens,
    positions,
    nope_dim: cutlass.Constexpr,
    emb_dim: cutlass.Constexpr,
    batch_size: cutlass.Constexpr,
    cp_rank: cutlass.Constexpr,
    cp_size: cutlass.Constexpr,
    inverse: cutlass.Constexpr,
    remove_interleaving: cutlass.Constexpr,
    packed_pairs: cutlass.Constexpr,
    threads: cutlass.Constexpr,
    heads_per_cta: cutlass.Constexpr,
):
    tid, _, _ = cute.arch.thread_idx()
    row, head_tile, _ = cute.arch.block_idx()
    pos = cutlass.Int64(row // batch_size)
    if cutlass.const_expr(positions is not None):
        pos = positions[row].to(cutlass.Int64)
    elif cutlass.const_expr(cu_seqlens is not None):
        pos = cutlass.Int64(-1)
        length = cutlass.Int64(0)
        # Upper bound skips duplicate/empty ends and preserves CP floor division.
        lo = cutlass.Int64(1)
        hi = cutlass.Int64(cute.size(cu_seqlens))
        while lo < hi:
            mid = (lo + hi) // 2
            end = cu_seqlens[mid].to(cutlass.Int64) // cp_size
            if end > row:
                hi = mid
            else:
                lo = mid + 1
        # The exclusive-bound sentinel denotes padding at/beyond the last end.
        if lo < cute.size(cu_seqlens):
            previous = cu_seqlens[lo - 1].to(cutlass.Int64) // cp_size
            end = cu_seqlens[lo].to(cutlass.Int64) // cp_size
            pos = row - previous
            length = end - previous
        if pos == -1:
            pos = cutlass.Int64(0)
        if cutlass.const_expr(cp_size > 1):
            if pos < length // 2:
                pos = pos + cp_rank * length // 2
            else:
                pos = pos - length // 2 + (2 * cp_size - cp_rank - 1) * length // 2

    if cutlass.const_expr(packed_pairs):
        q_pairs = cute.recast_tensor(q, cutlass.Uint32)
    half = emb_dim // 2
    count = cute.ceil_div(heads_per_cta * half, threads)
    left = cute.make_rmem_tensor((count,), cutlass.Float32)
    right = cute.make_rmem_tensor((count,), cutlass.Float32)
    # Adding `threads` leaves `flat % half` unchanged only under this guard.
    # Hoist the four same-address table loads once for each participating thread.
    if cutlass.const_expr(threads % half == 0 and heads_per_cta * half >= threads):
        table_pair = tid % half
        cl_once = cos[pos, table_pair]
        cr_once = cos[pos, half + table_pair]
        sl_once = sin[pos, table_pair]
        sr_once = sin[pos, half + table_pair]
    for slot in cutlass.range_constexpr(count):
        flat = tid + slot * threads
        head = head_tile * heads_per_cta + flat // half
        pair = flat % half
        if flat < heads_per_cta * half and head < q.shape[1]:
            if cutlass.const_expr(packed_pairs):
                # Bit reinterpretation preserves the original BF16 operands.
                values = cute.make_rmem_tensor((2,), q.element_type)
                word = cute.recast_tensor(values, cutlass.Uint32)
                word[0] = q_pairs[row, head, nope_dim // 2 + pair]
                x, y = values[0], values[1]
            else:
                x = q[row, head, nope_dim + 2 * pair]
                y = q[row, head, nope_dim + 2 * pair + 1]
            if cutlass.const_expr(threads % half == 0 and heads_per_cta * half >= threads):
                cl, cr, sl, sr = cl_once, cr_once, sl_once, sr_once
            else:
                cl = cos[pos, pair]
                cr = cos[pos, half + pair]
                sl = sin[pos, pair]
                sr = sin[pos, half + pair]
            if cutlass.const_expr(inverse):
                sl, sr = -sl, -sr
            # Floating promotion follows Triton: FP32 > FP16 > BF16.
            left[slot] = (x * cl - y * sl).to(cutlass.Float32)
            right[slot] = (y * cr + x * sr).to(cutlass.Float32)

    # Split-half stores alias other threads' interleaved inputs. Every warp
    # must finish loading before any thread writes to those addresses.
    if cutlass.const_expr(not remove_interleaving):
        cute.arch.sync_threads()
    for slot in cutlass.range_constexpr(count):
        flat = tid + slot * threads
        head = head_tile * heads_per_cta + flat // half
        pair = flat % half
        if flat < heads_per_cta * half and head < q.shape[1]:
            if cutlass.const_expr(packed_pairs and remove_interleaving):
                values = cute.make_rmem_tensor((2,), q.element_type)
                values[0] = left[slot].to(q.element_type)
                values[1] = right[slot].to(q.element_type)
                word = cute.recast_tensor(values, cutlass.Uint32)
                q_pairs[row, head, nope_dim // 2 + pair] = word[0]
            else:
                if cutlass.const_expr(remove_interleaving):
                    dst_l, dst_r = 2 * pair, 2 * pair + 1
                else:
                    dst_l, dst_r = pair, half + pair
                q[row, head, nope_dim + dst_l] = left[slot].to(q.element_type)
                q[row, head, nope_dim + dst_r] = right[slot].to(q.element_type)


@cute.jit
def _launch_forward(
    q: cute.Tensor,
    cos: cute.Tensor,
    sin: cute.Tensor,
    cu_seqlens,
    positions,
    nope_dim: cutlass.Constexpr,
    emb_dim: cutlass.Constexpr,
    batch_size: cutlass.Constexpr,
    cp_rank: cutlass.Constexpr,
    cp_size: cutlass.Constexpr,
    inverse: cutlass.Constexpr,
    remove_interleaving: cutlass.Constexpr,
    packed_pairs: cutlass.Constexpr,
    stream: cuda.CUstream,
):
    threads = 128
    heads_per_cta = min(16, q.shape[1])
    _mla_rope_fwd_inplace_kernel(
        q,
        cos,
        sin,
        cu_seqlens,
        positions,
        nope_dim,
        emb_dim,
        batch_size,
        cp_rank,
        cp_size,
        inverse,
        remove_interleaving,
        packed_pairs,
        threads,
        heads_per_cta,
    ).launch(
        grid=(q.shape[0], cute.ceil_div(q.shape[1], heads_per_cta), 1),
        block=(threads, 1, 1),
        stream=stream,
    )


@cute.kernel
def _mla_rope_bwd_inplace_kernel(
    q: cute.Tensor,
    cos: cute.Tensor,
    sin: cute.Tensor,
    cu_seqlens,
    positions,
    nope_dim: cutlass.Constexpr,
    emb_dim: cutlass.Constexpr,
    batch_size: cutlass.Constexpr,
    cp_rank: cutlass.Constexpr,
    cp_size: cutlass.Constexpr,
    inverse: cutlass.Constexpr,
    remove_interleaving: cutlass.Constexpr,
    packed_pairs: cutlass.Constexpr,
    threads: cutlass.Constexpr,
    heads_per_cta: cutlass.Constexpr,
):
    tid, _, _ = cute.arch.thread_idx()
    row, head_tile, _ = cute.arch.block_idx()
    pos = cutlass.Int64(row // batch_size)
    if cutlass.const_expr(positions is not None):
        pos = positions[row].to(cutlass.Int64)
    elif cutlass.const_expr(cu_seqlens is not None):
        pos = cutlass.Int64(-1)
        length = cutlass.Int64(0)
        previous = cu_seqlens[0].to(cutlass.Int64) // cp_size
        for seq in range(cute.size(cu_seqlens) - 1):
            end = cu_seqlens[seq + 1].to(cutlass.Int64) // cp_size
            if pos == -1 and end > row:
                pos = row - previous
                length = end - previous
            previous = end
        if pos == -1:
            pos = cutlass.Int64(0)
        if cutlass.const_expr(cp_size > 1):
            if pos < length // 2:
                pos = pos + cp_rank * length // 2
            else:
                pos = pos - length // 2 + (2 * cp_size - cp_rank - 1) * length // 2

    if cutlass.const_expr(packed_pairs):
        q_pairs = cute.recast_tensor(q, cutlass.Uint32)
    half = emb_dim // 2
    count = cute.ceil_div(heads_per_cta * half, threads)
    left = cute.make_rmem_tensor((count,), cutlass.Float32)
    right = cute.make_rmem_tensor((count,), cutlass.Float32)
    # Every slot has the same pair only when adding threads preserves remainder.
    # Preserve table types and the original per-slot inverse/arithmetic below.
    if cutlass.const_expr(threads % half == 0 and heads_per_cta * half >= threads):
        table_pair = tid % half
        cl_once = cos[pos, table_pair]
        cr_once = cos[pos, half + table_pair]
        sl_once = sin[pos, table_pair]
        sr_once = sin[pos, half + table_pair]
    for slot in cutlass.range_constexpr(count):
        flat = tid + slot * threads
        head = head_tile * heads_per_cta + flat // half
        pair = flat % half
        if flat < heads_per_cta * half and head < q.shape[1]:
            if cutlass.const_expr(remove_interleaving):
                src_l, src_r = 2 * pair, 2 * pair + 1
            else:
                src_l, src_r = pair, half + pair
            if cutlass.const_expr(packed_pairs and remove_interleaving):
                # Adjacent inputs form one word; split inputs keep scalar loads.
                values = cute.make_rmem_tensor((2,), q.element_type)
                word = cute.recast_tensor(values, cutlass.Uint32)
                word[0] = q_pairs[row, head, nope_dim // 2 + pair]
                x, y = values[0], values[1]
            else:
                x = q[row, head, nope_dim + src_l]
                y = q[row, head, nope_dim + src_r]
            if cutlass.const_expr(threads % half == 0 and heads_per_cta * half >= threads):
                cl, cr, sl, sr = cl_once, cr_once, sl_once, sr_once
            else:
                cl = cos[pos, pair]
                cr = cos[pos, half + pair]
                sl = sin[pos, pair]
                sr = sin[pos, half + pair]
            if cutlass.const_expr(inverse):
                sl, sr = -sl, -sr
            # CuTe and Triton both promote these floating operands to FP32
            # when a table is FP32, and otherwise preserve the narrower type.
            left[slot] = (x * cl + y * sr).to(cutlass.Float32)
            right[slot] = (-x * sl + y * cr).to(cutlass.Float32)

    # Split-input stores alias only within a head. When half divides 32,
    # every head belongs to one warp; wider/non-divisor widths need the CTA.
    if cutlass.const_expr(not remove_interleaving):
        if cutlass.const_expr(half <= 32 and 32 % half == 0):
            cute.arch.sync_warp()
        else:
            cute.arch.sync_threads()
    for slot in cutlass.range_constexpr(count):
        flat = tid + slot * threads
        head = head_tile * heads_per_cta + flat // half
        pair = flat % half
        if flat < heads_per_cta * half and head < q.shape[1]:
            if cutlass.const_expr(packed_pairs):
                # Backward always writes adjacent pairs, after original casts.
                values = cute.make_rmem_tensor((2,), q.element_type)
                values[0] = left[slot].to(q.element_type)
                values[1] = right[slot].to(q.element_type)
                word = cute.recast_tensor(values, cutlass.Uint32)
                q_pairs[row, head, nope_dim // 2 + pair] = word[0]
            else:
                q[row, head, nope_dim + 2 * pair] = left[slot].to(q.element_type)
                q[row, head, nope_dim + 2 * pair + 1] = right[slot].to(q.element_type)


@cute.jit
def _launch_backward(
    q: cute.Tensor,
    cos: cute.Tensor,
    sin: cute.Tensor,
    cu_seqlens,
    positions,
    nope_dim: cutlass.Constexpr,
    emb_dim: cutlass.Constexpr,
    batch_size: cutlass.Constexpr,
    cp_rank: cutlass.Constexpr,
    cp_size: cutlass.Constexpr,
    inverse: cutlass.Constexpr,
    remove_interleaving: cutlass.Constexpr,
    packed_pairs: cutlass.Constexpr,
    stream: cuda.CUstream,
):
    threads = 128
    heads_per_cta = min(16, q.shape[1])
    _mla_rope_bwd_inplace_kernel(
        q,
        cos,
        sin,
        cu_seqlens,
        positions,
        nope_dim,
        emb_dim,
        batch_size,
        cp_rank,
        cp_size,
        inverse,
        remove_interleaving,
        packed_pairs,
        threads,
        heads_per_cta,
    ).launch(
        grid=(q.shape[0], cute.ceil_div(q.shape[1], heads_per_cta), 1),
        block=(threads, 1, 1),
        stream=stream,
    )


class _DSv4RopeEntry:
    """TVM-FFI entrypoint with the kernel's static launch choices captured."""

    def __init__(self, heads, head_dim, rope_dim, backward):
        self.heads = heads
        self.head_dim = head_dim
        self.rope_dim = rope_dim
        self.backward = backward

    @cute.jit
    def __call__(self, q, cos, sin, cu_seqlens, stream):
        if cutlass.const_expr(self.backward):
            _launch_backward(
                q,
                cos,
                sin,
                cu_seqlens,
                None,
                self.head_dim - self.rope_dim,
                self.rope_dim,
                1,
                0,
                1,
                False,
                True,
                True,
                stream,
            )
        else:
            _launch_forward(
                q,
                cos,
                sin,
                cu_seqlens,
                None,
                self.head_dim - self.rope_dim,
                self.rope_dim,
                1,
                0,
                1,
                False,
                True,
                True,
                stream,
            )


def _compile_native(heads, head_dim, rope_dim, batch_size, backward):
    """Compile one symbolic-token RoPE entrypoint for common C++ dispatch."""
    rows = cute.sym_int32()
    positions = cute.sym_int32()
    q = cute.runtime.make_fake_compact_tensor(
        cutlass.BFloat16,
        (rows, heads, head_dim),
        stride_order=(2, 1, 0),
        memspace=cute.AddressSpace.gmem,
        assumed_align=4,
    )
    table = lambda: cute.runtime.make_fake_compact_tensor(
        cutlass.Float32,
        (positions, rope_dim),
        stride_order=(1, 0),
        memspace=cute.AddressSpace.gmem,
        assumed_align=4,
    )
    cu_seqlens = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32,
        (batch_size + 1,),
        stride_order=(0,),
        memspace=cute.AddressSpace.gmem,
        assumed_align=4,
    )
    return cute.compile(
        _DSv4RopeEntry(heads, head_dim, rope_dim, backward),
        q,
        table(),
        table(),
        cu_seqlens,
        cute.runtime.make_fake_stream(),
        options="--enable-tvm-ffi",
    )


def get_dsv4_rope_function(fn_name, heads, head_dim, rope_dim, batch_size, backward):
    """Compile and register the native callable requested by common C++."""
    if tvm_ffi.get_global_func(fn_name, allow_missing=True) is not None:
        return True
    if torch.cuda.get_device_capability() != (10, 0):
        return False
    compiled = _compile_native(heads, head_dim, rope_dim, batch_size, backward)
    native = getattr(compiled, "__tvm_ffi_object__", lambda: None)()
    tvm_ffi.register_global_func(fn_name, native if native is not None else compiled, override=True)
    return True


tvm_ffi.register_global_func("get_dsv4_rope_function", get_dsv4_rope_function, override=True)


def _prepare(t, cos, sin, cu_seqlens):
    """Validate and adapt TE's packed BSHD rows without changing device semantics."""
    emb_dim = cos.shape[-1]
    nope_dim = t.shape[-1] - emb_dim
    if t.ndim != 3 or t.stride(-1) != 1 or emb_dim <= 0 or emb_dim % 4 or nope_dim < 0:
        raise ValueError("CuTe RoPE requires packed [T,H,D] with a divisible-by-4 rotary tail.")
    if cos.ndim != 2 or sin.shape != cos.shape or cos.stride(-1) != 1 or sin.stride(-1) != 1:
        raise ValueError("COS/SIN must be matching [P,R] tables with unit channel stride.")
    if (
        cu_seqlens.ndim != 1
        or not cu_seqlens.is_contiguous()
        or cu_seqlens.dtype not in (torch.int32, torch.int64)
    ):
        raise ValueError("Sequence prefixes must be a contiguous int32/int64 vector.")
    for tensor in (t, cos, sin, cu_seqlens):
        if not tensor.is_cuda or tensor.device != t.device:
            raise ValueError("Input, tables, and sequence prefixes must share one CUDA device.")
    if t.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise ValueError("CuTe RoPE supports FP32, FP16, and BF16 inputs.")
    if cos.dtype not in (torch.float32, torch.float16, torch.bfloat16) or sin.dtype != cos.dtype:
        raise ValueError("COS/SIN must share a supported floating type.")
    packed_pairs = (
        t.dtype == torch.bfloat16
        and t.data_ptr() % 4 == 0
        and nope_dim % 2 == 0
        and all(stride % 2 == 0 for stride in t.stride()[:-1])
    )
    args = tuple(
        from_dlpack(x, assumed_align=4 if packed_pairs and i == 0 else None)
        for i, x in enumerate((t, cos, sin, cu_seqlens))
    )
    return args, nope_dim, emb_dim, packed_pairs


def forward_inplace(t, cos, sin, cu_seqlens):
    """Apply the retained Q-style forward kernel to packed interleaved rows."""
    if t.numel() == 0:
        return t
    args, nope_dim, emb_dim, packed_pairs = _prepare(t, cos, sin, cu_seqlens)
    q, cos_, sin_, cu_ = args
    _launch_forward(
        q,
        cos_,
        sin_,
        cu_,
        None,
        nope_dim,
        emb_dim,
        1,
        0,
        1,
        False,
        True,
        packed_pairs,
        cuda.CUstream(torch.cuda.current_stream(t.device).cuda_stream),
    )
    return t


def backward_inplace(t, cos, sin, cu_seqlens):
    """Apply the retained Q-style transpose kernel to packed interleaved gradients."""
    if t.numel() == 0:
        return t
    args, nope_dim, emb_dim, packed_pairs = _prepare(t, cos, sin, cu_seqlens)
    q, cos_, sin_, cu_ = args
    _launch_backward(
        q,
        cos_,
        sin_,
        cu_,
        None,
        nope_dim,
        emb_dim,
        1,
        0,
        1,
        False,
        True,
        packed_pairs,
        cuda.CUstream(torch.cuda.current_stream(t.device).cuda_stream),
    )
    return t
