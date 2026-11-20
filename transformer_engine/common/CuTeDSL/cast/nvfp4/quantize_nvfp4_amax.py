# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Global amax of a BF16 input, which nvte_compute_amax computes with these kernels for an NVFP4
output when the CuTeDSL backend is enabled."""

from typing import Optional

import cutlass
from cutlass import cute
from cutlass import BFloat16, Float32, Int32, Int64, Uint32
from cuda.bindings.driver import CUstream  # pylint: disable=no-name-in-module

from transformer_engine.common.CuTeDSL.utils import (
    abs_max_tree_bf16x2,
    abs_max_x2_bf16,
    create_l2_policy,
    ld_global_evict_last_v8,
    swap_halves_b32,
)
from transformer_engine.common.CuTeDSL.utils_nvfp4 import zero_b32
from .quantize_nvfp4_common import (
    CUTEDSL_DEBUG_LOGGING,
    NVFP4_SHAPE_ALIGNMENT,
    noop_flag_is_set,
)


@cute.jit
def ld_global_evict_last_v4(addr: Int64, policy: Int64):
    """16 bytes of input that is read again soon, as ld_global_evict_last_v8, with an evict_last
    `policy`: the .L2::evict_last qualifier needs a 256-bit load. 16-byte aligned."""
    return list(
        cute.arch.inline_ptx(
            "ld.global.L2::cache_hint.v4.b32 {{$w0}, {$w1}, {$w2}, {$w3}}, [{$r0}], {$r1};",
            write_only_types=[Uint32] * 4,
            read_only_args=[addr, policy],
        )
    )


@cute.jit
def min_s64(a: Int64, b: Int64) -> Int64:
    """The smaller of two 64-bit integers, without a branch."""
    return cute.arch.inline_ptx(
        "min.s64 {$w0}, {$r0}, {$r1};", write_only_types=[Int64], read_only_args=[a, b]
    )


class NVFP4GlobalAmaxKernel:
    """Global amax of a BF16 input into the rowwise amax and, if present, the columnwise amax:
    bit for bit the amax_kernel of TE's current scaling.

    One thread zeroes the amaxes. Then each CTA reduces a 64 KiB stripe, every thread holding
    _BYTES_PER_THREAD of loads in flight, and takes the maximum into the amaxes with an integer
    atomic max on the float bits. The magnitudes stay packed bf16 until then, so no float
    instruction can flush a subnormal, and a NaN loses to the zero the CTA's fold starts from, as
    in TE's kernel.
    """

    _THREADS = 256
    _BYTES_PER_THREAD = 256

    def __init__(self, columnwise_amax: bool):
        self.COLUMNWISE_AMAX = columnwise_amax

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,
        mAmaxRow: cute.Tensor,
        mAmaxCol: Optional[cute.Tensor],
        mNoop: cute.Pointer,
        stream: CUstream,
    ):
        if cutlass.const_expr(CUTEDSL_DEBUG_LOGGING):
            cute.printf(
                "[CuTeDSL] NVFP4GlobalAmaxKernel.__call__() with "
                f"columnwise_amax={self.COLUMNWISE_AMAX}\n"
            )
        M, N = mX.shape
        bytes_per_cta = self._THREADS * self._BYTES_PER_THREAD
        num_ctas = Int32((Int64(M) * Int64(N) * 2 + (bytes_per_cta - 1)) // bytes_per_cta)
        self.zero_kernel(mAmaxRow, mAmaxCol, mNoop).launch(
            grid=[1, 1, 1], block=[1, 1, 1], stream=stream
        )
        # An empty input only gets the zero, as in TE's launch_amax_kernel.
        if num_ctas > 0:
            # Rows and columns are multiples of 32, so the base alone decides the 32-byte
            # alignment.
            if mX.iterator.toint() % 32 == 0:
                self.amax_kernel(mX, mAmaxRow, mAmaxCol, mNoop, num_ctas, 32).launch(
                    grid=[num_ctas, 1, 1], block=[self._THREADS, 1, 1], stream=stream
                )
            else:
                self.amax_kernel(mX, mAmaxRow, mAmaxCol, mNoop, num_ctas, 16).launch(
                    grid=[num_ctas, 1, 1], block=[self._THREADS, 1, 1], stream=stream
                )

    @cute.kernel
    def zero_kernel(
        self,
        mAmaxRow: cute.Tensor,
        mAmaxCol: Optional[cute.Tensor],
        mNoop: cute.Pointer,
    ):
        """Device entry zeroing the amaxes that amax_kernel takes its maximum into."""
        skip = noop_flag_is_set(mNoop)
        if not skip:
            mAmaxRow[0] = Float32(0.0)
            if cutlass.const_expr(self.COLUMNWISE_AMAX):
                mAmaxCol[0] = Float32(0.0)

    @cute.kernel
    def amax_kernel(
        self,
        mX: cute.Tensor,
        mAmaxRow: cute.Tensor,
        mAmaxCol: Optional[cute.Tensor],
        mNoop: cute.Pointer,
        num_ctas: Int32,
        VEC_BYTES: cutlass.Constexpr,
    ):
        """Device entry reducing one stripe of the input per CTA."""
        THREADS = self._THREADS
        LOADS = self._BYTES_PER_THREAD // VEC_BYTES
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        M, N = mX.shape

        smem = cutlass.memory.SmemAllocator()
        warp_amax = smem.allocate_tensor(Int32, cute.make_layout(THREADS // 32), byte_alignment=16)

        # Under the noop flag nothing reads the input, as in TE's kernel. The flag is CTA-uniform.
        skip = noop_flag_is_set(mNoop)
        if not skip:
            # The stripes run from the back of the input to the front, so the front, which the
            # quantize kernel reads first, is the most recently read when the input exceeds L2.
            stripe = num_ctas - 1 - bidx
            first = Int64(stripe) * (THREADS * self._BYTES_PER_THREAD) + Int64(tidx * VEC_BYTES)
            last = Int64(M) * Int64(N) * 2 - VEC_BYTES
            x = mX.iterator.toint()
            policy = None
            if cutlass.const_expr(VEC_BYTES == 16):
                policy = create_l2_policy(True)
            words = []
            for i in cutlass.range_constexpr(LOADS):
                # Past the end of the input, a thread rereads its last vector, which leaves the
                # maximum as it is.
                addr = x + min_s64(first + i * THREADS * VEC_BYTES, last)
                if cutlass.const_expr(VEC_BYTES == 32):
                    words.extend(ld_global_evict_last_v8(addr))
                else:
                    words.extend(ld_global_evict_last_v4(addr, policy))

            # Packed abs-max over the thread's words, then its warp; both halves stay maxima.
            amax = Int32(abs_max_tree_bf16x2(words))
            for stage in cutlass.range_constexpr(5):
                amax = abs_max_x2_bf16(amax, cute.arch.shuffle_sync_bfly(amax, 1 << stage))
            if tidx % 32 == 0:
                warp_amax[tidx // 32] = amax
        cute.arch.sync_threads()

        if tidx == 0:
            if not skip:
                cta_amax = zero_b32()
                for w in cutlass.range_constexpr(THREADS // 32):
                    cta_amax = abs_max_x2_bf16(cta_amax, warp_amax[w])
                # Fold the halves and keep the magnitude, which is the float's bits.
                cta_amax = abs_max_x2_bf16(cta_amax, Int32(swap_halves_b32(Uint32(cta_amax))))
                cta_amax = cta_amax & Int32(0x7FFF0000)
                self._atomic_max(mAmaxRow, cta_amax)
                if cutlass.const_expr(self.COLUMNWISE_AMAX):
                    self._atomic_max(mAmaxCol, cta_amax)

    @cute.jit
    def _atomic_max(self, mAmax: cute.Tensor, amax_bits: Int32):
        """Atomic max of a non-negative float, given by its bits, into mAmax[0]."""
        amax_i32 = cute.make_tensor(
            cute.recast_ptr(mAmax.iterator, dtype=Int32), cute.make_layout(1)
        )
        cute.arch.atomic_max(amax_i32.iterator, amax_bits)


def compile_global_amax_function(columnwise_amax: bool):
    """Return the compiled CuTeDSL function computing the global amax. Its ABI, fixed by the fake
    tensors below, is (mX, mAmaxRow, mAmaxCol, mNoop, stream), as the C++ call site in
    `compute_amax_nvfp4_cutedsl.cuh` passes them; mAmaxCol is present iff columnwise_amax.
    """

    def _gmem(dtype, shape, stride_order, align):
        """Fake row-major GMEM tensor; `align` is the assumed base alignment in bytes."""
        return cute.runtime.make_fake_compact_tensor(
            dtype,
            shape,
            stride_order=stride_order,
            memspace=cute.AddressSpace.gmem,
            assumed_align=align,
        )

    sym_M = cute.sym_int32(divisibility=NVFP4_SHAPE_ALIGNMENT)
    sym_N = cute.sym_int32(divisibility=NVFP4_SHAPE_ALIGNMENT)
    in_fake = _gmem(BFloat16, (sym_M, sym_N), stride_order=(1, 0), align=16)
    amax_row_fake = _gmem(Float32, (1,), stride_order=(0,), align=4)
    amax_col_fake = _gmem(Float32, (1,), stride_order=(0,), align=4) if columnwise_amax else None
    noop_fake = cute.runtime.nullptr(Float32, mem_space=cute.AddressSpace.gmem, assumed_align=4)

    return cute.compile(
        NVFP4GlobalAmaxKernel(columnwise_amax),
        in_fake,  # mX
        amax_row_fake,  # mAmaxRow
        amax_col_fake,  # mAmaxCol
        noop_fake,  # mNoop
        cute.runtime.make_fake_stream(),  # stream
        options="--enable-tvm-ffi",
    )
