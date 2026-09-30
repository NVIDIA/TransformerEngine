# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Benchmark-only compact-input prototype using cuDNN's packed math helpers.

This experiment replaces the producer / shared-memory pipeline with coalesced
global loads into registers. It accepts TE's compact scales and emits both
GEMM scale layouts in one launch. It is not registered as a production backend.
"""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass.cute.typing import AddressSpace
from cudnn.moe_ep._megamoe_backend.cutedsl_src.kernel_src.rubin.training.mega.fwd_glu.glu_mxfp8_col_requant import (
    Mxfp8ColRequant,
    bits_f32,
    cvt_dn_fp8x2_portable,
    cvt_scaled_up_bf16x2,
    e8m0_raw_from_bf16,
    max_xorsign_abs_bf16x2,
)
import cutlass.cute.nvgpu.cpasync as cpasync


class CompactTmaRequantize(Mxfp8ColRequant):
    """Preserve the upstream TMA pipeline; producer converts compact scales.

    For this benchmark prototype, src_sf contains compact input scales followed
    by an equally sized allocation for the swizzled rowwise output scales.
    """

    @cute.jit
    def produce(
        self,
        smem_data_base,
        smem_sf_in_base,
        mbar_full,
        mbar_empty,
        tbl_data,
        tbl_sf,
        src_sf_base,
        bidx,
        grid_dim_x,
        total_tiles,
        lane_idx,
        tma_atom=None,
        tma_tensor=None,
        token_padding_block: cutlass.Constexpr = None,
    ):
        tok = cutlass.const_expr(self.TILE_TOK)
        width = cutlass.const_expr(self.TILE_HID)
        stages = cutlass.const_expr(self.NumStages)
        sf_bytes = cutlass.const_expr(self.SfTileBytes)
        box_h = cutlass.const_expr(self.TmaBoxHidU32)
        shared = cute.make_tensor(
            cute.make_ptr(cutlass.Uint32, smem_data_base, AddressSpace.smem, assumed_align=128),
            cute.make_layout((tok, box_h, stages), stride=(box_h, 1, tok * box_h)),
        )
        global_data = cute.group_modes(
            cute.local_tile(tma_tensor, (tok, box_h), (None, None)), 0, 2
        )
        shared_tile, global_tile = cpasync.tma_partition(
            tma_atom, 0, cute.make_layout(1), cute.group_modes(shared, 0, 2), global_data
        )
        cpasync.prefetch_descriptor(tma_atom)
        hidden_groups = cutlass.const_expr(self.hidden_groups)
        hidden = cutlass.const_expr(self.hidden)
        sf_stride = cutlass.const_expr(self.SfInStride)
        hidden_atoms = cutlass.const_expr(self._hidden_atoms)
        sf_capacity = cutlass.const_expr(self.max_total_tokens * (self.hidden // 32))
        step = cutlass.Int32(0)
        work = cutlass.Int32(bidx)
        while work < total_tiles * hidden_groups:
            stage = step % stages
            token_tile = work // hidden_groups
            hid_begin = (work % hidden_groups) * width
            if step >= stages:
                cute.arch.mbarrier_wait(mbar_empty + stage, ((step // stages) - 1) % 2)
            for i in cutlass.range_constexpr(0, tok * width // (32 * 32), 1):
                index = lane_idx + i * 32
                row = index // (width // 32)
                col = index % (width // 32)
                addr = (token_tile * tok + row) * (hidden // 32) + hid_begin // 32 + col
                val = cute.make_tensor(
                    cute.make_ptr(
                        cutlass.Uint8,
                        cutlass.Int64(src_sf_base) + cutlass.Int64(addr),
                        AddressSpace.gmem,
                        assumed_align=1,
                    ),
                    cute.make_layout((1,)),
                )[0]
                shared_offset = (col // 4) * sf_stride + (row % 32) * 16 + (row // 32) * 4 + col % 4
                cute.make_tensor(
                    cute.make_ptr(
                        cutlass.Uint8,
                        smem_sf_in_base + stage * sf_bytes + shared_offset,
                        AddressSpace.smem,
                        assumed_align=1,
                    ),
                    cute.make_layout((1,)),
                )[0] = val
                out_index = (
                    (token_tile * hidden_atoms + hid_begin // 128 + col // 4) * 512
                    + (row % 32) * 16
                    + (row // 32) * 4
                    + col % 4
                )
                cute.make_tensor(
                    cute.make_ptr(
                        cutlass.Uint8,
                        cutlass.Int64(src_sf_base) + sf_capacity + cutlass.Int64(out_index),
                        AddressSpace.gmem,
                        assumed_align=1,
                    ),
                    cute.make_layout((1,)),
                )[0] = val
            cute.arch.sync_warp()
            if lane_idx == 0:
                cute.arch.mbarrier_arrive_and_expect_tx(
                    mbar_full + stage, cutlass.Int32(tok * width)
                )
            cute.arch.sync_warp()
            cute.copy(
                tma_atom,
                global_tile[(None, token_tile, hid_begin // width)],
                shared_tile[(None, stage)],
                tma_bar_ptr=mbar_full + stage,
            )
            step = step + cutlass.Int32(1)
            work = work + grid_dim_x


class CompactRequantize:
    """Each lane owns four adjacent columns and one 32-token scale block."""

    def __init__(self, hidden, groups, token_blocks=4, hidden_blocks=1):
        self.hidden = hidden
        self.groups = groups
        self.token_blocks = token_blocks
        self.hidden_blocks = hidden_blocks

    @cute.jit
    def __call__(
        self,
        src: cute.Tensor,
        sf: cute.Tensor,
        offsets: cute.Tensor,
        dst: cute.Tensor,
        row_sf: cute.Tensor,
        col_sf: cute.Tensor,
        stream: cuda.CUstream,
    ):
        self.kernel(src, sf, offsets, dst, row_sf, col_sf).launch(
            grid=(
                cute.ceil_div(self.hidden, self.hidden_blocks * 128),
                cute.ceil_div(src.shape[0], self.token_blocks * 32),
                1,
            ),
            block=(self.token_blocks * self.hidden_blocks * 32, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        src: cute.Tensor,
        sf: cute.Tensor,
        offsets: cute.Tensor,
        dst: cute.Tensor,
        row_sf: cute.Tensor,
        col_sf: cute.Tensor,
    ):
        tid, _, _ = cute.arch.thread_idx()
        bx, by, _ = cute.arch.block_idx()
        lane = tid % 32
        warp = tid // 32
        row = (by * self.token_blocks + warp // self.hidden_blocks) * 32
        col = (bx * self.hidden_blocks + warp % self.hidden_blocks) * 128 + lane * 4
        if cutlass.const_expr(True):
            live = offsets[self.groups] // self.hidden
            if row < live and col < self.hidden:
                owner = cutlass.Int32(0)
                for g in cutlass.range(1, self.groups, 1):
                    if cutlass.Int64(row * self.hidden) >= offsets[g]:
                        owner = g
                start = cutlass.Int32(offsets[owner] // self.hidden)
                size = cutlass.Int32((offsets[owner + 1] - offsets[owner]) // self.hidden)
                src_u32 = cute.recast_ptr(src.iterator, dtype=cutlass.Int32)
                dst_u32 = cute.recast_ptr(dst.iterator, dtype=cutlass.Int32)
                d = [[None] * 2 for _ in range(32)]
                a0, a1 = cutlass.Int32(0), cutlass.Int32(0)
                for t in cutlass.range_constexpr(0, 32, 1):
                    index = (row + t) * self.hidden + col
                    word = cute.make_tensor(src_u32 + index // 4, cute.make_layout((1,)))[0]
                    raw = cutlass.Int32(sf[(row + t) * (self.hidden // 32) + col // 32]) & 255
                    lo = cvt_scaled_up_bf16x2(word, raw | (raw << 8), 0, cutlass.Float8E4M3FN)
                    hi = cvt_scaled_up_bf16x2(word, raw | (raw << 8), 1, cutlass.Float8E4M3FN)
                    d[t][0], d[t][1] = lo, hi
                    a0 = max_xorsign_abs_bf16x2(a0, lo)
                    a1 = max_xorsign_abs_bf16x2(a1, hi)
                    if lane % 8 == 0:
                        sf_index = (((row + t) // 128) * (self.hidden // 128) + col // 128) * 512
                        sf_index += ((row + t) % 32) * 16 + (((row + t) % 128) // 32) * 4
                        sf_index += (col // 32) % 4
                        row_sf[sf_index] = cutlass.Uint8(raw)
                a0 = a0 & 0x7FFF7FFF
                a1 = a1 & 0x7FFF7FFF
                raw_scales = [
                    e8m0_raw_from_bf16(a0 & 65535, 8),
                    e8m0_raw_from_bf16((a0 >> 16) & 65535, 8),
                    e8m0_raw_from_bf16(a1 & 65535, 8),
                    e8m0_raw_from_bf16((a1 >> 16) & 65535, 8),
                ]
                inv = [
                    bits_f32(cutlass.max((254 - raw) << 23, cutlass.Int32(0x400000)))
                    for raw in raw_scales
                ]
                for j in cutlass.range_constexpr(0, 4, 1):
                    sf_index = start * (self.hidden // 32)
                    sf_index += ((col + j) // 128 * (size // 128) + (row - start) // 128) * 512
                    sf_index += ((col + j) % 32) * 16 + (((col + j) % 128) // 32) * 4
                    sf_index += ((row - start) // 32) % 4
                    col_sf[sf_index] = cutlass.Uint8(raw_scales[j])
                for t in cutlass.range_constexpr(0, 32, 1):
                    q0 = cvt_dn_fp8x2_portable(d[t][0], inv[0], inv[1], cutlass.Float8E4M3FN)
                    q1 = cvt_dn_fp8x2_portable(d[t][1], inv[2], inv[3], cutlass.Float8E4M3FN)
                    index = (row + t) * (self.hidden // 4) + col // 4
                    cute.make_tensor(dst_u32 + index, cute.make_layout((1,)))[0] = (q0 & 65535) | (
                        q1 << 16
                    )
