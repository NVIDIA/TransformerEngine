# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""EP weight-gradient requantizer for the received rows.

When the copy used in the forward is quantized differently from the operand the backward weight
gradient needs, this re-quantizes the received rows into that backward layout. Unlike
``nvte_group_requantize``, it writes the columnwise operand with scales swizzled per expert. Only
MXFP8 (rowwise to columnwise) is supported.

Rowwise MXFP8 rows ([rows, cols] E4M3 data with compact E8M0 scales, one per 32 columns of a row)
become columnwise MXFP8 ([rows, cols] E4M3 data with one scale per 32 rows of a column, scales
GEMM-swizzled per expert slab), the operand layout of the MXFP8 weight gradients. Each value is
dequantized and quantized again with TE's MXFP8 rule. Operands: ``codes``, ``row_scales``,
``col_codes``, ``col_scales``, ``expert_row_ends`` [experts] int32 (every slab a multiple of 128
rows), optional ``signal`` [rows / ready_slot_rows] int32, stream.

A launch is one CTA per (128-row, 256-column) tile of the receive capacity, a programmatic dependent
of the kernel before it that lets the kernel behind it start at once. CTAs past the last expert row
end exit; with ``signal``, a tile starts once the dispatch has set the slots of its rows. The grid
completes after the kernel before it.
"""

import math
from dataclasses import dataclass
from typing import Optional

import cuda.bindings.driver as cuda
import cutlass
from cutlass import Float32, Float8E4M3FN, Int32, Int64, Uint8, Uint16, Uint32, cute
from cutlass.experimental.primitives import cvt_packfloat_f32

from ._loader import (
    compile_launch,
    SF_VEC_SIZE,
    fake_tensor,
    fp8_fake,
    import_gemm_modules,
    register_function,
)

# A thread's rows of a 32-row block, its bytes per row and the warps per CTA.
ROWS_PER_THREAD, VEC, NUM_WARPS = 16, 8, 2
# Lanes that share a group of VEC columns, each with its own rows of a block.
ROW_LANES = SF_VEC_SIZE // ROWS_PER_THREAD
# Columns per tile.
TILE_COLUMNS = 32 // ROW_LANES * VEC * NUM_WARPS
# Scale blocks along the rows per tile: one 128-row scale atom.
BLOCKS_PER_TILE = 4
# Rows per tile.
TILE_ROWS = BLOCKS_PER_TILE * SF_VEC_SIZE
# fp32(1 / 448), the reciprocal of the largest E4M3 value.
E4M3_MAX_RCP = 1.0 / 448.0
# 2^-127, the multiplier of E8M0 0.
E8M0_ZERO_MULTIPLIER = 2.0**-127
# Order of a thread's four column pairs. Any order gives the same bytes; this one takes 128
# registers, which fits 8 CTAs per SM.
PAIR_ORDER = (0, 3, 1, 2)


@dataclass(frozen=True)
class Mxfp8RequantConfig:
    """Static configuration of one compiled MXFP8 requant launch."""

    rows: int  # receive capacity in rows
    cols: int  # columns
    num_experts: int  # local experts
    ready_slot_rows: int = 0  # wait until signal[] is set for every slot of a tile's rows


class _Mxfp8RequantKernel:
    """MXFP8 requant of a ``cfg`` operand, one CTA per tile."""

    def __init__(self, cfg: Mxfp8RequantConfig):
        self.cfg = cfg
        if cfg.ready_slot_rows:
            scheduler = import_gemm_modules("moe_streaming_input_block_tile_scheduler")[0]
            self.poll_until_at_least = scheduler._poll_until_at_least
            self.wait_slots_ready = scheduler.wait_slots_ready

    @cute.jit
    def __call__(
        self,
        codes: cute.Pointer,
        row_scales: cute.Pointer,
        col_codes: cute.Pointer,
        col_scales: cute.Pointer,
        expert_row_ends: cute.Pointer,
        signal: Optional[cute.Pointer],
        stream: cuda.CUstream,
    ):
        self.tile_kernel(codes, row_scales, col_codes, col_scales, expert_row_ends, signal).launch(
            grid=[self.cfg.rows // TILE_ROWS * (self.cfg.cols // TILE_COLUMNS), 1, 1],
            block=[32 * NUM_WARPS, 1, 1],
            stream=stream,
            use_pdl=True,
        )

    @cute.kernel
    def tile_kernel(
        self,
        codes: cute.Pointer,
        row_scales: cute.Pointer,
        col_codes: cute.Pointer,
        col_scales: cute.Pointer,
        expert_row_ends: cute.Pointer,
        signal: Optional[cute.Pointer],
    ):
        """Tile ``blockIdx.x`` if it ends at or before the last expert end; CTA 0 waits for the
        kernel before it at exit, so the grid completes after that one."""
        num_experts = self.cfg.num_experts
        cute.arch.griddepcontrol_launch_dependents()
        ends = cute.make_tensor(expert_row_ends, cute.make_layout((num_experts,)))
        num_tiles = ends[num_experts - 1] // TILE_ROWS * (self.cfg.cols // TILE_COLUMNS)
        bidx, _, _ = cute.arch.block_idx()
        if bidx < num_tiles:
            if cutlass.const_expr(signal is not None):
                self.wait_rows_ready(signal, Int32(bidx))
            self.tile(codes, row_scales, col_codes, col_scales, expert_row_ends, Int32(bidx))
        if bidx == 0:
            cute.arch.griddepcontrol_wait()

    @cute.jit
    def wait_rows_ready(self, signal: cute.Pointer, tile: Int32):
        """Wait until the dispatch has set the ``signal`` slots of the rows of tile ``tile``: warp 0
        polls them relaxed, then acquires them once."""
        slot_rows = self.cfg.ready_slot_rows
        num_slots = max(TILE_ROWS // slot_rows, 1)
        tidx, _, _ = cute.arch.thread_idx()
        if tidx < 32:
            slots = signal + tile // (self.cfg.cols // TILE_COLUMNS) * TILE_ROWS // slot_rows
            for j in cutlass.range_constexpr((num_slots + 31) // 32):
                self.poll_until_at_least(slots + cutlass.min(tidx + j * 32, num_slots - 1), 0)
            self.wait_slots_ready(slots, Int32(num_slots))
        cute.arch.sync_threads()

    @cute.jit
    def tile(
        self,
        codes: cute.Pointer,
        row_scales: cute.Pointer,
        col_codes: cute.Pointer,
        col_scales: cute.Pointer,
        expert_row_ends: cute.Pointer,
        tile: Int32,
    ):
        """Requantize tile ``tile`` and write its scale words."""
        # pylint: disable=too-many-locals
        R, V, COLS = ROWS_PER_THREAD, VEC, self.cfg.cols
        num_experts = self.cfg.num_experts
        tidx, _, _ = cute.arch.thread_idx()
        # A thread owns VEC columns of the tile and every ROW_LANES-th row of each 32-row block.
        lane_row, lane_column = tidx % ROW_LANES, tidx // ROW_LANES
        row_group = tile // (COLS // TILE_COLUMNS)
        column = (tile % (COLS // TILE_COLUMNS)) * TILE_COLUMNS + lane_column * V
        warp_first_column = (tile % (COLS // TILE_COLUMNS)) * TILE_COLUMNS + (tidx // 32) * (R * V)
        access = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), Uint16, num_bits_per_copy=8 * V)
        r_codes = cute.make_rmem_tensor((V // 2, R), Uint16)
        r_out = cute.make_rmem_tensor((V // 2, R), Uint16)
        pair_row, pair_broadcast = cute.make_layout((2,)), cute.make_layout((2,), stride=(0,))
        codes_e4m3 = cute.recast_tensor(r_codes, Float8E4M3FN)
        out_e4m3 = cute.recast_tensor(r_out, Float8E4M3FN)
        r_values = cute.make_rmem_tensor((2, R), Float32)
        r_scales = cute.make_rmem_tensor((R,), Uint8)
        row_multiplier_bits = cute.make_rmem_tensor((R,), Int32)
        row_multiplier = cute.recast_tensor(row_multiplier_bits, Float32)
        multiplier_bits = cute.make_rmem_tensor((2,), Int32)
        multiplier = cute.recast_tensor(multiplier_bits, Float32)
        scaled_amax = cute.make_rmem_tensor((2,), Float32)
        # One word per column pair: this lane's column of the pair, one scale byte per 32-row block.
        packed = cute.make_rmem_tensor((V // 2,), Uint32)
        packed.fill(0)
        for block in cutlass.range(BLOCKS_PER_TILE, unroll=1):
            first_row = Int64(row_group) * TILE_ROWS + block * SF_VEC_SIZE + lane_row
            rows = cute.make_layout((V // 2, R), stride=(1, ROW_LANES * COLS // 2))
            offset = cute.assume(first_row * COLS + column, divby=V)
            cute.copy(
                access,
                cute.make_tensor(cute.recast_ptr(codes + offset, dtype=Uint16), rows),
                r_codes,
            )
            # Prefetch the next block's codes and row scales into L2.
            if block < BLOCKS_PER_TILE - 1:
                next_row = Int64(row_group) * TILE_ROWS + (block + 1) * SF_VEC_SIZE + tidx % 32
                cute.arch.prefetch(
                    (codes + (next_row * COLS + warp_first_column)).llvm_ptr, cache_level="L2"
                )
                cute.arch.prefetch(
                    (
                        row_scales
                        + (next_row * (COLS // SF_VEC_SIZE) + warp_first_column // SF_VEC_SIZE)
                    ).llvm_ptr,
                    cache_level="L2",
                )
            cute.autovec_copy(
                cute.make_tensor(
                    row_scales + (first_row * (COLS // SF_VEC_SIZE) + column // SF_VEC_SIZE),
                    cute.make_layout((R,), stride=(ROW_LANES * (COLS // SF_VEC_SIZE),)),
                ),
                r_scales,
            )
            # 2^(s - 127) from its bits; s = 0 gives 2^-127 and s = 255 NaN.
            for i in cutlass.range_constexpr(R):
                row_multiplier_bits[i] = Int32(r_scales[i]) << 23
                row_multiplier[i] = cute.arch.fmax(row_multiplier[i], E8M0_ZERO_MULTIPLIER)
                row_multiplier_bits[i] = row_multiplier_bits[i] | (
                    ((Int32(r_scales[i]) + 1) >> 8) << 22
                )
            for q in PAIR_ORDER:
                for i in cutlass.range_constexpr(R):
                    pair_codes = cute.make_tensor(codes_e4m3.iterator + 2 * q + V * i, pair_row)
                    pair_row_multiplier = cute.make_tensor(
                        row_multiplier.iterator + i, pair_broadcast
                    )
                    cute.make_tensor(r_values.iterator + 2 * i, pair_row).store(
                        pair_codes.load().to(Float32) * pair_row_multiplier.load()
                    )
                for v in cutlass.range_constexpr(2):
                    # The amax ignores NaN; the column's other rows sit in the neighboring lanes.
                    amax = Float32(0.0)
                    for i in cutlass.range_constexpr(R):
                        amax = cute.arch.fmax(amax, abs(r_values[v, i]))
                    for k in cutlass.range_constexpr(int(math.log2(ROW_LANES))):
                        amax = cute.arch.fmax(
                            amax, cute.arch.shuffle_sync_bfly(amax, offset=1 << k)
                        )
                    scaled_amax[v] = amax * E4M3_MAX_RCP
                # E8M0 of both columns of the pair, rounded up, the first column in the high byte.
                pair = cvt_packfloat_f32(
                    scaled_amax[0], scaled_amax[1], 0, "ue8m0x2", rnd="rp", sat="satfinite"
                )
                packed[q] = (packed[q] >> 8) | (Uint32(pair >> (8 * (1 - lane_row))) << 24)
                inverse = 0xFEFE - pair
                for h in cutlass.range_constexpr(2):
                    multiplier_bits[h] = cutlass.max(
                        ((inverse >> (8 * (1 - h))) & 0xFF) << 23, Int32(0x00400000)
                    )
                for i in cutlass.range_constexpr(R):
                    pair_values = cute.make_tensor(r_values.iterator + 2 * i, pair_row)
                    cute.make_tensor(out_e4m3.iterator + 2 * q + V * i, pair_row).store(
                        (pair_values.load() * multiplier.load()).to(Float8E4M3FN)
                    )
            cute.copy(
                access,
                r_out,
                cute.make_tensor(cute.recast_ptr(col_codes + offset, dtype=Uint16), rows),
            )
        # The tile's expert by a branchless lower bound over the row ends, then its start row.
        ends = cute.make_tensor(expert_row_ends, cute.make_layout((num_experts,)))
        group_row = row_group * TILE_ROWS
        expert, n = Int32(0), num_experts
        for _ in cutlass.range_constexpr(int(math.ceil(math.log2(num_experts)))):
            half = n // 2
            expert = cutlass.select_(ends[expert + half - 1] <= group_row, expert + half, expert)
            n -= half
        expert = cutlass.select_(ends[expert] <= group_row, expert + 1, expert)
        if expert < num_experts:
            start = cutlass.select_(expert > 0, ends[cutlass.max(expert - 1, Int32(0))], Int32(0))
            count = ends[expert] - start
            local_group = Int64(row_group - start // TILE_ROWS)
            # Int32 word of the expert's swizzled scales: its start, the 128-column atom, the
            # 128-row group in it, the column in its 32 and which of the four 32-column groups.
            c = column + lane_row
            index = (
                Int64(start) * (COLS // 128)
                + Int64(c // 128) * count
                + local_group * 128
                + (c % 32) * 4
                + (c // 32) % 4
            )
            for j in cutlass.range_constexpr(V // 2):
                word = cute.make_tensor(
                    col_scales + index + j * ROW_LANES * 4, cute.make_layout((1,))
                )
                word[0] = packed[j]


def _compile(cfg: Mxfp8RequantConfig):
    """Compile the launch wrapper of ``cfg`` with TVM-FFI."""
    assert cfg.rows % TILE_ROWS == 0 and cfg.cols % TILE_COLUMNS == 0, (cfg.rows, cfg.cols)
    slot_rows = cfg.ready_slot_rows
    # A tile waits for whole slots: one, or the slots it spans; the capacity is whole slots.
    assert not slot_rows or (
        (TILE_ROWS % slot_rows == 0 or slot_rows % TILE_ROWS == 0) and cfg.rows % slot_rows == 0
    ), (cfg.rows, slot_rows)
    requant = _Mxfp8RequantKernel(cfg)

    @cute.jit
    def launch(codes, row_scales, col_codes, col_scales, expert_row_ends, stream):
        requant(
            cute.recast_ptr(codes.iterator, dtype=Uint8),
            cute.recast_ptr(row_scales.iterator, dtype=Uint8),
            cute.recast_ptr(col_codes.iterator, dtype=Uint8),
            cute.recast_ptr(col_scales.iterator, dtype=Int32),
            expert_row_ends.iterator,
            None,
            stream,
        )

    @cute.jit
    def launch_ready(codes, row_scales, col_codes, col_scales, expert_row_ends, signal, stream):
        requant(
            cute.recast_ptr(codes.iterator, dtype=Uint8),
            cute.recast_ptr(row_scales.iterator, dtype=Uint8),
            cute.recast_ptr(col_codes.iterator, dtype=Uint8),
            cute.recast_ptr(col_scales.iterator, dtype=Int32),
            expert_row_ends.iterator,
            signal.iterator,
            stream,
        )

    rows, cols = cfg.rows, cfg.cols
    scales = rows * cols // SF_VEC_SIZE
    signal_fake = (fake_tensor(Int32, (rows // slot_rows,), (0,), 4),) if slot_rows else ()
    return compile_launch(
        launch_ready if slot_rows else launch,
        fp8_fake((rows, cols), (1, 0)),
        fake_tensor(cutlass.Float8E8M0FNU, (scales,), (0,), 16),
        fp8_fake((rows, cols), (1, 0)),
        fake_tensor(cutlass.Float8E8M0FNU, (scales,), (0,), 16),
        fake_tensor(Int32, (cfg.num_experts,), (0,), 16),
        *signal_fake,
    )


def get_ep_requant_function(fn_name, rows, cols, num_experts, ready_slot_rows=0) -> bool:
    """Compile and register the requant of this configuration as ``fn_name``; return whether it
    is. ``ready_slot_rows`` > 0: the function takes ``signal`` before the stream."""
    cfg = Mxfp8RequantConfig(rows, cols, num_experts, ready_slot_rows)
    return register_function(fn_name, lambda: _compile(cfg), "EP MXFP8 requant")
