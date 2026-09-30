# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
# SPDX-License-Identifier: BSD-3-Clause
# Adapted from cuDNN Frontend (Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES).
# The adapted implementation is covered by requantize_mxfp8.LICENSE.txt.

"""Grouped MXFP8 requantization adapted from cuDNN Frontend's Mxfp8ColRequant.

Source revision: 347e186e661fef677f78b2a576c9a64cb2e0a8b2, path
python/cudnn/moe_ep/_megamoe_backend/cutedsl_src/kernel_src/rubin/training/mega/
fwd_glu/glu_mxfp8_col_requant.py.

The input scales use TE's compact rowwise layout. One TMA pipeline loads both
payload and scales; consumers decode to BF16, compute columnwise scales and
emit row-major E4M3 payloads. Scale layouts and optional rowwise scale output
follow nvte_grouped_requantize. No framework is imported by this module.
"""

import logging
import cuda.bindings.driver as cuda
import cutlass
from cutlass import cute
from cutlass.cute.nvgpu import cpasync
from cutlass.cutlass_dsl import Float32, Int32, Int64, T, Uint8, dsl_user_op
from cutlass._mlir.dialects import arith, llvm
from cutlass.cute.typing import AddressSpace
from cutlass.memory import SmemAllocator
import tvm_ffi

logger = logging.getLogger("transformer_engine.cutedsl.requantize")


def _address_value(pointer_or_address, *, loc=None, ip=None):
    if isinstance(pointer_or_address, Int64):
        return pointer_or_address.ir_value()
    return pointer_or_address.toint(loc=loc, ip=ip).ir_value()


@dsl_user_op
def tma_load_1d(
    destination_smem,
    source_gmem,
    mbarrier_smem,
    num_bytes,
    *,
    loc=None,
    ip=None,
) -> None:
    """Issue a 1D GMEM-to-SMEM bulk copy."""
    llvm.inline_asm(
        None,
        [
            destination_smem.toint(loc=loc, ip=ip).ir_value(),
            _address_value(source_gmem, loc=loc, ip=ip),
            num_bytes.ir_value(),
            mbarrier_smem.toint(loc=loc, ip=ip).ir_value(),
        ],
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [$0], [$1], $2, [$3];",
        "r,l,r,r",
        has_side_effects=True,
        asm_dialect=0,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def cp_async_bulk_s2g(destination_gmem, source_smem, num_bytes, *, loc=None, ip=None) -> None:
    """Issue a 1D SMEM-to-GMEM bulk copy; the caller commits the group."""
    llvm.inline_asm(
        None,
        [
            _address_value(destination_gmem, loc=loc, ip=ip),
            source_smem.toint(loc=loc, ip=ip).ir_value(),
            num_bytes.ir_value(),
        ],
        "cp.async.bulk.global.shared::cta.bulk_group [$0], [$1], $2;",
        "l,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


def _fp8x2_mnemonic(fp8_type) -> str:
    if fp8_type is cutlass.Float8E4M3FN:
        return "e4m3x2"
    if fp8_type is cutlass.Float8E5M2:
        return "e5m2x2"
    raise TypeError(f"unsupported FP8 type {fp8_type}")


@dsl_user_op
def cvt_scaled_up_bf16x2(pair_b32, scale_b32, half: int, fp8_type, *, loc=None, ip=None) -> Int32:
    """Two FP8 values + their E8M0 scale -> BF16x2, in one SASS instruction."""
    mn = _fp8x2_mnemonic(fp8_type)
    asm = (
        "{\n"
        " .reg .b16 a0,a1,s0,s1;\n"
        " mov.b32 {a0,a1}, $1;\n"
        " mov.b32 {s0,s1}, $2;\n"
        f" cvt.rn.scaled::n2::ue8m0.bf16x2.{mn} $0, a{half}, s0;\n"
        "}"
    )
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [Int32(pair_b32).ir_value(loc=loc, ip=ip), Int32(scale_b32).ir_value(loc=loc, ip=ip)],
            asm,
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def cvt_dn_fp8x2_portable(v_bf16x2, inv_lo, inv_hi, fp8_type, *, loc=None, ip=None) -> Int32:
    """Two hidden columns of one token, as BF16x2, plus those two columns'
    exact FP32 reciprocal scales -> two FP8 bytes in bits [15:0] (byte 0 from
    the low BF16 half, byte 1 from the high half).

    Same rounding as ``cvt_scaled_dn_fp8x2``, not an approximation of it: BF16
    -> FP32 is an exact left shift, the reciprocal is an exact power of two, and
    ``cvt.rn.satfinite`` is the RNE-and-saturate the hardware instruction
    applies.  Taking a scale per half is what lets the caller skip the transpose
    that the one-scale-per-pair hardware instruction forces.
    """
    mn = _fp8x2_mnemonic(fp8_type)
    asm = (
        "{\n"
        " .reg .b32 a, b;\n"
        " .reg .b16 q;\n"
        " shl.b32 a, $1, 16;\n"
        " and.b32 b, $1, 0xffff0000;\n"
        " mul.f32 a, a, $2;\n"
        " mul.f32 b, b, $3;\n"
        # ``cvt d, a, b`` yields d[15:8] = cvt(a) and d[7:0] = cvt(b), so the
        # HIGH column has to be the first source for byte 0 to be the low one.
        f" cvt.rn.satfinite.{mn}.f32 q, b, a;\n"
        " cvt.u32.u16 $0, q;\n"
        "}"
    )
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [
                Int32(v_bf16x2).ir_value(loc=loc, ip=ip),
                Float32(inv_lo).ir_value(loc=loc, ip=ip),
                Float32(inv_hi).ir_value(loc=loc, ip=ip),
            ],
            asm,
            "=r,r,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def max_xorsign_abs_bf16x2(a, b, *, loc=None, ip=None) -> Int32:
    """Packed magnitude max of two BF16x2."""
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [Int32(a).ir_value(loc=loc, ip=ip), Int32(b).ir_value(loc=loc, ip=ip)],
            "max.xorsign.abs.bf16x2 $0, $1, $2;",
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def e8m0_raw_from_bf16(bf16_bits, limit_exponent: int, *, loc=None, ip=None) -> Int32:
    """E8M0 raw byte for a non-negative BF16 magnitude."""
    bits = Int32(bf16_bits) << Int32(16)
    biased = (bits + Int32(0x1FFFFF - (limit_exponent << 23))) >> Int32(23)
    return Int32(
        arith.select(
            (bits >= Int32(0x7F800000)).ir_value(loc=loc, ip=ip),
            Int32(254).ir_value(loc=loc, ip=ip),
            cutlass.max(Int32(0), biased).ir_value(loc=loc, ip=ip),
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def bits_f32(x: Int32, *, loc=None, ip=None) -> Float32:
    """Interpret an integer register as IEEE FP32 bits."""
    return Float32(llvm.bitcast(T.f32(), Int32(x).ir_value(loc=loc, ip=ip), loc=loc, ip=ip))


class GroupedRequantize:
    """Persistent, warp-specialized TMA requantization with compact input scales.

    A consumer lane owns four columns and 32 tokens. FP8 decoding and amax
    reduction stay in BF16 registers. Payloads remain row-major. Both GEMM
    scale layouts are emitted without a separate swizzle kernel.
    """

    TokensPerBlock = 32
    TILE_TOK = 128
    SfAtomNonK = 128
    SfAtomBytes = 512
    SfOutStride = 528
    ConsumerBarrierId = 1
    ProducerWarps = 1

    def __init__(
        self,
        hidden,
        num_groups,
        capacity,
        *,
        input_dtype=cutlass.Float8E4M3FN,
        swizzled=True,
        rowwise_output=True,
        uniform_rows=0,
        tile_hidden=None,
        stages=2,
        grid=None,
        sm_count=152,
    ):
        if hidden <= 0 or hidden % 128 or capacity <= 0 or capacity % 128 or num_groups < 1:
            raise ValueError(
                "Grouped MXFP8 requantization requires positive 128-aligned dimensions"
            )
        if input_dtype not in (cutlass.Float8E4M3FN, cutlass.Float8E5M2):
            raise ValueError("Input must be E4M3 or E5M2")
        if hidden >= 512 and hidden % 512:
            raise ValueError("Wide compact scale matrices require a 16-byte TMA row stride")
        if uniform_rows and (uniform_rows % 128 or uniform_rows * num_groups != capacity):
            raise ValueError("Uniform groups must be 128-aligned and fill the row capacity")
        self.hidden = int(hidden)
        self.num_experts = int(num_groups)
        self.quant_dtype = input_dtype
        self.sf_dtype = cutlass.Uint8
        self.swizzled = bool(swizzled)
        self.rowwise_output = bool(rowwise_output and swizzled)
        self.uniform_rows = int(uniform_rows)
        self.TILE_HID = tile_hidden or (128 if hidden == 128 else 256)
        if hidden % self.TILE_HID:
            self.TILE_HID = 128
        self.ColsPerLane = 4
        self.sp_lw = 4
        self.NumStages = stages
        self.TmaBoxHidU32 = self.TILE_HID // 4
        self.HidAtomsPerTile = self.TILE_HID // 128
        self.HidSegs = self.TILE_HID // 128
        self.ConsumerWarps = 4 * self.HidSegs
        self.ThreadsPerCta = (1 + self.ConsumerWarps) * 32
        self.MaxNTid = max(576, self.ThreadsPerCta)
        self._hidden_atoms = hidden // 128
        self.hidden_groups = hidden // self.TILE_HID
        # TMA requires a 16-byte row stride. Narrow full-width matrices use a
        # contiguous 1D bulk copy. Other tiles load at least 16 scale columns;
        # consumers ignore the extra columns (TMA zero-fills the final tile).
        self.scale_stride = hidden // 32 if hidden < 512 else max(16, self.TILE_HID // 32)
        self.SfTileBytes = 128 * self.scale_stride
        self.smem_sf_out_bytes = self.HidAtomsPerTile * self.SfOutStride
        self._data_limit_exponent = 8  # Output is always E4M3, including E5M2 input.
        tasks = capacity // 128 * self.hidden_groups
        self.grid = grid or min(tasks, max(1, min(24, tasks // (4 * 2 * sm_count))) * 2 * sm_count)
        span = 1
        while span < num_groups:
            span <<= 1
        self._search_steps = tuple(1 << i for i in reversed(range(span.bit_length() - 1)))
        self._search_needs_guard = (num_groups & (num_groups - 1)) != 0
        self.smem_bytes = (
            stages * (128 * self.TILE_HID + self.SfTileBytes)
            + self.smem_sf_out_bytes
            + (self.HidAtomsPerTile * 512 if self.rowwise_output else 0)
            + (num_groups + 1) * 4
            + 2 * stages * 8
            + 384
        )
        if self.smem_bytes > 227 * 1024:
            raise ValueError("Grouped requantization exceeds the shared-memory budget")

    @cute.jit
    def __call__(
        self,
        src_data: cute.Tensor,
        src_sf: cute.Tensor,
        offsets: cute.Tensor,
        dst_data: cute.Tensor,
        dst_row_sf: cute.Tensor,
        dst_col_sf: cute.Tensor,
        stream: cuda.CUstream,
    ):
        tok = cutlass.const_expr(self.TILE_TOK)
        width = cutlass.const_expr(self.TILE_HID // 4)
        src = cute.make_tensor(
            cute.recast_ptr(src_data.iterator, dtype=cutlass.Uint32),
            cute.make_layout((src_data.shape[0], self.hidden // 4), stride=(self.hidden // 4, 1)),
        )
        dst = cute.make_tensor(cute.recast_ptr(dst_data.iterator, dtype=cutlass.Uint32), src.layout)
        data_layout = cute.make_layout((tok, width), stride=(width, 1))
        load_atom, load_tensor = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(), src, data_layout, (tok, width)
        )
        store_atom, store_tensor = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(), dst, data_layout, (tok, width)
        )
        scale_atom, scale_tensor = None, None
        if cutlass.const_expr(self.hidden >= 512):
            scales = cute.make_tensor(
                cute.recast_ptr(src_sf.iterator, dtype=cutlass.Uint8),
                cute.make_layout(
                    (src_data.shape[0], self.hidden // 32), stride=(self.hidden // 32, 1)
                ),
            )
            scale_layout = cute.make_layout((tok, self.scale_stride), stride=(self.scale_stride, 1))
            scale_atom, scale_tensor = cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileG2SOp(), scales, scale_layout, (tok, self.scale_stride)
            )
        self.ws_kernel(
            src_sf,
            offsets,
            dst_row_sf,
            dst_col_sf,
            load_atom,
            load_tensor,
            store_atom,
            store_tensor,
            scale_atom,
            scale_tensor,
        ).launch(
            grid=(self.grid, 1, 1),
            block=(self.ThreadsPerCta, 1, 1),
            max_number_threads=(self.MaxNTid, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def ws_kernel(
        self,
        src_sf: cute.Tensor,
        offsets: cute.Tensor,
        dst_row_sf: cute.Tensor,
        dst_col_sf: cute.Tensor,
        load_atom=None,
        load_tensor=None,
        store_atom=None,
        store_tensor=None,
        scale_atom=None,
        scale_tensor=None,
    ):
        """Pipeline TMA loads, consumer requantization and fused scale stores."""
        tok = cutlass.const_expr(self.TILE_TOK)
        width = cutlass.const_expr(self.TILE_HID)
        stages = cutlass.const_expr(self.NumStages)
        sf_bytes = cutlass.const_expr(self.SfTileBytes)
        tid, _, _ = cute.arch.thread_idx()
        bid, _, _ = cute.arch.block_idx()
        grid, _, _ = cute.arch.grid_dim()
        warp, lane = tid // 32, tid % 32
        smem = SmemAllocator()
        full = smem.allocate_array(cutlass.Int64, stages)
        empty = smem.allocate_array(cutlass.Int64, stages)
        table = smem.allocate_tensor(cutlass.Int32, cute.make_layout((self.num_experts + 1,)), 16)
        sf_in = smem.allocate_array(cutlass.Uint8, stages * sf_bytes, byte_alignment=128)
        sf_out = smem.allocate_array(cutlass.Uint8, self.smem_sf_out_bytes, byte_alignment=128)
        data = smem.allocate_array(cutlass.Uint8, stages * tok * width, byte_alignment=128)
        row_sf = None
        row_dst = cutlass.Int64(0)
        if cutlass.const_expr(self.rowwise_output):
            row_sf = smem.allocate_array(
                cutlass.Uint8, self.HidAtomsPerTile * 512, byte_alignment=128
            )
            row_dst = dst_row_sf.iterator.toint()
        row_base = cutlass.Int32(0)
        if cutlass.const_expr(self.rowwise_output):
            row_base = row_sf.toint()
        if tid == 0:
            for s in cutlass.range_constexpr(0, stages, 1):
                cute.arch.mbarrier_init(full + s, 1)
                cute.arch.mbarrier_init(empty + s, self.ConsumerWarps)
        cute.arch.mbarrier_init_fence()
        for g in cutlass.range(tid, self.num_experts + 1, self.ThreadsPerCta):
            if cutlass.const_expr(self.uniform_rows > 0):
                table[g] = Int32(g * self.uniform_rows)
            else:
                table[g] = Int32(offsets[g] // self.hidden)
        cute.arch.sync_threads()
        tiles = Int32(table[self.num_experts]) // tok
        if warp == 0:
            self.produce(
                data.toint(),
                sf_in.toint(),
                full,
                empty,
                src_sf.iterator.toint(),
                bid,
                grid,
                tiles,
                lane,
                load_atom,
                load_tensor,
                scale_atom,
                scale_tensor,
            )
        else:
            self.consume_scaled(
                data.toint(),
                sf_in.toint(),
                sf_out.toint(),
                full,
                empty,
                table,
                dst_col_sf.iterator.toint(),
                bid,
                grid,
                tiles,
                warp - 1,
                lane,
                store_atom,
                store_tensor,
                row_sf_base=row_base,
                row_dst=row_dst,
            )

    @cute.jit
    def produce(
        self,
        data_base,
        sf_base,
        full,
        empty,
        src_sf_base,
        bid,
        grid,
        tiles,
        lane,
        data_atom=None,
        data_tensor=None,
        scale_atom=None,
        scale_tensor=None,
    ):
        """Load compact scale and payload tiles into the shared pipeline."""
        tok = cutlass.const_expr(self.TILE_TOK)
        width = cutlass.const_expr(self.TILE_HID)
        box_h = cutlass.const_expr(self.TmaBoxHidU32)
        stages = cutlass.const_expr(self.NumStages)
        sf_bytes = cutlass.const_expr(self.SfTileBytes)
        data = cute.make_tensor(
            cute.make_ptr(cutlass.Uint32, data_base, AddressSpace.smem, assumed_align=128),
            cute.make_layout((tok, box_h, stages), stride=(box_h, 1, tok * box_h)),
        )
        global_data = cute.group_modes(
            cute.local_tile(data_tensor, (tok, box_h), (None, None)), 0, 2
        )
        shared_tile, global_tile = cpasync.tma_partition(
            data_atom, 0, cute.make_layout(1), cute.group_modes(data, 0, 2), global_data
        )
        cpasync.prefetch_descriptor(data_atom)
        if cutlass.const_expr(self.hidden >= 512):
            sf = cute.make_tensor(
                cute.make_ptr(cutlass.Uint8, sf_base, AddressSpace.smem, assumed_align=128),
                cute.make_layout(
                    (tok, self.scale_stride, stages), stride=(self.scale_stride, 1, sf_bytes)
                ),
            )
            # The 16-byte scale box can overlap the next data tile's scales.
            # Its start is selected with a coordinate tensor, rather than
            # local_tile's box-width stepping.
            sf_tile = cute.local_tile(scale_tensor, (tok, self.scale_stride), (None, None))
            sf_shared, _ = cpasync.tma_partition(
                scale_atom,
                0,
                cute.make_layout(1),
                cute.group_modes(sf, 0, 2),
                cute.group_modes(sf_tile, 0, 2),
            )
            cpasync.prefetch_descriptor(scale_atom)
        step = Int32(0)
        work = Int32(bid)
        total = tiles * Int32(self.hidden_groups)
        while work < total:
            stage = step % Int32(stages)
            token_tile = work // Int32(self.hidden_groups)
            hid_begin = (work % Int32(self.hidden_groups)) * Int32(width)
            if step >= Int32(stages):
                cute.arch.mbarrier_wait(empty + stage, ((step // Int32(stages)) - 1) % 2)
            if lane == 0:
                cute.arch.mbarrier_arrive_and_expect_tx(full + stage, Int32(tok * width + sf_bytes))
            cute.arch.sync_warp()
            if cutlass.const_expr(self.hidden >= 512):
                # TMA scale coordinates must be 16-byte aligned. Two adjacent
                # 256-column data tiles share the same 16-column scale box.
                source = cute.domain_offset(
                    (token_tile * tok, (hid_begin // 32) // self.scale_stride * self.scale_stride),
                    scale_tensor,
                )
                tile = cute.local_tile(source, (tok, self.scale_stride), (0, 0))
                _, source_partition = cpasync.tma_partition(
                    scale_atom,
                    0,
                    cute.make_layout(1),
                    cute.group_modes(sf, 0, 2),
                    cute.group_modes(tile, 0, 2),
                )
                cute.copy(
                    scale_atom,
                    source_partition[(None,)],
                    sf_shared[(None, stage)],
                    tma_bar_ptr=full + stage,
                )
            else:
                if lane == 0:
                    tma_load_1d(
                        cute.make_ptr(
                            cutlass.Uint8,
                            sf_base + stage * sf_bytes,
                            AddressSpace.smem,
                            assumed_align=16,
                        ),
                        Int64(src_sf_base) + Int64(token_tile * sf_bytes),
                        full + stage,
                        Int32(sf_bytes),
                    )
            cute.copy(
                data_atom,
                global_tile[(None, token_tile, hid_begin // width)],
                shared_tile[(None, stage)],
                tma_bar_ptr=full + stage,
            )
            step = step + Int32(1)
            work = work + grid

    @cute.jit
    def emit_row_scales(self, src_base, dst_base, cw, lane):
        """Pack compact rowwise scales into shared 128-by-4 GEMM atoms."""
        row = cw * Int32(32) + lane
        if row < Int32(128):
            for a in cutlass.range_constexpr(0, self.HidAtomsPerTile, 1):
                value = cute.make_tensor(
                    cute.make_ptr(
                        cutlass.Int32,
                        src_base + row * self.scale_stride + a * 4,
                        AddressSpace.smem,
                        assumed_align=4,
                    ),
                    cute.make_layout((1,)),
                )[0]
                offset = a * 512 + (row % 32) * 16 + (row // 32) * 4
                cute.make_tensor(
                    cute.make_ptr(
                        cutlass.Int32, dst_base + offset, AddressSpace.smem, assumed_align=4
                    ),
                    cute.make_layout((1,)),
                )[0] = value

    @cute.jit
    def find_expert(self, tbl, key):
        """Find the nonempty group containing a 128-row tile."""
        E = cutlass.const_expr(self.num_experts)
        lo = Int32(0)
        for step in self._search_steps:
            probe = lo + Int32(step)
            if cutlass.const_expr(self._search_needs_guard):
                if probe < Int32(E) and Int32(tbl[probe]) <= key:
                    lo = probe
            elif Int32(tbl[probe]) <= key:
                lo = probe
        return lo

    @cute.jit
    def consume_scaled(
        self,
        smem_data_base,
        smem_sf_in_base,
        smem_sf_out_base,
        mbar_full,
        mbar_empty,
        tbl_data,
        dst_sf_base,
        bidx,
        grid_dim_x,
        total_tiles,
        cw,
        lane_idx,
        tma_atom_st=None,
        tma_tensor_st=None,
        row_sf_base=None,
        row_dst=None,
    ):
        """Decode/requantize and emit TE layouts with shared BF16 registers."""
        # CuTeDSL traces explicit conjunctions; retain these rather than chained comparisons.
        # pylint: disable=chained-comparison
        TOK = cutlass.const_expr(self.TILE_TOK)
        W = cutlass.const_expr(self.TILE_HID)
        S = cutlass.const_expr(self.NumStages)
        SFB = cutlass.const_expr(self.SfTileBytes)
        NB = cutlass.const_expr(self.TokensPerBlock)
        CONS_THREADS = cutlass.const_expr(self.ConsumerWarps * 32)
        HATOMS = cutlass.const_expr(self.HidAtomsPerTile)
        C = cutlass.const_expr(self.ColsPerLane)
        SEGW = cutlass.const_expr(32 * C)
        tb = cw // Int32(self.HidSegs)
        seg = cw % Int32(self.HidSegs)
        LW = cutlass.const_expr(self.sp_lw)
        ldsw = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), cutlass.Int32, num_bits_per_copy=LW * 8
        )
        data_lane_off = tb * Int32(NB * W) + seg * Int32(SEGW) + lane_idx * Int32(LW)
        hb0 = seg * Int32(C) + lane_idx * Int32(LW) // Int32(32)
        sf_lane_off = tb * Int32(32 * self.scale_stride)
        BOX_HS = cutlass.const_expr(self.TmaBoxHidU32)
        sDo = cute.make_tensor(
            cute.make_ptr(cutlass.Uint32, smem_data_base, AddressSpace.smem, assumed_align=128),
            cute.make_layout((TOK, BOX_HS, S), stride=(BOX_HS, 1, TOK * BOX_HS)),
        )
        gDo = cute.group_modes(cute.local_tile(tma_tensor_st, (TOK, BOX_HS), (None, None)), 0, 2)
        tDsDo, tDgDo = cpasync.tma_partition(
            tma_atom_st, 0, cute.make_layout(1), cute.group_modes(sDo, 0, 2), gDo
        )
        cpasync.prefetch_descriptor(tma_atom_st)
        t = Int32(0)
        work_idx = Int32(bidx)
        total_work = total_tiles * Int32(self.hidden_groups)
        while work_idx < total_work:
            stage = t % Int32(S)
            token_tile = work_idx // Int32(self.hidden_groups)
            hid_begin = work_idx % Int32(self.hidden_groups) * Int32(W)
            data_row0 = token_tile * Int32(TOK)
            owner = self.find_expert(tbl_data, data_row0)
            sf_expert_token_atom = Int32(tbl_data[owner]) // Int32(TOK)
            sf_token_atom = (data_row0 - Int32(tbl_data[owner])) // Int32(TOK)
            sf_token_atoms = (Int32(tbl_data[owner + Int32(1)]) - Int32(tbl_data[owner])) // Int32(
                TOK
            )
            cute.arch.mbarrier_wait(mbar_full + stage, t // Int32(S) % Int32(2))
            stage_data = smem_data_base + stage * Int32(TOK * W) + data_lane_off
            stage_sf = smem_sf_in_base + stage * Int32(SFB) + sf_lane_off
            stage_sf = stage_sf + (hid_begin // 32) % self.scale_stride
            sfout = smem_sf_out_base
            if cutlass.const_expr(self.rowwise_output):
                self.emit_row_scales(stage_sf - sf_lane_off, row_sf_base, cw, lane_idx)
            self._sp_tile_body(
                stage_data,
                stage_sf,
                sfout,
                ldsw,
                hb0,
                seg,
                tb,
                lane_idx,
                data_row0,
                hid_begin,
                dst_sf_base,
            )
            cute.arch.fence_proxy("async.shared", space="cta")
            cute.arch.barrier(barrier_id=self.ConsumerBarrierId, number_of_threads=CONS_THREADS)
            if cw == Int32(0):
                cute.copy(
                    tma_atom_st, tDsDo[None, stage], tDgDo[None, token_tile, hid_begin // Int32(W)]
                )
            if (
                cutlass.const_expr(self.swizzled)
                and cw == Int32(0)
                and lane_idx >= Int32(16)
                and (lane_idx < Int32(16) + Int32(HATOMS))
            ):
                atom = lane_idx - Int32(16)
                cp_async_bulk_s2g(
                    Int64(dst_sf_base)
                    + (
                        Int64(sf_expert_token_atom) * Int64(self._hidden_atoms)
                        + (Int64(hid_begin // Int32(self.SfAtomNonK)) + Int64(atom))
                        * Int64(sf_token_atoms)
                        + Int64(sf_token_atom)
                    )
                    * Int64(self.SfAtomBytes),
                    cute.make_ptr(
                        self.sf_dtype,
                        sfout + atom * Int32(self.SfOutStride),
                        AddressSpace.smem,
                        assumed_align=16,
                    ),
                    Int32(self.SfAtomBytes),
                )
            if cutlass.const_expr(self.rowwise_output):
                if (
                    cw == Int32(0)
                    and lane_idx >= Int32(24)
                    and lane_idx < Int32(24 + self.HidAtomsPerTile)
                ):
                    atom = lane_idx - Int32(24)
                    cp_async_bulk_s2g(
                        Int64(row_dst)
                        + (
                            Int64(token_tile) * self._hidden_atoms
                            + Int64(hid_begin // 128)
                            + Int64(atom)
                        )
                        * 512,
                        cute.make_ptr(
                            cutlass.Uint8,
                            row_sf_base + atom * 512,
                            AddressSpace.smem,
                            assumed_align=16,
                        ),
                        Int32(512),
                    )
            if cw == Int32(0):
                cute.arch.cp_async_bulk_commit_group()
            if cw == Int32(0):
                cute.arch.cp_async_bulk_wait_group(0, read=True)
            cute.arch.barrier(barrier_id=self.ConsumerBarrierId, number_of_threads=CONS_THREADS)
            if lane_idx == Int32(0):
                cute.arch.mbarrier_arrive(mbar_empty + stage)
            t = t + Int32(1)
            work_idx = work_idx + grid_dim_x

    @cute.jit
    def _sp_tile_body(
        self,
        stage_data,
        stage_sf_base,
        sfout,
        ldsw,
        hb0,
        seg,
        tb,
        lane_idx,
        data_row0,
        hid_begin,
        col_sf_base,
    ):
        """The single-pass arithmetic for one lane's share of one tile."""
        W = cutlass.const_expr(self.TILE_HID)
        NB = cutlass.const_expr(self.TokensPerBlock)
        C = cutlass.const_expr(self.ColsPerLane)
        LW = cutlass.const_expr(self.sp_lw)
        NWc = cutlass.const_expr(LW // 4)
        NPc = cutlass.const_expr(LW // 2)
        NCH = cutlass.const_expr(C // LW)
        SEGW = cutlass.const_expr(32 * C)
        QT = self.quant_dtype
        for ch in cutlass.range_constexpr(0, NCH, 1):
            base = stage_data + Int32(ch * 32 * LW)
            hbc = hb0 + Int32(ch * LW)
            sfb = stage_sf_base + hbc
            d = [[None] * NPc for _ in range(NB)]
            acc = [Int32(0)] * NPc
            for tt in cutlass.range_constexpr(0, NB, 1):
                words = self._load_words(base + Int32(tt * W), ldsw, NWc, LW)
                raw_sf = self._src_scale_raw(sfb + Int32(tt * self.scale_stride))
                s16 = raw_sf | raw_sf << Int32(8)
                for w in cutlass.range_constexpr(0, NWc, 1):
                    qw = Int32(words[w])
                    lo = cvt_scaled_up_bf16x2(qw, s16, 0, QT)
                    hi = cvt_scaled_up_bf16x2(qw, s16, 1, QT)
                    d[tt][2 * w] = lo
                    d[tt][2 * w + 1] = hi
                    acc[2 * w] = max_xorsign_abs_bf16x2(acc[2 * w], lo)
                    acc[2 * w + 1] = max_xorsign_abs_bf16x2(acc[2 * w + 1], hi)
            raws = [None] * LW
            for k in cutlass.range_constexpr(0, NPc, 1):
                a = acc[k] & Int32(2147450879)
                raws[2 * k] = e8m0_raw_from_bf16(a & Int32(65535), self._data_limit_exponent)
                raws[2 * k + 1] = e8m0_raw_from_bf16(
                    a >> Int32(16) & Int32(65535), self._data_limit_exponent
                )
            invs = [
                bits_f32(cutlass.max(Int32(254) - raws[j] << Int32(23), Int32(4194304)))
                for j in range(LW)
            ]
            col0 = seg * Int32(SEGW) + Int32(ch * 32 * LW) + lane_idx * Int32(LW)
            for j in cutlass.range_constexpr(0, LW, 1):
                col = col0 + Int32(j)
                off = (
                    sfout
                    + col // Int32(128) * Int32(self.SfOutStride)
                    + col % Int32(32) * Int32(16)
                    + col % Int32(128) // Int32(32) * Int32(4)
                    + tb
                )
                if cutlass.const_expr(self.swizzled):
                    out_t = cute.make_tensor(
                        cute.make_ptr(cutlass.Uint8, off, AddressSpace.smem, assumed_align=1),
                        cute.make_layout((1,)),
                    )
                else:
                    offset = (data_row0 // 32 + tb) * self.hidden + hid_begin + col
                    out_t = cute.make_tensor(
                        cute.make_ptr(
                            cutlass.Uint8,
                            Int64(col_sf_base) + Int64(offset),
                            AddressSpace.gmem,
                            assumed_align=1,
                        ),
                        cute.make_layout((1,)),
                    )
                out_t[0] = Uint8(raws[j])
            for tt in cutlass.range_constexpr(0, NB, 2):
                out0, out1 = self._requant_token_pair(d[tt], d[tt + 1], invs, NWc)
                self._store_words(base + Int32(tt * W), ldsw, out0, NWc, LW)
                self._store_words(base + Int32((tt + 1) * W), ldsw, out1, NWc, LW)

    def _requant_token_pair(self, d0, d1, invs, NWc):
        """Requantize two tokens using each column's exact reciprocal scale.

        Inline into the caller's trace: the BF16 register lists cannot cross a
        jitted function boundary. Pair columns to avoid a shared-memory transpose.
        """
        QT = cutlass.Float8E4M3FN
        out0 = cute.make_rmem_tensor((NWc,), cutlass.Int32)
        out1 = cute.make_rmem_tensor((NWc,), cutlass.Int32)
        for w in range(0, NWc, 1):
            k0 = 2 * w
            k1 = 2 * w + 1
            p00 = cvt_dn_fp8x2_portable(d0[k0], invs[2 * k0], invs[2 * k0 + 1], QT)
            p01 = cvt_dn_fp8x2_portable(d0[k1], invs[2 * k1], invs[2 * k1 + 1], QT)
            p10 = cvt_dn_fp8x2_portable(d1[k0], invs[2 * k0], invs[2 * k0 + 1], QT)
            p11 = cvt_dn_fp8x2_portable(d1[k1], invs[2 * k1], invs[2 * k1 + 1], QT)
            out0[w] = p00 & Int32(65535) | p01 << Int32(16)
            out1[w] = p10 & Int32(65535) | p11 << Int32(16)
        return (out0, out1)

    @cute.jit
    def _load_words(self, byte_addr, atom, NW, C):
        """One LDS of C raw bytes (no FP8 decode -- the scaled cvt does that)."""
        regs = cute.make_rmem_tensor((NW,), cutlass.Int32)
        cute.copy(
            atom,
            cute.make_tensor(
                cute.make_ptr(cutlass.Int32, byte_addr, AddressSpace.smem, assumed_align=C),
                cute.make_layout((NW,)),
            ),
            regs,
        )
        return regs

    @cute.jit
    def _store_words(self, byte_addr, atom, regs, NW, C):
        cute.copy(
            atom,
            regs,
            cute.make_tensor(
                cute.make_ptr(cutlass.Int32, byte_addr, AddressSpace.smem, assumed_align=C),
                cute.make_layout((NW,)),
            ),
        )

    @cute.jit
    def _src_scale_raw(self, byte_addr):
        """Raw source E8M0 byte, forced to 0..255.

        The mask is load-bearing: the pointer type says unsigned, but the
        emitted load sign-extends, and the caller packs this into both halves
        of a word with `raw | (raw << 8)`. Without the mask a scale byte >=
        0x80 poisons the E8M0 pair. Removing it took the mega suite from 28/28
        to 0/28.
        """
        return Int32(
            cute.make_tensor(
                cute.make_ptr(cutlass.Uint8, byte_addr, AddressSpace.smem, assumed_align=1),
                cute.make_layout((1,)),
            )[0]
        ) & Int32(255)


def get_mxfp8_requantization_function(
    fn_name, dtype, rows, hidden, groups, uniform_rows, swizzled, rowwise_output, sm_count
):
    """Compile/register one common C API specialization, or request CUDA fallback."""
    if tvm_ffi.get_global_func(fn_name, allow_missing=True) is not None:
        return True
    try:
        input_dtype = {"Float8E4M3": cutlass.Float8E4M3FN, "Float8E5M2": cutlass.Float8E5M2}[dtype]
        kernel = GroupedRequantize(
            hidden,
            groups,
            rows,
            input_dtype=input_dtype,
            uniform_rows=uniform_rows,
            swizzled=swizzled,
            rowwise_output=rowwise_output,
            sm_count=sm_count,
        )

        def tensor(element_type, shape):
            return cute.runtime.make_fake_compact_tensor(
                element_type,
                shape,
                stride_order=tuple(reversed(range(len(shape)))),
                memspace=cute.AddressSpace.gmem,
                assumed_align=16,
            )

        scales = rows * hidden // 32
        compiled = cute.compile(
            kernel,
            tensor(input_dtype, (rows, hidden)),
            tensor(cutlass.Float8E8M0FNU, (rows, hidden // 32)),
            None if uniform_rows else tensor(cutlass.Int64, (groups + 1,)),
            tensor(cutlass.Float8E4M3FN, (rows, hidden)),
            tensor(cutlass.Float8E8M0FNU, (scales,)) if rowwise_output else None,
            tensor(cutlass.Float8E8M0FNU, (scales,)),
            cute.runtime.make_fake_stream(),
            options="--enable-tvm-ffi",
        )
        native = getattr(compiled, "__tvm_ffi_object__", lambda: None)()
        tvm_ffi.register_global_func(
            fn_name, native if native is not None else compiled, override=True
        )
        return True
    except Exception as error:  # pylint: disable=broad-exception-caught
        logger.warning("CuTeDSL grouped requantization unavailable; using CUDA: %s", error)
        return False
