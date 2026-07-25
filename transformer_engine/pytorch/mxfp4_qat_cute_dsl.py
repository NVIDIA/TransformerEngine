# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""CuTe DSL implementation of MXFP4 weight fake-quantization.

Bit-identical to the CUDA kernel and the PyTorch reference: integer-bit scale
derivation and non-FTZ inline PTX on the value path. Requires the cutlass
CuTe DSL package.
"""
import functools

import torch

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir.dialects import llvm

__all__ = ["mxfp4_fake_quantize_cute_dsl"]

_MXFP4_BLOCK = 32
_THREADS = 256

_ASM_KW = dict(
    has_side_effects=False,
    is_align_stack=False,
    asm_dialect=llvm.AsmDialect.AD_ATT,
)


@dsl_user_op
def _is_nonfinite_f32(x: Float32, *, loc=None, ip=None) -> Int32:
    """1 if x is inf/NaN else 0 (PTX testp.finite)."""
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [Float32(x).ir_value(loc=loc, ip=ip)],
            "{\n  .reg .pred p;\n  testp.finite.f32 p, $1;\n  selp.s32 $0, 0, 1, p;\n}",
            "=r,f",
            loc=loc,
            ip=ip,
            **_ASM_KW,
        )
    )


@dsl_user_op
def _block_scale_from_amax(amax: Float32, *, loc=None, ip=None) -> Float32:
    """scale = 2^clamp(ceil(log2(amax/6)), -126, 125); amax == 0 -> 1.0."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(amax).ir_value(loc=loc, ip=ip)],
            "{\n"
            "  .reg .s32 b, k, mnt, e, t;\n"
            "  .reg .pred pz, pm, psub;\n"
            "  setp.eq.f32 pz, $1, 0f00000000;\n"
            "  mov.b32 b, $1;\n"
            "  shr.s32 k, b, 23;\n"
            "  and.b32 mnt, b, 8388607;\n"
            "  setp.gt.s32 pm, mnt, 4194304;\n"
            "  selp.s32 t, 1, 0, pm;\n"
            "  add.s32 e, k, -129;\n"
            "  add.s32 e, e, t;\n"
            "  setp.eq.s32 psub, k, 0;\n"
            "  selp.s32 e, -126, e, psub;\n"
            "  max.s32 e, e, -126;\n"
            "  min.s32 e, e, 125;\n"
            "  add.s32 e, e, 127;\n"
            "  shl.b32 e, e, 23;\n"
            "  mov.b32 $0, e;\n"
            "  @pz mov.f32 $0, 0f3F800000;\n"
            "}",
            "=f,f",
            loc=loc,
            ip=ip,
            **_ASM_KW,
        )
    )


@dsl_user_op
def _fake_quant_elem(x: Float32, scale: Float32, nonfinite: Int32, *, loc=None, ip=None) -> Float32:
    """RTNE onto the E2M1 grid at the given power-of-two scale; nonfinite -> NaN."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [
                Float32(x).ir_value(loc=loc, ip=ip),
                Float32(scale).ir_value(loc=loc, ip=ip),
                Int32(nonfinite).ir_value(loc=loc, ip=ip),
            ],
            "{\n"
            "  .reg .f32 y, ay, f, m, c, q, r, inv;\n"
            "  .reg .s32 ib;\n"
            "  .reg .pred p2, p4, pnf;\n"
            "  setp.ne.s32 pnf, $3, 0;\n"
            "  mov.b32 ib, $2;\n"
            "  sub.s32 ib, 2130706432, ib;\n"
            "  mov.b32 inv, ib;\n"
            "  mul.rn.f32 y, $1, inv;\n"
            "  abs.f32 ay, y;\n"
            "  min.f32 ay, ay, 0f40C00000;\n"
            "  mul.rn.f32 f, ay, 0f40000000;\n"
            "  cvt.rni.f32.f32 f, f;\n"
            "  mul.rn.f32 f, f, 0f3F000000;\n"
            "  cvt.rni.f32.f32 m, ay;\n"
            "  mul.rn.f32 c, ay, 0f3F000000;\n"
            "  cvt.rni.f32.f32 c, c;\n"
            "  mul.rn.f32 c, c, 0f40000000;\n"
            "  setp.le.f32 p2, ay, 0f40000000;\n"
            "  setp.le.f32 p4, ay, 0f40800000;\n"
            "  selp.f32 q, m, c, p4;\n"
            "  selp.f32 q, f, q, p2;\n"
            "  copysign.f32 r, $1, q;\n"
            "  mul.rn.f32 r, r, $2;\n"
            "  selp.f32 $0, 0f7FC00000, r, pnf;\n"
            "}",
            "=f,f,f,r",
            loc=loc,
            ip=ip,
            **_ASM_KW,
        )
    )


class _MXFP4FakeQuantCuteDsl:
    """Thread-per-1x32-block kernel; loads/stores via autovec_copy fragments."""

    def __init__(self, dtype):
        self.dtype = dtype

    @cute.jit
    def __call__(self, mX: cute.Tensor, mO: cute.Tensor, num_blocks: Int32, stream):
        self.kernel(mX, mO, num_blocks).launch(
            grid=[cute.ceil_div(num_blocks, _THREADS), 1, 1],
            block=[_THREADS, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(self, mX: cute.Tensor, mO: cute.Tensor, num_blocks: Int32):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        idx = bidx * _THREADS + tidx
        if idx < num_blocks:
            x_row = mX[idx, None]
            o_row = mO[idx, None]
            x_frag = cute.make_fragment_like(x_row)
            cute.autovec_copy(x_row, x_frag)
            o_frag = cute.make_fragment_like(o_row)

            amax = Float32(0.0)
            nonfinite = Int32(0)
            for i in cutlass.range_constexpr(_MXFP4_BLOCK):
                v = Float32(x_frag[i])
                nonfinite = nonfinite + _is_nonfinite_f32(v)
                amax = cute.arch.fmax(amax, cute.arch.fmax(v, Float32(0.0) - v))

            scale = _block_scale_from_amax(amax)
            for i in cutlass.range_constexpr(_MXFP4_BLOCK):
                v = Float32(x_frag[i])
                o_frag[i] = self.dtype(_fake_quant_elem(v, scale, nonfinite))
            cute.autovec_copy(o_frag, o_row)


@functools.lru_cache(maxsize=4)
def _compiled_kernel(dtype_key: str):
    cutlass_dtype = cutlass.BFloat16 if dtype_key == "bf16" else cutlass.Float32
    kernel_obj = _MXFP4FakeQuantCuteDsl(cutlass_dtype)
    sym_blocks = cute.sym_int()
    x_fake = cute.runtime.make_fake_compact_tensor(
        cutlass_dtype, (sym_blocks, _MXFP4_BLOCK), stride_order=(1, 0), assumed_align=16
    )
    o_fake = cute.runtime.make_fake_compact_tensor(
        cutlass_dtype, (sym_blocks, _MXFP4_BLOCK), stride_order=(1, 0), assumed_align=16
    )
    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    return cute.compile(
        kernel_obj, x_fake, o_fake, Int32(1), stream_fake, options="--enable-tvm-ffi"
    )


def mxfp4_fake_quantize_cute_dsl(weight: torch.Tensor) -> torch.Tensor:
    """Project ``weight`` onto the MXFP4 grid and return it in the input dtype."""
    if not weight.is_cuda:
        raise ValueError("MXFP4 QAT CuTe DSL kernel expects a CUDA tensor")
    if weight.dim() != 2:
        raise ValueError(f"MXFP4 QAT expects a 2D weight, got {tuple(weight.shape)}")
    if weight.dtype not in (torch.bfloat16, torch.float32):
        raise ValueError(
            f"MXFP4 QAT supports bf16/fp32 weights, got {weight.dtype} "
            "(fp16 cannot represent the full MXFP4 scale range)"
        )
    rows, cols = weight.shape
    if cols % _MXFP4_BLOCK != 0:
        raise ValueError(
            f"MXFP4 QAT needs the weight inner dim divisible by {_MXFP4_BLOCK}, got {cols}"
        )
    w = weight.contiguous()
    if w.data_ptr() % 16 != 0:
        w = w.clone()
    out = torch.empty_like(w)
    num_blocks = w.numel() // _MXFP4_BLOCK
    fn = _compiled_kernel("bf16" if weight.dtype == torch.bfloat16 else "fp32")
    fn(w.view(-1, _MXFP4_BLOCK), out.view(-1, _MXFP4_BLOCK), num_blocks)
    return out
