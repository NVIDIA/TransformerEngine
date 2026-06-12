import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.pipeline as pipeline
from cutlass import Float32, Int64, Int32, Int16, Uint8, Uint32
from cutlass._mlir.dialects import arith as mlir_arith
from cutlass._mlir.dialects import llvm
from cutlass.base_dsl.compiler import GPUArch
from cutlass.cute.runtime import make_ptr
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass.cute.arch import cvt_f32_bf16

from types import SimpleNamespace

# FP8E4M3 max representable value
FP8E4M3_MAX_NORM = 448.0
FP8E4M3_MAX_NORM_RCP = 1.0 / FP8E4M3_MAX_NORM
FP8E5M2_MAX_NORM = 57344.0
FP8E5M2_MAX_NORM_RCP = 1.0 / FP8E5M2_MAX_NORM

# NVFP4 (fp4e2m1) — 4-bit float, max representable value is 6.0
FP4_E2M1_MAX = 6.0
FP4_E2M1_MAX_RCP = 1.0 / FP4_E2M1_MAX
# Largest finite f32 — used to clamp the per-block scale inverse against
# division-by-zero (which produces +inf and then NaN downstream).
FP32_MAX = 3.4028234663852886e38

FP32_MANTISSA_BITS = 23


@dsl_user_op
def _bitcast_f32_to_i32(val: Float32, *, loc=None, ip=None) -> Int32:
    return Int32(mlir_arith.bitcast(T.i32(), val.ir_value(loc=loc, ip=ip), loc=loc, ip=ip))


@dsl_user_op
def _bitcast_i32_to_f32(val: Int32, *, loc=None, ip=None) -> Float32:
    return Float32(mlir_arith.bitcast(T.f32(), val.ir_value(loc=loc, ip=ip), loc=loc, ip=ip))


@dsl_user_op
def fabs_f32(val: Float32, *, loc=None, ip=None) -> Float32:
    val_i32 = _bitcast_f32_to_i32(val, loc=loc, ip=ip)
    abs_i32 = val_i32 & Int32(0x7FFFFFFF)
    return _bitcast_i32_to_f32(abs_i32, loc=loc, ip=ip)


@dsl_user_op
def float_to_e8m0(val: Float32, *, loc=None, ip=None) -> Int32:
    """Branchless float->E8M0: add mantissa mask to round up, clamp to 254."""
    val_i32 = _bitcast_f32_to_i32(val, loc=loc, ip=ip)
    rounded = val_i32 + Int32(0x7FFFFF)
    exponent = (rounded >> Int32(FP32_MANTISSA_BITS)) & Int32(0xFF)
    return Int32(mlir_arith.minsi(
        exponent.ir_value(loc=loc, ip=ip),
        Int32(254).ir_value(loc=loc, ip=ip), loc=loc, ip=ip))


@dsl_user_op
def exp2f_rcp(biased_exp: Int32, *, loc=None, ip=None) -> Float32:
    """2^(127 - biased_exp) with special-case handling."""
    new_exp = (Int32(254) - biased_exp) << Int32(FP32_MANTISSA_BITS)
    result = _bitcast_i32_to_f32(new_exp, loc=loc, ip=ip)
    for (cmp_val, repl_bits) in [(255, 0x7FFFFFFF), (254, 0x00400000), (0, 0x7F000000)]:
        cond = mlir_arith.cmpi(mlir_arith.CmpIPredicate.eq,
                               biased_exp.ir_value(loc=loc, ip=ip),
                               Int32(cmp_val).ir_value(loc=loc, ip=ip), loc=loc, ip=ip)
        alt = _bitcast_i32_to_f32(Int32(repl_bits), loc=loc, ip=ip)
        result = Float32(mlir_arith.select(
            cond, alt.ir_value(loc=loc, ip=ip),
            result.ir_value(loc=loc, ip=ip), loc=loc, ip=ip))
    return result


@dsl_user_op
def cvt_f32_to_fp8e4m3(val: Float32, *, loc=None, ip=None) -> Int32:
    """float32 -> fp8e4m3fn via PTX cvt.rn.satfinite.e4m3x2.f32."""
    zero = Float32(0.0)
    result_i16 = Int16(llvm.inline_asm(
        T.i16(),
        [zero.ir_value(loc=loc, ip=ip), val.ir_value(loc=loc, ip=ip)],
        "cvt.rn.satfinite.e4m3x2.f32 $0, $1, $2;",
        "=h,f,f", has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT))
    result_i32 = Int32(mlir_arith.extui(
        T.i32(), result_i16.ir_value(loc=loc, ip=ip), loc=loc, ip=ip))
    return result_i32 & Int32(0xFF)


@dsl_user_op
def cvt_f32_to_fp8e5m2(val: Float32, *, loc=None, ip=None) -> Int32:
    """float32 -> fp8e5m2 via PTX cvt.rn.satfinite.e5m2x2.f32."""
    zero = Float32(0.0)
    result_i16 = Int16(llvm.inline_asm(
        T.i16(),
        [zero.ir_value(loc=loc, ip=ip), val.ir_value(loc=loc, ip=ip)],
        "cvt.rn.satfinite.e5m2x2.f32 $0, $1, $2;",
        "=h,f,f", has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT))
    result_i32 = Int32(mlir_arith.extui(
        T.i32(), result_i16.ir_value(loc=loc, ip=ip), loc=loc, ip=ip))
    return result_i32 & Int32(0xFF)


@dsl_user_op
def fma_f32(a: Float32, b: Float32, c: Float32, *, loc=None, ip=None) -> Float32:
    """`fma.rn.f32 d, a, b, c;` — single-instruction fused multiply-add
    matching nvcc's FFMA. Used for explicit `partial += a * b` patterns
    where we need the same rounding as TE's compiler-fused FFMA."""
    return Float32(llvm.inline_asm(
        T.f32(),
        [a.ir_value(loc=loc, ip=ip),
         b.ir_value(loc=loc, ip=ip),
         c.ir_value(loc=loc, ip=ip)],
        "fma.rn.f32 $0, $1, $2, $3;",
        "=f,f,f,f", has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT))


@dsl_user_op
def tanh_approx(val: Float32, *, loc=None, ip=None) -> Float32:
    """`tanh.approx.f32` — fast tanh approximation. Matches CUDA `__tanhf`."""
    return Float32(llvm.inline_asm(
        T.f32(),
        [val.ir_value(loc=loc, ip=ip)],
        "tanh.approx.f32 $0, $1;",
        "=f,f", has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT))


@dsl_user_op
def pack_f32x2(lo: Float32, hi: Float32, *, loc=None, ip=None) -> Int64:
    """Pack two f32 scalars into a single 64-bit register (`floatx2` layout).

    Low 32 bits = `lo`, high 32 bits = `hi`. Uses `mov.b64 %dst, {%lo, %hi};`
    which lowers to a single register move — no actual memory traffic.
    """
    return Int64(llvm.inline_asm(
        T.i64(),
        [lo.ir_value(loc=loc, ip=ip), hi.ir_value(loc=loc, ip=ip)],
        "mov.b64 $0, {$1, $2};",
        "=l,f,f", has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT))


@dsl_user_op
def pack_i32x2(lo: Int32, hi: Int32, *, loc=None, ip=None) -> Int64:
    """i32 sibling of `pack_f32x2` — concat two i32 into a single b64 register.
    Used by NVFP4 to glue two `(bf16,bf16)`/`(f16,f16)` Int32 packs into the
    `Int64` operand the `mul_cvt.*x4` PTX expects."""
    return Int64(llvm.inline_asm(
        T.i64(),
        [lo.ir_value(loc=loc, ip=ip), hi.ir_value(loc=loc, ip=ip)],
        "mov.b64 $0, {$1, $2};",
        "=l,r,r", has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT))


@dsl_user_op
def _trunc_i32_to_i16(val: Int32, *, loc=None, ip=None) -> Int16:
    """Narrow an Int32 to Int16 by keeping the low 16 bits.

    Lives here because the existing arith-dialect narrowing pattern requires
    loc/ip kwargs (see other `mlir_arith.trunci` callers); wrapping it as a
    `@dsl_user_op` lets `@cute.jit` bodies use it without plumbing those in."""
    return Int16(mlir_arith.trunci(
        T.i16(), val.ir_value(loc=loc, ip=ip), loc=loc, ip=ip))


@dsl_user_op
def cvt_fp8e4m3_to_f32(byte_i32: Int32, *, loc=None, ip=None) -> Float32:
    """One fp8e4m3 byte (low 8 bits of `byte_i32`) → f32.

    PTX has no direct `cvt.f32.e4m3` for a scalar; route through the packed
    `cvt.rn.f16x2.e4m3x2` and then `cvt.f32.f16`. The high byte of the .b16
    register is forced to zero so the discarded high f16 lane is well-defined."""
    asm = (
        "{\n"
        ".reg .b32 masked; .reg .b16 b16; .reg .b16 b16_hi;\n\t"
        ".reg .b32 f16pair; .reg .b16 lo_f16; .reg .b16 hi_f16;\n\t"
        "and.b32 masked, $1, 0xFF;\n\t"
        "mov.b32 {b16, b16_hi}, masked;\n\t"
        "cvt.rn.f16x2.e4m3x2 f16pair, b16;\n\t"
        "mov.b32 {lo_f16, hi_f16}, f16pair;\n\t"
        "cvt.f32.f16 $0, lo_f16;\n\t"
        "}"
    )
    return Float32(llvm.inline_asm(
        T.f32(),
        [byte_i32.ir_value(loc=loc, ip=ip)],
        asm,
        "=f,r", has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT))

# ---------------------------------------------------------------------------
# 16-bit packed input PTX kit (bf16 / f16)
#
# bf16 and f16 share the same fast-path shape: packed-x2 amax via
# `max.xorsign.abs.<fmt>x2`, then per-lane widen-to-f32 + `mul.f32x2` +
# `cvt.rn.satfinite.<out>x2.f32`. Only the opcodes differ. Build one PTX kit
# per format at module load and let the kernel pick the right kit at JIT
# trace time via `cfg.DTYPE` — equivalent to a C++ template arg specialization
# on `IType`, with no runtime branch.
# ---------------------------------------------------------------------------
def _build_packed16_kit(in_fmt: str):
    """Build a kit of PTX wrappers for a 16-bit input format.

    `in_fmt` is the PTX format string ('bf16' or 'f16'). Returns a namespace
    with the per-format ops the rowwise/colwise inner loops need:

      abs_max_x2(Int32, Int32)  -> Int32   # `max.xorsign.abs.<fmt>x2`
      abs_max_scalar(Int16, Int16) -> Int16  # `max.xorsign.abs.<fmt>`
      bits_to_f32(Int16) -> Float32          # widen one 16-bit element
      x2_lo_to_f32(Int32) -> Float32         # extract+widen low half
      x2_hi_to_f32(Int32) -> Float32         # extract+widen high half
      mul_cvt_to_fp8x2(fp8_dtype) -> callable(Int32, Int64)->Int32
                                            # fused <fmt>x2 * f32x2 -> fp8x2
    """

    @dsl_user_op
    def abs_max_x2(a: Int32, b: Int32, *, loc=None, ip=None) -> Int32:
        return Int32(llvm.inline_asm(
            T.i32(),
            [a.ir_value(loc=loc, ip=ip), b.ir_value(loc=loc, ip=ip)],
            f"max.xorsign.abs.{in_fmt}x2 $0, $1, $2;",
            "=r,r,r", has_side_effects=False, is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT))
    
    @dsl_user_op
    def max_x2(a: Int32, b: Int32, *, loc=None, ip=None) -> Int32:
        return Int32(llvm.inline_asm(
            T.i32(),
            [a.ir_value(loc=loc, ip=ip), b.ir_value(loc=loc, ip=ip)],
            f"max.{in_fmt}x2 $0, $1, $2;",
            "=r,r,r", has_side_effects=False, is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT))

    @dsl_user_op
    def abs_max_scalar(a: Int16, b: Int16, *, loc=None, ip=None) -> Int16:
        return Int16(llvm.inline_asm(
            T.i16(),
            [a.ir_value(loc=loc, ip=ip), b.ir_value(loc=loc, ip=ip)],
            f"max.xorsign.abs.{in_fmt} $0, $1, $2;",
            "=h,h,h", has_side_effects=False, is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT))

    if in_fmt == "bf16":
        # bf16 == top 16 bits of f32 — widening is a free bit-shift.
        @dsl_user_op
        def bits_to_f32(bits: Int16, *, loc=None, ip=None) -> Float32:
            i32 = Int32(mlir_arith.extui(
                T.i32(), bits.ir_value(loc=loc, ip=ip), loc=loc, ip=ip))
            return _bitcast_i32_to_f32(i32 << Int32(16), loc=loc, ip=ip)

        @dsl_user_op
        def x2_lo_to_f32(bits: Int32, *, loc=None, ip=None) -> Float32:
            return _bitcast_i32_to_f32(
                (bits & Int32(0xFFFF)) << Int32(16), loc=loc, ip=ip)

        @dsl_user_op
        def x2_hi_to_f32(bits: Int32, *, loc=None, ip=None) -> Float32:
            # `(x >> 16) << 16` ≡ `x & 0xFFFF0000`, sidestepping signed-literal
            # issues. Sign bits from the arith-right shift get zeroed by the
            # left shift.
            return _bitcast_i32_to_f32(
                (bits >> Int32(16)) << Int32(16), loc=loc, ip=ip)

        @dsl_user_op
        def truncate_f32(val: Float32, *, loc=None, ip=None) -> Float32:
            """Round f32 to bf16 precision (round-to-nearest-even), keep f32.
            Matches C++'s `static_cast<float>(static_cast<bf16>(elt))`."""
            bf16_bits = Int16(llvm.inline_asm(
                T.i16(), [val.ir_value(loc=loc, ip=ip)],
                "cvt.rn.bf16.f32 $0, $1;",
                "=h,f", has_side_effects=False, is_align_stack=False,
                asm_dialect=llvm.AsmDialect.AD_ATT))
            i32 = Int32(mlir_arith.extui(
                T.i32(), bf16_bits.ir_value(loc=loc, ip=ip), loc=loc, ip=ip))
            return _bitcast_i32_to_f32(i32 << Int32(16), loc=loc, ip=ip)
    else:
        # f16 has its own bit layout; widening requires `cvt.f32.f16`.
        @dsl_user_op
        def bits_to_f32(bits: Int16, *, loc=None, ip=None) -> Float32:
            return Float32(llvm.inline_asm(
                T.f32(), [bits.ir_value(loc=loc, ip=ip)],
                "cvt.f32.f16 $0, $1;",
                "=f,h", has_side_effects=False, is_align_stack=False,
                asm_dialect=llvm.AsmDialect.AD_ATT))

        @dsl_user_op
        def x2_lo_to_f32(bits: Int32, *, loc=None, ip=None) -> Float32:
            lo_i16 = Int16(mlir_arith.trunci(
                T.i16(), bits.ir_value(loc=loc, ip=ip), loc=loc, ip=ip))
            return bits_to_f32(lo_i16, loc=loc, ip=ip)

        @dsl_user_op
        def x2_hi_to_f32(bits: Int32, *, loc=None, ip=None) -> Float32:
            hi_shifted = bits >> Int32(16)
            hi_i16 = Int16(mlir_arith.trunci(
                T.i16(), hi_shifted.ir_value(loc=loc, ip=ip), loc=loc, ip=ip))
            return bits_to_f32(hi_i16, loc=loc, ip=ip)

        @dsl_user_op
        def truncate_f32(val: Float32, *, loc=None, ip=None) -> Float32:
            """Round f32 to f16 precision, keep f32."""
            f16_bits = Int16(llvm.inline_asm(
                T.i16(), [val.ir_value(loc=loc, ip=ip)],
                "cvt.rn.f16.f32 $0, $1;",
                "=h,f", has_side_effects=False, is_align_stack=False,
                asm_dialect=llvm.AsmDialect.AD_ATT))
            return Float32(llvm.inline_asm(
                T.f32(), [f16_bits.ir_value(loc=loc, ip=ip)],
                "cvt.f32.f16 $0, $1;",
                "=f,h", has_side_effects=False, is_align_stack=False,
                asm_dialect=llvm.AsmDialect.AD_ATT))

    def _build_mul_cvt(out_fmt: str, relu: bool = False):
        """Build a fused `<in_fmt>x2 * f32x2 → fp8<out_fmt>x2` PTX wrapper.

        The shape is identical across (in_fmt, out_fmt) combos — only the
        widening opcode (`cvt.f32.<in_fmt>`) and the final saturating cvt
        (`cvt.rn.satfinite.<out_fmt>x2.f32`) differ.
        """
        out_op = "e4m3x2" if out_fmt == "e4m3" else "e5m2x2"
        asm = (
            "{\n"
            ".reg.b64 vp0; .reg.b64 vp1;\n\t"
            ".reg.b32 v1;  .reg.b32 v2;\n\t"
            ".reg.b16 vb1; .reg.b16 vb2;\n\t"
            "mov.b32 {vb1, vb2}, $1;\n\t"
            f"cvt.f32.{in_fmt} v1, vb1;\n\t"
            f"cvt.f32.{in_fmt} v2, vb2;\n\t"
            "mov.b64 vp0, {v1, v2};\n\t"
            "mul.f32x2 vp1, vp0, $2;\n\t"
            "mov.b64 {v2, v1}, vp1;\n\t"
            f"cvt.rn.satfinite{".relu" if relu else ""}.{out_op}.f32 $0, v1, v2;\n\t"
            "}"
        )

        @dsl_user_op
        def fn(val_2x: Int32, scale_2x: Int64, *, loc=None, ip=None) -> Int32:
            result_i16 = Int16(llvm.inline_asm(
                T.i16(),
                [val_2x.ir_value(loc=loc, ip=ip),
                 scale_2x.ir_value(loc=loc, ip=ip)],
                asm,
                "=h,r,l", has_side_effects=False, is_align_stack=False,
                asm_dialect=llvm.AsmDialect.AD_ATT))
            return Int32(mlir_arith.extui(
                T.i32(), result_i16.ir_value(loc=loc, ip=ip), loc=loc, ip=ip))
        return fn

    def mul_cvt_to_fp8x2(fp8_dtype: str, relu: bool = False):
        if fp8_dtype == "e5m2":
            return _build_mul_cvt("e5m2", relu)
        return _build_mul_cvt("e4m3", relu)

    # NVFP4 fused cast: <in_fmt>x4 × f32x2 → fp4e2m1x4 (4 fp4 packed in 16
    # bits). Same shape as `mul_cvt_to_fp8x2` but produces 4 elements at a
    # time because the `cvt.rn.satfinite.e2m1x2.f32` PTX consumes pairs and
    # writes a single byte (high nibble = first input, low nibble = second).
    # The shuffled `mov.b64 {v1, v0}, v01` lines after the muls undo the
    # PTX's hi/lo packing so the resulting byte is naturally
    # `(fp4(elt1) << 4) | fp4(elt0)` — matches TE's C++ asm.
    @dsl_user_op
    def mul_cvt_to_fp4x4(in_4x: Int64, scale_2x: Int64, *, loc=None, ip=None) -> Int32:
        asm = (
            "{\n"
            ".reg.b64 v01; .reg.b64 v23;\n\t"
            ".reg.b16 i0; .reg.b16 i1; .reg.b16 i2; .reg.b16 i3;\n\t"
            ".reg.b32 v0; .reg.b32 v1; .reg.b32 v2; .reg.b32 v3;\n\t"
            ".reg.b8 f0; .reg.b8 f1;\n\t"
            "mov.b64 {i0, i1, i2, i3}, $1;\n\t"
            f"cvt.f32.{in_fmt} v0, i0;\n\t"
            f"cvt.f32.{in_fmt} v1, i1;\n\t"
            f"cvt.f32.{in_fmt} v2, i2;\n\t"
            f"cvt.f32.{in_fmt} v3, i3;\n\t"
            "mov.b64 v01, {v0, v1};\n\t"
            "mov.b64 v23, {v2, v3};\n\t"
            "mul.f32x2 v01, v01, $2;\n\t"
            "mul.f32x2 v23, v23, $2;\n\t"
            "mov.b64 {v1, v0}, v01;\n\t"
            "mov.b64 {v3, v2}, v23;\n\t"
            "cvt.rn.satfinite.e2m1x2.f32 f0, v0, v1;\n\t"
            "cvt.rn.satfinite.e2m1x2.f32 f1, v2, v3;\n\t"
            "mov.b32 $0, {f0, f1, f0, f1};\n\t"
            "}"
        )
        return Int32(llvm.inline_asm(
            T.i32(),
            [in_4x.ir_value(loc=loc, ip=ip), scale_2x.ir_value(loc=loc, ip=ip)],
            asm,
            "=r,l,l", has_side_effects=False, is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT))

    return SimpleNamespace(
        abs_max_x2=abs_max_x2,
        max_x2=max_x2,
        abs_max_scalar=abs_max_scalar,
        bits_to_f32=bits_to_f32,
        x2_lo_to_f32=x2_lo_to_f32,
        x2_hi_to_f32=x2_hi_to_f32,
        truncate_f32=truncate_f32,
        mul_cvt_to_fp8x2=mul_cvt_to_fp8x2,
        mul_cvt_to_fp4x4=mul_cvt_to_fp4x4,
    )


_BF16_KIT = _build_packed16_kit("bf16")
_F16_KIT = _build_packed16_kit("f16")


def _is_packed16(dtype) -> bool:
    """True if `dtype` is one of the 16-bit packed input formats."""
    return dtype is cutlass.BFloat16 or dtype is cutlass.Float16


def _packed16_kit(dtype):
    """Trace-time selector — pick a Packed16Kit for the input dtype."""
    if dtype is cutlass.Float16:
        return _F16_KIT
    return _BF16_KIT


# ---------------------------------------------------------------------------
# Forward-activation registry
#
# Each entry is a Float32 → Float32 callable applied per element before the
# MXFP8 amax + cast. Selection is by Python string at JIT trace time, so the
# const-expr machinery treats `cfg.ACTIVATION` like a C++ template argument
# — no runtime branch in the inner loop, separate kernel cached per choice.
#
# Math primitives match CUDA fast-math intrinsics so outputs are bit-exact
# with PyTorch's CUDA implementations of the same activations:
#   tanh   -> tanh.approx.f32 (== __tanhf)
#   exp(x) -> exp2.approx.f32(x · log2(e)) (== __expf)
# ---------------------------------------------------------------------------
def _act_relu(x: Float32) -> Float32:
    return cute.arch.fmax(x, Float32(0.0))


def _act_gelu(x: Float32) -> Float32:
    """Tanh-approximation GELU. Constants and operator grouping match TE's
    `transformer_engine/common/util/math.h::gelu` exactly (factored form
    `x · (0.5 + 0.5·tanh(x·(a + b·x²)))`) so quantized output is bit-exact
    against the C++ fused IS_ACT path. Uses `cute.math.tanh(fastmath=False)`
    rather than the `tanh.approx.f32` PTX intrinsic — TE compiles activation
    kernels without `--use_fast_math` by default, so its `tanhf` is the
    IEEE-precise expansion."""
    A = Float32(0.79788456)       # sqrt(2/π) truncated to TE's 8-digit literal
    B = Float32(0.03567741)       # = sqrt(2/π) · 0.044715, same truncation
    return x * (Float32(0.5) + Float32(0.5) * cute.math.tanh(x * (A + B * x * x)))


def _act_silu(x: Float32) -> Float32:
    """SiLU/Swish: x · σ(x) = x / (1 + e^-x).
    Matches TE's `silu` (`val / (1 + expf(-val))`)."""
    return x / (Float32(1.0) + cute.arch.exp(-x))


def _act_qgelu(x: Float32) -> Float32:
    """Quick GELU: x · σ(1.702·x). Matches TE `qgelu_with_alpha(val, 1.702)` =
    `cval · (1 / (1 + expf(-1.702·cval)))` (multiply by sigmoid, not a divide)."""
    z = Float32(1.702) * x
    return x * (Float32(1.0) / (Float32(1.0) + cute.arch.exp(-z)))


def _act_srelu(x: Float32) -> Float32:
    """Squared ReLU: x>0 ? x·x : 0 == (max(0,x))². Matches TE `srelu`."""
    r = cute.arch.fmax(x, Float32(0.0))
    return r * r


SUPPORTED_ACTIVATIONS = {
    "relu": _act_relu,
    "gelu": _act_gelu,
    "silu": _act_silu,
    "qgelu": _act_qgelu,
    "srelu": _act_srelu,
}


# ---------------------------------------------------------------------------
# Backward-activation (dact) registry
#
# Each entry is the derivative act'(x) as a Float32 → Float32 callable, matching
# the corresponding `d<act>` in transformer_engine/common/util/math.h. The dact
# kernel computes `grad · act'(x)` per element before the MXFP8 amax + cast.
# Primitives mirror the forward registry (cute.math.tanh fastmath=False for
# gelu, cute.arch.exp for the sigmoid) so output is bit-exact with the C++ path.
# ---------------------------------------------------------------------------
@dsl_user_op
def _dact_drelu(x: Float32, *, loc=None, ip=None) -> Float32:
    """drelu: x > 0 ? 1 : 0. Matches math.h `drelu` (NaN → 0 via ordered compare)."""
    cond = mlir_arith.cmpf(mlir_arith.CmpFPredicate.OGT,
                           x.ir_value(loc=loc, ip=ip),
                           Float32(0.0).ir_value(loc=loc, ip=ip), loc=loc, ip=ip)
    return Float32(mlir_arith.select(cond,
                                     Float32(1.0).ir_value(loc=loc, ip=ip),
                                     Float32(0.0).ir_value(loc=loc, ip=ip), loc=loc, ip=ip))


def _dact_dsrelu(x: Float32) -> Float32:
    """dsrelu: fmax(2x, 0). Matches math.h `dsrelu`."""
    return cute.arch.fmax(Float32(2.0) * x, Float32(0.0))


def _sigmoid(x: Float32) -> Float32:
    """σ(x) = 1 / (1 + e^-x), same exp intrinsic as the forward silu/qgelu."""
    return Float32(1.0) / (Float32(1.0) + cute.arch.exp(-x))


def _dact_dsilu(x: Float32) -> Float32:
    """dsilu: x·σ(x)·(1-σ(x)) + σ(x). Matches math.h `dsilu`
    (`cval·dsigmoid + sigmoid`, dsigmoid = s·(1-s))."""
    s = _sigmoid(x)
    return x * (s * (Float32(1.0) - s)) + s


def _dact_dqgelu(x: Float32) -> Float32:
    """dqgelu (alpha=1.702): a·x·dσ(a·x) + σ(a·x). Matches math.h
    `dqgelu_with_alpha(val, 1.702)`."""
    a = Float32(1.702)
    ax = a * x
    s = _sigmoid(ax)
    return a * x * (s * (Float32(1.0) - s)) + s


def _dact_dgelu(x: Float32) -> Float32:
    """dgelu (tanh approximation). Matches math.h `dgelu` term-for-term;
    same tanh argument as the forward `_act_gelu`."""
    t = cute.math.tanh(
        Float32(0.79788456) * x * (Float32(1.0) + Float32(0.044715) * x * x),
        fastmath=False,
    )
    return (Float32(0.5) * x
            * ((Float32(1.0) - t * t) * (Float32(0.79788456) + Float32(0.1070322243) * x * x))
            + Float32(0.5) * (Float32(1.0) + t))


SUPPORTED_DACTIVATIONS = {
    "drelu": _dact_drelu,
    "dgelu": _dact_dgelu,
    "dsilu": _dact_dsilu,
    "dqgelu": _dact_dqgelu,
    "dsrelu": _dact_dsrelu,
}


@dsl_user_op
def cvt_f32x2_to_fp8e4m3x2(val_hi: Float32, val_lo: Float32, relu: bool = False,
                             *, loc=None, ip=None) -> Int32:
    """Convert two float32 values to two packed fp8e4m3fn bytes in one instruction.

    Returns an int32 where bits [7:0] = fp8(val_lo), bits [15:8] = fp8(val_hi).
    This mirrors ptx::mul_cvt_2x which converts 2 values in one instruction.
    """
    result_i16 = Int16(llvm.inline_asm(
        T.i16(),
        [val_hi.ir_value(loc=loc, ip=ip), val_lo.ir_value(loc=loc, ip=ip)],
        f"cvt.rn.satfinite{".relu" if relu else ""}.e4m3x2.f32 $0, $1, $2;",
        "=h,f,f", has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT))
    return Int32(mlir_arith.extui(
        T.i32(), result_i16.ir_value(loc=loc, ip=ip), loc=loc, ip=ip))


@dsl_user_op
def cvt_f32x2_to_fp8e5m2x2(val_hi: Float32, val_lo: Float32, relu: bool = False,
                             *, loc=None, ip=None) -> Int32:
    """e5m2 sibling of `cvt_f32x2_to_fp8e4m3x2`."""
    result_i16 = Int16(llvm.inline_asm(
        T.i16(),
        [val_hi.ir_value(loc=loc, ip=ip), val_lo.ir_value(loc=loc, ip=ip)],
        f"cvt.rn.satfinite{".relu" if relu else ""}.e5m2x2.f32 $0, $1, $2;",
        "=h,f,f", has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT))
    return Int32(mlir_arith.extui(
        T.i32(), result_i16.ir_value(loc=loc, ip=ip), loc=loc, ip=ip))


@dsl_user_op
def mul_cvt_f32x4_to_fp4x4(in01: Int64, in23: Int64, scale_2x: Int64,
                            *, loc=None, ip=None) -> Int32:
    """f32x4 sibling of `kit.mul_cvt_to_fp4x4` — for the NVFP4 colwise path
    where elements live on a strided column and we've already widened to f32
    for the amax reduction. `in01` = pack(f32_0, f32_1), `in23` similarly."""
    asm = (
        "{\n"
        ".reg.b64 v01; .reg.b64 v23;\n\t"
        ".reg.b32 v0; .reg.b32 v1; .reg.b32 v2; .reg.b32 v3;\n\t"
        ".reg.b8 f0; .reg.b8 f1;\n\t"
        "mov.b64 {v0, v1}, $1;\n\t"
        "mov.b64 {v2, v3}, $2;\n\t"
        "mov.b64 v01, {v0, v1};\n\t"
        "mov.b64 v23, {v2, v3};\n\t"
        "mul.f32x2 v01, v01, $3;\n\t"
        "mul.f32x2 v23, v23, $3;\n\t"
        "mov.b64 {v1, v0}, v01;\n\t"
        "mov.b64 {v3, v2}, v23;\n\t"
        "cvt.rn.satfinite.e2m1x2.f32 f0, v0, v1;\n\t"
        "cvt.rn.satfinite.e2m1x2.f32 f1, v2, v3;\n\t"
        "mov.b32 $0, {f0, f1, f0, f1};\n\t"
        "}"
    )
    return Int32(llvm.inline_asm(
        T.i32(),
        [in01.ir_value(loc=loc, ip=ip),
         in23.ir_value(loc=loc, ip=ip),
         scale_2x.ir_value(loc=loc, ip=ip)],
        asm,
        "=r,l,l,l", has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT))


def _cvt_f32_to_fp8(fp8_dtype: str):
    """Const-expr dispatch: pick the f32→fp8 scalar PTX op based on output dtype.

    `fp8_dtype` is the Python string from `cfg.FP8_DTYPE`, evaluated at JIT
    trace time; the unused branch is never traced.
    """
    if fp8_dtype == "e5m2":
        return cvt_f32_to_fp8e5m2
    return cvt_f32_to_fp8e4m3


def _cvt_f32x2_to_fp8x2(fp8_dtype: str):
    """Const-expr dispatch for the packed f32x2→fp8x2 cvt."""
    if fp8_dtype == "e5m2":
        return cvt_f32x2_to_fp8e5m2x2
    return cvt_f32x2_to_fp8e4m3x2

@cute.jit
def quantize_rowwise_mxfp8(
    sX_tile,        # (TILE_Y, TILE_X) bf16/fp16 smem view, post-TMA
    sO_row_tile,    # (TILE_Y, TILE_X) uint8 smem view (rowwise FP8 output)
    mS_row_stage,         # rowwise scale tensor (1D swizzled, or 2D linear)
    max_norm_rcp,
    tile_row_start, # Int32 — global row index of this stage's row 0
                    # (= tile_idx_y * TILE_Y). Used to mask OOB scale stores
                    # for irregular shapes.
    tile_col_start, # Int32 — global col index of this CTA's col 0
                    # (= bidx * TILE_X). Same purpose.
    M, N,           # Int32 — full tensor extents; OOB threads skip their
                    # scale store.
    ACTIVATION,
    DTYPE,
    ROWWISE,
    COLWISE,
    FP8_DTYPE,
    TILE_Y,
    SCALE_DIM,
    WAVES,
    THREADS_PER_WARP,
    THREADS_PER_BANK,
    PACK_SIZE,
    WITH_ACT=False,     # forward: apply activation to the element
    WITH_DACT=False,    # backward: out = grad · act'(act_input)
    sA_tile=None,       # (TILE_Y, TILE_X) activation-input smem tile (dact only)
    DBIAS_REDUCTION=False,  # rowwise-only dbias: accumulate per-column partials
    dbias_acc=None,         # rmem Float32[SCALE_DIM]; += this row's pre-truncate elt per column
):
    tidx, _, _ = cute.arch.thread_idx()

    # Match the C++ reference's thread layout: pairs of adjacent lanes
    # share a row (lanes 2k / 2k+1 both own row k), each pair covering
    # the two 32-element scale blocks of that row. Express as a cute
    # layout mapping `(tid_Y, tid_X) -> tidx` with stride (2, 1):
    # linear(tidx) = tid_Y*2 + tid_X, so `get_flat_coord` inverts to
    # `(tidx // 2, tidx % 2)` — semantically clearer than the raw
    # divmod, and readily reusable if we later partition via TiledCopy.
    # print(f"sX_tile: {sX_tile}")
    # print(f"sO_row_tile: {sO_row_tile}")
    # print(f"mS_row_stage: {mS_row_stage}")
    
    tiler, tv_layout = cute.make_layout_tv(
        thr_layout=cute.make_layout((TILE_Y, 2), stride=(2, 1)),
        val_layout=cute.make_layout((1, SCALE_DIM), stride=(0, 1))
    )
    # print(f"tv_layout: {tv_layout}")
    # print(f"tiler: {tiler}")
    
    sX_tv = cute.composition(sX_tile, tv_layout)
    sO_tv = cute.composition(sO_row_tile, tv_layout)

    # I/O Elements that belong to this thread
    sX_thread = sX_tv[tidx, None]   # shape (32,) bf16
    sO_thread = sO_tv[tidx, None]   # shape (32,) uint8

    # See https://kainzhong.github.io/CuTe-Layout-Visualizer/?key=tv-2-%2832%2C+2%29%3A%282%2C1%29-%281%2C+32%29%3A%280%2C1%29
    # print(f"sX_thread: {sX_thread}")
    # print(f"sO_thread: {sO_thread}")

    sO_thread_u32_ptr = cute.recast_ptr(sO_thread.iterator, dtype=Uint32)
    # Each wave it writes 32 bytes = 8 uint32s, so in 4 waves we write all 32 quantized elements.
    sO_thread_u32 = cute.make_tensor(
        sO_thread_u32_ptr,
        cute.make_layout((SCALE_DIM // 4,), stride=(1,)), # 1 uint32 is 4 fp8 elements
    )
    # print(f"sO_thread_u32: {sO_thread_u32}")

    FUSE_RELU = cutlass.const_expr(ACTIVATION == "relu")
    # For this fast paht we can read in pack of 2 instead of reading individual f16 / bf16 element.
    # dbias needs the per-element fp32 values to accumulate, so it forces the slow path.
    _row_fast = (_is_packed16(DTYPE) and (ACTIVATION is None or FUSE_RELU)
                 and not DBIAS_REDUCTION)

    if cutlass.const_expr(_row_fast):
        # If no activation, f16 / bf16 and rowwise quantization, we can read 2 f16 / bf16 at once in a pack
        # and use max.xorsign.abs.f16x2 / max.xorsign.abs.bf16x2 to compute
        kit = _packed16_kit(DTYPE)
        sX_thread_rw_i32 = cute.make_tensor(
            cute.recast_ptr(sX_thread.iterator, dtype=Int32),
            cute.make_layout((1, SCALE_DIM // 2), stride=(0, 1)), # 1 int32 is 2 fp16/bf16 elements
        )
        # print(f"sX_thread_rw_i32: {sX_thread_rw_i32}")
        # Each wave we read 2 packed i32, which is 4 fp16/bf16 elements (PACK_SIZE)
        # In total we have 8 waves where each wave reads 
        in_r = [[None, None] for _ in range(WAVES)]
        bank_group = (tidx % THREADS_PER_WARP) // THREADS_PER_BANK # Each 4 threads share the same bank, which forms a bank group
        offset = bank_group * 2 # Each bank group will read 2 i32 from their bank
        for w in cutlass.range_constexpr(WAVES):
            idx = (w * 2 + offset) % (SCALE_DIM // 2)
            in_r[w][0] = sX_thread_rw_i32[0, idx]
            in_r[w][1] = sX_thread_rw_i32[0, idx + 1]

        # 1. Packed-x2 amax — 2 PTX per wave, 16 total per thread.
        # Accumulates `|elt|` in both lanes (with xorsign-drifted signs);
        # final horizontal max reduces the two lanes to a single f32.
        amax_2x = Int32(0)
        # Each wave will use max.xorsign.abs.f16x2 or max.xorsign.abs.bf16x2 to compare 2 packed elements in parallel
        for w in cutlass.range_constexpr(WAVES):
            if cutlass.const_expr(FUSE_RELU):
                # If we fuse relu then we don't want to do abs since negative value will be set to 0 and they will lose comparison automatically
                amax_2x = kit.max_x2(amax_2x, in_r[w][0])
                amax_2x = kit.max_x2(amax_2x, in_r[w][1])
            else:
                amax_2x = kit.abs_max_x2(amax_2x, in_r[w][0])
                amax_2x = kit.abs_max_x2(amax_2x, in_r[w][1])
        if cutlass.const_expr(FUSE_RELU):
            # Compare the 2 packed max without abs
            amax_r = cute.arch.fmax(
                kit.x2_lo_to_f32(amax_2x),
                kit.x2_hi_to_f32(amax_2x),
            )
            # For relu the max is at least 0
            if cutlass.const_expr(FUSE_RELU):
                amax_r = cute.arch.fmax(amax_r, Float32(0.0))
        else:
            # Compare the 2 packed abs max
            amax_r = cute.arch.fmax(
                fabs_f32(kit.x2_lo_to_f32(amax_2x)),
                fabs_f32(kit.x2_hi_to_f32(amax_2x)),
            )
    else:
        # Since we need to do computation on individual f16 / bf16 elements, we can't read in pack
        sX_thread_rw = cute.make_tensor(
            sX_thread.iterator,
            cute.make_layout((1, SCALE_DIM), stride=(0, 1)),
        )
        in_r = [[None] * PACK_SIZE for _ in range(WAVES)]
        bank_group = (tidx % THREADS_PER_WARP) // THREADS_PER_BANK # Each 4 threads share the same bank, which forms a bank group
        offset = bank_group * 4 # Each bank group will read 4 f16 from their bank

        if cutlass.const_expr(WITH_DACT):
            # Backward: out = grad · act'(act_input). sX is grad, sA is act_input.
            dop = SUPPORTED_DACTIVATIONS[ACTIVATION]
            sA_thread = cute.composition(sA_tile, tv_layout)[tidx, None]
            sA_thread_rw = cute.make_tensor(
                sA_thread.iterator,
                cute.make_layout((1, SCALE_DIM), stride=(0, 1)),
            )
        elif cutlass.const_expr(WITH_ACT):
            op = SUPPORTED_ACTIVATIONS[ACTIVATION]

        if cutlass.const_expr(_is_packed16(DTYPE) and ACTIVATION is not None):
            kit_act = _packed16_kit(DTYPE)
        amax_r = Float32(0.0)
        for w in cutlass.range_constexpr(WAVES):
            idx = (w * PACK_SIZE + offset) % SCALE_DIM
            for e in cutlass.range_constexpr(PACK_SIZE):
                x = Float32(sX_thread_rw[0, idx + e])  # grad
                if cutlass.const_expr(WITH_DACT):
                    # out = grad · act'(act_input)
                    x = x * dop(Float32(sA_thread_rw[0, idx + e]))
                # If IS_ACT, apply activation function to x in f32
                elif cutlass.const_expr(WITH_ACT):
                    # If it's relu, we can handle it later
                    if not cutlass.const_expr(FUSE_RELU):
                        x = op(x)
                # dbias: accumulate this row's column (idx+e) value BEFORE the bf16
                # truncation (matches CUDA's `thread_dbias_rowwise[j] += elt`). idx+e
                # is a multiple-of-PACK_SIZE group + e, so it stays within [0, SCALE_DIM).
                if cutlass.const_expr(DBIAS_REDUCTION):
                    dbias_acc[idx + e] = dbias_acc[idx + e] + x
                # If 16-bit input with activation, truncate to IType
                if cutlass.const_expr(_is_packed16(DTYPE) and ACTIVATION is not None):
                    x = kit_act.truncate_f32(x) # TODO: Why not just qunatize from f32?
                in_r[w][e] = x
                if cutlass.const_expr(FUSE_RELU):
                    amax_r = cute.arch.fmax(amax_r, x) # For relu cases, we don't need abs since negative values will be 0 so they lose comparison automatically
                else:
                    amax_r = cute.arch.fmax(amax_r, fabs_f32(x))
        if cutlass.const_expr(FUSE_RELU):
            amax_r = cute.arch.fmax(amax_r, Float32(0.0)) # If relu, the amax is at least 0

    # 2. E8M0 scale → gmem. mS_row's layout already encodes the swizzle
    # when cfg.WITH_GEMM_SWIZZLED_SCALES=True, so 2D access just works.
    biased_exp_r = float_to_e8m0(amax_r * max_norm_rcp)
    # mS_row_stage has logical shape (32, 2) and we have 64 threads where each is mapped to one scale factor
    # The TV layout is equivalent to https://kainzhong.github.io/CuTe-Layout-Visualizer/?key=tv-2-%2832%2C+2%29%3A%282%2C+1%29-%281%29
    # but it's too trival so let's just index it directly without using layout
    # Note this is the logical layout, which is on top of the swizzled / non-swizzled scale factor layout that mappes the logical index to the physical offset
    # Irregular shapes: skip the scale store if this thread's logical row /
    # col-block lies past the input's actual extents. TMA already zero-fills
    # OOB input reads and drops OOB output writes; only the direct scale-byte
    # gmem store needs an explicit guard.
    scale_row = tile_row_start + tidx // 2
    scale_col_first_elt = tile_col_start + (tidx % 2) * SCALE_DIM
    if scale_row < M and scale_col_first_elt < N:
        mS_row_stage[(tidx // 2, tidx % 2)] = Uint8(biased_exp_r)

    # 3. scale + packed fp8 cast → smem as one u32 per wave.
    inv_scale_r = exp2f_rcp(biased_exp_r) # f32 reciprocal of the scale
    # Fetch the conversion function based on the FP8 format
    cvt_f32x2 = _cvt_f32x2_to_fp8x2(FP8_DTYPE)
    if cutlass.const_expr(_row_fast):
        kit_cast = _packed16_kit(DTYPE)
        mul_cvt_x2 = kit_cast.mul_cvt_to_fp8x2(FP8_DTYPE, FUSE_RELU)
        # Pack `(inv_scale_r, inv_scale_r)` as a single 64-bit f32x2 once;
        # the per-wave mul_cvt consumes this directly.
        scale_2x = pack_f32x2(inv_scale_r, inv_scale_r)

    bank_group = (tidx % THREADS_PER_WARP) // THREADS_PER_BANK # Each 4 threads share the same bank, which forms a bank group
    offset = bank_group * 4 # Each bank group will write 4 fp8 to
    for w in cutlass.range_constexpr(WAVES):
        idx = (w * 4 + offset) % SCALE_DIM
        idx = idx // 4
        if cutlass.const_expr(_row_fast):
            # One fused PTX per <fmt>x2 pair: <fmt>x2 × f32x2 → fp8x2.
            # Byte layout: byte[0]=fp8(lo * s), byte[1]=fp8(hi * s).
            p01 = mul_cvt_x2(in_r[w][0], scale_2x)
            p23 = mul_cvt_x2(in_r[w][1], scale_2x)
        else:
            # cvt PTX semantics: `cvt.rn.satfinite.<fmt>.f32 d, a, b` gives
            # d[15:8]=fp8(a), d[7:0]=fp8(b). Pass (v1, v0) so the u16 low
            # byte ends up as fp8(v0) and the high byte as fp8(v1).
            v0 = in_r[w][0] * inv_scale_r
            v1 = in_r[w][1] * inv_scale_r
            v2 = in_r[w][2] * inv_scale_r
            v3 = in_r[w][3] * inv_scale_r
            p01 = cvt_f32x2(v1, v0, FUSE_RELU)  # u16 little-endian: v0,v1
            p23 = cvt_f32x2(v3, v2, FUSE_RELU)  # u16 little-endian: v2,v3
        quad = (p23 << Int32(16)) | p01
        sO_thread_u32[idx] = Uint32(quad)

    return amax_r

@cute.jit
def quantize_colwise_mxfp8(
    sX_tile,        # (TILE_Y, TILE_X) bf16/fp16 smem view, post-TMA
    sO_col_tile,    # (TILE_Y, TILE_X) uint8 smem view (colwise FP8 output)
    mS_col_stage,         # colwise scale tensor (1D swizzled, or 2D linear)
    max_norm_rcp,
    tile_row_start, # Int32 — global row index of this stage's row 0
                    # (= tile_idx_y * TILE_Y). Used to mask OOB scale stores
                    # for irregular shapes.
    tile_col_start, # Int32 — global col index of this CTA's col 0
                    # (= bidx * TILE_X).
    M, N,           # Int32 — full tensor extents.
    ACTIVATION,
    DTYPE,
    FP8_DTYPE,
    SWIZZLE,
    TILE_X,
    TILE_Y,
    SCALE_DIM,
    WITH_ACT=False,     # forward: apply activation to the element
    WITH_DACT=False,    # backward: out = grad · act'(act_input)
    sA_tile=None,       # (TILE_Y, TILE_X) activation-input smem tile (dact only)
    WITH_DBIAS=False,   # also return this thread's column sum (pre-truncate)
):
    tidx, _, _ = cute.arch.thread_idx()

    # print(f"sX_tile: {sX_tile}")
    # print(f"sO_col_tile: {sO_col_tile}")
    # print(f"mS_col_stage: {mS_col_stage}")

    tiler, tv_layout = cute.make_layout_tv(
        thr_layout=cute.make_layout((1, TILE_X), stride=(TILE_X, 1)),
        val_layout=cute.make_layout((SCALE_DIM, 1), stride=(1, 1))
    )
    # print(f"tv_layout: {tv_layout}")

    sX_tv = cute.composition(sX_tile, tv_layout)
    sO_tv = cute.composition(sO_col_tile, tv_layout)

    # I/O Elements that belong to this thread
    sX_thread = sX_tv[tidx, None]
    sO_thread = sO_tv[tidx, None]

    # See https://kainzhong.github.io/CuTe-Layout-Visualizer/?key=tv-2-%281%2C+64%29%3A%2864%2C+1%29-%2832%2C+1%29%3A%281%2C+1%29
    # print(f"sX_thread: {sX_thread}") # shape (32,) bf16
    # print(f"sO_thread: {sO_thread}") # shape (32,) uint8

    # dbias needs the per-element fp32 values to sum, so it takes the f32 path
    # (never the i16 fast path) — matching CUDA, whose f16 fast path requires
    # `!IS_DBIAS` (quantize_mxfp8.cuh:219).
    HALF_PRECISION_PATH = _is_packed16(DTYPE) and ACTIVATION is None and not WITH_DBIAS
    dbias_partial = Float32(0.0)

    # 0. Load the 32-element column from smem into registers once (matches
    # C++'s `in_colwise_IType[i]` cache). Amax and cast both reuse these.
    if cutlass.const_expr(HALF_PRECISION_PATH):
        kit = _packed16_kit(DTYPE)
        # Per-thread Int16 view of the column. Same byte address as
        # `sX_thread` (bf16/fp16 are 16-bit, same width as Int16); the
        # element stride is TILE_X because the column elements are
        # TILE_X apart in the row-major tile.
        sX_thread_i16 = cute.make_tensor(
            cute.recast_ptr(sX_thread.iterator, dtype=Int16),
            cute.make_layout((SCALE_DIM,), stride=(TILE_X,)),
        )
        amax_bits = Int16(0)
        for i in cutlass.range_constexpr(SCALE_DIM):
            amax_bits = kit.abs_max_scalar(amax_bits, sX_thread_i16[i])
        amax_c = fabs_f32(kit.bits_to_f32(amax_bits))
    else:
        # Materialize the column into f32 registers — widen on read so
        # bf16/fp16 inputs become real fp32 values (a pointer recast to
        # Float32 would not widen; it would reinterpret the 16-bit bytes
        # as half of a 32-bit float).
        sX_thread_f32 = cute.make_rmem_tensor(
            layout_or_shape=cute.make_layout((SCALE_DIM,), stride=(1,)),
            dtype=Float32,
        )
        for i in cutlass.range_constexpr(SCALE_DIM):
            sX_thread_f32[i] = Float32(sX_thread[i])
        # Apply activation (fwd) or grad·act'(act_input) (bwd dact) in f32.
        if cutlass.const_expr(WITH_DACT):
            dop = SUPPORTED_DACTIVATIONS[ACTIVATION]
            sA_thread = cute.composition(sA_tile, tv_layout)[tidx, None]
            for i in cutlass.range_constexpr(SCALE_DIM):
                sX_thread_f32[i] = sX_thread_f32[i] * dop(Float32(sA_thread[i]))
        elif cutlass.const_expr(WITH_ACT):
            op = SUPPORTED_ACTIVATIONS[ACTIVATION]
            for i in cutlass.range_constexpr(SCALE_DIM):
                sX_thread_f32[i] = op(sX_thread_f32[i])
        # dbias = column sum of the (post-act/dact) value, taken BEFORE the bf16
        # truncation — matches CUDA's `partial_dbias_colwise += elt`.
        if cutlass.const_expr(WITH_DBIAS):
            for i in cutlass.range_constexpr(SCALE_DIM):
                dbias_partial += sX_thread_f32[i]
        # Numerical truncation through IType so amax/cast match C++.
        # Only needed when 16-bit input + activation; without activation
        # the widening was already exact.
        if cutlass.const_expr(_is_packed16(DTYPE) and ACTIVATION is not None):
            kit_act = _packed16_kit(DTYPE)
            for i in cutlass.range_constexpr(SCALE_DIM):
                sX_thread_f32[i] = kit_act.truncate_f32(sX_thread_f32[i])
        amax_c = Float32(0.0)
        for i in cutlass.range_constexpr(SCALE_DIM):
            amax_c = cute.arch.fmax(amax_c, fabs_f32(sX_thread_f32[i]))

    # 2. E8M0 scale → gmem. mS_col's layout already encodes the swizzle
    # when cfg.WITH_GEMM_SWIZZLED_SCALES=True, so 2D access just works.
    # Irregular shapes: skip when this stage's row range or this thread's
    # column lies past the input extents. TILE_Y == SCALE_DIM so each stage
    # is exactly one scale-row; valid iff `tile_row_start < M`.
    biased_exp_c = float_to_e8m0(amax_c * max_norm_rcp)
    scale_col = tile_col_start + tidx
    if tile_row_start < M and scale_col < N:
        if cutlass.const_expr(SWIZZLE):
            mS_col_stage[(0, tidx % 32, tidx // 32)] = Uint8(biased_exp_c)
        else:
            mS_col_stage[(0, tidx)] = Uint8(biased_exp_c)

    # 3. scale + FP8 cast → smem (one byte per (row, tidx)). Caller
    # flushes the whole (TILE_Y, TILE_X) tile with a TMA S2G.
    inv_scale_c = exp2f_rcp(biased_exp_c)
    cvt_to_fp8 = _cvt_f32_to_fp8(FP8_DTYPE)
    if cutlass.const_expr(HALF_PRECISION_PATH):
        kit_cast = _packed16_kit(DTYPE)
        for i in cutlass.range_constexpr(SCALE_DIM):
            v_f32 = kit_cast.bits_to_f32(sX_thread_i16[i])
            sO_thread[i] = Uint8(cvt_to_fp8(v_f32 * inv_scale_c))
    else:
        for i in cutlass.range_constexpr(SCALE_DIM):
            sO_thread[i] = Uint8(cvt_to_fp8(sX_thread_f32[i] * inv_scale_c))

    return amax_c, dbias_partial
