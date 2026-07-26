# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Per-step and end-to-end tests for the MXFP4 weight-QAT recipes."""
import ctypes
import os

for _n in ("libnvrtc.so.13", "libnvrtc.so.12", "libnvrtc.so"):
    try:
        ctypes.CDLL(_n, mode=ctypes.RTLD_GLOBAL)
        break
    except OSError:
        continue

import torch

import transformer_engine.pytorch as te
import transformer_engine_torch as tex
from transformer_engine.common.recipe import (
    MXFP4QATFloat8BlockScaling,
    MXFP4QATMXFP8BlockScaling,
)
from transformer_engine.pytorch import fp8_autocast
from transformer_engine.pytorch.mxfp4_qat import mxfp4_fake_quantize
from references.mxfp4_qat_reference import mxfp4_fake_quantize_reference
from transformer_engine.pytorch.tensor.float8_blockwise_tensor import Float8BlockQuantizer
from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer

torch.manual_seed(1234)
DEV = "cuda"
RUN_PERF = os.environ.get("NVTE_MXFP4_QAT_PERF", "0") == "1"
E2M1_GRID = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=DEV)
SHAPES = [(128, 256), (192, 448), (512, 1024)]

FP8_TOL_1OP = 0.045
FP8_TOL_2OP = 0.065
BF16_TOL = 0.005


def make_weight(m, n, dtype=torch.bfloat16, zero_blocks=0, outliers=0):
    w = (torch.randn(m, n, device=DEV, dtype=torch.float32) * 0.02).to(dtype)
    if outliers:
        idx = torch.randint(0, m * n, (outliers,), device=DEV)
        w.view(-1)[idx] *= 500.0
    if zero_blocks:
        for i in range(zero_blocks):
            r = torch.randint(0, m, (1,)).item()
            k = torch.randint(0, n // 32, (1,)).item()
            w[r, k * 32 : (k + 1) * 32] = 0.0
    return w


def rel_err(got, ref):
    got, ref = got.to(torch.float64), ref.to(torch.float64)
    return ((got - ref).norm() / ref.norm().clamp(min=1e-30)).item()


def fake_quant_ref_fp64(w):
    """Independent fp64 reference: nearest-with-ties-to-even onto the E2M1 grid."""
    m, n = w.shape
    w64 = w.to(torch.float64).view(m, n // 32, 32)
    amax = w64.abs().amax(dim=-1, keepdim=True)
    bits = amax.to(torch.float32).view(torch.int32)
    exp_field = bits >> 23
    mant = bits & 0x7FFFFF
    exp = exp_field - 129 + (mant > 0x400000).to(torch.int32)
    exp = torch.where(exp_field > 0, exp, torch.full_like(exp, -126)).clamp(-126, 125)
    exp = torch.where(amax > 0, exp, torch.zeros_like(exp))
    scale = torch.ldexp(torch.ones_like(amax), exp)
    y = (w64 / scale).clamp(min=-6.0, max=6.0)
    grid = E2M1_GRID.to(torch.float64)
    cand = torch.cat([-grid.flip(0)[:-1], grid])
    d = (y.unsqueeze(-1) - cand).abs()
    near = d.argmin(dim=-1)
    even = torch.tensor([abs(v) in (0.0, 1.0, 2.0, 4.0) for v in cand.tolist()], device=w.device)
    dmin = d.gather(-1, near.unsqueeze(-1)).squeeze(-1)
    tie_even = d.eq(dmin.unsqueeze(-1)) & even
    has_even_tie = tie_even.any(dim=-1)
    near_even = torch.where(has_even_tie, tie_even.to(torch.uint8).argmax(dim=-1), near)
    q = cand[near_even]
    return (q * scale).view(m, n).to(torch.float64)


def test_fake_quant_grid_scale_rtne():
    for m, n in SHAPES:
        w = make_weight(m, n, zero_blocks=4, outliers=4)
        w_hat = mxfp4_fake_quantize(w)
        assert w_hat.dtype == w.dtype

        wh32 = w_hat.to(torch.float32).view(m, n // 32, 32)
        amax = w.to(torch.float32).abs().view(m, n // 32, 32).amax(dim=-1, keepdim=True)
        bits = amax.view(torch.int32)
        exp_field = bits >> 23
        mant = bits & 0x7FFFFF
        exp = exp_field - 129 + (mant > 0x400000).to(torch.int32)
        exp = torch.where(exp_field > 0, exp, torch.full_like(exp, -126)).clamp(-126, 125)
        scale = torch.ldexp(torch.ones_like(amax), exp)
        scale = torch.where(amax > 0, scale, torch.ones_like(scale))

        vals = (wh32 / scale).abs()
        assert torch.isin(vals, E2M1_GRID).all(), "values off the E2M1 grid"
        nz = (amax.double() > 6.0 * 2.0**-126) & (amax.double() <= 6.0 * 2.0**125)
        assert ((scale.double() * 6.0)[nz] >= amax.double()[nz]).all()
        assert (((scale.double() / 2.0) * 6.0)[nz] < amax.double()[nz]).all()
        ref = fake_quant_ref_fp64(w)
        assert torch.equal(w_hat.to(torch.float64), ref), f"RTNE mismatch {m}x{n}"
        assert torch.equal(mxfp4_fake_quantize(w_hat), w_hat), "not idempotent"


def test_D_lossless_in_bf16():
    for m, n in SHAPES:
        w = make_weight(m, n, outliers=4)
        w_hat_bf16 = mxfp4_fake_quantize(w)
        w_hat_fp32 = mxfp4_fake_quantize(w.to(torch.float32))
        assert torch.equal(
            w_hat_bf16.to(torch.float32), w_hat_fp32
        ), "MXFP4-grid weight is not exact in bf16"


def test_E_mxfp8_rowwise_lossless():
    for m, n in SHAPES:
        w = make_weight(m, n, zero_blocks=4, outliers=4)
        w[0, :32] = torch.tensor(2.0**-127, dtype=torch.bfloat16)
        w[0, 0] = 2.0**-128
        w_hat = mxfp4_fake_quantize(w)
        q = MXFP8Quantizer(fp8_dtype=tex.DType.kFloat8E4M3, columnwise=False).quantize(w_hat)
        dq = q.dequantize(dtype=torch.float32)
        wh32 = w_hat.to(torch.float32)

        data = q._rowwise_data.view(torch.uint8)[:m, :n]
        codes = q._rowwise_scale_inv.view(torch.uint8)[:m, : n // 32].to(torch.int32)
        raw_vals = data.view(torch.float8_e4m3fn).to(torch.float32).view(m, n // 32, 32)
        raw_scale = torch.ldexp(torch.ones_like(codes, dtype=torch.float32), codes - 127)
        raw_dq = (raw_vals * raw_scale.unsqueeze(-1)).view(m, n)
        assert torch.equal(
            raw_dq, wh32
        ), f"RAW MXFP8 encoding of the MXFP4-grid weight is not lossless {m}x{n}"

        assert torch.equal(
            dq, wh32
        ), f"MXFP8 rowwise encoding of the MXFP4-grid weight is not lossless {m}x{n}"


def test_dequantize_e8m0_extreme_codes():
    """Real MXFP8 software dequantize with planted UE8M0 codes 0/255."""
    w = make_weight(32, 64)
    q = MXFP8Quantizer(fp8_dtype=tex.DType.kFloat8E4M3, columnwise=False).quantize(
        mxfp4_fake_quantize(w)
    )
    data = q._rowwise_data.view(torch.uint8)
    codes = q._rowwise_scale_inv.view(torch.uint8)
    data[0, :32] = 56
    codes[0, 0] = 0
    data[1, :32] = 56
    codes[1, 0] = 255
    dq = q.dequantize(dtype=torch.float32)
    assert (dq[0, :32] == 2.0**-127).all(), "UE8M0 code 0 must dequantize to 2^-127"
    assert torch.isnan(dq[1, :32]).all(), "UE8M0 code 255 must dequantize to NaN"


def test_G_blockwise_lossless_and_transpose():
    for m, n in SHAPES:
        w = make_weight(m, n, zero_blocks=4)
        w_hat = mxfp4_fake_quantize(w)
        q = Float8BlockQuantizer(
            fp8_dtype=tex.DType.kFloat8E4M3,
            rowwise=True,
            columnwise=True,
            force_pow_2_scales=True,
            block_scaling_dim=2,
        ).quantize(w_hat)
        dq = q.dequantize(dtype=torch.float32)
        assert torch.equal(
            dq, w_hat.to(torch.float32)
        ), f"128x128 blockwise encoding of the MXFP4-grid weight is not lossless {m}x{n}"
        q.update_usage(rowwise_usage=False, columnwise_usage=True)
        dqc = q.dequantize(dtype=torch.float32)
        assert torch.equal(
            dqc, w_hat.to(torch.float32)
        ), f"columnwise 128x128 dequant differs from the MXFP4-grid weight {m}x{n}"


def test_lossless_dequant_to_bf16():
    """Both weight encodings dequantize to bf16 bit-exactly (bwd override target dtype)."""
    for m, n in SHAPES:
        w = make_weight(m, n, zero_blocks=4, outliers=4)
        w[0, :32] = torch.tensor(2.0**-127, dtype=torch.bfloat16)
        w_hat = mxfp4_fake_quantize(w)

        q8 = MXFP8Quantizer(fp8_dtype=tex.DType.kFloat8E4M3, columnwise=False).quantize(w_hat)
        dq8 = q8.dequantize(dtype=torch.bfloat16)
        assert torch.equal(dq8, w_hat), f"mxfp8 rowwise dequant to bf16 not lossless {m}x{n}"

        w2_hat = mxfp4_fake_quantize(make_weight(m, n, zero_blocks=4))
        qb = Float8BlockQuantizer(
            fp8_dtype=tex.DType.kFloat8E4M3,
            rowwise=True,
            columnwise=True,
            force_pow_2_scales=True,
            block_scaling_dim=2,
        ).quantize(w2_hat)
        assert torch.equal(
            qb.dequantize(dtype=torch.bfloat16), w2_hat
        ), f"blockwise 128x128 dequant to bf16 not lossless {m}x{n}"


def test_C_mxfp8_colwise_bound():
    for m, n in SHAPES:
        w_hat = mxfp4_fake_quantize(make_weight(m, n, zero_blocks=4, outliers=4))
        q = MXFP8Quantizer(fp8_dtype=tex.DType.kFloat8E4M3).quantize(w_hat)
        q.update_usage(rowwise_usage=False, columnwise_usage=True)
        dq = q.dequantize(dtype=torch.float32)
        ref = w_hat.to(torch.float32)
        blk = ref.view(m // 32, 32, n)
        bound = blk.abs().amax(dim=1, keepdim=True) * 2.0**-3
        err = (dq.view(m // 32, 32, n) - blk).abs()
        assert (err <= bound + 1e-30).all(), f"colwise 32x1 requant error above bound {m}x{n}"


_INTREE_KERNELS = {}


def _intree_kernel(fast_math=False):
    """JIT-compile the in-tree kernel; fast_math=True builds the same source with --use_fast_math."""
    if fast_math not in _INTREE_KERNELS:
        import os
        from torch.utils.cpp_extension import load_inline

        src_dir = os.environ.get(
            "NVTE_MXFP4_QAT_TEST_SRC",
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "..",
                "..",
                "transformer_engine",
                "common",
            ),
        )
        cuda_src = r"""
#include <cuda.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include "cast/mxfp4/fake_quantize_mxfp4.cuh"

torch::Tensor mxfp4_fake_quantize_intree(torch::Tensor w) {
  TORCH_CHECK(w.is_cuda() && w.is_contiguous() && w.numel() % 32 == 0);
  auto out = torch::empty_like(w);
  auto stream = at::cuda::getCurrentCUDAStream();
  if (w.scalar_type() == at::kBFloat16) {
    transformer_engine::dispatch::mxfp4::fake_quantize_mxfp4_launch<nv_bfloat16>(
        reinterpret_cast<const nv_bfloat16 *>(w.data_ptr()),
        reinterpret_cast<nv_bfloat16 *>(out.data_ptr()), w.numel(), stream);
  } else if (w.scalar_type() == at::kFloat) {
    transformer_engine::dispatch::mxfp4::fake_quantize_mxfp4_launch<float>(
        w.data_ptr<float>(), out.data_ptr<float>(), w.numel(), stream);
  } else {
    TORCH_CHECK(false, "bf16/fp32 only");
  }
  return out;
}
"""
        _INTREE_KERNELS[fast_math] = load_inline(
            name="nvte_mxfp4_qat_intree_test" + ("_fastmath" if fast_math else ""),
            cpp_sources="torch::Tensor mxfp4_fake_quantize_intree(torch::Tensor w);",
            cuda_sources=cuda_src,
            functions=["mxfp4_fake_quantize_intree"],
            extra_include_paths=[src_dir],
            extra_cuda_cflags=[
                "-O3",
                "-U__CUDA_NO_HALF_OPERATORS__",
                "-U__CUDA_NO_HALF_CONVERSIONS__",
                "-U__CUDA_NO_HALF2_OPERATORS__",
                "-U__CUDA_NO_BFLOAT16_OPERATORS__",
                "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
            ]
            + (["--use_fast_math"] if fast_math else []),
            verbose=False,
        )
    return _INTREE_KERNELS[fast_math]


def _both_paths(w):
    got_kernel = _intree_kernel().mxfp4_fake_quantize_intree(w.contiguous())
    got_torch = mxfp4_fake_quantize_reference(w)
    return got_kernel, got_torch


def _assert_bits_equal(a, b, msg):
    assert a.dtype == b.dtype, f"{msg}: dtype mismatch"
    ib = torch.int16 if a.dtype == torch.bfloat16 else torch.int32
    assert torch.equal(a.view(ib), b.view(ib)), f"{msg}: raw bits differ"


def _assert_same_with_nan(a, b, msg):
    an, bn = torch.isnan(a), torch.isnan(b)
    assert torch.equal(an, bn), f"{msg}: NaN masks differ"
    _assert_bits_equal(
        torch.where(an, torch.zeros_like(a), a),
        torch.where(bn, torch.zeros_like(b), b),
        msg,
    )


PARITY_SHAPES = SHAPES + [(3, 416), (1, 32)]


def test_kernel_matches_torch_reference():
    for m, n in PARITY_SHAPES:
        for dtype in (torch.bfloat16, torch.float32):
            w = make_weight(m, n, dtype=dtype, zero_blocks=4, outliers=4)
            gk, gt = _both_paths(w)
            _assert_bits_equal(gk, gt, f"kernel != torch reference {m}x{n} {dtype}")


def test_edge_value_domains():
    m, n = 64, 128
    w = make_weight(m, n)

    z = torch.zeros(m, n, device=DEV, dtype=torch.bfloat16)
    gk, gt = _both_paths(z)
    assert torch.equal(gk, z) and torch.equal(gt, z)

    wz = w.clone()
    wz[0, :32] = -0.0
    gk, gt = _both_paths(wz)
    assert (gk[0, :32] == 0).all() and torch.signbit(gk[0, :32]).all(), "-0.0 sign lost"
    _assert_bits_equal(gk, gt, "-0 block")

    ws = w.clone()
    ws[1, :32] = torch.tensor(2.0**-133, dtype=torch.bfloat16)
    gk, gt = _both_paths(ws)
    assert torch.isfinite(gk[1, :32]).all() and torch.equal(gk, gt)
    ref = fake_quant_ref_fp64(ws)
    assert torch.equal(gt.to(torch.float64), ref), "subnormal block deviates from fp64 ref"

    wp = w.clone()
    wp[5, :32] = torch.tensor(2.0**-127, dtype=torch.bfloat16)
    gk, gt = _both_paths(wp)
    assert (gk[5, :32].to(torch.float32) == 2.0**-127).all(), "2^-127 grid point lost"
    _assert_bits_equal(gk, gt, "2^-127 grid point")
    ref = fake_quant_ref_fp64(wp)
    assert torch.equal(gt.to(torch.float64), ref), "2^-127 block deviates from fp64 ref"

    wh = w.clone()
    wh[2, :32] = torch.tensor(3.0e38, dtype=torch.bfloat16)
    wh[2, 0] = -3.0e38
    gk, gt = _both_paths(wh)
    assert torch.isfinite(gk[2, :32]).all() and torch.equal(gk, gt)
    ref = fake_quant_ref_fp64(wh)
    assert torch.equal(gt.to(torch.float64), ref), "huge block deviates from fp64 ref"

    for bad in (float("inf"), float("-inf"), float("nan")):
        wb = w.clone()
        wb[3, 40] = bad
        gk, gt = _both_paths(wb)
        _assert_same_with_nan(gk, gt, f"bad={bad}")
        assert torch.isnan(gk[3, 32:64]).all(), f"{bad}: block not NaN-poisoned"
        assert torch.isfinite(gk[3, :32]).all(), f"{bad}: neighbor block affected"
        clean_rows = torch.ones(m, dtype=torch.bool)
        clean_rows[3] = False
        assert torch.isfinite(gk[clean_rows]).all()

    try:
        mxfp4_fake_quantize(torch.zeros(32, 32, device=DEV, dtype=torch.float16))
        raise SystemExit("fp16 should be rejected")
    except ValueError:
        pass


def test_blockwise_recipe_is_128x128():
    r = MXFP4QATFloat8BlockScaling()
    assert r.x_block_scaling_dim == 1 and r.w_block_scaling_dim == 2
    assert r.grad_block_scaling_dim == 1
    q = Float8BlockQuantizer(
        fp8_dtype=tex.DType.kFloat8E4M3,
        rowwise=True,
        columnwise=True,
        force_pow_2_scales=True,
        block_scaling_dim=2,
    )
    assert q.block_len == 128
    w_hat = mxfp4_fake_quantize(make_weight(256, 512))
    qt = q.quantize(w_hat)
    assert tuple(qt._rowwise_scale_inv.shape)[0] == 256 // 128


def test_blockwise_over_ratio_is_bounded_not_lossless():
    w = make_weight(128, 128)
    w[0, :32] = torch.tensor(2.0**-14, dtype=torch.bfloat16)
    w[64, 0] = 100.0
    w_hat = mxfp4_fake_quantize(w)
    q = Float8BlockQuantizer(
        fp8_dtype=tex.DType.kFloat8E4M3,
        rowwise=True,
        columnwise=True,
        force_pow_2_scales=True,
        block_scaling_dim=2,
    ).quantize(w_hat)
    dq = q.dequantize(dtype=torch.float32)
    assert torch.isfinite(dq).all()
    err = (dq - w_hat.to(torch.float32)).abs().max().item()
    tile_amax = w_hat.to(torch.float32).abs().max().item()
    assert err <= tile_amax * 2.0**-9, "over-ratio loss larger than FP8 headroom"


def test_kernel_perf():
    if not RUN_PERF:
        print("  SKIP (set NVTE_MXFP4_QAT_PERF=1 to run benchmarks)")
        return
    rows, cols = 8192, 8192
    w = make_weight(rows, cols)
    k = _intree_kernel()
    for _ in range(3):
        k.mxfp4_fake_quantize_intree(w)
        mxfp4_fake_quantize_reference(w)
    torch.cuda.synchronize()

    def timeit(fn, iters=20):
        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / iters * 1e3

    t_kernel = timeit(lambda: k.mxfp4_fake_quantize_intree(w))
    t_torch = timeit(lambda: mxfp4_fake_quantize_reference(w))
    moved = rows * cols * 2 * 2
    print(
        f"  kernel: {t_kernel:.1f} us ({moved / t_kernel / 1e3:.0f} GB/s), "
        f"torch: {t_torch:.1f} us ({moved / t_torch / 1e3:.0f} GB/s)"
    )
    assert t_kernel < t_torch, "kernel slower than the torch reference"


def test_kernel_fast_math_immune():
    """The kernel stays bit-identical when compiled with --use_fast_math."""
    k = _intree_kernel(fast_math=True)
    for m, n in PARITY_SHAPES:
        for dtype in (torch.bfloat16, torch.float32):
            w = make_weight(m, n, dtype=dtype, zero_blocks=4, outliers=4)
            got = k.mxfp4_fake_quantize_intree(w.contiguous())
            ref = mxfp4_fake_quantize_reference(w)
            _assert_bits_equal(got, ref, f"fast-math build diverged {m}x{n} {dtype}")

    w = make_weight(64, 128)
    w[0, :32] = 0.0
    w[1, 5] = float("inf")
    w[2, 9] = float("nan")
    w[3, :32] = torch.tensor(2.0**-133, dtype=torch.bfloat16)
    w[4, :32] = torch.tensor(3.0e38, dtype=torch.bfloat16)
    w[5, :32] = torch.tensor(2.0**-127, dtype=torch.bfloat16)
    got = k.mxfp4_fake_quantize_intree(w.contiguous())
    ref = mxfp4_fake_quantize_reference(w)
    _assert_same_with_nan(got, ref, "fast-math edge")
    assert (
        got[5, :32].to(torch.float32) == 2.0**-127
    ).all(), "fast-math build flushed the 2^-127 grid point"


def test_exp2f_e8m0_all_codes():
    """All 256 UE8M0 codes through exp2f/exp2f_rcp extracted verbatim from ptx.cuh."""
    import re as _re
    from torch.utils.cpp_extension import load_inline

    src_dir = os.environ.get(
        "NVTE_MXFP4_QAT_TEST_SRC",
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "..", "transformer_engine", "common"
        ),
    )
    ptx_src = open(os.path.join(src_dir, "util", "ptx.cuh")).read()
    m_exp = _re.search(
        r"__device__ __forceinline__ float exp2f\(e8m0_t biased_exp\) \{.*?\n\}", ptx_src, _re.S
    )
    m_rcp = _re.search(
        r"template <>\n__device__ __forceinline__ float exp2f_rcp<float>\(e8m0_t biased_exp\)"
        r" \{.*?\n\}",
        ptx_src,
        _re.S,
    )
    assert m_exp and m_rcp, "could not extract exp2f/exp2f_rcp from ptx.cuh"
    cuda_src = (
        "#include <cuda.h>\n#include <torch/extension.h>\n"
        "#include <ATen/cuda/CUDAContext.h>\n"
        "namespace teptx {\n"
        "typedef unsigned char e8m0_t;\n"
        "constexpr unsigned FP32_MANTISSA_BITS = 23;\n"
        "template <typename T> __device__ __forceinline__ T exp2f_rcp(e8m0_t);\n"
        + m_rcp.group(0)
        + "\n"
        + m_exp.group(0)
        + "\n}\n"
        "__global__ void all_codes_kernel(float *e, float *r) {\n"
        "  const int b = threadIdx.x;\n"
        "  e[b] = teptx::exp2f(static_cast<teptx::e8m0_t>(b));\n"
        "  r[b] = teptx::exp2f_rcp<float>(static_cast<teptx::e8m0_t>(b));\n"
        "}\n"
        "void run_all_codes(torch::Tensor e, torch::Tensor r) {\n"
        "  all_codes_kernel<<<1, 256, 0, at::cuda::getCurrentCUDAStream()>>>(\n"
        "      e.data_ptr<float>(), r.data_ptr<float>());\n"
        "}\n"
    )
    mod = load_inline(
        name="nvte_mxfp4_qat_exp2f_test",
        cpp_sources="void run_all_codes(torch::Tensor e, torch::Tensor r);",
        cuda_sources=cuda_src,
        functions=["run_all_codes"],
        verbose=False,
    )
    e = torch.empty(256, dtype=torch.float32, device=DEV)
    r = torch.empty_like(e)
    mod.run_all_codes(e, r)
    torch.cuda.synchronize()
    codes = torch.arange(256, dtype=torch.int32)
    exp_ref = torch.ldexp(torch.ones(256), codes - 127)
    rcp_ref = torch.ldexp(torch.ones(256), 127 - codes)
    assert torch.isnan(e[255]) and torch.isnan(r[255]), "code 255 must be NaN"
    assert e.view(torch.int32)[255].item() == 0x7FFFFFFF, "code 255 NaN payload"
    assert r.view(torch.int32)[255].item() == 0x7FFFFFFF, "code 255 rcp NaN payload"
    assert torch.equal(e[:255].cpu(), exp_ref[:255]), "exp2f wrong (code 0 == 2^-127?)"
    assert torch.equal(r[:255].cpu(), rcp_ref[:255]), "exp2f_rcp wrong"


def test_ste_identity_gradient():
    """mxfp4_fake_quantize carries an identity (STE) gradient and matches the reference forward."""
    for dtype in (torch.bfloat16, torch.float32):
        w = make_weight(64, 128, dtype=dtype).requires_grad_(True)
        up = torch.randn_like(w)
        out = mxfp4_fake_quantize(w)
        out.backward(up)
        assert torch.equal(w.grad, up), f"STE gradient is not identity ({dtype})"
        _assert_bits_equal(out.detach(), mxfp4_fake_quantize_reference(w.detach()), "STE fwd")


def test_misaligned_and_noncontiguous():
    """Misaligned storage-offset views and non-contiguous inputs match an aligned copy bitwise."""
    for dtype, off in ((torch.bfloat16, 4), (torch.float32, 2)):
        m, n = 64, 128
        flat = (torch.randn(m * n + off, device=DEV, dtype=torch.float32) * 0.02).to(dtype)
        v = flat[off:].view(m, n)
        assert v.is_contiguous() and v.data_ptr() % 16 != 0
        got = mxfp4_fake_quantize(v)
        ref = mxfp4_fake_quantize(v.clone())
        _assert_bits_equal(got, ref, f"misaligned view {dtype}")

    nc = (torch.randn(64, 256, device=DEV, dtype=torch.float32) * 0.02)[:, ::2]
    assert not nc.is_contiguous()
    got = mxfp4_fake_quantize_reference(nc)
    assert torch.equal(got, mxfp4_fake_quantize_reference(nc.contiguous()))


def test_bf16_exhaustive_bit_patterns():
    """All 65536 bf16 bit patterns: kernel/torch parity, fp64 oracle on finite blocks."""
    bits = torch.arange(65536, dtype=torch.int32, device=DEV).to(torch.int16)
    w = bits.view(torch.bfloat16).view(2048, 32)
    gk, gt = _both_paths(w)
    _assert_same_with_nan(gk, gt, "bf16 exhaustive")
    finite = torch.isfinite(w.to(torch.float32)).all(dim=-1)
    ref = fake_quant_ref_fp64(w[finite])
    assert torch.equal(gt[finite].to(torch.float64), ref), "bf16 exhaustive vs fp64 ref"


def test_fp32_bit_fuzz():
    """Random raw fp32 bit patterns (subnormals/NaN/Inf included)."""
    g = torch.Generator().manual_seed(99)
    for _ in range(4):
        bits = torch.randint(-(2**31), 2**31 - 1, (4096, 32), generator=g, dtype=torch.int64)
        w = bits.to(torch.int32).cuda().view(torch.float32)
        gk, gt = _both_paths(w)
        _assert_same_with_nan(gk, gt, "fp32 fuzz")
        finite = torch.isfinite(w).all(dim=-1)
        if finite.any():
            ref = fake_quant_ref_fp64(w[finite])
            assert torch.equal(gt[finite].to(torch.float64), ref), "fp32 fuzz vs fp64 ref"


def test_scale_threshold_and_rtne_midpoints():
    """amax = 1.5*2^k ulp triples and all E2M1 RTNE midpoints vs explicit expected values."""
    rows = []
    for k in (-100, -10, 0, 42, 100):
        a = torch.tensor(1.5 * 2.0**k, dtype=torch.float32)
        ab = a.view(torch.int32)
        for delta in (-1, 0, 1):
            row = torch.zeros(32, dtype=torch.float32)
            row[0] = (ab + delta).view(torch.float32)
            rows.append(row)
    w = torch.stack(rows).cuda()
    gk, gt = _both_paths(w)
    _assert_bits_equal(gk, gt, "threshold parity")
    assert torch.equal(gt.to(torch.float64), fake_quant_ref_fp64(w)), "threshold vs fp64 ref"

    mids = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])
    expected = torch.tensor([0.0, 1.0, 1.0, 2.0, 2.0, 4.0, 4.0])
    for t in (-60, 0, 60):
        sc = 2.0**t
        block = torch.zeros(32)
        block[:7] = mids * sc
        block[7] = 6.0 * sc
        w = block.view(1, 32).to(torch.float32).cuda()
        gk, gt = _both_paths(w)
        _assert_bits_equal(gk, gt, f"midpoints t={t}")
        exp_row = torch.zeros(32)
        exp_row[:7] = expected * sc
        exp_row[7] = 6.0 * sc
        assert torch.equal(
            gt[0].cpu().to(torch.float32), exp_row.to(torch.float32)
        ), f"midpoint values t={t}"


def test_deployment_top_cap_domain():
    """amax <= 6*2^125 is deployment-bitwise; above, satfinite at the cap while TileKernels moves to scale 2^126."""
    s125 = 6.0 * 2.0**125
    w = torch.zeros(2, 32, dtype=torch.float32)
    w[0, 0] = s125
    just_above = (torch.tensor(s125, dtype=torch.float32).view(torch.int32) + 1).view(torch.float32)
    w[1, 0] = just_above
    wc = w.cuda()
    gk, gt = _both_paths(wc)
    _assert_bits_equal(gk, gt, "top-cap parity")
    assert gk[0, 0].item() == s125, "boundary block must be exact"
    assert gk[1, 0].item() == s125, "above-boundary block must satfinite at 6*2^125"
    assert torch.equal(gt.to(torch.float64), fake_quant_ref_fp64(wc)), "top-cap vs fp64 ref"


def test_tilekernels_cross_parity():
    """Bitwise parity with the TileKernels torch reference (round_sf=True) on the finite below-cap domain."""
    import sys

    tk_root = os.environ.get(
        "NVTE_MXFP4_QAT_TILEKERNELS", os.path.expanduser("~/Desktop/v4/TileKernels")
    )
    if os.path.isdir(tk_root) and tk_root not in sys.path:
        sys.path.insert(0, tk_root)
    try:
        from tile_kernels.torch.cast import cast as tk_cast, cast_back as tk_cast_back
    except Exception as e:
        print(f"  SKIP (tile_kernels unavailable: {type(e).__name__}: {e})")
        return

    torch.manual_seed(21)
    for m, n in SHAPES:
        w = make_weight(m, n, zero_blocks=4, outliers=4)
        w[0, :32] = torch.tensor(2.0**-127, dtype=torch.bfloat16)
        w[1, :32] = torch.tensor(2.0**-133, dtype=torch.bfloat16)
        w[2, :32] = torch.tensor(2.0**120, dtype=torch.bfloat16)
        data, sf = tk_cast(w, "e2m1", block_size=(1, 32), round_sf=True)
        tk_dq = tk_cast_back((data, sf), "bf16", block_size=(1, 32))
        ours = mxfp4_fake_quantize(w)
        _assert_bits_equal(tk_dq, ours, f"TileKernels cross parity {m}x{n}")


def test_blockwise_feasibility_enumeration():
    """Per-element E4M3-lattice prediction == TE 128x128 quantizer for exponent spreads d=0..16."""
    payloads = torch.tensor([0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
    boundary = None
    for d in range(17):
        w = torch.zeros(128, 128, dtype=torch.float32)
        w[0, 0] = 6.0
        vals = payloads * 2.0 ** (-d)
        w[1, :7] = vals
        w[1, 7:14] = -vals
        wc = w.cuda()
        assert torch.equal(mxfp4_fake_quantize(wc), wc), f"tile not on MXFP4 grid (d={d})"
        q = Float8BlockQuantizer(
            fp8_dtype=tex.DType.kFloat8E4M3,
            rowwise=True,
            columnwise=False,
            force_pow_2_scales=True,
            block_scaling_dim=2,
        ).quantize(wc)
        S = q._rowwise_scale_inv.reshape(-1)[0].item()
        dq = q.dequantize(dtype=torch.float32)
        pred = ((wc / S).to(torch.float8_e4m3fn).to(torch.float32) * S) == wc
        got = dq == wc
        assert torch.equal(pred, got), f"E4M3-lattice prediction mismatch at spread d={d}"
        if boundary is None and not bool(pred.all()):
            boundary = d
    assert (
        boundary is not None and boundary >= 12
    ), f"full-payload exactness should hold at least through spread 2^11, got {boundary}"
    print(
        f"  full-grid exact through spread d={boundary - 1}; first loss at d={boundary} "
        "(E4M3 subnormal lattice bound)"
    )


def test_fp16_pipeline_rejected():
    """fp16 weights and fp16 activation dtypes must fail loudly under MXFP4 QAT."""
    mod = te.Linear(256, 128, bias=False, params_dtype=torch.float16, device=DEV)
    x = torch.randn(64, 256, device=DEV, dtype=torch.float16)
    for recipe in (MXFP4QATMXFP8BlockScaling(), MXFP4QATFloat8BlockScaling()):
        try:
            with fp8_autocast(enabled=True, fp8_recipe=recipe):
                mod(x)
            raise SystemExit(f"fp16 pipeline must be rejected ({recipe.__class__.__name__})")
        except (NotImplementedError, ValueError):
            pass


def test_cross_impl_matrix():
    """TileKernels vs CUDA kernel vs torch reference, bitwise, over all edge domains and every quant/dequant hop."""
    import sys

    tk_root = os.environ.get(
        "NVTE_MXFP4_QAT_TILEKERNELS", os.path.expanduser("~/Desktop/v4/TileKernels")
    )
    if os.path.isdir(tk_root) and tk_root not in sys.path:
        sys.path.insert(0, tk_root)
    try:
        from tile_kernels.torch.cast import (
            cast as tk_cast,
            cast_back as tk_cast_back,
            get_min_clamp_val as tk_min_clamp,
        )
    except Exception as e:
        print(f"  SKIP (tile_kernels unavailable: {type(e).__name__}: {e})")
        return
    kern = _intree_kernel().mxfp4_fake_quantize_intree
    tref = mxfp4_fake_quantize_reference
    CAP = 6.0 * 2.0**125

    def tk_roundtrip(x, block, fmt):
        pair = tk_cast(x, "e2m1" if fmt == "e2m1" else "e4m3", block_size=block, round_sf=True)
        return tk_cast_back(pair, "fp32", block), tk_cast_back(pair, "bf16", block)

    torch.manual_seed(77)
    w = make_weight(64, 256, zero_blocks=2, outliers=4)
    w[0, :32] = 0.0
    w[1, :32] = -0.0
    w[2, :32] = torch.tensor(2.0**-127, dtype=torch.bfloat16)
    w[2, 0] = 2.0**-128
    w[3, :32] = torch.tensor(2.0**-133, dtype=torch.bfloat16)
    w[4, :32] = torch.tensor(2.0**120, dtype=torch.bfloat16)
    w[5, :7] = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], dtype=torch.bfloat16)
    w[5, 7] = 6.0
    a15 = torch.tensor(1.5, dtype=torch.bfloat16).view(torch.int16)
    w[6, 0] = (a15 - 1).view(torch.bfloat16)
    w[6, 32] = torch.tensor(1.5, dtype=torch.bfloat16)
    w[6, 64] = (a15 + 1).view(torch.bfloat16)

    for W in (w, w.to(torch.float32)):
        t = tref(W)
        _assert_bits_equal(kern(W.contiguous()), t, "fake-quant CUDA vs torch")
        tk32, tkb16 = tk_roundtrip(W, (1, 32), "e2m1")
        _assert_bits_equal(tk32, t.to(torch.float32), "fake-quant TK vs torch (fp32)")
        _assert_bits_equal(tkb16, t.to(torch.bfloat16), "fake-quant TK vs torch (bf16)")
    print("  fake-quant 3-way bitwise: PASS (bf16 + fp32 corpus)")

    wnf = w.clone()
    wnf[8, 40] = float("inf")
    wnf[9, 33] = float("-inf")
    wnf[10, 5] = float("nan")
    t = tref(wnf)
    _assert_same_with_nan(kern(wnf.contiguous()), t, "nonfinite CUDA")
    assert torch.isnan(t[8, 32:64]).all() and torch.isnan(t[9, 32:64]).all()
    assert torch.isnan(t[10, :32]).all() and torch.isfinite(t[11].to(torch.float32)).all()
    print("  nonfinite 2-way (whole-block NaN poisoning): PASS (TK policy differs, excluded)")

    wcap = torch.zeros(1, 32, dtype=torch.bfloat16, device=DEV)
    wcap[0, 0] = torch.tensor(3.2e38, dtype=torch.bfloat16)
    t = tref(wcap)
    assert t[0, 0].item() == CAP, "satfinite cap at 6*2^125"
    _assert_bits_equal(kern(wcap.contiguous()), t, "cap CUDA")
    tk32, _ = tk_roundtrip(wcap, (1, 32), "e2m1")
    assert not torch.equal(tk32, t.to(torch.float32)), "TK must diverge above 6*2^125"
    print(
        f"  above-cap domain: ours satfinite {CAP:.3e}, TK diverges as documented "
        f"(payload at 2^126 -> {tk32[0, 0].item()}): PASS"
    )

    w_hat = tref(w)
    wh32 = w_hat.to(torch.float32)
    q = MXFP8Quantizer(fp8_dtype=tex.DType.kFloat8E4M3, columnwise=True).quantize(w_hat)
    m, n = w_hat.shape
    data = q._rowwise_data.view(torch.uint8)[:m, :n]
    codes = q._rowwise_scale_inv.view(torch.uint8)[:m, : n // 32].to(torch.int32)
    raw = data.view(torch.float8_e4m3fn).to(torch.float32).view(m, n // 32, 32) * torch.ldexp(
        torch.ones_like(codes, dtype=torch.float32), codes - 127
    ).unsqueeze(-1)
    assert torch.equal(raw.view(m, n), wh32), "mxfp8 row RAW encode not lossless"
    e4m3_floor = float(tk_min_clamp(torch.float8_e4m3fn))
    blk_amax = wh32.abs().view(m, n // 32, 32).amax(dim=-1)
    ok = (blk_amax >= e4m3_floor) | (blk_amax == 0)
    okm = ok.unsqueeze(-1).expand(m, n // 32, 32).reshape(m, n)
    tk32, tkb16 = tk_roundtrip(w_hat, (1, 32), "e4m3")
    assert torch.equal(
        tk32.view(torch.int32)[okm], wh32.view(torch.int32)[okm]
    ), "mxfp8 row TK roundtrip (fp32)"
    assert torch.equal(
        tkb16.view(torch.int16)[okm], w_hat.view(torch.int16)[okm]
    ), "mxfp8 row TK roundtrip (bf16)"
    assert (tk32[~okm] == 0).all(), "TK e4m3 sub-floor blocks must flush to zero"
    assert (~ok).sum() > 0, "TK e4m3 floor case not exercised"
    dq = q.dequantize(dtype=torch.float32)
    assert torch.equal(dq, wh32), "mxfp8 row TE software dequant"
    print(
        "  mxfp8 row: TE raw encode lossless incl 2^-127; TK roundtrip bitwise on "
        f"amax >= {e4m3_floor:g} blocks ({int((~ok).sum())} sub-floor blocks flush in TK "
        "e4m3, a TK-only floor); TE software dequant exact incl 2^-127: PASS"
    )

    q.update_usage(rowwise_usage=False, columnwise_usage=True)
    te_col = q.dequantize(dtype=torch.float32)
    tk_col32, _ = tk_roundtrip(w_hat, (32, 1), "e4m3")
    col_amax = wh32.abs().view(m // 32, 32, n).amax(dim=1)
    cok = (col_amax >= e4m3_floor) | (col_amax == 0)
    cokm = cok.unsqueeze(1).expand(m // 32, 32, n).reshape(m, n)
    nzm = (wh32 != 0) & cokm
    zm = (wh32 == 0) & cokm
    assert torch.equal(
        te_col.view(torch.int32)[nzm], tk_col32.view(torch.int32)[nzm]
    ), "mxfp8 col TE vs TK (nonzero)"
    assert (te_col[zm] == 0).all() and (tk_col32[zm] == 0).all(), "mxfp8 col zeros"
    negz = zm & torch.signbit(wh32)
    assert negz.any()
    assert torch.signbit(tk_col32)[negz].all(), "TK col must preserve -0"
    assert not torch.signbit(te_col)[negz].any(), "TE col canonicalizes -0 -> +0"
    blk = wh32.view(m // 32, 32, n)
    bound = blk.abs().amax(dim=1, keepdim=True) * 2.0**-3
    assert ((te_col.view(m // 32, 32, n) - blk).abs() <= bound + 1e-30).all()
    print(
        "  mxfp8 col (32x1): TE == TK bitwise on nonzeros, zeros value-equal "
        "(TE col canonicalizes -0 -> +0, TK preserves the sign), bounded vs w_hat: PASS"
    )

    wb_hat = tref(make_weight(128, 256, zero_blocks=4))
    wb32 = wb_hat.to(torch.float32)
    qb = Float8BlockQuantizer(
        fp8_dtype=tex.DType.kFloat8E4M3,
        rowwise=True,
        columnwise=True,
        force_pow_2_scales=True,
        block_scaling_dim=2,
    ).quantize(wb_hat)
    assert torch.equal(qb.dequantize(dtype=torch.float32), wb32)
    assert torch.equal(qb.dequantize(dtype=torch.bfloat16), wb_hat)
    tk32, tkb16 = tk_roundtrip(wb_hat, (128, 128), "e4m3")
    _assert_bits_equal(tk32, wb32, "blockwise TK roundtrip (fp32)")
    _assert_bits_equal(tkb16, wb_hat, "blockwise TK roundtrip (bf16)")
    qb.update_usage(rowwise_usage=False, columnwise_usage=True)
    assert torch.equal(qb.dequantize(dtype=torch.float32), wb32), "blockwise col transpose"
    print(
        "  blockwise 128x128: TE row+col dequant == TK roundtrip == w_hat bitwise "
        "(fp32 + bf16 targets): PASS"
    )

    bits = torch.arange(65536, dtype=torch.int32, device=DEV).to(torch.int16)
    we = bits.view(torch.bfloat16).view(2048, 32)
    t = tref(we)
    _assert_same_with_nan(kern(we.contiguous()), t, "exhaustive CUDA")
    we32 = we.to(torch.float32)
    ok_rows = torch.isfinite(we32).all(dim=-1) & (we32.abs().amax(dim=-1) <= CAP)
    tk32, _ = tk_roundtrip(we[ok_rows], (1, 32), "e2m1")
    _assert_bits_equal(tk32, t[ok_rows].to(torch.float32), "exhaustive TK")
    print(
        "  bf16 exhaustive 65536 patterns: CUDA/torch full, "
        f"TK on {int(ok_rows.sum())}/2048 finite below-cap rows, bitwise: PASS"
    )

    g = torch.Generator().manual_seed(123)
    fb = torch.randint(-(2**31), 2**31 - 1, (512, 32), generator=g, dtype=torch.int64)
    wf = fb.to(torch.int32).cuda().view(torch.float32)
    t = tref(wf)
    _assert_same_with_nan(kern(wf.contiguous()), t, "fuzz CUDA")
    ok_rows = torch.isfinite(wf).all(dim=-1) & (wf.abs().amax(dim=-1) <= CAP)
    if ok_rows.any():
        tk32, _ = tk_roundtrip(wf[ok_rows], (1, 32), "e2m1")
        _assert_bits_equal(tk32, t[ok_rows], "fuzz TK")
    print(
        f"  fp32 bit-fuzz 512 blocks: CUDA/torch full, TK on {int(ok_rows.sum())} "
        "finite below-cap rows, bitwise: PASS"
    )


def test_recipe_switch_invalidates_cache():
    """base<->QAT recipe switches must requantize instead of reusing cached weight workspaces."""
    import warnings as _warnings
    from transformer_engine.common.recipe import MXFP8BlockScaling

    torch.manual_seed(3)
    x = torch.randn(64, 256, device=DEV, dtype=torch.bfloat16)
    mod = te.Linear(256, 128, bias=False, params_dtype=torch.bfloat16, device=DEV)
    base, qat = MXFP8BlockScaling(), MXFP4QATMXFP8BlockScaling()

    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore")
        with fp8_autocast(enabled=True, fp8_recipe=base):
            mod(x, is_first_microbatch=True)
        with fp8_autocast(enabled=True, fp8_recipe=qat):
            y_switch = mod(x, is_first_microbatch=False)
            y_fresh = mod(x, is_first_microbatch=True)
        assert torch.equal(
            y_switch, y_fresh
        ), "base->QAT switch reused an UNPROJECTED cached weight"
        with fp8_autocast(enabled=True, fp8_recipe=base):
            y2_switch = mod(x, is_first_microbatch=False)
            y2_fresh = mod(x, is_first_microbatch=True)
        assert torch.equal(y2_switch, y2_fresh), "QAT->base switch reused a PROJECTED cached weight"


def test_recipe_field_controls_qat():
    """mxfp4_qat_weights field (env NVTE_MXFP4_QAT) turns base recipes into QAT recipes."""
    import warnings as _warnings
    from transformer_engine.common.recipe import Float8BlockScaling, MXFP8BlockScaling

    base = MXFP8BlockScaling(mxfp4_qat_weights=True)
    assert base.mxfp4_qat() and MXFP4QATMXFP8BlockScaling().mxfp4_qat()
    assert not MXFP8BlockScaling().mxfp4_qat()

    torch.manual_seed(9)
    x = torch.randn(64, 256, device=DEV, dtype=torch.bfloat16)
    mod = te.Linear(256, 128, bias=False, params_dtype=torch.bfloat16, device=DEV)
    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore")
        with fp8_autocast(enabled=True, fp8_recipe=base):
            y_field = mod(x, is_first_microbatch=True)
        with fp8_autocast(enabled=True, fp8_recipe=MXFP4QATMXFP8BlockScaling()):
            y_class = mod(x, is_first_microbatch=True)
    assert torch.equal(y_field, y_class), "field-driven QAT differs from class-driven QAT"

    assert Float8BlockScaling(mxfp4_qat_weights=True).mxfp4_qat()
    try:
        Float8BlockScaling(mxfp4_qat_weights=True, w_block_scaling_dim=1)
        raise SystemExit("field-driven blockwise QAT must validate 2D weight blocks")
    except ValueError:
        pass


def test_ops_api_rejected():
    """te.ops bypasses the QAT weight hook and must reject MXFP4 QAT recipes."""
    import transformer_engine.pytorch.ops as te_ops

    op = te_ops.Linear(256, 128, bias=False, device=DEV, dtype=torch.bfloat16)
    x = torch.randn(64, 256, device=DEV, dtype=torch.bfloat16)
    for recipe in (MXFP4QATMXFP8BlockScaling(), MXFP4QATFloat8BlockScaling()):
        try:
            with fp8_autocast(enabled=True, fp8_recipe=recipe):
                op(x)
            raise SystemExit(f"ops API must reject {recipe.__class__.__name__}")
        except NotImplementedError:
            pass


def _run_module(module_kind, recipe, override, fuse):
    torch.manual_seed(7)
    m_splits = [192, 320]
    tokens, in_f, out_f = sum(m_splits), 256, 512
    recipe.backward_override = override

    if module_kind == "linear":
        mod = te.Linear(
            in_f,
            out_f,
            bias=False,
            params_dtype=torch.bfloat16,
            device=DEV,
            fuse_wgrad_accumulation=fuse,
        )
        params = [mod.weight]
    else:
        mod = te.GroupedLinear(
            2,
            in_f,
            out_f,
            bias=False,
            params_dtype=torch.bfloat16,
            device=DEV,
            fuse_wgrad_accumulation=fuse,
        )
        params = [mod.weight0, mod.weight1]
    with torch.no_grad():
        for p in params:
            p.copy_((torch.randn_like(p, dtype=torch.float32) * 0.02).to(p.dtype))
    w_hats = [mxfp4_fake_quantize(p.detach()) for p in params]
    if fuse:
        for p in params:
            p.main_grad = torch.zeros_like(p, dtype=torch.float32)
            p.grad_added_to_main_grad = False

    x = (torch.randn(tokens, in_f, device=DEV, dtype=torch.float32) * 0.5).to(torch.bfloat16)
    x.requires_grad_(True)
    with fp8_autocast(enabled=True, fp8_recipe=recipe):
        if module_kind == "linear":
            out = mod(x)
        else:
            out = mod(x, m_splits)

    offs = [0, m_splits[0], tokens] if module_kind == "grouped" else [0, tokens]
    nsplit = len(params)

    def cat_ref(weights, mat, transpose):
        return torch.cat(
            [
                mat[offs[i] : offs[i + 1]].to(torch.float32)
                @ (
                    weights[i].detach().to(torch.float32).t()
                    if transpose
                    else weights[i].detach().to(torch.float32)
                )
                for i in range(nsplit)
            ]
        )

    ref_out_hat = cat_ref(w_hats, x, True)
    ref_out_master = cat_ref(params, x, True)
    e_hat, e_master = rel_err(out, ref_out_hat), rel_err(out, ref_out_master)
    assert e_hat < FP8_TOL_1OP, f"fwd err {e_hat}"
    assert e_hat < e_master, "fwd closer to master than to the MXFP4-grid weight"

    g = (torch.randn_like(out, dtype=torch.float32) * 0.1).to(out.dtype)
    out.backward(g)

    ref_dx_hat = cat_ref(w_hats, g, False)
    ref_dx_master = cat_ref(params, g, False)
    e_h, e_m = rel_err(x.grad, ref_dx_hat), rel_err(x.grad, ref_dx_master)
    if override == "high_precision":
        assert e_m < BF16_TOL, f"hp dgrad err vs master {e_m}"
        assert e_m < e_h, "hp dgrad must use the unquantized master (B)"
    elif override == "dequantized":
        assert e_h < BF16_TOL, f"dequantized dgrad err vs w_hat {e_h}"
        assert e_h < e_m, "dequantized dgrad must use the MXFP4-grid weight (A)"
    else:
        assert e_h < FP8_TOL_1OP, f"mode-None dgrad err {e_h}"
        assert e_h < e_m, "mode-None dgrad must use the MXFP4-grid weight (C)"

    if module_kind == "linear":
        with fp8_autocast(enabled=True, fp8_recipe=recipe):
            out_c1 = mod(x.detach(), is_first_microbatch=True)
            out_c2 = mod(x.detach(), is_first_microbatch=False)
        assert rel_err(out_c2, ref_out_hat) < FP8_TOL_1OP, "cached-weight fwd drifted"
        assert rel_err(out_c1, out_c2) < 1e-6, "workspace update changed the weights"

    wgrad_tol = BF16_TOL if override == "high_precision" else FP8_TOL_2OP
    for i, p in enumerate(params):
        ref_wg = g[offs[i] : offs[i + 1]].to(torch.float32).t() @ x[offs[i] : offs[i + 1]].to(
            torch.float32
        )
        if fuse:
            assert p.grad_added_to_main_grad is True, "fused-wgrad flag not set (G)"
            e = rel_err(p.main_grad, ref_wg)
            assert e < wgrad_tol, f"main_grad err {e} (G)"
        else:
            assert p.grad is not None, "STE did not deliver a weight grad (G)"
            e = rel_err(p.grad, ref_wg)
            assert e < wgrad_tol, f"wgrad err {e} (G)"


def test_e2e_matrix():
    for module_kind in ("linear", "grouped"):
        for recipe_cls in (MXFP4QATMXFP8BlockScaling, MXFP4QATFloat8BlockScaling):
            for override in (None, "dequantized", "high_precision"):
                for fuse in (False, True):
                    _run_module(module_kind, recipe_cls(), override, fuse)
                    print(
                        f"  e2e OK {module_kind} {recipe_cls.__name__} "
                        f"override={override} fuse={fuse}"
                    )


TESTS = [
    test_fake_quant_grid_scale_rtne,
    test_D_lossless_in_bf16,
    test_E_mxfp8_rowwise_lossless,
    test_dequantize_e8m0_extreme_codes,
    test_G_blockwise_lossless_and_transpose,
    test_C_mxfp8_colwise_bound,
    test_lossless_dequant_to_bf16,
    test_kernel_matches_torch_reference,
    test_edge_value_domains,
    test_blockwise_recipe_is_128x128,
    test_blockwise_over_ratio_is_bounded_not_lossless,
    test_exp2f_e8m0_all_codes,
    test_ste_identity_gradient,
    test_misaligned_and_noncontiguous,
    test_bf16_exhaustive_bit_patterns,
    test_fp32_bit_fuzz,
    test_scale_threshold_and_rtne_midpoints,
    test_deployment_top_cap_domain,
    test_tilekernels_cross_parity,
    test_blockwise_feasibility_enumeration,
    test_recipe_switch_invalidates_cache,
    test_recipe_field_controls_qat,
    test_ops_api_rejected,
    test_fp16_pipeline_rejected,
    test_cross_impl_matrix,
    test_kernel_perf,
    test_kernel_fast_math_immune,
    test_e2e_matrix,
]

if __name__ == "__main__":
    import sys
    import traceback

    failed = 0
    for t in TESTS:
        try:
            t()
            torch.cuda.synchronize()
            print(f"PASS {t.__name__}")
        except Exception:
            failed += 1
            print(f"FAIL {t.__name__}")
            traceback.print_exc()
    print(f"{len(TESTS) - failed}/{len(TESTS)} passed")
    sys.exit(1 if failed else 0)
