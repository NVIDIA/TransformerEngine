# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Recompile-robustness tests for the CuTeDSL MXFP8 quantize backend.

The CuTeDSL backend JIT-compiles one kernel per distinct config (input dtype ×
fp8 format × direction × activation × dbias × swizzle), registers it in the
TVM-FFI global registry under a config key, and fetches it per call. These tests
stress that compile/cache machinery rather than numerics:

  * many distinct configs each compile and produce finite, correct output;
  * interleaving configs never clobbers a cached kernel (the right kernel is
    served for each key, regardless of what else was compiled);
  * a single symbolic-shape kernel handles many (M, N) shapes from one compile;
  * repeated calls are bit-for-bit deterministic.

This is backend-specific, so it only runs when the CuTeDSL MXFP8 backend is
actually active in the process. Run it with::

    NVTE_ENABLE_CUTEDSL_QUANT_BACKEND=1 CUTE_DSL_ARCH=sm_100a \\
        python -m pytest tests/pytorch/mxfp8/test_mxfp8_cutedsl_recompile.py

otherwise every test skips (the env var is read once, at the first quantize).
"""
# TODO: review this file

import pytest
import torch
import torch.nn.functional as F

import transformer_engine.pytorch as te
import transformer_engine_torch as tex
from transformer_engine.pytorch import MXFP8Quantizer

recipe_available, reason_for_no_recipe = te.is_mxfp8_available(return_reason=True)

_FP8 = {"e4m3": tex.DType.kFloat8E4M3, "e5m2": tex.DType.kFloat8E5M2}
_DT = {"bf16": torch.bfloat16, "fp16": torch.float16}
_FWD = {"plain", "gelu", "silu", "relu", "qgelu", "srelu"}
_FWD_FN = {
    "gelu": tex.gelu,
    "silu": tex.silu,
    "relu": tex.relu,
    "qgelu": tex.qgelu,
    "srelu": tex.srelu,
}
_DACT_FN = {
    "dgelu": tex.dgelu,
    "dsilu": tex.dsilu,
    "drelu": tex.drelu,
    "dqgelu": tex.dqgelu,
    "dsrelu": tex.dsrelu,
}
_DBIAS_DACT_FN = {
    f"dbias_{k}": getattr(tex, f"dbias_{k}")
    for k in ("dgelu", "dsilu", "drelu", "dqgelu", "dsrelu")
}

# A diverse set of configs to interleave/repeat: mixed dtypes, fp8 formats,
# directions, and the plain / forward-act / dact / dbias / dbias+dact families.
_CONFIGS = [
    # (combo, rowwise, columnwise, in_dtype, fp8)
    ("plain", True, True, "bf16", "e4m3"),
    ("plain", True, False, "bf16", "e4m3"),
    ("plain", False, True, "bf16", "e4m3"),
    ("plain", True, True, "bf16", "e5m2"),
    ("plain", True, True, "fp16", "e4m3"),
    ("gelu", True, True, "bf16", "e4m3"),
    ("relu", True, True, "bf16", "e4m3"),
    ("silu", True, True, "bf16", "e4m3"),
    ("dgelu", True, True, "bf16", "e4m3"),
    ("dbias", True, True, "bf16", "e4m3"),
    ("dbias_dsilu", True, True, "bf16", "e4m3"),
    ("dbias_dqgelu", True, False, "bf16", "e4m3"),
]


def _inputs(M, N, in_dtype, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    dt = _DT[in_dtype]
    x = torch.empty(M, N, dtype=dt, device="cuda").uniform_(-4.0, 4.0, generator=g)
    ain = torch.empty(M, N, dtype=dt, device="cuda").uniform_(-3.0, 3.0, generator=g)
    return x, ain


def _run(combo, x, ain, rowwise, columnwise, fp8, swizzle=False):
    """Quantize via the public dispatch; returns (mxfp8_tensor, dbias_or_None).

    swizzle=True requests cuBLAS-swizzled scale layout (optimize_for_gemm)."""
    q = MXFP8Quantizer(fp8_dtype=_FP8[fp8], rowwise=rowwise, columnwise=columnwise)
    if swizzle:
        q.optimize_for_gemm = True
    if combo == "plain":
        return q(x), None
    if combo in _FWD_FN:
        return _FWD_FN[combo](x, q), None
    if combo in _DACT_FN:
        return _DACT_FN[combo](x, ain, q), None
    if combo == "dbias":
        db, out = tex.bgrad_quantize(x, q)
        return out, db
    if combo in _DBIAS_DACT_FN:
        db, out = _DBIAS_DACT_FN[combo](x, ain, q)
        return out, db
    raise ValueError(f"unknown combo {combo!r}")


def _signature(out, db, rowwise, columnwise):
    """Bit-level fingerprint of a quantized result, for golden comparison.

    The scale tensors are allocated at a 128-padded shape; only the meaningful
    region is written by the kernel, so we slice to it (M, ceil(N/32)) rowwise /
    (ceil(M/32), N) columnwise). Comparing the padding would spuriously fail —
    it's uninitialized and reflects whatever was in the (dirty) allocator pool.
    M, N are read from the data tensor, which is exactly (M, N), unpadded."""
    parts = []
    if rowwise:
        data = out._rowwise_data.view(torch.uint8)
        M, N = data.shape
        parts += [data.clone(), out._rowwise_scale_inv[:M, : (N + 31) // 32].clone()]
    if columnwise:
        data = out._columnwise_data.view(torch.uint8)
        M, N = data.shape
        parts += [data.clone(), out._columnwise_scale_inv[: (M + 31) // 32, :N].clone()]
    if db is not None:
        parts.append(db.clone())
    return parts


def _sig_equal(a, b):
    return len(a) == len(b) and all(torch.equal(p, q) for p, q in zip(a, b))


def _ref_fwd(combo, xf):
    if combo == "plain":
        return xf
    if combo == "gelu":
        return F.gelu(xf, approximate="tanh")
    if combo == "silu":
        return F.silu(xf)
    if combo == "relu":
        return F.relu(xf)
    if combo == "qgelu":
        return xf * torch.sigmoid(1.702 * xf)
    if combo == "srelu":
        return F.relu(xf) ** 2
    raise ValueError(combo)


@pytest.fixture(scope="module", autouse=True)
def _require_active_cutedsl_backend():
    """Skip unless the CuTeDSL backend is actually active (it registers its kernel
    under a config key in the TVM-FFI registry on first use)."""
    if not recipe_available:
        pytest.skip(reason_for_no_recipe)
    # Trigger one quantize, then confirm the CuTeDSL kernel registered itself.
    x = torch.randn(64, 64, dtype=torch.bfloat16, device="cuda")
    MXFP8Quantizer(fp8_dtype=tex.DType.kFloat8E4M3, rowwise=True, columnwise=True)(x)
    active = False
    try:
        import tvm_ffi

        active = (
            tvm_ffi.get_global_func(
                "cutedsl_mxfp8_bf16_e4m3_1_1_0_0_0_0_0_0_none", allow_missing=True
            )
            is not None
        )
    except Exception:
        active = False
    if not active:
        pytest.skip(
            "CuTeDSL MXFP8 backend not active in this process; run with "
            "NVTE_ENABLE_CUTEDSL_QUANT_BACKEND=1 set before the first quantize."
        )


@pytest.mark.skipif(not recipe_available, reason=reason_for_no_recipe)
def test_interleaved_configs_do_not_clobber_each_other():
    """Compile + capture a golden output for every config, then re-run them all in
    reverse order. Each must reproduce its golden bit-for-bit — compiling/running
    other configs must never corrupt a cached kernel or serve the wrong one."""
    M, N = 256, 512
    golden = {}
    for combo, rw, cw, dt, fp8 in _CONFIGS:
        x, ain = _inputs(M, N, dt)
        out, db = _run(combo, x, ain, rw, cw, fp8)
        golden[(combo, rw, cw, dt, fp8)] = _signature(out, db, rw, cw)

    for cfg in reversed(_CONFIGS):
        combo, rw, cw, dt, fp8 = cfg
        x, ain = _inputs(M, N, dt)
        out, db = _run(combo, x, ain, rw, cw, fp8)
        assert _sig_equal(_signature(out, db, rw, cw), golden[cfg]), (
            f"config {cfg} produced different output after other configs were "
            "(re)compiled — cached kernel was clobbered or mis-keyed"
        )


@pytest.mark.skipif(not recipe_available, reason=reason_for_no_recipe)
def test_cached_kernel_stable_while_new_configs_compile():
    """A fixed probe config run once for golden, then re-run after compiling each
    other config. The probe output must never change — a newly compiled kernel
    must not evict or overwrite the probe's cached kernel."""
    M, N = 320, 640
    p_combo, p_rw, p_cw, p_dt, p_fp8 = ("gelu", True, True, "bf16", "e4m3")
    px, pain = _inputs(M, N, p_dt)
    out, db = _run(p_combo, px, pain, p_rw, p_cw, p_fp8)
    golden = _signature(out, db, p_rw, p_cw)

    for combo, rw, cw, dt, fp8 in _CONFIGS:
        x, ain = _inputs(M, N, dt)
        _run(combo, x, ain, rw, cw, fp8)  # (re)compile / run another config
        out, db = _run(p_combo, px, pain, p_rw, p_cw, p_fp8)
        assert _sig_equal(
            _signature(out, db, p_rw, p_cw), golden
        ), f"probe ({p_combo}) output changed after running config {combo!r}"


@pytest.mark.skipif(not recipe_available, reason=reason_for_no_recipe)
@pytest.mark.parametrize("combo", ["plain", "gelu", "dbias_dsilu"])
def test_one_symbolic_kernel_handles_many_shapes(combo):
    """The kernel is compiled once with symbolic (M, N) (divisible by 32). Feeding
    many shapes through that single compile must all give finite output (and, for
    forward combos, output close to the reference)."""
    rw = cw = True
    fp8, dt = "e4m3", "bf16"
    shapes = [(32, 32), (64, 64), (32, 2048), (2048, 32), (256, 512), (1024, 1536), (2048, 2048)]
    for M, N in shapes:
        x, ain = _inputs(M, N, dt)
        out, _ = _run(combo, x, ain, rw, cw, fp8)
        deq = out.dequantize(dtype=torch.float32)
        assert torch.isfinite(deq).all(), f"{combo} {M}x{N}: non-finite output"
        if combo in _FWD:
            ref = _ref_fwd(combo, x.float())
            rel = (deq - ref).norm() / ref.norm().clamp_min(1e-6)
            assert rel < 0.12, f"{combo} {M}x{N}: rel_err={rel:.4f}"


@pytest.mark.skipif(not recipe_available, reason=reason_for_no_recipe)
def test_repeated_calls_are_deterministic():
    """The same config + same input, called repeatedly, must be bit-for-bit
    identical (the cached kernel is stable across reuse)."""
    M, N = 256, 512
    for combo, rw, cw, dt, fp8 in _CONFIGS:
        x, ain = _inputs(M, N, dt)
        sigs = [_signature(*_run(combo, x, ain, rw, cw, fp8), rw, cw) for _ in range(4)]
        for i in range(1, len(sigs)):
            assert _sig_equal(
                sigs[i], sigs[0]
            ), f"config ({combo},{dt},{fp8}) call {i} differs from call 0"


@pytest.mark.skipif(not recipe_available, reason=reason_for_no_recipe)
@pytest.mark.parametrize("direction", ["both", "row"])
@pytest.mark.parametrize("fp8", ["e4m3", "e5m2"])
@pytest.mark.parametrize("combo", ["plain", "gelu", "relu"])
def test_distinct_configs_compile_and_are_correct(combo, fp8, direction):
    """Each distinct (combo, fp8, direction) is its own compile. Verify it produces
    finite output close to the reference — i.e. a freshly compiled kernel is not
    garbage (catches e.g. a write-index regression in a recompiled kernel)."""
    rw = direction in ("both", "row")
    cw = direction in ("both", "col")
    M, N = 256, 512
    x, _ = _inputs(M, N, "bf16")
    out, _ = _run(combo, x, None, rw, cw, fp8)
    deq = out.dequantize(dtype=torch.float32)
    assert torch.isfinite(deq).all()
    ref = _ref_fwd(combo, x.float())
    rel = (deq - ref).norm() / ref.norm().clamp_min(1e-6)
    tol = 0.12 if fp8 == "e4m3" else 0.30
    assert rel < tol, f"{combo}/{fp8}/{direction}: rel_err={rel:.4f}"


# ---------------------------------------------------------------------------
# Numerical parity vs an fp32 reference, mirroring tests/cpp/operator/
# test_cast_mxfp8.cu. The C++ gtests never exercise the CuTeDSL backend (it is
# registered from Python), so this re-runs the C++ methodology with the backend
# active. Same case selection as the C++ test:
#   * ops: GeLU family only (the C++ test has SiLU/ReLU/QGeLU/SReLU commented
#     out) -> CAST_ONLY=plain, CAST_DBIAS=dbias, CAST_ACT=gelu, CAST_DACT=dgelu,
#     CAST_DBIAS_DACT=dbias_dgelu
#   * direction = the C++ block_size: {1,32}=row, {32,1}=col, {32,32}=both
#   * three orthogonal sweeps (the C++ INSTANTIATE_TEST_SUITE_P blocks) instead
#     of one giant cross product
#   * no swizzle (the C++ cast test doesn't cover it; see
#     test_mxfp8_quantize_swizzle_fusion for the swizzled layout)
# Comparison also mirrors the C++ test: e8m0 scales bit-exact (zero tolerance,
# valid where the reference value matches the kernel input exactly, i.e. the
# no-activation ops), FP8 data within fp8 atol/rtol, dbias relaxed. Activation
# ops use a relative-error bound instead of bit-exact scales because the torch
# reference activation isn't bit-identical to TE's device activation (the C++
# test gets bit-exactness only by reusing TE's own host activation).
_PT_DT = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
_FP8_MAX_RCP = {"e4m3": 1.0 / 448.0, "e5m2": 1.0 / 57344.0}
_FP8_T = {"e4m3": torch.float8_e4m3fn, "e5m2": torch.float8_e5m2}

# Shapes: %32 (so the CuTeDSL backend handles them; non-%32 falls back to CUDA),
# mixing %128 and %32-not-%128 (the kernels' partial-tile / OOB edge).
_PARITY_SHAPES = [(128, 128), (256, 1024), (512, 512), (160, 160), (128, 1056), (256, 384)]
# (op for _run, is_activation): the GeLU-family ProcessingMethods.
_CPP_OPS = [
    ("plain", False),
    ("dbias", False),
    ("gelu", True),
    ("dgelu", True),
    ("dbias_dgelu", True),
]
_CPP_DIRECTIONS = ["row", "col", "both"]


def _parity_inputs(M, N, in_dtype, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    dt = _PT_DT[in_dtype]
    x = torch.empty(M, N, dtype=dt, device="cuda").uniform_(-4.0, 4.0, generator=g)
    ain = torch.empty(M, N, dtype=dt, device="cuda").uniform_(-3.0, 3.0, generator=g)
    return x, ain


def _ref_value(op, x, ain):
    """fp32 reference of the (pre-quantization) tensor the kernel quantizes."""
    xf = x.float()
    if op in ("plain", "dbias"):
        return xf
    if op == "gelu":
        return _ref_fwd("gelu", xf)
    # dgelu / dbias_dgelu: grad (x) * d(gelu)/d(input), via autograd of the
    # matching forward so it tracks TE's exact tanh-gelu derivative.
    av = ain.float().detach().requires_grad_(True)
    _ref_fwd("gelu", av).backward(xf)
    return av.grad


def _ref_e8m0(amax_rcp):
    """fp32 (amax * max_reciprocal) -> e8m0 scale byte. Round-up of the biased
    exponent; bit-identical to the Blackwell cvt.rp.ue8m0x2 the kernel uses."""
    bits = amax_rcp.contiguous().view(torch.int32)
    exp = ((bits + 0x7FFFFF) >> 23) & 0xFF
    return exp.clamp(max=254).to(torch.uint8)


def _kernel_scale_data(out, d, M, N, fp8):
    """(e8m0 scales, dequantized output) for the meaningful region, direction d."""
    if d == "row":
        sc = out._rowwise_scale_inv[:M, : (N + 31) // 32]
        data = out._rowwise_data.view(_FP8_T[fp8]).float()
        deq = data * torch.exp2(sc.float() - 127.0).repeat_interleave(32, dim=1)[:, :N]
    else:
        sc = out._columnwise_scale_inv[: (M + 31) // 32, :N]
        data = out._columnwise_data.view(_FP8_T[fp8]).float()
        deq = data * torch.exp2(sc.float() - 127.0).repeat_interleave(32, dim=0)[:M, :]
    return sc, deq


def _ref_scales(v, d, fp8):
    """Reference e8m0 scales for value v (fp32), direction d."""
    M, N = v.shape
    if d == "row":
        amax = v.reshape(M, N // 32, 32).abs().amax(-1)  # (M, N//32)
    else:
        amax = v.reshape(M // 32, 32, N).abs().amax(1)  # (M//32, N)
    return _ref_e8m0(amax * _FP8_MAX_RCP[fp8])


def _check_parity(op, is_act, direction, M, N, in_dtype, fp8):
    rw = direction in ("row", "both")
    cw = direction in ("col", "both")
    x, ain = _parity_inputs(M, N, in_dtype)
    v = _ref_value(op, x, ain)
    out, db = _run(op, x, ain, rw, cw, fp8)

    tol = 0.12 if fp8 == "e4m3" else 0.30
    for d in (["row"] if rw else []) + (["col"] if cw else []):
        sc, deq = _kernel_scale_data(out, d, M, N, fp8)
        assert torch.isfinite(deq).all(), f"{op}/{d}/{fp8}/{in_dtype} {M}x{N}: non-finite"
        # Data: dequant within MXFP8 granularity (the C++ fp8 atol/rtol bar).
        rel = (deq - v).norm() / v.norm().clamp_min(1e-6)
        assert rel < tol, f"{op}/{d}/{fp8}/{in_dtype} {M}x{N}: rel_err={rel:.4f}"
        # Scales: bit-exact vs the fp32 reference (C++ zero-tolerance) — only for
        # no-activation ops, where the reference value equals the kernel input.
        if not is_act:
            assert torch.equal(
                sc, _ref_scales(v, d, fp8)
            ), f"{op}/{d}/{fp8}/{in_dtype} {M}x{N}: e8m0 scales differ from reference"

    if db is not None:
        dref = v.sum(dim=0)
        drel = (db.float() - dref).norm() / dref.norm().clamp_min(1e-6)
        assert drel < 0.1, f"{op}/{in_dtype} {M}x{N}: dbias rel_err={drel:.4f}"


# Sweep 1 — CAST_ONLY across all shapes/directions/dtypes/formats
# (C++ OperatorTest_FusedCastMXFP8_CastOnly).
@pytest.mark.skipif(not recipe_available, reason=reason_for_no_recipe)
@pytest.mark.parametrize("shape", _PARITY_SHAPES, ids=lambda s: f"{s[0]}x{s[1]}")
@pytest.mark.parametrize("in_dtype", ["bf16", "fp16", "fp32"])
@pytest.mark.parametrize("fp8", ["e4m3", "e5m2"])
@pytest.mark.parametrize("direction", _CPP_DIRECTIONS)
def test_parity_cast_only(direction, fp8, in_dtype, shape):
    _check_parity("plain", False, direction, *shape, in_dtype, fp8)


# Sweep 2 — all ops/directions/shapes at bf16/e4m3
# (C++ OperatorTest_FusedCastMXFP8_Sizes).
@pytest.mark.skipif(not recipe_available, reason=reason_for_no_recipe)
@pytest.mark.parametrize("shape", _PARITY_SHAPES, ids=lambda s: f"{s[0]}x{s[1]}")
@pytest.mark.parametrize("direction", _CPP_DIRECTIONS)
@pytest.mark.parametrize("op,is_act", _CPP_OPS, ids=[o for o, _ in _CPP_OPS])
def test_parity_ops_and_sizes(op, is_act, direction, shape):
    _check_parity(op, is_act, direction, *shape, "bf16", "e4m3")


# Sweep 3 — all ops/dtypes/formats at a fixed both-direction shape
# (C++ OperatorTest_FusedCastMXFP8_Dtypes, {256,384}, block {32,32}).
@pytest.mark.skipif(not recipe_available, reason=reason_for_no_recipe)
@pytest.mark.parametrize("in_dtype", ["bf16", "fp16", "fp32"])
@pytest.mark.parametrize("fp8", ["e4m3", "e5m2"])
@pytest.mark.parametrize("op,is_act", _CPP_OPS, ids=[o for o, _ in _CPP_OPS])
def test_parity_dtypes(op, is_act, fp8, in_dtype):
    _check_parity(op, is_act, "both", 256, 384, in_dtype, fp8)
