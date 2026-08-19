# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Cross-backend bit-exactness tests for the CuTeDSL MXFP8 quantize kernels."""

import ctypes
import os

import pytest
import torch

import transformer_engine.pytorch as te
import transformer_engine_torch as tex
import tvm_ffi

from transformer_engine.common import _get_shared_object_file
from transformer_engine.pytorch import MXFP8Quantizer

recipe_available, reason_for_no_recipe = te.is_mxfp8_available(return_reason=True)

# The already-loaded core lib (dlopen refcounts: this returns the same handle,
# so the call mutates the same dispatcher singleton the quantize ops read).
CORE_LIB = ctypes.CDLL(str(_get_shared_object_file("core")))
# We need this API to manually enable & disable the CuTeDSL backend for the tests
if not hasattr(CORE_LIB, "nvte_set_cutedsl_quant_backend"):
    raise RuntimeError(
        "libtransformer_engine.so lacks nvte_set_cutedsl_quant_backend -- rebuild the "
        "Transformer Engine core library."
    )

# The CuTeDSL entrypoint is registered only when NVTE_ENABLE_CUTEDSL_QUANT_BACKEND
# is set (see common/__init__.py); without it there is nothing to compare against
# the CUDA path, so skip these runs.
cutedsl_enabled = os.environ.get("NVTE_ENABLE_CUTEDSL_QUANT_BACKEND", "0") != "0"
pytestmark = pytest.mark.skipif(
    not (recipe_available and cutedsl_enabled),
    reason=reason_for_no_recipe or "NVTE_ENABLE_CUTEDSL_QUANT_BACKEND is not set",
)

# We reject irregular shapes in transformer_engine/pytorch/csrc/quantizer.cpp's MXFP8Quantizer::get_scale_shape
# and CuTeDSL's divisibility assumption also strictly requires 32x32 alignment.
MATRIX_SIZES = [
    (128, 128),
    (256, 1024),
    (512, 512),
    (8192, 7168),
]
# (block_rows, block_cols): (1,32)=rowwise, (32,1)=colwise, (32,32)=both.
BLOCK_SIZES = [(1, 32), (32, 1), (32, 32)]

# Only GeLU activation tests are used (SiLU/ReLU/QGeLU/SReLU commented out
# in the C++ test as well).
IDENTITY = {"name": "Identity", "act": None, "dact": None, "dbias_dact": None, "desc": "none"}
GELU = {
    "name": "GeLU",
    "act": tex.gelu,
    "dact": tex.dgelu,
    "dbias_dact": tex.dbias_dgelu,
    "desc": "gelu",
}
METHOD_FUSION_CASES = [
    ("CAST_ONLY", IDENTITY),
    ("CAST_DBIAS", IDENTITY),
    ("CAST_ACT", GELU),
    ("CAST_DACT", GELU),
    ("CAST_DBIAS_DACT", GELU),
]
METHOD_FUSION_IDS = [f"{m}X{f['name']}" for m, f in METHOD_FUSION_CASES]

IN_DTYPES = [torch.float32, torch.bfloat16, torch.float16]
FP8_DTYPES = [tex.DType.kFloat8E4M3, tex.DType.kFloat8E5M2]
FP8_TO_KEY = {
    tex.DType.kFloat8E4M3: "fp8_e4m3fn",
    tex.DType.kFloat8E5M2: "fp8_e5m2",
}

SWIZZLE_MODES = [False, True]

get_shape_id = lambda s: f"{s[0]}x{s[1]}"
get_block_id = lambda b: f"{b[0]}x{b[1]}"
DTYPE_TO_STR = {torch.float32: "fp32", torch.bfloat16: "bf16", torch.float16: "fp16"}
get_dtype_id = DTYPE_TO_STR.get
FP8_TO_STR = {tex.DType.kFloat8E4M3: "e4m3", tex.DType.kFloat8E5M2: "e5m2"}
get_fp8_id = FP8_TO_STR.get
get_swizzle_id = lambda s: "swizzled" if s else "non-swizzled"


def set_cutedsl_backend(enabled):
    CORE_LIB.nvte_set_cutedsl_quant_backend(1 if enabled else 0)


@pytest.fixture(scope="module", autouse=True)
def _restore_backend_choice_from_env():
    """Restore the flag that decides the CuTeDSL / CUDA backend choice when this pytest module is done."""
    yield
    flag = os.getenv("NVTE_ENABLE_CUTEDSL_QUANT_BACKEND")
    set_cutedsl_backend(flag is not None and not flag.startswith("0"))


def generate_inputs(M, N, in_dtype, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)

    def fill():
        # Mirrors InputsFillCase::uniform in fillCase_special (tests/cpp/test_common.cu) where the uniform range is [-2, 1]
        # and we apply a random sign flip
        v = torch.empty(M, N, dtype=torch.float32, device="cuda").uniform_(-2.0, 1.0, generator=g)
        negate = (
            torch.empty(M, N, dtype=torch.float32, device="cuda").uniform_(-1.0, 1.0, generator=g)
            < 0.0
        )
        return torch.where(negate, -v, v).to(in_dtype)

    return fill(), fill()


def run_quantize(method, act, x, ain, rowwise, columnwise, fp8_dtype, swizzled):
    """Quantize via the public dispatch; returns (mxfp8_tensor, dbias_or_None)."""
    q = MXFP8Quantizer(fp8_dtype=fp8_dtype, rowwise=rowwise, columnwise=columnwise)
    # Emit scales in the GEMM-swizzled layout (MXFP8QuantConfig::swizzled).
    q.optimize_for_gemm = swizzled
    if method == "CAST_ONLY":
        return q(x), None
    if method == "CAST_DBIAS":
        db, out = tex.bgrad_quantize(x, q)
        return out, db
    if method == "CAST_ACT":
        return act["act"](x, q), None
    if method == "CAST_DACT":
        return act["dact"](x, ain, q), None
    if method == "CAST_DBIAS_DACT":
        db, out = act["dbias_dact"](x, ain, q)
        return out, db
    raise ValueError(f"unknown method {method!r}")


def get_cfg_key(method, act, in_dtype, fp8_dtype, rowwise, colwise, swizzled):
    """Mirror of MXFP8QuantConfig::to_key (quantize_mxfp8_cutedsl.cuh): the name the CuTeDSL backend registers its compiled kernel under for this config.
    Used to check if the CuTeDSL implmentation is registered
    """
    with_dbias = method in ("CAST_DBIAS", "CAST_DBIAS_DACT")
    with_dact = method in ("CAST_DACT", "CAST_DBIAS_DACT")
    with_act = method == "CAST_ACT"
    desc = "none"
    if with_act:
        desc = act["desc"]
    elif with_dact:
        desc = f"d{act['desc']}"
    # with_amax is hardcoded to False for now because there is no way to obtain this value and validate in python
    # trailing False is use_2d_quantization; these cases never request 2D block scaling
    flags = (rowwise, colwise, swizzled, False, with_dbias, with_dact, with_act, False)
    return (
        "cutedsl_mxfp8_"
        + DTYPE_TO_STR[in_dtype]
        + "_"
        + FP8_TO_KEY[fp8_dtype]
        + "_"
        + "_".join("1" if f else "0" for f in flags)
        + "_"
        + desc
    )


def extract_quantized_output(out, rowwise, columnwise, swizzled):
    """Extract the bytes to compare between backends.

    Linear layout: the scale padding is uninitialized, so only the meaningful region is
    compared. Swizzled layout: the meaningful scales are scattered by the swizzle, so the
    top-left slice is meaningless; both backends zero the padding (see zero_scales_kernel),
    so compare the whole buffer instead -- which also covers the padding-zeroing itself.
    """
    parts = {}
    if rowwise:
        d = out._rowwise_data.view(torch.uint8)
        M, N = d.shape
        parts["rowwise data"] = d.clone()
        s = out._rowwise_scale_inv
        parts["rowwise scales"] = (s if swizzled else s[:M, : (N + 31) // 32]).clone()
    if columnwise:
        d = out._columnwise_data.view(torch.uint8)
        M, N = d.shape
        parts["colwise data"] = d.clone()
        s = out._columnwise_scale_inv
        parts["colwise scales"] = (s if swizzled else s[: (M + 31) // 32, :N]).clone()
    return parts


def run_test_case(method, act, shape, block_size, in_dtype, fp8_dtype, swizzled=False):
    """Assert the CuTeDSL and CUDA backends produce bit-identical outputs for the
    same input and config.
    """
    M, N = shape
    rowwise = block_size[1] != 1
    columnwise = block_size[0] != 1
    x, act_input = generate_inputs(M, N, in_dtype)

    set_cutedsl_backend(False)
    out_cuda, dbias_cuda = run_quantize(
        method, act, x, act_input, rowwise, columnwise, fp8_dtype, swizzled
    )
    cuda_output = extract_quantized_output(out_cuda, rowwise, columnwise, swizzled)

    set_cutedsl_backend(True)
    try:
        out_cutedsl, dbias_cutedsl = run_quantize(
            method, act, x, act_input, rowwise, columnwise, fp8_dtype, swizzled
        )
        cutedsl_output = extract_quantized_output(out_cutedsl, rowwise, columnwise, swizzled)
    finally:
        set_cutedsl_backend(False)

    # Guard against a silent CUDA fallback: every config in the matrix is one the
    # CuTeDSL backend supports, so its kernel must have been registered under the
    # config key. If not, the backend rejected or missed the config and the
    # comparison above was CUDA vs CUDA.
    key = get_cfg_key(method, act, in_dtype, fp8_dtype, rowwise, columnwise, swizzled)
    assert tvm_ffi.get_global_func(key, allow_missing=True) is not None, (
        f"CuTeDSL kernel not registered for {key}; the CuTeDSL backend fell back "
        "to CUDA and this case compared CUDA against itself"
    )

    layout = "swizzled" if swizzled else "linear"
    tag = (
        f"{method}/{act["name"]}/{M}x{N}/{DTYPE_TO_STR[in_dtype]}/{FP8_TO_STR[fp8_dtype]}/{layout}"
    )
    for name, cuda_bytes in cuda_output.items():
        assert torch.equal(
            cutedsl_output[name], cuda_bytes
        ), f"{tag}: {name} differ between backends"
    if dbias_cuda is not None:
        # CuTeDSL kernel does dbias reduction in a slightly different order than the CUDA kernel,
        # due to the non-associativity of floating-point addition, this will not be bit-identical.
        torch.testing.assert_close(dbias_cutedsl, dbias_cuda)


# Test cases with only cast kernels (mirrors C++ test's OperatorTest_FusedCastMXFP8_CastOnly).
@pytest.mark.parametrize("shape", MATRIX_SIZES, ids=get_shape_id)
@pytest.mark.parametrize("block_size", BLOCK_SIZES, ids=get_block_id)
@pytest.mark.parametrize("in_dtype", IN_DTYPES, ids=get_dtype_id)
@pytest.mark.parametrize("fp8_dtype", FP8_DTYPES, ids=get_fp8_id)
@pytest.mark.parametrize("swizzled", SWIZZLE_MODES, ids=get_swizzle_id)
def test_cast_only(swizzled, fp8_dtype, in_dtype, block_size, shape):
    run_test_case("CAST_ONLY", IDENTITY, shape, block_size, in_dtype, fp8_dtype, swizzled)


# Test cases with varying matrix shapes and block shapes
# (OperatorTest_FusedCastMXFP8_Sizes).
@pytest.mark.parametrize("shape", MATRIX_SIZES, ids=get_shape_id)
@pytest.mark.parametrize("block_size", BLOCK_SIZES, ids=get_block_id)
@pytest.mark.parametrize("method,act", METHOD_FUSION_CASES, ids=METHOD_FUSION_IDS)
@pytest.mark.parametrize("swizzled", SWIZZLE_MODES, ids=get_swizzle_id)
def test_sizes(swizzled, method, act, block_size, shape):
    run_test_case(method, act, shape, block_size, torch.bfloat16, tex.DType.kFloat8E4M3, swizzled)


# Test cases with varying dtypes (OperatorTest_FusedCastMXFP8_Dtypes).
@pytest.mark.parametrize("in_dtype", IN_DTYPES, ids=get_dtype_id)
@pytest.mark.parametrize("fp8_dtype", FP8_DTYPES, ids=get_fp8_id)
@pytest.mark.parametrize("method,act", METHOD_FUSION_CASES, ids=METHOD_FUSION_IDS)
@pytest.mark.parametrize("swizzled", SWIZZLE_MODES, ids=get_swizzle_id)
def test_dtypes(swizzled, method, act, fp8_dtype, in_dtype):
    run_test_case(method, act, (256, 384), (32, 32), in_dtype, fp8_dtype, swizzled)
