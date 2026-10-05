# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Cross-backend bit-exactness tests for the CuTeDSL MXFP8 quantize kernels, driven from pytorch."""

# Optional CuTeDSL dependencies must be checked before importing Transformer Engine.
# pylint: disable=wrong-import-position

import ctypes
import os
import subprocess
import sys
import textwrap

import pytest
import torch

tvm_ffi = pytest.importorskip("tvm_ffi")
pytest.importorskip("cutlass")

import transformer_engine.pytorch as te
import transformer_engine_torch as tex

from transformer_engine.common import _get_shared_object_file
from transformer_engine.common.CuTeDSL.utils import device_compute_capability
from transformer_engine.pytorch import MXFP8Quantizer

recipe_available, reason_for_no_recipe = te.is_mxfp8_available(return_reason=True)

CORE_LIB = ctypes.CDLL(str(_get_shared_object_file("core")))
cutedsl_built = hasattr(CORE_LIB, "nvte_is_cutedsl_backend_built") and bool(
    CORE_LIB.nvte_is_cutedsl_backend_built()
)
cutedsl_enabled = os.environ.get("NVTE_ENABLE_CUTEDSL_BACKEND", "0") != "0"
if not cutedsl_built:
    skip_reason = "Transformer Engine was built without NVTE_WITH_CUTEDSL=1"
elif not cutedsl_enabled:
    skip_reason = "NVTE_ENABLE_CUTEDSL_BACKEND is not set"
else:
    skip_reason = reason_for_no_recipe
pytestmark = pytest.mark.skipif(
    not (recipe_available and cutedsl_built and cutedsl_enabled),
    reason=skip_reason,
)

# We reject irregular shapes in transformer_engine/pytorch/csrc/quantizer.cpp's MXFP8Quantizer::get_scale_shape
# and CuTeDSL's divisibility assumption also strictly requires 32x32 alignment.
MATRIX_SIZES = [
    (128, 128),
    (256, 1024),
    (512, 512),
    (8192, 7168),
    # N is 32- but not 64-divisible, so it does not tile the general kernel's TILE_COLS
    # exactly and the entry keeps the per-thread scale-bounds guard.
    (128, 96),
    # M is 32- but not 64-divisible, so the last CTA's block hangs past M and num_tiles
    # clamps it. Covers the clamp that makes TILE_ROWS (not TILE_ROWS*NUM_TILES) the bound.
    (96, 128),
]
# (block_rows, block_cols): (1,32)=rowwise, (32,1)=colwise, (32,32)=both.
BLOCK_SIZES = [(1, 32), (32, 1), (32, 32)]

# Every activation the CuTeDSL backend implements; "desc" is the name it goes by in the
# config key (Activation in common/util/cutedsl_utils.h). Gated variants are absent because
# they go to a separate TE/common kernel that the backend does not cover.
IDENTITY = {"name": "Identity", "act": None, "dact": None, "dbias_dact": None, "desc": "none"}
ACTIVATIONS = [
    {
        "name": "GeLU",
        "act": tex.gelu,
        "dact": tex.dgelu,
        "dbias_dact": tex.dbias_dgelu,
        "desc": "gelu",
    },
    {
        "name": "ReLU",
        "act": tex.relu,
        "dact": tex.drelu,
        "dbias_dact": tex.dbias_drelu,
        "desc": "relu",
    },
    {
        "name": "SiLU",
        "act": tex.silu,
        "dact": tex.dsilu,
        "dbias_dact": tex.dbias_dsilu,
        "desc": "silu",
    },
    {
        "name": "QGeLU",
        "act": tex.qgelu,
        "dact": tex.dqgelu,
        "dbias_dact": tex.dbias_dqgelu,
        "desc": "qgelu",
    },
    {
        "name": "SReLU",
        "act": tex.srelu,
        "dact": tex.dsrelu,
        "dbias_dact": tex.dbias_dsrelu,
        "desc": "srelu",
    },
]
METHOD_FUSION_CASES = [
    ("CAST_ONLY", IDENTITY),
    ("CAST_DBIAS", IDENTITY),
] + [
    (method, act) for method in ("CAST_ACT", "CAST_DACT", "CAST_DBIAS_DACT") for act in ACTIVATIONS
]
METHOD_FUSION_IDS = [f"{m}X{f['name']}" for m, f in METHOD_FUSION_CASES]

IN_DTYPES = [torch.float32, torch.bfloat16, torch.float16]
FP8_DTYPES = [tex.DType.kFloat8E4M3, tex.DType.kFloat8E5M2]
FP8_TO_KEY = {
    tex.DType.kFloat8E4M3: "Float8E4M3",
    tex.DType.kFloat8E5M2: "Float8E5M2",
}

SWIZZLE_MODES = [False, True]

get_shape_id = lambda s: f"{s[0]}x{s[1]}"
get_block_id = lambda b: f"{b[0]}x{b[1]}"
DTYPE_TO_STR = {
    torch.float32: "Float32",
    torch.bfloat16: "BFloat16",
    torch.float16: "Float16",
}
get_dtype_id = DTYPE_TO_STR.get
FP8_TO_STR = {tex.DType.kFloat8E4M3: "e4m3", tex.DType.kFloat8E5M2: "e5m2"}
get_fp8_id = FP8_TO_STR.get
get_swizzle_id = lambda s: "swizzled" if s else "non-swizzled"


def set_cutedsl_backend(enabled):
    CORE_LIB.nvte_set_cutedsl_backend(1 if enabled else 0)


def test_enable_cutedsl_backend_after_import():
    """Test if we can manually enable the CuTeDSL backend without enabling CuTeDSL backend in the beginning."""
    script = textwrap.dedent("""
        import ctypes

        import transformer_engine.pytorch  # pylint: disable=unused-import
        import tvm_ffi

        from transformer_engine.common import _get_shared_object_file

        entrypoint = "get_mxfp8_quantization_function"
        assert tvm_ffi.get_global_func(entrypoint, allow_missing=True) is None

        core = ctypes.CDLL(str(_get_shared_object_file("core")))
        setter = core.nvte_set_cutedsl_backend
        setter.argtypes = [ctypes.c_int]
        setter.restype = None
        setter(1)

        assert tvm_ffi.get_global_func(entrypoint, allow_missing=True) is not None
        """)
    env = os.environ.copy()
    env["NVTE_ENABLE_CUTEDSL_BACKEND"] = "0"
    result = subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.fixture(scope="module", autouse=True)
def _restore_backend_choice_from_env():
    """Restore the flag that decides the CuTeDSL / CUDA backend choice when this pytest module is done."""
    yield
    flag = os.getenv("NVTE_ENABLE_CUTEDSL_BACKEND")
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
    Used to check if the CuTeDSL implementation is registered
    """
    with_dbias = method in ("CAST_DBIAS", "CAST_DBIAS_DACT")
    with_dact = method in ("CAST_DACT", "CAST_DBIAS_DACT")
    with_act = method == "CAST_ACT"
    desc = "none"
    if with_act:
        desc = act["desc"]
    elif with_dact:
        desc = f"d{act['desc']}"
    # with_amax is hardcoded to False for now because currently MXFP8 quantizers never allocate an amax
    # The last False is for use_2d_quantization; these cases never request 2D block scaling
    flags = (rowwise, colwise, swizzled, False, with_dbias, with_dact, with_act, False)
    major, minor = device_compute_capability()
    return (
        f"cutedsl_mxfp8_sm{major * 10 + minor}_"
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
        f"{method}/{act['name']}/{M}x{N}/{DTYPE_TO_STR[in_dtype]}/{FP8_TO_STR[fp8_dtype]}/{layout}"
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


def test_cast_only_rowwise_specialized_grid_y_overflow():
    """Fall back when the specialized rowwise launch would exceed CUDA's grid.y limit."""
    # The specialized kernel covers four rows per CTA, so this shape would require
    # grid.y=65536. The general kernel must handle it instead.
    run_test_case(
        "CAST_ONLY",
        IDENTITY,
        (4 * 65536, 128),
        (1, 32),
        torch.float16,
        tex.DType.kFloat8E4M3,
    )


def test_cast_only_bidimensional_specialized_grid_y_overflow():
    """Fall back when the specialized bidimensional launch exceeds CUDA's grid.y limit."""
    # The specialized kernel covers 32 rows per CTA, so this shape would require
    # grid.y=65536. Use the minimum valid N to limit the test's memory footprint.
    run_test_case(
        "CAST_ONLY",
        IDENTITY,
        (32 * 65536, 32),
        (32, 32),
        torch.float16,
        tex.DType.kFloat8E4M3,
    )


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


# Grouped quantization (nvte_group_quantize / nvte_group_quantize_dbias). Every member's first
# dim is a multiple of 128, which both grouped kernels require. VARYING_LAST_DIM and
# VARYING_BOTH_DIMS members also keep their last dims 128-aligned, as both kernels' per-member
# scale layout assumes. The fused activations have no PyTorch binding; the CuTeDSL path for
# them is covered by tests/cpp/operator/test_cast_mxfp8_grouped.cu run with
# NVTE_ENABLE_CUTEDSL_BACKEND=1.
# (name, shape representation in the config key, per-member shapes)
GROUP_CASES = [
    ("same_both", "sbd", [(256, 512)] * 3),
    ("single_member", "sbd", [(384, 384)]),
    # N is 32- but not 128-divisible, so the rowwise and colwise scales carry zeroed padding.
    ("same_both_n96", "sbd", [(128, 96)] * 2),
    ("varying_first", "vfd", [(128, 256), (384, 256), (256, 256)]),
    ("varying_first_n160", "vfd", [(128, 160), (256, 160)]),
    # Partial 32-element blocks (e.g. N=144) are covered by the C++ grouped tests;
    # the PyTorch MXFP8 quantizer requires dimensions divisible by 32.
    ("varying_last", "vld", [(256, 128), (256, 384), (256, 256)]),
    ("varying_both", "vbd", [(128, 256), (256, 128), (384, 512)]),
    # Multiple chunks in both dimensions, including a final half-width 128-column strip.
    ("varying_both_multichunk", "vbd", [(128, 128), (256, 384), (384, 640)]),
    ("varying_first_empty", "vfd", [(128, 256), (0, 256), (384, 256)]),
    ("varying_both_empty", "vbd", [(128, 128), (0, 256), (256, 384)]),
]
SINGLE_TENSOR_GROUP_CASES = [c for c in GROUP_CASES if c[1] in ("sbd", "vfd")]
get_group_case_id = lambda c: c[0]


def group_dims(members):
    """(logical shape, first_dims, last_dims) of the grouped tensor made of `members`."""
    first_dims = [m for m, _ in members]
    last_dims = [n for _, n in members]
    same_first = len(set(first_dims)) == 1
    same_last = len(set(last_dims)) == 1
    if same_last:
        logical_shape = (sum(first_dims), last_dims[0])
    elif same_first:
        logical_shape = (first_dims[0], sum(last_dims))
    else:
        logical_shape = (1, sum(m * n for m, n in members))
    to_tensor = lambda dims: torch.tensor(dims, dtype=torch.int64, device="cuda")
    return (
        logical_shape,
        None if same_first else to_tensor(first_dims),
        None if same_last else to_tensor(last_dims),
    )


def run_group_quantize(members, in_dtype, fp8_dtype, rowwise, columnwise, swizzled, dbias):
    """Quantize the concatenated members; returns (grouped output, dbias or None)."""
    logical_shape, first_dims, last_dims = group_dims(members)
    x, _ = generate_inputs(*logical_shape, in_dtype)
    q = MXFP8Quantizer(fp8_dtype=fp8_dtype, rowwise=rowwise, columnwise=columnwise)
    q.optimize_for_gemm = swizzled
    if dbias:
        return tex.bgrad_group_quantize(x, q, len(members), first_dims, last_dims)
    return tex.group_quantize(x, q, len(members), first_dims, last_dims), None


def extract_group_quantized_output(out, rowwise, columnwise):
    """Extract the bytes to compare between backends.

    Unlike the single-tensor kernels, both grouped backends zero the scale padding, so the
    whole data and scale buffers are compared.
    """
    parts = {}
    if rowwise:
        parts["rowwise data"] = out.rowwise_data.view(torch.uint8).clone()
        parts["rowwise scales"] = out.scale_inv.view(torch.uint8).clone()
    if columnwise:
        parts["colwise data"] = out.columnwise_data.view(torch.uint8).clone()
        parts["colwise scales"] = out.columnwise_scale_inv.view(torch.uint8).clone()
    return parts


def get_group_cfg_key(shape_rep, in_dtype, fp8_dtype, rowwise, colwise, swizzled, dbias):
    """Mirror of MXFP8GroupQuantConfig::to_key (group_quantize_mxfp8_cutedsl.cuh) for the
    configs reachable from PyTorch (no fused activation)."""
    major, minor = device_compute_capability()
    flags = (swizzled, dbias, False, False)  # swizzled, with_dbias, with_dact, with_act
    return (
        f"cutedsl_group_mxfp8_sm{major * 10 + minor}_{DTYPE_TO_STR[in_dtype]}_"
        f"{FP8_TO_KEY[fp8_dtype]}_{int(rowwise)}_{int(colwise)}_{shape_rep}_"
        + "_".join("1" if f else "0" for f in flags)
        + "_none"
    )


def assert_group_cutedsl_registered(*key_args):
    """Guard against a silent CUDA fallback; see run_test_case."""
    key = get_group_cfg_key(*key_args)
    assert tvm_ffi.get_global_func(key, allow_missing=True) is not None, (
        f"CuTeDSL kernel not registered for {key}; the CuTeDSL backend fell back "
        "to CUDA and this case compared CUDA against itself"
    )


def run_group_test_case(
    members, shape_rep, block_size, in_dtype, fp8_dtype, swizzled=False, dbias=False
):
    """Assert the CuTeDSL and CUDA grouped backends produce bit-identical outputs, including
    dbias, which both accumulate in the same order."""
    rowwise = block_size[1] != 1
    columnwise = block_size[0] != 1
    args = (members, in_dtype, fp8_dtype, rowwise, columnwise, swizzled, dbias)

    set_cutedsl_backend(False)
    out_cuda, dbias_cuda = run_group_quantize(*args)
    cuda_output = extract_group_quantized_output(out_cuda, rowwise, columnwise)

    set_cutedsl_backend(True)
    try:
        out_cutedsl, dbias_cutedsl = run_group_quantize(*args)
        cutedsl_output = extract_group_quantized_output(out_cutedsl, rowwise, columnwise)
    finally:
        set_cutedsl_backend(False)

    assert_group_cutedsl_registered(
        shape_rep, in_dtype, fp8_dtype, rowwise, columnwise, swizzled, dbias
    )
    tag = f"group/{members}/{DTYPE_TO_STR[in_dtype]}/{FP8_TO_STR[fp8_dtype]}"
    for name, cuda_bytes in cuda_output.items():
        assert torch.equal(
            cutedsl_output[name], cuda_bytes
        ), f"{tag}: {name} differ between backends"
    if dbias:
        assert torch.equal(
            dbias_cutedsl.view(torch.uint8), dbias_cuda.view(torch.uint8)
        ), f"{tag}: dbias differs between backends"


@pytest.mark.parametrize("case", GROUP_CASES, ids=get_group_case_id)
@pytest.mark.parametrize("block_size", BLOCK_SIZES, ids=get_block_id)
@pytest.mark.parametrize("in_dtype", IN_DTYPES, ids=get_dtype_id)
@pytest.mark.parametrize("fp8_dtype", FP8_DTYPES, ids=get_fp8_id)
def test_group_cast_only(fp8_dtype, in_dtype, block_size, case):
    _, shape_rep, members = case
    run_group_test_case(members, shape_rep, block_size, in_dtype, fp8_dtype)


@pytest.mark.parametrize("case", GROUP_CASES, ids=get_group_case_id)
@pytest.mark.parametrize("block_size", BLOCK_SIZES, ids=get_block_id)
def test_group_swizzled(block_size, case):
    _, shape_rep, members = case
    run_group_test_case(
        members, shape_rep, block_size, torch.bfloat16, tex.DType.kFloat8E4M3, swizzled=True
    )


@pytest.mark.parametrize("case", SINGLE_TENSOR_GROUP_CASES, ids=get_group_case_id)
@pytest.mark.parametrize("block_size", BLOCK_SIZES, ids=get_block_id)
@pytest.mark.parametrize("in_dtype", IN_DTYPES, ids=get_dtype_id)
@pytest.mark.parametrize("swizzled", SWIZZLE_MODES, ids=get_swizzle_id)
def test_group_dbias(swizzled, in_dtype, block_size, case):
    _, shape_rep, members = case
    run_group_test_case(
        members,
        shape_rep,
        block_size,
        in_dtype,
        tex.DType.kFloat8E4M3,
        swizzled=swizzled,
        dbias=True,
    )


@pytest.mark.parametrize("cutedsl", [False, True], ids=["cuda", "cutedsl"])
def test_group_noop(cutedsl):
    """A set cast-noop flag leaves a reused grouped output untouched; a clear one does not."""
    members = [(256, 512)] * 3
    logical_shape, _, _ = group_dims(members)
    x1, x2 = generate_inputs(*logical_shape, torch.bfloat16)
    q = MXFP8Quantizer(fp8_dtype=tex.DType.kFloat8E4M3, rowwise=True, columnwise=True)
    set_cutedsl_backend(cutedsl)
    try:
        out = tex.group_quantize(x1, q, len(members), None)
        first = extract_group_quantized_output(out, True, True)
        noop = torch.ones(1, dtype=torch.float32, device="cuda")
        tex.group_quantize(x2, q, len(members), None, noop_flag=noop, output=out)
        skipped = extract_group_quantized_output(out, True, True)
        noop.zero_()
        tex.group_quantize(x2, q, len(members), None, noop_flag=noop, output=out)
        quantized = extract_group_quantized_output(out, True, True)
        expected = extract_group_quantized_output(
            tex.group_quantize(x2, q, len(members), None), True, True
        )
    finally:
        set_cutedsl_backend(False)
    if cutedsl:
        assert_group_cutedsl_registered(
            "sbd", torch.bfloat16, tex.DType.kFloat8E4M3, True, True, False, False
        )
    for name, first_bytes in first.items():
        assert torch.equal(skipped[name], first_bytes), f"{name} changed under a set noop flag"
        assert torch.equal(quantized[name], expected[name]), f"{name} wrong under a clear flag"


def test_group_2d_quantization_fallback():
    """2D block scaling is left to the CUDA kernel, which must still be what runs."""
    members = [(128, 256), (384, 256), (256, 256)]
    logical_shape, first_dims, _ = group_dims(members)
    x, _ = generate_inputs(*logical_shape, torch.bfloat16)
    outputs = []
    for cutedsl in (False, True):
        q = MXFP8Quantizer(
            fp8_dtype=tex.DType.kFloat8E4M3,
            rowwise=True,
            columnwise=True,
            with_2d_quantization=True,
        )
        set_cutedsl_backend(cutedsl)
        try:
            out = tex.group_quantize(x, q, len(members), first_dims)
            outputs.append(extract_group_quantized_output(out, True, True))
        finally:
            set_cutedsl_backend(False)
    for name, cuda_bytes in outputs[0].items():
        assert torch.equal(outputs[1][name], cuda_bytes), f"{name} differ between backends"
