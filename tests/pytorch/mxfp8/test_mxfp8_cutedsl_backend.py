# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Cross-backend bit-exactness tests for the CuTeDSL MXFP8 quantize kernels."""

import ctypes
import os
from typing import Callable, NamedTuple, Optional

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
if not hasattr(CORE_LIB, "nvte_set_cutedsl_quant_backend"):
    raise RuntimeError(
        "libtransformer_engine.so lacks nvte_set_cutedsl_quant_backend -- rebuild the "
        "Transformer Engine core library."
    )

pytestmark = pytest.mark.skipif(not recipe_available, reason=reason_for_no_recipe)


class Fusion(NamedTuple):
    """An ActivationType from the C++ test: its tex ops per ProcessingMethod and
    the activation desc used in the CuTeDSL config key."""

    name: str
    act: Optional[Callable]  # CAST_ACT:        act(x, quantizer)
    dact: Optional[Callable]  # CAST_DACT:       dact(grad, act_input, quantizer)
    dbias_dact: Optional[Callable]  # CAST_DBIAS_DACT: dbias_dact(grad, act_input, quantizer)
    desc: str


# --- Case matrix, mirroring test_cast_mxfp8.cu ---
# The C++ multi-dim sizes flattened to the 2D (rows, cols) the kernels see:
# {8,32,1024} -> (256, 1024), {16,8,4,512} -> (512, 512). The C++ list also has
# non-32-divisible shapes ({1,16}, {16,48}, {993,512}, {1024}) that exercise the
# CUDA kernels' partial-block edges; the CuTeDSL backend's contract is
# 32-divisible flat dims (the dispatcher falls back to CUDA otherwise), so those
# cases are omitted here rather than vacuously comparing CUDA against CUDA.
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
IDENTITY = Fusion("Identity", None, None, None, "none")
GELU = Fusion("GeLU", tex.gelu, tex.dgelu, tex.dbias_dgelu, "gelu")
# SILU = Fusion("SiLU", tex.silu, tex.dsilu, tex.dbias_dsilu, "silu")
# RELU = Fusion("ReLU", tex.relu, tex.drelu, tex.dbias_drelu, "relu")
# QGELU = Fusion("QGeLU", tex.qgelu, tex.dqgelu, tex.dbias_dqgelu, "qgelu")
# SRELU = Fusion("SReLU", tex.srelu, tex.dsrelu, tex.dbias_dsrelu, "srelu")

# Valid (ProcessingMethod, ActivationType) pairs. The C++ test crosses the two
# axes and GTEST_SKIPs the mismatched half; only the meaningful pairs are
# generated here. A newly enabled activation adds its ACT/DACT/DBIAS_DACT pairs.
METHOD_FUSION_CASES = [
    ("CAST_ONLY", IDENTITY),
    ("CAST_DBIAS", IDENTITY),
    ("CAST_ACT", GELU),
    ("CAST_DACT", GELU),
    ("CAST_DBIAS_DACT", GELU),
]
METHOD_FUSION_IDS = [f"{m}X{f.name}" for m, f in METHOD_FUSION_CASES]
IN_DTYPES = [torch.float32, torch.bfloat16, torch.float16]
FP8_DTYPES = [tex.DType.kFloat8E4M3, tex.DType.kFloat8E5M2]

# Description strings for pytest case ids and the CuTeDSL config key.
DTYPE_TO_STR = {torch.float32: "fp32", torch.bfloat16: "bf16", torch.float16: "fp16"}
FP8_TO_STR = {tex.DType.kFloat8E4M3: "e4m3", tex.DType.kFloat8E5M2: "e5m2"}

get_shape_id = lambda s: f"{s[0]}x{s[1]}"
get_block_id = lambda b: f"{b[0]}x{b[1]}"
get_dtype_id = DTYPE_TO_STR.get
get_fp8_id = FP8_TO_STR.get


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
    x = torch.empty(M, N, dtype=in_dtype, device="cuda").uniform_(-2.0, 1.0, generator=g)
    ain = torch.empty(M, N, dtype=in_dtype, device="cuda").uniform_(-2.0, 1.0, generator=g)
    return x, ain


def run_quantize(method, act, x, ain, rowwise, columnwise, fp8_dtype):
    """Quantize via the public dispatch; returns (mxfp8_tensor, dbias_or_None)."""
    q = MXFP8Quantizer(fp8_dtype=fp8_dtype, rowwise=rowwise, columnwise=columnwise)
    if method == "CAST_ONLY":
        return q(x), None
    if method == "CAST_DBIAS":
        db, out = tex.bgrad_quantize(x, q)
        return out, db
    if method == "CAST_ACT":
        return act.act(x, q), None
    if method == "CAST_DACT":
        return act.dact(x, ain, q), None
    if method == "CAST_DBIAS_DACT":
        db, out = act.dbias_dact(x, ain, q)
        return out, db
    raise ValueError(f"unknown method {method!r}")


def get_cfg_key(method, act, in_dtype, fp8_dtype, rowwise, colwise):
    """Mirror of MXFP8QuantConfig::to_key (quantize_mxfp8_cutedsl.cuh): the name the CuTeDSL backend registers its compiled kernel under for this config.
    Used to check if the CuTeDSL implmentation is registered
    """
    with_dbias = method in ("CAST_DBIAS", "CAST_DBIAS_DACT")
    with_dact = method in ("CAST_DACT", "CAST_DBIAS_DACT")
    with_act = method == "CAST_ACT"
    desc = "none"
    if with_act:
        desc = act.desc
    elif with_dact:
        desc = f"d{act.desc}"
    flags = (rowwise, colwise, False, False, with_dbias, with_dact, with_act, False)
    return (
        "cutedsl_mxfp8_"
        + DTYPE_TO_STR[in_dtype]
        + "_"
        + FP8_TO_STR[fp8_dtype]
        + "_"
        + "_".join("1" if f else "0" for f in flags)
        + "_"
        + desc
    )


def extract_quantized_output(out, rowwise, columnwise):
    """Extract the meaningful bytes from the MXFP8Quantizer output for comparison between backends. The scale padding is uninitialized, so only the meaningful
    region is compared.
    """
    parts = {}
    if rowwise:
        d = out._rowwise_data.view(torch.uint8)
        M, N = d.shape
        parts["rowwise data"] = d.clone()
        parts["rowwise scales"] = out._rowwise_scale_inv[:M, : (N + 31) // 32].clone()
    if columnwise:
        d = out._columnwise_data.view(torch.uint8)
        M, N = d.shape
        parts["colwise data"] = d.clone()
        parts["colwise scales"] = out._columnwise_scale_inv[: (M + 31) // 32, :N].clone()
    return parts


def run_test_case(method, act, shape, block_size, in_dtype, fp8_dtype):
    """Assert the CuTeDSL and CUDA backends produce bit-identical outputs for the
    same input and config.
    """
    M, N = shape
    rowwise = block_size[1] != 1
    columnwise = block_size[0] != 1
    x, act_input = generate_inputs(M, N, in_dtype)

    set_cutedsl_backend(False)
    out_cuda, dbias_cuda = run_quantize(method, act, x, act_input, rowwise, columnwise, fp8_dtype)
    cuda_output = extract_quantized_output(out_cuda, rowwise, columnwise)

    set_cutedsl_backend(True)
    try:
        out_cutedsl, dbias_cutedsl = run_quantize(
            method, act, x, act_input, rowwise, columnwise, fp8_dtype
        )
        cutedsl_output = extract_quantized_output(out_cutedsl, rowwise, columnwise)
    finally:
        set_cutedsl_backend(False)

    # Guard against a silent CUDA fallback: every config in the matrix is one the
    # CuTeDSL backend supports, so its kernel must have been registered under the
    # config key. If not, the backend rejected or missed the config and the
    # comparison above was CUDA vs CUDA.
    key = get_cfg_key(method, act, in_dtype, fp8_dtype, rowwise, columnwise)
    assert tvm_ffi.get_global_func(key, allow_missing=True) is not None, (
        f"CuTeDSL kernel not registered for {key}; the CuTeDSL backend fell back "
        "to CUDA and this case compared CUDA against itself"
    )

    tag = f"{method}/{act.name}/{M}x{N}/{DTYPE_TO_STR[in_dtype]}/{FP8_TO_STR[fp8_dtype]}"
    for name, cuda_bytes in cuda_output.items():
        assert torch.equal(
            cutedsl_output[name], cuda_bytes
        ), f"{tag}: {name} differ between backends"
    if dbias_cuda is not None:
        torch.testing.assert_close(dbias_cutedsl, dbias_cuda)


# Test cases with only cast kernels (mirrors C++ test's OperatorTest_FusedCastMXFP8_CastOnly).
@pytest.mark.parametrize("shape", MATRIX_SIZES, ids=get_shape_id)
@pytest.mark.parametrize("block_size", BLOCK_SIZES, ids=get_block_id)
@pytest.mark.parametrize("in_dtype", IN_DTYPES, ids=get_dtype_id)
@pytest.mark.parametrize("fp8_dtype", FP8_DTYPES, ids=get_fp8_id)
def test_cast_only(fp8_dtype, in_dtype, block_size, shape):
    run_test_case("CAST_ONLY", IDENTITY, shape, block_size, in_dtype, fp8_dtype)


# Test cases with varying matrix shapes and block shapes
# (OperatorTest_FusedCastMXFP8_Sizes).
@pytest.mark.parametrize("shape", MATRIX_SIZES, ids=get_shape_id)
@pytest.mark.parametrize("block_size", BLOCK_SIZES, ids=get_block_id)
@pytest.mark.parametrize("method,act", METHOD_FUSION_CASES, ids=METHOD_FUSION_IDS)
def test_sizes(method, act, block_size, shape):
    run_test_case(method, act, shape, block_size, torch.bfloat16, tex.DType.kFloat8E4M3)


# Test cases with varying dtypes (OperatorTest_FusedCastMXFP8_Dtypes).
@pytest.mark.parametrize("in_dtype", IN_DTYPES, ids=get_dtype_id)
@pytest.mark.parametrize("fp8_dtype", FP8_DTYPES, ids=get_fp8_id)
@pytest.mark.parametrize("method,act", METHOD_FUSION_CASES, ids=METHOD_FUSION_IDS)
def test_dtypes(method, act, fp8_dtype, in_dtype):
    run_test_case(method, act, (256, 384), (32, 32), in_dtype, fp8_dtype)
