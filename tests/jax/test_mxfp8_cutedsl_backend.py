# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Cross-backend bit-exactness tests for the CuTeDSL MXFP8 quantize kernels, driven from JAX.

JAX companion to tests/pytorch/mxfp8/test_mxfp8_cutedsl_backend.py: the CuTeDSL dispatch
lives in TE/common, so this checks that the JAX FFI path reaches it and produces the same
bytes as the CUDA kernels.
"""

import ctypes
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import tvm_ffi

from utils import assert_allclose

from transformer_engine.common import _get_shared_object_file
from transformer_engine.jax import cpp_extensions as tex
from transformer_engine.jax.quantize import (
    QuantizerFactory,
    QuantizeLayout,
    ScaledTensor1x,
    ScalingMode,
    helper,
)

recipe_available, reason_for_no_recipe = helper.is_scaling_mode_supported(
    ScalingMode.MXFP8_1D_SCALING
)

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

# CuTeDSL's divisibility assumption strictly requires 32x32 alignment, and the JAX
# MXFP8 scale shapes additionally require 128-alignment for the fused dact paths.
MATRIX_SIZES = [
    (128, 128),
    (256, 1024),
    (512, 512),
    (8192, 7168),
]
# QuantizeLayout.COLWISE is absent on purpose: every quantize wrapper diverts colwise-only to a
# pure-JAX implementation before reaching the FFI (_quantize_dbias_impl, act_lu,
# quantize_dact_dbias), so it cannot exercise the kernels. The colwise-only kernels themselves are
# covered by tests/pytorch/mxfp8/test_mxfp8_cutedsl_backend.py.
Q_LAYOUTS = [QuantizeLayout.ROWWISE, QuantizeLayout.ROWWISE_COLWISE]

# Only GeLU activation tests are used (SiLU/ReLU/QGeLU/SReLU commented out
# in the C++ test as well). Gated variants go to a separate TE/common kernel
# that the CuTeDSL backend does not cover.
ACT_TYPE = ("gelu",)
METHODS = ["CAST_ONLY", "CAST_DBIAS", "CAST_ACT", "CAST_DACT", "CAST_DBIAS_DACT"]

IN_DTYPES = [jnp.float32, jnp.bfloat16, jnp.float16]
FP8_DTYPES = [jnp.float8_e4m3fn, jnp.float8_e5m2]
FP8_TO_KEY = {
    jnp.float8_e4m3fn: "fp8_e4m3fn",
    jnp.float8_e5m2: "fp8_e5m2",
}

get_shape_id = lambda s: f"{s[0]}x{s[1]}"
get_layout_id = lambda l: "rowwise" if l == QuantizeLayout.ROWWISE else "bidim"
DTYPE_TO_STR = {jnp.float32: "fp32", jnp.bfloat16: "bf16", jnp.float16: "fp16"}
get_dtype_id = DTYPE_TO_STR.get
FP8_TO_STR = {jnp.float8_e4m3fn: "e4m3", jnp.float8_e5m2: "e5m2"}
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
    keys = jax.random.split(jax.random.PRNGKey(seed), 4)

    def fill(k_value, k_sign):
        # Mirrors InputsFillCase::uniform in fillCase_special (tests/cpp/test_common.cu) where the
        # uniform range is [-2, 1] and we apply a random sign flip
        v = jax.random.uniform(k_value, (M, N), jnp.float32, -2.0, 1.0)
        negate = jax.random.uniform(k_sign, (M, N), jnp.float32, -1.0, 1.0) < 0.0
        return jnp.where(negate, -v, v).astype(in_dtype)

    x = fill(keys[0], keys[1])
    # The activation input is replicated along the -2 axis, one entry per activation
    # in the (possibly gated) activation type.
    act_input = jnp.expand_dims(fill(keys[2], keys[3]), axis=-2)
    return x, act_input


def run_quantize(method, x, act_input, q_layout, fp8_dtype):
    """Quantize via the public dispatch; returns (scaled_tensor, dbias_or_None)."""
    quantizer = QuantizerFactory.create(
        scaling_mode=ScalingMode.MXFP8_1D_SCALING, q_dtype=fp8_dtype, q_layout=q_layout
    )
    if method == "CAST_ONLY":
        return tex.quantize(x, quantizer=quantizer), None
    if method == "CAST_DBIAS":
        return tex.quantize_dbias(x, quantizer=quantizer)
    if method == "CAST_ACT":
        return tex.act_lu(act_input, ACT_TYPE, quantizer=quantizer), None
    if method == "CAST_DACT":
        out, _ = tex.quantize_dact_dbias(
            x, act_input, ACT_TYPE, is_dbias=False, quantizer=quantizer
        )
        return out, None
    if method == "CAST_DBIAS_DACT":
        return tex.quantize_dact_dbias(x, act_input, ACT_TYPE, is_dbias=True, quantizer=quantizer)
    raise ValueError(f"unknown method {method!r}")


def get_cfg_key(method, in_dtype, fp8_dtype, q_layout):
    """Mirror of MXFP8QuantConfig::to_key (quantize_mxfp8_cutedsl.cuh): the name the CuTeDSL backend
    registers its compiled kernel under for this config.
    Used to check if the CuTeDSL implementation is registered
    """
    with_dbias = method in ("CAST_DBIAS", "CAST_DBIAS_DACT")
    with_dact = method in ("CAST_DACT", "CAST_DBIAS_DACT")
    with_act = method == "CAST_ACT"
    desc = "none"
    if with_act:
        desc = "gelu"
    elif with_dact:
        desc = "dgelu"
    # MXFP8 never asks TE/common for an amax, and JAX quantize emits scales in the linear
    # (non-swizzled) layout -- the GEMM swizzle happens later, in JAX (see gemm.swizzled_scale).
    # trailing False is use_2d_quantization; JAX never requests 2D block scaling
    flags = (True, q_layout.has_colwise, False, False, with_dbias, with_dact, with_act, False)
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


def extract_quantized_output(out, dbias):
    """Pull the values to compare between backends onto the host.

    Materializing here is what makes the backend toggle safe: JAX dispatch is
    asynchronous, so the FFI handler that reads the toggle may not have run yet when the
    Python call returns.

    ScaledTensor1x carries scale_inv already trimmed to the unpadded shape (see its
    __post_init__), so there is no uninitialized scale padding to exclude.
    """
    tensors = [out] if isinstance(out, ScaledTensor1x) else [out.rowwise_tensor, out.colwise_tensor]
    parts = {}
    for t in tensors:
        name = "colwise" if t.is_colwise else "rowwise"
        parts[f"{name} data"] = np.asarray(t.data.view(jnp.uint8))
        parts[f"{name} scales"] = np.asarray(t.scale_inv.view(jnp.uint8))
    return parts, None if dbias is None else np.asarray(dbias)


def run_test_case(method, shape, q_layout, in_dtype, fp8_dtype):
    """Assert the CuTeDSL and CUDA backends produce bit-identical outputs for the
    same input and config.
    """
    M, N = shape
    x, act_input = generate_inputs(M, N, in_dtype)

    set_cutedsl_backend(False)
    cuda_output, dbias_cuda = extract_quantized_output(
        *run_quantize(method, x, act_input, q_layout, fp8_dtype)
    )

    set_cutedsl_backend(True)
    try:
        cutedsl_output, dbias_cutedsl = extract_quantized_output(
            *run_quantize(method, x, act_input, q_layout, fp8_dtype)
        )
    finally:
        set_cutedsl_backend(False)

    # Guard against a silent CUDA fallback: every config in the matrix is one the
    # CuTeDSL backend supports, so its kernel must have been registered under the
    # config key. If not, the backend rejected or missed the config and the
    # comparison above was CUDA vs CUDA.
    key = get_cfg_key(method, in_dtype, fp8_dtype, q_layout)
    assert tvm_ffi.get_global_func(key, allow_missing=True) is not None, (
        f"CuTeDSL kernel not registered for {key}; the CuTeDSL backend fell back "
        "to CUDA and this case compared CUDA against itself"
    )

    tag = (
        f"{method}/{get_layout_id(q_layout)}/{M}x{N}/"
        f"{DTYPE_TO_STR[in_dtype]}/{FP8_TO_STR[fp8_dtype]}"
    )
    for name, cuda_bytes in cuda_output.items():
        assert np.array_equal(
            cutedsl_output[name], cuda_bytes
        ), f"{tag}: {name} differ between backends"
    if dbias_cuda is not None:
        # CuTeDSL kernel does dbias reduction in a slightly different order than the CUDA kernel,
        # due to the non-associativity of floating-point addition, this will not be bit-identical.
        assert_allclose(dbias_cutedsl, dbias_cuda, err_msg=f"{tag}: dbias differs between backends")


# Test cases with only cast kernels (mirrors C++ test's OperatorTest_FusedCastMXFP8_CastOnly).
@pytest.mark.parametrize("shape", MATRIX_SIZES, ids=get_shape_id)
@pytest.mark.parametrize("q_layout", Q_LAYOUTS, ids=get_layout_id)
@pytest.mark.parametrize("in_dtype", IN_DTYPES, ids=get_dtype_id)
@pytest.mark.parametrize("fp8_dtype", FP8_DTYPES, ids=get_fp8_id)
def test_cast_only(fp8_dtype, in_dtype, q_layout, shape):
    run_test_case("CAST_ONLY", shape, q_layout, in_dtype, fp8_dtype)


# Test cases with varying matrix shapes and quantize layouts
# (OperatorTest_FusedCastMXFP8_Sizes).
@pytest.mark.parametrize("shape", MATRIX_SIZES, ids=get_shape_id)
@pytest.mark.parametrize("q_layout", Q_LAYOUTS, ids=get_layout_id)
@pytest.mark.parametrize("method", METHODS)
def test_sizes(method, q_layout, shape):
    run_test_case(method, shape, q_layout, jnp.bfloat16, jnp.float8_e4m3fn)


# Test cases with varying dtypes (OperatorTest_FusedCastMXFP8_Dtypes).
@pytest.mark.parametrize("in_dtype", IN_DTYPES, ids=get_dtype_id)
@pytest.mark.parametrize("fp8_dtype", FP8_DTYPES, ids=get_fp8_id)
@pytest.mark.parametrize("method", METHODS)
def test_dtypes(method, fp8_dtype, in_dtype):
    run_test_case(method, (256, 384), QuantizeLayout.ROWWISE_COLWISE, in_dtype, fp8_dtype)
