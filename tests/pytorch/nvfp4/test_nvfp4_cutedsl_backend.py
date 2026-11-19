# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Cross-backend bit-exactness tests for the CuTeDSL NVFP4 quantize kernels, driven from pytorch.

Each case quantizes the same input with the CUDA backend and with the CuTeDSL backend and requires
identical bytes. The CuTeDSL backend picks one of three kernels per call: the register-resident
(g2r) rowwise or rowwise + columnwise kernel when the input is eligible, else the general
TMA-staged (g2s) kernel. Only configs the CuTeDSL backend serves are tested.

Stochastic rounding is checked statistically rather than byte for byte, since the g2r kernels draw
Philox per output block and the CUDA kernel per thread.

A quantizer that optimizes for GEMM gets GEMM-swizzled scales from the CuTeDSL kernels themselves
on 128-aligned shapes, and from the swizzle after the CUDA kernel otherwise; both give the same
bytes.
"""

# pylint: disable=wrong-import-position

import ctypes
import os
import re

import pytest
import torch

tvm_ffi = pytest.importorskip("tvm_ffi")
pytest.importorskip("cutlass")

import transformer_engine.pytorch as te

from transformer_engine.common import _get_shared_object_file
from transformer_engine.common.CuTeDSL.utils import device_compute_capability
from transformer_engine.pytorch import NVFP4Quantizer
from transformer_engine.pytorch.cpp_extensions import general_gemm

recipe_available, reason_for_no_recipe = te.is_nvfp4_available(return_reason=True)

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

NVFP4_BLOCK_SIZE = 16
SEED = 1234

# The CuTeDSL backend takes BF16 input whose flattened dims are multiples of 32.
MATRIX_SIZES = [
    (32, 32),
    (32, 1024),
    (1024, 32),
    (96, 160),
    (128, 128),
    (256, 1024),
    (512, 512),
    (8, 32, 1024),  # rank > 2, flattened to (256, 1024)
    (8192, 7168),
]
# (rowwise, columnwise). The CuTeDSL backend serves row-scaled NVFP4 rowwise only.
DIRECTIONS = [(True, False), (True, True)]

get_shape_id = lambda s: "x".join(str(d) for d in s)
get_direction_id = lambda d: "rowwise-columnwise" if d[1] else "rowwise"
get_fast_math_id = lambda f: "fastmath" if f else "exact"
get_optimize_for_gemm_id = lambda o: "swizzled" if o else "compact"


def set_cutedsl_backend(enabled):
    """Switch the backend and NVTE_ENABLE_CUTEDSL_BACKEND, which quantizers read to choose
    between GEMM-swizzled scales from the kernel and a swizzle after it."""
    os.environ["NVTE_ENABLE_CUTEDSL_BACKEND"] = "1" if enabled else "0"
    CORE_LIB.nvte_set_cutedsl_backend(1 if enabled else 0)


@pytest.fixture(scope="module", autouse=True)
def _restore_backend_choice_from_env():
    """Restore the flag that decides the CuTeDSL / CUDA backend choice when this pytest module is done."""
    flag = os.getenv("NVTE_ENABLE_CUTEDSL_BACKEND")
    yield
    set_cutedsl_backend(flag is not None and not flag.startswith("0"))
    if flag is None:
        del os.environ["NVTE_ENABLE_CUTEDSL_BACKEND"]
    else:
        os.environ["NVTE_ENABLE_CUTEDSL_BACKEND"] = flag


def flat_dims(shape):
    return int(torch.tensor(shape[:-1]).prod()), shape[-1]


def generate_input(shape, seed=SEED):
    """Uniform noise scaled by a random power of two per 16-element block, so neighbouring blocks
    get different E4M3 scales."""
    rows, cols = flat_dims(shape)
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.empty(rows, cols, device="cuda").uniform_(-6.0, 6.0, generator=g)
    exponents = torch.randint(
        -6, 7, (rows, cols // NVFP4_BLOCK_SIZE, 1), device="cuda", generator=g
    )
    x = (x.view(rows, -1, NVFP4_BLOCK_SIZE) * torch.exp2(exponents.float())).view(rows, cols)
    return x.to(torch.bfloat16).view(shape)


def make_quantizer(
    rowwise, columnwise, row_scaled=False, stochastic_rounding=False, optimize_for_gemm=False
):
    quantizer = NVFP4Quantizer(
        rowwise=rowwise,
        columnwise=columnwise,
        row_scaled_nvfp4=row_scaled,
        stochastic_rounding=stochastic_rounding,
    )
    quantizer.optimize_for_gemm = optimize_for_gemm
    return quantizer


def kernel_writes_swizzled_scales(shape, row_scaled=False, stochastic_rounding=False):
    """Whether a quantizer that optimizes for GEMM gets its GEMM-swizzled scales from the CuTeDSL
    kernel rather than the later swizzle. Mirror of nvfp4_gemm_swizzled_scales_alignment
    (cutedsl_gemm_swizzled_scales.h) for the rowwise E4M3 quantizers with an amax used here."""
    major, minor = device_compute_capability()
    rows, cols = flat_dims(shape)
    return (
        100 <= major * 10 + minor <= 110
        and not (stochastic_rounding and major != 10)
        and not row_scaled
        and rows % 128 == 0
        and cols % 128 == 0
    )


def quantize(quantizer, x, noop_flag=None, out=None):
    # Stochastic rounding draws its Philox seed from the default CUDA generator.
    torch.cuda.manual_seed(SEED)
    if out is None:
        out = quantizer.make_empty(x.shape, dtype=x.dtype, device="cuda")
    return out.quantize_(x, noop_flag=noop_flag)


def get_cfg_key(stochastic_rounding, fast_math, row_scaled, columnwise, swizzled=False):
    """Mirror of NVFP4QuantConfig::to_key (quantize_transpose_nvfp4_cutedsl.cuh)."""
    major, minor = device_compute_capability()
    flags = (stochastic_rounding, fast_math, row_scaled, columnwise, swizzled)
    return f"cutedsl_nvfp4_sm{major * 10 + minor}_" + "_".join("1" if f else "0" for f in flags)


def get_amax_key(columnwise_amax):
    """Mirror of NVFP4GlobalAmaxConfig::to_key (compute_amax_nvfp4_cutedsl.cuh)."""
    major, minor = device_compute_capability()
    return f"cutedsl_nvfp4_amax_sm{major * 10 + minor}_{int(columnwise_amax)}"


def extract_quantized_output(out, shape, rowwise, columnwise):
    """The bytes both backends must agree on. Scale padding is not written, so only the valid
    region of the scales is taken."""
    rows, cols = flat_dims(shape)
    parts = {}
    if rowwise:
        parts["rowwise data"] = out._rowwise_data.view(torch.uint8).clone()
        parts["rowwise scales"] = out._rowwise_scale_inv.view(torch.uint8)[
            :rows, : cols // NVFP4_BLOCK_SIZE
        ].clone()
        parts["rowwise amax"] = out._amax_rowwise.clone()
    if columnwise:
        parts["columnwise data"] = out._columnwise_data.view(torch.uint8).clone()
        parts["columnwise scales"] = out._columnwise_scale_inv.view(torch.uint8)[
            :cols, : rows // NVFP4_BLOCK_SIZE
        ].clone()
        parts["columnwise amax"] = out._amax_columnwise.clone()
    return parts


def run_test_case(shape, rowwise, columnwise, row_scaled=False, x=None, optimize_for_gemm=False):
    """Assert the CuTeDSL and CUDA backends produce bit-identical outputs for the same input."""
    if x is None:
        x = generate_input(shape)
    fast_math = os.environ.get("NVTE_USE_FAST_MATH", "0") == "1"
    quantizer = make_quantizer(rowwise, columnwise, row_scaled, optimize_for_gemm=optimize_for_gemm)

    set_cutedsl_backend(False)
    out_cuda = quantize(quantizer, x)
    cuda_output = extract_quantized_output(out_cuda, shape, rowwise, columnwise)

    set_cutedsl_backend(True)
    try:
        out_cutedsl = quantize(quantizer, x)
        cutedsl_output = extract_quantized_output(out_cutedsl, shape, rowwise, columnwise)
    finally:
        set_cutedsl_backend(False)

    assert out_cutedsl._with_gemm_swizzled_scales == optimize_for_gemm
    # Guard against a silent CUDA fallback, which would compare CUDA against itself.
    swizzled = optimize_for_gemm and kernel_writes_swizzled_scales(shape, row_scaled)
    key = get_cfg_key(False, fast_math, row_scaled, columnwise, swizzled)
    assert (
        tvm_ffi.get_global_func(key, allow_missing=True) is not None
    ), f"CuTeDSL kernel not registered for {key}; the backend fell back to CUDA"
    # quantize_ computes the global amax with nvte_compute_amax, which the CuTeDSL backend serves
    # outside row-scaled NVFP4.
    if not row_scaled:
        amax_key = get_amax_key(columnwise)
        assert (
            tvm_ffi.get_global_func(amax_key, allow_missing=True) is not None
        ), f"CuTeDSL amax kernel not registered for {amax_key}; the amax fell back to CUDA"

    tag = f"{get_shape_id(shape)}/{get_direction_id((rowwise, columnwise))}"
    for name, cuda_bytes in cuda_output.items():
        assert torch.equal(cutedsl_output[name], cuda_bytes), f"{tag}: {name} differ"


@pytest.mark.parametrize("shape", MATRIX_SIZES, ids=get_shape_id)
@pytest.mark.parametrize("direction", DIRECTIONS, ids=get_direction_id)
@pytest.mark.parametrize("fast_math", [False, True], ids=get_fast_math_id)
@pytest.mark.parametrize("optimize_for_gemm", [False, True], ids=get_optimize_for_gemm_id)
def test_cast(optimize_for_gemm, fast_math, direction, shape, monkeypatch):
    monkeypatch.setenv("NVTE_USE_FAST_MATH", "1" if fast_math else "0")
    run_test_case(shape, *direction, optimize_for_gemm=optimize_for_gemm)


@pytest.mark.parametrize("fast_math", [False, True], ids=get_fast_math_id)
def test_row_scaled(fast_math, monkeypatch):
    monkeypatch.setenv("NVTE_USE_FAST_MATH", "1" if fast_math else "0")
    run_test_case((256, 1024), True, False, row_scaled=True)


def test_row_scaled_columnwise_falls_back_to_cuda(monkeypatch):
    """Row-scaled NVFP4 with a columnwise output is not implemented by the CuTeDSL kernels, so the
    backend must decline it and let the CUDA kernel produce the same bytes."""
    monkeypatch.setenv("NVTE_USE_FAST_MATH", "0")
    shape = (256, 1024)
    x = generate_input(shape)
    quantizer = make_quantizer(True, True, row_scaled=True)

    set_cutedsl_backend(False)
    cuda_output = extract_quantized_output(quantize(quantizer, x), shape, True, True)

    set_cutedsl_backend(True)
    try:
        fallback_output = extract_quantized_output(quantize(quantizer, x), shape, True, True)
    finally:
        set_cutedsl_backend(False)

    key = get_cfg_key(False, False, True, True)
    assert tvm_ffi.get_global_func(key, allow_missing=True) is None, f"{key} was compiled"
    for name, cuda_bytes in cuda_output.items():
        assert torch.equal(fallback_output[name], cuda_bytes), f"{name} differ"


@pytest.mark.skipif(
    device_compute_capability()[0] != 10, reason="CuTeDSL stochastic rounding needs SM 10.x"
)
@pytest.mark.parametrize("shape", [(512, 1024), (512, 1152), (8192, 7168)], ids=get_shape_id)
@pytest.mark.parametrize("direction", DIRECTIONS, ids=get_direction_id)
def test_stochastic_rounding_swizzled_scales(direction, shape):
    """Under stochastic rounding the scales do not depend on the draws, so the GEMM-swizzled
    scales the CuTeDSL kernels write equal those of the CUDA backend and its swizzle. The draws
    do not depend on the scale layout, so the data equal those of the CuTeDSL kernels with
    compact scales."""
    rowwise, columnwise = direction
    x = generate_input(shape)
    quantizer = make_quantizer(
        rowwise, columnwise, stochastic_rounding=True, optimize_for_gemm=True
    )
    compact_quantizer = make_quantizer(rowwise, columnwise, stochastic_rounding=True)

    set_cutedsl_backend(False)
    cuda_output = extract_quantized_output(quantize(quantizer, x), shape, rowwise, columnwise)
    set_cutedsl_backend(True)
    try:
        out = quantize(quantizer, x)
        compact_output = extract_quantized_output(
            quantize(compact_quantizer, x), shape, rowwise, columnwise
        )
    finally:
        set_cutedsl_backend(False)
    cutedsl_output = extract_quantized_output(out, shape, rowwise, columnwise)

    assert out._with_gemm_swizzled_scales
    swizzled = kernel_writes_swizzled_scales(shape, stochastic_rounding=True)
    key = get_cfg_key(True, False, False, columnwise, swizzled)
    assert tvm_ffi.get_global_func(key, allow_missing=True) is not None, key
    for name, cuda_bytes in cuda_output.items():
        if "data" not in name:
            assert torch.equal(cutedsl_output[name], cuda_bytes), f"{name} differ"
        else:
            assert torch.equal(cutedsl_output[name], compact_output[name]), f"{name} differ"


@pytest.mark.parametrize("layout", ["TN", "NN", "NT"])
def test_gemm_of_swizzled_scales(layout):
    """A GEMM of operands whose scales the CuTeDSL kernels swizzled equals the GEMM of the CUDA
    backend's compact operands, which the GEMM swizzles itself. The operands are 128-aligned, so
    every layout reads scales the CuTeDSL kernel wrote."""
    M, N, K = 384, 256, 512
    a_shape = (N, K) if layout[0] == "T" else (K, N)
    b_shape = (K, M) if layout[1] == "T" else (M, K)
    if not kernel_writes_swizzled_scales(a_shape):
        pytest.skip("the CuTeDSL kernels write GEMM-swizzled scales on SM 100-110 only")
    a, b = generate_input(a_shape), generate_input(b_shape, seed=SEED + 1)

    set_cutedsl_backend(True)
    try:
        quantizer = make_quantizer(True, True, optimize_for_gemm=True)
        a_swizzled, b_swizzled = quantize(quantizer, a), quantize(quantizer, b)
    finally:
        set_cutedsl_backend(False)
    quantizer = make_quantizer(True, True)
    a_compact, b_compact = quantize(quantizer, a), quantize(quantizer, b)
    assert a_swizzled._with_gemm_swizzled_scales and b_swizzled._with_gemm_swizzled_scales
    assert not a_compact._with_gemm_swizzled_scales and not b_compact._with_gemm_swizzled_scales

    expected = general_gemm(a_compact, b_compact, out_dtype=torch.float32, layout=layout)[0]
    actual = general_gemm(a_swizzled, b_swizzled, out_dtype=torch.float32, layout=layout)[0]
    assert torch.equal(actual, expected)


def test_swizzled_scales_skip_the_separate_swizzle():
    """On a 128-aligned shape the CuTeDSL kernels write the GEMM-swizzled scales themselves, so the
    call launches a CuTeDSL quantize kernel and no swizzle kernel, unlike the CUDA backend's call,
    and gives the same bytes. The vector-blockwise kernel writes them too, so the CuTeDSL kernel
    is asserted to have run."""
    shape = (256, 1024)
    if not kernel_writes_swizzled_scales(shape):
        pytest.skip("the CuTeDSL kernels write GEMM-swizzled scales on SM 100-110 only")
    x = generate_input(shape)
    quantizer = make_quantizer(True, True, optimize_for_gemm=True)

    def profiled_quantize(enabled):
        set_cutedsl_backend(enabled)
        quantize(quantizer, x)  # compiles the kernels outside the profile
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
            out = quantize(quantizer, x)
            torch.cuda.synchronize()
        names = [e.name for e in prof.events()]
        swizzles = [n for n in names if re.search(r"swizzle_\w*kernel", n)]
        quantizes = [n for n in names if re.search(r"NVFP4Quantize\w*Kernel", n)]
        return extract_quantized_output(out, shape, True, True), swizzles, quantizes

    try:
        cuda_output, cuda_swizzles, cuda_quantizes = profiled_quantize(False)
        cutedsl_output, cutedsl_swizzles, cutedsl_quantizes = profiled_quantize(True)
    finally:
        set_cutedsl_backend(False)

    assert cuda_swizzles, "the CUDA backend's call launched no swizzle kernel"
    assert not cutedsl_swizzles, f"the CuTeDSL backend's call launched {cutedsl_swizzles}"
    assert cutedsl_quantizes, "the CuTeDSL backend's call launched no CuTeDSL quantize kernel"
    assert not cuda_quantizes, f"the CUDA backend's call launched {cuda_quantizes}"
    for name, cuda_bytes in cuda_output.items():
        assert torch.equal(cutedsl_output[name], cuda_bytes), f"{name} differ"


@pytest.mark.parametrize("direction", DIRECTIONS, ids=get_direction_id)
@pytest.mark.parametrize(
    "kind", ["zero-blocks", "negative-zero-blocks", "all-zero", "all-negative-zero"]
)
def test_zero_blocks(kind, direction):
    """Zero block amaxes (and, for all-zero, a zero global amax) hit the clamped reciprocal;
    negative zero checks the sign of zero survives the scaling multiply as in CUDA, and that the
    global amax of an all-negative-zero input is +0."""
    shape = (256, 1024)
    if kind == "all-zero":
        x = torch.zeros(shape, dtype=torch.bfloat16, device="cuda")
    elif kind == "all-negative-zero":
        x = torch.full(shape, -0.0, dtype=torch.bfloat16, device="cuda")
    else:
        x = generate_input(shape)
        x.view(-1, NVFP4_BLOCK_SIZE)[::3].fill_(-0.0 if kind == "negative-zero-blocks" else 0.0)
    run_test_case(shape, *direction, x=x)


@pytest.mark.parametrize("direction", DIRECTIONS, ids=get_direction_id)
@pytest.mark.parametrize("shape", [(96, 160), (2080, 1056)], ids=get_shape_id)
@pytest.mark.parametrize("position", ["first", "last"])
def test_amax_at_either_end(position, shape, direction):
    """The global amax the quantization computes reaches the first and the last element of the
    input, also when the input fills only part of the amax kernel's last stripe."""
    x = generate_input(shape)
    x.view(-1)[0 if position == "first" else -1] = 1024.0
    run_test_case(shape, *direction, x=x)


@pytest.mark.parametrize("shape", [(256, 1024), (256, 48)], ids=get_shape_id)
@pytest.mark.parametrize(
    "direction",
    [(True, False), (True, True), (False, True)],
    ids=["rowwise", "rowwise-columnwise", "columnwise"],
)
@pytest.mark.parametrize("kind", ["uniform", "inf", "nan", "subnormal"])
def test_global_amax_matches_cuda_backend(kind, direction, shape):
    """The amaxes nvte_compute_amax writes for an NVFP4 output are bit for bit those of the CUDA
    backend, also for a columnwise-only output, whose quantize kernel is not a CuTeDSL one, and
    for inputs with an infinity, NaNs or only subnormals. The CuTeDSL amax kernel serves shapes
    whose flattened dims are multiples of 32, and TE's amax kernel the others."""
    rowwise, columnwise = direction
    x = generate_input(shape)
    if kind == "inf":
        x[3, 5], x[100, 7] = float("inf"), float("-inf")
    elif kind == "nan":
        x.view(-1)[::97] = float("nan")
    elif kind == "subnormal":
        # Without their exponent bits, the values are bf16 subnormals or zeros of either sign.
        x = (x.view(torch.int16) & ~0x7F80).view(torch.bfloat16)
    quantizer = make_quantizer(rowwise, columnwise)

    def profiled_amaxes(enabled):
        set_cutedsl_backend(enabled)
        try:
            quantize(quantizer, x)  # compiles the kernels outside the profile
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
                out = quantize(quantizer, x)
                torch.cuda.synchronize()
        finally:
            set_cutedsl_backend(False)
        names = [e.name for e in prof.events()]
        launched = {
            "cutedsl": any(re.search(r"amax_kernel\w*NVFP4GlobalAmaxKernel", n) for n in names),
            "cuda": any(re.search(r"\bamax_kernel\b", n) for n in names),
        }
        amaxes = extract_quantized_output(out, shape, rowwise, columnwise)
        return {k: v.view(torch.int32) for k, v in amaxes.items() if "amax" in k}, launched

    cuda_amaxes, cuda_launched = profiled_amaxes(False)
    cutedsl_amaxes, cutedsl_launched = profiled_amaxes(True)

    served = all(d % 32 == 0 for d in flat_dims(shape))
    assert cuda_launched == {"cutedsl": False, "cuda": True}, cuda_launched
    assert cutedsl_launched == {"cutedsl": served, "cuda": not served}, cutedsl_launched
    for name, cuda_bits in cuda_amaxes.items():
        assert torch.equal(cutedsl_amaxes[name], cuda_bits), f"{name} differ"


@pytest.mark.parametrize("direction", DIRECTIONS, ids=get_direction_id)
def test_cuda_graph(direction):
    """A quantization that computes its global amax, captured in a CUDA graph, replays on new
    input to the bytes the CUDA backend gives for that input."""
    rowwise, columnwise = direction
    shape = (256, 1024)
    quantizer = make_quantizer(rowwise, columnwise)
    x = generate_input(shape)

    set_cutedsl_backend(True)
    try:
        out = quantize(quantizer, x)  # compiles the kernels outside the capture
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out.quantize_(x)
        x.copy_(generate_input(shape, seed=SEED + 1))
        graph.replay()
        replayed = extract_quantized_output(out, shape, rowwise, columnwise)
    finally:
        set_cutedsl_backend(False)

    expected = extract_quantized_output(quantize(quantizer, x), shape, rowwise, columnwise)
    for name, cuda_bytes in expected.items():
        assert torch.equal(replayed[name], cuda_bytes), f"{name} differ"


@pytest.mark.parametrize("direction", DIRECTIONS, ids=get_direction_id)
@pytest.mark.parametrize("optimize_for_gemm", [False, True], ids=get_optimize_for_gemm_id)
def test_unaligned_input_uses_g2s(optimize_for_gemm, direction):
    """A 16-byte-aligned input is not eligible for the g2r kernels' 32-byte loads, so the g2s
    kernel serves it."""
    shape = (256, 384)
    rows, cols = shape
    buffer = torch.empty(rows * cols + 8, dtype=torch.bfloat16, device="cuda")
    x = buffer[8:].view(shape)
    x.copy_(generate_input(shape))
    assert x.data_ptr() % 32 == 16
    run_test_case(shape, *direction, x=x, optimize_for_gemm=optimize_for_gemm)


def test_rowwise_g2r_grid_y_overflow_uses_g2s():
    """The rowwise g2r kernel covers 4 rows per CTA, so this shape needs grid.y = 65536 and the
    g2s kernel must serve it."""
    run_test_case((4 * 65536, 32), True, False)


def test_bidimensional_g2r_grid_y_overflow_uses_g2s():
    """The rowwise + columnwise g2r kernel covers 64 rows per CTA, so this shape needs
    grid.y = 65536 and the g2s kernel must serve it."""
    run_test_case((64 * 65536, 32), True, True)


@pytest.mark.parametrize("flag_is_set", [True, False], ids=["noop-set", "noop-clear"])
def test_cast_noop_flag(flag_is_set):
    """With the noop flag set the kernel must leave the output untouched."""
    shape = (256, 1024)
    quantizer = make_quantizer(True, False)
    x = generate_input(shape)
    flag = torch.full((1,), 1.0 if flag_is_set else 0.0, dtype=torch.float32, device="cuda")

    set_cutedsl_backend(True)
    try:
        expected = extract_quantized_output(quantize(quantizer, x), shape, True, False)
        out = quantize(quantizer, generate_input(shape, seed=SEED + 1))
        before = extract_quantized_output(out, shape, True, False)
        quantize(quantizer, x, noop_flag=flag, out=out)
        after = extract_quantized_output(out, shape, True, False)
    finally:
        set_cutedsl_backend(False)

    reference = before if flag_is_set else expected
    for name in ("rowwise data", "rowwise scales", "rowwise amax"):
        assert torch.equal(after[name], reference[name]), f"{name} differ"


@pytest.mark.skipif(
    device_compute_capability()[0] != 10, reason="CuTeDSL stochastic rounding needs SM 10.x"
)
@pytest.mark.parametrize("direction", DIRECTIONS, ids=get_direction_id)
def test_stochastic_rounding(direction):
    """Every 16-element block holds one exact 6.0, which makes the encode coefficient exactly 1.0,
    and the other elements are 2.75: 3/4 of the way from 2 to 3 on the E2M1 grid. Correct
    stochastic rounding sends them to 3 with probability 0.75 and is deterministic for a fixed
    seed."""
    rowwise, columnwise = direction
    shape = (256, 1024)
    x = torch.full(shape, 2.75, dtype=torch.bfloat16, device="cuda")
    x[::NVFP4_BLOCK_SIZE, :] = 6.0
    x[:, ::NVFP4_BLOCK_SIZE] = 6.0
    probe = x == 2.75
    quantizer = make_quantizer(rowwise, columnwise, stochastic_rounding=True)

    set_cutedsl_backend(True)
    try:
        out = extract_quantized_output(quantize(quantizer, x), shape, rowwise, columnwise)
        out_again = extract_quantized_output(quantize(quantizer, x), shape, rowwise, columnwise)
    finally:
        set_cutedsl_backend(False)
    key = get_cfg_key(True, False, False, columnwise)
    assert tvm_ffi.get_global_func(key, allow_missing=True) is not None, key

    masks = {"rowwise data": probe}
    if columnwise:
        masks["columnwise data"] = probe.T.contiguous()
    for name, mask in masks.items():
        assert torch.equal(out[name], out_again[name]), f"{name}: not deterministic"
        data = out[name]
        nibbles = torch.stack([data & 0xF, data >> 4], dim=-1).flatten()[mask.flatten()]
        # E2M1 codes 4 and 5 are 2.0 and 3.0.
        assert bool(((nibbles == 4) | (nibbles == 5)).all()), f"{name}: probe outside {{2, 3}}"
        up_fraction = (nibbles == 5).float().mean().item()
        assert abs(up_fraction - 0.75) < 0.02, f"{name}: rounded up {up_fraction:.4f}, not 0.75"


@pytest.mark.skipif(
    device_compute_capability()[0] != 10, reason="CuTeDSL stochastic rounding needs SM 10.x"
)
@pytest.mark.parametrize("direction", DIRECTIONS, ids=get_direction_id)
def test_stochastic_rounding_cuda_graph(direction):
    """Stochastic rounding captured in a CUDA graph reads the Philox seed and offset when the graph
    is replayed. Two quantizes in one graph must match two eager calls from the same generator
    state, which checks each call's offset within the graph, and the next replay must draw anew.
    The seed has its top bit set, so it must reach the kernel intact both by value (eager) and
    through a pointer (replay)."""
    rowwise, columnwise = direction
    shape = (256, 1024)
    seed = 2**63 + SEED
    x = generate_input(shape)
    quantizer = make_quantizer(rowwise, columnwise, stochastic_rounding=True)
    outs = [quantizer.make_empty(shape, dtype=x.dtype, device="cuda") for _ in range(2)]

    def quantize_all():
        for out in outs:
            out.quantize_(x)

    def outputs():
        return [extract_quantized_output(out, shape, rowwise, columnwise) for out in outs]

    set_cutedsl_backend(True)
    try:
        # Warm up outside capture, so the kernel is compiled before the graph is captured.
        warmup_stream = torch.cuda.Stream()
        warmup_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warmup_stream):
            torch.cuda.manual_seed(seed + 1)
            quantize_all()
            other_seed = outputs()
            torch.cuda.manual_seed(seed)
            quantize_all()
            eager = outputs()
        torch.cuda.current_stream().wait_stream(warmup_stream)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            quantize_all()
        torch.cuda.manual_seed(seed)
        graph.replay()
        replay = outputs()
        graph.replay()
        replay_again = outputs()
    finally:
        set_cutedsl_backend(False)
    key = get_cfg_key(True, False, False, columnwise)
    assert tvm_ffi.get_global_func(key, allow_missing=True) is not None, key

    for i, expected in enumerate(eager):
        for name, value in expected.items():
            assert torch.equal(replay[i][name], value), f"call {i}: {name}: replay differs"
    for name in ["rowwise data", "columnwise data"] if columnwise else ["rowwise data"]:
        assert not torch.equal(eager[1][name], eager[0][name]), f"{name}: calls share draws"
        assert not torch.equal(other_seed[0][name], eager[0][name]), f"{name}: seed is not used"
        assert not torch.equal(replay_again[0][name], replay[0][name]), f"{name}: replays repeat"


def _philox4x32(key, counter, rounds):
    """Plain Philox4x32 (curanddx's philox4x32_native_state)."""
    mask = 0xFFFFFFFF
    k0, k1 = key
    c = list(counter)
    for r in range(rounds):
        if r > 0:
            k0, k1 = (k0 + 0x9E3779B9) & mask, (k1 + 0xBB67AE85) & mask
        p0, p1 = 0xD2511F53 * c[0], 0xCD9E8D57 * c[2]
        c = [(p1 >> 32) ^ c[1] ^ k0, p1 & mask, (p0 >> 32) ^ c[3] ^ k1, p0 & mask]
    return c


def test_block_philox_matches_philox4x32():
    """The g2r kernels' per-block draw is plain Philox4x32 of the counter
    {offset_lo, offset_hi, subsequence_lo, subsequence_hi} with key = seed, where
    subsequence_hi = logical_hi ^ hi(offset_lo * M0) ^ seed_hi (philox_rng.py)."""
    # pylint: disable=import-outside-toplevel
    import cutlass
    from cutlass import cute, Int32, Uint32, Uint64
    from cutlass.cute.runtime import from_dlpack
    from transformer_engine.common.CuTeDSL import philox_rng

    if not philox_rng.BLOCK_PHILOX_FOLD_SUPPORTED:
        pytest.skip("the per-block draw needs at least 3 Philox rounds")

    @cute.kernel
    def draw_kernel(mIn: cute.Tensor, mOut: cute.Tensor):
        t, _, _ = cute.arch.thread_idx()
        word = []
        for i in cutlass.range_constexpr(6):
            word.append(Uint32(mIn[t, i]))
        seed = Uint64(word[0]) | (Uint64(word[1]) << Uint64(32))
        offset = Uint64(word[2]) | (Uint64(word[3]) << Uint64(32))
        fold = philox_rng.make_block_philox_fold(
            philox_rng.make_block_philox(seed, offset), word[4]
        )
        draw = philox_rng.block_philox_draws([word[5]], fold)[0]
        for i in cutlass.range_constexpr(4):
            mOut[t, i] = Int32(draw[i])

    @cute.jit
    def draw(mIn: cute.Tensor, mOut: cute.Tensor):
        draw_kernel(mIn, mOut).launch(grid=[1, 1, 1], block=[mIn.shape[0], 1, 1])

    generator = torch.Generator().manual_seed(SEED)
    words = torch.randint(0, 2**32, (128, 6), dtype=torch.int64, generator=generator)
    words[0] = 0
    words[1] = 2**32 - 1
    draws = torch.zeros((128, 4), dtype=torch.int32, device="cuda")
    draw(from_dlpack(words.to(torch.int32).cuda()), from_dlpack(draws))

    for row, out in zip(words.tolist(), draws.cpu().tolist()):
        seed_lo, seed_hi, offset_lo, offset_hi, sub_lo, logical_hi = row
        sub_hi = logical_hi ^ ((offset_lo * 0xD2511F53) >> 32) ^ seed_hi
        expected = _philox4x32(
            (seed_lo, seed_hi), (offset_lo, offset_hi, sub_lo, sub_hi), philox_rng.NUM_PHILOX_ROUNDS
        )
        assert [w & 0xFFFFFFFF for w in out] == expected, f"draw for {row}"
