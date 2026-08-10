# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Cross-backend bit-exactness tests for the CuTeDSL NVFP4 quantize-transpose kernel.

Every case quantizes the same input twice, once with the CuTeDSL backend disabled and once with
it enabled, and requires the two results to agree byte for byte. The CUDA C++ kernels are the
reference: `quantize_transpose_nvfp4_tuned_1D.cuh` is what the CuTeDSL kernel reimplements, and
its numerics are in turn pinned against a pure-PyTorch reference by test_nvfp4_quantize_exact.py,
so matching it bit for bit is the whole contract.

Configurations the CuTeDSL kernel does not own are kept in the matrix, but it is worth being
precise about how little they check: both runs execute the same CUDA kernel, so the comparison
only says that turning the backend on is inert. That is not nothing, since the dispatcher walks a
wall of NVTE_CHECKs and builds a config key before it declines and the Python entrypoint compiles
without being allowed to let anything escape, but it says nothing about the CuTeDSL kernel itself.
What pins the kernel's coverage down is `test_kernel_coverage`, which asks the entrypoint
directly; a config moves between the two categories by editing `kernel_implements`.

Which backend ran is therefore never inferred, it is observed: every kernel the CuTeDSL backend
can serve is wrapped in a counting proxy at import (see `_install_call_counters`), and every case
states how many times the dispatcher must call into one. A comparison that silently ran the CUDA
kernel twice would otherwise pass while testing nothing, which is the one failure this file cannot
afford -- and it is not hypothetical: a kernel that fails to compile, a dispatcher check that
rejects the config, and a backend switch that does not take effect all produce exactly that.
Counting catches all three, including the last, since the run that is meant to be the CUDA
reference asserts that no CuTeDSL kernel ran either.

Stochastic rounding is deliberately not compared byte for byte, even though the CuTeDSL kernel
implements it (and, as it happens, currently matches the CUDA kernel bit for bit, having
replicated its Philox seeding, thread arrangement and draw order): which random bits an element
gets follows from the work decomposition, and the CUDA kernel seeds Philox from the chunk
coordinate a CTA starts on without reseeding when it steals a chunk through cluster launch
control, so two implementations, or even two launches, may legitimately disagree without either
being wrong. test_stochastic_rounding checks the property that defines the rounding instead,
and test_nvfp4_sr_quantize.py checks the rounding itself on the CUDA side.

RHT is out of scope: it is a pre-transform with kernels and tests of its own, and it splits the
quantization into per-direction calls, so "which backend ran" stops being one fact per case.
"""

import collections
import ctypes
import itertools
import math
import os
from contextlib import contextmanager
from dataclasses import dataclass, replace

import pytest
import torch

import transformer_engine.pytorch as te
import tvm_ffi
from transformer_engine.common import (
    _get_shared_object_file,
    _load_tvm_ffi_library,
    _register_cutedsl_backends,
)
from transformer_engine.pytorch import NVFP4Quantizer

recipe_available, reason_for_no_recipe = te.is_nvfp4_available(return_reason=True)

# The already-loaded core lib (dlopen refcounts: this returns the same handle, so the call
# mutates the same dispatcher singleton the quantize ops read).
CORE_LIB = ctypes.CDLL(str(_get_shared_object_file("core")))
if not hasattr(CORE_LIB, "nvte_set_cutedsl_quant_backend"):
    raise RuntimeError(
        "libtransformer_engine.so lacks nvte_set_cutedsl_quant_backend -- rebuild the "
        "Transformer Engine core library."
    )

# TE loads tvm-ffi and registers the CuTeDSL entrypoints at import time, but only when
# NVTE_ENABLE_CUTEDSL_QUANT_BACKEND is set. These tests choose the backend through the C++ setter
# instead, so they have to make sure the Python side is wired up regardless of the environment.
# Both calls are idempotent and report whether they succeeded.
backend_available = _load_tvm_ffi_library() and _register_cutedsl_backends()

pytestmark = [
    pytest.mark.skipif(not recipe_available, reason=reason_for_no_recipe),
    pytest.mark.skipif(
        not backend_available, reason="the CuTeDSL backend could not be registered via tvm-ffi"
    ),
]

# Elements sharing one E4M3 block scale.
NVFP4_BLOCK_SIZE = 16
# Flat-dim divisibility the tuned-1D path requires (16B alignment for its TMA descriptors).
TUNED_1D_ALIGNMENT = 32
# Name the CuTeDSL NVFP4 compile-on-demand entrypoint is registered under (kEntrypointName in
# quantize_transpose_nvfp4_cutedsl.cuh).
ENTRYPOINT = "get_nvfp4_quantization_function"
# Filled into the output buffers before every quantization: anything a kernel fails to write then
# shows up as a difference instead of as whatever the caching allocator recycled into the block.
POISON_BYTE = 0xA5
SEED = 1234


@dataclass(frozen=True)
class Case:
    """An NVFP4Quantizer configuration together with the input it is exercised with."""

    id: str
    shape: tuple = (256, 384)
    dtype: torch.dtype = torch.bfloat16
    rowwise: bool = True
    columnwise: bool = False
    row_scaled: bool = False
    two_d: bool = False
    stochastic_rounding: bool = False
    fast_math: bool = False
    use_4over6: bool = False


# The configuration the CuTeDSL kernel currently covers; other cases are derived from it.
ROWWISE = Case(id="rowwise")

get_case_id = lambda case: case.id
get_shape_id = lambda shape: "x".join(str(d) for d in shape)
get_dtype_id = lambda dtype: str(dtype).removeprefix("torch.")


def flat_dims(shape):
    """The 2D dims the kernels see; a rank > 2 input is flattened over its leading dims."""
    return math.prod(shape[:-1]), shape[-1]


def ceil_div(numerator, denominator):
    return -(-numerator // denominator)


def make_quantizer(case):
    return NVFP4Quantizer(
        rowwise=case.rowwise,
        columnwise=case.columnwise,
        with_2d_quantization=case.two_d,
        stochastic_rounding=case.stochastic_rounding,
        row_scaled_nvfp4=case.row_scaled,
        nvfp4_use_4over6=case.use_4over6,
    )


def cutedsl_flags(case):
    """The four booleans an NVFP4QuantConfig is made of, as the dispatcher reads them off the
    quantization config and the output tensor: a transposed output is requested by allocating
    columnwise data, and fast math comes from NVTE_USE_FAST_MATH."""
    return (case.stochastic_rounding, case.fast_math, case.row_scaled, case.columnwise)


def cutedsl_key(flags):
    """Mirror of NVFP4QuantConfig::to_key: the name the compiled kernel for these flags is
    registered under, and the key the C++ side caches its lookup result against."""
    return "cutedsl_nvfp4_" + "_".join("1" if flag else "0" for flag in flags)


def kernel_implements(flags):
    """Whether the CuTeDSL kernel compiles a kernel for these flags. Every feature of the CUDA
    tuned-1D kernel is implemented; the one refusal left is the flag combination that is not a
    real configuration (NVFP4QuantizeConfig raises for it): row-scaled quantization has no
    transposed output."""
    _, _, row_scaled, return_transpose = flags
    return not (row_scaled and return_transpose)


def dispatcher_offers(case):
    """Whether the C++ dispatcher hands this config to the CuTeDSL backend at all. Mirrors the
    NVTE_NVFP4_1D_SCALING branch of dispatch/quantize.cuh and the 2D early-out in
    quantize_transpose_nvfp4_cutedsl.cuh: 4over6 has kernels of its own, the optimized (tuned-1D)
    path is taken only for bf16 with 32-divisible flat dims and an allocated rowwise output, and
    the CuTeDSL entry for 2D scaling declines before it even builds a config key."""
    rows, cols = flat_dims(case.shape)
    return (
        not case.use_4over6
        and not case.two_d
        and case.dtype is torch.bfloat16
        and rows % TUNED_1D_ALIGNMENT == 0
        and cols % TUNED_1D_ALIGNMENT == 0
        and case.rowwise
    )


def expect_cutedsl(case):
    """Whether the CuTeDSL kernel is the one that produces this case's output."""
    return dispatcher_offers(case) and kernel_implements(cutedsl_flags(case))


def set_cutedsl_backend(enabled):
    CORE_LIB.nvte_set_cutedsl_quant_backend(1 if enabled else 0)


# --- Observing which backend ran ---

# How many times the C++ dispatcher has called into a CuTeDSL kernel, per config key.
_cutedsl_calls_by_key = collections.Counter()


def _install_call_counters():
    """Replace every kernel the CuTeDSL backend can serve with a proxy that counts its calls.

    A registered key says a kernel exists, not that the dispatcher reached it, and a key another
    case registered answers for every case after it; counting invocations is what makes "the
    CuTeDSL backend produced this output" a fact rather than an inference. The proxies have to go
    in before the first quantization of the process: the C++ side resolves each key once through
    `TVMFFICentral::lazyload_function` and caches the handle for good, so a proxy installed after
    that is never seen. Compiling the kernels here rather than letting the dispatcher trigger it
    is what makes that possible, and costs one compilation per supported config at import.
    """
    from transformer_engine.common.CuTeDSL.cast.nvfp4.quantize_transpose import (
        get_nvfp4_quantization_function,
    )

    def counting_proxy(key, kernel):
        def proxy(*args):
            _cutedsl_calls_by_key[key] += 1
            return kernel(*args)

        return proxy

    for flags in itertools.product([False, True], repeat=4):
        if not kernel_implements(flags):
            continue
        key = cutedsl_key(flags)
        # A config kernel_implements() claims is served but that does not compile is left alone:
        # test_kernel_coverage is where that contradiction is reported.
        if not get_nvfp4_quantization_function(key, *flags):
            continue
        tvm_ffi.register_global_func(
            key, counting_proxy(key, tvm_ffi.get_global_func(key)), override=True
        )


@contextmanager
def cutedsl_calls(expected, tag, key=None):
    """Require exactly `expected` CuTeDSL kernel invocations inside the block, all of them of the
    kernel compiled for `key` when one is named.

    Which key ran matters as much as whether any did: the flags are what a compiled kernel is, so
    a case served by the kernel for another flag combination -- exact math answering for fast
    math, say -- is testing something other than what it says.
    """
    before = collections.Counter(_cutedsl_calls_by_key)
    yield
    made = _cutedsl_calls_by_key - before
    actual = sum(made.values())
    if actual != expected:
        if expected == 0:
            raise AssertionError(
                f"{tag}: the CuTeDSL backend ran {actual} time(s) ({dict(made)}) where it was "
                "expected to stay out of the way, so this case did not exercise the backend it "
                "names"
            )
        raise AssertionError(
            f"{tag}: the CuTeDSL backend ran {actual} time(s) instead of {expected}, so the CUDA "
            "kernel served a config this case is about. Either the dispatcher declined it, or a "
            "quantization earlier in this pytest session made the C++ side cache the kernel "
            "handle before the counting proxies were installed"
        )
    if key is not None and expected > 0 and set(made) != {key}:
        raise AssertionError(
            f"{tag}: the CuTeDSL kernels that ran were {dict(made)}, not {expected} call(s) of "
            f"{key}, so this case was served by a kernel compiled for another config"
        )


if backend_available and recipe_available:
    _install_call_counters()


@pytest.fixture(scope="module", autouse=True)
def _restore_backend_choice_from_env():
    """Restore the flag that decides the CuTeDSL / CUDA backend choice when this pytest module is done."""
    yield
    flag = os.getenv("NVTE_ENABLE_CUTEDSL_QUANT_BACKEND")
    set_cutedsl_backend(flag is not None and not flag.startswith("0"))


@contextmanager
def fast_math(enabled):
    """NVTE_USE_FAST_MATH is what puts use_fast_math in the quantization config, and TE reads it
    afresh on every quantize call, so setting it here is enough to switch the config."""
    name = "NVTE_USE_FAST_MATH"
    previous = os.environ.get(name)
    os.environ[name] = "1" if enabled else "0"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = previous


def make_input(case, seed=SEED):
    """Uniform noise with a per-scaling-block power-of-two magnitude, so that neighbouring blocks
    land on different E4M3 block scales and the elements at the top of a block saturate the E2M1
    range."""
    rows, cols = flat_dims(case.shape)
    generator = torch.Generator(device="cuda").manual_seed(seed)
    values = torch.empty((rows, cols), dtype=torch.float32, device="cuda").uniform_(
        -6.0, 6.0, generator=generator
    )
    if cols % NVFP4_BLOCK_SIZE == 0:
        blocks = cols // NVFP4_BLOCK_SIZE
        exponents = torch.randint(
            -6, 7, (rows, blocks, 1), device="cuda", generator=generator
        ).float()
        values = (values.view(rows, blocks, NVFP4_BLOCK_SIZE) * torch.exp2(exponents)).view(
            rows, cols
        )
    return values.to(case.dtype).view(case.shape)


def make_input_with_zero_blocks(case, negative=False):
    """Whole scaling blocks of zeros, which is where the block amax is zero and both kernels divide
    by it and clamp the coefficient. Random data never lands there."""
    x = make_input(case)
    x.view(-1, NVFP4_BLOCK_SIZE)[::3].fill_(-0.0 if negative else 0.0)
    return x


def make_output(case):
    """An output tensor of this config's shape, poisoned so that every byte the comparison looks
    at has to have been written by the kernel under test."""
    out = make_quantizer(case).make_empty(case.shape, dtype=case.dtype, device="cuda")
    for buffer in (
        out._rowwise_data,
        out._columnwise_data,
        out._rowwise_scale_inv,
        out._columnwise_scale_inv,
    ):
        if buffer is not None:
            buffer.view(torch.uint8).fill_(POISON_BYTE)
    return out


def quantize_into(out, x, noop_flag=None):
    """Quantize through the same C++ dispatch a plain quantizer call takes, into a preallocated
    output so that its buffers can be poisoned and a cast-noop flag can be passed."""
    # The Philox state for stochastic rounding is drawn from the default CUDA generator when the
    # quantize op runs, so seeding here is what makes the two backends see the same randomness.
    torch.cuda.manual_seed(SEED)
    return out.quantize_(x, noop_flag=noop_flag)


def quantize(case, x):
    return quantize_into(make_output(case), x)


def quantized_parts(out, case, with_amax=True):
    """The bytes of a quantized tensor that both backends must agree on. The scale buffers are
    padded to (round_up(rows, 128), round_up(ceil(cols / 16), 4)) and no kernel writes the
    padding, so only the valid region is taken."""
    rows, cols = flat_dims(case.shape)
    parts = {}
    if case.rowwise:
        parts["rowwise data"] = out._rowwise_data.view(torch.uint8).clone()
        parts["rowwise scales"] = out._rowwise_scale_inv.view(torch.uint8)[
            :rows, : ceil_div(cols, NVFP4_BLOCK_SIZE)
        ].clone()
    if case.columnwise:
        parts["columnwise data"] = out._columnwise_data.view(torch.uint8).clone()
        parts["columnwise scales"] = out._columnwise_scale_inv.view(torch.uint8)[
            :cols, : ceil_div(rows, NVFP4_BLOCK_SIZE)
        ].clone()
    if with_amax:
        # The amax comes from a pass of its own rather than from the quantize kernel, but it is
        # what fixes the global encode scale, so comparing it separates a wrong amax from wrong
        # quantization when a case fails.
        if out._amax_rowwise is not None:
            parts["rowwise amax"] = out._amax_rowwise.clone()
        if out._amax_columnwise is not None:
            parts["columnwise amax"] = out._amax_columnwise.clone()
    return parts


def assert_parts_equal(expected, actual, tag):
    for name, expected_bytes in expected.items():
        assert torch.equal(actual[name], expected_bytes), f"{tag}: {name} does not match"


def relative_dequantize_error(out, x):
    """How far the dequantized output sits from the input, relative to the input's norm."""
    dequantized = out.dequantize(dtype=torch.bfloat16).float()
    reference = x.float()
    error = torch.linalg.vector_norm(dequantized - reference) / torch.linalg.vector_norm(reference)
    return error.item()


def run_case(case, x=None):
    """Assert the CuTeDSL and CUDA backends produce bit-identical output for this config, and
    that the case exercised the backend it was meant to."""
    if x is None:
        x = make_input(case)

    with fast_math(case.fast_math):
        set_cutedsl_backend(False)
        # The reference run has to be CUDA, which is only true if the backend switch took effect.
        with cutedsl_calls(0, f"{case.id} (CUDA reference run)"):
            cuda_parts = quantized_parts(quantize(case, x), case)

        key = cutedsl_key(cutedsl_flags(case))
        set_cutedsl_backend(True)
        try:
            with cutedsl_calls(1 if expect_cutedsl(case) else 0, case.id, key):
                cutedsl_parts = quantized_parts(quantize(case, x), case)
        finally:
            set_cutedsl_backend(False)

    registered = tvm_ffi.get_global_func(key, allow_missing=True) is not None
    if expect_cutedsl(case):
        # Redundant with the call count above, but it separates a kernel that failed to compile
        # from a dispatcher that declined a kernel it has.
        assert registered, (
            f"{case.id}: nothing is registered under {key}, so the CuTeDSL backend fell back to "
            "CUDA and this case compared CUDA against itself"
        )
    elif not kernel_implements(cutedsl_flags(case)):
        assert not registered, (
            f"{case.id}: a kernel is registered under {key}, so the CuTeDSL backend now covers a "
            "config this test expects it to decline; update kernel_implements()"
        )
    # A config the dispatcher never offers (2D scaling, 4over6, non-bf16, flat dims not divisible
    # by 32) shares its key with configs it does offer, so registration says nothing about it.
    # What such a case checks is that turning the backend on leaves the CUDA result alone.

    assert_parts_equal(cuda_parts, cutedsl_parts, case.id)


# --- The kernel's own account of what it supports ---


@pytest.mark.parametrize(
    "flags",
    list(itertools.product([False, True], repeat=4)),
    ids=lambda flags: cutedsl_key(flags).removeprefix("cutedsl_nvfp4_"),
)
def test_kernel_coverage(flags):
    """Ask the compile-on-demand entrypoint directly which flag combinations it compiles a kernel
    for. This is the one place the kernel's coverage is asserted, so a config that starts or stops
    being supported is reported here instead of showing up as a puzzling fallback elsewhere.

    The probe registers under a name of its own so that it cannot make a later case's
    registration check pass by proxy.
    """
    # Imports cleanly only with the CuTeDSL stack present, which the module-level skip guarantees.
    from transformer_engine.common.CuTeDSL.cast.nvfp4.quantize_transpose import (
        get_nvfp4_quantization_function,
    )

    assert tvm_ffi.get_global_func(ENTRYPOINT, allow_missing=True) is not None, (
        f"{ENTRYPOINT} is not in the tvm-ffi registry, so the C++ dispatcher has no way to ask for "
        "a CuTeDSL NVFP4 kernel at all"
    )

    stochastic_rounding, use_fast_math, row_scaled, return_transpose = flags
    supported = get_nvfp4_quantization_function(
        "test_probe_" + cutedsl_key(flags),
        stochastic_rounding,
        use_fast_math,
        row_scaled,
        return_transpose,
    )
    assert bool(supported) == kernel_implements(flags), (
        f"the CuTeDSL NVFP4 backend reports supported={bool(supported)} for {cutedsl_key(flags)}, "
        "which contradicts kernel_implements()"
    )


# --- Shapes, over the config the kernel serves ---

# Flat dims must be multiples of 32 for the tuned-1D path to be chosen at all; within that, the
# interesting axis is how the shape sits relative to the kernel's 64x64 chunk, since everything
# that does not tile exactly is covered by predication.
SHAPES = [
    (32, 32),  # one partial chunk, in both dimensions
    (32, 1024),  # a single row of chunks, partial
    (1024, 32),  # a single column of chunks, partial
    (96, 160),  # partial chunks along both edges
    (128, 128),
    (256, 1024),
    (512, 512),
    (8, 32, 1024),  # rank > 2, flattened to (256, 1024)
    (8192, 7168),  # a shape from a real model
]


@pytest.mark.parametrize("shape", SHAPES, ids=get_shape_id)
@pytest.mark.parametrize("use_fast_math", [False, True], ids=["exact", "fastmath"])
def test_shapes(shape, use_fast_math):
    variant = "fastmath" if use_fast_math else "exact"
    case = replace(
        ROWWISE,
        id=f"{get_shape_id(shape)}-{variant}",
        shape=shape,
        fast_math=use_fast_math,
    )
    assert expect_cutedsl(case)
    run_case(case)


# Flat dims that are not multiples of 32 leave the optimized path entirely, so they exercise the
# dispatcher's other fallback rather than the CuTeDSL kernel.
@pytest.mark.parametrize("shape", [(48, 48), (16, 1024), (256, 400)], ids=get_shape_id)
def test_unaligned_shapes_fall_back(shape):
    case = replace(ROWWISE, id=f"unaligned-{get_shape_id(shape)}", shape=shape)
    assert not expect_cutedsl(case)
    run_case(case)


# --- Input dtypes ---


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32, torch.float16], ids=get_dtype_id)
def test_input_dtypes(dtype):
    """Only bf16 reaches the CuTeDSL kernel; fp32 and fp16 must take the CUDA fallback and be
    left untouched by the backend choice."""
    case = replace(ROWWISE, id=f"dtype-{get_dtype_id(dtype)}", dtype=dtype)
    assert expect_cutedsl(case) == (dtype is torch.bfloat16)
    run_case(case)


# --- Quantizer variations ---

# The NVFP4 quantize configurations reachable from the PyTorch API whose output is deterministic,
# minus RHT and amax reduction. The tuned-1D configs (plain, fast-math, columnwise, row-scaled)
# are served by the CuTeDSL kernel; the 2D and 4over6 ones check that enabling the backend is
# inert for a config it is never offered. Stochastic rounding is not here, since its bytes are
# not a cross-backend contract.
VARIANT_CASES = [
    ROWWISE,
    replace(ROWWISE, id="rowwise-fastmath", fast_math=True),
    replace(ROWWISE, id="rowwise-columnwise", columnwise=True),
    replace(ROWWISE, id="rowwise-columnwise-fastmath", columnwise=True, fast_math=True),
    replace(ROWWISE, id="columnwise-only", rowwise=False, columnwise=True),
    replace(ROWWISE, id="row-scaled", row_scaled=True),
    replace(ROWWISE, id="row-scaled-fastmath", row_scaled=True, fast_math=True),
    replace(ROWWISE, id="2d-quantization", columnwise=True, two_d=True),
    replace(
        ROWWISE,
        id="2d-quantization-columnwise-only",
        rowwise=False,
        columnwise=True,
        two_d=True,
    ),
    replace(ROWWISE, id="4over6", use_4over6=True),
    replace(ROWWISE, id="4over6-columnwise", use_4over6=True, columnwise=True),
]


@pytest.mark.parametrize("case", VARIANT_CASES, ids=get_case_id)
def test_config_variants(case):
    run_case(case)


# Larger shapes for the variants that do produce a transposed output, since a transpose crosses
# chunk boundaries in a way a single chunk cannot show.
@pytest.mark.parametrize("shape", [(64, 64), (256, 1024), (512, 512)], ids=get_shape_id)
def test_transpose_shapes(shape):
    case = replace(ROWWISE, id=f"transpose-{get_shape_id(shape)}", shape=shape, columnwise=True)
    run_case(case)


# --- Stochastic rounding, which is not a byte-for-byte contract ---


def nibble_counts(data_bytes, probe_mask):
    """Count E2M1 nibbles over the probe positions of a packed FP4 tensor."""
    lo = data_bytes & 0xF
    hi = data_bytes >> 4
    nibbles = torch.stack([lo, hi], dim=-1).flatten()  # element order: (byte, low-then-high nibble)
    picked = nibbles[probe_mask.flatten()]
    return torch.bincount(picked.long(), minlength=16), picked.numel()


@pytest.mark.parametrize("columnwise", [False, True], ids=["rowwise", "rowwise-columnwise"])
def test_stochastic_rounding(columnwise):
    """The CuTeDSL kernel serves stochastic rounding, but per the module docstring its bytes are
    deliberately not compared against CUDA's, even though the current implementation happens to
    match it byte for byte (same Philox seeding, thread arrangement and draw order): that
    equality would not survive either kernel legitimately reorganizing its work. What is checked
    instead is the property that defines stochastic rounding. The input is built so every scaling
    block carries one exact 6.0 (making the encode coefficient exactly 1.0 in both directions)
    and probe values of 2.75, which sit at 3/4 of the way from 2 to 3 on the E2M1 lattice, so a
    correct rounder must send them to 3 with probability 0.75, and to 2 otherwise. With ~200k
    probes the tolerance below is dozens of standard deviations, yet fails a round-to-nearest
    (would give 1.0) or wrongly-seeded implementation outright. Determinism for a fixed Philox
    seed is also asserted, since TE draws the seed from the framework RNG.
    """
    case = replace(
        ROWWISE,
        id="stochastic-rounding" + ("-columnwise" if columnwise else ""),
        shape=(256, 1024),
        stochastic_rounding=True,
        columnwise=columnwise,
    )
    assert expect_cutedsl(case)
    rows, cols = flat_dims(case.shape)

    # 6.0 on every 16th row and column anchors every scaling block of both directions; the rest
    # are probes. The global amax 6.0 makes the global encode scale 448/6 . 6/448-exact, the
    # anchored block amaxes make every block decode scale exactly 448 (an E4M3 lattice point),
    # and the encode coefficient works out to exactly 1.0, so probes arrive at the converter
    # as exactly 2.75.
    x = torch.full((rows, cols), 2.75, dtype=torch.bfloat16, device="cuda")
    x[::NVFP4_BLOCK_SIZE, :] = 6.0
    x[:, ::NVFP4_BLOCK_SIZE] = 6.0
    probe = torch.ones(rows, cols, dtype=torch.bool, device="cuda")
    probe[::NVFP4_BLOCK_SIZE, :] = False
    probe[:, ::NVFP4_BLOCK_SIZE] = False

    key = cutedsl_key(cutedsl_flags(case))
    set_cutedsl_backend(True)
    try:
        with cutedsl_calls(2, case.id, key):
            out = quantize(case, x)
            out_again = quantize(case, x)
    finally:
        set_cutedsl_backend(False)

    assert_parts_equal(
        quantized_parts(out, case),
        quantized_parts(out_again, case),
        f"{case.id} (same seed must give the same bytes)",
    )

    checked = {"rowwise data": probe}
    if columnwise:
        checked["columnwise data"] = probe.T.contiguous()
    for name, mask in checked.items():
        counts, total = nibble_counts(quantized_parts(out, case)[name].view(torch.uint8), mask)
        # E2M1 codes: 4 -> 2.0, 5 -> 3.0. Everything else means the scales are wrong.
        assert counts[4] + counts[5] == total, (
            f"{case.id}: {name} probes landed outside {{2.0, 3.0}}:"
            f" {dict(enumerate(counts.tolist()))}"
        )
        up_fraction = counts[5].item() / total
        assert (
            abs(up_fraction - 0.75) < 0.02
        ), f"{case.id}: {name} rounded 2.75 up with frequency {up_fraction:.4f}, expected 0.75"

    error = relative_dequantize_error(out, x)
    assert (
        error < 0.3
    ), f"{case.id}: relative error {error:.4f} is too large for an NVFP4 round trip"


# --- Zeros, where the block amax and the global encode scale degenerate ---


@pytest.mark.parametrize("use_fast_math", [False, True], ids=["exact", "fastmath"])
@pytest.mark.parametrize("kind", ["zero-blocks", "negative-zero-blocks", "all-zero"])
def test_zero_blocks(kind, use_fast_math):
    """A block of zeros makes the block amax zero, which both kernels feed to a reciprocal and
    clamp; an all-zero tensor does the same to the global encode scale.

    Negative zero is the input that decides how the scaling multiply is spelled. The CUDA kernels
    scale with `mul.f32x2` for an f32 coefficient and with `fma.rn.f32.bf16 v, v_h, coeff, zero`
    for a bf16 one, and IEEE addition of that +0 addend turns a -0 product into +0 where a plain
    multiply keeps the sign. E2M1 has a signed zero, so the two encode a -0 element as 0x0 and
    0x8; the kernel uses the same instruction as CUDA on each path, which is what makes this case
    pass for fast math as well.
    """
    case = replace(
        ROWWISE,
        id=f"{kind}-{'fastmath' if use_fast_math else 'exact'}",
        shape=(256, 1024),
        fast_math=use_fast_math,
    )
    if kind == "all-zero":
        x = torch.zeros(case.shape, dtype=case.dtype, device="cuda")
    else:
        x = make_input_with_zero_blocks(case, negative=kind == "negative-zero-blocks")
    run_case(case, x=x)


# --- The cast-noop flag ---


@pytest.mark.parametrize("flag_is_set", [True, False], ids=["noop-set", "noop-clear"])
def test_cast_noop_flag(flag_is_set):
    """The CuTeDSL kernel dereferences the cast-noop flag on device and must leave the output
    exactly as it found it when the flag reads 1.0, and quantize as usual when it reads 0.0."""
    case = replace(ROWWISE, id=f"noop-{'set' if flag_is_set else 'clear'}", shape=(256, 1024))
    assert expect_cutedsl(case)

    x = make_input(case)
    previous_x = make_input(case, seed=SEED + 1)
    flag = torch.full((1,), 1.0 if flag_is_set else 0.0, dtype=torch.float32, device="cuda")

    set_cutedsl_backend(True)
    try:
        # Every one of the three quantizations has to be the CuTeDSL kernel's, the suppressed one
        # included: a kernel that is not launched at all also leaves the output untouched.
        with cutedsl_calls(3, case.id, cutedsl_key(cutedsl_flags(case))):
            # Quantizing x with the flag clear is what the flag has to suppress, and the state the
            # output is left in beforehand is the quantization of a different tensor.
            quantized_x = quantized_parts(quantize(case, x), case, with_amax=False)

            out = make_output(case)
            quantize_into(out, previous_x)
            before = quantized_parts(out, case, with_amax=False)
            quantize_into(out, x, noop_flag=flag)
            after = quantized_parts(out, case, with_amax=False)
    finally:
        set_cutedsl_backend(False)

    if flag_is_set:
        # Both halves matter: nothing changed, and something would have changed.
        assert_parts_equal(before, after, case.id)
        assert not torch.equal(
            before["rowwise data"], quantized_x["rowwise data"]
        ), f"{case.id}: the two inputs quantize to the same bytes, so the flag suppressed nothing"
    else:
        assert_parts_equal(quantized_x, after, case.id)


# --- A numerical floor, independent of the CUDA kernels ---


@pytest.mark.parametrize("use_fast_math", [False, True], ids=["exact", "fastmath"])
def test_dequantized_output_tracks_input(use_fast_math):
    """Dequantizing the CuTeDSL output has to land near the input. Bit-exactness against the CUDA
    kernel is the real contract, but it is a comparison between two implementations of the same
    idea; this is a floor that does not depend on either of them being right.
    """
    case = replace(ROWWISE, id="dequantize", shape=(256, 1024), fast_math=use_fast_math)
    assert expect_cutedsl(case)

    x = make_input(case)
    with fast_math(use_fast_math):
        set_cutedsl_backend(True)
        try:
            with cutedsl_calls(1, case.id, cutedsl_key(cutedsl_flags(case))):
                out = quantize(case, x)
        finally:
            set_cutedsl_backend(False)

    # NVFP4 keeps about two mantissa bits per element, so a few percent of relative error is
    # expected; an order of magnitude more than that means the output is not a quantization of x.
    error = relative_dequantize_error(out, x)
    assert error < 0.2, f"relative error {error:.4f} is too large for an NVFP4 round trip"
