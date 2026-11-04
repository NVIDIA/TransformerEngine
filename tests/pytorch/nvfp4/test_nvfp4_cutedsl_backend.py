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
"""

# pylint: disable=wrong-import-position

import ctypes
import os

import pytest
import torch

tvm_ffi = pytest.importorskip("tvm_ffi")
pytest.importorskip("cutlass")

import transformer_engine.pytorch as te

from transformer_engine.common import _get_shared_object_file
from transformer_engine.common.CuTeDSL.utils import device_compute_capability
from transformer_engine.pytorch import NVFP4Quantizer

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
# (rowwise, columnwise). Row-scaled NVFP4 has no columnwise output.
DIRECTIONS = [(True, False), (True, True)]

get_shape_id = lambda s: "x".join(str(d) for d in s)
get_direction_id = lambda d: "rowwise-columnwise" if d[1] else "rowwise"
get_fast_math_id = lambda f: "fastmath" if f else "exact"


def set_cutedsl_backend(enabled):
    CORE_LIB.nvte_set_cutedsl_backend(1 if enabled else 0)


@pytest.fixture(scope="module", autouse=True)
def _restore_backend_choice_from_env():
    """Restore the flag that decides the CuTeDSL / CUDA backend choice when this pytest module is done."""
    yield
    flag = os.getenv("NVTE_ENABLE_CUTEDSL_BACKEND")
    set_cutedsl_backend(flag is not None and not flag.startswith("0"))


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


def make_quantizer(rowwise, columnwise, row_scaled=False, stochastic_rounding=False):
    return NVFP4Quantizer(
        rowwise=rowwise,
        columnwise=columnwise,
        row_scaled_nvfp4=row_scaled,
        stochastic_rounding=stochastic_rounding,
    )


def quantize(quantizer, x, noop_flag=None, out=None):
    # Stochastic rounding draws its Philox seed from the default CUDA generator.
    torch.cuda.manual_seed(SEED)
    if out is None:
        out = quantizer.make_empty(x.shape, dtype=x.dtype, device="cuda")
    return out.quantize_(x, noop_flag=noop_flag)


def get_cfg_key(stochastic_rounding, fast_math, row_scaled, columnwise):
    """Mirror of NVFP4QuantConfig::to_key (quantize_transpose_nvfp4_cutedsl.cuh)."""
    major, minor = device_compute_capability()
    flags = (stochastic_rounding, fast_math, row_scaled, columnwise)
    return f"cutedsl_nvfp4_sm{major * 10 + minor}_" + "_".join("1" if f else "0" for f in flags)


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


def run_test_case(shape, rowwise, columnwise, row_scaled=False, x=None):
    """Assert the CuTeDSL and CUDA backends produce bit-identical outputs for the same input."""
    if x is None:
        x = generate_input(shape)
    fast_math = os.environ.get("NVTE_USE_FAST_MATH", "0") == "1"
    quantizer = make_quantizer(rowwise, columnwise, row_scaled)

    set_cutedsl_backend(False)
    out_cuda = quantize(quantizer, x)
    cuda_output = extract_quantized_output(out_cuda, shape, rowwise, columnwise)

    set_cutedsl_backend(True)
    try:
        out_cutedsl = quantize(quantizer, x)
        cutedsl_output = extract_quantized_output(out_cutedsl, shape, rowwise, columnwise)
    finally:
        set_cutedsl_backend(False)

    # Guard against a silent CUDA fallback, which would compare CUDA against itself.
    key = get_cfg_key(False, fast_math, row_scaled, columnwise)
    assert (
        tvm_ffi.get_global_func(key, allow_missing=True) is not None
    ), f"CuTeDSL kernel not registered for {key}; the backend fell back to CUDA"

    tag = f"{get_shape_id(shape)}/{get_direction_id((rowwise, columnwise))}"
    for name, cuda_bytes in cuda_output.items():
        assert torch.equal(cutedsl_output[name], cuda_bytes), f"{tag}: {name} differ"


@pytest.mark.parametrize("shape", MATRIX_SIZES, ids=get_shape_id)
@pytest.mark.parametrize("direction", DIRECTIONS, ids=get_direction_id)
@pytest.mark.parametrize("fast_math", [False, True], ids=get_fast_math_id)
def test_cast(fast_math, direction, shape, monkeypatch):
    monkeypatch.setenv("NVTE_USE_FAST_MATH", "1" if fast_math else "0")
    run_test_case(shape, *direction)


@pytest.mark.parametrize("fast_math", [False, True], ids=get_fast_math_id)
def test_row_scaled(fast_math, monkeypatch):
    monkeypatch.setenv("NVTE_USE_FAST_MATH", "1" if fast_math else "0")
    run_test_case((256, 1024), True, False, row_scaled=True)


@pytest.mark.parametrize("direction", DIRECTIONS, ids=get_direction_id)
@pytest.mark.parametrize("kind", ["zero-blocks", "negative-zero-blocks", "all-zero"])
def test_zero_blocks(kind, direction):
    """Zero block amaxes (and, for all-zero, a zero global amax) hit the clamped reciprocal;
    negative zero checks the sign of zero survives the scaling multiply as in CUDA."""
    shape = (256, 1024)
    if kind == "all-zero":
        x = torch.zeros(shape, dtype=torch.bfloat16, device="cuda")
    else:
        x = generate_input(shape)
        x.view(-1, NVFP4_BLOCK_SIZE)[::3].fill_(-0.0 if kind == "negative-zero-blocks" else 0.0)
    run_test_case(shape, *direction, x=x)


@pytest.mark.parametrize("direction", DIRECTIONS, ids=get_direction_id)
def test_unaligned_input_uses_g2s(direction):
    """A 16-byte-aligned input is not eligible for the g2r kernels' 32-byte loads, so the g2s
    kernel serves it."""
    shape = (256, 384)
    rows, cols = shape
    buffer = torch.empty(rows * cols + 8, dtype=torch.bfloat16, device="cuda")
    x = buffer[8:].view(shape)
    x.copy_(generate_input(shape))
    assert x.data_ptr() % 32 == 16
    run_test_case(shape, *direction, x=x)


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
    for name in ("rowwise data", "rowwise scales"):
        assert torch.equal(after[name], reference[name]), f"{name} differ"


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
