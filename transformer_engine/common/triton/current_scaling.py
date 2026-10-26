# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Two-phase FP8 current-scaling quantization kernels written with OpenAI Triton."""

import triton
import triton.language as tl

try:
    from triton.language.extra import libdevice
except ImportError:  # older Triton
    from triton.language.extra.cuda import libdevice


@triton.jit
def _finalize_scale(ws_ptr, scale_inv_ptr, n_prog, eps, FP8MAX: tl.constexpr, POW2: tl.constexpr):
    """Executed by every CTA after its amax atomic; the last one turns amax into the scale.

    Mirrors TE's compute_scale_from_amax. The workspace ``ws = [amax, counter, scale]`` is
    handed back zeroed so no memset launch is needed (CUDA-graph safe).
    """
    cnt = tl.atomic_add(ws_ptr + 1, 1.0)
    if cnt == n_prog - 1:
        amax = tl.atomic_xchg(ws_ptr, 0.0)
        tl.atomic_xchg(ws_ptr + 1, 0.0)
        bad = (amax == float("inf")) | (amax != amax)
        amax = tl.where(amax < eps, eps, amax)
        bad = bad | (amax == 0.0)
        scale = libdevice.div_rn(FP8MAX, amax)
        scale = tl.where(scale == float("inf"), 3.4028234663852886e38, scale)
        if POW2:
            scale = (scale.to(tl.uint32, bitcast=True) & 0xFF800000).to(tl.float32, bitcast=True)
        scale = tl.where(bad, 1.0, scale)
        tl.store(ws_ptr + 2, scale)
        tl.store(scale_inv_ptr, libdevice.rcp_rn(scale))


@triton.jit
def current_scaling_quantize_kernel(
    x_ptr,
    out_ptr,
    ws_ptr,
    scale_inv_ptr,
    n,
    eps,
    BLOCK: tl.constexpr,
    CAST: tl.constexpr,
    FP8MAX: tl.constexpr,
    POW2: tl.constexpr,
):
    """CAST=False: persistent amax pass, one atomic per CTA, last CTA computes the scale.
    CAST=True: one block per CTA, in reverse order so the amax pass's tail is still in L2."""
    n_prog = tl.num_programs(0)
    pid = tl.program_id(0)
    if CAST:
        offs = (n_prog - 1 - pid).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        m = offs < n
        x = tl.load(x_ptr + offs, mask=m, other=0.0).to(tl.float32)
        scale = tl.load(ws_ptr + 2)
        tl.store(out_ptr + offs, (x * scale).to(out_ptr.dtype.element_ty), mask=m)
    else:
        amax = tl.zeros([BLOCK], tl.float32)
        for blk in tl.range(pid, tl.cdiv(n, BLOCK), n_prog):
            offs = blk.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
            x = tl.load(x_ptr + offs, mask=offs < n, other=0.0).to(tl.float32)
            amax = tl.maximum(amax, tl.abs(x))
        tl.atomic_max(ws_ptr, tl.max(amax, axis=0))
        _finalize_scale(ws_ptr, scale_inv_ptr, n_prog, eps, FP8MAX, POW2)
