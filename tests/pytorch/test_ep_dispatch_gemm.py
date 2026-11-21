# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Signal-aware grouped GEMM that follows the EP dispatch, loaded through TVM-FFI."""

# pylint: disable=wrong-import-position

import ctypes
import os

import pytest
import torch

# The optional CuTeDSL dependencies are imported only when the overlap is requested, where a
# missing one fails the module.
if not os.environ.get("NVTE_MOE_OVERLAP"):
    pytest.skip("NVTE_MOE_OVERLAP is not set", allow_module_level=True)

import tvm_ffi

from transformer_engine.common.CuTeDSL.ep_gemm import (
    combine_gemm,
    dispatch_gemm,
    requantize,
    wgrad_gemm,
)

pytestmark = [
    pytest.mark.skipif(
        not torch.cuda.is_available()
        or torch.cuda.get_device_capability() not in ((10, 0), (10, 3), (10, 7)),
        reason="the kernels need SM 10.0, 10.3 or 10.7",
    ),
]

TILE = dispatch_gemm.ROWS_PER_TILE
# SM 10.7 has no MXFP8 tail tiles.
MXFP8_TAILS = torch.cuda.is_available() and torch.cuda.get_device_capability() != (10, 7)


def _unit_e8m0(count, device):
    """``count`` unit E8M0 scales (127), which read the same in any scale layout."""
    return torch.full((count,), 127, dtype=torch.uint8, device=device).view(torch.float8_e8m0fnu)


def _run_gemm(rows_per_expert, rows_alignment, mxfp8, tail_clusters, device_layout):
    """Launch the forward GEMM with every row tile ready, and with ``tail_clusters`` a second
    launch behind it on the same workspace; return (a, b, alpha, prob, c, d, claims) with MXFP8
    operands at unit scales in float32, the MXFP8 ``d`` dequantized and the final claim counter.
    ``device_layout``: the row ends follow expert 0's offset, as the dispatch writes them."""
    device = torch.device("cuda")
    hidden, n, experts = 1024, 1024, len(rows_per_expert)
    rows = sum(rows_per_expert) + TILE  # spare tile: the capacity exceeds the rows in use
    scaling_mode = dispatch_gemm.MXFP8_1D_SCALING if mxfp8 else dispatch_gemm.DELAYED_TENSOR_SCALING
    cfg = dispatch_gemm.DispatchGemmConfig(
        rows,
        hidden,
        n,
        experts,
        num_clusters=8,
        scaling_mode=scaling_mode,
        rows_alignment=rows_alignment,
        device_layout=device_layout,
    )
    launches = []
    for num_clusters in (cfg.num_clusters, tail_clusters):
        if num_clusters == 0:
            continue
        name = (
            f"test_ep_dispatch_gemm_{rows}_{rows_alignment}_{int(mxfp8)}_{num_clusters}"
            f"_{int(device_layout)}"
        )
        assert dispatch_gemm.get_ep_dispatch_gemm_function(
            name,
            rows,
            hidden,
            n,
            experts,
            num_clusters,
            False,
            mxfp8,
            rows_alignment,
            device_layout,
        )
        launches.append(tvm_ffi.get_global_func(name))

    gen = torch.Generator(device=device).manual_seed(0)
    dtype = torch.float8_e4m3fn if mxfp8 else torch.bfloat16
    a = torch.rand(rows, hidden, device=device, generator=gen).to(dtype)
    b = (torch.rand(experts, n, hidden, device=device, generator=gen) * 2 * hidden**-0.5).to(dtype)
    # Powers of two: alpha and prob cost no tolerance.
    alpha = torch.pow(2.0, (torch.arange(experts, device=device) % 5 - 2).float())
    prob = torch.pow(2.0, (torch.arange(rows, device=device) % 3 - 1).float())
    ends = torch.tensor(rows_per_expert, device=device).cumsum(0).to(torch.int32)
    if device_layout:
        # Offsets then the total: expert 0's offset is the word before the 16-byte aligned ends.
        layout = torch.zeros(4 + experts, dtype=torch.int32, device=device)
        layout[4:] = ends
        ends = layout[4:]
    d_cols = n // 2
    c = torch.zeros(rows, n, dtype=torch.bfloat16, device=device)
    d = torch.zeros(rows, d_cols, dtype=dtype, device=device)
    ws_bytes = dispatch_gemm.workspace_layout(cfg)[0]
    workspace = torch.zeros(ws_bytes, dtype=torch.uint8, device=device)
    _, counter_offset, signal_offset, _ = dispatch_gemm.workspace_layout(cfg)
    words = workspace.view(torch.int32)
    # Slot s is ready once signal[s] == s.
    words[signal_offset // 4 :][: cfg.signal_capacity] = torch.arange(
        cfg.signal_capacity, dtype=torch.int32, device=device
    )
    counter = words[counter_offset // 4]
    stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
    if not mxfp8:
        for gemm in launches:
            gemm(a, b, c, d, ends, alpha, prob, workspace, stream)
        torch.cuda.synchronize()
        return a.float(), b.float(), alpha, prob, c, d.float(), counter.item()

    # The forward takes no prob.
    sfd = torch.zeros(rows * d_cols // 32, dtype=torch.uint8, device=device)
    norm_const = torch.ones(1, dtype=torch.float32, device=device)
    for gemm in launches:
        gemm(
            a, b, _unit_e8m0(experts * n * hidden // 32, device),
            _unit_e8m0(rows * hidden // 32, device), c, d,
            sfd.view(torch.float8_e8m0fnu), ends, alpha, norm_const, workspace, stream,
        )  # fmt: skip
    torch.cuda.synchronize()
    # GEMM-swizzled scales of [rows, d_cols]: 128 x 4 atoms, rows 32 apart interleaved.
    scale_cols = d_cols // 32
    scales = sfd.view(rows // 128, scale_cols // 4, 32, 4, 4).permute(0, 3, 2, 1, 4)
    scales = torch.pow(2.0, scales.reshape(rows, scale_cols).float() - 127)
    d_float = d.float() * scales.repeat_interleave(32, dim=1)
    return a.float(), b.float(), alpha, torch.ones_like(prob), c, d_float, counter.item()


@pytest.mark.parametrize(
    "rows_alignment,mxfp8,rows_per_expert",
    [
        (TILE, False, [2 * TILE, TILE, 0, 3 * TILE]),
        (TILE, True, [2 * TILE, TILE, 0, 3 * TILE]),
        # Tail tiles: an expert of only a tail, a tail after full tiles, an empty expert.
        (128, False, [128, 2 * TILE + 128, 0, TILE]),
        pytest.param(
            128,
            True,
            [128, 2 * TILE + 128, 0, TILE],
            marks=pytest.mark.skipif(not MXFP8_TAILS, reason="no MXFP8 tail tiles on SM 10.7"),
        ),
        # Every tail height, and the tails that run inside the last full tile of their expert.
        (64, False, [64, 128, 192, 2 * TILE + 64, 2 * TILE + 128, 0, TILE]),
    ],
)
@pytest.mark.parametrize("tail_clusters", [0, 4])
@pytest.mark.parametrize("device_layout", [False, True])
def test_dispatch_gemm_matches_reference(
    rows_alignment, mxfp8, rows_per_expert, tail_clusters, device_layout
):
    """Grouped GEMM over padded slabs matches a float32 reference, also with experts padded to
    fewer rows than a tile, with a tail launch sharing the tiles and with the dispatch-written
    layout; rows past the last expert stay 0."""
    a, b, alpha, prob, c, d, claims = _run_gemm(
        rows_per_expert, rows_alignment, mxfp8, tail_clusters, device_layout
    )
    if tail_clusters:
        # The tail's schedulers claimed from the counter they share with the base's.
        assert claims > _run_gemm(rows_per_expert, rows_alignment, mxfp8, 0, device_layout)[-1]
    used = sum(rows_per_expert)
    ref_c = torch.zeros(a.shape[0], b.shape[1], dtype=torch.float32, device=a.device)
    start = 0
    for e, count in enumerate(rows_per_expert):
        ref_c[start : start + count] = a[start : start + count] @ b[e].t() * alpha[e]
        start += count
    col = torch.arange(b.shape[1] // 2, device=a.device)
    gate = 64 * (col // 32) + col % 32
    ref_d = ref_c[:, gate + 32] * torch.nn.functional.silu(ref_c[:, gate]) * prob[:, None]
    # BF16 outputs round once; the MXFP8 d may round to either neighbor of its E4M3 value.
    for got, ref, tol in ((c, ref_c, 2.0**-8), (d, ref_d, 2.0**-3 if mxfp8 else 2.0**-8)):
        err = (got[:used].float() - ref[:used]).abs() / ref[:used].abs().clamp_min(1e-6)
        assert err.max() <= tol
        assert torch.count_nonzero(got[used:]) == 0


def _run_combine(rows_per_expert, rows_alignment, mxfp8, backward):
    """Launch the combine GEMM over padded slabs and return (a, b, alpha, prob, d, used) with
    MXFP8 operands at unit scales in float32. The forward weight is [E, n, k], backward [E, k, n];
    ``d`` [rows, n] is BF16 in both recipes. Rows past the last expert stay 0."""
    device = torch.device("cuda")
    k, n, experts = 1024, 1024, len(rows_per_expert)
    rows = sum(rows_per_expert) + TILE  # spare tile: the capacity exceeds the rows in use
    name = f"test_ep_combine_gemm_{rows}_{rows_alignment}_{int(mxfp8)}_{int(backward)}"
    assert combine_gemm.get_ep_combine_gemm_function(
        name, rows, k, n, experts, 8, 0, False, backward, mxfp8, rows_alignment
    )
    gemm = tvm_ffi.get_global_func(name)

    gen = torch.Generator(device=device).manual_seed(0)
    dtype = torch.float8_e4m3fn if mxfp8 else torch.bfloat16
    w_shape = (experts, k, n) if backward else (experts, n, k)
    a = torch.rand(rows, k, device=device, generator=gen).to(dtype)
    b = (torch.rand(w_shape, device=device, generator=gen) * 2 * k**-0.5).to(dtype)
    # Powers of two: alpha and prob cost no tolerance.
    alpha = torch.pow(2.0, (torch.arange(experts, device=device) % 5 - 2).float())
    prob = torch.pow(2.0, (torch.arange(rows, device=device) % 3 - 1).float())
    ends = torch.tensor(rows_per_expert, device=device).cumsum(0).to(torch.int32)
    d = torch.zeros(rows, n, dtype=torch.bfloat16, device=device)
    ws_bytes = combine_gemm.get_ep_combine_gemm_workspace_layout(
        rows, k, n, experts, 8, 0, False, backward, mxfp8, rows_alignment
    )[0]
    workspace = torch.zeros(ws_bytes, dtype=torch.uint8, device=device)
    stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
    if mxfp8:
        gemm(
            a, b, _unit_e8m0(experts * n * k // 32, device),
            _unit_e8m0(rows * k // 32, device), d, ends, alpha, prob, workspace, stream,
        )  # fmt: skip
    else:
        gemm(a, b, d, ends, alpha, prob, workspace, stream)
    torch.cuda.synchronize()
    return a.float(), b.float(), alpha, prob, d, sum(rows_per_expert)


@pytest.mark.parametrize(
    "backward,mxfp8,rows_alignment,rows_per_expert",
    [
        (False, False, TILE, [2 * TILE, TILE, 0, 3 * TILE]),
        (False, True, TILE, [2 * TILE, TILE, 0, 3 * TILE]),
        (True, False, TILE, [2 * TILE, TILE, 0, 3 * TILE]),
        (True, True, TILE, [2 * TILE, TILE, 0, 3 * TILE]),
        # Tail tiles: an expert of only a tail, a tail after full tiles, an empty expert.
        (False, False, 128, [128, 2 * TILE + 128, 0, TILE]),
    ],
)
def test_combine_gemm_matches_reference(backward, mxfp8, rows_alignment, rows_per_expert):
    """Grouped combine GEMM over padded slabs, per expert ``d = (a @ weight) * alpha * prob``,
    matches a float32 reference; rows past the last expert stay 0."""
    a, b, alpha, prob, d, used = _run_combine(rows_per_expert, rows_alignment, mxfp8, backward)
    ref = torch.zeros(a.shape[0], b.shape[2] if backward else b.shape[1], device=a.device)
    start = 0
    for e, count in enumerate(rows_per_expert):
        sl = slice(start, start + count)
        weight = b[e] if backward else b[e].t()
        ref[sl] = a[sl] @ weight * alpha[e] * prob[sl, None]
        start += count
    # BF16 output rounds once, whether the operands are BF16 or MXFP8.
    err = (d[:used].float() - ref[:used]).abs() / ref[:used].abs().clamp_min(1e-6)
    assert err.max() <= 2.0**-8
    assert torch.count_nonzero(d[used:]) == 0


def _run_wgrad(rows_per_expert, mxfp8, fp32_out, accumulate):
    """Launch the weight gradient over padded slabs and return (grad, x, init, weight_grad) with
    MXFP8 operands at unit scales in float32. ``weight_grad[e] = grad[e]^T @ x[e]``, written or
    added to ``init``."""
    device = torch.device("cuda")
    m, n, experts = 256, 512, len(rows_per_expert)
    rows = sum(rows_per_expert) + TILE  # spare tile: the capacity exceeds the rows in use
    routed = max(sum(rows_per_expert) // experts, 1)
    args = (rows, m, n, experts, 8, fp32_out, accumulate, False, 0, mxfp8, False, TILE, routed)
    name = f"test_ep_wgrad_{rows}_{int(mxfp8)}_{int(fp32_out)}_{int(accumulate)}"
    assert wgrad_gemm.get_ep_wgrad_function(name, *args)
    launches = [tvm_ffi.get_global_func(name)]
    if mxfp8 and wgrad_gemm.get_ep_wgrad_reads_descriptors(*args):
        # A descriptors-only launch writes the per-expert descriptors the GEMM launch reads.
        desc = (rows, m, n, experts, 1, fp32_out, accumulate, False, 0, mxfp8, True, TILE, routed)
        assert wgrad_gemm.get_ep_wgrad_function(name + "_desc", *desc)
        launches.insert(0, tvm_ffi.get_global_func(name + "_desc"))

    gen = torch.Generator(device=device).manual_seed(0)
    dtype = torch.float8_e4m3fn if mxfp8 else torch.bfloat16
    # Powers of two keep every product and sum exact in float32; signs on grad add cancellation.
    signs = 1 - 2 * torch.randint(0, 2, (rows, m), device=device, generator=gen)
    grad = (
        signs
        * torch.pow(2.0, -torch.randint(0, 2, (rows, m), device=device, generator=gen).float())
    ).to(dtype)
    x = torch.pow(2.0, -torch.randint(0, 2, (rows, n), device=device, generator=gen).float()).to(
        dtype
    )
    out_dtype = torch.float32 if fp32_out else torch.bfloat16
    init = torch.pow(2.0, (torch.arange(experts * m * n, device=device) % 3 - 1).float()).reshape(
        experts, m, n
    )
    weight_grad = (init if accumulate else torch.zeros_like(init)).to(out_dtype, copy=True)
    ends = torch.tensor(rows_per_expert, device=device).cumsum(0).to(torch.int32)
    ws_bytes = wgrad_gemm.get_ep_wgrad_workspace_layout(*args)[0]
    workspace = torch.zeros(ws_bytes, dtype=torch.uint8, device=device)
    stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
    for gemm in launches:
        if mxfp8:
            gemm(
                grad, x, _unit_e8m0(rows * m // 32, device), _unit_e8m0(rows * n // 32, device),
                weight_grad, ends, workspace, stream,
            )  # fmt: skip
        else:
            gemm(grad, x, weight_grad, ends, workspace, stream)
    torch.cuda.synchronize()
    base = init if accumulate else torch.zeros_like(init)
    return grad.float(), x.float(), base, weight_grad.float()


@pytest.mark.parametrize(
    "mxfp8,fp32_out,accumulate,rows_per_expert",
    [
        (False, False, False, [2 * TILE, TILE, 0, 3 * TILE]),  # BF16 written
        (False, True, False, [2 * TILE, TILE, 0, 3 * TILE]),  # FP32 written
        (False, False, True, [TILE, 0, 2 * TILE, TILE]),  # BF16 accumulated
        (True, False, False, [2 * TILE, TILE, 0, 3 * TILE]),  # MXFP8 operands
        (True, True, False, [2 * TILE, TILE, 0, 3 * TILE]),  # MXFP8, FP32 written
        (True, False, True, [TILE, 0, 2 * TILE, TILE]),  # MXFP8, BF16 accumulated
        (True, True, True, [TILE, 0, 2 * TILE, TILE]),  # MXFP8, FP32 accumulated
    ],
)
def test_wgrad_gemm_matches_reference(mxfp8, fp32_out, accumulate, rows_per_expert):
    """Grouped weight gradient matches a float32 reference; empty experts keep their input."""
    grad, x, base, weight_grad = _run_wgrad(rows_per_expert, mxfp8, fp32_out, accumulate)
    ref = base.clone()
    start = 0
    for e, count in enumerate(rows_per_expert):
        sl = slice(start, start + count)
        ref[e] += grad[sl].t() @ x[sl]
        start += count
    # A written FP32 gradient is exact; a BF16 one rounds once.
    tol = 2.0**-16 if fp32_out else 2.0**-8
    err = (weight_grad - ref).abs() / ref.abs().clamp_min(1e-6)
    assert err.max() <= tol


@pytest.mark.skipif(MXFP8_TAILS, reason="SM 10.7 only")
def test_wgrad_gemm_rejects_one_tile_capacity():
    """SM 10.7 refuses an added or FP32 MXFP8 weight gradient over 256 receive rows or fewer."""
    cfg = wgrad_gemm.WgradConfig(
        TILE,
        256,
        512,
        2,
        8,
        fp32_out=True,
        scaling_mode=wgrad_gemm.MXFP8_1D_SCALING,
        rows_alignment=TILE,
        routed_rows=TILE,
    )
    with pytest.raises(ValueError, match="above 256 rows"):
        wgrad_gemm.build_kernel(cfg)


def _read_col_scales(words, rows_per_expert, cols, device):
    """Columnwise E8M0 scale per row and column from the GEMM-swizzled per-expert scale words."""
    scale = torch.ones(sum(rows_per_expert), cols, device=device)
    col = torch.arange(cols, device=device)
    start = 0
    for count in rows_per_expert:
        if count:
            r = torch.arange(start, start + count, device=device)
            group = r // 128 - start // 128
            index = (
                start * (cols // 128)
                + (col // 128) * count
                + group[:, None] * 128
                + (col % 32)[None, :] * 4
                + ((col // 32) % 4)[None, :]
            )
            block = (r // 32) % 4
            e8m0 = (words[index] >> (8 * block[:, None])) & 0xFF
            scale[start : start + count] = torch.pow(2.0, e8m0.float() - 127)
        start += count
    return scale


def _run_requant(rows_per_expert, cols, ready_slot_rows=0, nan_scale=None):
    """Launch the MXFP8 rowwise-to-columnwise requant and return (codes, col_codes, scale words,
    used). Row scales are unit, so the rowwise values are the codes themselves. With
    ``ready_slot_rows``, the rows and then their dispatch flags arrive late, on another stream.
    ``nan_scale`` is a (row, 32-column block) whose rowwise scale is the E8M0 NaN."""
    device = torch.device("cuda")
    experts = len(rows_per_expert)
    # Spare rows, a whole slot: the capacity exceeds the rows in use.
    rows = sum(rows_per_expert) + max(128, ready_slot_rows)
    name = f"test_ep_requant_{rows}_{cols}_{ready_slot_rows}"
    assert requantize.get_ep_requant_function(name, rows, cols, experts, ready_slot_rows)
    fn = tvm_ffi.get_global_func(name)

    gen = torch.Generator(device=device).manual_seed(0)
    # Powers of two: the columnwise requant represents them exactly, so it costs no tolerance.
    exps = torch.randint(-2, 2, (rows, cols), device=device, generator=gen).float()
    signs = 1 - 2 * torch.randint(0, 2, (rows, cols), device=device, generator=gen)
    codes = (signs * torch.pow(2.0, exps)).to(torch.float8_e4m3fn)
    row_scales = _unit_e8m0(rows * cols // 32, device)
    if nan_scale is not None:
        row_scales.view(torch.uint8)[nan_scale[0] * (cols // 32) + nan_scale[1]] = 255
    col_codes = torch.zeros(rows, cols, dtype=torch.float8_e4m3fn, device=device)
    col_scales = _unit_e8m0(rows * cols // 32, device)
    ends = torch.tensor(rows_per_expert, device=device).cumsum(0).to(torch.int32)
    stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
    if ready_slot_rows:
        received = torch.zeros_like(codes)
        signal = torch.full((rows // ready_slot_rows,), -1, dtype=torch.int32, device=device)
        # Enqueued first, so a run that serializes kernels does not hang.
        arrival = torch.cuda.Stream()
        arrival.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(arrival):
            torch.cuda._sleep(10**7)
            received.copy_(codes)
            signal.fill_(0)
        fn(received, row_scales, col_codes, col_scales, ends, signal, stream)
    else:
        fn(codes, row_scales, col_codes, col_scales, ends, stream)
    torch.cuda.synchronize()
    words = col_scales.view(torch.uint8).view(torch.int32)
    return codes.float(), col_codes.float(), words, sum(rows_per_expert)


@pytest.mark.parametrize("ready_slot_rows", [0, 128, 256])
@pytest.mark.parametrize(
    "cols,rows_per_expert",
    [
        (256, [128, 256, 0, 128]),
        (512, [256, 0, 128, 128]),
    ],
)
def test_requantize_matches_reference(cols, rows_per_expert, ready_slot_rows):
    """MXFP8 rowwise-to-columnwise requant reproduces the dequantized rows exactly for power-of-two
    values; rows past the last expert stay 0. With ``ready_slot_rows``, it waits for rows that
    arrive after its launch."""
    codes, col_codes, words, used = _run_requant(rows_per_expert, cols, ready_slot_rows)
    scale = _read_col_scales(words, rows_per_expert, cols, codes.device)
    assert torch.equal(col_codes[:used] * scale, codes[:used])
    assert torch.count_nonzero(col_codes[used:]) == 0


def test_requantize_nan_scale():
    """A rowwise E8M0 scale of 255 makes its 32 values NaN and leaves every other value finite."""
    cols, row, block = 256, 5, 2
    _, col_codes, _, used = _run_requant([128], cols, nan_scale=(row, block))
    expected = torch.zeros(used, cols, dtype=torch.bool, device=col_codes.device)
    expected[row, block * 32 : (block + 1) * 32] = True
    assert torch.equal(col_codes[:used].isnan(), expected)
