# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Single-GPU tests for the NVFP4 EP token rows: the pack (RN and stochastic rounding,
power-of-two token scales), the BF16 decode, and the direct receive that builds the expert GEMM's
NVFP4 GroupedTensor from packed rows (rowwise codes reused, columnwise RHT quantized in one pass).
"""

import os
import random

import pytest
import torch

import transformer_engine.pytorch as te
import transformer_engine_torch as tex
from transformer_engine.common.recipe import NVFP4BlockScaling
from transformer_engine.pytorch.ep import _nvfp4_direct_receive, _nvfp4_row_bytes
from transformer_engine.pytorch.tensor.nvfp4_tensor import (
    NVFP4Quantizer,
    get_hadamard_matrix,
    get_wgrad_sign_vector,
)

recipe_available, reason_for_no_recipe = te.is_nvfp4_available(return_reason=True)
pytestmark = pytest.mark.skipif(not recipe_available, reason=reason_for_no_recipe)

E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
LUT = torch.cat([E2M1, -E2M1])
NVFP4_MAX = 6.0 * 448.0
FAST_MATH = os.environ.get("NVTE_USE_FAST_MATH", "0") == "1"


def _tokens(total, hidden, seed, spread=(-2, 1)):
    """BF16 tokens whose magnitudes are log-spaced over ``spread`` decades, in random order."""
    gen = torch.Generator(device="cuda")
    gen.manual_seed(seed)
    x = torch.randn(total, hidden, generator=gen, device="cuda")
    x = x * torch.logspace(*spread, total, device="cuda")[:, None]
    perm = torch.randperm(total, generator=gen, device="cuda")
    return x[perm].to(torch.bfloat16)


def _pack(x, stochastic=False, seed=None):
    if seed is not None:
        torch.cuda.manual_seed(seed)
    return tex.ep_nvfp4_pack(x, stochastic)


def _unpack(packed, hidden):
    return tex.ep_nvfp4_unpack(packed, hidden)


def _receive(packed, hidden, counts, stochastic=False):
    return _nvfp4_direct_receive(packed, hidden, counts, torch.bfloat16, stochastic)


def _swz(r, c, ncols):
    return (
        ((r // 128) * ((ncols + 3) // 4) + c // 4) * 512
        + (r % 32) * 16
        + ((r // 32) % 4) * 4
        + (c % 4)
    )


def _codes(data_bytes):
    d = data_bytes.to(torch.int64)
    return torch.stack([d & 15, d >> 4], dim=-1).flatten(-2)


def _token_scales(packed, hidden):
    off = hidden // 2 + hidden // 16
    return packed[:, off : off + 4].contiguous().view(torch.float32).flatten()


def _ref_pack(x):
    """Reference RN pack in PyTorch: codes [T, H], block scales (uint8) [T, H/16], token scales."""
    x = x.float()
    num_tokens, hidden = x.shape
    amax = x.abs().amax(dim=1)
    bits = (amax / NVFP4_MAX).view(torch.int32)
    exp = bits & 0x7F800000
    dec = torch.where((bits & 0x7FFFFF) == 0, exp, exp + 0x800000).clamp_min(0x800000)
    decode = torch.where(amax == 0, 0.0, dec.view(torch.float32))
    encode = torch.where(amax == 0, 1.0, 1.0 / decode)
    blocks = x.view(num_tokens, hidden // 16, 16)
    bmax = blocks.abs().amax(dim=-1)
    bscale = (bmax * (encode * (1.0 / 6.0))[:, None]).clamp_max(448.0).to(torch.float8_e4m3fn)
    rcp = (1.0 / (bscale.float() * decode[:, None])).clamp_max(torch.finfo(torch.float32).max)
    q = (blocks * rcp[..., None]).view(num_tokens, hidden)
    a = q.abs()
    code = torch.zeros_like(q, dtype=torch.int64)
    for thr, inclusive in ((0.25, True), (0.75, False), (1.25, True), (1.75, False), (2.5, True)):
        code += (a > thr) if inclusive else (a >= thr)
    code += (a >= 3.5).long() + (a > 5.0).long()
    code += torch.where(q < 0, 8, 0)
    return code, bscale.view(torch.uint8), decode


def _dequant_rowwise(g, total, hidden):
    lut = LUT.to("cuda")
    data = g.rowwise_data[: total * hidden // 2].view(total, hidden // 2)
    vals = lut[_codes(data)]
    r = torch.arange(total, device="cuda")[:, None]
    c = torch.arange(hidden // 16, device="cuda")[None, :]
    sb = g.scale_inv[_swz(r, c, hidden // 16)].view(torch.float8_e4m3fn).float()
    counts = g.first_dims
    group_of_row = torch.repeat_interleave(torch.arange(counts.numel(), device="cuda"), counts)
    gscale = (g.amax / NVFP4_MAX)[group_of_row][:, None]
    return vals * sb.repeat_interleave(16, dim=1) * gscale


def _dequant_columnwise_inverse_rht(g, hidden):
    """Columnwise operand dequantized and mapped back through the inverse RHT: [total, hidden]."""
    lut = LUT.to("cuda")
    signs = get_wgrad_sign_vector("cuda")
    had = get_hadamard_matrix(16, "cuda")  # symmetric, orthonormal (1/4 scaled)
    out = []
    off = 0
    for gi, n in enumerate(g.first_dims.tolist()):
        if n == 0:
            continue
        data = g.columnwise_data[off * hidden // 2 : (off + n) * hidden // 2].view(hidden, n // 2)
        vals = lut[_codes(data)]  # [hidden, n]
        k = torch.arange(hidden, device="cuda")[:, None]
        b = torch.arange(n // 16, device="cuda")[None, :]
        sc = g.columnwise_scale_inv[off * hidden // 16 + _swz(k, b, n // 16)]
        sc = sc.view(torch.float8_e4m3fn).float().repeat_interleave(16, dim=1)
        y = vals * sc * (g.columnwise_amax[gi] / NVFP4_MAX)
        # y = H (s * v) per 16-token block  =>  v = s * (H y)
        yb = y.view(hidden, n // 16, 16)
        v = torch.einsum("ij,kbj->kbi", had, yb) * signs
        out.append(v.reshape(hidden, n).t())
        off += n
    return torch.cat(out, dim=0)


def _te_group_quantize(x, counts, hidden, stochastic=False):
    q = NVFP4Quantizer(
        rowwise=True,
        columnwise=True,
        with_rht=True,
        with_post_rht_amax=True,
        stochastic_rounding=stochastic,
    )
    q.optimize_for_gemm = True
    q.internal = True
    return tex.group_quantize(
        x, q, counts.numel(), counts, tensor_offsets=tex.splits_to_offsets(counts, hidden)
    )


def _rel(a, b):
    return ((a - b).norm() / b.norm().clamp_min(1e-30)).item()


def test_row_bytes_layout():
    assert _nvfp4_row_bytes(7168) == 4048
    x = _tokens(4, 7168, seed=0)
    assert tuple(_pack(x).shape) == (4, 4048)


@pytest.mark.skipif(FAST_MATH, reason="the reference uses IEEE division")
@pytest.mark.parametrize("hidden", [512, 1024, 7168])
def test_pack_matches_reference(hidden):
    x = _tokens(300, hidden, seed=hidden)
    x[5] = 0
    x[7, :16] = 0  # an all-zero block
    packed = _pack(x)
    code, bscale, decode = _ref_pack(x)
    got_code = _codes(packed[:, : hidden // 2])
    # Zero magnitudes may carry either sign.
    assert torch.equal(
        torch.where(got_code & 7 == 0, 0, got_code), torch.where(code & 7 == 0, 0, code)
    )
    assert torch.equal(packed[:, hidden // 2 : hidden // 2 + hidden // 16], bscale)
    assert torch.equal(_token_scales(packed, hidden), decode)
    assert (packed[:, hidden // 2 + hidden // 16 + 4 :] == 0).all()


def test_pack_pow2_scales_and_decode():
    hidden = 1024
    x = _tokens(64, hidden, seed=0)
    x[3] = 0
    packed = _pack(x)
    ts = _token_scales(packed, hidden)
    assert ts[3].item() == 0.0
    nz = ts[ts > 0]
    assert torch.equal(torch.frexp(nz).mantissa, torch.full_like(nz, 0.5)), "not powers of two"
    amax = x.float().abs().amax(dim=1)
    ok = amax > 0
    assert (amax[ok] <= NVFP4_MAX * ts[ok]).all()
    assert (amax[ok] > NVFP4_MAX * ts[ok] / 2).all()
    dec = _unpack(packed, hidden).float()
    assert dec[3].abs().max().item() == 0.0
    # Per-token scales: at least as accurate as TE's per-tensor NVFP4 cast of the same tokens.
    te_dec = NVFP4Quantizer(rowwise=True, columnwise=False)(x).dequantize().float()
    assert _rel(dec, x.float()) <= 1.02 * _rel(te_dec, x.float())


def test_unpack_bounded_by_counts():
    hidden = 512
    packed = _pack(_tokens(256, hidden, seed=1))
    counts = torch.tensor([64, 0, 32], dtype=torch.int64, device="cuda")
    out = torch.full((256, hidden), 7.0, dtype=torch.bfloat16, device="cuda")
    tex.ep_nvfp4_unpack(packed, hidden, counts, out)
    assert torch.equal(out[:96], _unpack(packed, hidden)[:96])
    assert (out[96:] == 7.0).all()


@pytest.mark.parametrize("hidden", [512, 7168])
def test_pack_sr_scales_neighbors_and_seed(hidden):
    x = _tokens(128, hidden, seed=3)
    x[2] = 0
    rn = _pack(x)
    a = _pack(x, stochastic=True, seed=11)
    b = _pack(x, stochastic=True, seed=11)
    c = _pack(x, stochastic=True, seed=12)
    assert torch.equal(a, b), "same seed must give the same bits"
    assert not torch.equal(a, c), "different seeds must draw different bits"
    # Block scales, token scales and padding are those of the RN pack.
    assert torch.equal(a[:, hidden // 2 :], rn[:, hidden // 2 :])
    # Every value is one of the two grid neighbours of the scaled input.
    dec = _unpack(a, hidden).float()
    dec_rn = _unpack(rn, hidden).float()
    assert dec[2].abs().max().item() == 0.0
    off = hidden // 2
    sb = a[:, off : off + hidden // 16].contiguous().view(torch.float8_e4m3fn).float()
    ts = _token_scales(a, hidden)[:, None]
    step_scale = (sb * ts).repeat_interleave(16, dim=1)  # value of one E2M1 unit (1.0)
    # E2M1 spacing is <= 2 units, so |SR - x| <= 2 * scale and |SR - RN| <= 2 * scale.
    assert ((dec - x.float()).abs() <= 2 * step_scale + 1e-30).all()
    assert ((dec - dec_rn).abs() <= 2 * step_scale + 1e-30).all()


def test_pack_sr_unbiased():
    hidden = 512
    x = _tokens(64, hidden, seed=5, spread=(-0.3, 0.3))
    n = 512
    acc = torch.zeros(64, hidden, dtype=torch.float64, device="cuda")
    torch.cuda.manual_seed(1000)
    for _ in range(n):
        acc += _unpack(_pack(x, stochastic=True), hidden).double()
    mean = acc / n
    rn = _unpack(_pack(x), hidden).double()
    xf = x.double()
    bias = _rel(mean, xf)
    rn_err = _rel(rn, xf)
    signed = ((mean - xf).sum() / xf.abs().sum()).item()
    print(f"SR pack: rel(mean of {n}) = {bias:.5f}, RN rel err = {rn_err:.5f}, signed {signed:.2e}")
    assert bias < 0.25 * rn_err
    assert abs(signed) < 1e-3


@pytest.mark.parametrize("hidden", [512, 7168])
@pytest.mark.parametrize("counts_list", [[256, 0, 512, 256], [768], [0, 256], [128, 384]])
def test_direct_receive_rowwise_exact_and_columnwise_quality(hidden, counts_list):
    counts = torch.tensor(counts_list, dtype=torch.int64, device="cuda")
    total = sum(counts_list)
    capacity = total + 256  # rows past the counts must be ignored
    # Token magnitudes within ~8x of each other (like normalized activations), so re-expressed
    # block scales stay in E4M3's normal range and the rowwise reuse is exact.
    x = _tokens(capacity, hidden, seed=total + hidden, spread=(-0.45, 0.45))
    x[1] = 0  # a zero token
    packed = _pack(x)
    decoded = _unpack(packed, hidden)
    g = _receive(packed, hidden, counts)
    torch.cuda.synchronize()
    assert g._with_gemm_swizzled_scales
    assert not g.quantizer.stochastic_rounding

    # Per-expert second-level scales: 6*448*G and 4x that, G = max token scale of the expert.
    ts = _token_scales(packed[:total], hidden)
    starts = [0] + torch.tensor(counts_list).cumsum(0).tolist()
    for gi in range(len(counts_list)):
        seg = ts[starts[gi] : starts[gi + 1]]
        G = seg.max().item() if seg.numel() and seg.max().item() > 0 else 1.0
        assert g.amax[gi].item() == NVFP4_MAX * G
        assert g.columnwise_amax[gi].item() == 4 * NVFP4_MAX * G

    # Rowwise: the sender's codes and values are reused exactly.
    n_data = total * hidden // 2
    assert torch.equal(g.rowwise_data[:n_data], packed[:total, : hidden // 2].reshape(-1))
    assert torch.equal(_dequant_rowwise(g, total, hidden), decoded[:total].float())

    # Columnwise: as close to the decoded tokens as TE's own RHT quantization of them.
    ref = _te_group_quantize(decoded[:total].contiguous(), counts, hidden)
    got_err = _rel(_dequant_columnwise_inverse_rht(g, hidden), decoded[:total].float())
    te_err = _rel(_dequant_columnwise_inverse_rht(ref, hidden), decoded[:total].float())
    print(f"columnwise rel err direct={got_err:.5f} te={te_err:.5f}")
    assert got_err < 1.05 * te_err + 1e-4


def test_direct_receive_wide_token_range_rowwise_quality():
    """With token magnitudes spread over 1000x, small tokens' block scales underflow against the
    expert scale; the rowwise error stays at TE's per-tensor level."""
    hidden = 1024
    counts = torch.tensor([512, 512], dtype=torch.int64, device="cuda")
    x = _tokens(1024, hidden, seed=7, spread=(-2, 1))
    packed = _pack(x)
    decoded = _unpack(packed, hidden)
    g = _receive(packed, hidden, counts)
    ref = _te_group_quantize(decoded.contiguous(), counts, hidden)
    got_err = _rel(_dequant_rowwise(g, 1024, hidden), x.float())
    te_err = _rel(_dequant_rowwise(ref, 1024, hidden), x.float())
    print(f"rowwise rel err vs original: direct={got_err:.5f} decode+TE={te_err:.5f}")
    assert got_err <= te_err * 1.02


@pytest.mark.parametrize("hidden", [512, 7168])
@pytest.mark.parametrize("counts_list", [[256, 0, 512, 256], [768], [0, 256]])
def test_direct_receive_stochastic_rounding(hidden, counts_list):
    counts = torch.tensor(counts_list, dtype=torch.int64, device="cuda")
    total = sum(counts_list)
    capacity = total + 256
    x = _tokens(capacity, hidden, seed=total + hidden, spread=(-0.45, 0.45))
    x[1] = 0
    packed = _pack(x, stochastic=True, seed=77)
    decoded = _unpack(packed, hidden)
    g = _receive(packed, hidden, counts, stochastic=True)
    g_rn = _receive(packed, hidden, counts)
    torch.cuda.synchronize()
    assert g._with_gemm_swizzled_scales
    assert g.quantizer.stochastic_rounding
    # Rowwise and the per-expert scales do not depend on the rounding mode of the receive.
    n_data = total * hidden // 2
    assert torch.equal(g.rowwise_data[:n_data], g_rn.rowwise_data[:n_data])
    assert torch.equal(g.amax, g_rn.amax) and torch.equal(g.columnwise_amax, g_rn.columnwise_amax)
    assert torch.equal(_dequant_rowwise(g, total, hidden), decoded[:total].float())
    # Columnwise: a single SR draw is about as close to the decoded rows as TE's own RHT + SR
    # quantization of them, and SR is unbiased (the mean over draws converges).
    ref = _te_group_quantize(decoded[:total].contiguous(), counts, hidden, stochastic=True)
    target = decoded[:total].float()
    got_err = _rel(_dequant_columnwise_inverse_rht(g, hidden), target)
    te_err = _rel(_dequant_columnwise_inverse_rht(ref, hidden), target)
    rn_err = _rel(_dequant_columnwise_inverse_rht(g_rn, hidden), target)
    acc = torch.zeros_like(target, dtype=torch.float64)
    draws = 32
    for _ in range(draws):
        acc += _dequant_columnwise_inverse_rht(
            _receive(packed, hidden, counts, stochastic=True), hidden
        ).double()
    mean_err = _rel(acc / draws, target.double())
    print(
        f"columnwise SR rel err: direct={got_err:.5f} te_sr={te_err:.5f} direct_rn={rn_err:.5f}"
        f" mean_of_{draws}={mean_err:.5f}"
    )
    assert got_err < 1.15 * te_err + 1e-4
    assert mean_err < 0.4 * got_err


def test_direct_receive_cuda_graph_reads_live_counts():
    hidden = 512
    counts = torch.tensor([256, 256], dtype=torch.int64, device="cuda")
    packed = _pack(_tokens(768, hidden, seed=1, spread=(-0.45, 0.45)))
    out = {}

    def step():
        out["g"] = _receive(packed, hidden, counts)

    step()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        step()
    counts.copy_(torch.tensor([512, 256], device="cuda"))
    packed.copy_(_pack(_tokens(768, hidden, seed=2, spread=(-0.45, 0.45))))
    graph.replay()
    torch.cuda.synchronize()
    g = out["g"]
    decoded = _unpack(packed, hidden)
    assert torch.equal(_dequant_rowwise(g, 768, hidden), decoded.float())
    eager = _receive(packed, hidden, counts)
    n = 768 * hidden // 2
    assert torch.equal(g.columnwise_data[:n], eager.columnwise_data[:n])
    assert torch.equal(g.columnwise_amax, eager.columnwise_amax)


def test_sr_pack_and_receive_cuda_graph_fresh_bits():
    hidden = 512
    counts = torch.tensor([256, 256], dtype=torch.int64, device="cuda")
    x = _tokens(512, hidden, seed=9, spread=(-0.45, 0.45))
    out = {}

    def step():
        packed = tex.ep_nvfp4_pack(x, True)
        out["packed"] = packed
        out["g"] = _receive(packed, hidden, counts, stochastic=True)

    step()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        step()
    snaps = []
    for _ in range(2):
        graph.replay()
        torch.cuda.synchronize()
        g = out["g"]
        snaps.append((out["packed"].clone(), g.columnwise_data[: 512 * hidden // 2].clone()))
        dec = _unpack(out["packed"], hidden)
        assert torch.equal(_dequant_rowwise(g, 512, hidden), dec.float())
    assert not torch.equal(snaps[0][0], snaps[1][0]), "pack must draw fresh bits per replay"
    assert not torch.equal(snaps[0][1], snaps[1][1]), "receive must draw fresh bits per replay"


def _grouped_mlp(group_size, hidden, recipe, weight_requires_grad):
    with te.quantized_model_init(enabled=True, recipe=recipe):
        fc1 = te.ops.GroupedLinear(
            group_size,
            hidden,
            2 * hidden,
            bias=False,
            device="cuda",
            dtype=torch.bfloat16,
            split_alignment=256,
        )
        fc2 = te.ops.GroupedLinear(
            group_size,
            hidden,
            hidden,
            bias=False,
            device="cuda",
            dtype=torch.bfloat16,
            split_alignment=256,
        )
        module = te.ops.Sequential(fc1, te.ops.ScaledSwiGLU(glu_interleave_size=32), fc2)
    with torch.no_grad():
        for param in module.parameters():
            param.requires_grad_(requires_grad=weight_requires_grad)
    return module, fc1, fc2


def _split_sizes(group_size):
    split = [256 * i for i in range(group_size)]
    random.Random(0).shuffle(split)
    return torch.tensor(split, dtype=torch.int64, device="cuda"), sum(split)


@pytest.mark.parametrize("weight_requires_grad", (False, True))
def test_fused_grouped_mlp_with_direct_receive(weight_requires_grad):
    """The fused grouped MLP fed the direct-receive GroupedTensor is at least as accurate (vs an
    unquantized reference) as when it group-quantizes the decoded BF16 rows itself."""
    if not te.ops.fused.GroupedMLP_CuTeGEMMGLU.is_supported():
        pytest.skip("Fused grouped MLP (CuTeDSL) is not supported on this system")
    group_size, hidden = 4, 512
    split_sizes, total = _split_sizes(group_size)
    packed = _pack(_tokens(total, hidden, seed=3))
    x_dec = _unpack(packed, hidden)
    probs = torch.rand((total,), dtype=torch.bfloat16, device="cuda")
    dy = torch.rand((total, hidden), dtype=torch.bfloat16, device="cuda") - 0.5
    recipe = NVFP4BlockScaling(disable_stochastic_rounding=True)
    module, fc1, fc2 = _grouped_mlp(group_size, hidden, recipe, weight_requires_grad)

    def _run(x, quantized=True):
        with te.autocast(enabled=quantized, recipe=recipe):
            y = module(x, split_sizes, probs, split_sizes)
        grads = []
        if weight_requires_grad:
            y.backward(dy)
            for lin in (fc1, fc2):
                for idx in range(group_size):
                    w = getattr(lin, f"weight{idx}")
                    grads.append(w.grad.detach().float().clone())
                    w.grad = None
        return y.detach().float(), grads

    y_dec, g_dec = _run(x_dec)
    y_dir, g_dir = _run(_receive(packed, hidden, split_sizes))
    forward_ops = module._module_groups[0]._forward_ops
    assert isinstance(forward_ops[0][0], te.ops.fused.GroupedMLP_CuTeGEMMGLU)

    # Both runs share weights and all other quantization; they differ only in how the FC1 input
    # is quantized (direct: the sender's codes and a bound-scaled RHT operand; decode path: TE
    # requantizes the decoded BF16 rows). Judge each against the same module run unquantized on
    # the decoded rows, and require the direct path to be no less accurate.
    y_ref, g_ref = _run(x_dec, quantized=False)
    names = ["y"] + [f"fc{i}.w{j}" for i in (1, 2) for j in range(group_size)]
    err_dir = {"y": _rel(y_dir, y_ref)}
    err_dec = {"y": _rel(y_dec, y_ref)}
    err_dir.update({n: _rel(a, b) for n, a, b in zip(names[1:], g_dir, g_ref)})
    err_dec.update({n: _rel(a, b) for n, a, b in zip(names[1:], g_dec, g_ref)})
    print("rel err vs unquantized: direct", {k: round(v, 4) for k, v in err_dir.items()})
    print("rel err vs unquantized: decode", {k: round(v, 4) for k, v in err_dec.items()})
    for n in err_dir:
        assert err_dir[n] <= 1.1 * err_dec[n] + 1e-3, (n, err_dir[n], err_dec[n])


class _InjectGrad(torch.autograd.Function):
    """Identity whose backward returns a prepared grad (the combine backward's GroupedTensor)."""

    @staticmethod
    def forward(ctx, y, holder):  # type: ignore[override]
        ctx.holder = holder
        return y.view_as(y)

    @staticmethod
    def backward(ctx, _g):  # type: ignore[override]
        return ctx.holder["grad"], None


@pytest.mark.parametrize("weight_requires_grad", (False, True))
def test_fused_grouped_mlp_backward_with_direct_grad(weight_requires_grad):
    """The fused grouped MLP backward fed FC2's grad_output as the stochastically rounded
    direct-receive GroupedTensor is as accurate (vs an unquantized reference) as when it
    SR-quantizes the BF16 grad itself, except for FC2's wgrad, whose doubly rounded columnwise
    operand adds variance."""
    if not te.ops.fused.GroupedMLP_CuTeGEMMGLU.is_supported():
        pytest.skip("Fused grouped MLP (CuTeDSL) is not supported on this system")
    group_size, hidden = 4, 512
    split_sizes, total = _split_sizes(group_size)
    x = _tokens(total, hidden, seed=3, spread=(-0.3, 0.3))
    probs = torch.rand((total,), dtype=torch.bfloat16, device="cuda")
    dy = _tokens(total, hidden, seed=4, spread=(-0.3, 0.3))
    recipe = NVFP4BlockScaling()
    module, fc1, fc2 = _grouped_mlp(group_size, hidden, recipe, weight_requires_grad)

    def _run(direct, quantized=True, seed=0):
        torch.manual_seed(seed)
        xin = x.detach().clone().requires_grad_(True)
        with te.autocast(enabled=quantized, recipe=recipe):
            y = module(xin, split_sizes, probs, split_sizes)
        if direct:
            grad = _receive(
                _pack(dy, stochastic=True, seed=seed + 1), hidden, split_sizes, stochastic=True
            )
            _InjectGrad.apply(y, {"grad": grad}).backward(torch.zeros_like(y))
        else:
            y.backward(dy)
        grads = [xin.grad.detach().float().clone()]
        if weight_requires_grad:
            for lin in (fc1, fc2):
                for idx in range(group_size):
                    w = getattr(lin, f"weight{idx}")
                    grads.append(w.grad.detach().float().clone())
                    w.grad = None
        return grads

    ref = _run(False, quantized=False)
    names = ["dx"] + [f"fc{i}.w{j}" for i in (1, 2) for j in range(group_size)]
    names = names[: len(ref)]
    # Average errors over a few SR draws to compare the two noisy paths fairly.
    err_dir = {n: 0.0 for n in names}
    err_te = {n: 0.0 for n in names}
    draws = 3
    for s in range(draws):
        gd = _run(True, seed=10 + s)
        gt = _run(False, seed=10 + s)
        for n, a, b, r in zip(names, gd, gt, ref):
            assert torch.isfinite(a).all(), n
            err_dir[n] += _rel(a, r) / draws
            err_te[n] += _rel(b, r) / draws
    forward_ops = module._module_groups[0]._forward_ops
    assert isinstance(forward_ops[0][0], te.ops.fused.GroupedMLP_CuTeGEMMGLU)
    print("rel err vs unquantized: direct", {k: round(v, 4) for k, v in err_dir.items()})
    print("rel err vs unquantized: te    ", {k: round(v, 4) for k, v in err_te.items()})
    for n in names:
        # FC2's wgrad reads the columnwise operand, which is rounded twice (sender SR, then RHT +
        # SR on the receiver) where TE rounds once: both unbiased, but the variances add up.
        # dgrad and FC1's wgrad see a single rounding.
        factor = 1.25 if n.startswith("fc2") else 1.05
        assert err_dir[n] <= factor * err_te[n] + 2e-3, (n, err_dir[n], err_te[n])
