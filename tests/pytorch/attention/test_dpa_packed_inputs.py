# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Tests for declarative packed QKV/KV inputs to DotProductAttention.

Instead of slicing a fused-projection buffer into q/k/v views (which TE then
reverse-engineers via pointer-based layout detection), callers can pass the
packed tensor directly (``qkv_layer``/``kv_layer`` + ``qkv_interleave_dim``).
Q/K/V are derived as zero-copy views and the exact layout string (e.g.
``bs3hd``) is declared, not detected -- including for thd and FP8 DPA.
"""

import pytest
import torch

import transformer_engine.pytorch  # noqa: F401  (loads libtransformer_engine.so)
import transformer_engine_torch as tex
from transformer_engine.pytorch import DotProductAttention, MultiheadAttention
from transformer_engine.pytorch.attention.dot_product_attention import (
    dot_product_attention as dpa_module,
)
import transformer_engine.pytorch.attention.dot_product_attention.utils as dpa_utils
from transformer_engine.pytorch.attention.dot_product_attention.utils import (
    combine_and_quantize,
)
from transformer_engine.pytorch.cpp_extensions.fused_attn import (
    fused_attn_fwd,
)
from transformer_engine.pytorch.tensor.float8_tensor import Float8Quantizer

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")

_B, _S, _H, _D = 2, 128, 8, 64
_DTYPE = torch.bfloat16


def _cu_seqlens():
    return torch.arange(0, (_B + 1) * _S, _S, dtype=torch.int32, device="cuda")


def _fused_backend_supported():
    try:
        q = torch.randn(_B, _S, _H, _D, dtype=_DTYPE, device="cuda")
        fused_attn_fwd(
            True,
            _S,
            _S,
            _cu_seqlens(),
            _cu_seqlens(),
            q,
            q.clone(),
            q.clone(),
            _DTYPE,
            tex.NVTE_Fused_Attn_Backend.NVTE_F16_arbitrary_seqlen,
            dropout=0.0,
            qkv_layout="bshd_bshd_bshd",
            o_format="bshd",
            attn_bias_type="no_bias",
            attn_mask_type="no_mask",
        )
        return True
    except Exception:
        return False


requires_fused = pytest.mark.skipif(
    not (torch.cuda.is_available() and _fused_backend_supported()),
    reason="F16_arbitrary_seqlen fused attention backend is not supported on this device",
)


def _force_backend(monkeypatch, backend):
    """Force a single attention backend via env and invalidate the selection cache."""
    flash, fused = {"flash": ("1", "0"), "fused": ("0", "1")}[backend]
    monkeypatch.setenv("NVTE_FLASH_ATTN", flash)
    monkeypatch.setenv("NVTE_FUSED_ATTN", fused)
    monkeypatch.setenv("NVTE_UNFUSED_ATTN", "0")
    if backend == "flash":
        # flash-attn bwd uses atomics unless deterministic
        monkeypatch.setenv("NVTE_ALLOW_NONDETERMINISTIC_ALGO", "0")
    dpa_module._attention_backends["backend_selection_requires_update"] = True


def _make_dpa(qkv_format, num_gqa_groups=None):
    return DotProductAttention(
        _H,
        _D,
        num_gqa_groups=num_gqa_groups,
        attention_dropout=0.0,
        qkv_format=qkv_format,
        attn_mask_type="no_mask",
    )


def _assert_bit_exact(result, reference, names=("out", "dq", "dk", "dv")):
    for name, x, y in zip(names, result, reference):
        assert torch.equal(x.contiguous(), y.contiguous()), f"{name} differs"


def _fresh_parts(qkv_format, num_heads, seed=0):
    torch.manual_seed(seed)
    shape = (_B, _S, num_heads, _D) if qkv_format == "bshd" else (_S, _B, num_heads, _D)
    return [torch.randn(*shape, dtype=_DTYPE, device="cuda") for _ in range(3)]


def _dpa_separate_baseline(qkv_format, num_gqa_groups=None):
    """Fwd+bwd on contiguous separate q/k/v leaves; returns (out, dq, dk, dv)."""
    hg = num_gqa_groups or _H
    torch.manual_seed(0)
    q_shape = (_B, _S, _H, _D) if qkv_format == "bshd" else (_S, _B, _H, _D)
    kv_shape = (_B, _S, hg, _D) if qkv_format == "bshd" else (_S, _B, hg, _D)
    q = torch.randn(*q_shape, dtype=_DTYPE, device="cuda")
    k = torch.randn(*kv_shape, dtype=_DTYPE, device="cuda")
    v = torch.randn(*kv_shape, dtype=_DTYPE, device="cuda")
    q, k, v = [x.clone().requires_grad_() for x in (q, k, v)]
    dpa_module._attention_backends["backend_selection_requires_update"] = True
    out = _make_dpa(qkv_format, num_gqa_groups)(q, k, v)
    out.backward(torch.ones_like(out))
    return out, q.grad, k.grad, v.grad


# ---------------------------------------------------------------------------
# 1. Dense eager equivalence (fused backend), fwd + input grads, bit-exact
# ---------------------------------------------------------------------------


@requires_fused
@pytest.mark.parametrize(
    "qkv_format, interleave_dim",
    [
        pytest.param("bshd", -3, id="bshd_qkv_dim-3"),  # (a) bs3hd
        pytest.param("bshd", -2, id="bshd_qkv_dim-2"),  # (b) bsh3d (Megatron-style)
        pytest.param("sbhd", -3, id="sbhd_qkv_dim-3"),  # (c) sb3hd
    ],
)
def test_dpa_fused_qkv_layer_dense(monkeypatch, qkv_format, interleave_dim):
    """qkv_layer packed input is bit-exact vs separate contiguous q/k/v, and grads
    flow back into the packed tensor itself."""
    _force_backend(monkeypatch, "fused")
    reference = _dpa_separate_baseline(qkv_format)

    torch.manual_seed(0)
    q_shape = (_B, _S, _H, _D) if qkv_format == "bshd" else (_S, _B, _H, _D)
    parts = [torch.randn(*q_shape, dtype=_DTYPE, device="cuda") for _ in range(3)]
    stack_dim = len(q_shape) + interleave_dim + 1  # -3 -> before h, -2 -> before d
    qkv = torch.stack(parts, dim=stack_dim).requires_grad_()

    dpa_module._attention_backends["backend_selection_requires_update"] = True
    out = _make_dpa(qkv_format)(qkv_layer=qkv, qkv_interleave_dim=interleave_dim)
    out.backward(torch.ones_like(out))

    assert qkv.grad is not None and qkv.grad.shape == qkv.shape
    grads = [qkv.grad.select(stack_dim, i) for i in range(3)]
    _assert_bit_exact((out, *grads), reference)


@requires_fused
@pytest.mark.parametrize(
    "num_gqa_groups",
    [pytest.param(None, id="mha_kv"), pytest.param(2, id="gqa_kv")],  # (d) and (e)
)
def test_dpa_fused_kv_layer_dense(monkeypatch, num_gqa_groups):
    """kv_layer packed input (with separate query) is bit-exact vs separate
    contiguous q/k/v; grads flow into the packed kv tensor."""
    _force_backend(monkeypatch, "fused")
    reference = _dpa_separate_baseline("bshd", num_gqa_groups)

    hg = num_gqa_groups or _H
    torch.manual_seed(0)
    q = torch.randn(_B, _S, _H, _D, dtype=_DTYPE, device="cuda")
    k = torch.randn(_B, _S, hg, _D, dtype=_DTYPE, device="cuda")
    v = torch.randn(_B, _S, hg, _D, dtype=_DTYPE, device="cuda")
    q = q.clone().requires_grad_()
    kv = torch.stack([k, v], dim=2).requires_grad_()  # [b,s,2,hg,d]

    dpa_module._attention_backends["backend_selection_requires_update"] = True
    out = _make_dpa("bshd", num_gqa_groups)(query_layer=q, kv_layer=kv)
    out.backward(torch.ones_like(out))

    assert kv.grad is not None and kv.grad.shape == kv.shape
    _assert_bit_exact((out, q.grad, kv.grad[:, :, 0], kv.grad[:, :, 1]), reference)


# ---------------------------------------------------------------------------
# 2. Flash backend smoke
# ---------------------------------------------------------------------------


def test_dpa_flash_qkv_layer(monkeypatch):
    """Flash backend: packed qkv_layer [b,s,3,h,d] is bit-exact vs separate."""
    _force_backend(monkeypatch, "flash")
    try:
        reference = _dpa_separate_baseline("bshd")
    except Exception as exc:
        pytest.skip(f"flash attention backend not available: {exc}")

    torch.manual_seed(0)
    parts = [torch.randn(_B, _S, _H, _D, dtype=_DTYPE, device="cuda") for _ in range(3)]
    qkv = torch.stack(parts, dim=2).requires_grad_()
    dpa_module._attention_backends["backend_selection_requires_update"] = True
    out = _make_dpa("bshd")(qkv_layer=qkv)
    out.backward(torch.ones_like(out))
    grads = [qkv.grad[:, :, i] for i in range(3)]
    _assert_bit_exact((out, *grads), reference)


# ---------------------------------------------------------------------------
# 3. Validation errors
# ---------------------------------------------------------------------------


def test_dpa_packed_input_validation():
    dpa = _make_dpa("bshd")
    qkv = torch.randn(_B, _S, 3, _H, _D, dtype=_DTYPE, device="cuda")
    kv = torch.randn(_B, _S, 2, _H, _D, dtype=_DTYPE, device="cuda")
    k = torch.randn(_B, _S, _H, _D, dtype=_DTYPE, device="cuda")

    with pytest.raises(ValueError, match="must be None when qkv_layer is provided"):
        dpa(qkv_layer=qkv, key_layer=k)
    with pytest.raises(ValueError, match="query_layer is required when kv_layer"):
        dpa(kv_layer=kv)
    with pytest.raises(ValueError, match="qkv_interleave_dim must be -3"):
        dpa(qkv_layer=qkv, qkv_interleave_dim=-1)
    with pytest.raises(ValueError, match="mutually exclusive"):
        dpa(qkv_layer=qkv, kv_layer=kv)
    with pytest.raises(ValueError, match="must have size 3 at dim"):
        dpa(qkv_layer=kv)  # 2 at the interleave dim, not 3
    with pytest.raises(ValueError, match="stride 1 in its last"):
        dpa(qkv_layer=qkv.transpose(-2, -1))  # declared layout would lie about memory
    with pytest.raises(ValueError, match="required unless packed"):
        dpa()


# ---------------------------------------------------------------------------
# 4. torch.compile: no data_ptr/UntypedStorage graph breaks with qkv_layer
# ---------------------------------------------------------------------------


@requires_fused
def test_dpa_torch_compile_qkv_layer_no_pointer_graph_breaks(monkeypatch):
    _force_backend(monkeypatch, "fused")
    torch._dynamo.reset()
    torch._dynamo.utils.counters.clear()

    torch.manual_seed(0)
    parts = [torch.randn(_B, _S, _H, _D, dtype=_DTYPE, device="cuda") for _ in range(3)]
    qkv = torch.stack(parts, dim=2).requires_grad_()
    dpa = _make_dpa("bshd")

    def fn(x):
        return dpa(qkv_layer=x)

    dpa_module._attention_backends["backend_selection_requires_update"] = True
    eager_out = fn(qkv)  # eager warm-up: backend selection happens outside dynamo
    compiled_out = torch.compile(fn)(qkv)
    compiled_out.backward(torch.ones_like(compiled_out))

    breaks = dict(torch._dynamo.utils.counters["graph_break"])
    torch._dynamo.reset()
    pointer_breaks = {
        reason: count
        for reason, count in breaks.items()
        if "data_ptr" in reason or "UntypedStorage" in reason
    }
    assert not pointer_breaks, f"pointer-based graph breaks with qkv_layer: {pointer_breaks}"
    assert torch.equal(compiled_out, eager_out), "compiled output differs from eager"


# ---------------------------------------------------------------------------
# 5. FP8 combine refactor: combined= path is bit-identical to combine_tensors path
# ---------------------------------------------------------------------------


def _fp8_quantizer():
    return Float8Quantizer(
        scale=torch.ones(1, dtype=torch.float32, device="cuda"),
        amax=torch.zeros(1, dtype=torch.float32, device="cuda"),
        fp8_dtype=tex.DType.kFloat8E4M3,
    )


def test_combine_and_quantize_combined_matches_views():
    """Quantizing the caller's packed buffer directly (combined=) produces the same
    _data bits and scale_inv as rebuilding the packed buffer from q/k/v views via
    combine_tensors (the old set_-based path)."""
    torch.manual_seed(0)
    qkv = torch.randn(_B, _S, 3, _H, _D, dtype=_DTYPE, device="cuda")
    q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]

    old = combine_and_quantize("bs3hd", q, k, v, _fp8_quantizer())
    new = combine_and_quantize("bs3hd", q, k, v, _fp8_quantizer(), combined=qkv)

    assert old[3] == new[3] == "bs3hd"
    for name, x, y in zip(("q", "k", "v"), old[:3], new[:3]):
        assert torch.equal(x._data, y._data), f"{name} fp8 bits differ"
        assert torch.equal(x._scale_inv, y._scale_inv), f"{name} scale_inv differs"


def test_combine_and_quantize_combined_kv_matches_views():
    """Same for the kv-packed (group 2) layout."""
    torch.manual_seed(0)
    q = torch.randn(_B, _S, _H, _D, dtype=_DTYPE, device="cuda")
    kv = torch.randn(_B, _S, 2, _H, _D, dtype=_DTYPE, device="cuda")
    k, v = kv[:, :, 0], kv[:, :, 1]

    old = combine_and_quantize("bshd_bs2hd", q, k, v, _fp8_quantizer())
    new = combine_and_quantize("bshd_bs2hd", q, k, v, _fp8_quantizer(), combined=kv)

    for name, x, y in zip(("q", "k", "v"), old[:3], new[:3]):
        assert torch.equal(x._data, y._data), f"{name} fp8 bits differ"
        assert torch.equal(x._scale_inv, y._scale_inv), f"{name} scale_inv differs"


# ---------------------------------------------------------------------------
# 6. thd declarative: t3hd is declared, get_qkv_layout is never called
# ---------------------------------------------------------------------------


def test_dpa_thd_qkv_layer_declared_no_detection(monkeypatch):
    """Packed thd input (qkv_layer [t,3,h,d]) declares 't3hd' without calling
    get_qkv_layout; full forward+backward runs if a thd backend is available."""
    calls = []
    orig_get_qkv_layout = dpa_utils.get_qkv_layout

    def counting(*args, **kwargs):
        calls.append(kwargs.get("qkv_format"))
        return orig_get_qkv_layout(*args, **kwargs)

    monkeypatch.setattr(dpa_utils, "get_qkv_layout", counting)

    seen_layouts = []
    orig_get_backend = dpa_utils.get_attention_backend

    def recording(params):
        seen_layouts.append(params.qkv_layout)
        return orig_get_backend(params)

    monkeypatch.setattr(dpa_utils, "get_attention_backend", recording)

    torch.manual_seed(0)
    t = _B * _S
    qkv = torch.randn(t, 3, _H, _D, dtype=_DTYPE, device="cuda", requires_grad=True)
    cu = _cu_seqlens()
    dpa = DotProductAttention(
        _H, _D, attention_dropout=0.0, qkv_format="thd", attn_mask_type="padding"
    )
    dpa_module._attention_backends["backend_selection_requires_update"] = True
    ran_full = True
    try:
        out = dpa(
            qkv_layer=qkv,
            cu_seqlens_q=cu,
            cu_seqlens_kv=cu,
            max_seqlen_q=_S,
            max_seqlen_kv=_S,
        )
        out.backward(torch.ones_like(out))
        assert qkv.grad is not None and qkv.grad.shape == qkv.shape
    except ValueError:
        # No thd-capable backend on this device; the layout step (the subject of
        # this test) runs before backend dispatch, so the assertions still hold.
        ran_full = False

    assert not calls, "get_qkv_layout must not be called for declared packed thd input"
    assert seen_layouts == ["t3hd"], f"expected declared layout 't3hd', got {seen_layouts}"
    if not ran_full:
        pytest.skip("layout declaration verified; no thd backend available for full run")


# ---------------------------------------------------------------------------
# 7. MultiheadAttention adoption: the fused QKV/KV projection output is passed
#    packed (qkv_layer/kv_layer) to DotProductAttention
# ---------------------------------------------------------------------------


def _spy_packed_dpa(monkeypatch, record):
    """Record (qkv_layer given, kv_layer given, qkv_interleave_dim) per DPA call."""
    orig = DotProductAttention.forward

    def spy(self, *args, **kwargs):
        record.append(
            (
                kwargs.get("qkv_layer") is not None,
                kwargs.get("kv_layer") is not None,
                kwargs.get("qkv_interleave_dim", None),
            )
        )
        return orig(self, *args, **kwargs)

    monkeypatch.setattr(DotProductAttention, "forward", spy)


def _strip_packed_dpa(monkeypatch):
    """Reference path: convert packed DPA inputs back to separate contiguous q/k/v."""
    orig = DotProductAttention.forward

    def stripped(self, query_layer=None, key_layer=None, value_layer=None, *args, **kwargs):
        qkv = kwargs.pop("qkv_layer", None)
        kv = kwargs.pop("kv_layer", None)
        dim = kwargs.pop("qkv_interleave_dim", -3)
        if qkv is not None:
            query_layer, key_layer, value_layer = (
                qkv.select(dim, i).contiguous() for i in range(3)
            )
        elif kv is not None:
            key_layer, value_layer = (kv.select(dim, i).contiguous() for i in range(2))
        return orig(self, query_layer, key_layer, value_layer, *args, **kwargs)

    monkeypatch.setattr(DotProductAttention, "forward", stripped)


def _run_mha(mha, x, encoder_output=None):
    dpa_module._attention_backends["backend_selection_requires_update"] = True
    if encoder_output is not None:
        out = mha(x, encoder_output=encoder_output)
    else:
        out = mha(x)
    out.backward(torch.ones_like(out))
    wgrads = [p.grad.clone() for p in mha.parameters() if p.grad is not None]
    xgrad = x.grad.clone()
    x.grad = None
    mha.zero_grad(set_to_none=True)
    return out, xgrad, wgrads


def _assert_mha_equal(result, reference):
    out, xgrad, wgrads = result
    out_ref, xgrad_ref, wgrads_ref = reference
    assert torch.equal(out, out_ref), "output differs"
    assert torch.equal(xgrad, xgrad_ref), "input grad differs"
    assert len(wgrads) == len(wgrads_ref)
    for i, (w, w_ref) in enumerate(zip(wgrads, wgrads_ref)):
        assert torch.equal(w, w_ref), f"weight grad {i} differs"


@requires_fused
@pytest.mark.parametrize(
    "interleaved", [pytest.param(True, id="interleaved"), pytest.param(False, id="non_interleaved")]
)
def test_mha_self_attention_packed_pass_through(monkeypatch, interleaved):
    """MHA self-attention passes its packed projection output straight to DPA as
    qkv_layer (with the matching interleave dim), bit-exact vs the same MHA with
    packed inputs converted back to separate contiguous q/k/v."""
    _force_backend(monkeypatch, "fused")
    hidden = _H * _D
    torch.manual_seed(0)
    mha = MultiheadAttention(
        hidden,
        _H,
        attention_dropout=0.0,
        attn_mask_type="no_mask",
        qkv_format="sbhd",
        fuse_qkv_params=True,
        qkv_weight_interleaved=interleaved,
        params_dtype=_DTYPE,
        device="cuda",
    )
    torch.manual_seed(1)
    x = torch.randn(_S, _B, hidden, dtype=_DTYPE, device="cuda", requires_grad=True)

    record = []
    _spy_packed_dpa(monkeypatch, record)
    result = _run_mha(mha, x)
    assert record == [(True, False, -2 if interleaved else -3)], f"unexpected DPA call: {record}"
    monkeypatch.undo()

    _force_backend(monkeypatch, "fused")
    _strip_packed_dpa(monkeypatch)
    reference = _run_mha(mha, x)
    _assert_mha_equal(result, reference)


@requires_fused
@pytest.mark.parametrize(
    "interleaved", [pytest.param(True, id="interleaved"), pytest.param(False, id="non_interleaved")]
)
def test_mha_cross_attention_packed_kv_pass_through(monkeypatch, interleaved):
    """MHA cross-attention passes its packed KV projection output to DPA as
    kv_layer, bit-exact vs the separate contiguous reference."""
    _force_backend(monkeypatch, "fused")
    hidden = _H * _D
    torch.manual_seed(0)
    mha = MultiheadAttention(
        hidden,
        _H,
        attention_dropout=0.0,
        attn_mask_type="no_mask",
        qkv_format="sbhd",
        attention_type="cross",
        fuse_qkv_params=True,
        qkv_weight_interleaved=interleaved,
        params_dtype=_DTYPE,
        device="cuda",
    )
    torch.manual_seed(1)
    x = torch.randn(_S, _B, hidden, dtype=_DTYPE, device="cuda", requires_grad=True)
    enc = torch.randn(_S, _B, hidden, dtype=_DTYPE, device="cuda")

    record = []
    _spy_packed_dpa(monkeypatch, record)
    result = _run_mha(mha, x, encoder_output=enc)
    assert record == [(False, True, -2 if interleaved else -3)], f"unexpected DPA call: {record}"
    monkeypatch.undo()

    _force_backend(monkeypatch, "fused")
    _strip_packed_dpa(monkeypatch)
    reference = _run_mha(mha, x, encoder_output=enc)
    _assert_mha_equal(result, reference)


@requires_fused
def test_mha_gqa_falls_back_to_views(monkeypatch):
    """GQA (np != ng) is not a uniform 3-interleave: MHA must keep the legacy
    sliced-views path and still work."""
    _force_backend(monkeypatch, "fused")
    hidden = _H * _D
    torch.manual_seed(0)
    mha = MultiheadAttention(
        hidden,
        _H,
        num_gqa_groups=2,
        attention_dropout=0.0,
        attn_mask_type="no_mask",
        qkv_format="sbhd",
        fuse_qkv_params=True,
        params_dtype=_DTYPE,
        device="cuda",
    )
    torch.manual_seed(1)
    x = torch.randn(_S, _B, hidden, dtype=_DTYPE, device="cuda", requires_grad=True)

    record = []
    _spy_packed_dpa(monkeypatch, record)
    out, _, _ = _run_mha(mha, x)
    assert record == [(False, False, -3)], f"GQA must not use the packed path: {record}"
    assert out.shape == (_S, _B, hidden)


@requires_fused
def test_mha_rope_falls_back_to_views(monkeypatch):
    """RoPE needs the individual q/k slices: MHA must keep the legacy path."""
    _force_backend(monkeypatch, "fused")
    from transformer_engine.pytorch.attention.rope import RotaryPositionEmbedding

    hidden = _H * _D
    torch.manual_seed(0)
    mha = MultiheadAttention(
        hidden,
        _H,
        attention_dropout=0.0,
        attn_mask_type="no_mask",
        qkv_format="sbhd",
        fuse_qkv_params=True,
        params_dtype=_DTYPE,
        device="cuda",
    )
    rope = RotaryPositionEmbedding(_D)(max_seq_len=_S).to("cuda")
    torch.manual_seed(1)
    x = torch.randn(_S, _B, hidden, dtype=_DTYPE, device="cuda", requires_grad=True)

    record = []
    _spy_packed_dpa(monkeypatch, record)
    dpa_module._attention_backends["backend_selection_requires_update"] = True
    out = mha(x, rotary_pos_emb=rope)
    assert record == [(False, False, -3)], f"RoPE must not use the packed path: {record}"
    assert out.shape == (_S, _B, hidden)
