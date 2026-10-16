# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""cuDNN OSS through the real TE common-library backward dispatcher."""

import ctypes
import os

import pytest
import torch


@pytest.fixture
def frost_enabled(monkeypatch):
    """Require native execution instead of accepting a silent backend fallback."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("cuDNN OSS integration currently requires SM100")
    monkeypatch.setenv("CUDNN_FRONTEND_ENABLE_FROST_ENGINES", "1")
    pytest.importorskip("cudnn")
    pytest.importorskip("tvm_ffi")
    from transformer_engine.common import (
        _get_shared_object_file,
        _load_tvm_ffi_library,
        _register_cutedsl_backends,
    )

    assert _load_tvm_ffi_library()
    assert _register_cutedsl_backends()

    core = ctypes.CDLL(str(_get_shared_object_file("core")))
    setter = core.nvte_set_cutedsl_backend
    setter.argtypes, setter.restype = [ctypes.c_int], None
    setter(1)
    monkeypatch.setenv("NVTE_FUSED_ATTN_FROST", "2")
    monkeypatch.setenv("NVTE_FUSED_ATTN_FROST_UNIFORM", "1")
    yield
    setter(int(os.environ.get("NVTE_ENABLE_CUTEDSL_BACKEND", "0") != "0"))


def _case(dim, thd, unequal, causal, seed=31):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    q_lengths, k_lengths = ((129, 63), (193, 97)) if unequal else ((128, 128), (256, 256))
    heads, kv_heads = 4, 1
    inputs, expected = [], []
    for sq, sk in zip(q_lengths, k_lengths):

        def draw(s, h):
            return (torch.randn(s, h, dim, generator=generator) * 0.5).bfloat16()

        q, k, v, do = draw(sq, heads), draw(sk, kv_heads), draw(sk, kv_heads), draw(sq, heads)
        qr, kr, vr = (x.double().transpose(0, 1).requires_grad_() for x in (q, k, v))
        scores = (qr @ kr.repeat_interleave(heads // kv_heads, 0).transpose(-1, -2)) * dim**-0.5
        if causal:
            scores = scores.masked_fill(
                torch.arange(sk)[None, :] > torch.arange(sq)[:, None], -torch.inf
            )
        result = scores.softmax(-1) @ vr.repeat_interleave(heads // kv_heads, 0)
        grads = torch.autograd.grad(result, (qr, kr, vr), do.transpose(0, 1).double())
        inputs.append((q, k, v, do))
        expected.append(tuple(x.transpose(0, 1).detach() for x in grads))
    combine = torch.cat if thd else torch.stack
    q, k, v, do = (combine([row[i] for row in inputs]).cuda() for i in range(4))
    reference = tuple(combine([row[i] for row in expected]) for i in range(3))
    prefixes = [
        torch.tensor([0, lengths[0], sum(lengths)], device="cuda", dtype=torch.int32)
        for lengths in (q_lengths, k_lengths)
    ]
    return q, k, v, do, prefixes, reference, max(q_lengths), max(k_lengths)


def _calls(case, dim, thd, causal):
    from transformer_engine.pytorch.cpp_extensions.fused_attn import (
        FusedAttnBackend,
        fused_attn_fwd,
        fused_attn_bwd,
    )

    q, k, v, do, (cu_q, cu_kv), reference, sq, sk = case
    fmt = "thd" if thd else "bshd"
    mask = (
        ("padding_causal" if causal else "padding") if thd else ("causal" if causal else "no_mask")
    )
    options = dict(
        max_seqlen_q=sq,
        max_seqlen_kv=sk,
        cu_seqlens_q=cu_q,
        cu_seqlens_kv=cu_kv,
        q=q,
        k=k,
        v=v,
        fake_dtype=torch.bfloat16,
        fused_attention_backend=FusedAttnBackend.F16_arbitrary_seqlen,
        cu_seqlens_q_padded=cu_q,
        cu_seqlens_kv_padded=cu_kv,
        attn_scale=dim**-0.5,
        qkv_layout="_".join([fmt] * 3),
        o_format=fmt,
        attn_mask_type=mask,
        dropout=0.0,
    )
    o, aux = fused_attn_fwd(is_training=True, **options)

    def backward():
        return fused_attn_bwd(
            o=o,
            d_o=do,
            aux_ctx_tensors=aux,
            do_format=fmt,
            dqkv_layout=options["qkv_layout"],
            deterministic=False,
            **options
        )[:3]

    return backward, reference


def _check(actual, reference):
    for got, want in zip(actual, reference):
        torch.testing.assert_close(got.double().cpu(), want, atol=0.025, rtol=0.025)
        error = (got.double().cpu() - want).square().mean().sqrt()
        assert error / want.square().mean().sqrt().clamp_min(1e-8) < 0.02


@pytest.mark.parametrize(
    "dim,thd,unequal",
    [(128, False, False), (128, True, True), (256, False, False), (256, True, False)],
)
@pytest.mark.parametrize("causal", [False, True])
def test_frost_backward_and_rebinding(frost_enabled, dim, thd, unequal, causal):
    """New pointers and changed inputs reuse the cached native plan."""
    del frost_enabled
    for seed in (31, 47):
        backward, reference = _calls(_case(dim, thd, unequal, causal, seed), dim, thd, causal)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            result = backward()
        torch.cuda.current_stream().wait_stream(stream)
        _check(result, reference)


@pytest.mark.parametrize("dim,thd", [(128, True), (256, True)])
def test_frost_backward_replay(frost_enabled, dim, thd):
    """Repeated replay overwrites poisoned gradients and reinitializes scratch."""
    del frost_enabled
    backward, reference = _calls(_case(dim, thd, False, True), dim, thd, True)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        _ = backward()
    torch.cuda.current_stream().wait_stream(stream)
    # Removing preparation must not affect cached C++ dispatch or graph capture.
    import tvm_ffi

    def unexpected_python(*args, **kwargs):
        raise AssertionError("Steady-state backward re-entered Python preparation")

    original = tvm_ffi.get_global_func("get_frost_attention_backward")
    tvm_ffi.register_global_func("get_frost_attention_backward", unexpected_python, override=True)
    try:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            result = backward()
        for _ in range(3):
            for value in result:
                value.fill_(float("nan"))
            graph.replay()
            _check(result, reference)
    finally:
        tvm_ffi.register_global_func("get_frost_attention_backward", original, override=True)


def test_frost_d256_ragged_fallback(frost_enabled, monkeypatch):
    """D256 unequal THD uses the existing backend unless strict mode is requested."""
    del frost_enabled
    monkeypatch.setenv("NVTE_FUSED_ATTN_FROST_UNIFORM", "0")
    monkeypatch.setenv("NVTE_FUSED_ATTN_FROST", "1")
    backward, reference = _calls(_case(256, True, True, True), 256, True, True)
    _check(backward(), reference)
    monkeypatch.setenv("NVTE_FUSED_ATTN_FROST", "2")
    with pytest.raises(RuntimeError, match="Requested cuDNN OSS backward unavailable"):
        backward()
