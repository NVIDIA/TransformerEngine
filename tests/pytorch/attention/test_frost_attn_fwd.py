# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Actual TE common-library FROST forward, including training Stats and replay."""

import ctypes
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from test_frost_attn import frost_enabled  # noqa: F401


@pytest.fixture
def forward_enabled(frost_enabled, monkeypatch):
    del frost_enabled
    monkeypatch.setenv("NVTE_FUSED_ATTN_FROST_FWD", "2")
    monkeypatch.setenv("NVTE_FUSED_ATTN_FROST_FWD_COMPACT", "1")


def _case(heads=4, kv_heads=1, seed=51):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    q, k, v, do = [
        (torch.randn(512, h, 128, generator=gen) * 0.5).bfloat16().cuda()
        for h in (heads, kv_heads, kv_heads, heads)
    ]
    cuq = torch.tensor([0, 129, 192], device="cuda", dtype=torch.int32)
    cuk = torch.tensor([0, 193, 290], device="cuda", dtype=torch.int32)
    return q, k, v, do, cuq, cuk, cuq.clone(), cuk.clone()


def _forward(case):
    from transformer_engine.pytorch.cpp_extensions.fused_attn import (
        FusedAttnBackend,
        fused_attn_fwd,
    )

    q, k, v, _, cuq, cuk, oq, ok = case
    return fused_attn_fwd(
        is_training=True,
        max_seqlen_q=256,
        max_seqlen_kv=256,
        cu_seqlens_q=cuq,
        cu_seqlens_kv=cuk,
        q=q,
        k=k,
        v=v,
        fake_dtype=torch.bfloat16,
        fused_attention_backend=FusedAttnBackend.F16_arbitrary_seqlen,
        cu_seqlens_q_padded=oq,
        cu_seqlens_kv_padded=ok,
        attn_scale=128**-0.5,
        qkv_layout="thd_thd_thd",
        o_format="thd",
        attn_mask_type="padding_causal",
        dropout=0.0,
    )


def _reference(case):
    q, k, v, do, cuq, cuk, _, _ = [x.cpu() for x in case]
    outputs, stats, gradients = [], [], []
    for i in range(2):
        qs = slice(int(cuq[i]), int(cuq[i + 1]))
        ks = slice(int(cuk[i]), int(cuk[i + 1]))
        qr, kr, vr = [x.double().transpose(0, 1).requires_grad_() for x in (q[qs], k[ks], v[ks])]
        repeat = q.shape[1] // k.shape[1]
        score = qr @ kr.repeat_interleave(repeat, 0).transpose(-1, -2) * 128**-0.5
        mask = torch.arange(kr.shape[1])[None, :] > torch.arange(qr.shape[1])[:, None]
        score = score.masked_fill(mask, -torch.inf)
        o = score.softmax(-1) @ vr.repeat_interleave(repeat, 0)
        grads = torch.autograd.grad(o, (qr, kr, vr), do[qs].transpose(0, 1).double())
        outputs.append(o.transpose(0, 1).detach())
        stats.append(score.logsumexp(-1).transpose(0, 1).detach())
        gradients.append(tuple(g.transpose(0, 1).detach() for g in grads))
    return (
        torch.cat(outputs),
        torch.cat(stats),
        tuple(torch.cat([g[i] for g in gradients]) for i in range(3)),
    )


def _check_forward(result, expected):
    o, aux = result
    ref, lse, _ = expected
    torch.testing.assert_close(o[: len(ref)].double().cpu(), ref, atol=0.015, rtol=0.015)
    torch.testing.assert_close(aux[0][: len(lse), :, 0].double().cpu(), lse, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("heads,kv_heads", [(4, 1), (8, 2), (4, 4)])
@pytest.mark.parametrize("seed", [51, 73])
def test_frost_forward_training(forward_enabled, heads, kv_heads, seed):
    """Check O, natural-log packed Stats and gradients against CPU FP64."""
    del forward_enabled
    from transformer_engine.pytorch.cpp_extensions.fused_attn import (
        FusedAttnBackend,
        fused_attn_bwd,
    )

    case = _case(heads, kv_heads, seed)
    expected = _reference(case)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        result = _forward(case)
    torch.cuda.current_stream().wait_stream(stream)
    _check_forward(result, expected)
    q, k, v, do, cuq, cuk, oq, ok = case
    o, aux = result
    actual = fused_attn_bwd(
        max_seqlen_q=256,
        max_seqlen_kv=256,
        cu_seqlens_q=cuq,
        cu_seqlens_kv=cuk,
        q=q,
        k=k,
        v=v,
        o=o,
        d_o=do,
        fake_dtype=torch.bfloat16,
        fused_attention_backend=FusedAttnBackend.F16_arbitrary_seqlen,
        aux_ctx_tensors=aux,
        cu_seqlens_q_padded=oq,
        cu_seqlens_kv_padded=ok,
        attn_scale=128**-0.5,
        qkv_layout="thd_thd_thd",
        dqkv_layout="thd_thd_thd",
        o_format="thd",
        do_format="thd",
        attn_mask_type="padding_causal",
        dropout=0.0,
        deterministic=False,
    )[:3]
    for got, want in zip(actual, expected[2]):
        torch.testing.assert_close(got[: len(want)].double().cpu(), want, atol=0.025, rtol=0.025)
        relative = (
            got[: len(want)].double().cpu() - want
        ).square().mean().sqrt() / want.square().mean().sqrt().clamp_min(1e-8)
        assert relative < 0.02


@pytest.mark.parametrize("heads,kv_heads", [(4, 1), (8, 2), (4, 4)])
def test_frost_forward_native_replay(forward_enabled, heads, kv_heads):
    """No Python hot dispatch; graph reads changed lengths, origins and inputs."""
    del forward_enabled
    import tvm_ffi

    case = _case(heads, kv_heads)
    expected = _reference(case)
    _check_forward(_forward(case), expected)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        _forward(case)
    torch.cuda.current_stream().wait_stream(stream)
    names = ("get_frost_attention_forward", "get_frost_attention_forward_constant")
    originals = {name: tvm_ffi.get_global_func(name) for name in names}

    def unexpected_python(*args):
        raise AssertionError("Steady-state forward entered Python preparation")

    try:
        for name in names:
            tvm_ffi.register_global_func(name, unexpected_python, override=True)
        # A second allocation uses the same declaration with new pointers.
        other = _case(heads, kv_heads, 97)
        _check_forward(_forward(other), _reference(other))
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            result = _forward(case)
        for q_lengths, k_lengths in (
            ((128, 128), (128, 256)),
            ((0, 0), (0, 0)),
            ((63, 129), (97, 193)),
        ):
            for cu, origin, lengths in (
                (case[4], case[6], q_lengths),
                (case[5], case[7], k_lengths),
            ):
                cu.copy_(
                    torch.tensor([0, lengths[0], sum(lengths)], dtype=torch.int32, device="cuda")
                )
                origin.copy_(cu)
            for tensor in case[:3]:
                tensor.neg_()
            result[0].fill_(float("nan"))
            result[1][0].fill_(float("nan"))
            graph.replay()
            if sum(q_lengths):
                _check_forward(result, _reference(case))
            else:
                torch.cuda.synchronize()
    finally:
        for name, original in originals.items():
            tvm_ffi.register_global_func(name, original, override=True)


def test_frost_forward_compact_decline(forward_enabled, monkeypatch):
    """The compact-layout promise is required even after a plan has been cached."""
    del forward_enabled
    case = _case()
    expected = _reference(case)
    _check_forward(_forward(case), expected)
    monkeypatch.setenv("NVTE_FUSED_ATTN_FROST_FWD_COMPACT", "0")
    monkeypatch.setenv("NVTE_FUSED_ATTN_FROST_FWD", "1")
    _check_forward(_forward(case), expected)
    monkeypatch.setenv("NVTE_FUSED_ATTN_FROST_FWD", "2")
    with pytest.raises(RuntimeError, match="Requested cuDNN OSS forward unavailable"):
        _forward(case)


def _invalid_metadata(kind):
    from transformer_engine.common import (
        _get_shared_object_file,
        _load_tvm_ffi_library,
        _register_cutedsl_backends,
    )

    assert _load_tvm_ffi_library() and _register_cutedsl_backends()
    core = ctypes.CDLL(str(_get_shared_object_file("core")))
    core.nvte_set_cutedsl_backend.argtypes, core.nvte_set_cutedsl_backend.restype = [
        ctypes.c_int
    ], None
    core.nvte_set_cutedsl_backend(1)
    os.environ["NVTE_FUSED_ATTN_FROST_FWD"] = "2"
    os.environ["NVTE_FUSED_ATTN_FROST_FWD_COMPACT"] = "1"
    case = _case()
    _check_forward(_forward(case), _reference(case))
    graph = None
    if kind == "graph_origin":
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            _forward(case)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            _forward(case)
        case[7][-1] += 1
    elif kind in ("q_origin", "kv_origin"):
        case[6 if kind == "q_origin" else 7][1] += 1
    else:
        cu, origin = (
            (case[4], case[6])
            if kind in ("q_limit", "q_decrease", "q_base")
            else (case[5], case[7])
        )
        if kind == "q_decrease":
            cu[-1] = 100
        elif kind == "q_base":
            cu[0] = 1
        else:
            cu[-1] = 513
        origin.copy_(cu)
    print("EXPECT_FORWARD_GUARD_TRAP", kind, flush=True)
    if graph is None:
        _forward(case)
    else:
        graph.replay()
    torch.cuda.synchronize()
    raise AssertionError("Invalid forward metadata escaped the device guard")


@pytest.mark.parametrize(
    "kind", ["q_origin", "kv_origin", "q_limit", "kv_limit", "q_decrease", "q_base", "graph_origin"]
)
def test_frost_forward_metadata_guard(forward_enabled, kind):
    del forward_enabled
    command = [sys.executable] + (["-S"] if sys.flags.no_site else [])
    code = (
        "import json, runpy, sys; sys.path[:] = json.loads(sys.argv[1]);"
        " runpy.run_path(sys.argv[2])['_invalid_metadata'](sys.argv[3])"
    )
    result = subprocess.run(
        command + ["-c", code, json.dumps(sys.path), str(Path(__file__).resolve()), kind],
        text=True,
        capture_output=True,
        timeout=240,
        check=False,
    )
    output = result.stdout + result.stderr
    assert result.returncode != 0, output
    assert "EXPECT_FORWARD_GUARD_TRAP " + kind in output, output
    assert "NVTE_FUSED_ATTN_FROST_FWD_COMPACT: invalid sequence/offset" in output, output
