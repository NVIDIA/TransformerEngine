# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""DSv4 core against a dense oracle, including packed sequence boundaries."""

import os

import pytest
import torch

from transformer_engine.pytorch.attention.sparse_attention import dsa_rope, dsv4_attention


@pytest.mark.parametrize("variant", ["hca", "csa"])
@pytest.mark.parametrize("head_dim", [512, 576])
def test_dsv4_forward_backward(variant, head_dim):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("DSv4 requires SM100")
    cudnn = pytest.importorskip("cudnn")
    dsa = getattr(cudnn, "DSA", None)
    if not all(
        hasattr(dsa, operation)
        for operation in (
            "sparse_attention_forward_wrapper",
            "sparse_attention_backward_wrapper",
        )
    ):
        pytest.skip("DSv4 requires cuDNN Frontend 1.29 sparse attention APIs")
    torch.manual_seed(17)
    ratio = 128 if variant == "hca" else 4
    lengths = [256, 128]
    comp_lengths = [n // ratio for n in lengths]
    total, total_comp = sum(lengths), sum(comp_lengths)
    cu = torch.tensor([0, 256, total], device="cuda", dtype=torch.int32)
    cu_comp = torch.tensor([0, comp_lengths[0], total_comp], device="cuda", dtype=torch.int32)
    query = (
        torch.randn(total, 64, head_dim, device="cuda", dtype=torch.bfloat16) * 0.1
    ).requires_grad_()
    local = (
        torch.randn(total, head_dim, device="cuda", dtype=torch.bfloat16) * 0.1
    ).requires_grad_()
    compressed = (
        torch.randn(total_comp, head_dim, device="cuda", dtype=torch.bfloat16) * 0.1
    ).requires_grad_()
    sink = torch.randn(64, device="cuda", requires_grad=True)
    leaves = (query, local, compressed, sink)
    reference_leaves = tuple(x.detach().float().requires_grad_() for x in leaves)
    indices = None
    if variant == "csa":
        # Select alternating compressed entries; distinct global IDs expose
        # mistakes at the second sequence's compressed offset.
        indices = torch.full((total, 32), -1, device="cuda", dtype=torch.int32)
        start = comp_start = 0
        for length, count in zip(lengths, comp_lengths):
            for position in range(length):
                ids = torch.arange(
                    comp_start, comp_start + (position + 1) // ratio, 2, device="cuda"
                )
                indices[start + position, : ids.numel()] = ids
            start += length
            comp_start += count

    output = dsv4_attention.DSv4Attention(window_size=32, ratio=ratio)(
        *leaves,
        cu,
        cu_comp,
        indices=indices,
        max_compressed_seqlen=max(comp_lengths) if indices is None else None,
    )
    q, k, c, s = reference_leaves
    reference = []
    start = comp_start = 0
    for length, count in zip(lengths, comp_lengths):
        kv = torch.cat((k[start : start + length], c[comp_start : comp_start + count]))
        pos = torch.arange(length, device="cuda")
        local_mask = (pos[None, :] <= pos[:, None]) & (pos[None, :] > pos[:, None] - 32)
        comp_ids = torch.arange(count, device="cuda")
        comp_mask = (comp_ids[None, :] + 1) * ratio <= pos[:, None] + 1
        if indices is not None:
            comp_mask &= (comp_ids % 2 == 0)[None, :]
        mask = torch.cat((local_mask, comp_mask), dim=-1)
        logits = torch.einsum("thd,kd->thk", q[start : start + length], kv) * (head_dim**-0.5)
        logits = logits.masked_fill(~mask[:, None, :], float("-inf"))
        probabilities = torch.cat((logits, s[None, :, None].expand(length, -1, -1)), -1).softmax(-1)
        reference.append(torch.einsum("thk,kd->thd", probabilities[..., :-1], kv[..., :512]))
        start += length
        comp_start += count
    expected = torch.cat(reference)
    torch.testing.assert_close(output.float(), expected, atol=1e-3, rtol=1e-2)
    gradient = torch.randn_like(output)
    output.backward(gradient)
    expected.backward(gradient.float())
    for actual, ref in zip(leaves, reference_leaves):
        assert actual.grad is not None and torch.isfinite(actual.grad).all()
        # Relative L2 error tolerates BF16 reduction rounding near zero entries.
        relative_error = (actual.grad.float() - ref.grad).norm() / ref.grad.norm().clamp_min(1e-8)
        assert relative_error < 0.02, relative_error.item()


def test_dsv4_rejects_unsupported_metadata():
    with pytest.raises(ValueError, match="positive"):
        dsv4_attention.DSv4Attention(window_size=0, ratio=4)
    core = dsv4_attention.DSv4Attention(window_size=32, ratio=4)
    with pytest.raises(ValueError, match="query"):
        core(torch.empty(1, 4, 512), None, None, None, None, None)


def test_dsv4_cute_rope_batched_forward_backward(monkeypatch):
    """The packed CuTe bridge must reset positions at each BSHD sequence."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("CuTe DSv4 RoPE requires SM100")
    if dsa_rope._cute_rope_module() is None:
        pytest.skip("CuTe DSL is unavailable")
    if os.environ.get("NVTE_ENABLE_CUTEDSL_BACKEND", "0") == "0":
        pytest.skip("native CuTe DSL dispatch is disabled")

    torch.manual_seed(23)
    batch, seq, heads, dim, width = 2, 17, 64, 512, 64
    x = torch.randn(batch, seq, heads, dim, device="cuda", dtype=torch.bfloat16)
    (cos, sin), _ = dsa_rope.rotary_embeddings(seq, 4, width, 160000.0, x.device)
    cu = torch.arange(batch + 1, device=x.device, dtype=torch.int32) * seq
    gradient = torch.randn_like(x)

    module = dsa_rope._cute_rope_module()
    monkeypatch.setattr(
        module,
        "forward_inplace",
        lambda *_: pytest.fail("native forward fell back to its Python launcher"),
    )
    monkeypatch.setattr(
        module,
        "backward_inplace",
        lambda *_: pytest.fail("native backward fell back to its Python launcher"),
    )
    candidate_input = x.detach().requires_grad_()
    candidate = dsa_rope.apply_rotary(candidate_input, cos, sin, cu)
    candidate_grad = torch.autograd.grad(candidate, candidate_input, gradient)[0]

    reference_input = x.detach().requires_grad_()
    reference = dsa_rope._apply_rotary_eager(reference_input, cos, sin)
    reference_grad = torch.autograd.grad(reference, reference_input, gradient)[0]

    assert candidate.data_ptr() != candidate_input.data_ptr()
    torch.testing.assert_close(candidate, reference, atol=2e-2, rtol=2e-2)
    grad_delta = candidate_grad.float() - reference_grad.float()
    assert grad_delta.abs().max() <= 0.03125
    assert grad_delta.norm() / reference_grad.float().norm() < 0.002

    query_source = x.detach().requires_grad_()
    query_input = query_source.clone()
    query_ptr = query_input.data_ptr()
    query = dsa_rope._apply_rotary_query(query_input, cos, sin, cu)
    query_grad = torch.autograd.grad(query, query_source, gradient)[0]

    assert query.data_ptr() == query_ptr
    torch.testing.assert_close(query, reference, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(query_grad, reference_grad, atol=0.03125, rtol=0.002)
