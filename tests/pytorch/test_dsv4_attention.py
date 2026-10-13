# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""DSA primitives and DSv4 layer wiring, including packed sequence boundaries."""

import pytest
import torch

from transformer_engine.pytorch.attention.sparse_attention import (
    Compressor,
    Indexer,
    compressor,
    dsa_cudnn,
    dsa_rope,
    dsv4_attention,
)


@pytest.mark.parametrize("variant", ["hca", "csa"])
@pytest.mark.parametrize("head_dim", [512, 576])
def test_dsa_sparse_attention_forward_backward(variant, head_dim):
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

    output = dsa_cudnn.sparse_attention(
        *leaves,
        cu,
        cu_comp,
        window_size=32,
        ratio=ratio,
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
        dsa_cudnn.sparse_attention(
            torch.empty(1, 64, 512),
            torch.empty(1, 512),
            torch.empty(1, 512),
            torch.empty(64),
            None,
            None,
            window_size=0,
            ratio=4,
        )
    with pytest.raises(ValueError, match="query"):
        dsa_cudnn.sparse_attention(
            torch.empty(1, 4, 512),
            None,
            None,
            None,
            None,
            None,
            window_size=32,
            ratio=4,
        )


def test_dsv4_attention_rejects_fp8(monkeypatch):
    from transformer_engine.pytorch.quantization import FP8GlobalStateManager

    monkeypatch.setattr(FP8GlobalStateManager, "is_fp8_enabled", lambda: True)
    layer = dsv4_attention.DSv4Attention.__new__(dsv4_attention.DSv4Attention)
    torch.nn.Module.__init__(layer)
    layer.qkv_format, layer.hidden_size = "sbhd", 8
    with pytest.raises(NotImplementedError, match="BF16 only"):
        layer(torch.empty(4, 1, 8))


@pytest.mark.parametrize("heads", [1, 64])
def test_dsa_triton_rope_batched_forward_backward(monkeypatch, heads):
    """The packed Triton kernels must reset positions at each BSHD sequence."""
    if not torch.cuda.is_available():
        pytest.skip("Triton DSv4 RoPE requires CUDA")
    if dsa_rope._triton_rope_module() is None:
        pytest.skip("Triton DSL is unavailable")
    torch.manual_seed(23)
    batch, seq, dim, width = 2, 17, 512, 64
    x = torch.randn(batch, seq, heads, dim, device="cuda", dtype=torch.bfloat16)
    (cos, sin), _ = dsa_rope.rotary_embeddings(seq, 4, width, 160000.0, x.device)
    cu = torch.arange(batch + 1, device=x.device, dtype=torch.int32) * seq
    gradient = torch.randn_like(x)

    eager = dsa_rope._apply_rotary_eager
    monkeypatch.setattr(
        dsa_rope,
        "_apply_rotary_eager",
        lambda *_: pytest.fail("Triton dispatch fell back to eager RoPE"),
    )
    candidate_input = x.detach().requires_grad_()
    candidate = dsa_rope.apply_rotary(candidate_input, cos, sin, cu)
    candidate_grad = torch.autograd.grad(candidate, candidate_input, gradient)[0]

    reference_input = x.detach().requires_grad_()
    reference = eager(reference_input, cos, sin)
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


def test_dsa_rope_non_power_of_two_forward_backward(monkeypatch):
    """An accepted width of 96 must fall back for both rotary entry points."""
    if not torch.cuda.is_available():
        pytest.skip("DSv4 rotary dispatch requires CUDA")
    if dsa_rope._triton_rope_module() is None:
        pytest.skip("Triton DSL is unavailable")
    torch.manual_seed(41)
    x = torch.randn(17, 2, 128, device="cuda", dtype=torch.bfloat16)
    (cos, sin), _ = dsa_rope.rotary_embeddings(17, 4, 96, 160000.0, x.device)
    cu = torch.tensor([0, 17], device=x.device, dtype=torch.int32)
    monkeypatch.setattr(
        dsa_rope, "_launch_triton", lambda *_: pytest.fail("Width 96 reached Triton")
    )
    assert not dsa_rope._can_use_triton(x, cos, sin, cu)
    gradient = torch.randn_like(x)
    reference_input = x.detach().requires_grad_()
    reference = dsa_rope._apply_rotary_eager(reference_input, cos, sin, cu)
    reference_grad = torch.autograd.grad(reference, reference_input, gradient)[0]
    for apply in (dsa_rope.apply_rotary, dsa_rope._apply_rotary_query):
        candidate_input = x.detach().requires_grad_()
        candidate = apply(candidate_input, cos, sin, cu)
        candidate_grad = torch.autograd.grad(candidate, candidate_input, gradient)[0]
        torch.testing.assert_close(candidate, reference)
        torch.testing.assert_close(candidate_grad, reference_grad)


@pytest.mark.parametrize("index_heads", [32, 64])
def test_dsa_indexer_backward(index_heads):
    """Train the standalone Indexer and its Compressor against a dense KL reference."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("DSv4 requires SM100")
    cudnn = pytest.importorskip("cudnn")
    if not all(
        hasattr(getattr(cudnn, "DSA", None), op)
        for op in (
            "dense_indexer_score_recompute_wrapper",
            "dense_attn_score_recompute_wrapper",
            "dense_indexer_backward_wrapper",
        )
    ):
        pytest.skip("DSv4 requires cuDNN dense indexer training APIs")
    torch.manual_seed(43)
    seq, ratio, window = 256, 4, 32
    layer = Indexer(32, 16, 128, index_heads, 8, ratio, 1e-6, "cuda", torch.bfloat16, fused=True)
    x = torch.randn(1, seq, 32, device="cuda", dtype=torch.bfloat16)
    q_residual = torch.randn(1, seq, 16, device="cuda", dtype=torch.bfloat16)
    cu = torch.tensor([0, seq], device=x.device, dtype=torch.int32)
    cu_comp = torch.tensor([0, seq // ratio], device=x.device, dtype=torch.int32)
    token_rope, compressed_rope = dsa_rope.rotary_embeddings(seq, ratio, 64, 160000.0, x.device)
    ids, query, key, weights = layer(
        x,
        q_residual,
        cu,
        cu_comp,
        token_rope,
        compressed_rope,
        total_comp=seq // ratio,
        max_seqlen=seq,
        return_context=True,
    )
    assert ids.shape == (seq, 8)
    qi, ki, wi = (t.unsqueeze(0) for t in (query, key, weights))
    qa = (torch.randn(1, seq, 64, 512, device=x.device, dtype=torch.bfloat16) * 0.1).float()
    kc = (torch.randn(1, seq // ratio, 512, device=x.device, dtype=torch.bfloat16) * 0.1).float()
    local_kv = (torch.randn(seq, 512, device=x.device, dtype=torch.bfloat16) * 0.1).float()
    sink = torch.randn(64, device=x.device)
    pos = torch.arange(seq, device=x.device)
    valid = (torch.arange(seq // ratio, device=x.device) + 1) * ratio <= pos[:, None] + 1
    local_mask = (pos[None, :] <= pos[:, None]) & (pos[None, :] > pos[:, None] - window)
    scale_i, scale_a = (index_heads * 128) ** -0.5, 512**-0.5
    comp_logits = (torch.einsum("bshd,bkd->bshk", qa, kc) * scale_a).masked_fill(
        ~valid[None, :, None, :], -torch.inf
    )
    local_logits = (torch.einsum("bshd,kd->bshk", qa, local_kv) * scale_a).masked_fill(
        ~local_mask[None, :, None, :], -torch.inf
    )
    # The dense teacher denominator includes local keys, ALL compressed keys, and sink.
    full_lse = torch.logaddexp(
        torch.logaddexp(local_logits.logsumexp(-1), comp_logits.logsumexp(-1)),
        sink[None, None, :],
    ).contiguous()
    target = (comp_logits - full_lse[..., None]).exp().sum(2)
    target = target / target.sum(-1, keepdim=True).clamp_min(1e-30)
    index_score, attn_score = dsa_cudnn.dense_indexer_loss_scores(
        qi,
        ki,
        wi,
        qa.to(torch.bfloat16),
        kc.to(torch.bfloat16),
        full_lse,
        ratio=ratio,
        index_scale=scale_i,
        attention_scale=scale_a,
    )
    actual_target = attn_score["out"] / attn_score["denom"].unsqueeze(-1).clamp_min(1e-30)
    # Masked score-buffer entries are not valid teacher probabilities.
    actual_target = actual_target.masked_fill(~valid[None], 0)
    torch.testing.assert_close(actual_target, target, atol=2e-4, rtol=2e-2)

    qr, kr, wr = (t.detach().float().requires_grad_() for t in (qi, ki, wi))
    logits = (torch.einsum("bshd,bkd->bshk", qr, kr).relu() * wr[..., None]).sum(2) * scale_i
    logits = logits.masked_fill(~valid[None], -torch.inf)
    logits = torch.where(valid.any(-1)[None, :, None], logits, 0)
    log_prob = logits.log_softmax(-1)
    actual_log_prob = index_score["out"] - index_score["denom"].unsqueeze(-1)
    torch.testing.assert_close(actual_log_prob[0][valid], log_prob[0][valid], atol=2e-3, rtol=2e-2)
    coefficient, upstream = 0.01, 0.7
    loss = (
        coefficient * upstream * (target * (target.clamp_min(1e-30).log() - log_prob)).sum() / seq
    )
    reference_grads = torch.autograd.grad(loss, (qr, kr, wr))
    actual_grads = dsa_cudnn.dense_indexer_loss_backward(
        qi,
        ki,
        wi,
        index_score,
        attn_score,
        torch.tensor(upstream, device=x.device),
        ratio=ratio,
        index_scale=scale_i,
        loss_coeff=coefficient,
    )
    for actual, expected in zip(actual_grads, reference_grads):
        assert actual.shape == expected.shape and torch.isfinite(actual).all()
        error = (actual.float() - expected).norm() / expected.norm().clamp_min(1e-8)
        assert error < 0.03, error.item()
    parameters = tuple(layer.parameters())
    candidate = torch.autograd.grad((qi, ki, wi), parameters, actual_grads)
    # TE Linear releases its backward workspace; compare using a fresh forward.
    _, ref_q, ref_k, ref_w = layer(
        x,
        q_residual,
        cu,
        cu_comp,
        token_rope,
        compressed_rope,
        total_comp=seq // ratio,
        max_seqlen=seq,
        return_context=True,
    )
    reference = torch.autograd.grad(
        (ref_q, ref_k, ref_w),
        parameters,
        tuple(g[0].to(t.dtype) for g, t in zip(reference_grads, (ref_q, ref_k, ref_w))),
    )
    for actual, expected in zip(candidate, reference):
        assert torch.isfinite(actual).all() and actual.float().norm() > 0
        error = (actual.float() - expected.float()).norm() / expected.float().norm().clamp_min(1e-8)
        assert error < 0.03, error.item()


def test_dsv4_thd_compressed_prefixes():
    """Floor each sequence independently, retaining a safe allocation capacity."""
    for lengths, capacity in (([129, 257], 96), ([131, 257], 97)):
        x = torch.empty(sum(lengths), 8)
        cu = torch.tensor([0, lengths[0], sum(lengths)], dtype=torch.int32)
        cu_comp, total_comp, maximum = dsv4_attention._packed_prefixes(x, cu, 257, 4)
        assert cu_comp.tolist() == [0, 32, 96]
        assert total_comp == capacity
        assert maximum == 257


def test_dsa_compressor_clears_unused_capacity(monkeypatch):
    """Unused cuDNN output must not contaminate normalization weight gradients."""
    layer = Compressor.__new__(Compressor)
    torch.nn.Module.__init__(layer)
    layer.kv_proj = layer.gate_proj = torch.nn.Identity()
    layer.kv_norm = torch.nn.RMSNorm(4)
    layer.position_bias = torch.nn.Parameter(torch.zeros(4, 4))
    layer.projected, layer.ratio, layer.overlap, layer.weight_width = 4, 4, False, 0
    monkeypatch.setattr(
        compressor,
        "compress",
        lambda *args, **kwargs: torch.cat((torch.ones(1, 4), torch.full((1, 4), float("nan")))),
    )
    cu = torch.tensor([0, 3, 8], dtype=torch.int32)
    cu_comp = torch.tensor([0, 0, 1], dtype=torch.int32)
    output = layer(torch.empty(8, 4), cu, cu_comp, total_comp=2)
    output.sum().backward()
    assert torch.isfinite(output).all()
    assert torch.isfinite(layer.kv_norm.weight.grad).all()
    assert torch.count_nonzero(output[1]) == 0


@pytest.mark.parametrize(
    "prefix,maximum",
    [
        (None, 8),
        ([0, 8], 8),
        (torch.tensor([0, 8], dtype=torch.int64), 8),
        (torch.tensor([[0, 8]], dtype=torch.int32), 8),
        (torch.tensor([0], dtype=torch.int32), 8),
        (torch.tensor([0, 0, 8, 8], dtype=torch.int32)[::2], 8),
        (torch.tensor([0, 8], dtype=torch.int32), None),
        (torch.tensor([0, 8], dtype=torch.int32), 0),
    ],
)
def test_dsv4_thd_rejects_malformed_metadata(prefix, maximum):
    with pytest.raises(ValueError, match="THD requires"):
        dsv4_attention._packed_prefixes(torch.empty(8, 4), prefix, maximum, 4)


def test_dsv4_thd_rejects_hidden_batch_axis_and_multi_sequence_context():
    # These checks precede projections and do not require GPU-backed TE layers.
    layer = dsv4_attention.DSv4Attention.__new__(dsv4_attention.DSv4Attention)
    torch.nn.Module.__init__(layer)
    layer.qkv_format, layer.hidden_size = "thd", 8
    layer.is_csa, layer.compression_ratio = True, 4
    with pytest.raises(ValueError, match="selected format"):
        layer(torch.empty(16, 1, 8))
    with pytest.raises(ValueError, match="THD requires"):
        layer(torch.empty(16, 8))
    with pytest.raises(ValueError, match="batch=1"):
        layer(
            torch.empty(16, 8),
            cu_seqlens_q=torch.tensor([0, 8, 16], dtype=torch.int32),
            max_seqlen_q=8,
            return_indexer_context=True,
        )


def test_dsa_packed_rope_eager_forward_backward():
    """Reset per-sequence positions and safely rotate unused compressed capacity."""
    torch.manual_seed(31)
    cu = torch.tensor([0, 3, 8], dtype=torch.int32)
    (cos, sin), _ = dsa_rope.rotary_embeddings(5, 1, 4, 12345.0, "cpu")
    x = torch.randn(9, 2, 8, requires_grad=True)
    reference_input = x.detach().clone().requires_grad_()
    actual = dsa_rope.apply_rotary(x, cos, sin, cu)
    expected = torch.cat(
        [
            dsa_rope._apply_rotary_eager(
                reference_input[start:end].unsqueeze(0),
                cos[:, : end - start],
                sin[:, : end - start],
            ).squeeze(0)
            for start, end in ((0, 3), (3, 8), (8, 9))
        ]
    )
    gradient = torch.randn_like(actual)
    actual.backward(gradient)
    expected.backward(gradient)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(x.grad, reference_input.grad)


@pytest.mark.parametrize("variant,ratio", [("hca", 128), ("csa", 4)])
@pytest.mark.parametrize("lengths", [[129, 257], [255, 383]])
def test_dsv4_hybrid_thd_forward_backward(variant, ratio, lengths):
    """Packed full-layer output and gradients match independent dense sequences."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("DSv4 requires SM100")
    cudnn = pytest.importorskip("cudnn")
    if not hasattr(getattr(cudnn, "DSA", None), "sparse_attention_forward_wrapper"):
        pytest.skip("DSv4 requires cuDNN Frontend 1.29 sparse attention APIs")
    torch.manual_seed(37)
    layer = dsv4_attention.DSv4Attention(
        hidden_size=32,
        q_lora_rank=16,
        layer_type=(
            "compressed_sparse_attention" if variant == "csa" else "heavily_compressed_attention"
        ),
        head_dim=512,
        rope_head_dim=64,
        sliding_window=32,
        compression_ratio=ratio,
        o_groups=8,
        o_lora_rank=8,
        index_topk=8,
        rope=(
            dsa_rope._DSv4RotaryEmbedding(ratio, 64, 12345.0, "cuda") if variant == "csa" else None
        ),
        params_dtype=torch.bfloat16,
        qkv_format="thd",
    )
    x = torch.randn(sum(lengths), 32, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    cu = torch.tensor([0, lengths[0], sum(lengths)], device=x.device, dtype=torch.int32)
    actual = layer(x, cu_seqlens_q=cu, max_seqlen_q=max(lengths))
    assert actual.shape == x.shape
    gradient = torch.randn_like(actual)
    actual.backward(gradient)
    packed_grads = {
        name: p.grad.clone() for name, p in layer.named_parameters() if p.grad is not None
    }
    layer.zero_grad(set_to_none=True)
    layer.qkv_format = "sbhd" if variant == "hca" else "bshd"
    batch_axis = 1 if layer.qkv_format == "sbhd" else 0
    reference_input = x.detach().clone().requires_grad_()
    parts = []
    for part in reference_input.split(lengths):
        if variant == "csa":
            output, context = layer(part.unsqueeze(batch_axis), return_indexer_context=True)
            assert all(context[name].requires_grad for name in ("index_q", "index_k", "index_w"))
            assert not any(
                context[name].requires_grad
                for name in ("attn_q", "local_kv", "compressed_kv", "sink")
            )
        else:
            output = layer(part.unsqueeze(batch_axis))
        parts.append(output.squeeze(batch_axis))
    expected = torch.cat(parts)
    expected.backward(gradient)
    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)
    for packed_grad, reference_grad in [(x.grad, reference_input.grad)] + [
        (packed_grads[name], p.grad) for name, p in layer.named_parameters() if name in packed_grads
    ]:
        assert reference_grad is not None
        assert torch.isfinite(packed_grad).all() and torch.isfinite(reference_grad).all()
        error = (packed_grad.float() - reference_grad.float()).norm()
        assert error / reference_grad.float().norm().clamp_min(1e-8) < 0.03
