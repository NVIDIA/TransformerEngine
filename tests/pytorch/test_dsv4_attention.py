# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""DSA primitives and DSv4 layer wiring, including packed sequence boundaries."""

import pytest
import torch

from transformer_engine.pytorch import dense_indexer_loss_scores, dense_indexer_loss_backward
from transformer_engine.pytorch.attention import rope
from transformer_engine.pytorch.attention.sparse_attention import (
    Compressor,
    Indexer,
    compressor,
    dsa_cudnn,
    dsv4_attention,
)

# Sparse attention


@pytest.mark.parametrize("variant", ["hca", "csa"])
@pytest.mark.parametrize("head_dim", [512, 576])
def test_dsa_sparse_attention(variant, head_dim):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in (
        (10, 0),
        (10, 3),
        (10, 7),
    ):
        pytest.skip("cuDNN sparse attention requires SM100, SM103, or SM107")
    cudnn = pytest.importorskip("cudnn")
    dsa = getattr(cudnn, "DSA", None)
    if not all(
        hasattr(dsa, operation)
        for operation in (
            "sparse_attention_forward_wrapper",
            "sparse_attention_backward_wrapper",
        )
    ):
        pytest.skip("cuDNN sparse attention APIs are unavailable")
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
        # With ratio=4, sequence one has 64 compressed tokens. Sequence two
        # therefore selects packed indices 64, 66, ... rather than 0, 2, ... .
        # Comparing with its separate reference catches a missing sequence offset.
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


def test_dsa_rejects_unsupported_metadata():
    with pytest.raises(ValueError, match="window_size and ratio must be positive"):
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
    with pytest.raises(ValueError, match=r"query \[T,64,D\]"):
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


# Compressor


def test_dsa_compressor_clears_unused_capacity(monkeypatch):
    """Clear extra allocated compressed output before it affects normalization gradients."""
    layer = Compressor.__new__(Compressor)
    torch.nn.Module.__init__(layer)
    layer.fused_proj = torch.nn.Identity()
    layer.kv_norm = torch.nn.RMSNorm(4)
    layer.ape = torch.nn.Parameter(torch.zeros(4, 4))
    layer.projected, layer.ratio, layer.overlap, layer.weight_width = 4, 4, False, 0
    monkeypatch.setattr(
        compressor,
        "compress",
        lambda *args, **kwargs: torch.cat((torch.ones(1, 4), torch.full((1, 4), float("nan")))),
    )
    # Lengths 3 and 5 at ratio 4 produce one valid compressed entry, but T // ratio allocates two.
    cu = torch.tensor([0, 3, 8], dtype=torch.int32)
    cu_comp = torch.tensor([0, 0, 1], dtype=torch.int32)
    output = layer(torch.empty(8, 8), cu, cu_comp, total_comp=2)
    output.sum().backward()
    assert torch.isfinite(output).all()
    assert torch.isfinite(layer.kv_norm.weight.grad).all()
    assert torch.count_nonzero(output[1]) == 0


# Indexer


def _require_dense_indexer_training():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in (
        (10, 0),
        (10, 3),
        (10, 7),
    ):
        pytest.skip("cuDNN indexer training requires SM100, SM103, or SM107")
    cudnn = pytest.importorskip("cudnn")
    if not all(
        hasattr(getattr(cudnn, "DSA", None), op)
        for op in (
            "dense_indexer_score_recompute_wrapper",
            "dense_attn_score_recompute_wrapper",
            "dense_indexer_backward_wrapper",
        )
    ):
        pytest.skip("cuDNN dense indexer training APIs are unavailable")
    return cudnn


def _dense_teacher_logits(query, local_kv, compressed_kv, sink, *, ratio, window, scale):
    query = query.float()
    pos = torch.arange(query.shape[1], device=query.device)
    valid = (torch.arange(compressed_kv.shape[1], device=query.device) + 1) * ratio <= (
        pos[:, None] + 1
    )
    local_mask = (pos[None, :] <= pos[:, None]) & (pos[None, :] > pos[:, None] - window)
    comp_logits = (
        torch.einsum("bshd,bkd->bshk", query, compressed_kv.float()) * scale
    ).masked_fill(~valid[None, :, None, :], -torch.inf)
    local_logits = (torch.einsum("bshd,kd->bshk", query, local_kv.float()) * scale).masked_fill(
        ~local_mask[None, :, None, :], -torch.inf
    )
    # The denominator covers the local window, all eligible compressed KV, and the sink.
    full_lse = torch.logaddexp(
        torch.logaddexp(local_logits.logsumexp(-1), comp_logits.logsumexp(-1)),
        sink[None, None, :],
    ).contiguous()
    return comp_logits, full_lse, valid


@pytest.mark.parametrize("index_heads", [32, 64])
def test_dsa_indexer_backward(index_heads):
    """Train the standalone Indexer and its Compressor against a dense KL reference."""
    _require_dense_indexer_training()
    torch.manual_seed(43)
    seq, ratio, window = 256, 4, 32
    layer = Indexer(32, 16, 128, index_heads, 8, ratio, 1e-6, "cuda", torch.bfloat16)
    x = torch.randn(1, seq, 32, device="cuda", dtype=torch.bfloat16)
    q_residual = torch.randn(1, seq, 16, device="cuda", dtype=torch.bfloat16)
    cu = torch.tensor([0, seq], device=x.device, dtype=torch.int32)
    cu_comp = torch.tensor([0, seq // ratio], device=x.device, dtype=torch.int32)
    token_rope, compressed_rope = rope.build_dsv4_rope_tables(seq, ratio, 64, 160000.0, x.device)
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
    scale_i, scale_a = (index_heads * 128) ** -0.5, 512**-0.5
    comp_logits, full_lse, valid = _dense_teacher_logits(
        qa, local_kv, kc, sink, ratio=ratio, window=window, scale=scale_a
    )
    target = (comp_logits - full_lse[..., None]).exp().sum(2)
    target = target / target.sum(-1, keepdim=True).clamp_min(1e-30)
    index_score, attn_score = dense_indexer_loss_scores(
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
    actual_grads = dense_indexer_loss_backward(
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


# DSv4 layer wiring


def test_dsv4_indexer_auxiliary_backward():
    """Backpropagate through the returned CSA context and isolate Indexer gradients."""
    cudnn = _require_dense_indexer_training()
    if not hasattr(cudnn.DSA, "sparse_attention_forward_wrapper"):
        pytest.skip("cuDNN sparse attention APIs are unavailable")

    torch.manual_seed(43)
    seq, ratio, window = 256, 4, 32
    layer = dsv4_attention.DSv4Attention(
        hidden_size=32,
        q_lora_rank=16,
        layer_type="compressed_sparse_attention",
        head_dim=512,
        rope_head_dim=64,
        sliding_window=window,
        compression_ratio=ratio,
        o_groups=8,
        o_lora_rank=8,
        index_n_heads=32,
        index_topk=8,
        params_dtype=torch.bfloat16,
        qkv_format="bshd",
    )
    x = torch.randn(1, seq, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    _, context = layer(x, return_indexer_context=True)
    qi, ki, wi = (context[name].unsqueeze(0) for name in ("index_q", "index_k", "index_w"))
    qa, kc = (context[name].unsqueeze(0) for name in ("attn_q", "compressed_kv"))
    scale_i, scale_a = (qi.shape[-2] * qi.shape[-1]) ** -0.5, qa.shape[-1] ** -0.5
    _, full_lse, _ = _dense_teacher_logits(
        qa, context["local_kv"], kc, context["sink"], ratio=ratio, window=window, scale=scale_a
    )
    index_score, attn_score = dense_indexer_loss_scores(
        qi, ki, wi, qa, kc, full_lse, ratio=ratio, index_scale=scale_i, attention_scale=scale_a
    )
    grads = dense_indexer_loss_backward(
        qi,
        ki,
        wi,
        index_score,
        attn_score,
        torch.ones((), device=x.device),
        ratio=ratio,
        index_scale=scale_i,
        loss_coeff=0.01,
    )
    torch.autograd.backward((qi, ki, wi), grads)

    assert x.grad is None
    for name, param in layer.named_parameters():
        if name.startswith("indexer."):
            assert param.grad is not None, name
            assert torch.isfinite(param.grad).all() and torch.count_nonzero(param.grad) > 0, name
        else:
            assert param.grad is None, name


def test_dsv4_attention_rejects_fp8(monkeypatch):
    from transformer_engine.pytorch.quantization import FP8GlobalStateManager

    monkeypatch.setattr(FP8GlobalStateManager, "is_fp8_enabled", lambda: True)
    layer = dsv4_attention.DSv4Attention.__new__(dsv4_attention.DSv4Attention)
    torch.nn.Module.__init__(layer)
    layer.qkv_format, layer.hidden_size = "sbhd", 8
    with pytest.raises(NotImplementedError, match="BF16 only"):
        layer(torch.empty(4, 1, 8))


def test_dsv4_thd_compressed_prefixes():
    """Floor each sequence independently, retaining a safe allocation capacity."""
    for lengths, capacity in (([129, 257], 96), ([131, 257], 97)):
        x = torch.empty(sum(lengths), 8)
        cu = torch.tensor([0, lengths[0], sum(lengths)], dtype=torch.int32)
        cu_comp, total_comp, maximum = dsv4_attention._packed_prefixes(x, cu, 257, 4)
        assert cu_comp.tolist() == [0, 32, 96]
        assert total_comp == capacity
        assert maximum == 257


@pytest.mark.parametrize(
    "prefix,maximum,expected",
    [
        ([0, 8], 8, "contiguous INT32 cu_seqlens_q"),
        (torch.tensor([0, 8], dtype=torch.int64), 8, "contiguous INT32 cu_seqlens_q"),
        (torch.tensor([[0, 8]], dtype=torch.int32), 8, "contiguous INT32 cu_seqlens_q"),
        (torch.tensor([0], dtype=torch.int32), 8, "contiguous INT32 cu_seqlens_q"),
        (torch.tensor([0, 0, 8, 8], dtype=torch.int32)[::2], 8, "contiguous INT32 cu_seqlens_q"),
        (
            torch.tensor([0, 8], dtype=torch.int32),
            None,
            "integer max_seqlen_q >= compression_ratio",
        ),
        (torch.tensor([0, 8], dtype=torch.int32), 0, "integer max_seqlen_q >= compression_ratio"),
    ],
)
def test_dsv4_thd_rejects_malformed_metadata(prefix, maximum, expected):
    with pytest.raises(ValueError, match=expected):
        dsv4_attention._packed_prefixes(torch.empty(8, 4), prefix, maximum, 4)


def test_dsv4_thd_rejects_hidden_batch_axis_and_multi_sequence_context():
    # These checks precede projections and do not require GPU-backed TE layers.
    layer = dsv4_attention.DSv4Attention.__new__(dsv4_attention.DSv4Attention)
    torch.nn.Module.__init__(layer)
    layer.qkv_format, layer.hidden_size = "thd", 8
    layer.is_csa, layer.compression_ratio = True, 4
    with pytest.raises(ValueError, match="selected format"):
        layer(torch.empty(16, 1, 8))
    with pytest.raises(ValueError, match="contiguous INT32 cu_seqlens_q"):
        layer(torch.empty(16, 8))
    with pytest.raises(ValueError, match="batch=1"):
        layer(
            torch.empty(16, 8),
            cu_seqlens_q=torch.tensor([0, 8, 16], dtype=torch.int32),
            max_seqlen_q=8,
            return_indexer_context=True,
        )


@pytest.mark.parametrize("variant,ratio", [("hca", 128), ("csa", 4)])
def test_dsv4_attention_thd(variant, ratio):
    """Packed full-layer output and gradients match separate batch-size-1 calls."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in (
        (10, 0),
        (10, 3),
        (10, 7),
    ):
        pytest.skip("DSv4 requires SM100, SM103, or SM107")
    cudnn = pytest.importorskip("cudnn")
    if not hasattr(getattr(cudnn, "DSA", None), "sparse_attention_forward_wrapper"):
        pytest.skip("DSv4 requires cuDNN Frontend 1.29 sparse attention APIs")
    lengths = [255, 383]
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
            # A warm CPU cache must move on a same-length CUDA request.
            rope.DSv4RotaryEmbedding(ratio, 64, 12345.0, "cpu", max_seqlen=max(lengths))
            if variant == "csa"
            else None
        ),
        params_dtype=torch.bfloat16,
        qkv_format="thd",
    )
    x = torch.randn(sum(lengths), 32, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    cu = torch.tensor([0, lengths[0], sum(lengths)], device=x.device, dtype=torch.int32)
    actual = layer(x, cu_seqlens_q=cu, max_seqlen_q=max(lengths))
    assert actual.shape == x.shape
    if variant == "csa":
        assert layer.rope.cos.device == x.device
        assert layer.rope.cos.dtype == torch.float32
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
