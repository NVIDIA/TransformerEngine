# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""SM107 gradient comparison through TE's explicit backward dispatch."""
from dataclasses import replace
from itertools import accumulate
from unittest.mock import patch

import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "lengths,spans",
    [
        ((3, 125), (16, 128)),
        ((257, 1025), (384, 1152)),
        ((4097, 8191), (4097, 8191)),
    ],
)
def test_native_compact_dispatch_matches_stock(lengths, spans):
    if torch.cuda.get_device_capability() != (10, 7):
        pytest.skip("Compact GQA requires SM107")
    api = pytest.importorskip("cudnn.sdpa.bwd.compact_gqa")
    from transformer_engine.pytorch.attention import packed_sequence as metadata
    from transformer_engine.pytorch.attention.dot_product_attention import compact_gqa
    from transformer_engine.pytorch.attention.dot_product_attention.backends import (
        FusedAttnBwdArgs,
        _fused_attn_backward_impl,
    )
    from transformer_engine.pytorch.cpp_extensions.fused_attn import (
        FusedAttnBackend,
        fused_attn_fwd,
    )

    torch.manual_seed(2300)
    logical = tuple(accumulate(lengths, initial=0))
    physical = tuple(accumulate(spans, initial=0))
    cu = torch.tensor(logical, device="cuda", dtype=torch.int32)
    padded = torch.tensor(physical, device="cuda", dtype=torch.int32)
    metadata.register_cu_seqlens(cu, logical)
    metadata.register_cu_seqlens(padded, physical)
    q = torch.randn(sum(spans), 8, 256, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(sum(spans), 1, 256, device="cuda", dtype=torch.bfloat16)
    v, d_out = torch.randn_like(k), torch.randn_like(q)
    backend = FusedAttnBackend["F16_arbitrary_seqlen"]
    out, aux = fused_attn_fwd(
        True,
        max(lengths),
        max(lengths),
        cu,
        cu,
        q,
        k,
        v,
        torch.bfloat16,
        backend,
        cu_seqlens_q_padded=padded,
        cu_seqlens_kv_padded=padded,
        attn_scale=1 / 16,
        dropout=0.0,
        qkv_layout="thd_thd_thd",
        o_format="thd",
        attn_mask_type="padding_causal",
    )
    args = FusedAttnBwdArgs(
        q=q,
        k=k,
        v=v,
        out=out,
        grad_output=d_out,
        cu_seqlens_q=cu,
        cu_seqlens_kv=cu,
        cu_seqlens_q_padded=padded,
        cu_seqlens_kv_padded=padded,
        softmax_stats=aux[0],
        rng_state=aux[1],
        max_seqlen_q=max(lengths),
        max_seqlen_kv=max(lengths),
        attn_scale=1 / 16,
        dropout_p=0.0,
        fast_zero_fill=True,
        qkv_layout="thd_thd_thd",
        dqkv_layout="thd_thd_thd",
        o_format="thd",
        attn_bias_type="no_bias",
        attn_mask_type="padding_causal",
        softmax_type="vanilla",
        window_size=(-1, -1),
        fused_attention_backend=backend,
        deterministic=False,
        use_FAv2_bwd=False,
        nominal_dtype=torch.bfloat16,
        fp8=False,
        is_input_fp8=False,
    )
    with patch.object(compact_gqa, "_ENABLED", False):
        expected = _fused_attn_backward_impl(args)
    with metadata.attention_backend_workspace():
        args = replace(args, backend_workspace=metadata._get_attention_backend_workspace())
        with patch.object(compact_gqa, "_ENABLED", True), patch.object(
            api, "compact_gqa_backward", wraps=api.compact_gqa_backward
        ) as call:
            actual = _fused_attn_backward_impl(args)
            assert call.call_count == 1
            # Unregistered prefixes must dispatch to stock.
            unregistered = replace(args, cu_seqlens_q=cu.clone(), cu_seqlens_kv=cu.clone())
            fallback = _fused_attn_backward_impl(unregistered)
            assert call.call_count == 1
    for result, reference, stock in zip(actual[:3], expected[:3], fallback[:3]):
        torch.testing.assert_close(result, reference, atol=0.025, rtol=0.025)
        torch.testing.assert_close(stock, reference, atol=0.025, rtol=0.025)
    if max(lengths) <= 1025:
        # Check valid tokens against independent FP32 causal attention.
        for start, length in zip(physical, lengths):
            q_ref, k_ref, v_ref = [
                tensor[start : start + length].float().detach().requires_grad_()
                for tensor in (q, k, v)
            ]
            scores = torch.einsum("thd,shd->hts", q_ref, k_ref.expand(-1, 8, -1)) / 16
            mask = torch.ones(length, length, device=q.device, dtype=torch.bool).tril()
            probability = scores.masked_fill(~mask, float("-inf")).softmax(dim=-1)
            output = torch.einsum("hts,shd->thd", probability, v_ref.expand(-1, 8, -1))
            gradients = torch.autograd.grad(
                output, (q_ref, k_ref, v_ref), d_out[start : start + length].float()
            )
            for result, reference in zip(actual[:3], gradients):
                torch.testing.assert_close(
                    result[start : start + length].float(), reference, atol=0.025, rtol=0.025
                )
