# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Compact GQA metadata, workspace, dispatch, and GPU numerics."""
import gc
import weakref
from dataclasses import replace
from itertools import accumulate
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from transformer_engine.pytorch.attention import packed_sequence as metadata
from transformer_engine.pytorch.attention.dot_product_attention import compact_gqa


@pytest.mark.parametrize("offsets", [(0,), (0, 1), (0, 0, 257, 4097), (0, 16384, 65536)])
def test_registered_metadata_never_reads_tensor(offsets, monkeypatch):
    tensor = torch.tensor(offsets, dtype=torch.int32)
    monkeypatch.setattr(torch.Tensor, "tolist", lambda *_: pytest.fail("Unexpected readback"))
    metadata.register_cu_seqlens(tensor, offsets)
    assert metadata._get_cu_seqlens(tensor) == offsets


def test_mutation_and_clone_require_registration():
    tensor = torch.tensor([0, 3, 8], dtype=torch.int32)
    metadata.register_cu_seqlens(tensor, [0, 3, 8])
    assert metadata._get_cu_seqlens(tensor.clone()) is None
    alias = tensor.view_as(tensor)
    assert metadata._get_cu_seqlens(alias) == (0, 3, 8)
    alias[-1] = 9
    assert metadata._get_cu_seqlens(tensor) is None
    metadata.register_cu_seqlens(tensor, [0, 3, 9])
    assert metadata._get_cu_seqlens(tensor) == (0, 3, 9)


def test_registration_does_not_retain_tensor_or_mutable_host_list():
    tensor = torch.tensor([0, 3], dtype=torch.int32)
    key = (tensor.device, tensor.data_ptr())
    host = [0, 3]
    metadata.register_cu_seqlens(tensor, host)
    host[-1] = 7
    assert metadata._get_cu_seqlens(tensor) == (0, 3)
    ref = weakref.ref(tensor)
    del tensor
    gc.collect()
    assert ref() is None
    assert key not in metadata._PREFIXES


@pytest.mark.parametrize("offsets", [(1, 3), (0, 3, 2), (), (0, 2**31)])
def test_invalid_offsets(offsets):
    with pytest.raises(ValueError):
        metadata.register_cu_seqlens(torch.zeros(len(offsets), dtype=torch.int32), offsets)


def test_tensor_metadata_is_not_implicitly_copied_to_host():
    tensor = torch.tensor([0, 3], dtype=torch.int32)
    with pytest.raises(TypeError):
        metadata.register_cu_seqlens(tensor, tensor)


def test_shape_and_dtype_mismatch():
    with pytest.raises(ValueError):
        metadata.register_cu_seqlens(torch.tensor([0, 3]), [0, 3])
    with pytest.raises(ValueError):
        metadata.register_cu_seqlens(torch.zeros(3, dtype=torch.int32), [0, 3])


def test_inference_tensor_is_not_retained():
    with torch.inference_mode():
        tensor = torch.tensor([0, 3], dtype=torch.int32)
        metadata.register_cu_seqlens(tensor, [0, 3])
        assert metadata._get_cu_seqlens(tensor) is None


def test_workspace_nested_and_exception_cleanup():
    assert metadata._get_attention_backend_workspace() is None
    with pytest.raises(RuntimeError, match="training failed"):
        with metadata.attention_backend_workspace():
            owner = metadata._get_attention_backend_workspace()
            tensor = torch.empty(10)
            ref = weakref.ref(tensor)
            owner.buffers["test"] = tensor
            del tensor
            with metadata.attention_backend_workspace():
                assert metadata._get_attention_backend_workspace() is owner
            assert owner.active and ref() is not None
            raise RuntimeError("training failed")
    assert not owner.active and not owner.buffers and ref() is None
    assert metadata._get_attention_backend_workspace() is None


def inputs(lengths=(3, 5)):
    offsets = [0]
    for length in lengths:
        offsets.append(offsets[-1] + length)
    tokens = offsets[-1]
    q = torch.zeros((tokens, 8, 256), dtype=torch.bfloat16)
    k = torch.zeros((tokens, 1, 256), dtype=torch.bfloat16)
    cu = torch.tensor(offsets, dtype=torch.int32)
    metadata.register_cu_seqlens(cu, offsets)
    return SimpleNamespace(
        q=q,
        k=k,
        v=k.clone(),
        out=q.clone(),
        cu_seqlens_q=cu,
        cu_seqlens_kv=cu,
        cu_seqlens_q_padded=None,
        cu_seqlens_kv_padded=None,
        max_seqlen_q=max(lengths),
        max_seqlen_kv=max(lengths),
        backend_workspace=metadata._get_attention_backend_workspace(),
        fp8=False,
        is_input_fp8=False,
        use_FAv2_bwd=False,
        deterministic=False,
        qkv_layout="thd_thd_thd",
        dqkv_layout="thd_thd_thd",
        o_format="thd",
        nominal_dtype=torch.bfloat16,
        dropout_p=0.0,
        attn_bias_type="no_bias",
        softmax_type="vanilla",
        attn_mask_type="padding_causal",
        attn_scale=1 / 16,
        window_size=(-1, -1),
    )


def test_disabled_dispatch_does_not_inspect_or_import_backend(monkeypatch):
    monkeypatch.setattr(compact_gqa, "_ENABLED", False)
    assert compact_gqa.try_compact_gqa_backward(None, None, None) is None


def test_cpu_inputs_use_stock():
    args = inputs()
    assert not compact_gqa._eligible(args, args.q, torch.zeros(8, 8))


def test_unregistered_and_mutated_prefixes_use_stock():
    args = inputs()
    assert compact_gqa._packing(args) == ((0, 3, 8), (3, 5))
    args.cu_seqlens_q[-1] = 7
    assert compact_gqa._packing(args) is None


def test_padding_and_mismatched_kv():
    args = inputs()
    padded = torch.tensor([0, 4, 12], dtype=torch.int32)
    metadata.register_cu_seqlens(padded, [0, 4, 12])
    args.q = torch.empty((12, 8, 256))
    args.cu_seqlens_q_padded = args.cu_seqlens_kv_padded = padded
    assert compact_gqa._packing(args) == ((0, 4, 12), (3, 5))
    other = torch.tensor([0, 4, 8], dtype=torch.int32)
    metadata.register_cu_seqlens(other, [0, 4, 8])
    args.cu_seqlens_kv = other
    assert compact_gqa._packing(args) is None


def test_workspace_reuse_resize_and_release(monkeypatch):
    import sys

    monkeypatch.setattr(compact_gqa, "_ENABLED", True)
    monkeypatch.setattr(compact_gqa, "_eligible", lambda *_: True)
    monkeypatch.setattr(compact_gqa, "_PLANS", {})
    monkeypatch.setattr(compact_gqa, "_BOUND_WORKSPACES", {})
    monkeypatch.setattr(torch.cuda, "current_stream", lambda *_: SimpleNamespace(cuda_stream=3))
    initialized_capacity = 0

    def initialize_workspace(buffer, stream, *, max_seqlen):
        nonlocal initialized_capacity
        initialized_capacity = max_seqlen

    def backward(q, k, v, *args, sequence_lengths, **kwargs):
        assert max(sequence_lengths) <= initialized_capacity
        return {"dq": q, "dk": k, "dv": v}

    plan = SimpleNamespace(
        compile=Mock(),
        initialize_workspace=Mock(side_effect=initialize_workspace),
        scratch_workspace_bytes=lambda n: 256 + n,
    )
    api = ModuleType("cudnn.sdpa.bwd.compact_gqa")
    api.CompactGqaBackward = Mock(return_value=plan)
    api.compact_gqa_backward = Mock(side_effect=backward)
    monkeypatch.setitem(sys.modules, api.__name__, api)
    with metadata.attention_backend_workspace():
        owner = metadata._get_attention_backend_workspace()
        for lengths in ((3, 5), (3, 5), (7, 9), (3, 5)):
            args = inputs(lengths)
            lse = torch.zeros(args.q.shape[0], 8)
            result = compact_gqa.try_compact_gqa_backward(args, args.q, lse)
            assert result[0] is args.q
        assert plan.compile.call_count == 1
        assert plan.initialize_workspace.call_count == 2
        assert api.CompactGqaBackward.call_count == 1
        assert api.compact_gqa_backward.call_args.kwargs["sequence_lengths"] == (3, 5)
        assert owner.buffers
    assert not owner.buffers and not owner.active
    assert compact_gqa.try_compact_gqa_backward(args, args.q, lse) is None


@pytest.mark.parametrize(
    "field,value",
    [
        ("fp8", True),
        ("is_input_fp8", True),
        ("use_FAv2_bwd", True),
        ("deterministic", True),
        ("dropout_p", 0.1),
        ("attn_bias_type", "post_scale_bias"),
        ("softmax_type", "off-by-one"),
        ("attn_mask_type", "no_mask"),
        ("qkv_layout", "bshd_bshd_bshd"),
        ("window_size", (128, 0)),
    ],
)
def test_unsupported_modes(field, value, monkeypatch):
    class CudaMetadataTensor(torch.Tensor):
        @property
        def is_cuda(self):
            return True

    args = inputs()
    args.q = args.q.as_subclass(CudaMetadataTensor)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *_: (10, 7))
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    lse = torch.zeros(8, 8)
    assert compact_gqa._eligible(args, args.q, lse)
    setattr(args, field, value)
    assert not compact_gqa._eligible(args, args.q, lse)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "lengths,spans",
    [
        ((3, 125), (16, 128)),
        ((257, 1025), (384, 1152)),
        ((4097, 8191), (4097, 8191)),
    ],
)
def test_native_compact_dispatch_matches_stock(lengths, spans, monkeypatch):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7):
        pytest.skip("Compact GQA requires SM107")
    monkeypatch.setattr(compact_gqa, "_PLANS", {})
    monkeypatch.setattr(compact_gqa, "_BOUND_WORKSPACES", {})
    small = ((3, 125), (16, 128))
    cached_plan = None
    allocated = 0
    try:
        with metadata.attention_backend_workspace():
            owner = metadata._get_attention_backend_workspace()
            for current_lengths, current_spans in (small, (lengths, spans), small):
                _check_native_compact_dispatch(current_lengths, current_spans)
                assert len(compact_gqa._PLANS) == 1
                plan = next(iter(compact_gqa._PLANS.values()))
                if cached_plan is None:
                    cached_plan = plan
                assert plan is cached_plan
                assert len(owner.buffers) == 1
                _, capacity = next(iter(owner.buffers.values()))
                allocated = max(allocated, *current_lengths)
                assert capacity == allocated
    finally:
        torch.cuda.synchronize()
        for plan in compact_gqa._PLANS.values():
            plan.close()


def _check_native_compact_dispatch(lengths, spans):
    """Compare one pack with stock and an independent reference."""
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
