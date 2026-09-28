# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Dispatch contracts; GPU numerics are tested separately."""
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from transformer_engine.pytorch.attention import packed_sequence as metadata
from transformer_engine.pytorch.attention.dot_product_attention import compact_gqa


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
    plan = SimpleNamespace(
        compile=Mock(), initialize_workspace=Mock(), scratch_workspace_bytes=lambda n: 256 + n
    )
    api = ModuleType("cudnn.sdpa.bwd.compact_gqa")
    api.CompactGqaBackward = Mock(return_value=plan)
    api.compact_gqa_backward = Mock(
        side_effect=lambda q, k, v, *args, **kwargs: {"dq": q, "dk": k, "dv": v}
    )
    monkeypatch.setitem(sys.modules, api.__name__, api)
    with metadata.attention_backend_workspace():
        owner = metadata._get_attention_backend_workspace()
        for lengths in ((3, 5), (3, 5), (7, 9)):
            args = inputs(lengths)
            lse = torch.zeros(args.q.shape[0], 8)
            result = compact_gqa.try_compact_gqa_backward(args, args.q, lse)
            assert result[0] is args.q
        assert plan.compile.call_count == 1
        assert plan.initialize_workspace.call_count == 2
        assert api.compact_gqa_backward.call_args.kwargs["sequence_lengths"] == (7, 9)
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
