# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Tests for per-sequence mask types in packed THD attention."""

import pytest
import torch

from transformer_engine.pytorch import DotProductAttention


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")

NUM_HEADS = 16
HEAD_DIM = 64


def _make_dpa(dtype: torch.dtype, attention_type: str = "self") -> DotProductAttention:
    return DotProductAttention(
        num_attention_heads=NUM_HEADS,
        kv_channels=HEAD_DIM,
        attention_dropout=0.0,
        qkv_format="thd",
        attn_mask_type="padding",
        attention_type=attention_type,
        tp_size=1,
        tp_group=None,
        layer_number=1,
    ).to(device="cuda", dtype=dtype)


def _make_cu_seqlens(lengths):
    lengths = torch.tensor(lengths, dtype=torch.int32, device="cuda")
    return torch.cat((lengths.new_zeros(1), torch.cumsum(lengths, dim=0, dtype=torch.int32)))


def _make_policy(sequence_ids):
    return {
        mask_type: torch.tensor(ids, dtype=torch.int64, device="cuda")
        for mask_type, ids in sequence_ids.items()
    }


def _per_sequence_scalar_reference(
    attention: DotProductAttention,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_kv: torch.Tensor,
    mask_type_per_seq,
    *,
    cu_seqlens_q_padded=None,
    cu_seqlens_kv_padded=None,
    window_size_per_mask_type=None,
) -> torch.Tensor:
    """Reference each physical sequence through an ordinary scalar-mask call."""
    q_physical = cu_seqlens_q if cu_seqlens_q_padded is None else cu_seqlens_q_padded
    kv_physical = cu_seqlens_kv if cu_seqlens_kv_padded is None else cu_seqlens_kv_padded
    outputs = [None] * (cu_seqlens_q.numel() - 1)
    for mask_type, sequence_ids in mask_type_per_seq.items():
        for sequence_id in sequence_ids.tolist():
            q_length = int((cu_seqlens_q[sequence_id + 1] - cu_seqlens_q[sequence_id]).item())
            kv_length = int((cu_seqlens_kv[sequence_id + 1] - cu_seqlens_kv[sequence_id]).item())
            q_start = int(q_physical[sequence_id].item())
            kv_start = int(kv_physical[sequence_id].item())
            q_end = q_start + q_length
            kv_end = kv_start + kv_length
            sequence_cu_seqlens_q = torch.tensor(
                (0, q_length), dtype=torch.int32, device=query.device
            )
            sequence_cu_seqlens_kv = torch.tensor(
                (0, kv_length), dtype=torch.int32, device=query.device
            )
            policy_window_size = None
            if window_size_per_mask_type is not None:
                policy_window_size = window_size_per_mask_type.get(mask_type)
            outputs[sequence_id] = attention(
                # THD scalar attention requires zero-offset storage. clone() is
                # differentiable, so gradients still reach the packed inputs.
                query[q_start:q_end].clone(),
                key[kv_start:kv_end].clone(),
                value[kv_start:kv_end].clone(),
                qkv_format="thd",
                cu_seqlens_q=sequence_cu_seqlens_q,
                cu_seqlens_kv=sequence_cu_seqlens_kv,
                max_seqlen_q=q_length,
                max_seqlen_kv=kv_length,
                attn_mask_type=mask_type,
                window_size=policy_window_size,
            )

    output = query.new_zeros((query.shape[0], NUM_HEADS * HEAD_DIM))
    for sequence_id, sequence_output in enumerate(outputs):
        q_length = int((cu_seqlens_q[sequence_id + 1] - cu_seqlens_q[sequence_id]).item())
        q_start = int(q_physical[sequence_id].item())
        output[q_start : q_start + q_length] = sequence_output
    return output


@pytest.mark.parametrize(
    "mask_type_per_seq",
    (
        {"padding": (0, 2), "padding_causal": (1, 3)},
        {"padding": (0, 1, 2, 3)},
        {"padding_causal": (0, 1, 2, 3)},
        {"padding": (0, 2), "padding_causal_bottom_right": (1, 3)},
    ),
)
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
def test_thd_mask_types_match_scalar_forward_and_backward(mask_type_per_seq, dtype):
    """Per-sequence masks must match independent scalar-policy attention."""
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("BF16 is not supported by this GPU")

    torch.manual_seed(1234)
    cu_seqlens = _make_cu_seqlens((7, 3, 5, 4))
    token_count = 19
    mask_type_per_seq = _make_policy(mask_type_per_seq)

    def make_input():
        return (
            0.1 * torch.randn(token_count, NUM_HEADS, HEAD_DIM, dtype=dtype, device="cuda")
        ).requires_grad_()

    query, key, value = make_input(), make_input(), make_input()
    reference_query = query.detach().clone().requires_grad_()
    reference_key = key.detach().clone().requires_grad_()
    reference_value = value.detach().clone().requires_grad_()

    attention = _make_dpa(dtype)
    output = attention(
        query,
        key,
        value,
        qkv_format="thd",
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_kv=cu_seqlens,
        max_seqlen_q=7,
        max_seqlen_kv=7,
        attn_mask_type_per_seq=mask_type_per_seq,
    )
    reference_output = _per_sequence_scalar_reference(
        attention,
        reference_query,
        reference_key,
        reference_value,
        cu_seqlens,
        cu_seqlens,
        mask_type_per_seq,
    )

    tolerances = {"atol": 1.0e-3, "rtol": 1.0e-3}
    if dtype == torch.bfloat16:
        tolerances = {"atol": 1.5e-2, "rtol": 1.5e-2}
    torch.testing.assert_close(output, reference_output, **tolerances)

    output_grad = torch.randn_like(output)
    output.backward(output_grad)
    reference_output.backward(output_grad)
    for actual_grad, reference_grad in (
        (query.grad, reference_query.grad),
        (key.grad, reference_key.grad),
        (value.grad, reference_value.grad),
    ):
        torch.testing.assert_close(actual_grad, reference_grad, **tolerances)


def test_thd_mask_types_compose_with_existing_inter_sequence_padding():
    """Policy gaps and pre-existing THD padding must share the physical layout."""
    torch.manual_seed(1234)
    dtype = torch.float16
    cu_seqlens = _make_cu_seqlens((7, 3, 5, 4))
    cu_seqlens_padded = torch.tensor((0, 9, 12, 19, 23), dtype=torch.int32, device="cuda")
    mask_type_per_seq = _make_policy({"padding": (0, 2), "padding_causal": (1, 3)})

    def make_input():
        return (
            0.1 * torch.randn(23, NUM_HEADS, HEAD_DIM, dtype=dtype, device="cuda")
        ).requires_grad_()

    query, key, value = make_input(), make_input(), make_input()
    reference_query = query.detach().clone().requires_grad_()
    reference_key = key.detach().clone().requires_grad_()
    reference_value = value.detach().clone().requires_grad_()
    attention = _make_dpa(dtype)

    output = attention(
        query,
        key,
        value,
        qkv_format="thd",
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_kv=cu_seqlens,
        cu_seqlens_q_padded=cu_seqlens_padded,
        cu_seqlens_kv_padded=cu_seqlens_padded,
        max_seqlen_q=7,
        max_seqlen_kv=7,
        attn_mask_type_per_seq=mask_type_per_seq,
    )
    reference_output = _per_sequence_scalar_reference(
        attention,
        reference_query,
        reference_key,
        reference_value,
        cu_seqlens,
        cu_seqlens,
        mask_type_per_seq,
        cu_seqlens_q_padded=cu_seqlens_padded,
        cu_seqlens_kv_padded=cu_seqlens_padded,
    )
    torch.testing.assert_close(output, reference_output, atol=1.0e-3, rtol=1.0e-3)

    output_grad = torch.randn_like(output)
    output.backward(output_grad)
    reference_output.backward(output_grad)
    for actual_grad, reference_grad in (
        (query.grad, reference_query.grad),
        (key.grad, reference_key.grad),
        (value.grad, reference_value.grad),
    ):
        torch.testing.assert_close(actual_grad, reference_grad, atol=1.0e-3, rtol=1.0e-3)


def test_thd_mask_types_support_cross_attention_and_per_policy_windows():
    """Mask groups may use different Q/KV lengths and sliding windows."""
    torch.manual_seed(1234)
    dtype = torch.float16
    cu_seqlens_q = _make_cu_seqlens((5, 3, 4))
    cu_seqlens_kv = _make_cu_seqlens((7, 2, 5))
    mask_type_per_seq = _make_policy({"padding": (0, 2), "padding_causal_bottom_right": (1,)})
    windows = {"padding": (-1, -1), "padding_causal_bottom_right": (2, 0)}

    query = (
        0.1 * torch.randn(12, NUM_HEADS, HEAD_DIM, dtype=dtype, device="cuda")
    ).requires_grad_()
    key = (0.1 * torch.randn(14, NUM_HEADS, HEAD_DIM, dtype=dtype, device="cuda")).requires_grad_()
    value = (0.1 * torch.randn_like(key)).requires_grad_()
    reference_query = query.detach().clone().requires_grad_()
    reference_key = key.detach().clone().requires_grad_()
    reference_value = value.detach().clone().requires_grad_()
    attention = _make_dpa(dtype, attention_type="cross")

    output = attention(
        query,
        key,
        value,
        qkv_format="thd",
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_kv=cu_seqlens_kv,
        max_seqlen_q=5,
        max_seqlen_kv=7,
        attn_mask_type_per_seq=mask_type_per_seq,
        window_size_per_mask_type=windows,
    )
    reference_output = _per_sequence_scalar_reference(
        attention,
        reference_query,
        reference_key,
        reference_value,
        cu_seqlens_q,
        cu_seqlens_kv,
        mask_type_per_seq,
        window_size_per_mask_type=windows,
    )
    torch.testing.assert_close(output, reference_output, atol=1.0e-3, rtol=1.0e-3)

    output_grad = torch.randn_like(output)
    output.backward(output_grad)
    reference_output.backward(output_grad)
    for actual_grad, reference_grad in (
        (query.grad, reference_query.grad),
        (key.grad, reference_key.grad),
        (value.grad, reference_value.grad),
    ):
        torch.testing.assert_close(actual_grad, reference_grad, atol=1.0e-3, rtol=1.0e-3)


def test_thd_mask_types_support_cuda_graph_capture():
    """The sync-free policy metadata path must be replayable in a CUDA graph."""
    torch.manual_seed(1234)
    dtype = torch.float16
    cu_seqlens = _make_cu_seqlens((7, 3, 5, 4))
    mask_type_per_seq = _make_policy({"padding": (0, 2), "padding_causal": (1, 3)})
    query = 0.1 * torch.randn(19, NUM_HEADS, HEAD_DIM, dtype=dtype, device="cuda")
    key = 0.1 * torch.randn_like(query)
    value = 0.1 * torch.randn_like(query)
    attention = _make_dpa(dtype).eval()

    def run_attention():
        return attention(
            query,
            key,
            value,
            qkv_format="thd",
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_kv=cu_seqlens,
            max_seqlen_q=7,
            max_seqlen_kv=7,
            attn_mask_type_per_seq=mask_type_per_seq,
        )

    warmup_stream = torch.cuda.Stream()
    warmup_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup_stream):
        for _ in range(3):
            run_attention()
    torch.cuda.current_stream().wait_stream(warmup_stream)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_output = run_attention()
    graph.replay()
    actual_output = captured_output.clone()
    expected_output = run_attention()
    torch.testing.assert_close(actual_output, expected_output, atol=1.0e-3, rtol=1.0e-3)


@pytest.mark.parametrize(
    ("invalid_case", "error"),
    (
        ("policy_dtype", "one-dimensional int32 or int64"),
        ("missing_sequence", "assign every sequence exactly once by count"),
        ("mask_type", "only padding mask types"),
        ("window_key", "mask not present"),
    ),
)
def test_thd_mask_types_reject_invalid_inputs(invalid_case, error):
    """Reject malformed policy metadata before launching attention."""
    query = torch.randn(4, NUM_HEADS, HEAD_DIM, device="cuda", dtype=torch.float16)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    cu_seqlens = torch.tensor((0, 2, 4), dtype=torch.int32, device="cuda")
    attention = _make_dpa(torch.float16)
    policies = _make_policy({"padding": (0,), "padding_causal": (1,)})
    windows = None

    if invalid_case == "policy_dtype":
        policies["padding"] = torch.tensor((0,), dtype=torch.float32, device="cuda")
    elif invalid_case == "missing_sequence":
        policies = _make_policy({"padding": (0,)})
    elif invalid_case == "mask_type":
        policies = _make_policy({"no_mask": (0,), "padding": (1,)})
    else:
        windows = {"padding_causal_bottom_right": (-1, 0)}

    with pytest.raises(ValueError, match=error):
        attention(
            query,
            key,
            value,
            qkv_format="thd",
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_kv=cu_seqlens,
            max_seqlen_q=2,
            max_seqlen_kv=2,
            attn_mask_type_per_seq=policies,
            window_size_per_mask_type=windows,
        )


def test_thd_mask_types_empty_batch_preserves_input_gradients():
    """An empty packed batch needs zero gradients, not a disconnected output."""
    attention = _make_dpa(torch.float16)
    query = torch.empty(
        0, NUM_HEADS, HEAD_DIM, device="cuda", dtype=torch.float16, requires_grad=True
    )
    key = torch.empty_like(query, requires_grad=True)
    value = torch.empty_like(query, requires_grad=True)
    cu_seqlens = torch.zeros(1, dtype=torch.int32, device="cuda")

    output = attention(
        query,
        key,
        value,
        qkv_format="thd",
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_kv=cu_seqlens,
        max_seqlen_q=0,
        max_seqlen_kv=0,
        attn_mask_type_per_seq={"padding": torch.empty(0, dtype=torch.int64, device="cuda")},
    )
    assert output.shape == (0, NUM_HEADS * HEAD_DIM)
    output.sum().backward()
    for input_tensor in (query, key, value):
        torch.testing.assert_close(input_tensor.grad, torch.zeros_like(input_tensor))
