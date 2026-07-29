# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tests for mixed causal/non-causal packed THD attention."""

import pytest
import torch

from transformer_engine.pytorch import DotProductAttention


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")

NUM_HEADS = 16
HEAD_DIM = 64


def _make_dpa(dtype: torch.dtype) -> DotProductAttention:
    return DotProductAttention(
        num_attention_heads=NUM_HEADS,
        kv_channels=HEAD_DIM,
        attention_dropout=0.0,
        qkv_format="thd",
        attn_mask_type="padding",
        tp_size=1,
        tp_group=None,
        layer_number=1,
    ).to(device="cuda", dtype=dtype)


def _per_sequence_scalar_reference(
    attention: DotProductAttention,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cu_seqlens: torch.Tensor,
    sequence_is_causal: torch.Tensor,
) -> torch.Tensor:
    """Reference every packed sequence through its scalar-policy DPA call."""
    outputs = []
    start = 0
    for length, is_causal in zip(
        (cu_seqlens[1:] - cu_seqlens[:-1]).tolist(), sequence_is_causal.tolist()
    ):
        end = start + length
        sequence_cu_seqlens = torch.tensor((0, length), dtype=torch.int32, device=query.device)
        outputs.append(
            attention(
                # THD scalar attention requires tensors with a zero storage
                # offset. clone() is differentiable, so the reference still
                # accumulates gradients into the original packed inputs.
                query[start:end].clone(),
                key[start:end].clone(),
                value[start:end].clone(),
                qkv_format="thd",
                cu_seqlens_q=sequence_cu_seqlens,
                cu_seqlens_kv=sequence_cu_seqlens,
                max_seqlen_q=length,
                max_seqlen_kv=length,
                attn_mask_type="padding_causal" if is_causal else "padding",
                window_size=(-1, 0) if is_causal else (-1, -1),
                bottom_right_diagonal=False,
            )
        )
        start = end

    return torch.cat(outputs, dim=0)


@pytest.mark.parametrize(
    "sequence_is_causal",
    (
        (False, True, False, True),
        (False, False, False, False),
        (True, True, True, True),
    ),
)
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
def test_mixed_thd_matches_scalar_forward_and_backward(sequence_is_causal, dtype):
    """Mixed THD dispatch must match attention applied to each packed sequence."""
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("BF16 is not supported by this GPU")

    torch.manual_seed(1234)
    lengths = torch.tensor((7, 3, 5, 4), dtype=torch.int32, device="cuda")
    cu_seqlens = torch.cat((lengths.new_zeros(1), torch.cumsum(lengths, dim=0, dtype=torch.int32)))
    token_count = int(lengths.sum().item())
    sequence_is_causal = torch.tensor(sequence_is_causal, dtype=torch.bool, device="cuda")

    def make_input():
        return (
            0.1 * torch.randn(token_count, NUM_HEADS, HEAD_DIM, dtype=dtype, device="cuda")
        ).requires_grad_()

    query, key, value = make_input(), make_input(), make_input()
    reference_query = query.detach().clone().requires_grad_()
    reference_key = key.detach().clone().requires_grad_()
    reference_value = value.detach().clone().requires_grad_()

    attention = _make_dpa(dtype)
    mixed_output = attention(
        query,
        key,
        value,
        qkv_format="thd",
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_kv=cu_seqlens,
        max_seqlen_q=int(lengths.max().item()),
        max_seqlen_kv=int(lengths.max().item()),
        thd_sequence_is_causal=sequence_is_causal,
    )
    reference_output = _per_sequence_scalar_reference(
        attention,
        reference_query,
        reference_key,
        reference_value,
        cu_seqlens,
        sequence_is_causal,
    )

    tolerances = {"atol": 1.0e-3, "rtol": 1.0e-3}
    if dtype == torch.bfloat16:
        tolerances = {"atol": 1.5e-2, "rtol": 1.5e-2}
    torch.testing.assert_close(mixed_output, reference_output, **tolerances)

    output_grad = torch.randn_like(mixed_output)
    mixed_output.backward(output_grad)
    reference_output.backward(output_grad)
    for mixed_grad, reference_grad in (
        (query.grad, reference_query.grad),
        (key.grad, reference_key.grad),
        (value.grad, reference_value.grad),
    ):
        torch.testing.assert_close(mixed_grad, reference_grad, **tolerances)


@pytest.mark.parametrize(
    ("invalid_case", "error"),
    (
        ("policy_dtype", "one-dimensional torch.bool"),
        ("sliding_window", "sliding-window attention"),
    ),
)
def test_mixed_thd_rejects_unsupported_inputs(invalid_case, error):
    """Reject metadata that the grouping path cannot preserve."""
    query = torch.randn(4, NUM_HEADS, HEAD_DIM, device="cuda", dtype=torch.float16)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    cu_seqlens = torch.tensor((0, 2, 4), dtype=torch.int32, device="cuda")
    attention = _make_dpa(torch.float16)

    kwargs = {
        "thd_sequence_is_causal": torch.tensor((False, True), dtype=torch.bool, device="cuda"),
    }
    if invalid_case == "policy_dtype":
        kwargs["thd_sequence_is_causal"] = torch.tensor((0, 1), dtype=torch.int32, device="cuda")
    else:
        kwargs["window_size"] = (4, 0)

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
            **kwargs,
        )


def test_mixed_thd_empty_batch_preserves_input_gradients():
    """An all-empty packed batch needs zero gradients, not a disconnected output."""
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
        thd_sequence_is_causal=torch.empty(0, dtype=torch.bool, device="cuda"),
    )
    assert output.shape == (0, NUM_HEADS * HEAD_DIM)
    output.sum().backward()
    for input_tensor in (query, key, value):
        torch.testing.assert_close(input_tensor.grad, torch.zeros_like(input_tensor))
