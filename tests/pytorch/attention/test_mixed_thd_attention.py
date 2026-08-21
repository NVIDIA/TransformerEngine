# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Tests for per-sequence mask types in packed THD attention."""

import pytest
import torch

from transformer_engine.pytorch import DotProductAttention
from transformer_engine.pytorch.attention.dot_product_attention import (
    dot_product_attention as dpa_module,
)


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


def _make_policies(policy_specs):
    return [
        {
            "sequence_ids": torch.tensor(sequence_ids, dtype=torch.int64, device="cuda"),
            "mask_type": mask_type,
            "window_size": window_size,
        }
        for mask_type, sequence_ids, window_size in policy_specs
    ]


def _per_sequence_scalar_reference(
    attention: DotProductAttention,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_kv: torch.Tensor,
    policies,
    *,
    cu_seqlens_q_padded=None,
    cu_seqlens_kv_padded=None,
) -> torch.Tensor:
    """Reference each physical sequence through an ordinary scalar-mask call."""
    q_physical = cu_seqlens_q if cu_seqlens_q_padded is None else cu_seqlens_q_padded
    kv_physical = cu_seqlens_kv if cu_seqlens_kv_padded is None else cu_seqlens_kv_padded
    outputs = [None] * (cu_seqlens_q.numel() - 1)
    for policy in policies:
        mask_type = policy["mask_type"]
        sequence_ids = policy["sequence_ids"]
        policy_window_size = policy["window_size"]
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
    "policy_specs",
    (
        (("padding", (0, 2), (-1, -1)), ("padding_causal", (1, 3), (-1, 0))),
        (("padding", (0, 1, 2, 3), (-1, -1)),),
        (("padding_causal", (0, 1, 2, 3), (-1, 0)),),
        (
            ("padding", (0, 2), (-1, -1)),
            ("padding_causal_bottom_right", (1, 3), (-1, 0)),
        ),
    ),
)
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
def test_thd_mask_types_match_scalar_forward_and_backward(policy_specs, dtype):
    """Per-sequence masks must match independent scalar-policy attention."""
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("BF16 is not supported by this GPU")

    torch.manual_seed(1234)
    cu_seqlens = _make_cu_seqlens((7, 3, 5, 4))
    token_count = 19
    policies = _make_policies(policy_specs)

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
        attn_mask_type_and_window_size_per_seq_policies=policies,
    )
    reference_output = _per_sequence_scalar_reference(
        attention,
        reference_query,
        reference_key,
        reference_value,
        cu_seqlens,
        cu_seqlens,
        policies,
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
    policies = _make_policies((("padding", (0, 2), (-1, -1)), ("padding_causal", (1, 3), (-1, 0))))

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
        attn_mask_type_and_window_size_per_seq_policies=policies,
    )
    reference_output = _per_sequence_scalar_reference(
        attention,
        reference_query,
        reference_key,
        reference_value,
        cu_seqlens,
        cu_seqlens,
        policies,
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
    policies = _make_policies(
        (("padding", (0, 2), (-1, -1)), ("padding_causal_bottom_right", (1,), (2, 0)))
    )

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
        attn_mask_type_and_window_size_per_seq_policies=policies,
    )
    reference_output = _per_sequence_scalar_reference(
        attention,
        reference_query,
        reference_key,
        reference_value,
        cu_seqlens_q,
        cu_seqlens_kv,
        policies,
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


@pytest.mark.parametrize(
    "policy_specs",
    (
        (
            ("padding_causal", (0,), (1, 0)),
            ("padding_causal", (2,), (3, 0)),
            ("padding", (1, 3), (2, 2)),
        ),
        (
            ("padding", (1, 3), (2, 2)),
            ("padding_causal", (2,), (3, 0)),
            ("padding_causal", (0,), (1, 0)),
        ),
    ),
)
def test_thd_mask_types_support_repeated_masks_and_policy_order(policy_specs):
    """Repeated mask types may use distinct windows in any policy-list order."""
    torch.manual_seed(1234)
    dtype = torch.float16
    cu_seqlens = _make_cu_seqlens((7, 3, 5, 4))
    policies = _make_policies(policy_specs)

    def make_input():
        return (
            0.1 * torch.randn(19, NUM_HEADS, HEAD_DIM, dtype=dtype, device="cuda")
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
        attn_mask_type_and_window_size_per_seq_policies=policies,
    )
    reference_output = _per_sequence_scalar_reference(
        attention,
        reference_query,
        reference_key,
        reference_value,
        cu_seqlens,
        cu_seqlens,
        policies,
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


def test_thd_mask_type_runtime_dispatch_uses_backend_selection(monkeypatch):
    """Route each policy according to backend support for inter-sequence padding."""
    policies = [
        {
            "sequence_ids": torch.tensor((0, 2), dtype=torch.int64, device="cuda"),
            "mask_type": "padding",
            "window_size": (-1, -1),
            "bottom_right_diagonal": True,
        },
        {
            "sequence_ids": torch.tensor((1,), dtype=torch.int64, device="cuda"),
            "mask_type": "padding_causal",
            "window_size": (2, 0),
            "bottom_right_diagonal": False,
        },
    ]
    observed_params = []

    def fake_get_attention_backend(attention_params):
        observed_params.append(attention_params)
        available_backends = [False, attention_params.attn_mask_type == "padding", False]
        return False, None, available_backends[1], None, False, available_backends

    monkeypatch.setattr(dpa_module.dpa_utils, "get_attention_backend", fake_get_attention_backend)
    padded_policies, grouped_policies = DotProductAttention._partition_thd_mask_policies(
        policies,
        {},
    )

    assert padded_policies[0] is policies[0]
    assert grouped_policies[0] is policies[1]
    assert [params.batch_size for params in observed_params] == [2, 1]
    assert [params.attn_mask_type for params in observed_params] == [
        "padding",
        "padding_causal",
    ]
    assert [params.window_size for params in observed_params] == [(-1, -1), (2, 0)]
    assert [params.bottom_right_diagonal for params in observed_params] == [True, False]
    assert all(params.pad_between_seqs for params in observed_params)


def test_thd_mask_type_runtime_dispatch_combines_backend_outputs(monkeypatch):
    """Policies routed to different backend representations share one output."""
    attention = _make_dpa(torch.float16)
    cu_seqlens = _make_cu_seqlens((1, 1))
    policies = [
        {
            "sequence_ids": torch.tensor((0,), dtype=torch.int64, device="cuda"),
            "mask_type": "padding",
            "window_size": (-1, -1),
            "bottom_right_diagonal": True,
        },
        {
            "sequence_ids": torch.tensor((1,), dtype=torch.int64, device="cuda"),
            "mask_type": "padding_causal",
            "window_size": (-1, 0),
            "bottom_right_diagonal": False,
        },
    ]
    query = torch.zeros(2, NUM_HEADS, HEAD_DIM, dtype=torch.float16, device="cuda")
    padded_output = query.new_zeros((2, NUM_HEADS * HEAD_DIM))
    grouped_output = torch.zeros_like(padded_output)
    padded_output[0] = 1
    grouped_output[1] = 2

    monkeypatch.setattr(
        attention,
        "_partition_thd_mask_policies",
        lambda *_args, **_kwargs: ([policies[0]], [policies[1]]),
    )
    monkeypatch.setattr(
        attention,
        "_forward_thd_mask_types_with_padding",
        lambda *_args, **_kwargs: padded_output,
    )
    monkeypatch.setattr(
        attention,
        "_forward_thd_mask_types_grouped",
        lambda *_args, **_kwargs: grouped_output,
    )

    output = attention._forward_thd_mask_types(
        query,
        query,
        query,
        cu_seqlens,
        cu_seqlens,
        None,
        None,
        policies,
        1,
        1,
        attention_params_kwargs={},
        checkpoint_core_attention=False,
        fast_zero_fill=True,
        pad_between_seqs=False,
        bf16_backward=None,
        num_splits=None,
    )
    torch.testing.assert_close(output, padded_output + grouped_output)


def test_thd_mask_types_support_cuda_graph_capture(monkeypatch):
    """The sync-free policy metadata path must be replayable in a CUDA graph."""
    if dpa_module.dpa_utils.get_device_compute_capability() != (9, 0):
        pytest.skip("CUDA graph capture requires a Hopper GPU for the FA3 test path")
    if not dpa_module.FlashAttentionUtils.v3_is_installed:
        pytest.skip("CUDA graph capture requires the Hopper/FA3 padding implementation")

    monkeypatch.setenv("NVTE_FLASH_ATTN", "1")
    monkeypatch.setenv("NVTE_FLASH_ATTN_V3", "1")
    monkeypatch.setenv("NVTE_FUSED_ATTN", "0")
    monkeypatch.setenv("NVTE_UNFUSED_ATTN", "0")

    torch.manual_seed(1234)
    dtype = torch.float16
    cu_seqlens = _make_cu_seqlens((7, 3, 5, 4))
    policies = _make_policies((("padding", (0, 2), (-1, -1)), ("padding_causal", (1, 3), (-1, 0))))
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
            attn_mask_type_and_window_size_per_seq_policies=policies,
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
        ("policy_keys", "must contain exactly"),
        ("scalar_mask", "do not also pass"),
    ),
)
def test_thd_mask_types_reject_invalid_inputs(invalid_case, error):
    """Reject malformed policy metadata before launching attention."""
    query = torch.randn(4, NUM_HEADS, HEAD_DIM, device="cuda", dtype=torch.float16)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    cu_seqlens = torch.tensor((0, 2, 4), dtype=torch.int32, device="cuda")
    attention = _make_dpa(torch.float16)
    policies = _make_policies((("padding", (0,), (-1, -1)), ("padding_causal", (1,), (-1, 0))))
    scalar_mask_type = None

    if invalid_case == "policy_dtype":
        policies[0]["sequence_ids"] = torch.tensor((0,), dtype=torch.float32, device="cuda")
    elif invalid_case == "missing_sequence":
        policies = _make_policies((("padding", (0,), (-1, -1)),))
    elif invalid_case == "mask_type":
        policies[0]["mask_type"] = "no_mask"
    elif invalid_case == "policy_keys":
        del policies[0]["window_size"]
    else:
        scalar_mask_type = "padding"

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
            attn_mask_type=scalar_mask_type,
            attn_mask_type_and_window_size_per_seq_policies=policies,
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
        attn_mask_type_and_window_size_per_seq_policies=[
            {
                "sequence_ids": torch.empty(0, dtype=torch.int64, device="cuda"),
                "mask_type": "padding",
                "window_size": (-1, -1),
            }
        ],
    )
    assert output.shape == (0, NUM_HEADS * HEAD_DIM)
    output.sum().backward()
    for input_tensor in (query, key, value):
        torch.testing.assert_close(input_tensor.grad, torch.zeros_like(input_tensor))
