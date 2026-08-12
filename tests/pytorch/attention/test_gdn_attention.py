# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Tests for Gated DeltaNet through the DotProductAttention API."""

import importlib.util
import math
from typing import Optional, Tuple

import pytest
import torch
import torch.nn.functional as F

from transformer_engine.pytorch import DotProductAttention


def _gdn_available() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        from cudnn.linear_attention.ops import gated_delta_net  # noqa: F401
    except (AttributeError, ImportError):
        return False
    try:
        has_cutlass = importlib.util.find_spec("cutlass") is not None
        has_cu_tile = importlib.util.find_spec("cuda.tile") is not None
    except (ImportError, ModuleNotFoundError):
        return False
    return has_cutlass or has_cu_tile


pytestmark = pytest.mark.skipif(
    not _gdn_available(), reason="GDN requires a supported cuDNN frontend kernel runtime"
)

_FWD_TOL = 2e-2
_STATE_TOL = 2e-2
_BWD_TOL = 4e-2


def _rms_ratio(actual: torch.Tensor, expected: torch.Tensor) -> float:
    """Return relative RMS error, which is appropriate for bfloat16 results."""
    actual = actual.detach().double()
    expected = expected.detach().double()
    return (
        (actual - expected).square().mean().sqrt()
        / expected.square().mean().sqrt().clamp_min(1e-12)
    ).item()


def _assert_rms_close(
    actual: torch.Tensor, expected: torch.Tensor, tolerance: float, name: str
) -> None:
    ratio = _rms_ratio(actual, expected)
    assert ratio < tolerance, f"{name} RMS ratio {ratio:.4g} >= {tolerance}"


def _gdn_recurrence(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    state: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Evaluate GDN for tensors in [batch, heads, sequence, ...] layout."""
    outputs = []
    for token_idx in range(q.shape[2]):
        q_t = q[:, :, token_idx]
        k_t = k[:, :, token_idx]
        v_t = v[:, :, token_idx]
        alpha_t = alpha[:, :, token_idx]
        beta_t = beta[:, :, token_idx]

        prediction = torch.matmul(k_t.unsqueeze(-2), state).squeeze(-2)
        residual = v_t - alpha_t.unsqueeze(-1) * prediction
        state = alpha_t[..., None, None] * state + beta_t[..., None, None] * (
            k_t.unsqueeze(-1) @ residual.unsqueeze(-2)
        )
        outputs.append(torch.matmul(q_t.unsqueeze(-2), state).squeeze(-2))

    return torch.stack(outputs, dim=2), state


def _gdn_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    scale: Optional[float] = None,
    initial_state: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pure PyTorch FP64 implementation of the Gated DeltaNet recurrence.

    Inputs use [batch, sequence, heads, dimension] layout. Packed inputs use a
    singleton batch dimension and provide sequence boundaries via cu_seqlens.
    """
    qk_dim = q.shape[-1]
    if scale is None:
        scale = 1.0 / math.sqrt(qk_dim)

    output_heads = max(q.shape[2], v.shape[2])

    def _expand_heads(tensor: torch.Tensor) -> torch.Tensor:
        repeats = output_heads // tensor.shape[2]
        return tensor.repeat_interleave(repeats, dim=2) if repeats > 1 else tensor

    q_ref = _expand_heads(q.double() * scale)
    k_ref = _expand_heads(k.double())
    v_ref = _expand_heads(v.double())
    alpha_ref = _expand_heads(g.double().exp())
    beta_ref = _expand_heads(beta.double())

    # [batch, sequence, heads, ...] -> [batch, heads, sequence, ...]
    q_ref = q_ref.permute(0, 2, 1, 3)
    k_ref = k_ref.permute(0, 2, 1, 3)
    v_ref = v_ref.permute(0, 2, 1, 3)
    alpha_ref = alpha_ref.permute(0, 2, 1)
    beta_ref = beta_ref.permute(0, 2, 1)

    def _initial_state(sequence_idx: Optional[int] = None) -> torch.Tensor:
        if initial_state is None:
            return torch.zeros(
                1 if sequence_idx is not None else q.shape[0],
                output_heads,
                qk_dim,
                v.shape[-1],
                dtype=torch.float64,
                device=q.device,
            )
        if sequence_idx is None:
            return initial_state.double()
        return initial_state[sequence_idx : sequence_idx + 1].double()

    if cu_seqlens is None:
        output, final_state = _gdn_recurrence(
            q_ref, k_ref, v_ref, alpha_ref, beta_ref, _initial_state()
        )
        return output.permute(0, 2, 1, 3), final_state

    assert q.shape[0] == 1
    bounds = cu_seqlens.tolist()
    outputs = []
    final_states = []
    for sequence_idx, (start, end) in enumerate(zip(bounds[:-1], bounds[1:])):
        state = _initial_state(sequence_idx)
        if start == end:
            final_states.append(state)
            continue
        output, state = _gdn_recurrence(
            q_ref[:, :, start:end],
            k_ref[:, :, start:end],
            v_ref[:, :, start:end],
            alpha_ref[:, :, start:end],
            beta_ref[:, :, start:end],
            state,
        )
        outputs.append(output)
        final_states.append(state)

    if outputs:
        packed_output = torch.cat(outputs, dim=2).permute(0, 2, 1, 3)
    else:
        packed_output = q_ref.new_zeros(1, 0, output_heads, v.shape[-1])
    return packed_output, torch.cat(final_states, dim=0)


def _inputs(batch, sequence, q_heads, v_heads, qk_dim=64, v_dim=64):
    torch.manual_seed(1234)
    q = torch.randn(batch, sequence, q_heads, qk_dim, device="cuda", dtype=torch.bfloat16)
    k = F.normalize(torch.randn_like(q, dtype=torch.float32), dim=-1).to(torch.bfloat16)
    v = torch.randn(batch, sequence, v_heads, v_dim, device="cuda", dtype=torch.bfloat16)
    output_heads = max(q_heads, v_heads)
    g = torch.rand(batch, sequence, output_heads, device="cuda", dtype=torch.float32).log()
    beta = torch.rand(batch, sequence, output_heads, device="cuda", dtype=torch.float32)
    return q, k, v, g, beta


def test_gdn_thd_forward_final_state_and_backward():
    """THD GDN matches a PyTorch recurrence in forward and backward."""
    batch, sequence, heads, dim = 2, 128, 2, 64
    q, k, v, g, beta = (
        tensor.reshape(batch * sequence, *tensor.shape[2:]).requires_grad_(True)
        for tensor in _inputs(batch, sequence, heads, heads, dim, dim)
    )
    cu_seqlens = torch.arange(batch + 1, device="cuda", dtype=torch.int32) * sequence
    initial_state = (
        torch.randn(batch, heads, dim, dim, device="cuda", dtype=torch.float32) * 0.05
    ).requires_grad_()

    attention = DotProductAttention(
        num_attention_heads=heads,
        kv_channels=dim,
        qkv_format="thd",
        attn_mask_type="padding_causal",
    )
    output, final_state = attention(
        q,
        k,
        v,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_kv=cu_seqlens,
        g=g,
        beta=beta,
        initial_state=initial_state,
        output_final_state=True,
    )
    reference_inputs = {
        name: tensor.detach().double().reshape(1, -1, *tensor.shape[1:]).requires_grad_()
        for name, tensor in (("q", q), ("k", k), ("v", v), ("g", g), ("beta", beta))
    }
    initial_state_ref = initial_state.detach().double().requires_grad_()
    output_ref, final_state_ref = _gdn_reference(
        reference_inputs["q"],
        reference_inputs["k"],
        reference_inputs["v"],
        reference_inputs["g"],
        reference_inputs["beta"],
        initial_state=initial_state_ref,
        cu_seqlens=cu_seqlens,
    )
    output_ref = output_ref.squeeze(0).flatten(-2)

    _assert_rms_close(output, output_ref, _FWD_TOL, "output")
    _assert_rms_close(final_state, final_state_ref, _STATE_TOL, "final state")

    output_weight = torch.randn_like(output, dtype=torch.float32)
    state_weight = torch.randn_like(final_state, dtype=torch.float32)
    ((output.float() * output_weight).sum() + (final_state.float() * state_weight).sum()).backward()
    (
        (output_ref * output_weight.double()).sum()
        + (final_state_ref * state_weight.double()).sum()
    ).backward()

    for name, tensor in (("q", q), ("k", k), ("v", v), ("g", g), ("beta", beta)):
        assert tensor.grad is not None, f"no gradient for {name}"
        assert torch.isfinite(tensor.grad).all(), f"non-finite gradient for {name}"
        reference_grad = reference_inputs[name].grad.reshape_as(tensor)
        _assert_rms_close(tensor.grad, reference_grad, _BWD_TOL, f"d{name}")
    _assert_rms_close(initial_state.grad, initial_state_ref.grad, _BWD_TOL, "dinitial_state")


@pytest.mark.parametrize("qkv_format", ["bshd", "sbhd"])
def test_gdn_dense_layout_and_grouped_value_heads(qkv_format):
    """Dense TE layouts and grouped heads match a pure PyTorch reference."""
    batch, sequence, q_heads, v_heads, dim = 1, 128, 1, 2, 64
    q, k, v, g, beta = _inputs(batch, sequence, q_heads, v_heads, dim, dim)
    with torch.no_grad():
        output_ref, _ = _gdn_reference(
            F.normalize(q.float(), dim=-1),
            F.normalize(k.float(), dim=-1),
            v,
            g,
            beta,
        )

    if qkv_format == "sbhd":
        q, k, v, g, beta = (tensor.transpose(0, 1).contiguous() for tensor in (q, k, v, g, beta))
        expected = output_ref.reshape(batch, sequence, -1).transpose(0, 1).contiguous()
    else:
        expected = output_ref.reshape(batch, sequence, -1)

    attention = DotProductAttention(
        num_attention_heads=q_heads,
        kv_channels=dim,
        qkv_format=qkv_format,
        attn_mask_type="causal",
    )
    output = attention(q, k, v, g=g, beta=beta, use_qk_l2norm_in_kernel=True)
    assert output.shape == expected.shape
    _assert_rms_close(output, expected, _FWD_TOL, "output")


def test_gdn_requires_both_gates():
    """A partial GDN invocation fails before entering a softmax-attention backend."""
    q, k, v, g, _ = _inputs(1, 128, 1, 1)
    attention = DotProductAttention(
        num_attention_heads=1,
        kv_channels=64,
        qkv_format="bshd",
        attn_mask_type="causal",
    )
    with pytest.raises(ValueError, match="requires both g and beta"):
        attention(q, k, v, g=g)
