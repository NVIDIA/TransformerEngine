# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Tests for Gated DeltaNet through the DotProductAttention API."""

import importlib.util
import math
import os
from typing import Optional, Tuple

import pytest
import torch
import torch.nn.functional as F

from transformer_engine.pytorch import DotProductAttention, autocast, is_fp8_available


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


_GDN_AVAILABLE = _gdn_available()
if os.getenv("NVTE_GDN_TEST_REQUIRED", "0") == "1" and not _GDN_AVAILABLE:
    raise RuntimeError(
        "NVTE_GDN_TEST_REQUIRED=1, but the cuDNN frontend GDN op or its "
        "cutedsl runtime is unavailable."
    )

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
requires_gdn = pytest.mark.skipif(
    not _GDN_AVAILABLE, reason="GDN requires a supported cuDNN frontend kernel runtime"
)

_FWD_TOL = {torch.bfloat16: 2e-2, torch.float16: 1e-2}
_STATE_TOL = {torch.bfloat16: 2e-2, torch.float16: 1e-2}
_BWD_TOL = {torch.bfloat16: 4e-2, torch.float16: 2e-2}


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
    """Evaluate GDN for tensors in [batch, heads, sequence, ...] layout.

    ``state`` follows the cuDNN frontend's [..., v_head_dim, qk_head_dim] convention.
    """
    outputs = []
    for token_idx in range(q.shape[2]):
        q_t = q[:, :, token_idx]
        k_t = k[:, :, token_idx]
        v_t = v[:, :, token_idx]
        alpha_t = alpha[:, :, token_idx]
        beta_t = beta[:, :, token_idx]

        prediction = torch.matmul(state, k_t.unsqueeze(-1)).squeeze(-1)
        residual = v_t - alpha_t.unsqueeze(-1) * prediction
        state = alpha_t[..., None, None] * state + beta_t[..., None, None] * (
            residual.unsqueeze(-1) @ k_t.unsqueeze(-2)
        )
        outputs.append(torch.matmul(state, q_t.unsqueeze(-1)).squeeze(-1))

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
    ``initial_state``/the returned final state use the cuDNN frontend's
    [batch, heads, v_head_dim, qk_head_dim] convention throughout.
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
                v.shape[-1],
                qk_dim,
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


def _inputs(
    batch,
    sequence,
    q_heads,
    v_heads,
    qk_dim=64,
    v_dim=64,
    dtype=torch.bfloat16,
):
    torch.manual_seed(1234)
    q = torch.randn(batch, sequence, q_heads, qk_dim, device="cuda", dtype=dtype)
    k = F.normalize(torch.randn_like(q, dtype=torch.float32), dim=-1).to(dtype)
    v = torch.randn(batch, sequence, v_heads, v_dim, device="cuda", dtype=dtype)
    output_heads = max(q_heads, v_heads)
    g = torch.rand(batch, sequence, output_heads, device="cuda", dtype=torch.float32).log()
    beta = torch.rand(batch, sequence, output_heads, device="cuda", dtype=torch.float32)
    return q, k, v, g, beta


@requires_gdn
@pytest.mark.parametrize("checkpoint_core_attention", [False, True], ids=["eager", "checkpoint"])
@pytest.mark.parametrize(
    ("qk_dim", "v_dim"),
    [(64, 64), (128, 128), (64, 128)],
    ids=["qk64_v64_cutile", "qk128_v128_frost", "qk64_v128"],
)
@pytest.mark.parametrize("use_qk_l2norm_in_kernel", [False, True], ids=["no_l2norm", "l2norm"])
def test_gdn_thd_forward_final_state_and_backward(
    checkpoint_core_attention, qk_dim, v_dim, use_qk_l2norm_in_kernel
):
    """THD GDN matches a PyTorch recurrence in forward and backward.

    head_dim=128 exercises the FROST engine, while head_dim=64 falls back to cuTile.
    """
    batch, sequence, heads = 2, 128, 2
    q, k, v, g, beta = (
        tensor.reshape(batch * sequence, *tensor.shape[2:]).requires_grad_(True)
        for tensor in _inputs(batch, sequence, heads, heads, qk_dim, v_dim)
    )
    cu_seqlens = torch.arange(batch + 1, device="cuda", dtype=torch.int32) * sequence
    initial_state = (
        torch.randn(batch, heads, v_dim, qk_dim, device="cuda", dtype=torch.float32) * 0.05
    ).requires_grad_()

    attention = DotProductAttention(
        num_attention_heads=heads,
        kv_channels=(qk_dim, v_dim),
        qkv_format="thd",
        attn_mask_type="padding_causal",
    )
    output, final_state = attention(
        q,
        k,
        v,
        cu_seqlens_q=cu_seqlens,
        g=g,
        beta=beta,
        initial_state=initial_state,
        output_final_state=True,
        checkpoint_core_attention=checkpoint_core_attention,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
    )
    reference_inputs = {
        name: tensor.detach().double().reshape(1, -1, *tensor.shape[1:]).requires_grad_()
        for name, tensor in (("q", q), ("k", k), ("v", v), ("g", g), ("beta", beta))
    }
    initial_state_ref = initial_state.detach().double().requires_grad_()
    reference_q = reference_inputs["q"]
    reference_k = reference_inputs["k"]
    if use_qk_l2norm_in_kernel:
        reference_q = F.normalize(reference_q, dim=-1)
        reference_k = F.normalize(reference_k, dim=-1)
    output_ref, final_state_ref = _gdn_reference(
        reference_q,
        reference_k,
        reference_inputs["v"],
        reference_inputs["g"],
        reference_inputs["beta"],
        initial_state=initial_state_ref,
        cu_seqlens=cu_seqlens,
    )
    output_ref = output_ref.squeeze(0).flatten(-2)

    _assert_rms_close(output, output_ref, _FWD_TOL[q.dtype], "output")
    _assert_rms_close(final_state, final_state_ref, _STATE_TOL[q.dtype], "final state")

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
        _assert_rms_close(tensor.grad, reference_grad, _BWD_TOL[q.dtype], f"d{name}")
    assert initial_state.grad is not None, "no gradient for initial_state"
    _assert_rms_close(
        initial_state.grad,
        initial_state_ref.grad,
        _BWD_TOL[q.dtype],
        "dinitial_state",
    )


@pytest.mark.parametrize("qkv_format", ["bshd", "sbhd"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
@requires_gdn
def test_gdn_dense_layout(qkv_format, dtype):
    """Dense TE layouts and supported input dtypes match a PyTorch reference."""
    batch, sequence, heads, dim = 1, 128, 2, 64
    q, k, v, g, beta = _inputs(batch, sequence, heads, heads, dim, dim, dtype)
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
        num_attention_heads=heads,
        kv_channels=dim,
        qkv_format=qkv_format,
        attn_mask_type="causal",
    )
    output = attention(q, k, v, g=g, beta=beta, use_qk_l2norm_in_kernel=True)
    assert output.shape == expected.shape
    _assert_rms_close(output, expected, _FWD_TOL[dtype], "output")


def test_gdn_rejects_value_head_count_that_changes_output_width():
    """DPA's configured output width cannot be changed by the runtime V tensor."""
    q, k, v, g, beta = _inputs(1, 128, 1, 2)
    attention = DotProductAttention(
        num_attention_heads=1,
        kv_channels=64,
        qkv_format="bshd",
        attn_mask_type="causal",
    )
    with pytest.raises(ValueError, match="GDN V must have 1 heads"):
        attention(q, k, v, g=g, beta=beta)


@requires_gdn
def test_gdn_state_round_trip_matches_single_shot():
    """A final state can seed the next chunk without changing the result."""
    batch, sequence, heads, dim = 2, 128, 2, 64
    split = 64
    q, k, v, g, beta = _inputs(batch, sequence, heads, heads, dim, dim)
    attention = DotProductAttention(
        num_attention_heads=heads,
        kv_channels=dim,
        qkv_format="bshd",
        attn_mask_type="causal",
    )

    with torch.no_grad():
        full_output, full_state = attention(q, k, v, g=g, beta=beta, output_final_state=True)
        first_output, first_state = attention(
            q[:, :split],
            k[:, :split],
            v[:, :split],
            g=g[:, :split],
            beta=beta[:, :split],
            output_final_state=True,
        )
        second_output, chunked_state = attention(
            q[:, split:],
            k[:, split:],
            v[:, split:],
            g=g[:, split:],
            beta=beta[:, split:],
            initial_state=first_state,
            output_final_state=True,
        )

    chunked_output = torch.cat((first_output, second_output), dim=1)
    _assert_rms_close(chunked_output, full_output, _FWD_TOL[q.dtype], "chunked output")
    _assert_rms_close(chunked_state, full_state, _STATE_TOL[q.dtype], "chunked state")


@requires_gdn
@pytest.mark.parametrize(
    "bounds",
    [(0, 48, 160), (0, 0, 64, 160)],
    ids=["unequal", "leading-empty"],
)
def test_gdn_thd_ragged_sequences(bounds):
    """Packed GDN handles unequal lengths and empty sequences."""
    total_tokens, heads, dim = bounds[-1], 2, 64
    q, k, v, g, beta = (
        tensor.squeeze(0) for tensor in _inputs(1, total_tokens, heads, heads, dim, dim)
    )
    cu_seqlens = torch.tensor(bounds, device="cuda", dtype=torch.int32)
    initial_state = torch.randn(
        len(bounds) - 1,
        heads,
        dim,
        dim,
        device="cuda",
        dtype=torch.float32,
    )
    attention = DotProductAttention(
        num_attention_heads=heads,
        kv_channels=dim,
        qkv_format="thd",
        attn_mask_type="padding_causal",
    )

    with torch.no_grad():
        output, final_state = attention(
            q,
            k,
            v,
            cu_seqlens_q=cu_seqlens,
            g=g,
            beta=beta,
            initial_state=initial_state,
            output_final_state=True,
        )
        output_ref, final_state_ref = _gdn_reference(
            q.unsqueeze(0),
            k.unsqueeze(0),
            v.unsqueeze(0),
            g.unsqueeze(0),
            beta.unsqueeze(0),
            initial_state=initial_state,
            cu_seqlens=cu_seqlens,
        )

    _assert_rms_close(
        output,
        output_ref.squeeze(0).flatten(-2),
        _FWD_TOL[q.dtype],
        "ragged output",
    )
    _assert_rms_close(
        final_state,
        final_state_ref,
        _STATE_TOL[q.dtype],
        "ragged final state",
    )


def test_gdn_requires_both_gates():
    """A partial GDN invocation fails before entering a softmax-attention backend."""
    q, k, v, g, _ = _inputs(1, 128, 1, 1)
    attention = DotProductAttention(
        num_attention_heads=1,
        kv_channels=64,
        qkv_format="bshd",
        attn_mask_type="causal",
    )
    with pytest.raises(ValueError, match="require both g and beta"):
        attention(q, k, v, g=g)


@pytest.mark.skipif(not is_fp8_available(), reason="FP8 is not available")
def test_gdn_rejects_fp8_autocast():
    """GDN must not silently run in high precision inside FP8 autocast."""
    q, k, v, g, beta = _inputs(1, 128, 1, 1)
    attention = DotProductAttention(
        num_attention_heads=1,
        kv_channels=64,
        qkv_format="bshd",
        attn_mask_type="causal",
    )
    with autocast(enabled=True), pytest.raises(ValueError, match="does not support FP8 autocast"):
        attention(q, k, v, g=g, beta=beta)


def test_gdn_runs_te_forward_lifecycle(monkeypatch):
    """GDN calls pair prepare_forward with end_forward even without the kernel runtime."""
    q, k, v, g, beta = _inputs(1, 128, 1, 1)
    attention = DotProductAttention(
        num_attention_heads=1,
        kv_channels=64,
        qkv_format="bshd",
        attn_mask_type="causal",
    )
    events = []
    prepare_forward = attention.prepare_forward
    end_forward = attention.end_forward

    def traced_prepare_forward(*args, **kwargs):
        events.append("prepare")
        return prepare_forward(*args, **kwargs)

    def traced_end_forward():
        events.append("end")
        return end_forward()

    def fake_gdn_forward(query, key, value, gate_g, gate_beta, initial_state, **kwargs):
        del key, gate_g, gate_beta, initial_state, kwargs
        return value.reshape(*query.shape[:-2], -1)

    monkeypatch.setattr(attention, "prepare_forward", traced_prepare_forward)
    monkeypatch.setattr(attention, "end_forward", traced_end_forward)
    monkeypatch.setattr(attention.gdn_attention, "forward", fake_gdn_forward)

    output = attention(q, k, v, g=g, beta=beta)
    assert output.shape == (1, 128, 64)
    assert events == ["prepare", "end"]
