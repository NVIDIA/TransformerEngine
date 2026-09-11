# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Tests for Gated DeltaNet v2 through the GatedDeltaNet2Attention API."""

import importlib.util
import math
import os
from typing import Optional, Tuple

import pytest
import torch
import torch.nn.functional as F

from transformer_engine.pytorch import GatedDeltaNet2Attention, autocast, is_fp8_available


def _gdn2_op_available() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        from cudnn.linear_attention.ops import gated_delta_net_v2  # noqa: F401
    except (AttributeError, ImportError):
        return False
    try:
        return importlib.util.find_spec("cutlass") is not None
    except (ImportError, ModuleNotFoundError):
        return False


def _gdn_op_available() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        from cudnn.linear_attention.ops import gated_delta_net  # noqa: F401
    except (AttributeError, ImportError):
        return False
    return True


def _gdn2_supported_arch() -> bool:
    """GDN-2's only cuDNN frontend engine is FROST, on SM100/SM103/SM107."""
    if not torch.cuda.is_available():
        return False
    major, minor = torch.cuda.get_device_capability()
    return (major, minor) in {(10, 0), (10, 1), (10, 2), (10, 3), (10, 7)}


_GDN2_OP_AVAILABLE = _gdn2_op_available()
if os.getenv("NVTE_GDN2_TEST_REQUIRED", "0") == "1" and not _GDN2_OP_AVAILABLE:
    raise RuntimeError(
        "NVTE_GDN2_TEST_REQUIRED=1, but the cuDNN frontend GDN-2 op or its "
        "cutedsl runtime is unavailable."
    )

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
requires_gdn2 = pytest.mark.skipif(
    not (_GDN2_OP_AVAILABLE and _gdn2_supported_arch()),
    reason="GDN-2 requires the cuDNN frontend FROST kernel runtime on SM100/SM103/SM107",
)
requires_gdn_and_gdn2 = pytest.mark.skipif(
    not (_GDN2_OP_AVAILABLE and _gdn2_supported_arch() and _gdn_op_available()),
    reason="the GDN/GDN-2 comparison requires both cuDNN frontend kernel runtimes",
)

_FWD_TOL = {torch.bfloat16: 2e-2, torch.float16: 1e-2}
_STATE_TOL = {torch.bfloat16: 2e-2, torch.float16: 1e-2}
_BWD_TOL = {torch.bfloat16: 4e-2, torch.float16: 2e-2}
# beta and w are kernel-native in the Q/K/V dtype, so their gradients come back
# with the io dtype's rounding rather than float32's.
_GATE_BWD_TOL = {torch.bfloat16: 6e-2, torch.float16: 3e-2}


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


def _gdn2_recurrence(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    w: torch.Tensor,
    state: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Evaluate GDN-2 for tensors in [batch, heads, sequence, ...] layout.

    ``state`` follows the cuDNN frontend's [..., v_head_dim, qk_head_dim]
    convention, so the per-key-channel gates scale its columns. The kernel
    applies the gates in this order: decay the state, erase from the decayed
    state, then write the gated value.
    """
    outputs = []
    for token_idx in range(q.shape[2]):
        q_t = q[:, :, token_idx]
        k_t = k[:, :, token_idx]
        v_t = v[:, :, token_idx]
        alpha_t = alpha[:, :, token_idx]
        beta_t = beta[:, :, token_idx]
        w_t = w[:, :, token_idx]

        state = alpha_t.unsqueeze(-2) * state
        erase = torch.matmul(state, (beta_t * k_t).unsqueeze(-1)).squeeze(-1)
        residual = w_t * v_t - erase
        state = state + residual.unsqueeze(-1) @ k_t.unsqueeze(-2)
        outputs.append(torch.matmul(state, q_t.unsqueeze(-1)).squeeze(-1))

    return torch.stack(outputs, dim=2), state


def _gdn2_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    w: torch.Tensor,
    *,
    scale: Optional[float] = None,
    initial_state: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    use_beta_sigmoid: bool = False,
    allow_neg_eigval: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pure PyTorch FP64 implementation of the Gated DeltaNet v2 recurrence.

    Inputs use [batch, sequence, heads, dimension] layout, with ``g``/``beta``
    carrying one value per query/key channel and ``w`` one per value channel.
    Packed inputs use a singleton batch dimension and provide sequence
    boundaries via cu_seqlens. ``initial_state``/the returned final state use
    the cuDNN frontend's [batch, heads, v_head_dim, qk_head_dim] convention
    throughout.
    """
    qk_dim = q.shape[-1]
    if scale is None:
        scale = 1.0 / math.sqrt(qk_dim)

    heads = q.shape[2]
    q_ref = q.double() * scale
    k_ref = k.double()
    v_ref = v.double()
    alpha_ref = g.double().exp()
    beta_ref = beta.double()
    if use_beta_sigmoid:
        beta_ref = beta_ref.sigmoid() * (2.0 if allow_neg_eigval else 1.0)
    w_ref = w.double()

    # [batch, sequence, heads, ...] -> [batch, heads, sequence, ...]
    q_ref, k_ref, v_ref, alpha_ref, beta_ref, w_ref = (
        tensor.permute(0, 2, 1, 3) for tensor in (q_ref, k_ref, v_ref, alpha_ref, beta_ref, w_ref)
    )

    def _initial_state(sequence_idx: Optional[int] = None) -> torch.Tensor:
        if initial_state is None:
            return torch.zeros(
                1 if sequence_idx is not None else q.shape[0],
                heads,
                v.shape[-1],
                qk_dim,
                dtype=torch.float64,
                device=q.device,
            )
        if sequence_idx is None:
            return initial_state.double()
        return initial_state[sequence_idx : sequence_idx + 1].double()

    if cu_seqlens is None:
        output, final_state = _gdn2_recurrence(
            q_ref, k_ref, v_ref, alpha_ref, beta_ref, w_ref, _initial_state()
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
        output, state = _gdn2_recurrence(
            q_ref[:, :, start:end],
            k_ref[:, :, start:end],
            v_ref[:, :, start:end],
            alpha_ref[:, :, start:end],
            beta_ref[:, :, start:end],
            w_ref[:, :, start:end],
            state,
        )
        outputs.append(output)
        final_states.append(state)

    if outputs:
        packed_output = torch.cat(outputs, dim=2).permute(0, 2, 1, 3)
    else:
        packed_output = q_ref.new_zeros(1, 0, heads, v.shape[-1])
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
    """Random GDN-2 inputs in [batch, sequence, heads, dimension] layout.

    The gates are drawn the way the kernel takes them: ``g`` in float32 log
    space, ``beta`` and ``w`` in the Q/K/V dtype.
    """
    torch.manual_seed(1234)
    q = torch.randn(batch, sequence, q_heads, qk_dim, device="cuda", dtype=dtype)
    k = F.normalize(torch.randn_like(q, dtype=torch.float32), dim=-1).to(dtype)
    v = torch.randn(batch, sequence, v_heads, v_dim, device="cuda", dtype=dtype)
    output_heads = max(q_heads, v_heads)
    # A per-channel decay in (0.5, 1] keeps the fp64 reference well conditioned.
    g = (
        torch.empty(batch, sequence, output_heads, qk_dim, device="cuda", dtype=torch.float32)
        .uniform_(0.5, 1.0)
        .log()
    )
    beta = torch.rand(batch, sequence, output_heads, qk_dim, device="cuda").to(dtype)
    w = torch.rand(batch, sequence, output_heads, v_dim, device="cuda").to(dtype)
    return q, k, v, g, beta, w


@requires_gdn2
@pytest.mark.parametrize("checkpoint_core_attention", [False, True], ids=["eager", "checkpoint"])
@pytest.mark.parametrize(
    ("qk_dim", "v_dim"),
    [(64, 64), (128, 128), (64, 128)],
    ids=["qk64_v64", "qk128_v128", "qk64_v128"],
)
@pytest.mark.parametrize("use_qk_l2norm_in_kernel", [False, True], ids=["no_l2norm", "l2norm"])
def test_gdn2_thd_forward_final_state_and_backward(
    checkpoint_core_attention, qk_dim, v_dim, use_qk_l2norm_in_kernel
):
    """THD GDN-2 matches a PyTorch recurrence in forward and backward."""
    batch, sequence, heads = 2, 128, 2
    q, k, v, g, beta, w = (
        tensor.reshape(batch * sequence, *tensor.shape[2:]).requires_grad_(True)
        for tensor in _inputs(batch, sequence, heads, heads, qk_dim, v_dim)
    )
    cu_seqlens = torch.arange(batch + 1, device="cuda", dtype=torch.int32) * sequence
    initial_state = (
        torch.randn(batch, heads, v_dim, qk_dim, device="cuda", dtype=torch.float32) * 0.05
    ).requires_grad_()

    attention = GatedDeltaNet2Attention(
        num_attention_heads=heads,
        kv_channels=(qk_dim, v_dim),
        qkv_format="thd",
    )
    output, final_state = attention(
        q,
        k,
        v,
        cu_seqlens=cu_seqlens,
        g=g,
        beta=beta,
        w=w,
        initial_state=initial_state,
        output_final_state=True,
        checkpoint_core_attention=checkpoint_core_attention,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
    )
    reference_inputs = {
        name: tensor.detach().double().reshape(1, -1, *tensor.shape[1:]).requires_grad_()
        for name, tensor in (("q", q), ("k", k), ("v", v), ("g", g), ("beta", beta), ("w", w))
    }
    initial_state_ref = initial_state.detach().double().requires_grad_()
    reference_q = reference_inputs["q"]
    reference_k = reference_inputs["k"]
    if use_qk_l2norm_in_kernel:
        reference_q = F.normalize(reference_q, dim=-1)
        reference_k = F.normalize(reference_k, dim=-1)
    output_ref, final_state_ref = _gdn2_reference(
        reference_q,
        reference_k,
        reference_inputs["v"],
        reference_inputs["g"],
        reference_inputs["beta"],
        reference_inputs["w"],
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

    for name, tensor in (("q", q), ("k", k), ("v", v), ("g", g), ("beta", beta), ("w", w)):
        assert tensor.grad is not None, f"no gradient for {name}"
        assert torch.isfinite(tensor.grad).all(), f"non-finite gradient for {name}"
        tolerance = _GATE_BWD_TOL[q.dtype] if name in ("beta", "w") else _BWD_TOL[q.dtype]
        reference_grad = reference_inputs[name].grad.reshape_as(tensor)
        _assert_rms_close(tensor.grad, reference_grad, tolerance, f"d{name}")
    assert initial_state.grad is not None, "no gradient for initial_state"
    _assert_rms_close(
        initial_state.grad,
        initial_state_ref.grad,
        _BWD_TOL[q.dtype],
        "dinitial_state",
    )


@pytest.mark.parametrize("qkv_format", ["bshd", "sbhd"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
@requires_gdn2
def test_gdn2_dense_layout(qkv_format, dtype):
    """Dense TE layouts and supported input dtypes match a PyTorch reference."""
    batch, sequence, heads, dim = 1, 128, 2, 64
    q, k, v, g, beta, w = _inputs(batch, sequence, heads, heads, dim, dim, dtype)
    with torch.no_grad():
        output_ref, _ = _gdn2_reference(
            F.normalize(q.float(), dim=-1),
            F.normalize(k.float(), dim=-1),
            v,
            g,
            beta,
            w,
        )

    if qkv_format == "sbhd":
        q, k, v, g, beta, w = (
            tensor.transpose(0, 1).contiguous() for tensor in (q, k, v, g, beta, w)
        )
        expected = output_ref.reshape(batch, sequence, -1).transpose(0, 1).contiguous()
    else:
        expected = output_ref.reshape(batch, sequence, -1)

    attention = GatedDeltaNet2Attention(
        num_attention_heads=heads,
        kv_channels=dim,
        qkv_format=qkv_format,
    )
    output = attention(q, k, v, g=g, beta=beta, w=w, use_qk_l2norm_in_kernel=True)
    assert output.shape == expected.shape
    _assert_rms_close(output, expected, _FWD_TOL[dtype], "output")


@requires_gdn2
def test_gdn2_state_round_trip_matches_single_shot():
    """A final state can seed the next chunk without changing the result."""
    batch, sequence, heads, dim = 2, 128, 2, 64
    split = 64
    q, k, v, g, beta, w = _inputs(batch, sequence, heads, heads, dim, dim)
    attention = GatedDeltaNet2Attention(
        num_attention_heads=heads,
        kv_channels=dim,
        qkv_format="bshd",
    )

    with torch.no_grad():
        full_output, full_state = attention(
            q, k, v, g=g, beta=beta, w=w, output_final_state=True
        )
        first_output, first_state = attention(
            q[:, :split],
            k[:, :split],
            v[:, :split],
            g=g[:, :split],
            beta=beta[:, :split],
            w=w[:, :split],
            output_final_state=True,
        )
        second_output, chunked_state = attention(
            q[:, split:],
            k[:, split:],
            v[:, split:],
            g=g[:, split:],
            beta=beta[:, split:],
            w=w[:, split:],
            initial_state=first_state,
            output_final_state=True,
        )

    chunked_output = torch.cat((first_output, second_output), dim=1)
    _assert_rms_close(chunked_output, full_output, _FWD_TOL[q.dtype], "chunked output")
    _assert_rms_close(chunked_state, full_state, _STATE_TOL[q.dtype], "chunked state")


@requires_gdn2
@pytest.mark.parametrize(
    "bounds",
    [(0, 48, 160), (0, 0, 64, 160)],
    ids=["unequal", "leading-empty"],
)
def test_gdn2_thd_ragged_sequences(bounds):
    """Packed GDN-2 handles unequal lengths and empty sequences."""
    total_tokens, heads, dim = bounds[-1], 2, 64
    q, k, v, g, beta, w = (
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
    attention = GatedDeltaNet2Attention(
        num_attention_heads=heads,
        kv_channels=dim,
        qkv_format="thd",
    )

    with torch.no_grad():
        output, final_state = attention(
            q,
            k,
            v,
            cu_seqlens=cu_seqlens,
            g=g,
            beta=beta,
            w=w,
            initial_state=initial_state,
            output_final_state=True,
        )
        output_ref, final_state_ref = _gdn2_reference(
            q.unsqueeze(0),
            k.unsqueeze(0),
            v.unsqueeze(0),
            g.unsqueeze(0),
            beta.unsqueeze(0),
            w.unsqueeze(0),
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


@requires_gdn2
@pytest.mark.parametrize("allow_neg_eigval", [False, True], ids=["projection", "reflection"])
def test_gdn2_fused_beta_sigmoid(allow_neg_eigval):
    """The kernel's fused beta activation matches applying it in the reference."""
    batch, sequence, heads, dim = 1, 128, 2, 64
    q, k, v, g, _, w = _inputs(batch, sequence, heads, heads, dim, dim)
    # Raw logits, which the kernel turns into the erase gate itself.
    beta = torch.randn(batch, sequence, heads, dim, device="cuda", dtype=q.dtype)

    attention = GatedDeltaNet2Attention(
        num_attention_heads=heads,
        kv_channels=dim,
        qkv_format="bshd",
    )
    with torch.no_grad():
        output = attention(
            q,
            k,
            v,
            g=g,
            beta=beta,
            w=w,
            use_qk_l2norm_in_kernel=True,
            use_beta_sigmoid_in_kernel=True,
            allow_neg_eigval=allow_neg_eigval,
        )
        output_ref, _ = _gdn2_reference(
            F.normalize(q.float(), dim=-1),
            F.normalize(k.float(), dim=-1),
            v,
            g,
            beta,
            w,
            use_beta_sigmoid=True,
            allow_neg_eigval=allow_neg_eigval,
        )

    _assert_rms_close(
        output,
        output_ref.reshape(batch, sequence, -1),
        _FWD_TOL[q.dtype],
        "fused-sigmoid output",
    )


@requires_gdn2
def test_gdn2_beta_guard_runs():
    """The erase-side beta safeguard produces finite outputs and gradients."""
    batch, sequence, heads, dim = 1, 128, 2, 64
    q, k, v, g, beta, w = _inputs(batch, sequence, heads, heads, dim, dim)
    # A high-contrast beta is what the guard exists for.
    beta = (beta.float() * 4.0).to(beta.dtype)
    q, k, v, g, beta, w = (tensor.detach().requires_grad_() for tensor in (q, k, v, g, beta, w))

    attention = GatedDeltaNet2Attention(
        num_attention_heads=heads,
        kv_channels=dim,
        qkv_format="bshd",
    )
    output = attention(
        q,
        k,
        v,
        g=g,
        beta=beta,
        w=w,
        use_qk_l2norm_in_kernel=True,
        beta_guard=True,
    )
    assert output.shape == (batch, sequence, heads * dim)
    assert torch.isfinite(output).all(), "non-finite beta-guarded output"

    output.float().sum().backward()
    for name, tensor in (("q", q), ("k", k), ("v", v), ("g", g), ("beta", beta), ("w", w)):
        assert tensor.grad is not None, f"no gradient for {name}"
        assert torch.isfinite(tensor.grad).all(), f"non-finite gradient for {name}"


@requires_gdn_and_gdn2
def test_gdn2_matches_gdn_with_scalar_gates():
    """Broadcasting GDN's scalar gates over the channels reproduces GDN."""
    from transformer_engine.pytorch import GatedDeltaNetAttention

    batch, sequence, heads, dim = 1, 128, 2, 128
    q, k, v, _, _, _ = _inputs(batch, sequence, heads, heads, dim, dim)
    g = torch.rand(batch, sequence, heads, device="cuda", dtype=torch.float32).log()
    beta = torch.rand(batch, sequence, heads, device="cuda", dtype=torch.float32)

    gdn = GatedDeltaNetAttention(num_attention_heads=heads, kv_channels=dim, qkv_format="bshd")
    gdn2 = GatedDeltaNet2Attention(
        num_attention_heads=heads, kv_channels=dim, qkv_format="bshd"
    )
    with torch.no_grad():
        expected = gdn(q, k, v, g=g, beta=beta, use_qk_l2norm_in_kernel=True)
        # GDN applies beta to both the erase and the write; GDN-2 splits those
        # into the per-key erase gate and the per-value write gate.
        output = gdn2(
            q,
            k,
            v,
            g=g.unsqueeze(-1).expand(-1, -1, -1, dim).contiguous(),
            beta=beta.unsqueeze(-1).expand(-1, -1, -1, dim).to(q.dtype).contiguous(),
            w=beta.unsqueeze(-1).expand(-1, -1, -1, dim).to(q.dtype).contiguous(),
            use_qk_l2norm_in_kernel=True,
        )

    # Looser than _FWD_TOL: this compares two bf16 kernels, and GDN-2 reads the
    # gates at the io dtype where GDN reads them in float32.
    _assert_rms_close(output, expected, 3e-2, "GDN-2 against GDN")


def test_gdn2_rejects_value_head_count_that_changes_output_width():
    """GatedDeltaNet2Attention's output width cannot be changed by the runtime V tensor."""
    q, k, v, g, beta, w = _inputs(1, 128, 1, 2)
    attention = GatedDeltaNet2Attention(
        num_attention_heads=1,
        kv_channels=64,
        qkv_format="bshd",
    )
    with pytest.raises(ValueError, match="GDN2 V must have 1 heads"):
        attention(q, k, v, g=g, beta=beta, w=w)


def test_gdn2_requires_all_gates():
    """A partial GDN-2 invocation fails before entering a softmax-attention backend."""
    q, k, v, g, beta, _ = _inputs(1, 128, 1, 1)
    attention = GatedDeltaNet2Attention(
        num_attention_heads=1,
        kv_channels=64,
        qkv_format="bshd",
    )
    with pytest.raises(ValueError, match="requires all of g, beta, and w; got no w"):
        attention(q, k, v, g=g, beta=beta)


def test_gdn2_rejects_scalar_gates():
    """GDN-2's gates are channel-wise; GDN's scalar gates are a shape error."""
    q, k, v, _, _, w = _inputs(1, 128, 1, 1)
    scalar = torch.rand(1, 128, 1, device="cuda", dtype=q.dtype)
    attention = GatedDeltaNet2Attention(
        num_attention_heads=1,
        kv_channels=64,
        qkv_format="bshd",
    )
    with pytest.raises(ValueError, match="GDN2 g and beta must both have shape"):
        attention(q, k, v, g=scalar, beta=scalar, w=w)


def test_gdn2_rejects_gate_dtypes():
    """beta and w are kernel-native in the Q/K/V dtype."""
    q, k, v, g, beta, w = _inputs(1, 128, 1, 1)
    attention = GatedDeltaNet2Attention(
        num_attention_heads=1,
        kv_channels=64,
        qkv_format="bshd",
    )
    with pytest.raises(TypeError, match="GDN2 beta and w must have the same dtype"):
        attention(q, k, v, g=g, beta=beta.float(), w=w)
    with pytest.raises(TypeError, match="GDN2 g must have dtype"):
        attention(q, k, v, g=g.double(), beta=beta, w=w)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"allow_neg_eigval": True}, "allow_neg_eigval requires use_beta_sigmoid_in_kernel"),
        ({"beta_guard": True}, "beta_guard requires use_qk_l2norm_in_kernel"),
    ],
    ids=["neg_eigval_without_sigmoid", "guard_without_l2norm"],
)
def test_gdn2_rejects_unsupported_flag_combinations(kwargs, message):
    """Flags the kernel couples are checked in TE, with a TE-level message."""
    q, k, v, g, beta, w = _inputs(1, 128, 1, 1)
    attention = GatedDeltaNet2Attention(
        num_attention_heads=1,
        kv_channels=64,
        qkv_format="bshd",
    )
    with pytest.raises(ValueError, match=message):
        attention(q, k, v, g=g, beta=beta, w=w, **kwargs)


@pytest.mark.skipif(not is_fp8_available(), reason="FP8 is not available")
def test_gdn2_rejects_fp8_autocast():
    """GDN-2 must not silently run in high precision inside FP8 autocast."""
    q, k, v, g, beta, w = _inputs(1, 128, 1, 1)
    attention = GatedDeltaNet2Attention(
        num_attention_heads=1,
        kv_channels=64,
        qkv_format="bshd",
    )
    with autocast(enabled=True), pytest.raises(ValueError, match="does not support FP8 autocast"):
        attention(q, k, v, g=g, beta=beta, w=w)


def test_gdn2_module_carries_no_quantization_state():
    """LinearAttentionBase is a plain nn.Module: no parameters, no FP8 state."""
    from transformer_engine.pytorch.module.base import TransformerEngineBaseModule

    attention = GatedDeltaNet2Attention(
        num_attention_heads=1,
        kv_channels=64,
        qkv_format="bshd",
    )
    assert isinstance(attention, torch.nn.Module)
    assert not isinstance(attention, TransformerEngineBaseModule)
    assert list(attention.parameters()) == []
    # No FP8 meta means no `_extra_state` entries to carry in a checkpoint.
    assert dict(attention.state_dict()) == {}


def test_gdn2_requires_tensor_parallel_group_when_sharded():
    """The TP handshake is enforced in TE rather than deferred to the kernel."""
    q, k, v, g, beta, w = _inputs(1, 128, 1, 1)
    attention = GatedDeltaNet2Attention(
        num_attention_heads=2,
        kv_channels=64,
        qkv_format="bshd",
        tp_size=2,
    )
    assert attention.num_attention_heads_per_partition == 1
    assert not attention.tp_group_initialized
    with pytest.raises(RuntimeError, match="Tensor parallel group not initialized"):
        attention(q, k, v, g=g, beta=beta, w=w)

    attention.set_tensor_parallel_group(None)
    assert attention.tp_group is None
    assert attention.tp_group_initialized


def test_gdn2_runs_te_forward_lifecycle(monkeypatch):
    """GDN-2 calls pair prepare_forward with end_forward even without the kernel runtime."""
    q, k, v, g, beta, w = _inputs(1, 128, 1, 1)
    attention = GatedDeltaNet2Attention(
        num_attention_heads=1,
        kv_channels=64,
        qkv_format="bshd",
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

    def fake_gdn2_forward(query, key, value, gate_g, gate_beta, gate_w, initial_state, **kwargs):
        del key, gate_g, gate_beta, gate_w, initial_state, kwargs
        return value.reshape(*query.shape[:-2], -1)

    monkeypatch.setattr(attention, "prepare_forward", traced_prepare_forward)
    monkeypatch.setattr(attention, "end_forward", traced_end_forward)
    monkeypatch.setattr(attention.gdn2_attention, "forward", fake_gdn2_forward)

    output = attention(q, k, v, g=g, beta=beta, w=w)
    assert output.shape == (1, 128, 64)
    assert events == ["prepare", "end"]
