# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Tests for Gated DeltaProduct through the GatedDeltaProductAttention API."""

import importlib.util
import math
import os
from typing import Optional, Tuple

import pytest
import torch
import torch.nn.functional as F

from transformer_engine.pytorch import GatedDeltaProductAttention


def _gdp_op_available() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        from cudnn.linear_attention.ops import gated_delta_product  # noqa: F401
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


def _gdp_supported_arch() -> bool:
    """GDP's only cuDNN frontend engine is FROST, on SM100/SM103/SM107."""
    if not torch.cuda.is_available():
        return False
    major, minor = torch.cuda.get_device_capability()
    return (major, minor) in {(10, 0), (10, 3), (10, 7)}


_GDP_OP_AVAILABLE = _gdp_op_available()
if os.getenv("NVTE_GDP_TEST_REQUIRED", "0") == "1" and not _GDP_OP_AVAILABLE:
    raise RuntimeError(
        "NVTE_GDP_TEST_REQUIRED=1, but the cuDNN frontend GDP op or its "
        "cutedsl runtime is unavailable."
    )

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
requires_gdp = pytest.mark.skipif(
    not (_GDP_OP_AVAILABLE and _gdp_supported_arch()),
    reason="GDP requires the cuDNN frontend FROST kernel runtime on SM100/SM103/SM107",
)
requires_gdn_and_gdp = pytest.mark.skipif(
    not (_GDP_OP_AVAILABLE and _gdp_supported_arch() and _gdn_op_available()),
    reason="the GDN/GDP comparison requires both cuDNN frontend kernel runtimes",
)

_FWD_TOL = {torch.bfloat16: 2e-2, torch.float16: 1e-2}
_STATE_TOL = {torch.bfloat16: 2e-2, torch.float16: 1e-2}
_BWD_TOL = {torch.bfloat16: 4e-2, torch.float16: 2e-2}
# a_log and dt_bias hold one value per head, so their gradients are a reduction
# over every token and sub-token and carry that accumulation's rounding.
_PARAM_BWD_TOL = {torch.bfloat16: 6e-2, torch.float16: 3e-2}


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


def _gdp_recurrence(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    state: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Evaluate GDP for tensors in [batch, heads, sequence, ...] layout.

    ``k``, ``v`` and ``beta`` carry their num_householder axis after the
    sequence axis. ``state`` follows the cuDNN frontend's
    [..., v_head_dim, qk_head_dim] convention. The decay acts once per token,
    before that token's Householder updates, and the readout follows the last
    one.
    """
    num_householder = k.shape[3]
    outputs = []
    for token_idx in range(q.shape[2]):
        state = alpha[:, :, token_idx][..., None, None] * state
        for update_idx in range(num_householder):
            k_j = k[:, :, token_idx, update_idx]
            v_j = v[:, :, token_idx, update_idx]
            beta_j = beta[:, :, token_idx, update_idx]

            prediction = torch.matmul(state, k_j.unsqueeze(-1)).squeeze(-1)
            residual = v_j - prediction
            state = state + beta_j[..., None, None] * (residual.unsqueeze(-1) @ k_j.unsqueeze(-2))
        outputs.append(torch.matmul(state, q[:, :, token_idx].unsqueeze(-1)).squeeze(-1))

    return torch.stack(outputs, dim=2), state


def _gdp_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    scale: Optional[float] = None,
    initial_state: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    use_beta_sigmoid: bool = False,
    allow_neg_eigval: bool = False,
    gate_domain: str = "log",
    safe_gate: bool = False,
    a_log: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pure PyTorch FP64 implementation of the Gated DeltaProduct recurrence.

    Inputs use the module's layout: ``q``/``g`` are
    [batch, sequence, heads, ...] on the real-token timeline, while
    ``k``/``v``/``beta`` insert a num_householder axis after the sequence axis.
    Packed inputs use a singleton batch dimension and provide sequence
    boundaries via cu_seqlens, which count real tokens. ``initial_state``/the
    returned final state use the cuDNN frontend's
    [batch, heads, v_head_dim, qk_head_dim] convention throughout.
    """
    qk_dim = q.shape[-1]
    if scale is None:
        scale = 1.0 / math.sqrt(qk_dim)

    heads = q.shape[2]
    if safe_gate:
        # ln(alpha) = -exp(a_log) * softplus(g + dt_bias), with the per-head
        # parameters broadcasting over the token dimensions.
        amplitude = 1.0 if a_log is None else a_log.double().exp()
        bias = 0.0 if dt_bias is None else dt_bias.double()
        alpha_ref = (-amplitude * F.softplus(g.double() + bias)).exp()
    elif gate_domain == "linear":
        alpha_ref = g.double().clamp_min(1e-10)
    else:
        alpha_ref = g.double().exp()
    beta_ref = beta.double()
    if use_beta_sigmoid:
        beta_ref = beta_ref.sigmoid() * (2.0 if allow_neg_eigval else 1.0)

    # -> [batch, heads, sequence, ...], with the householder axis kept after the
    # sequence axis so that slicing a sequence slices its sub-tokens with it.
    q_ref = (q.double() * scale).permute(0, 2, 1, 3)
    alpha_ref = alpha_ref.permute(0, 2, 1)
    k_ref, v_ref = (tensor.double().permute(0, 3, 1, 2, 4) for tensor in (k, v))
    beta_ref = beta_ref.permute(0, 3, 1, 2)

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
        output, final_state = _gdp_recurrence(
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
        output, state = _gdp_recurrence(
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
        packed_output = q_ref.new_zeros(1, 0, heads, v.shape[-1])
    return packed_output, torch.cat(final_states, dim=0)


def _inputs(
    batch,
    sequence,
    heads,
    num_householder,
    qk_dim=64,
    v_dim=64,
    dtype=torch.bfloat16,
):
    """Random GDP inputs in the module's layout.

    ``q``/``g`` live on the real-token timeline; ``k``/``v``/``beta`` carry a
    num_householder axis after the sequence axis. The gates are drawn the way
    the kernel takes them: ``g`` in float32 log space and ``beta`` in float32.
    """
    torch.manual_seed(1234)
    q = torch.randn(batch, sequence, heads, qk_dim, device="cuda", dtype=dtype)
    k = F.normalize(
        torch.randn(
            batch, sequence, num_householder, heads, qk_dim, device="cuda", dtype=torch.float32
        ),
        dim=-1,
    ).to(dtype)
    v = torch.randn(batch, sequence, num_householder, heads, v_dim, device="cuda", dtype=dtype)
    # A decay in (0.5, 1] keeps the fp64 reference well conditioned.
    g = (
        torch.empty(batch, sequence, heads, device="cuda", dtype=torch.float32)
        .uniform_(0.5, 1.0)
        .log()
    )
    beta = torch.rand(batch, sequence, num_householder, heads, device="cuda", dtype=torch.float32)
    return q, k, v, g, beta


@requires_gdp
@pytest.mark.parametrize("checkpoint_core_attention", [False, True], ids=["eager", "checkpoint"])
@pytest.mark.parametrize("num_householder", [1, 2, 3], ids=["n1", "n2", "n3"])
@pytest.mark.parametrize(
    ("qk_dim", "v_dim"),
    [(64, 64), (128, 128), (64, 128)],
    ids=["qk64_v64", "qk128_v128", "qk64_v128"],
)
def test_gdp_thd_forward_final_state_and_backward(
    checkpoint_core_attention, num_householder, qk_dim, v_dim
):
    """THD GDP matches a PyTorch recurrence in forward and backward."""
    batch, sequence, heads = 2, 128, 2
    q, k, v, g, beta = _inputs(batch, sequence, heads, num_householder, qk_dim, v_dim)
    # THD flattens only the real-token timeline; the householder axis stays put.
    q, k, v, g, beta = (
        tensor.reshape(batch * sequence, *tensor.shape[2:]).requires_grad_(True)
        for tensor in (q, k, v, g, beta)
    )
    cu_seqlens = torch.arange(batch + 1, device="cuda", dtype=torch.int32) * sequence
    initial_state = (
        torch.randn(batch, heads, v_dim, qk_dim, device="cuda", dtype=torch.float32) * 0.05
    ).requires_grad_()

    attention = GatedDeltaProductAttention(
        num_attention_heads=heads,
        kv_channels=(qk_dim, v_dim),
        num_householder=num_householder,
        qkv_format="thd",
    )
    output, final_state = attention(
        q,
        k,
        v,
        cu_seqlens=cu_seqlens,
        g=g,
        beta=beta,
        initial_state=initial_state,
        output_final_state=True,
        checkpoint_core_attention=checkpoint_core_attention,
    )

    reference_inputs = {
        name: tensor.detach().double().reshape(1, -1, *tensor.shape[1:]).requires_grad_()
        for name, tensor in (("q", q), ("k", k), ("v", v), ("g", g), ("beta", beta))
    }
    initial_state_ref = initial_state.detach().double().requires_grad_()
    output_ref, final_state_ref = _gdp_reference(
        reference_inputs["q"],
        reference_inputs["k"],
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


@requires_gdp
@pytest.mark.parametrize("qkv_format", ["bshd", "sbhd"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
def test_gdp_dense_layout(qkv_format, dtype):
    """Dense TE layouts and supported input dtypes match a PyTorch reference."""
    batch, sequence, heads, dim, num_householder = 1, 128, 2, 64, 2
    q, k, v, g, beta = _inputs(batch, sequence, heads, num_householder, dim, dim, dtype)
    with torch.no_grad():
        output_ref, _ = _gdp_reference(
            F.normalize(q.float(), dim=-1),
            F.normalize(k.float(), dim=-1),
            v,
            g,
            beta,
        )

    if qkv_format == "sbhd":
        # sbhd swaps the batch and sequence axes; the householder axis follows
        # the sequence axis in either layout.
        q, g = (tensor.transpose(0, 1).contiguous() for tensor in (q, g))
        k, v, beta = (tensor.transpose(0, 1).contiguous() for tensor in (k, v, beta))
        expected = output_ref.reshape(batch, sequence, -1).transpose(0, 1).contiguous()
    else:
        expected = output_ref.reshape(batch, sequence, -1)

    attention = GatedDeltaProductAttention(
        num_attention_heads=heads,
        kv_channels=dim,
        num_householder=num_householder,
        qkv_format=qkv_format,
    )
    output = attention(q, k, v, g=g, beta=beta, use_qk_l2norm_in_kernel=True)
    assert output.shape == expected.shape
    _assert_rms_close(output, expected, _FWD_TOL[dtype], "output")


@requires_gdp
def test_gdp_state_round_trip_matches_single_shot():
    """A final state can seed the next chunk without changing the result."""
    batch, sequence, heads, dim, num_householder = 2, 128, 2, 64, 2
    split = 64
    q, k, v, g, beta = _inputs(batch, sequence, heads, num_householder, dim, dim)
    attention = GatedDeltaProductAttention(
        num_attention_heads=heads,
        kv_channels=dim,
        num_householder=num_householder,
        qkv_format="bshd",
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


@requires_gdp
@pytest.mark.parametrize(
    "bounds",
    [(0, 48, 160), (0, 0, 64, 160)],
    ids=["unequal", "leading-empty"],
)
def test_gdp_thd_ragged_sequences(bounds):
    """Packed GDP handles unequal lengths and empty sequences.

    ``cu_seqlens`` counts real tokens even though K/V/beta carry
    ``num_householder`` rows per token.
    """
    total_tokens, heads, dim, num_householder = bounds[-1], 2, 64, 2
    q, k, v, g, beta = (
        tensor.squeeze(0)
        for tensor in _inputs(1, total_tokens, heads, num_householder, dim, dim)
    )
    cu_seqlens = torch.tensor(bounds, device="cuda", dtype=torch.int32)
    initial_state = torch.randn(
        len(bounds) - 1, heads, dim, dim, device="cuda", dtype=torch.float32
    )
    attention = GatedDeltaProductAttention(
        num_attention_heads=heads,
        kv_channels=dim,
        num_householder=num_householder,
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
            initial_state=initial_state,
            output_final_state=True,
        )
        output_ref, final_state_ref = _gdp_reference(
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


@requires_gdp
@pytest.mark.parametrize("allow_neg_eigval", [False, True], ids=["projection", "reflection"])
def test_gdp_fused_beta_sigmoid(allow_neg_eigval):
    """The kernel's fused beta activation matches applying it in the reference."""
    batch, sequence, heads, dim, num_householder = 1, 128, 2, 64, 2
    q, k, v, g, _ = _inputs(batch, sequence, heads, num_householder, dim, dim)
    # Raw logits, which the kernel turns into the write gate itself.
    beta = torch.randn(batch, sequence, num_householder, heads, device="cuda", dtype=torch.float32)

    attention = GatedDeltaProductAttention(
        num_attention_heads=heads,
        kv_channels=dim,
        num_householder=num_householder,
        qkv_format="bshd",
    )
    with torch.no_grad():
        output = attention(
            q,
            k,
            v,
            g=g,
            beta=beta,
            use_qk_l2norm_in_kernel=True,
            use_beta_sigmoid_in_kernel=True,
            allow_neg_eigval=allow_neg_eigval,
        )
        output_ref, _ = _gdp_reference(
            F.normalize(q.float(), dim=-1),
            F.normalize(k.float(), dim=-1),
            v,
            g,
            beta,
            use_beta_sigmoid=True,
            allow_neg_eigval=allow_neg_eigval,
        )

    _assert_rms_close(
        output,
        output_ref.reshape(batch, sequence, -1),
        _FWD_TOL[q.dtype],
        "fused-sigmoid output",
    )


@requires_gdp
@pytest.mark.parametrize("checkpoint_core_attention", [False, True], ids=["eager", "checkpoint"])
@pytest.mark.parametrize(
    ("with_a_log", "with_dt_bias"),
    [(False, False), (True, False), (False, True), (True, True)],
    ids=["bare", "a_log", "dt_bias", "both"],
)
def test_gdp_safe_gate(with_a_log, with_dt_bias, checkpoint_core_attention):
    """The fused safe gate matches -exp(a_log) * softplus(g + dt_bias).

    Parametrized over activation checkpointing because a_log/dt_bias take
    gradients, and TE's reentrant checkpoint only returns gradients for the
    positional arguments it forwards.
    """
    batch, sequence, heads, dim, num_householder = 1, 128, 2, 64, 2
    q, k, v, _, beta = _inputs(batch, sequence, heads, num_householder, dim, dim)
    q, k, v, beta = (tensor.detach().requires_grad_() for tensor in (q, k, v, beta))
    # Raw logits, which the kernel maps to a log decay itself.
    g = torch.randn(batch, sequence, heads, device="cuda", dtype=torch.float32).requires_grad_()
    # A small amplitude keeps alpha near 1, so the decay does not wash the
    # sequence out before the gradient comparison can see it.
    a_log = None
    if with_a_log:
        a_log = torch.full((heads,), -2.0, device="cuda", dtype=torch.float32).requires_grad_()
    dt_bias = None
    if with_dt_bias:
        dt_bias = (
            torch.empty(heads, device="cuda", dtype=torch.float32)
            .uniform_(-0.5, 0.5)
            .requires_grad_()
        )

    attention = GatedDeltaProductAttention(
        num_attention_heads=heads,
        kv_channels=dim,
        num_householder=num_householder,
        qkv_format="bshd",
    )
    output = attention(
        q,
        k,
        v,
        g=g,
        beta=beta,
        safe_gate=True,
        a_log=a_log,
        dt_bias=dt_bias,
        checkpoint_core_attention=checkpoint_core_attention,
    )

    reference = {
        name: tensor.detach().double().requires_grad_()
        for name, tensor in (("q", q), ("k", k), ("v", v), ("g", g), ("beta", beta))
    }
    a_log_ref = a_log.detach().double().requires_grad_() if with_a_log else None
    dt_bias_ref = dt_bias.detach().double().requires_grad_() if with_dt_bias else None
    output_ref, _ = _gdp_reference(
        reference["q"],
        reference["k"],
        reference["v"],
        reference["g"],
        reference["beta"],
        safe_gate=True,
        a_log=a_log_ref,
        dt_bias=dt_bias_ref,
    )
    output_ref = output_ref.reshape(batch, sequence, -1)
    _assert_rms_close(output, output_ref, _FWD_TOL[q.dtype], "safe-gate output")

    weight = torch.randn_like(output, dtype=torch.float32)
    (output.float() * weight).sum().backward()
    (output_ref * weight.double()).sum().backward()

    for name in ("q", "k", "v", "g", "beta"):
        tensor = {"q": q, "k": k, "v": v, "g": g, "beta": beta}[name]
        assert tensor.grad is not None, f"no gradient for {name}"
        assert torch.isfinite(tensor.grad).all(), f"non-finite gradient for {name}"
        _assert_rms_close(tensor.grad, reference[name].grad, _BWD_TOL[q.dtype], f"d{name}")
    for name, tensor, tensor_ref in (
        ("a_log", a_log, a_log_ref),
        ("dt_bias", dt_bias, dt_bias_ref),
    ):
        if tensor is None:
            continue
        assert tensor.grad is not None, f"no gradient for {name}"
        _assert_rms_close(tensor.grad, tensor_ref.grad, _PARAM_BWD_TOL[q.dtype], f"d{name}")


@requires_gdp
def test_gdp_gate_domain_linear():
    """gate_domain='linear' reads g as the decay itself rather than its log."""
    batch, sequence, heads, dim, num_householder = 1, 128, 2, 64, 2
    q, k, v, g, beta = _inputs(batch, sequence, heads, num_householder, dim, dim)
    # _inputs draws g in log space; the linear domain wants alpha directly.
    alpha = g.exp()

    attention = GatedDeltaProductAttention(
        num_attention_heads=heads,
        kv_channels=dim,
        num_householder=num_householder,
        qkv_format="bshd",
    )
    with torch.no_grad():
        output = attention(q, k, v, g=alpha, beta=beta, gate_domain="linear")
        output_ref, _ = _gdp_reference(q, k, v, alpha, beta, gate_domain="linear")

    _assert_rms_close(
        output,
        output_ref.reshape(batch, sequence, -1),
        _FWD_TOL[q.dtype],
        "linear-domain output",
    )


@requires_gdp
def test_gdp_gate_domains_agree():
    """Passing alpha in the linear domain matches passing ln(alpha) in the log domain."""
    batch, sequence, heads, dim, num_householder = 1, 128, 2, 64, 2
    q, k, v, g, beta = _inputs(batch, sequence, heads, num_householder, dim, dim)

    attention = GatedDeltaProductAttention(
        num_attention_heads=heads,
        kv_channels=dim,
        num_householder=num_householder,
        qkv_format="bshd",
    )
    with torch.no_grad():
        log_domain = attention(q, k, v, g=g, beta=beta)
        linear_domain = attention(q, k, v, g=g.exp(), beta=beta, gate_domain="linear")

    _assert_rms_close(linear_domain, log_domain, _FWD_TOL[q.dtype], "linear against log domain")


@requires_gdp
def test_gdp_extra_householder_updates_change_the_output():
    """A second Householder update is not a no-op on the same first update.

    Guards against the num_householder axis being silently dropped: the n=2
    result must differ from the n=1 result that shares its first sub-token.
    """
    batch, sequence, heads, dim = 1, 128, 2, 64
    q, k, v, g, beta = _inputs(batch, sequence, heads, 2, dim, dim)

    with torch.no_grad():
        two = GatedDeltaProductAttention(
            num_attention_heads=heads, kv_channels=dim, num_householder=2, qkv_format="bshd"
        )(q, k, v, g=g, beta=beta)
        one = GatedDeltaProductAttention(
            num_attention_heads=heads, kv_channels=dim, num_householder=1, qkv_format="bshd"
        )(q, k[:, :, :1], v[:, :, :1], g=g, beta=beta[:, :, :1])

    assert _rms_ratio(two, one) > 1e-2, "the second Householder update did not affect the output"


@requires_gdn_and_gdp
def test_gdp_with_one_householder_matches_gdn():
    """num_householder=1 is exactly Gated DeltaNet."""
    from transformer_engine.pytorch import GatedDeltaNetAttention

    batch, sequence, heads, dim = 1, 128, 2, 128
    q, k, v, g, beta = _inputs(batch, sequence, heads, 1, dim, dim)

    gdn = GatedDeltaNetAttention(num_attention_heads=heads, kv_channels=dim, qkv_format="bshd")
    gdp = GatedDeltaProductAttention(
        num_attention_heads=heads, kv_channels=dim, num_householder=1, qkv_format="bshd"
    )
    with torch.no_grad():
        # GDN has no householder axis, so drop the singleton one.
        expected = gdn(
            q,
            k.squeeze(2),
            v.squeeze(2),
            g=g,
            beta=beta.squeeze(2),
            use_qk_l2norm_in_kernel=True,
        )
        output = gdp(q, k, v, g=g, beta=beta, use_qk_l2norm_in_kernel=True)

    _assert_rms_close(output, expected, _FWD_TOL[q.dtype], "GDP against GDN")


def test_gdp_requires_both_gates():
    """A partial GDP invocation fails before entering a softmax-attention backend."""
    q, k, v, g, _ = _inputs(1, 128, 1, 2)
    attention = GatedDeltaProductAttention(
        num_attention_heads=1,
        kv_channels=64,
        num_householder=2,
        qkv_format="bshd",
    )
    with pytest.raises(ValueError, match="requires both g and beta"):
        attention(q, k, v, g=g)
