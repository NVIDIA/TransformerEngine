# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Tests for Kimi Delta Attention through the KimiDeltaAttention API."""

import importlib.util
import math
import os
from typing import Optional, Tuple

import pytest
import torch
import torch.nn.functional as F

from transformer_engine.pytorch import KimiDeltaAttention


def _kda_op_available() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        from cudnn.linear_attention.ops import kimi_delta_attention  # noqa: F401
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


def _kda_supported_arch() -> bool:
    """KDA's default cuDNN frontend engine is FROST, on SM100/SM103/SM107."""
    if not torch.cuda.is_available():
        return False
    major, minor = torch.cuda.get_device_capability()
    return (major, minor) in {(10, 0), (10, 3), (10, 7)}


_KDA_OP_AVAILABLE = _kda_op_available()
if os.getenv("NVTE_KDA_TEST_REQUIRED", "0") == "1" and not _KDA_OP_AVAILABLE:
    raise RuntimeError(
        "NVTE_KDA_TEST_REQUIRED=1, but the cuDNN frontend KDA op or its "
        "cutedsl runtime is unavailable."
    )

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
requires_kda = pytest.mark.skipif(
    not (_KDA_OP_AVAILABLE and _kda_supported_arch()),
    reason="KDA requires the cuDNN frontend FROST kernel runtime on SM100/SM103/SM107",
)
requires_gdn_and_kda = pytest.mark.skipif(
    not (_KDA_OP_AVAILABLE and _kda_supported_arch() and _gdn_op_available()),
    reason="the GDN/KDA comparison requires both cuDNN frontend kernel runtimes",
)

_FWD_TOL = {torch.bfloat16: 2e-2, torch.float16: 1e-2}
_STATE_TOL = {torch.bfloat16: 2e-2, torch.float16: 1e-2}
_BWD_TOL = {torch.bfloat16: 4e-2, torch.float16: 2e-2}
# A gate gradient accumulates over the whole suffix of the sequence, so it
# carries more of the recurrence's rounding than the Q/K/V gradients do.
_GATE_BWD_TOL = {torch.bfloat16: 6e-2, torch.float16: 3e-2}
# a_log holds one value per head and dt_bias one per head and channel, so their
# gradients are a reduction over every token and carry that accumulation's
# rounding.
_PARAM_BWD_TOL = {torch.bfloat16: 6e-2, torch.float16: 3e-2}

# The kernel's own safe-gate lower bound, used when gate_lower_bound is None.
_DEFAULT_GATE_LOWER_BOUND = -5.0


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


def _kda_recurrence(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    state: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Evaluate KDA for tensors in [batch, heads, sequence, ...] layout.

    ``state`` follows the cuDNN frontend's [..., v_head_dim, qk_head_dim]
    convention, so the per-key-channel decay scales its columns. The decay is
    applied first and the delta-rule correction reads the already-decayed
    state; ``beta`` is a single write strength per token and head.
    """
    outputs = []
    for token_idx in range(q.shape[2]):
        q_t = q[:, :, token_idx]
        k_t = k[:, :, token_idx]
        v_t = v[:, :, token_idx]
        alpha_t = alpha[:, :, token_idx]
        beta_t = beta[:, :, token_idx]

        state = alpha_t.unsqueeze(-2) * state
        erase = torch.matmul(state, k_t.unsqueeze(-1)).squeeze(-1)
        residual = v_t - erase
        state = state + beta_t[..., None, None] * (residual.unsqueeze(-1) @ k_t.unsqueeze(-2))
        outputs.append(torch.matmul(state, q_t.unsqueeze(-1)).squeeze(-1))

    return torch.stack(outputs, dim=2), state


def _kda_reference(
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
    safe_gate: bool = False,
    gate_lower_bound: Optional[float] = None,
    gate_domain: str = "log",
    a_log: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pure PyTorch FP64 implementation of the Kimi Delta Attention recurrence.

    Inputs use [batch, sequence, heads, dimension] layout, with ``g`` carrying
    one value per query/key channel and ``beta`` one per head. Packed inputs use
    a singleton batch dimension and provide sequence boundaries via cu_seqlens.
    ``initial_state``/the returned final state use the cuDNN frontend's
    [batch, heads, v_head_dim, qk_head_dim] convention throughout.
    """
    qk_dim = q.shape[-1]
    if scale is None:
        scale = 1.0 / math.sqrt(qk_dim)

    heads = q.shape[2]
    q_ref = q.double() * scale
    k_ref = k.double()
    v_ref = v.double()

    g_ref = g.double()
    if safe_gate:
        lower_bound = (
            _DEFAULT_GATE_LOWER_BOUND if gate_lower_bound is None else float(gate_lower_bound)
        )
        amplitude = 1.0 if a_log is None else a_log.double().exp()[:, None]
        bias = 0.0 if dt_bias is None else dt_bias.double()
        g_ref = lower_bound * torch.sigmoid(amplitude * (g_ref + bias))
    if gate_domain == "linear":
        # The kernel floors a linear-domain gate before using it.
        alpha_ref = g_ref.clamp_min(1e-10)
    else:
        alpha_ref = g_ref.exp()

    beta_ref = beta.double()
    if use_beta_sigmoid:
        beta_ref = beta_ref.sigmoid() * (2.0 if allow_neg_eigval else 1.0)

    # [batch, sequence, heads, ...] -> [batch, heads, sequence, ...]
    q_ref, k_ref, v_ref, alpha_ref = (
        tensor.permute(0, 2, 1, 3) for tensor in (q_ref, k_ref, v_ref, alpha_ref)
    )
    beta_ref = beta_ref.permute(0, 2, 1)

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
        output, final_state = _kda_recurrence(
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
        output, state = _kda_recurrence(
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
    qk_dim=64,
    v_dim=64,
    dtype=torch.bfloat16,
):
    """Random KDA inputs in [batch, sequence, heads, dimension] layout.

    The gates are drawn the way the kernel takes them: ``g`` in float32 log
    space with one value per query/key channel, and ``beta`` as a float32
    scalar per token and head.
    """
    torch.manual_seed(1234)
    q = torch.randn(batch, sequence, heads, qk_dim, device="cuda", dtype=dtype)
    k = F.normalize(torch.randn_like(q, dtype=torch.float32), dim=-1).to(dtype)
    v = torch.randn(batch, sequence, heads, v_dim, device="cuda", dtype=dtype)
    # A per-channel decay in (0.5, 1] keeps the fp64 reference well conditioned.
    g = (
        torch.empty(batch, sequence, heads, qk_dim, device="cuda", dtype=torch.float32)
        .uniform_(0.5, 1.0)
        .log()
    )
    beta = torch.rand(batch, sequence, heads, device="cuda", dtype=torch.float32)
    return q, k, v, g, beta


def _safe_gate_logits(batch, sequence, heads, qk_dim):
    """Raw ``g`` logits for the safe gate, centred to keep the decay near 1.

    The safe gate is ``gate_lower_bound * sigmoid(exp(a_log) * (g + dt_bias))``,
    so logits centred on zero sit at the sigmoid's midpoint and give a decay of
    ``exp(-2.5) ~ 0.08`` per token: the state washes out within three tokens and
    nothing downstream of that is left to compare. Unlike the softplus gates of
    the Gated DeltaNet family, a small ``a_log`` does not move this operating
    point -- only a negative centre does.
    """
    return (
        torch.randn(batch, sequence, heads, qk_dim, device="cuda", dtype=torch.float32) - 4.0
    ).requires_grad_()


@requires_kda
@pytest.mark.parametrize("checkpoint_core_attention", [False, True], ids=["eager", "checkpoint"])
@pytest.mark.parametrize(
    ("qk_dim", "v_dim"),
    [(64, 64), (128, 128), (64, 128)],
    ids=["qk64_v64", "qk128_v128", "qk64_v128"],
)
@pytest.mark.parametrize("use_qk_l2norm_in_kernel", [False, True], ids=["no_l2norm", "l2norm"])
def test_kda_thd_forward_final_state_and_backward(
    checkpoint_core_attention, qk_dim, v_dim, use_qk_l2norm_in_kernel
):
    """THD KDA matches a PyTorch recurrence in forward and backward."""
    batch, sequence, heads = 2, 128, 2
    q, k, v, g, beta = (
        tensor.reshape(batch * sequence, *tensor.shape[2:]).requires_grad_(True)
        for tensor in _inputs(batch, sequence, heads, qk_dim, v_dim)
    )
    cu_seqlens = torch.arange(batch + 1, device="cuda", dtype=torch.int32) * sequence
    initial_state = (
        torch.randn(batch, heads, v_dim, qk_dim, device="cuda", dtype=torch.float32) * 0.05
    ).requires_grad_()

    attention = KimiDeltaAttention(
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
    output_ref, final_state_ref = _kda_reference(
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
        tolerance = _GATE_BWD_TOL[q.dtype] if name in ("g", "beta") else _BWD_TOL[q.dtype]
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
@requires_kda
def test_kda_dense_layout(qkv_format, dtype):
    """Dense TE layouts and supported input dtypes match a PyTorch reference."""
    batch, sequence, heads, dim = 1, 128, 2, 64
    q, k, v, g, beta = _inputs(batch, sequence, heads, dim, dim, dtype)
    with torch.no_grad():
        output_ref, _ = _kda_reference(
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

    attention = KimiDeltaAttention(
        num_attention_heads=heads,
        kv_channels=dim,
        qkv_format=qkv_format,
    )
    output = attention(q, k, v, g=g, beta=beta, use_qk_l2norm_in_kernel=True)
    assert output.shape == expected.shape
    _assert_rms_close(output, expected, _FWD_TOL[dtype], "output")


@requires_kda
def test_kda_beta_in_io_dtype():
    """beta is accepted at the Q/K/V dtype as well as float32."""
    batch, sequence, heads, dim = 1, 128, 2, 64
    q, k, v, g, beta = _inputs(batch, sequence, heads, dim, dim)
    attention = KimiDeltaAttention(
        num_attention_heads=heads,
        kv_channels=dim,
        qkv_format="bshd",
    )
    with torch.no_grad():
        expected = attention(q, k, v, g=g, beta=beta, use_qk_l2norm_in_kernel=True)
        output = attention(q, k, v, g=g, beta=beta.to(q.dtype), use_qk_l2norm_in_kernel=True)
    _assert_rms_close(output, expected, _FWD_TOL[q.dtype], "io-dtype beta output")


@requires_kda
def test_kda_state_round_trip_matches_single_shot():
    """A final state can seed the next chunk without changing the result."""
    batch, sequence, heads, dim = 2, 128, 2, 64
    split = 64
    q, k, v, g, beta = _inputs(batch, sequence, heads, dim, dim)
    attention = KimiDeltaAttention(
        num_attention_heads=heads,
        kv_channels=dim,
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


@requires_kda
@pytest.mark.parametrize(
    "bounds",
    [(0, 48, 160), (0, 0, 64, 160)],
    ids=["unequal", "leading-empty"],
)
def test_kda_thd_ragged_sequences(bounds):
    """Packed KDA handles unequal lengths and empty sequences."""
    total_tokens, heads, dim = bounds[-1], 2, 64
    q, k, v, g, beta = (tensor.squeeze(0) for tensor in _inputs(1, total_tokens, heads, dim, dim))
    cu_seqlens = torch.tensor(bounds, device="cuda", dtype=torch.int32)
    initial_state = torch.randn(
        len(bounds) - 1,
        heads,
        dim,
        dim,
        device="cuda",
        dtype=torch.float32,
    )
    attention = KimiDeltaAttention(
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
            initial_state=initial_state,
            output_final_state=True,
        )
        output_ref, final_state_ref = _kda_reference(
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


@requires_kda
@pytest.mark.parametrize("allow_neg_eigval", [False, True], ids=["projection", "reflection"])
def test_kda_fused_beta_sigmoid(allow_neg_eigval):
    """The kernel's fused beta activation matches applying it in the reference."""
    batch, sequence, heads, dim = 1, 128, 2, 64
    q, k, v, g, _ = _inputs(batch, sequence, heads, dim, dim)
    # Raw logits, which the kernel turns into the write strength itself.
    beta = torch.randn(batch, sequence, heads, device="cuda", dtype=torch.float32)

    attention = KimiDeltaAttention(
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
            use_qk_l2norm_in_kernel=True,
            use_beta_sigmoid_in_kernel=True,
            allow_neg_eigval=allow_neg_eigval,
        )
        output_ref, _ = _kda_reference(
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


@requires_kda
@pytest.mark.parametrize("with_params", [False, True], ids=["bare", "a_log_dt_bias"])
@pytest.mark.parametrize("gate_lower_bound", [None, -3.0], ids=["default_bound", "bound-3"])
def test_kda_safe_gate(with_params, gate_lower_bound):
    """The fused safe-gate transform matches applying it in the reference."""
    batch, sequence, heads, dim = 1, 128, 2, 64
    q, k, v, _, beta = _inputs(batch, sequence, heads, dim, dim)
    # Raw logits, which the kernel turns into the log decay itself.
    g = _safe_gate_logits(batch, sequence, heads, dim).detach()
    a_log = None
    dt_bias = None
    if with_params:
        a_log = torch.randn(heads, device="cuda", dtype=torch.float32) * 0.5
        dt_bias = torch.randn(heads, dim, device="cuda", dtype=torch.float32) * 0.5

    attention = KimiDeltaAttention(
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
            use_qk_l2norm_in_kernel=True,
            safe_gate=True,
            gate_lower_bound=gate_lower_bound,
            a_log=a_log,
            dt_bias=dt_bias,
        )
        output_ref, _ = _kda_reference(
            F.normalize(q.float(), dim=-1),
            F.normalize(k.float(), dim=-1),
            v,
            g,
            beta,
            safe_gate=True,
            gate_lower_bound=gate_lower_bound,
            a_log=a_log,
            dt_bias=dt_bias,
        )

    _assert_rms_close(
        output,
        output_ref.reshape(batch, sequence, -1),
        _FWD_TOL[q.dtype],
        "safe-gate output",
    )


@requires_kda
@pytest.mark.parametrize("checkpoint_core_attention", [False, True], ids=["eager", "checkpoint"])
def test_kda_safe_gate_parameter_gradients(checkpoint_core_attention):
    """The safe-gate parameters get gradients matching the reference.

    Parametrized over activation checkpointing because a_log/dt_bias take
    gradients, and TE's reentrant checkpoint only returns gradients for the
    positional arguments it forwards.
    """
    batch, sequence, heads, dim = 1, 128, 2, 64
    q, k, v, _, beta = _inputs(batch, sequence, heads, dim, dim)
    g = _safe_gate_logits(batch, sequence, heads, dim)
    a_log = (torch.randn(heads, device="cuda", dtype=torch.float32) * 0.5).requires_grad_()
    dt_bias = (torch.randn(heads, dim, device="cuda", dtype=torch.float32) * 0.5).requires_grad_()

    attention = KimiDeltaAttention(
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
        use_qk_l2norm_in_kernel=True,
        safe_gate=True,
        a_log=a_log,
        dt_bias=dt_bias,
        checkpoint_core_attention=checkpoint_core_attention,
    )

    reference_inputs = {
        name: tensor.detach().double().requires_grad_()
        for name, tensor in (("g", g), ("a_log", a_log), ("dt_bias", dt_bias))
    }
    output_ref, _ = _kda_reference(
        F.normalize(q.double(), dim=-1),
        F.normalize(k.double(), dim=-1),
        v.double(),
        reference_inputs["g"],
        beta,
        safe_gate=True,
        a_log=reference_inputs["a_log"],
        dt_bias=reference_inputs["dt_bias"],
    )

    output_weight = torch.randn_like(output, dtype=torch.float32)
    (output.float() * output_weight).sum().backward()
    (output_ref.reshape(batch, sequence, -1) * output_weight.double()).sum().backward()

    for name, tensor in (("g", g), ("a_log", a_log), ("dt_bias", dt_bias)):
        assert tensor.grad is not None, f"no gradient for {name}"
        assert torch.isfinite(tensor.grad).all(), f"non-finite gradient for {name}"
        tolerance = _GATE_BWD_TOL[q.dtype] if name == "g" else _PARAM_BWD_TOL[q.dtype]
        _assert_rms_close(
            tensor.grad,
            reference_inputs[name].grad.reshape_as(tensor),
            tolerance,
            f"d{name}",
        )


@requires_kda
def test_kda_linear_gate_domain():
    """gate_domain='linear' takes the decay itself rather than its logarithm."""
    batch, sequence, heads, dim = 1, 128, 2, 64
    q, k, v, g, beta = _inputs(batch, sequence, heads, dim, dim)
    alpha = g.exp()

    attention = KimiDeltaAttention(
        num_attention_heads=heads,
        kv_channels=dim,
        qkv_format="bshd",
    )
    with torch.no_grad():
        expected = attention(q, k, v, g=g, beta=beta, use_qk_l2norm_in_kernel=True)
        output = attention(
            q,
            k,
            v,
            g=alpha,
            beta=beta,
            use_qk_l2norm_in_kernel=True,
            gate_domain="linear",
        )

    _assert_rms_close(output, expected, _FWD_TOL[q.dtype], "linear-domain output")


@requires_gdn_and_kda
def test_kda_matches_gdn_with_scalar_decay():
    """KDA reduces to GDN when its per-channel decay is constant across channels.

    A scalar decay commutes with the delta-rule correction, so KDA's
    decay-first order collapses onto GDN's. Feeding KDA a channel-broadcast
    copy of GDN's scalar g -- and the same scalar beta, which both variants
    apply to the erase and the write alike -- must therefore reproduce GDN.
    """
    from transformer_engine.pytorch import GatedDeltaNetAttention

    batch, sequence, heads, dim = 1, 128, 2, 128
    q, k, v, _, _ = _inputs(batch, sequence, heads, dim, dim)
    g = torch.rand(batch, sequence, heads, device="cuda", dtype=torch.float32).log()
    beta = torch.rand(batch, sequence, heads, device="cuda", dtype=torch.float32)

    gdn = GatedDeltaNetAttention(num_attention_heads=heads, kv_channels=dim, qkv_format="bshd")
    kda = KimiDeltaAttention(num_attention_heads=heads, kv_channels=dim, qkv_format="bshd")
    with torch.no_grad():
        expected = gdn(q, k, v, g=g, beta=beta, use_qk_l2norm_in_kernel=True)
        output = kda(
            q,
            k,
            v,
            g=g.unsqueeze(-1).expand(-1, -1, -1, dim).contiguous(),
            beta=beta,
            use_qk_l2norm_in_kernel=True,
        )

    # Looser than _FWD_TOL: this compares two independently scheduled kernels.
    _assert_rms_close(output, expected, 3e-2, "KDA against GDN")
