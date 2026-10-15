# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
from copy import deepcopy
from itertools import product
from typing import Optional

import pytest
import torch
from torch._dynamo.utils import counters

from transformer_engine.pytorch.router import (
    QBHistogramMode,
    RoutingMapFormat,
    fused_topk_with_score_function,
    fused_compute_score_for_moe_aux_loss,
    fused_moe_aux_loss,
    mark_qb_bin_bounds_validated,
)
import transformer_engine_torch as tex

seed = 42
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)


@pytest.mark.parametrize("topk", [1, 10, 32])
@pytest.mark.parametrize("histogram_mode", [None, "two_kernel", "fused_atomic"])
@pytest.mark.parametrize("layout", ["bytemap", "bitmap_u8", "dense"])
@pytest.mark.parametrize("mixed_scores", [False, True])
def test_sigmoid_bias_preserves_tiny_scores(topk, histogram_mode, layout, mixed_scores):
    """Bias affects selection, but must not round away the selected raw scores."""
    num_tokens, num_experts, num_bins = 17, 512, 1000
    logits = torch.linspace(-8.0, -4.0, num_experts, device="cuda").repeat(num_tokens, 1)
    logits[:, :topk] = -16.0 - torch.arange(topk, device="cuda") % 9
    if mixed_scores:
        logits[:, 0] = -1.0
    logits.requires_grad_(True)
    expert_bias = torch.zeros(num_experts, device="cuda")
    expert_bias[:topk] = 0.75 + 0.001 * torch.arange(topk, device="cuda")
    expert_bias -= expert_bias.mean()
    bin_bounds = torch.tensor([-1.0, 1.0], device="cuda")
    histogram = torch.zeros(num_experts, num_bins, dtype=torch.int32, device="cuda")
    reference_probs, reference_map = topk_score_function_pytorch(
        logits,
        topk,
        scaling_factor=3.16,
        score_function="sigmoid",
        expert_bias=expert_bias,
    )
    kwargs = {}
    if histogram_mode is not None:
        kwargs.update(
            qb_histogram=histogram,
            qb_bin_bounds=bin_bounds,
            qb_histogram_mode=histogram_mode,
        )
    if layout == "dense":
        kwargs["topk_indices"] = torch.empty(num_tokens, topk, dtype=torch.int16, device="cuda")
    fused_logits = logits.detach().clone().requires_grad_(True)
    probs, routing = fused_topk_with_score_function(
        fused_logits,
        topk,
        False,
        None,
        None,
        3.16,
        "sigmoid",
        expert_bias,
        routing_map_format="bitmap_u8" if layout == "bitmap_u8" else "bytemap",
        **kwargs,
    )
    if layout == "dense":
        routing = topk_indices_to_routing_map(routing, num_experts)
    elif layout == "bitmap_u8":
        reference_map = _bytemap_to_bitmap_u8(reference_map)
    torch.testing.assert_close(routing, reference_map)
    # An absolute tolerance would hide precisely the tiny values under test.
    torch.testing.assert_close(probs, reference_probs, atol=0, rtol=5e-6)
    assert torch.all(probs[:, :topk] > 0)
    if histogram_mode is not None:
        reference = qb_topk_score_function_pytorch(logits, topk, expert_bias, bin_bounds, num_bins)
        torch.testing.assert_close(histogram, reference["histogram"])
    # Nonuniform upstream gradients exercise the normalization derivative.
    upstream = torch.linspace(-0.7, 1.3, num_experts, device="cuda").expand_as(probs)
    probs.backward(upstream)
    reference_probs.backward(upstream)
    torch.testing.assert_close(fused_logits.grad, logits.grad, atol=2e-7, rtol=2e-4)
    if mixed_scores and topk > 1:
        torch.testing.assert_close(
            fused_logits.grad[:, 1:topk], logits.grad[:, 1:topk], atol=0, rtol=2e-4
        )


def _get_tolerances(dtype: torch.dtype, num_experts: int):
    """Return (atol, rtol) scaled by the number of experts.

    With many experts the fused and reference kernels accumulate
    floating-point reductions (e.g. normalization sums) in different
    orders, causing O(num_experts * machine_eps) rounding divergence.
    Scale the default tolerances accordingly so that small expert
    counts keep tight checks while large counts (1024+) get the
    headroom they need.
    """
    # Default tolerances for torch.testing.assert_close
    base_atol, base_rtol = 1e-5, 1.3e-6
    if dtype == torch.float16:
        return 2e-4, 2e-3
    if dtype == torch.bfloat16:
        return 2e-3, 2e-2
    eps = 2e-7
    # The worst-case rounding error from summing N values is O(N * eps).
    # Use 2 * num_experts * eps as the tolerance floor so tests pass for
    # large expert counts while remaining tight for small ones.
    atol = max(base_atol, 2 * num_experts * eps)
    rtol = max(base_rtol, 2 * num_experts * eps)
    return atol, rtol


def _router_function(fn, execution):
    if execution == "eager":
        return fn
    torch._dynamo.reset()
    return torch.compile(fn, fullgraph=True)


def _compile_logits(dtype):
    logits = torch.linspace(-2, 2, 32, device="cuda")
    logits = logits + torch.linspace(-0.1, 0.1, 64, device="cuda").unsqueeze(1)
    return logits.reshape(2, 32, 32).to(dtype).requires_grad_(True)


def _check_compile_reference(actual, expected, dtype):
    atol, rtol = {
        torch.float32: (1e-6, 2e-5),
        torch.float16: (2e-4, 2e-3),
        torch.bfloat16: (2e-3, 2e-2),
    }[dtype]
    torch.testing.assert_close(actual.float(), expected.float(), atol=atol, rtol=rtol)


@pytest.mark.parametrize(
    "histogram_mode,index_dtype",
    [(None, torch.int32), ("two_kernel", None), ("fused_atomic", torch.int16)],
)
def test_router_compile_replay(histogram_mode, index_dtype):
    torch._dynamo.reset()
    counters.clear()
    histogram = torch.full((32, 31), 7, device="cuda", dtype=torch.int32)
    bounds = torch.tensor([-1.1, 1.2], device="cuda")
    bias = torch.linspace(0.3, 0, 32, device="cuda")
    counts = torch.arange(1, 33, device="cuda", dtype=torch.int32)
    total = torch.tensor(64, device="cuda", dtype=torch.int64)
    indices = (
        torch.full((2, 32, 4), -1, device="cuda", dtype=index_dtype)
        if index_dtype is not None
        else None
    )

    def router(logits, bounds, total):
        outputs = fused_topk_with_score_function(
            logits,
            4,
            False,
            None,
            None,
            1.2,
            "sigmoid",
            bias,
            routing_map_format="bitmap_u8" if indices is None else "bytemap",
            topk_indices=indices,
            overwrite_topk_indices=False,
            qb_histogram=histogram if histogram_mode else None,
            qb_bin_bounds=bounds if histogram_mode else None,
            qb_histogram_mode=histogram_mode,
            return_qb_histogram_delta=histogram_mode is not None,
            assume_valid_qb_bin_bounds=histogram_mode is not None,
        )
        _, scores = fused_compute_score_for_moe_aux_loss(logits, 4, "sigmoid")
        loss = fused_moe_aux_loss(scores.reshape(-1, 32), counts, total, 32, 4, 0.01)
        return (outputs[0].float().square().sum() + loss, *outputs[1:])

    compiled = torch.compile(router, fullgraph=True, mode="reduce-overhead")
    for step, new_bounds in enumerate(([-1.1, 1.2], [-0.3, 0.7], [-0.8, 1.4])):
        torch.compiler.cudagraph_mark_step_begin()
        bounds.copy_(bounds.new_tensor(new_bounds))
        total.fill_(64 + 17 * step)
        logits = _compile_logits(torch.bfloat16)
        if step == 1:
            logits = logits.flip(-1).detach().requires_grad_(True)
        eager_logits = logits.detach().clone().requires_grad_(True)
        actual = compiled(logits, bounds, total)
        expected = router(eager_logits, bounds, total)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(histogram, torch.full_like(histogram, 7 + step))
        if indices is not None:
            torch.testing.assert_close(indices, torch.full_like(indices, -1 - step))
            indices.sub_(1)
        histogram.add_(1)
        bounds.add_(0.1)
        actual_grad = torch.autograd.grad(actual[0], logits)
        expected_grad = torch.autograd.grad(expected[0], eager_logits)
        torch.testing.assert_close(actual_grad, expected_grad)
    with torch.no_grad():
        torch.compiler.cudagraph_mark_step_begin()
        torch.testing.assert_close(compiled(logits, bounds, total), router(logits, bounds, total))
    assert not counters["inductor"]["cudagraph_skips"]


@pytest.mark.parametrize(
    "dtype,index_dtype",
    [(torch.float32, None), (torch.float16, torch.int16), (torch.bfloat16, torch.int64)],
)
def test_router_compile_dynamic(dtype, index_dtype):
    torch._dynamo.reset()
    counts = torch.arange(1, 131, device="cuda", dtype=torch.int32)

    def router_loss(logits):
        indices = (
            torch.empty(logits.shape[0], 4, dtype=index_dtype, device=logits.device)
            if index_dtype is not None
            else None
        )
        probs, routing_map = fused_topk_with_score_function(
            logits,
            4,
            False,
            None,
            None,
            None,
            "sigmoid",
            None,
            "bytemap" if indices is not None else "bitmap_u8",
            topk_indices=indices,
            overwrite_topk_indices=False,
        )
        _, scores = fused_compute_score_for_moe_aux_loss(logits, 4, "sigmoid", 1)
        aux_loss = fused_moe_aux_loss(scores, counts, logits.shape[0], 130, 4, 0.01)
        return probs.float().square().sum() + aux_loss, routing_map

    compiled = torch.compile(router_loss, fullgraph=True, dynamic=True)
    for num_tokens in (16, 33, 64):
        logits = torch.randn(num_tokens, 130, device="cuda", dtype=dtype, requires_grad=True)
        eager_logits = logits.detach().clone().requires_grad_(True)
        actual, routing_map = compiled(logits)
        expected, eager_map = router_loss(eager_logits)
        assert routing_map.shape == (num_tokens, 4 if index_dtype is not None else 17)
        torch.testing.assert_close(routing_map, eager_map)
        torch.testing.assert_close(actual, expected)
        (actual_grad,) = torch.autograd.grad(actual, logits)
        (eager_grad,) = torch.autograd.grad(expected, eager_logits)
        _check_compile_reference(actual_grad, eager_grad, dtype)


@pytest.mark.parametrize(
    "dtype,routing_map_format,index_dtype,histogram_mode",
    [
        (torch.float32, 0, None, None),
        (torch.float16, 1, None, None),
        (torch.bfloat16, 0, torch.int16, None),
        (torch.float32, 0, torch.int32, None),
        (torch.bfloat16, 0, torch.int64, None),
        (torch.float32, 0, None, QBHistogramMode.TWO_KERNEL),
        (torch.float16, 1, None, QBHistogramMode.FUSED_ATOMIC),
        (torch.bfloat16, 0, torch.int16, QBHistogramMode.TWO_KERNEL),
        (torch.float32, 0, torch.int32, QBHistogramMode.FUSED_ATOMIC),
        (torch.bfloat16, 0, torch.int64, QBHistogramMode.FUSED_ATOMIC),
    ],
)
def test_router_compile_opcheck(dtype, routing_map_format, index_dtype, histogram_mode):
    from transformer_engine.pytorch import router

    logits = torch.randn(16, 130, device="cuda", dtype=dtype, requires_grad=True)
    indices = (
        torch.full((16, 4), -1, device="cuda", dtype=index_dtype)
        if index_dtype is not None
        else None
    )
    if histogram_mode is not None:
        bias = torch.zeros(130, device="cuda")
        histogram = torch.full((130, 64), 7, device="cuda", dtype=torch.int32)
        bounds = torch.tensor([-1.0, 1.0], device="cuda")
        torch.library.opcheck(
            router._qb_forward,
            (
                logits,
                4,
                None,
                bias,
                routing_map_format,
                indices,
                histogram,
                bounds,
                int(histogram_mode),
            ),
        )
    else:
        torch.library.opcheck(
            router._topk_forward,
            (logits, 4, False, None, None, None, "sigmoid", None, routing_map_format, indices),
        )
        if index_dtype in (None, torch.int16):
            torch.library.opcheck(router._score_forward, (logits, 4, "softmax", routing_map_format))
            counts = torch.ones(130, device="cuda", dtype=torch.int32)
            count = torch.tensor(16, device="cuda", dtype=torch.int64)
            for token_count in (None, count):
                torch.library.opcheck(
                    router._aux_loss_forward, (logits, counts, 16, token_count, 130, 4, 0.01)
                )


@pytest.mark.parametrize(
    "qb,overwrite,delta,assume_valid",
    [
        (False, True, False, False),
        (True, True, False, False),
        (True, False, False, False),
        (True, False, False, True),
        (True, False, True, False),
        (True, True, True, True),
    ],
)
def test_router_compile_eager_fallback(qb, overwrite, delta, assume_valid):
    torch._dynamo.reset()
    logits = torch.randn(32, 32, device="cuda", requires_grad=True)
    bias = torch.zeros(32, device="cuda")
    indices = torch.empty(32, 4, device="cuda", dtype=torch.int32)
    histogram = torch.zeros(32, 16, device="cuda", dtype=torch.int32)
    bounds = torch.tensor([-2, 2], device="cuda", dtype=torch.float32)

    def router(x):
        return fused_topk_with_score_function(
            x,
            4,
            False,
            None,
            None,
            None,
            "sigmoid",
            bias,
            topk_indices=indices if not qb or (delta and assume_valid) else None,
            qb_histogram=histogram if qb else None,
            qb_bin_bounds=bounds if qb else None,
            qb_histogram_mode="two_kernel" if qb else None,
            **({} if overwrite else {"overwrite_topk_indices": False}),
            return_qb_histogram_delta=delta,
            assume_valid_qb_bin_bounds=assume_valid,
        )

    with pytest.warns(UserWarning, match="Falling back to eager execution"):
        actual = torch.compile(router)(logits)
    expected = router(logits)
    torch.testing.assert_close(actual, expected)
    torch._dynamo.reset()
    with pytest.raises(torch._dynamo.exc.Unsupported):
        torch.compile(router, fullgraph=True)(logits)


@pytest.mark.parametrize("delta", [False, True])
def test_qb_trusted_bounds_skip_value_validation(monkeypatch, delta):
    from transformer_engine.pytorch import router

    logits = torch.randn(8, 16, device="cuda", requires_grad=True)
    bias = torch.zeros(16, device="cuda")
    bounds = torch.tensor([-1.0, 1.0], device="cuda")
    options = dict(
        qb_histogram=torch.zeros(16, 32, device="cuda", dtype=torch.int32),
        qb_bin_bounds=bounds,
        qb_histogram_mode="two_kernel",
        return_qb_histogram_delta=delta,
    )

    def unexpected_validation(_):
        raise AssertionError("trusted QB bounds must not read values on the CPU")

    with monkeypatch.context() as patch:
        patch.setattr(router, "_validate_qb_bin_bounds", unexpected_validation)
        fused_topk_with_score_function(
            logits,
            4,
            False,
            None,
            None,
            None,
            "sigmoid",
            bias,
            **options,
            assume_valid_qb_bin_bounds=True,
        )
    bounds.zero_()
    with pytest.raises(ValueError, match="finite with lower < upper"):
        fused_topk_with_score_function(
            logits, 4, False, None, None, None, "sigmoid", bias, **options
        )


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("invalid", ["shape", "dtype", "device"])
def test_qb_trusted_bounds_check_metadata(compiled, invalid):
    torch._dynamo.reset()
    logits = torch.randn(8, 16, device="cuda")
    bias = torch.zeros(16, device="cuda")
    histogram = torch.zeros(16, 32, device="cuda", dtype=torch.int32)
    bounds = torch.tensor([-1.0, 1.0], device="cuda")
    if invalid == "shape":
        bounds = bounds.reshape(1, 2)
    elif invalid == "dtype":
        bounds = bounds.half()
    else:
        bounds = bounds.cpu()

    def router(logits, bounds):
        return fused_topk_with_score_function(
            logits,
            4,
            False,
            None,
            None,
            None,
            "sigmoid",
            bias,
            qb_histogram=histogram,
            qb_bin_bounds=bounds,
            qb_histogram_mode="two_kernel",
            return_qb_histogram_delta=True,
            assume_valid_qb_bin_bounds=True,
        )

    if compiled:
        router = torch.compile(router, fullgraph=True)
    with pytest.raises(RuntimeError, match="QB bin_bounds"):
        router(logits, bounds)


# Pytorch-based group topk
def group_limited_topk(
    scores: torch.Tensor,
    topk: int,
    num_tokens: int,
    num_experts: int,
    num_groups: int,
    group_topk: int,
):
    group_scores = (
        scores.view(num_tokens, num_groups, -1).topk(topk // group_topk, dim=-1)[0].sum(dim=-1)
    )
    group_idx = torch.topk(group_scores, k=group_topk, dim=-1, sorted=False)[1]
    group_mask = torch.zeros_like(group_scores)
    group_mask.scatter_(1, group_idx, 1)

    # Mask the experts based on selection groups
    score_mask = (
        group_mask.unsqueeze(-1)
        .expand(num_tokens, num_groups, num_experts // num_groups)
        .reshape(num_tokens, -1)
    )

    masked_scores = scores.masked_fill(~score_mask.bool(), float("-inf"))
    probs, top_indices = torch.topk(masked_scores, k=topk, dim=-1)

    return probs, top_indices


# Pytorch-based topk softmax/sigmoid
def topk_score_function_pytorch(
    logits: torch.Tensor,
    topk: int,
    use_pre_softmax: bool = False,
    num_groups: Optional[int] = None,
    group_topk: Optional[int] = None,
    scaling_factor: Optional[float] = None,
    score_function: str = "softmax",
    expert_bias: Optional[torch.Tensor] = None,
):
    num_tokens, num_experts = logits.shape

    def compute_topk(scores, topk, num_groups=None, group_topk=None):
        if group_topk:
            return group_limited_topk(
                scores=scores,
                topk=topk,
                num_tokens=num_tokens,
                num_experts=num_experts,
                num_groups=num_groups,
                group_topk=group_topk,
            )
        else:
            return torch.topk(scores, k=topk, dim=1)

    if score_function == "softmax":
        if use_pre_softmax:
            scores = torch.softmax(logits, dim=-1, dtype=torch.float32)
            probs, top_indices = compute_topk(scores, topk, num_groups, group_topk)
        else:
            scores, top_indices = compute_topk(logits, topk, num_groups, group_topk)
            probs = torch.softmax(scores, dim=-1, dtype=torch.float32)
    elif score_function in ("sigmoid", "sqrtsoftplus"):
        if score_function == "sigmoid":
            scores = torch.sigmoid(logits.float())
        else:
            scores = torch.nn.functional.softplus(logits.float()).sqrt()
        if expert_bias is not None:
            scores_for_routing = scores + expert_bias
            _, top_indices = compute_topk(scores_for_routing, topk, num_groups, group_topk)
            scores = torch.gather(scores, dim=1, index=top_indices)
        else:
            scores, top_indices = compute_topk(scores, topk, num_groups, group_topk)
        probs = scores / (scores.sum(dim=-1, keepdim=True) + 1e-20) if topk > 1 else scores
    else:
        raise ValueError(f"Invalid score_function: {score_function}")

    if scaling_factor:
        probs = probs * scaling_factor

    probs = probs.type_as(logits)

    topk_masked_gates = torch.zeros_like(logits).scatter(1, top_indices, probs)
    topk_map = torch.zeros_like(logits).int().scatter(1, top_indices, 1).bool()

    return topk_masked_gates, topk_map


def qb_topk_score_function_pytorch(
    logits: torch.Tensor,
    topk: int,
    expert_bias: torch.Tensor,
    bin_bounds: torch.Tensor,
    num_bins: int,
    histogram: Optional[torch.Tensor] = None,
):
    """Pure-PyTorch reference for Kimi K3 QB routing and histogram accumulation."""
    original_shape = logits.shape
    num_experts = original_shape[-1]
    raw_scores = torch.sigmoid(logits.float()).reshape(-1, num_experts)
    biased_scores = raw_scores + expert_bias
    topk_plus_one_scores, topk_plus_one_indices = torch.topk(biased_scores, k=topk + 1, dim=-1)
    cutoff = topk_plus_one_scores.min(dim=-1).values
    cutoff_candidates = topk_plus_one_indices.masked_fill(
        topk_plus_one_scores != cutoff.unsqueeze(1), -1
    )
    dropped_expert = cutoff_candidates.max(dim=-1).values
    topk_indices = topk_plus_one_indices[
        topk_plus_one_indices != dropped_expert.unsqueeze(1)
    ].reshape(-1, topk)

    selected_raw_scores = torch.gather(raw_scores, 1, topk_indices)
    if topk > 1:
        selected_probs = selected_raw_scores / (
            selected_raw_scores.sum(dim=-1, keepdim=True) + 1e-20
        )
    else:
        selected_probs = selected_raw_scores
    probs = torch.zeros_like(raw_scores).scatter(1, topk_indices, selected_probs)
    routing_map = torch.zeros_like(raw_scores, dtype=torch.bool).scatter(1, topk_indices, True)

    lower, upper = bin_bounds[0], bin_bounds[1]
    required_bias = cutoff.unsqueeze(1) - raw_scores
    bin_scale = num_bins / (upper - lower)
    bin_indices = torch.floor((required_bias - lower) * bin_scale).to(torch.int64)
    bin_indices.clamp_(0, num_bins - 1)
    expert_offsets = torch.arange(num_experts, device=logits.device, dtype=torch.int64) * num_bins
    flat_indices = (bin_indices + expert_offsets).reshape(-1)
    counts = torch.bincount(flat_indices, minlength=num_experts * num_bins)
    counts = counts.reshape(num_experts, num_bins).to(torch.int32)
    if histogram is None:
        histogram = torch.zeros_like(counts)
    histogram.add_(counts)

    return {
        "probs": probs.reshape(original_shape).to(logits.dtype),
        "routing_map": routing_map.reshape(original_shape),
        "topk_indices": topk_indices.reshape(*original_shape[:-1], topk),
        "raw_scores": raw_scores.reshape(original_shape),
        "cutoff": cutoff.reshape(original_shape[:-1]),
        "bin_indices": bin_indices.reshape(original_shape),
        "histogram": histogram,
    }


# Pytorch-based compute routing scores for aux loss
def compute_scores_for_aux_loss_pytorch(
    logits: torch.Tensor, topk: int, score_function: str
) -> torch.Tensor:
    if score_function == "softmax":
        scores = torch.softmax(logits, dim=-1, dtype=torch.float32)
    elif score_function == "sigmoid":
        scores = torch.sigmoid(logits.float())
        scores = scores / (scores.sum(dim=-1, keepdim=True) + 1e-20)
    elif score_function == "sqrtsoftplus":
        scores = torch.nn.functional.softplus(logits.float()).sqrt()
        scores = scores / (scores.sum(dim=-1, keepdim=True) + 1e-20)
    else:
        raise ValueError(f"Invalid score_function: {score_function}")

    _, top_indices = torch.topk(scores, k=topk, dim=1)
    routing_map = torch.zeros_like(logits).int().scatter(1, top_indices, 1).bool()
    return routing_map, scores


# Pytorch-based aux loss
def aux_loss_pytorch(
    probs: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    total_num_tokens: int,
    topk: int,
    num_experts: int,
    moe_aux_loss_coeff: float,
):
    aggregated_probs_per_expert = probs.sum(dim=0)
    aux_loss = torch.sum(aggregated_probs_per_expert * tokens_per_expert) * (
        num_experts * moe_aux_loss_coeff / (topk * total_num_tokens * total_num_tokens)
    )
    return aux_loss


def topk_indices_to_routing_map(topk_indices: torch.Tensor, num_experts: int) -> torch.Tensor:
    """Convert dense [num_tokens, topk] top-k indices to a bool routing map."""
    routing_map = torch.zeros(
        topk_indices.size(0), num_experts, dtype=torch.bool, device=topk_indices.device
    )
    routing_map.scatter_(1, topk_indices.long(), True)
    return routing_map


def run_comparison(
    dtype,
    num_tokens,
    num_experts,
    topk,
    use_pre_softmax,
    num_groups,
    group_topk,
    scaling_factor,
    score_function,
    enable_bias,
    topk_output_mode="sparse",
    topk_index_dtype=torch.int16,
    execution="eager",
):
    if topk >= num_experts:
        pytest.skip(f"topk ({topk}) >= num_experts ({num_experts})")
    if group_topk is not None and num_groups is not None:
        group_size = num_experts // num_groups
        per_group_topk = topk // group_topk
        if per_group_topk >= group_size:
            pytest.skip(f"per-group topk ({per_group_topk}) >= group_size ({group_size})")
    # Set some parameters
    if score_function in ("sigmoid", "sqrtsoftplus"):
        # Construct logits with a narrow range to avoid very small activation values,
        # which would cause precision loss when adding/subtracting expert bias in float32.
        offset = torch.arange(-num_tokens // 2, num_tokens // 2, dtype=dtype, device="cuda") * 1e-4
        logits = (
            torch.arange(-num_experts // 2, num_experts // 2, device="cuda", dtype=dtype) * 1e-2
        )
        logits = logits.unsqueeze(0).repeat(num_tokens, 1) + offset.unsqueeze(1)
    else:
        logits = (
            torch.arange(
                -num_tokens * num_experts // 2,
                num_tokens * num_experts // 2,
                device="cuda",
                dtype=dtype,
            )
            * 1e-4
        )
        logits = logits.view(num_tokens, num_experts)
    if dtype != torch.float32:
        logits = torch.linspace(-2, 2, num_experts, device="cuda").repeat(num_tokens, 1)
        logits = logits.to(dtype)
    logits.requires_grad = True
    if enable_bias and score_function in ("sigmoid", "sqrtsoftplus"):
        expert_bias = torch.arange(num_experts, device="cuda", dtype=torch.float32) * 0.1
        expert_bias = torch.flip(expert_bias, dims=[0])
        expert_bias.requires_grad = True
    else:
        expert_bias = None

    # Clone the input tensor
    logits_clone = deepcopy(logits)
    logits_clone.requires_grad = True
    if expert_bias is not None:
        expert_bias_clone = deepcopy(expert_bias)
        expert_bias_clone.requires_grad = True
    else:
        expert_bias_clone = None

    logits = logits.detach().float().requires_grad_(True)

    # Run the original implementation
    # We do not support the capacity factor case
    probs, routing_map = topk_score_function_pytorch(
        logits=logits,
        topk=topk,
        use_pre_softmax=use_pre_softmax,
        num_groups=num_groups,
        group_topk=group_topk,
        scaling_factor=scaling_factor,
        score_function=score_function,
        expert_bias=expert_bias,
    )

    topk_indices = None
    if topk_output_mode == "dense":
        topk_indices = torch.full((num_tokens, topk), -1, device="cuda", dtype=topk_index_dtype)

    # Run the fused implementation
    router = _router_function(fused_topk_with_score_function, execution)
    probs_fused, routing_output_fused = router(
        logits=logits_clone,
        topk=topk,
        use_pre_softmax=use_pre_softmax,
        num_groups=num_groups,
        group_topk=group_topk,
        scaling_factor=scaling_factor,
        score_function=score_function,
        expert_bias=expert_bias_clone,
        topk_indices=topk_indices,
        overwrite_topk_indices=execution == "eager",
    )
    assert probs_fused.dtype == dtype and not routing_output_fused.requires_grad
    if topk_output_mode == "dense":
        assert (routing_output_fused.data_ptr() == topk_indices.data_ptr()) == (
            execution == "eager"
        )
        if execution != "eager":
            torch.testing.assert_close(topk_indices, torch.full_like(topk_indices, -1))
            topk_indices.fill_(-2)
        assert routing_output_fused.dtype == topk_index_dtype
        routing_map_fused = topk_indices_to_routing_map(routing_output_fused, num_experts)
    else:
        routing_map_fused = routing_output_fused

    atol, rtol = _get_tolerances(dtype, num_experts)
    torch.testing.assert_close(probs, probs_fused.float(), atol=atol, rtol=rtol)
    torch.testing.assert_close(routing_map, routing_map_fused)

    grad = torch.randn((*probs_fused.shape, 2), device="cuda", dtype=dtype)[..., 0]
    probs.backward(grad.float())
    probs_fused.backward(grad)
    torch.testing.assert_close(logits.grad, logits_clone.grad.float(), atol=atol, rtol=rtol)


_SIGMOID_TOPK_CASES = [
    ("eager", *case)
    for case in product(
        [torch.float32],
        [2048, 7168, 8992],
        [1024, 128, 32],
        [4, 8, 16, 32],
        [None, 4],
        [None, 1.2],
        [True, False],
        [None, torch.int16, torch.int32, torch.int64],
    )
] + [
    ("compile", torch.float32, 64, 32, 4, None, None, False, None),
    ("compile", torch.float16, 64, 32, 4, None, 1.2, True, torch.int16),
    ("compile", torch.bfloat16, 64, 32, 4, 4, 1.2, True, torch.int32),
    ("compile", torch.float32, 64, 130, 16, None, None, True, torch.int64),
    ("compile", torch.bfloat16, 64, 32, 1, None, None, False, None),
]


@pytest.mark.parametrize(
    "execution,dtype,num_tokens,num_experts,topk,group_topk,scaling_factor,enable_bias,topk_index_dtype",
    _SIGMOID_TOPK_CASES,
)
def test_topk_sigmoid(
    dtype,
    num_tokens,
    num_experts,
    topk,
    group_topk,
    scaling_factor,
    enable_bias,
    topk_index_dtype,
    execution,
):
    num_groups = 8 if group_topk else None
    run_comparison(
        dtype=dtype,
        num_tokens=num_tokens,
        num_experts=num_experts,
        topk=topk,
        use_pre_softmax=False,
        num_groups=num_groups,
        group_topk=group_topk,
        scaling_factor=scaling_factor,
        score_function="sigmoid",
        enable_bias=enable_bias,
        topk_output_mode="dense" if topk_index_dtype is not None else "sparse",
        topk_index_dtype=topk_index_dtype or torch.int16,
        execution=execution,
    )


@pytest.mark.parametrize(
    "execution,dtype,num_tokens,num_experts,topk,group_topk,scaling_factor,enable_bias,topk_index_dtype",
    _SIGMOID_TOPK_CASES,
)
def test_topk_sqrtsoftplus(
    dtype,
    num_tokens,
    num_experts,
    topk,
    group_topk,
    scaling_factor,
    enable_bias,
    topk_index_dtype,
    execution,
):
    num_groups = 8 if group_topk else None
    run_comparison(
        dtype=dtype,
        num_tokens=num_tokens,
        num_experts=num_experts,
        topk=topk,
        use_pre_softmax=False,
        num_groups=num_groups,
        group_topk=group_topk,
        scaling_factor=scaling_factor,
        score_function="sqrtsoftplus",
        enable_bias=enable_bias,
        topk_output_mode="dense" if topk_index_dtype is not None else "sparse",
        topk_index_dtype=topk_index_dtype or torch.int16,
        execution=execution,
    )


@pytest.mark.parametrize(
    "execution,dtype,histogram_mode,topk,routing_output_mode,overwrite,delta",
    [
        ("eager", torch.float32, *case, overwrite, False)
        for case in product(
            ["two_kernel", "fused_atomic"],
            [8, 16],
            ["bytemap", "bitmap_u8", "dense_int16", "dense_int32", "dense_int64"],
        )
        for overwrite in ([False, True] if case[2].startswith("dense") else [True])
    ]
    + [
        (execution, dtype, histogram_mode, topk, layout, False, True)
        for execution in ["eager", "compile"]
        for dtype, histogram_mode, topk, layout in [
            (torch.float32, "two_kernel", 1, "bytemap"),
            (torch.float16, "fused_atomic", 4, "bitmap_u8"),
            (torch.bfloat16, "two_kernel", 16, "dense_int16"),
            (torch.float32, "fused_atomic", 16, "dense_int32"),
            (torch.bfloat16, "fused_atomic", 4, "dense_int64"),
            (torch.float16, "two_kernel", 4, "bytemap"),
        ]
    ],
)
def test_qb_topk_histogram(
    histogram_mode,
    topk,
    routing_output_mode,
    overwrite,
    execution,
    dtype,
    delta,
):
    num_tokens, num_experts, num_bins = (64, 32, 31) if delta else (257, 896, 1000)
    logits = torch.randn(
        num_tokens, num_experts, device="cuda", dtype=torch.float32, requires_grad=True
    )
    if delta:
        logits = (
            _compile_logits(dtype).reshape(num_tokens, num_experts).detach().requires_grad_(True)
        )
    reference_logits = logits.detach().float().requires_grad_(True)
    expert_bias = torch.linspace(-0.2, 0.2, num_experts, device="cuda", dtype=torch.float32)
    bin_bounds = torch.stack((expert_bias.min() - 1.0, expert_bias.max() + 1.0))
    reference_histogram = torch.zeros(num_experts, num_bins, device="cuda", dtype=torch.int32)
    reference = qb_topk_score_function_pytorch(
        reference_logits,
        topk,
        expert_bias,
        bin_bounds,
        num_bins,
        reference_histogram,
    )

    fused_logits = logits.detach().clone().requires_grad_(True)
    fused_histogram = torch.full_like(reference_histogram, 7 if delta else 0)
    histogram_version = fused_histogram._version
    dense_dtype = {
        "dense_int16": torch.int16,
        "dense_int32": torch.int32,
        "dense_int64": torch.int64,
    }.get(routing_output_mode)
    topk_indices = (
        torch.full((num_tokens, topk), -1, device="cuda", dtype=dense_dtype)
        if dense_dtype is not None
        else None
    )
    routing_map_format = (
        RoutingMapFormat.BITMAP_U8
        if routing_output_mode == "bitmap_u8"
        else RoutingMapFormat.BYTEMAP
    )
    router = _router_function(fused_topk_with_score_function, execution)
    outputs = router(
        logits=fused_logits,
        topk=topk,
        use_pre_softmax=False,
        num_groups=None,
        group_topk=None,
        scaling_factor=None,
        score_function="sigmoid",
        expert_bias=expert_bias,
        routing_map_format=routing_map_format,
        topk_indices=topk_indices,
        qb_histogram=fused_histogram,
        qb_bin_bounds=bin_bounds,
        qb_histogram_mode=histogram_mode,
        overwrite_topk_indices=overwrite,
        return_qb_histogram_delta=delta,
        assume_valid_qb_bin_bounds=delta,
    )
    fused_probs, fused_routing_output = outputs[:2]
    if delta:
        _check_compile_reference(fused_probs, reference["probs"], dtype)
    else:
        torch.testing.assert_close(fused_probs, reference["probs"])
    assert not fused_routing_output.requires_grad
    if delta:
        assert not outputs[2].requires_grad
        assert outputs[2].data_ptr() != fused_histogram.data_ptr()
        assert fused_histogram._version == histogram_version
        torch.testing.assert_close(fused_histogram, torch.full_like(fused_histogram, 7))
        torch.testing.assert_close(outputs[2], reference["histogram"])
        fused_histogram.add_(1)
        bin_bounds.add_(0.1)
    else:
        torch.testing.assert_close(fused_histogram, reference["histogram"])
    if dense_dtype is not None:
        assert (fused_routing_output.data_ptr() == topk_indices.data_ptr()) == overwrite
        if not overwrite:
            torch.testing.assert_close(topk_indices, torch.full_like(topk_indices, -1))
            topk_indices.fill_(-2)
        fused_routing_map = topk_indices_to_routing_map(fused_routing_output, num_experts)
        torch.testing.assert_close(fused_routing_map, reference["routing_map"])
    elif routing_output_mode == "bitmap_u8":
        torch.testing.assert_close(
            fused_routing_output,
            _bytemap_to_bitmap_u8(reference["routing_map"]),
        )
    else:
        torch.testing.assert_close(fused_routing_output, reference["routing_map"])

    grad = torch.randn((*fused_probs.shape, 2), device="cuda", dtype=dtype)[..., 0]
    reference["probs"].backward(grad.float())
    fused_probs.backward(grad)
    if delta:
        _check_compile_reference(fused_logits.grad, reference_logits.grad, dtype)
    else:
        torch.testing.assert_close(fused_logits.grad, reference_logits.grad, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize(
    "execution,dtype,histogram_mode",
    [
        ("eager", torch.float32, "two_kernel"),
        ("eager", torch.float32, "fused_atomic"),
        ("compile", torch.float32, "two_kernel"),
        ("compile", torch.bfloat16, "fused_atomic"),
    ],
)
def test_qb_histogram_accumulates_microbatches(execution, dtype, histogram_mode):
    functional = execution == "compile"
    num_experts, num_bins = (130, 64) if functional else (64, 1000)
    expert_bias = torch.linspace(-0.1, 0.1, num_experts, device="cuda")
    bounds = torch.stack((expert_bias.min() - 1.0, expert_bias.max() + 1.0))
    histogram = torch.full(
        (num_experts, num_bins), 7 if functional else 0, device="cuda", dtype=torch.int32
    )
    reference_histogram = histogram.clone()

    def route(logits, histogram, bounds):
        outputs = fused_topk_with_score_function(
            logits,
            8,
            False,
            None,
            None,
            None,
            "sigmoid",
            expert_bias,
            routing_map_format="bitmap_u8" if functional else "bytemap",
            qb_histogram=histogram,
            qb_bin_bounds=bounds,
            qb_histogram_mode=histogram_mode,
            return_qb_histogram_delta=functional,
            assume_valid_qb_bin_bounds=functional,
        )
        return outputs[0], outputs[1], histogram + outputs[2] if functional else histogram

    if functional:
        torch._dynamo.reset()
        route = torch.compile(route, fullgraph=True, dynamic=True)
    for num_tokens in (17, 33, 64) if functional else (127, 193):
        logits = torch.randn(
            num_tokens, num_experts, device="cuda", dtype=dtype, requires_grad=True
        )
        reference = qb_topk_score_function_pytorch(
            logits, 8, expert_bias, bounds, num_bins, reference_histogram
        )
        input_histogram = histogram
        probs, routing, histogram = route(logits, histogram, bounds)
        torch.testing.assert_close(histogram, reference["histogram"])
        _check_compile_reference(probs, reference["probs"], dtype)
        expected_map = (
            _bytemap_to_bitmap_u8(reference["routing_map"])
            if functional
            else reference["routing_map"]
        )
        torch.testing.assert_close(routing, expected_map)
        if functional:
            assert histogram.data_ptr() != input_histogram.data_ptr()
            actual_grad = torch.autograd.grad(probs.float().square().sum(), logits)[0]
            ref_grad = torch.autograd.grad(reference["probs"].float().square().sum(), logits)[0]
            _check_compile_reference(actual_grad, ref_grad, dtype)
            bounds.add_(0.03)


def test_qb_topk_argument_validation():
    logits = torch.randn(16, 32, device="cuda", dtype=torch.float32)
    expert_bias = torch.zeros(32, device="cuda", dtype=torch.float32)
    histogram = torch.zeros(32, 1000, device="cuda", dtype=torch.int32)
    bin_bounds = torch.tensor([-1.0, 1.0], device="cuda", dtype=torch.float32)
    with pytest.raises(ValueError, match="provided together"):
        fused_topk_with_score_function(
            logits,
            4,
            False,
            None,
            None,
            None,
            "sigmoid",
            expert_bias,
            qb_histogram=histogram,
        )
    with pytest.raises(ValueError, match="only supports"):
        fused_topk_with_score_function(
            logits,
            4,
            False,
            None,
            None,
            None,
            "softmax",
            expert_bias,
            qb_histogram=histogram,
            qb_bin_bounds=bin_bounds,
            qb_histogram_mode="two_kernel",
        )

    with pytest.raises(ValueError, match="return_qb_histogram_delta requires"):
        fused_topk_with_score_function(
            logits,
            4,
            False,
            None,
            None,
            None,
            "sigmoid",
            None,
            return_qb_histogram_delta=True,
        )


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires at least two CUDA devices")
@pytest.mark.parametrize("histogram_mode", ["two_kernel", "fused_atomic"])
def test_qb_topk_uses_logits_device(histogram_mode):
    current_device = torch.cuda.current_device()
    logits_device = (current_device + 1) % torch.cuda.device_count()
    device = torch.device("cuda", logits_device)
    num_tokens, num_experts, topk, num_bins = 17, 32, 4, 64
    logits = torch.randn(num_tokens, num_experts, device=device, dtype=torch.float32)
    expert_bias = torch.zeros(num_experts, device=device, dtype=torch.float32)
    histogram = torch.zeros(num_experts, num_bins, device=device, dtype=torch.int32)
    bin_bounds = torch.tensor([-1.0, 1.0], device=device, dtype=torch.float32)

    probs, routing_map = fused_topk_with_score_function(
        logits,
        topk,
        False,
        None,
        None,
        None,
        "sigmoid",
        expert_bias,
        qb_histogram=histogram,
        qb_bin_bounds=bin_bounds,
        qb_histogram_mode=histogram_mode,
    )
    torch.cuda.synchronize(logits_device)

    assert probs.device == device
    assert routing_map.device == device
    assert histogram.sum().item() == num_tokens * num_experts
    assert torch.cuda.current_device() == current_device


@pytest.mark.parametrize("histogram_mode", ["two_kernel", "fused_atomic"])
@pytest.mark.parametrize("invalid_bounds", ["equal", "reversed", "nonfinite"])
@pytest.mark.parametrize("delta", [False, True])
def test_qb_topk_rejects_invalid_bin_bounds(histogram_mode, invalid_bounds, delta):
    logits = torch.randn(8, 16, device="cuda", dtype=torch.float32)
    expert_bias = torch.zeros(16, device="cuda", dtype=torch.float32)
    histogram = torch.zeros(16, 32, device="cuda", dtype=torch.int32)
    bounds = {
        "equal": [1.0, 1.0],
        "reversed": [1.0, -1.0],
        "nonfinite": [float("nan"), 1.0],
    }[invalid_bounds]
    bin_bounds = torch.tensor(bounds, device="cuda", dtype=torch.float32)
    with pytest.raises(ValueError, match="finite with lower < upper"):
        fused_topk_with_score_function(
            logits,
            4,
            False,
            None,
            None,
            None,
            "sigmoid",
            expert_bias,
            qb_histogram=histogram,
            qb_bin_bounds=bin_bounds,
            qb_histogram_mode=histogram_mode,
            return_qb_histogram_delta=delta,
        )


@pytest.mark.parametrize("histogram_mode", ["two_kernel", "fused_atomic"])
def test_qb_topk_revalidates_updated_bin_bounds(histogram_mode):
    logits = torch.randn(8, 16, device="cuda", dtype=torch.float32)
    expert_bias = torch.zeros(16, device="cuda", dtype=torch.float32)
    histogram = torch.zeros(16, 32, device="cuda", dtype=torch.int32)
    bin_bounds = torch.tensor([-1.0, 1.0], device="cuda", dtype=torch.float32)
    fused_topk_with_score_function(
        logits,
        4,
        False,
        None,
        None,
        None,
        "sigmoid",
        expert_bias,
        qb_histogram=histogram,
        qb_bin_bounds=bin_bounds,
        qb_histogram_mode=histogram_mode,
    )
    bin_bounds.fill_(0.0)
    with pytest.raises(ValueError, match="finite with lower < upper"):
        fused_topk_with_score_function(
            logits,
            4,
            False,
            None,
            None,
            None,
            "sigmoid",
            expert_bias,
            qb_histogram=histogram,
            qb_bin_bounds=bin_bounds,
            qb_histogram_mode=histogram_mode,
        )


@pytest.mark.parametrize(
    "histogram_mode",
    [QBHistogramMode.TWO_KERNEL, QBHistogramMode.FUSED_ATOMIC],
)
@pytest.mark.parametrize("use_dense_indices", [False, True])
def test_qb_raw_binding_rejects_invalid_bin_bounds_recoverably(histogram_mode, use_dense_indices):
    logits = torch.randn(8, 16, device="cuda", dtype=torch.float32)
    expert_bias = torch.zeros(16, device="cuda", dtype=torch.float32)
    histogram = torch.zeros(16, 32, device="cuda", dtype=torch.int32)
    topk_indices = (
        torch.empty(8, 4, device="cuda", dtype=torch.int32) if use_dense_indices else None
    )
    invalid_bounds = torch.tensor([1.0, 1.0], device="cuda", dtype=torch.float32)
    with pytest.raises(RuntimeError, match="finite with lower < upper"):
        tex.fused_topk_with_score_function_qb_fwd(
            logits,
            4,
            None,
            expert_bias,
            int(RoutingMapFormat.BYTEMAP),
            topk_indices,
            histogram,
            invalid_bounds,
            histogram_mode,
        )

    # The validation error must not poison the CUDA context.
    valid_bounds = torch.tensor([-1.0, 1.0], device="cuda", dtype=torch.float32)
    tex.fused_topk_with_score_function_qb_fwd(
        logits,
        4,
        None,
        expert_bias,
        int(RoutingMapFormat.BYTEMAP),
        topk_indices,
        histogram,
        valid_bounds,
        histogram_mode,
    )
    torch.cuda.synchronize()


@pytest.mark.parametrize("histogram_mode", ["two_kernel", "fused_atomic"])
@pytest.mark.parametrize("use_dense_indices", [False, True])
def test_qb_topk_cuda_graph_uses_mutable_bounds(histogram_mode, use_dense_indices):
    logits = torch.randn(8, 16, device="cuda", dtype=torch.float32)
    expert_bias = torch.zeros(16, device="cuda", dtype=torch.float32)
    histogram = torch.zeros(16, 32, device="cuda", dtype=torch.int32)
    bin_bounds = torch.tensor([-1.0, 1.0], device="cuda", dtype=torch.float32)
    topk_indices = (
        torch.empty(8, 4, device="cuda", dtype=torch.int32) if use_dense_indices else None
    )

    def run_router():
        return fused_topk_with_score_function(
            logits,
            4,
            False,
            None,
            None,
            None,
            "sigmoid",
            expert_bias,
            topk_indices=topk_indices,
            qb_histogram=histogram,
            qb_bin_bounds=bin_bounds,
            qb_histogram_mode=histogram_mode,
        )

    run_router()
    bounds_data_ptr = bin_bounds.data_ptr()
    initial_bounds = bin_bounds.clone()

    histogram.zero_()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        probs, routing_output = run_router()

    bin_bounds.copy_(torch.tensor([-0.75, 0.25], device="cuda"))
    assert bin_bounds.data_ptr() == bounds_data_ptr
    histogram.zero_()
    graph.replay()
    torch.cuda.synchronize()
    reference = qb_topk_score_function_pytorch(
        logits, 4, expert_bias, bin_bounds, histogram.shape[1]
    )
    torch.testing.assert_close(probs, reference["probs"])
    if use_dense_indices:
        torch.testing.assert_close(
            topk_indices_to_routing_map(routing_output, logits.shape[1]),
            reference["routing_map"],
        )
    else:
        torch.testing.assert_close(routing_output, reference["routing_map"])
    torch.testing.assert_close(histogram, reference["histogram"])
    initial_reference = qb_topk_score_function_pytorch(
        logits, 4, expert_bias, initial_bounds, histogram.shape[1]
    )
    assert not torch.equal(histogram, initial_reference["histogram"])


@pytest.mark.parametrize("histogram_mode", ["two_kernel", "fused_atomic"])
def test_qb_topk_cuda_graph_captures_bounds_update(histogram_mode):
    logits = torch.randn(8, 16, device="cuda", dtype=torch.float32)
    expert_bias = torch.zeros(16, device="cuda", dtype=torch.float32)
    histogram = torch.zeros(16, 32, device="cuda", dtype=torch.int32)
    bin_bounds = torch.tensor([-1.0, 1.0], device="cuda", dtype=torch.float32)
    next_bounds = torch.tensor([-0.75, 0.25], device="cuda", dtype=torch.float32)

    def run_iteration():
        probs, routing_map = fused_topk_with_score_function(
            logits,
            4,
            False,
            None,
            None,
            None,
            "sigmoid",
            expert_bias,
            qb_histogram=histogram,
            qb_bin_bounds=bin_bounds,
            qb_histogram_mode=histogram_mode,
        )
        bin_bounds.copy_(next_bounds)
        mark_qb_bin_bounds_validated(bin_bounds)
        return probs, routing_map

    # Match full-iteration capture: an eager warmup ends with a bounds update, and the same
    # update is part of the captured iteration.
    run_iteration()
    histogram.zero_()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        probs, routing_map = run_iteration()
    histogram.zero_()
    graph.replay()
    torch.cuda.synchronize()

    reference = qb_topk_score_function_pytorch(
        logits, 4, expert_bias, next_bounds, histogram.shape[1]
    )
    torch.testing.assert_close(probs, reference["probs"])
    torch.testing.assert_close(routing_map, reference["routing_map"])
    torch.testing.assert_close(histogram, reference["histogram"])
    torch.testing.assert_close(bin_bounds, next_bounds)


@pytest.mark.parametrize("histogram_mode", ["two_kernel", "fused_atomic"])
def test_qb_topk_eager_call_revalidates_after_captured_bounds_update(histogram_mode):
    logits = torch.randn(8, 16, device="cuda", dtype=torch.float32)
    expert_bias = torch.zeros(16, device="cuda", dtype=torch.float32)
    histogram = torch.zeros(16, 32, device="cuda", dtype=torch.int32)
    bin_bounds = torch.tensor([-1.0, 1.0], device="cuda", dtype=torch.float32)
    next_bounds = torch.tensor([-0.75, 0.25], device="cuda", dtype=torch.float32)

    def run_router():
        return fused_topk_with_score_function(
            logits,
            4,
            False,
            None,
            None,
            None,
            "sigmoid",
            expert_bias,
            qb_histogram=histogram,
            qb_bin_bounds=bin_bounds,
            qb_histogram_mode=histogram_mode,
        )

    run_router()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        bin_bounds.copy_(next_bounds)
        mark_qb_bin_bounds_validated(bin_bounds)

    # Replayed device writes do not advance bin_bounds._version. The capture-time marker must not
    # let a subsequent eager router call inherit trust in the replay-produced values.
    next_bounds.zero_()
    graph.replay()
    torch.cuda.synchronize()
    with pytest.raises(ValueError, match="finite with lower < upper"):
        run_router()


@pytest.mark.parametrize("histogram_mode", ["two_kernel", "fused_atomic"])
def test_qb_topk_cuda_graph_rejects_unvalidated_bounds(histogram_mode):
    logits = torch.randn(8, 16, device="cuda", dtype=torch.float32)
    expert_bias = torch.zeros(16, device="cuda", dtype=torch.float32)
    histogram = torch.zeros(16, 32, device="cuda", dtype=torch.int32)
    bin_bounds = torch.tensor([-1.0, 1.0], device="cuda", dtype=torch.float32)

    with pytest.raises(RuntimeError, match="current version must be validated"):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            fused_topk_with_score_function(
                logits,
                4,
                False,
                None,
                None,
                None,
                "sigmoid",
                expert_bias,
                qb_histogram=histogram,
                qb_bin_bounds=bin_bounds,
                qb_histogram_mode=histogram_mode,
            )


@pytest.mark.parametrize("histogram_mode", ["two_kernel", "fused_atomic"])
def test_qb_topk_cuda_graph_rejects_stale_bounds_validation(histogram_mode):
    logits = torch.randn(8, 16, device="cuda", dtype=torch.float32)
    expert_bias = torch.zeros(16, device="cuda", dtype=torch.float32)
    histogram = torch.zeros(16, 32, device="cuda", dtype=torch.int32)
    bin_bounds = torch.tensor([-1.0, 1.0], device="cuda", dtype=torch.float32)

    def run_router():
        return fused_topk_with_score_function(
            logits,
            4,
            False,
            None,
            None,
            None,
            "sigmoid",
            expert_bias,
            qb_histogram=histogram,
            qb_bin_bounds=bin_bounds,
            qb_histogram_mode=histogram_mode,
        )

    run_router()
    bin_bounds.zero_()
    with pytest.raises(RuntimeError, match="current version must be validated"):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run_router()


@pytest.mark.parametrize(
    "histogram_mode",
    [QBHistogramMode.TWO_KERNEL, QBHistogramMode.FUSED_ATOMIC],
)
def test_qb_topk_plus_one_tie_and_bin_clamping(histogram_mode):
    logits = torch.tensor(
        [
            [3.0, 3.0, 3.0, -1.0, -2.0, -3.0, -4.0, -5.0],
            [5.0, 4.0, 3.0, 2.0, -2.0, -3.0, -4.0, -5.0],
        ],
        device="cuda",
        dtype=torch.float32,
    )
    _, num_experts = logits.shape
    topk = 2
    num_bins = 8
    expert_bias = torch.zeros(num_experts, device="cuda", dtype=torch.float32)
    bin_bounds = torch.tensor([-0.05, 0.05], device="cuda", dtype=torch.float32)
    reference = qb_topk_score_function_pytorch(
        logits,
        topk,
        expert_bias,
        bin_bounds,
        num_bins,
    )
    histogram = torch.zeros(num_experts, num_bins, device="cuda", dtype=torch.int32)

    probs, routing_map, raw_scores, cutoff, histogram = tex.fused_topk_with_score_function_qb_fwd(
        logits,
        topk,
        None,
        expert_bias,
        int(RoutingMapFormat.BYTEMAP),
        None,
        histogram,
        bin_bounds,
        histogram_mode,
    )

    torch.testing.assert_close(probs, reference["probs"])
    torch.testing.assert_close(routing_map, reference["routing_map"])
    torch.testing.assert_close(raw_scores, reference["raw_scores"])
    torch.testing.assert_close(cutoff, reference["cutoff"])
    torch.testing.assert_close(histogram, reference["histogram"])
    assert histogram[:, 0].sum() > 0
    assert histogram[:, -1].sum() > 0
    # The first token has exactly Top-(k+1) equal scores. The deterministic
    # compaction rule drops the largest expert ID at the cutoff.
    assert routing_map[0, :3].tolist() == [True, True, False]


@pytest.mark.parametrize(
    "execution,dtype,num_tokens,num_experts,topk,use_pre_softmax,group_topk,scaling_factor,topk_index_dtype",
    [
        ("eager", *case)
        for case in product(
            [torch.float32],
            [2048, 7168, 14234],
            [1024, 128, 32],
            [4, 8, 16, 32],
            [True, False],
            [None, 4],
            [None, 1.2],
            [None, torch.int16, torch.int32, torch.int64],
        )
    ]
    + [
        ("compile", torch.float32, 64, 32, 4, True, None, None, None),
        ("compile", torch.float16, 64, 32, 4, False, None, 1.2, torch.int16),
        ("compile", torch.bfloat16, 64, 32, 4, False, 4, 1.2, torch.int32),
        ("compile", torch.float32, 64, 130, 16, True, None, None, torch.int64),
        ("compile", torch.bfloat16, 64, 32, 4, True, None, 1.2, None),
    ],
)
def test_topk_softmax(
    dtype,
    num_tokens,
    num_experts,
    topk,
    use_pre_softmax,
    group_topk,
    scaling_factor,
    topk_index_dtype,
    execution,
):
    num_groups = 8 if group_topk else None
    run_comparison(
        dtype=dtype,
        num_tokens=num_tokens,
        num_experts=num_experts,
        topk=topk,
        use_pre_softmax=use_pre_softmax,
        num_groups=num_groups,
        group_topk=group_topk,
        scaling_factor=scaling_factor,
        score_function="softmax",
        enable_bias=False,
        topk_output_mode="dense" if topk_index_dtype is not None else "sparse",
        topk_index_dtype=topk_index_dtype or torch.int16,
        execution=execution,
    )


@pytest.mark.parametrize(
    "execution,topk_index_dtype", product(["eager", "compile"], [None, torch.int16])
)
def test_topk_preserves_leading_dims(topk_index_dtype, execution):
    num_tokens = 128
    num_experts = 32
    topk = 4
    logits = torch.randn(num_tokens, 2, num_experts, device="cuda", dtype=torch.float32)
    topk_indices = None
    if topk_index_dtype is not None:
        topk_indices = torch.empty(num_tokens, 2, topk, device="cuda", dtype=topk_index_dtype)

    router = _router_function(fused_topk_with_score_function, execution)
    probs, routing_output = router(
        logits=logits,
        topk=topk,
        use_pre_softmax=False,
        num_groups=None,
        group_topk=None,
        scaling_factor=None,
        score_function="softmax",
        expert_bias=None,
        topk_indices=topk_indices,
        overwrite_topk_indices=execution == "eager",
    )

    assert probs.shape == logits.shape
    expected_routing_shape = topk_indices.shape if topk_indices is not None else logits.shape
    assert routing_output.shape == expected_routing_shape


@pytest.mark.parametrize(
    "execution,dtype,num_tokens,num_experts,topk,score_function,routing_map_format",
    [
        ("eager", *case, "bytemap")
        for case in product(
            [torch.float32],
            [2048, 7168],
            [1024, 256, 128, 32],
            [1, 4, 8, 16, 32],
            ["softmax", "sigmoid", "sqrtsoftplus"],
        )
    ]
    + [
        ("compile", dtype, 64, 32, 4, score, "bitmap_u8" if i % 2 else "bytemap")
        for i, (dtype, score) in enumerate(
            product(
                [torch.float32, torch.float16, torch.bfloat16],
                ["softmax", "sigmoid", "sqrtsoftplus"],
            )
        )
    ],
)
def test_fused_scores_for_aux_loss(
    dtype,
    num_tokens,
    num_experts,
    topk,
    score_function,
    execution,
    routing_map_format,
):
    if topk >= num_experts:
        pytest.skip(f"topk ({topk}) >= num_experts ({num_experts})")
    if score_function in ("sigmoid", "sqrtsoftplus"):
        # Construct logits with a narrow range to avoid very small activation values
        offset = torch.arange(-num_tokens // 2, num_tokens // 2, dtype=dtype, device="cuda") * 1e-4
        logits = (
            torch.arange(-num_experts // 2, num_experts // 2, device="cuda", dtype=dtype) * 1e-2
        )
        logits = logits.unsqueeze(0).repeat(num_tokens, 1) + offset.unsqueeze(1)
    else:
        logits = (
            torch.arange(
                -num_tokens * num_experts // 2,
                num_tokens * num_experts // 2,
                device="cuda",
                dtype=dtype,
            )
            * 1e-4
        )
        logits = logits.view(num_tokens, num_experts)
    if dtype != torch.float32:
        logits = torch.linspace(-2, 2, num_experts, device="cuda").repeat(num_tokens, 1).to(dtype)
    logits.requires_grad = True

    logits_clone = deepcopy(logits)
    logits_clone.requires_grad = True

    logits = logits.detach().float().requires_grad_(True)
    routing_map, scores = compute_scores_for_aux_loss_pytorch(
        logits=logits,
        topk=topk,
        score_function=score_function,
    )

    scores_fn = _router_function(fused_compute_score_for_moe_aux_loss, execution)
    routing_map_fused, scores_fused = scores_fn(
        logits=logits_clone,
        topk=topk,
        score_function=score_function,
        routing_map_format=routing_map_format,
    )

    assert scores_fused.dtype == torch.float32
    assert not routing_map_fused.requires_grad
    if routing_map_format == "bitmap_u8":
        routing_map = _bytemap_to_bitmap_u8(routing_map)
    atol, rtol = _get_tolerances(torch.float32, num_experts)
    torch.testing.assert_close(scores, scores_fused, atol=atol, rtol=rtol)
    torch.testing.assert_close(routing_map, routing_map_fused)

    grad = torch.randn((*scores_fused.shape, 2), device="cuda")[..., 0]
    scores.backward(grad)
    scores_fused.backward(grad)
    assert logits_clone.grad.dtype == dtype
    atol, rtol = _get_tolerances(dtype, num_experts)
    torch.testing.assert_close(logits.grad, logits_clone.grad.float(), atol=atol, rtol=rtol)


@pytest.mark.parametrize(
    "execution,dtype,num_tokens,num_experts,topk,expert_multiplier,tensor_count",
    [
        ("eager", *case, False)
        for case in product(
            [torch.float32], [2048, 7168, 14234], [1024, 256, 128, 32], [4, 32], [1, 2]
        )
    ]
    + [
        ("compile", dtype, 64, 32, 4, 2 if tensor_count else 1, tensor_count)
        for dtype, tensor_count in product(
            [torch.float32, torch.float16, torch.bfloat16], [False, True]
        )
    ],
)
def test_fused_moe_aux_loss(
    dtype, num_tokens, num_experts, topk, expert_multiplier, execution, tensor_count
):
    if topk >= num_experts:
        pytest.skip(f"topk ({topk}) >= num_experts ({num_experts})")
    # Sequence aux loss batches independent sequences along the expert dimension.
    num_cols = num_experts * expert_multiplier
    # Construct the special probs to avoid inf in the sigmoid function
    offset = torch.arange(-num_tokens // 2, num_tokens // 2, dtype=dtype, device="cuda") * 1e-4
    probs = torch.arange(-num_cols // 2, num_cols // 2, device="cuda", dtype=dtype) * 1e-2
    probs = probs.unsqueeze(0).repeat(num_tokens, 1) + offset.unsqueeze(1)
    probs = probs.view(num_tokens, num_cols)
    probs.requires_grad = True

    tokens_per_expert = torch.randint(1, 1000, (num_cols,), device="cuda", dtype=torch.int32)
    coeff = 0.01

    probs_clone = deepcopy(probs)
    probs_clone.requires_grad = True

    probs = probs.detach().float().requires_grad_(True)
    aux_loss = aux_loss_pytorch(
        probs=probs,
        tokens_per_expert=tokens_per_expert,
        total_num_tokens=num_tokens,
        topk=topk,
        num_experts=num_experts,
        moe_aux_loss_coeff=coeff,
    )

    total = (
        torch.tensor(num_tokens, device="cuda", dtype=torch.int64) if tensor_count else num_tokens
    )
    loss_fn = _router_function(fused_moe_aux_loss, execution)
    aux_loss_fused = loss_fn(
        probs=probs_clone,
        tokens_per_expert=tokens_per_expert,
        total_num_tokens=total,
        num_experts=num_experts,
        topk=topk,
        coeff=coeff,
    )

    assert aux_loss_fused.shape == () and aux_loss_fused.dtype == dtype
    atol, rtol = _get_tolerances(dtype, num_cols)
    torch.testing.assert_close(aux_loss, aux_loss_fused.float(), atol=atol, rtol=rtol)

    # Backward
    aux_loss.backward()
    aux_loss_fused.backward()

    torch.testing.assert_close(probs.grad, probs_clone.grad.float(), atol=atol, rtol=rtol)


def test_fused_moe_aux_loss_cuda_graph_capture():
    """CUDA-graph-safe path: total_num_tokens is a device tensor whose value
    changes between replays. Forward and backward must both observe the new
    value via the device-side coefficient computation."""
    dtype = torch.float32
    num_tokens = 4096
    num_experts = 128
    topk = 4
    num_cols = num_experts
    coeff = 0.01

    offset = torch.arange(-num_tokens // 2, num_tokens // 2, dtype=dtype, device="cuda") * 1e-4
    probs = (
        torch.arange(-num_cols // 2, num_cols // 2, device="cuda", dtype=dtype) * 1e-2
    ).unsqueeze(0).repeat(num_tokens, 1) + offset.unsqueeze(1)
    probs = probs.contiguous().requires_grad_(True)
    tokens_per_expert = torch.randint(1, 1000, (num_cols,), device="cuda", dtype=torch.int32)

    total_num_tokens_dev = torch.tensor(num_tokens, dtype=torch.int64, device="cuda")

    # Warmup on a side stream to satisfy CUDA Graph capture requirements.
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            warmup_out = fused_moe_aux_loss(
                probs=probs,
                tokens_per_expert=tokens_per_expert,
                total_num_tokens=total_num_tokens_dev,
                num_experts=num_experts,
                topk=topk,
                coeff=coeff,
            )
            torch.autograd.grad(warmup_out, probs)
        del warmup_out
    torch.cuda.current_stream().wait_stream(s)

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = fused_moe_aux_loss(
            probs=probs,
            tokens_per_expert=tokens_per_expert,
            total_num_tokens=total_num_tokens_dev,
            num_experts=num_experts,
            topk=topk,
            coeff=coeff,
        )
        (grad_probs,) = torch.autograd.grad(out, probs)

    atol, rtol = _get_tolerances(dtype, num_cols)
    # Replay with several distinct token counts; the captured graph must pick
    # up each new value through total_num_tokens_dev.
    for new_total in (num_tokens, num_tokens // 2, num_tokens * 2 - 17):
        total_num_tokens_dev.fill_(new_total)
        g.replay()
        torch.cuda.synchronize()
        ref_probs = probs.detach().clone().requires_grad_(True)
        ref = aux_loss_pytorch(
            probs=ref_probs,
            tokens_per_expert=tokens_per_expert,
            total_num_tokens=new_total,
            topk=topk,
            num_experts=num_experts,
            moe_aux_loss_coeff=coeff,
        )
        (ref_grad_probs,) = torch.autograd.grad(ref, ref_probs)
        torch.testing.assert_close(out, ref, atol=atol, rtol=rtol)
        torch.testing.assert_close(grad_probs, ref_grad_probs, atol=atol, rtol=rtol)


def _bytemap_to_bitmap_u8(bytemap: torch.Tensor) -> torch.Tensor:
    """Reference packer: bool[T, E] -> uint8[T, ceil(E/8)] LSB-first.

    Matches numpy.packbits(..., bitorder='little')
    """
    flat = bytemap.to(torch.uint8).cpu().numpy()
    import numpy as np

    return torch.from_numpy(np.packbits(flat, axis=-1, bitorder="little")).to(bytemap.device)


@pytest.mark.parametrize(
    "execution,dtype,num_tokens,num_experts,topk,score_function",
    [
        ("eager", torch.float32, *shape, score)
        for shape, score in product(
            [(128, 32, 4), (256, 128, 8), (256, 130, 8), (128, 1024, 16)],
            ["softmax", "sigmoid", "sqrtsoftplus"],
        )
    ]
    + [
        ("compile", torch.float32, 64, 130, 4, score)
        for score in ["softmax", "sigmoid", "sqrtsoftplus"]
    ],
)
def test_topk_bitmap_vs_bytemap(dtype, num_tokens, num_experts, topk, score_function, execution):
    """fused_topk_with_score_function should produce identical probs and an
    LSB-packed bitmap routing_map when routing_map_format=BITMAP_U8, and
    backward gradients should match the bytemap path exactly."""
    if topk >= num_experts:
        pytest.skip(f"topk ({topk}) >= num_experts ({num_experts})")
    if score_function in ("sigmoid", "sqrtsoftplus"):
        offset = torch.arange(-num_tokens // 2, num_tokens // 2, dtype=dtype, device="cuda") * 1e-4
        logits = (
            torch.arange(-num_experts // 2, num_experts // 2, device="cuda", dtype=dtype) * 1e-2
        )
        logits = logits.unsqueeze(0).repeat(num_tokens, 1) + offset.unsqueeze(1)
    else:
        logits = (
            torch.arange(
                -num_tokens * num_experts // 2,
                num_tokens * num_experts // 2,
                device="cuda",
                dtype=dtype,
            )
            * 1e-4
        )
        logits = logits.view(num_tokens, num_experts)

    logits_byte = logits.detach().clone().requires_grad_(True)
    logits_bit = logits.detach().clone().requires_grad_(True)

    router = _router_function(fused_topk_with_score_function, execution)
    probs_byte, routing_map_byte = router(
        logits=logits_byte,
        topk=topk,
        use_pre_softmax=False,
        num_groups=None,
        group_topk=None,
        scaling_factor=None,
        score_function=score_function,
        expert_bias=None,
        routing_map_format=RoutingMapFormat.BYTEMAP,
    )
    probs_bit, routing_map_bit = router(
        logits=logits_bit,
        topk=topk,
        use_pre_softmax=False,
        num_groups=None,
        group_topk=None,
        scaling_factor=None,
        score_function=score_function,
        expert_bias=None,
        routing_map_format=RoutingMapFormat.BITMAP_U8,
    )

    assert probs_byte.dtype == probs_bit.dtype
    torch.testing.assert_close(probs_byte, probs_bit, atol=0.0, rtol=0.0)

    expected_shape = (num_tokens, (num_experts + 7) // 8)
    assert (
        routing_map_bit.shape == expected_shape
    ), f"Bitmap shape {tuple(routing_map_bit.shape)} != {expected_shape}"
    assert routing_map_bit.dtype == torch.uint8
    assert routing_map_byte.dtype == torch.bool

    packed_expected = _bytemap_to_bitmap_u8(routing_map_byte)
    torch.testing.assert_close(routing_map_bit, packed_expected, atol=0, rtol=0)

    # Backward must be bit-identical across formats.
    grad = torch.randn_like(probs_byte)
    probs_byte.backward(grad)
    probs_bit.backward(grad)
    torch.testing.assert_close(logits_byte.grad, logits_bit.grad, atol=0.0, rtol=0.0)


@pytest.mark.parametrize(
    "execution,dtype,num_tokens,num_experts,topk,score_function",
    [
        ("eager", torch.float32, *shape, score)
        for shape, score in product(
            [(128, 32, 4), (256, 128, 8), (256, 130, 8)], ["softmax", "sigmoid", "sqrtsoftplus"]
        )
    ]
    + [
        ("compile", torch.float32, 64, 130, 4, score)
        for score in ["softmax", "sigmoid", "sqrtsoftplus"]
    ],
)
def test_score_for_aux_loss_bitmap_vs_bytemap(
    dtype, num_tokens, num_experts, topk, score_function, execution
):
    """fused_compute_score_for_moe_aux_loss: bitmap routing_map must equal
    LSB-packed bytemap; scores must be bit-identical across formats."""
    if topk >= num_experts:
        pytest.skip(f"topk ({topk}) >= num_experts ({num_experts})")
    offset = torch.arange(-num_tokens // 2, num_tokens // 2, dtype=dtype, device="cuda") * 1e-4
    logits = torch.arange(-num_experts // 2, num_experts // 2, device="cuda", dtype=dtype) * 1e-2
    logits = logits.unsqueeze(0).repeat(num_tokens, 1) + offset.unsqueeze(1)

    logits_byte = logits.detach().clone().requires_grad_(True)
    logits_bit = logits.detach().clone().requires_grad_(True)

    router = _router_function(fused_compute_score_for_moe_aux_loss, execution)
    routing_map_byte, scores_byte = router(
        logits=logits_byte,
        topk=topk,
        score_function=score_function,
        routing_map_format="bytemap",
    )
    routing_map_bit, scores_bit = router(
        logits=logits_bit,
        topk=topk,
        score_function=score_function,
        routing_map_format="bitmap_u8",
    )

    torch.testing.assert_close(scores_byte, scores_bit, atol=0.0, rtol=0.0)

    expected_shape = (num_tokens, (num_experts + 7) // 8)
    assert routing_map_bit.shape == expected_shape
    assert routing_map_bit.dtype == torch.uint8
    assert routing_map_byte.dtype == torch.bool
    packed_expected = _bytemap_to_bitmap_u8(routing_map_byte)
    torch.testing.assert_close(routing_map_bit, packed_expected, atol=0, rtol=0)

    # Backward parity through scores.
    grad = torch.randn_like(scores_byte)
    scores_byte.backward(grad)
    scores_bit.backward(grad)
    torch.testing.assert_close(logits_byte.grad, logits_bit.grad, atol=0.0, rtol=0.0)


def profile_topk_softmax(
    dtype,
    num_tokens,
    num_experts,
    topk,
    enable_bias,
    use_pre_softmax,
):
    group_topk = 4
    scaling_factor = 1.2
    test_topk_sigmoid(
        torch.float32,
        num_tokens,
        num_experts,
        topk,
        group_topk,
        scaling_factor,
        enable_bias,
    )
    test_topk_softmax(
        torch.float32,
        num_tokens,
        num_experts,
        topk,
        use_pre_softmax,
        group_topk,
        scaling_factor,
    )
    test_topk_sqrtsoftplus(
        torch.float32,
        num_tokens,
        num_experts,
        topk,
        group_topk,
        scaling_factor,
        enable_bias,
    )
