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
import transformer_engine.pytorch.router as te_router
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
        logits, topk, scaling_factor=3.16, score_function="sigmoid", expert_bias=expert_bias
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


def _make_logits(dtype):
    logits = torch.linspace(-2, 2, 32, device="cuda")
    logits = logits + torch.linspace(-0.1, 0.1, 64, device="cuda").unsqueeze(1)
    return logits.reshape(2, 32, 32).to(dtype).requires_grad_(True)


def _check_reference(actual, expected, dtype):
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
@pytest.mark.parametrize("deterministic", [False, True])
def test_router_compile_replay(histogram_mode, index_dtype, deterministic):
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

    def router(logits, bounds, total, indices, histogram):
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
            qb_histogram=histogram if histogram_mode else None,
            qb_bin_bounds=bounds if histogram_mode else None,
            qb_histogram_mode=histogram_mode,
        )
        _, scores = fused_compute_score_for_moe_aux_loss(logits, 4, "sigmoid")
        loss = fused_moe_aux_loss(
            scores.reshape(-1, 32), counts, total, 32, 4, 0.01, deterministic=deterministic
        )
        return (outputs[0].float().square().sum() + loss, *outputs[1:])

    reference_histogram = histogram.clone()
    reference_indices = torch.empty_like(indices) if indices is not None else None
    torch._dynamo.mark_static_address(histogram)
    if indices is not None:
        torch._dynamo.mark_static_address(indices)
    compiled = torch.compile(router, fullgraph=True, mode="reduce-overhead")
    for step, new_bounds in enumerate(([-1.1, 1.2], [-0.3, 0.7], [-0.8, 1.4])):
        torch.compiler.cudagraph_mark_step_begin()
        bounds.copy_(bounds.new_tensor(new_bounds))
        total.fill_(64 + 17 * step)
        logits = _make_logits(torch.bfloat16)
        if step == 1:
            logits = logits.flip(-1).detach().requires_grad_(True)
        eager_logits = logits.detach().clone().requires_grad_(True)
        actual = compiled(logits, bounds, total, indices, histogram)
        expected = router(eager_logits, bounds, total, reference_indices, reference_histogram)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(histogram, reference_histogram)
        if indices is not None:
            assert actual[1].data_ptr() == indices.data_ptr()
            torch.testing.assert_close(indices, reference_indices)
        histogram.add_(1)
        reference_histogram.add_(1)
        bounds.add_(0.1)
        actual_grad = torch.autograd.grad(actual[0], logits)
        expected_grad = torch.autograd.grad(expected[0], eager_logits)
        torch.testing.assert_close(actual_grad, expected_grad)
    for grad_context, inference_logits in (
        (torch.no_grad, logits),
        (torch.inference_mode, logits),
        (torch.enable_grad, logits.detach()),
    ):
        with grad_context():
            torch.compiler.cudagraph_mark_step_begin()
            actual = compiled(inference_logits, bounds, total, indices, histogram)
            expected = router(
                inference_logits, bounds, total, reference_indices, reference_histogram
            )
            torch.testing.assert_close(actual, expected)
            torch.testing.assert_close(histogram, reference_histogram)
            assert not actual[0].requires_grad
            if indices is not None:
                assert actual[1].data_ptr() == indices.data_ptr()
    assert not counters["inductor"]["cudagraph_skips"]


@pytest.mark.parametrize(
    "dtype,index_dtype",
    [(torch.float32, None), (torch.float16, torch.int16), (torch.bfloat16, torch.int64)],
)
@pytest.mark.parametrize("deterministic", [False, True])
def test_router_compile_dynamic(dtype, index_dtype, deterministic):
    torch._dynamo.reset()
    counters.clear()
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
        )
        _, scores = fused_compute_score_for_moe_aux_loss(logits, 4, "sigmoid", 1)
        aux_loss = fused_moe_aux_loss(
            scores, counts, logits.shape[0], 130, 4, 0.01, deterministic=deterministic
        )
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
        _check_reference(actual_grad, eager_grad, dtype)
        assert counters["stats"]["unique_graphs"] == 1


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
    forward_logits = logits.detach()
    if histogram_mode is not None:
        bias = torch.zeros(130, device="cuda")
        histogram = torch.full((130, 64), 7, device="cuda", dtype=torch.int32)
        bounds = torch.tensor([-1.0, 1.0], device="cuda")
        forward = router._qb_indices_forward if indices is not None else router._qb_map_forward
        forward_args = (
            forward_logits,
            4,
            None,
            bias,
            routing_map_format,
            *((indices,) if indices is not None else ()),
            histogram,
            bounds,
            int(histogram_mode),
        )
    else:
        forward = router._topk_indices_forward if indices is not None else router._topk_map_forward
        forward_args = (
            forward_logits,
            4,
            False,
            None,
            None,
            None,
            "sigmoid",
            None,
            routing_map_format,
            *((indices,) if indices is not None else ()),
        )
    torch.library.opcheck(forward, forward_args)
    outputs = forward(*forward_args)
    torch.library.opcheck(
        router._topk_backward,
        (
            indices if indices is not None else outputs[1],
            outputs[-1],
            torch.randn_like(outputs[0]),
            4,
            False,
            None,
            "sigmoid",
            routing_map_format,
            indices is not None,
        ),
    )

    if histogram_mode is None and index_dtype in (None, torch.int16):
        torch.library.opcheck(router._score_forward, (logits, 4, "softmax", routing_map_format))
        scores, _, intermediate = router._score_forward(
            forward_logits, 4, "softmax", routing_map_format
        )
        torch.library.opcheck(
            router._score_backward,
            (intermediate, torch.randn_like(scores), 4, "softmax", dtype),
        )
        counts = torch.ones(130, device="cuda", dtype=torch.int32)
        count = torch.tensor(16, device="cuda", dtype=torch.int64)
        for token_count, deterministic in product((None, count), (False, True)):
            torch.library.opcheck(
                router._aux_loss_forward,
                (logits, counts, 16, token_count, 130, 4, 0.01, deterministic),
            )
            loss, const_buf = router._aux_loss_forward(
                forward_logits, counts, 16, token_count, 130, 4, 0.01, deterministic
            )
            torch.library.opcheck(
                router._aux_loss_backward, (const_buf, counts, 16, 130, torch.randn_like(loss))
            )


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
    )
    assert probs_fused.dtype == dtype and not routing_output_fused.requires_grad
    if topk_output_mode == "dense":
        assert routing_output_fused.data_ptr() == topk_indices.data_ptr()
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
    "execution,dtype,histogram_mode,topk,routing_output_mode,shape",
    [
        ("eager", torch.float32, *case, (257, 896, 1000))
        for case in product(
            ["two_kernel", "fused_atomic"],
            [8, 16],
            ["bytemap", "bitmap_u8", "dense_int16", "dense_int32", "dense_int64"],
        )
    ]
    + [
        (execution, dtype, histogram_mode, topk, layout, (64, 32, 31))
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
def test_qb_topk_histogram(histogram_mode, topk, routing_output_mode, execution, dtype, shape):
    num_tokens, num_experts, num_bins = shape
    small = num_tokens == 64
    logits = torch.randn(num_tokens, num_experts, device="cuda", dtype=dtype, requires_grad=True)
    if small:
        logits = _make_logits(dtype).reshape(num_tokens, num_experts).detach().requires_grad_(True)
    reference_logits = logits.detach().float().requires_grad_(True)
    expert_bias = torch.linspace(-0.2, 0.2, num_experts, device="cuda", dtype=torch.float32)
    bin_bounds = torch.stack((expert_bias.min() - 1.0, expert_bias.max() + 1.0))
    reference_histogram = torch.full(
        (num_experts, num_bins), 7 if small else 0, device="cuda", dtype=torch.int32
    )
    reference = qb_topk_score_function_pytorch(
        reference_logits,
        topk,
        expert_bias,
        bin_bounds,
        num_bins,
        reference_histogram,
    )

    fused_logits = logits.detach().clone().requires_grad_(True)
    fused_histogram = torch.full_like(reference_histogram, 7 if small else 0)
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
    )
    fused_probs, fused_routing_output = outputs[:2]
    if small:
        _check_reference(fused_probs, reference["probs"], dtype)
    else:
        torch.testing.assert_close(fused_probs, reference["probs"])
    assert not fused_routing_output.requires_grad
    assert len(outputs) == 2
    assert fused_histogram._version > histogram_version
    torch.testing.assert_close(fused_histogram, reference["histogram"])
    fused_histogram.add_(1)
    bin_bounds.add_(0.1)
    if dense_dtype is not None:
        assert fused_routing_output.data_ptr() == topk_indices.data_ptr()
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
    if small:
        _check_reference(fused_logits.grad, reference_logits.grad, dtype)
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
    compiled = execution == "compile"
    num_experts, num_bins = (130, 64) if compiled else (64, 1000)
    expert_bias = torch.linspace(-0.1, 0.1, num_experts, device="cuda")
    bounds = torch.stack((expert_bias.min() - 1.0, expert_bias.max() + 1.0))
    histogram = torch.full(
        (num_experts, num_bins), 7 if compiled else 0, device="cuda", dtype=torch.int32
    )
    reference_histogram = histogram.clone()

    def route(logits, histogram, bounds, indices):
        return fused_topk_with_score_function(
            logits,
            8,
            False,
            None,
            None,
            None,
            "sigmoid",
            expert_bias,
            routing_map_format="bitmap_u8" if compiled and indices is None else "bytemap",
            topk_indices=indices,
            qb_histogram=histogram,
            qb_bin_bounds=bounds,
            qb_histogram_mode=histogram_mode,
        )

    if compiled:
        torch._dynamo.reset()
        route = torch.compile(route, fullgraph=True, dynamic=True)
    pending = []
    for num_tokens in (17, 33, 64) if compiled else (127, 193):
        logits = torch.randn(
            num_tokens, num_experts, device="cuda", dtype=dtype, requires_grad=True
        )
        reference = qb_topk_score_function_pytorch(
            logits, 8, expert_bias, bounds, num_bins, reference_histogram
        )
        indices = (
            torch.empty(num_tokens, 8, device="cuda", dtype=torch.int32)
            if histogram_mode == "fused_atomic"
            else None
        )
        probs, routing = route(logits, histogram, bounds, indices)
        torch.testing.assert_close(histogram, reference["histogram"])
        _check_reference(probs, reference["probs"], dtype)
        if indices is not None:
            assert routing.data_ptr() == indices.data_ptr()
            torch.testing.assert_close(
                topk_indices_to_routing_map(routing, num_experts), reference["routing_map"]
            )
        else:
            expected_map = (
                _bytemap_to_bitmap_u8(reference["routing_map"])
                if compiled
                else reference["routing_map"]
            )
            torch.testing.assert_close(routing, expected_map)
        pending.append((logits, probs, reference["probs"]))
        if compiled:
            bounds.add_(0.03)
    for logits, probs, reference_probs in reversed(pending):
        actual_grad = torch.autograd.grad(probs.float().square().sum(), logits)[0]
        ref_grad = torch.autograd.grad(reference_probs.float().square().sum(), logits)[0]
        _check_reference(actual_grad, ref_grad, dtype)


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
@pytest.mark.parametrize("use_dense_indices", [False, True])
@pytest.mark.parametrize(
    "invalid_bounds",
    [[1.0, 1.0], [1.0, -1.0], [float("nan"), 1.0], [float("-inf"), float("inf")]],
    ids=["equal", "reversed", "nan", "infinite"],
)
def test_qb_topk_invalid_bin_bounds_stay_in_histogram(
    histogram_mode, use_dense_indices, invalid_bounds
):
    num_tokens, num_experts, num_bins, topk = 8, 16, 32, 4
    logits = torch.randn(num_tokens, num_experts, device="cuda", dtype=torch.float32)
    expert_bias = torch.zeros(num_experts, device="cuda", dtype=torch.float32)
    histogram = torch.zeros(num_experts, num_bins, device="cuda", dtype=torch.int32)
    bin_bounds = torch.tensor(invalid_bounds, device="cuda", dtype=torch.float32)
    topk_indices = (
        torch.empty(num_tokens, topk, device="cuda", dtype=torch.int32)
        if use_dense_indices
        else None
    )

    # bin_bounds values are not validated, so invalid bounds must neither raise nor write outside
    # the histogram.
    probs, routing_output = fused_topk_with_score_function(
        logits,
        topk,
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
    torch.cuda.synchronize()

    # Routing does not read the bounds, and every (token, expert) pair is counted in some bin.
    valid_bounds = torch.tensor([-1.0, 1.0], device="cuda", dtype=torch.float32)
    reference = qb_topk_score_function_pytorch(logits, topk, expert_bias, valid_bounds, num_bins)
    torch.testing.assert_close(probs, reference["probs"])
    if use_dense_indices:
        torch.testing.assert_close(
            topk_indices_to_routing_map(routing_output, num_experts), reference["routing_map"]
        )
    else:
        torch.testing.assert_close(routing_output, reference["routing_map"])
    assert histogram.sum().item() == num_tokens * num_experts


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


def test_mark_qb_bin_bounds_validated_is_deprecated():
    bin_bounds = torch.tensor([-1.0, 1.0], device="cuda", dtype=torch.float32)
    with pytest.warns(DeprecationWarning, match="mark_qb_bin_bounds_validated is deprecated"):
        mark_qb_bin_bounds_validated(bin_bounds)


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


@pytest.mark.parametrize(
    "path",
    [
        "forward",
        "graph_safe_forward",
        "deterministic_forward",
        "deterministic_graph_safe_forward",
        "backward",
    ],
)
def test_fused_moe_aux_loss_rejects_int_offset_overflow(path):
    num_rows, num_cols = 8_388_609, 256  # Above INT_MAX elements.
    bytes_needed = num_rows * num_cols * 2
    torch.cuda.empty_cache()
    free_bytes, _ = torch.cuda.mem_get_info()
    if free_bytes < bytes_needed + (1 << 30):
        pytest.skip("Needs 4.0 GiB for BF16 input/output plus 1 GiB headroom")

    tokens_per_expert = torch.ones(num_cols, device="cuda", dtype=torch.int32)
    try:
        if path == "backward":
            tex.fused_moe_aux_loss_bwd(
                Const_buf=torch.ones(2, device="cuda", dtype=torch.float32),
                tokens_per_expert=tokens_per_expert,
                num_rows=num_rows,
                num_cols=num_cols,
                grad_aux_loss=torch.ones((), device="cuda", dtype=torch.bfloat16),
            )
        else:
            probs = torch.empty((num_rows, num_cols), device="cuda", dtype=torch.bfloat16)
            arguments = dict(
                probs=probs,
                tokens_per_expert=tokens_per_expert,
                num_experts=num_cols,
                num_rows=num_rows,
                num_cols=num_cols,
                topk=1,
                coeff=0.01,
            )
            if path.startswith("deterministic_"):
                arguments.update(
                    deterministic=True,
                    workspace=te_router._get_moe_aux_loss_workspace(probs.device),
                )
            if path in ("forward", "deterministic_forward"):
                tex.fused_moe_aux_loss_fwd(total_num_tokens=num_rows, **arguments)
            else:
                total = torch.tensor(num_rows, device="cuda", dtype=torch.int64)
                tex.fused_moe_aux_loss_fwd_graph_safe(total_num_tokens=total, **arguments)
    except torch.cuda.OutOfMemoryError:
        pytest.skip("Could not allocate the BF16 input/output tensor")
    except RuntimeError as exc:
        assert "num_rows * num_cols exceeds INT_MAX" in str(exc)
    else:
        pytest.fail("Expected the int-offset guard to reject the shape")


@pytest.mark.parametrize(
    "path",
    [
        "forward",
        "graph_safe_forward",
        "deterministic_forward",
        "deterministic_graph_safe_forward",
        "backward",
    ],
)
def test_fused_moe_aux_loss_largest_accepted_shape(path):
    # Exactly INT_MAX elements with one column: loop counters and grid math must not overflow.
    num_rows, num_cols = 2**31 - 1, 1
    bytes_needed = num_rows * num_cols * 2
    torch.cuda.empty_cache()
    free_bytes, _ = torch.cuda.mem_get_info()
    if free_bytes < bytes_needed + (1 << 30):
        pytest.skip("Needs 4.0 GiB for BF16 input/output plus 1 GiB headroom")

    tokens_per_expert = torch.ones(num_cols, device="cuda", dtype=torch.int32)
    try:
        if path == "backward":
            grad_probs = tex.fused_moe_aux_loss_bwd(
                Const_buf=torch.tensor([0.25, 0.0], device="cuda", dtype=torch.float32),
                tokens_per_expert=tokens_per_expert,
                num_rows=num_rows,
                num_cols=num_cols,
                grad_aux_loss=torch.tensor(2.0, device="cuda", dtype=torch.bfloat16),
            )
            for start in range(0, num_rows, 1 << 24):
                assert torch.all(grad_probs[start : start + (1 << 24)] == 0.5)
        else:
            probs = torch.zeros((num_rows, num_cols), device="cuda", dtype=torch.bfloat16)
            probs[-2:] = 1  # Nonzero only where the row counter is closest to INT_MAX.
            arguments = dict(
                probs=probs,
                tokens_per_expert=tokens_per_expert,
                num_experts=num_cols,
                num_rows=num_rows,
                num_cols=num_cols,
                topk=1,
                coeff=1.0,
            )
            if path.startswith("deterministic_"):
                arguments.update(
                    deterministic=True,
                    workspace=te_router._get_moe_aux_loss_workspace(probs.device),
                )
            if path in ("forward", "deterministic_forward"):
                aux_loss, _ = tex.fused_moe_aux_loss_fwd(total_num_tokens=1, **arguments)
            else:
                total = torch.tensor(1, device="cuda", dtype=torch.int64)
                aux_loss, _ = tex.fused_moe_aux_loss_fwd_graph_safe(
                    total_num_tokens=total, **arguments
                )
            torch.testing.assert_close(aux_loss, probs.new_tensor(2.0), atol=0, rtol=0)
    except torch.cuda.OutOfMemoryError:
        pytest.skip("Could not allocate the BF16 input/output tensor")


def _moe_aux_loss_determinism_inputs(
    dtype, num_tokens, num_experts, expert_multiplier, token_count_dtype
):
    """Asymmetric positive scores spanning several orders of magnitude."""
    generator = torch.Generator(device="cuda").manual_seed(1234)
    scores = torch.rand(
        (num_tokens, expert_multiplier, num_experts), device="cuda", generator=generator
    ).pow(4)
    scores *= torch.logspace(-4, 0, num_experts, device="cuda")
    scores /= scores.sum(dim=-1, keepdim=True)
    probs = scores.reshape(num_tokens, -1).to(dtype).requires_grad_(True)
    tokens_per_expert = torch.randint(
        1, num_tokens + 1, (num_experts * expert_multiplier,), device="cuda", generator=generator
    ).to(token_count_dtype)
    if tokens_per_expert.is_floating_point():
        # Global aux loss may use averaged, non-integral expert counts.
        tokens_per_expert /= 3
    return probs, tokens_per_expert


def _assert_bitwise_equal(actual, expected):
    # Compare storage bits, including for the scalar output and low-precision dtypes.
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    assert torch.equal(actual.reshape(-1).view(torch.uint8), expected.reshape(-1).view(torch.uint8))


@pytest.mark.parametrize("tensor_total", [False, True], ids=["host_total", "tensor_total"])
@pytest.mark.parametrize("deterministic", [None, False, True], ids=["default", "atomic", "fixed"])
def test_fused_moe_aux_loss_deterministic_dispatch(monkeypatch, tensor_total, deterministic):
    """Check Python dispatch and autograd argument count without launching CUDA."""
    probs = torch.ones(3, 2, requires_grad=True)
    tokens_per_expert = torch.tensor([1, 2], dtype=torch.int32)
    total_num_tokens = torch.tensor(3, dtype=torch.int64) if tensor_total else 3
    constants = torch.tensor([3.0, 4.0])
    workspace = torch.empty(256, dtype=torch.float32)
    calls = []
    workspace_requests = []

    def fake_workspace(device):
        workspace_requests.append(device)
        return workspace

    def fake_forward(path, **kwargs):
        calls.append((path, kwargs["deterministic"]))
        assert kwargs["probs"] is probs
        assert kwargs["num_rows"] == 3 and kwargs["num_cols"] == 2
        assert kwargs["workspace"] is (workspace if deterministic is True else None)
        return torch.tensor(2.0), constants

    def fake_scalar_forward(**kwargs):
        return fake_forward("host", **kwargs)

    def fake_tensor_forward(**kwargs):
        return fake_forward("tensor", **kwargs)

    def fake_backward(**kwargs):
        assert kwargs["Const_buf"] is constants
        assert kwargs["tokens_per_expert"] is tokens_per_expert
        return torch.full_like(probs, 7.0) * kwargs["grad_aux_loss"]

    monkeypatch.setattr(tex, "fused_moe_aux_loss_fwd", fake_scalar_forward)
    monkeypatch.setattr(tex, "fused_moe_aux_loss_fwd_graph_safe", fake_tensor_forward)
    monkeypatch.setattr(tex, "fused_moe_aux_loss_bwd", fake_backward)
    monkeypatch.setattr(te_router, "_get_moe_aux_loss_workspace", fake_workspace)
    kwargs = {} if deterministic is None else {"deterministic": deterministic}
    out = fused_moe_aux_loss(probs, tokens_per_expert, total_num_tokens, 2, 1, 0.01, **kwargs)
    out.backward(torch.tensor(0.5))

    assert calls == [("tensor" if tensor_total else "host", deterministic is True)]
    assert workspace_requests == ([probs.device] if deterministic is True else [])
    torch.testing.assert_close(probs.grad, torch.full_like(probs, 3.5))


def test_fused_moe_aux_loss_workspace_reuse():
    """Eager calls share scratch only when ordered on the same device/stream."""
    device = torch.device("cuda", torch.cuda.current_device())
    shape, dtype = te_router._get_moe_aux_loss_workspace_config()
    workspaces = []
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    for stream in streams:
        with torch.cuda.stream(stream):
            workspace = te_router._get_moe_aux_loss_workspace(device)
            again = te_router._get_moe_aux_loss_workspace(device)
        assert workspace.data_ptr() == again.data_ptr()
        assert workspace.device == device and workspace.dtype == dtype
        assert tuple(workspace.shape) == shape
        workspaces.append(workspace)
    assert workspaces[0].data_ptr() != workspaces[1].data_ptr()


@pytest.mark.parametrize("device_total", [False, True], ids=["host_total", "device_total"])
@pytest.mark.parametrize(
    "workspace_kind,match",
    [
        ("missing", "requires a workspace"),
        ("empty", "must not be empty"),
        ("undersized", "needs at least"),
        ("wrong_dtype", "contiguous FP32"),
        ("noncontiguous", "contiguous FP32"),
        ("cpu", "same CUDA device"),
    ],
)
def test_fused_moe_aux_loss_invalid_workspace(device_total, workspace_kind, match):
    """Reject unsafe scratch before either forward variant launches kernels."""
    num_tokens, num_experts = 257, 8
    probs = torch.ones(num_tokens, num_experts, device="cuda")
    counts = torch.ones(num_experts, dtype=torch.int32, device="cuda")
    shape, dtype = te_router._get_moe_aux_loss_workspace_config()
    workspace = torch.empty(shape, dtype=dtype, device="cuda")
    if workspace_kind == "missing":
        workspace = None
    elif workspace_kind == "empty":
        workspace = workspace[:0]
    elif workspace_kind == "undersized":
        workspace = workspace[:1]
    elif workspace_kind == "wrong_dtype":
        workspace = workspace.half()
    elif workspace_kind == "noncontiguous":
        workspace = workspace.repeat_interleave(2)[::2]
    elif workspace_kind == "cpu":
        workspace = torch.empty(shape, dtype=dtype)
    total = (
        torch.tensor(num_tokens, dtype=torch.int64, device="cuda") if device_total else num_tokens
    )
    forward = tex.fused_moe_aux_loss_fwd_graph_safe if device_total else tex.fused_moe_aux_loss_fwd
    with pytest.raises(RuntimeError, match=match):
        forward(
            probs, counts, total, num_experts, num_tokens, num_experts, 1, 0.01, True, workspace
        )


@pytest.mark.parametrize("execution", ["eager", "compile"])
def test_fused_moe_aux_loss_cuda_graph_workspace_isolation(monkeypatch, execution):
    """Graphs captured on one stream need private scratch for concurrent replay."""
    num_tokens, num_experts, topk, coeff = 1025, 128, 4, 0.01
    probs_a, counts_a = _moe_aux_loss_determinism_inputs(
        torch.float32, num_tokens, num_experts, 1, torch.int32
    )
    inputs = [
        (probs_a, counts_a, torch.tensor(num_tokens, dtype=torch.int64, device="cuda")),
        (
            probs_a.detach().flip((1,)).requires_grad_(True),
            counts_a.roll(3) + 1,
            torch.tensor(num_tokens * 2 + 1, dtype=torch.int64, device="cuda"),
        ),
    ]

    loss_fn = _router_function(fused_moe_aux_loss, execution)

    def run(probs, counts, total):
        out = loss_fn(probs, counts, total, num_experts, topk, coeff, deterministic=True)
        (grad,) = torch.autograd.grad(out, probs)
        return out, grad

    capture_stream = torch.cuda.Stream()
    capture_stream.wait_stream(torch.cuda.current_stream())
    expected = []
    with torch.cuda.stream(capture_stream):
        eager_workspace = te_router._get_moe_aux_loss_workspace(probs_a.device)
        for args in inputs:
            for _ in range(3):
                warmup_out, warmup_grad = run(*args)
            expected.append((warmup_out.detach().clone(), warmup_grad.clone()))
        del warmup_out, warmup_grad
    torch.cuda.current_stream().wait_stream(capture_stream)

    # Record addresses only, so the test does not extend scratch tensor lifetimes.
    captured_workspace_ptrs = []
    get_workspace = te_router._get_moe_aux_loss_workspace

    def trace_workspace(device, *, cache=True):
        workspace = get_workspace(device, cache=cache)
        if torch.cuda.is_current_stream_capturing():
            captured_workspace_ptrs.append(workspace.data_ptr())
        return workspace

    monkeypatch.setattr(te_router, "_get_moe_aux_loss_workspace", trace_workspace)
    graphs, captured_outputs = [], []
    for args in inputs:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=capture_stream):
            outputs = run(*args)
        graphs.append(graph)
        captured_outputs.append(outputs)
    assert len(captured_workspace_ptrs) == 2
    assert captured_workspace_ptrs[0] != captured_workspace_ptrs[1]
    assert eager_workspace.data_ptr() not in captured_workspace_ptrs
    with torch.cuda.stream(capture_stream):
        assert get_workspace(probs_a.device).data_ptr() == eager_workspace.data_ptr()

    replay_streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    for stream in replay_streams:
        stream.wait_stream(torch.cuda.current_stream())
    for _ in range(8):
        for stream, graph in zip(replay_streams, graphs):
            with torch.cuda.stream(stream):
                graph.replay()
    for stream in replay_streams:
        torch.cuda.current_stream().wait_stream(stream)

    for (out, grad), (expected_out, expected_grad), (probs, counts, total) in zip(
        captured_outputs, expected, inputs
    ):
        _assert_bitwise_equal(out, expected_out)
        _assert_bitwise_equal(grad, expected_grad)
        ref_probs = probs.detach().clone().requires_grad_(True)
        ref = aux_loss_pytorch(ref_probs, counts, int(total.item()), topk, num_experts, coeff)
        (ref_grad,) = torch.autograd.grad(ref, ref_probs)
        torch.testing.assert_close(out, ref, atol=1e-6, rtol=2e-5)
        torch.testing.assert_close(grad, ref_grad, atol=1e-6, rtol=2e-5)


@pytest.mark.parametrize("device_total", [False, True], ids=["host_total", "device_total"])
@pytest.mark.parametrize(
    "dtype,num_tokens,num_experts,expert_multiplier,token_count_dtype",
    [
        pytest.param(torch.float32, 4097, 128, 1, torch.int32, id="multi_cta"),
        pytest.param(torch.float32, 257, 128, 3, torch.float32, id="sequence_global"),
        pytest.param(torch.float32, 3, 7, 1, torch.int64, id="small_partial_count"),
        pytest.param(torch.float32, 129, 1025, 1, torch.int32, id="wide"),
        pytest.param(torch.float16, 4097, 128, 1, torch.int32, id="fp16"),
        pytest.param(torch.bfloat16, 257, 128, 1, torch.bfloat16, id="bf16"),
    ],
)
@pytest.mark.parametrize("execution", ["eager", "compile"])
def test_fused_moe_aux_loss_deterministic(
    dtype, num_tokens, num_experts, expert_multiplier, token_count_dtype, device_total, execution
):
    """Each launch must produce identical loss/gradient bits on the same GPU."""
    probs, tokens_per_expert = _moe_aux_loss_determinism_inputs(
        dtype, num_tokens, num_experts, expert_multiplier, token_count_dtype
    )
    total_num_tokens = (
        torch.tensor(num_tokens, dtype=torch.int64, device="cuda") if device_total else num_tokens
    )
    topk, coeff = 4, 0.01
    grad_output = torch.tensor(0.37, dtype=dtype, device="cuda")

    # Accumulate the reference in FP32, then compare in the public output dtype.
    ref_probs = probs.detach().float().requires_grad_(True)
    ref = aux_loss_pytorch(
        ref_probs, tokens_per_expert.float(), num_tokens, topk, num_experts, coeff
    )
    (ref_grad,) = torch.autograd.grad(ref, ref_probs, grad_outputs=grad_output.float())
    finfo = torch.finfo(dtype)
    rtol = max(2e-5, 2 * finfo.eps)
    atol = finfo.tiny * finfo.eps
    loss_fn = _router_function(fused_moe_aux_loss, execution)
    expected_out = expected_grad = None
    for _ in range(32):
        out = loss_fn(
            probs, tokens_per_expert, total_num_tokens, num_experts, topk, coeff, deterministic=True
        )
        (grad,) = torch.autograd.grad(out, probs, grad_outputs=grad_output)
        if expected_out is None:
            torch.testing.assert_close(out, ref.to(dtype), atol=atol, rtol=rtol)
            torch.testing.assert_close(grad, ref_grad.to(dtype), atol=atol, rtol=rtol)
            expected_out, expected_grad = out.detach().clone(), grad.clone()
        else:
            _assert_bitwise_equal(out, expected_out)
            _assert_bitwise_equal(grad, expected_grad)


@pytest.mark.parametrize("deterministic", [False, True], ids=["atomic", "fixed"])
@pytest.mark.parametrize(
    "num_tokens,num_experts,expert_multiplier,token_count_dtype",
    [
        pytest.param(4096, 128, 1, torch.int32, id="multi_cta"),
        pytest.param(3, 7, 2, torch.float32, id="small_sequence_global"),
    ],
)
def test_fused_moe_aux_loss_cuda_graph_capture(
    num_tokens, num_experts, expert_multiplier, token_count_dtype, deterministic
):
    """CUDA-graph-safe path: total_num_tokens is a device tensor whose value
    changes between replays. Forward and backward must both observe the new
    value via the device-side coefficient computation."""
    dtype = torch.float32
    topk = 4
    num_cols = num_experts * expert_multiplier
    coeff = 0.01

    probs, tokens_per_expert = _moe_aux_loss_determinism_inputs(
        dtype, num_tokens, num_experts, expert_multiplier, token_count_dtype
    )
    original_probs = probs.detach().clone()
    original_counts = tokens_per_expert.clone()

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
                deterministic=deterministic,
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
            deterministic=deterministic,
        )
        (grad_probs,) = torch.autograd.grad(out, probs)

    atol, rtol = _get_tolerances(dtype, num_cols)
    # Replay A -> B -> A in the same graph. Restore both inputs and the device
    # token count so stale partial sums or coefficients cannot pass unnoticed.
    original_out = original_grad = None
    for changed_inputs, new_total in (
        (False, num_tokens),
        (True, num_tokens * 2 + 1),
        (False, num_tokens),
    ):
        with torch.no_grad():
            probs.copy_(original_probs.flip((1,)) if changed_inputs else original_probs)
            tokens_per_expert.copy_(
                original_counts.roll(3) + 1 if changed_inputs else original_counts
            )
        total_num_tokens_dev.fill_(new_total)
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
        expected_out = expected_grad = None
        for _ in range(16):
            g.replay()
            if expected_out is None or not deterministic:
                torch.testing.assert_close(out, ref, atol=atol, rtol=rtol)
                torch.testing.assert_close(grad_probs, ref_grad_probs, atol=atol, rtol=rtol)
                expected_out, expected_grad = out.detach().clone(), grad_probs.clone()
            else:
                _assert_bitwise_equal(out, expected_out)
                _assert_bitwise_equal(grad_probs, expected_grad)
        if deterministic and not changed_inputs:
            if original_out is None:
                original_out, original_grad = expected_out, expected_grad
            else:
                _assert_bitwise_equal(expected_out, original_out)
                _assert_bitwise_equal(expected_grad, original_grad)


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
