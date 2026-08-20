# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
import torch
from typing import Optional
from transformer_engine.pytorch.router import (
    QBHistogramMode,
    RoutingMapFormat,
    fused_topk_with_score_function,
    fused_compute_score_for_moe_aux_loss,
    fused_moe_aux_loss,
)
import transformer_engine_torch as tex
import pytest
from copy import deepcopy

seed = 42
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)


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
    # TODO: account for fp16, bf16 as dtype
    if dtype != torch.float32:
        raise NotImplementedError("tolerances implemented for fp32 only")
    eps = 2e-7
    # The worst-case rounding error from summing N values is O(N * eps).
    # Use 2 * num_experts * eps as the tolerance floor so tests pass for
    # large expert counts while remaining tight for small ones.
    atol = max(base_atol, 2 * num_experts * eps)
    rtol = max(base_rtol, 2 * num_experts * eps)
    return atol, rtol


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
    logits.requires_grad = True
    if enable_bias and score_function in ("sigmoid", "sqrtsoftplus"):
        expert_bias = torch.arange(num_experts, device="cuda", dtype=dtype) * 0.1
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
        topk_indices = torch.empty((num_tokens, topk), device="cuda", dtype=topk_index_dtype)

    # Run the fused implementation
    probs_fused, routing_output_fused = fused_topk_with_score_function(
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
    if topk_output_mode == "dense":
        assert routing_output_fused.data_ptr() == topk_indices.data_ptr()
        assert routing_output_fused.dtype == topk_index_dtype
        routing_map_fused = topk_indices_to_routing_map(routing_output_fused, num_experts)
    else:
        routing_map_fused = routing_output_fused

    atol, rtol = _get_tolerances(dtype, num_experts)
    torch.testing.assert_close(probs, probs_fused, atol=atol, rtol=rtol)
    torch.testing.assert_close(routing_map, routing_map_fused)

    # Fake the loss
    loss = torch.sum(probs)
    loss_fused = torch.sum(probs_fused)

    # Backward the loss
    loss.backward()
    loss_fused.backward()

    # Check the gradient
    torch.testing.assert_close(logits.grad, logits_clone.grad, atol=atol, rtol=rtol)


@pytest.mark.parametrize("dtype", [torch.float32])
@pytest.mark.parametrize("num_tokens", [2048, 7168, 8992])
@pytest.mark.parametrize("num_experts", [1024, 128, 32])
@pytest.mark.parametrize("topk", [4, 8, 16, 32])
@pytest.mark.parametrize("group_topk", [None, 4])
@pytest.mark.parametrize("scaling_factor", [None, 1.2])
@pytest.mark.parametrize("enable_bias", [True, False])
@pytest.mark.parametrize("topk_index_dtype", [None, torch.int16, torch.int32, torch.int64])
def test_topk_sigmoid(
    dtype,
    num_tokens,
    num_experts,
    topk,
    group_topk,
    scaling_factor,
    enable_bias,
    topk_index_dtype,
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
    )


@pytest.mark.parametrize("dtype", [torch.float32])
@pytest.mark.parametrize("num_tokens", [2048, 7168, 8992])
@pytest.mark.parametrize("num_experts", [1024, 128, 32])
@pytest.mark.parametrize("topk", [4, 8, 16, 32])
@pytest.mark.parametrize("group_topk", [None, 4])
@pytest.mark.parametrize("scaling_factor", [None, 1.2])
@pytest.mark.parametrize("enable_bias", [True, False])
@pytest.mark.parametrize("topk_index_dtype", [None, torch.int16, torch.int32, torch.int64])
def test_topk_sqrtsoftplus(
    dtype,
    num_tokens,
    num_experts,
    topk,
    group_topk,
    scaling_factor,
    enable_bias,
    topk_index_dtype,
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
    )


@pytest.mark.parametrize("histogram_mode", ["two_kernel", "fused_atomic"])
@pytest.mark.parametrize("topk", [8, 16])
@pytest.mark.parametrize(
    "routing_output_mode",
    ["bytemap", "bitmap_u8", "dense_int16", "dense_int32", "dense_int64"],
)
def test_qb_topk_histogram(histogram_mode, topk, routing_output_mode):
    num_tokens = 257
    num_experts = 896
    num_bins = 1000
    logits = torch.randn(
        num_tokens, num_experts, device="cuda", dtype=torch.float32, requires_grad=True
    )
    expert_bias = torch.linspace(-0.2, 0.2, num_experts, device="cuda", dtype=torch.float32)
    bin_bounds = torch.stack((expert_bias.min() - 1.0, expert_bias.max() + 1.0))
    reference_histogram = torch.zeros(num_experts, num_bins, device="cuda", dtype=torch.int32)
    reference = qb_topk_score_function_pytorch(
        logits,
        topk,
        expert_bias,
        bin_bounds,
        num_bins,
        reference_histogram,
    )

    fused_logits = logits.detach().clone().requires_grad_(True)
    fused_histogram = torch.zeros_like(reference_histogram)
    dense_dtype = {
        "dense_int16": torch.int16,
        "dense_int32": torch.int32,
        "dense_int64": torch.int64,
    }.get(routing_output_mode)
    topk_indices = (
        torch.empty(num_tokens, topk, device="cuda", dtype=dense_dtype)
        if dense_dtype is not None
        else None
    )
    routing_map_format = (
        RoutingMapFormat.BITMAP_U8
        if routing_output_mode == "bitmap_u8"
        else RoutingMapFormat.BYTEMAP
    )
    fused_probs, fused_routing_output = fused_topk_with_score_function(
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
    torch.testing.assert_close(fused_probs, reference["probs"])
    if dense_dtype is not None:
        fused_routing_map = topk_indices_to_routing_map(fused_routing_output, num_experts)
        torch.testing.assert_close(fused_routing_map, reference["routing_map"])
    elif routing_output_mode == "bitmap_u8":
        torch.testing.assert_close(
            fused_routing_output,
            _bytemap_to_bitmap_u8(reference["routing_map"]),
        )
    else:
        torch.testing.assert_close(fused_routing_output, reference["routing_map"])
    torch.testing.assert_close(fused_histogram, reference["histogram"])

    grad = torch.randn_like(fused_probs)
    reference["probs"].backward(grad)
    fused_probs.backward(grad)
    torch.testing.assert_close(fused_logits.grad, logits.grad, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("histogram_mode", ["two_kernel", "fused_atomic"])
def test_qb_histogram_accumulates_microbatches(histogram_mode):
    num_experts = 64
    topk = 8
    num_bins = 1000
    expert_bias = torch.linspace(-0.1, 0.1, num_experts, device="cuda", dtype=torch.float32)
    bin_bounds = torch.stack((expert_bias.min() - 1.0, expert_bias.max() + 1.0))
    reference_histogram = torch.zeros(num_experts, num_bins, device="cuda", dtype=torch.int32)
    fused_histogram = torch.zeros_like(reference_histogram)

    for num_tokens in (127, 193):
        logits = torch.randn(num_tokens, num_experts, device="cuda", dtype=torch.float32)
        qb_topk_score_function_pytorch(
            logits,
            topk,
            expert_bias,
            bin_bounds,
            num_bins,
            reference_histogram,
        )
        fused_topk_with_score_function(
            logits=logits,
            topk=topk,
            use_pre_softmax=False,
            num_groups=None,
            group_topk=None,
            scaling_factor=None,
            score_function="sigmoid",
            expert_bias=expert_bias,
            qb_histogram=fused_histogram,
            qb_bin_bounds=bin_bounds,
            qb_histogram_mode=histogram_mode,
        )

    torch.testing.assert_close(fused_histogram, reference_histogram)


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
@pytest.mark.parametrize("invalid_bounds", ["equal", "reversed", "nonfinite"])
def test_qb_topk_rejects_invalid_bin_bounds(histogram_mode, invalid_bounds):
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
        )


def test_qb_topk_revalidates_updated_bin_bounds():
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
        qb_histogram_mode="fused_atomic",
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
            qb_histogram_mode="fused_atomic",
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
def test_qb_topk_cuda_graph_uses_prevalidated_bounds(histogram_mode):
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
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        probs, routing_map = run_router()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.isfinite(probs).all()
    assert routing_map.sum().item() == logits.shape[0] * 4


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


@pytest.mark.parametrize("dtype", [torch.float32])
@pytest.mark.parametrize("num_tokens", [2048, 7168, 14234])
@pytest.mark.parametrize("num_experts", [1024, 128, 32])
@pytest.mark.parametrize("topk", [4, 8, 16, 32])
@pytest.mark.parametrize("use_pre_softmax", [True, False])
@pytest.mark.parametrize("group_topk", [None, 4])
@pytest.mark.parametrize("scaling_factor", [None, 1.2])
@pytest.mark.parametrize("topk_index_dtype", [None, torch.int16, torch.int32, torch.int64])
def test_topk_softmax(
    dtype,
    num_tokens,
    num_experts,
    topk,
    use_pre_softmax,
    group_topk,
    scaling_factor,
    topk_index_dtype,
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
    )


@pytest.mark.parametrize("topk_index_dtype", [None, torch.int16])
def test_topk_preserves_leading_dims(topk_index_dtype):
    num_tokens = 128
    num_experts = 32
    topk = 4
    logits = torch.randn(num_tokens, 2, num_experts, device="cuda", dtype=torch.float32)
    topk_indices = None
    if topk_index_dtype is not None:
        topk_indices = torch.empty(num_tokens, 2, topk, device="cuda", dtype=topk_index_dtype)

    probs, routing_output = fused_topk_with_score_function(
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


@pytest.mark.parametrize("dtype", [torch.float32])
@pytest.mark.parametrize("num_tokens", [2048, 7168])
@pytest.mark.parametrize("num_experts", [1024, 256, 128, 32])
@pytest.mark.parametrize("topk", [1, 4, 8, 16, 32])
@pytest.mark.parametrize("score_function", ["softmax", "sigmoid", "sqrtsoftplus"])
def test_fused_scores_for_aux_loss(dtype, num_tokens, num_experts, topk, score_function):
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
    logits.requires_grad = True

    logits_clone = deepcopy(logits)
    logits_clone.requires_grad = True

    routing_map, scores = compute_scores_for_aux_loss_pytorch(
        logits=logits,
        topk=topk,
        score_function=score_function,
    )

    routing_map_fused, scores_fused = fused_compute_score_for_moe_aux_loss(
        logits=logits_clone,
        topk=topk,
        score_function=score_function,
    )

    atol, rtol = _get_tolerances(dtype, num_experts)
    torch.testing.assert_close(scores, scores_fused, atol=atol, rtol=rtol)
    torch.testing.assert_close(routing_map, routing_map_fused)

    loss = torch.sum(scores)
    loss.backward()
    loss_fused = torch.sum(scores_fused)
    loss_fused.backward()

    torch.testing.assert_close(logits.grad, logits_clone.grad, atol=atol, rtol=rtol)


@pytest.mark.parametrize("dtype", [torch.float32])
@pytest.mark.parametrize("num_tokens", [2048, 7168, 14234])
@pytest.mark.parametrize("num_experts", [1024, 256, 128, 32])
@pytest.mark.parametrize("topk", [4, 32])
@pytest.mark.parametrize("expert_multiplier", [1, 2])
def test_fused_moe_aux_loss(dtype, num_tokens, num_experts, topk, expert_multiplier):
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

    aux_loss = aux_loss_pytorch(
        probs=probs,
        tokens_per_expert=tokens_per_expert,
        total_num_tokens=num_tokens,
        topk=topk,
        num_experts=num_experts,
        moe_aux_loss_coeff=coeff,
    )

    aux_loss_fused = fused_moe_aux_loss(
        probs=probs_clone,
        tokens_per_expert=tokens_per_expert,
        total_num_tokens=num_tokens,
        num_experts=num_experts,
        topk=topk,
        coeff=coeff,
    )

    atol, rtol = _get_tolerances(dtype, num_cols)
    torch.testing.assert_close(aux_loss, aux_loss_fused, atol=atol, rtol=rtol)

    # Backward
    aux_loss.backward()
    aux_loss_fused.backward()

    torch.testing.assert_close(probs.grad, probs_clone.grad, atol=atol, rtol=rtol)


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


@pytest.mark.parametrize("dtype", [torch.float32])
@pytest.mark.parametrize(
    "num_tokens,num_experts,topk",
    [(128, 32, 4), (256, 128, 8), (256, 130, 8), (128, 1024, 16)],
)
@pytest.mark.parametrize("score_function", ["softmax", "sigmoid", "sqrtsoftplus"])
def test_topk_bitmap_vs_bytemap(dtype, num_tokens, num_experts, topk, score_function):
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

    probs_byte, routing_map_byte = fused_topk_with_score_function(
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
    probs_bit, routing_map_bit = fused_topk_with_score_function(
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

    # Backward parity: grad of probs.sum() must be bit-identical across formats.
    probs_byte.sum().backward()
    probs_bit.sum().backward()
    torch.testing.assert_close(logits_byte.grad, logits_bit.grad, atol=0.0, rtol=0.0)


@pytest.mark.parametrize("dtype", [torch.float32])
@pytest.mark.parametrize(
    "num_tokens,num_experts,topk",
    [(128, 32, 4), (256, 128, 8), (256, 130, 8)],
)
@pytest.mark.parametrize("score_function", ["softmax", "sigmoid", "sqrtsoftplus"])
def test_score_for_aux_loss_bitmap_vs_bytemap(dtype, num_tokens, num_experts, topk, score_function):
    """fused_compute_score_for_moe_aux_loss: bitmap routing_map must equal
    LSB-packed bytemap; scores must be bit-identical across formats."""
    if topk >= num_experts:
        pytest.skip(f"topk ({topk}) >= num_experts ({num_experts})")
    offset = torch.arange(-num_tokens // 2, num_tokens // 2, dtype=dtype, device="cuda") * 1e-4
    logits = torch.arange(-num_experts // 2, num_experts // 2, device="cuda", dtype=dtype) * 1e-2
    logits = logits.unsqueeze(0).repeat(num_tokens, 1) + offset.unsqueeze(1)

    logits_byte = logits.detach().clone().requires_grad_(True)
    logits_bit = logits.detach().clone().requires_grad_(True)

    routing_map_byte, scores_byte = fused_compute_score_for_moe_aux_loss(
        logits=logits_byte,
        topk=topk,
        score_function=score_function,
        routing_map_format="bytemap",
    )
    routing_map_bit, scores_bit = fused_compute_score_for_moe_aux_loss(
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
    scores_byte.sum().backward()
    scores_bit.sum().backward()
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
