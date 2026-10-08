# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Graph-safe grouped row-scaled NVFP4 cast (amax + cast) from device-side routing.

Exercises ``transformer_engine.pytorch.ep.nvfp4_group_row_scaled_cast`` against the
host-split row-scaled cast (``tex.split_quantize`` with row-scaled NVFP4 quantizers).
"""

import pytest
import torch

import transformer_engine.pytorch as te
import transformer_engine_torch as tex
from transformer_engine.pytorch import NVFP4Quantizer
from transformer_engine.pytorch.ep import nvfp4_group_row_scaled_cast

recipe_available, reason_for_no_recipe = te.is_nvfp4_available(return_reason=True)


def _row_scaled_quantizer() -> NVFP4Quantizer:
    return NVFP4Quantizer(
        fp4_dtype=te.DType.kFloat4E2M1,
        rowwise=True,
        columnwise=False,
        with_rht=False,
        with_post_rht_amax=False,
        row_scaled_nvfp4=True,
    )


def _host_split_reference(x_valid: torch.Tensor, split_sections: list[int]):
    quantizers = [_row_scaled_quantizer() for _ in split_sections]
    return tex.split_quantize(x_valid, split_sections, quantizers)


def _gather_reference(reference, split_sections: list[int], N: int):
    """Concatenate the per-expert host-split outputs into contiguous [sum_M, *] buffers."""
    live = [(r, m) for r, m in zip(reference, split_sections) if m > 0]
    data = torch.cat([r._rowwise_data.view(torch.uint8).reshape(m, N // 2) for r, m in live], dim=0)
    scale = torch.cat([r._rowwise_scale_inv.reshape(m, -1) for r, m in live], dim=0)
    amax = torch.cat([r._amax_rowwise.reshape(m) for r, m in live], dim=0)
    return data, scale, amax


def _assert_matches_reference(grouped, reference, split_sections: list[int], N: int) -> None:
    """Compare the grouped output's raw device buffers (valid region) to the reference.

    Reads raw buffers rather than ``split_into_quantized_tensors`` because the wrapper's
    per-expert shapes are frozen at capture time while the device buffers are recomputed.
    """
    sum_m = sum(split_sections)
    ref_data, ref_scale, ref_amax = _gather_reference(reference, split_sections, N)

    capacity = int(grouped.logical_shape[0])
    scale_stride = grouped.scale_inv.numel() // capacity
    g_data = grouped.rowwise_data.view(torch.uint8).reshape(capacity, N // 2)[:sum_m]
    g_scale = grouped.scale_inv.view(torch.uint8).reshape(capacity, scale_stride)[:sum_m]
    g_amax = grouped.amax.reshape(-1)[:sum_m]

    torch.testing.assert_close(g_data, ref_data, atol=0, rtol=0)
    torch.testing.assert_close(g_scale[:, : ref_scale.shape[1]], ref_scale, atol=0, rtol=0)
    torch.testing.assert_close(g_amax, ref_amax, atol=0, rtol=0)


@pytest.mark.skipif(not recipe_available, reason=reason_for_no_recipe)
@pytest.mark.parametrize(
    "split_sections, N",
    [
        ([128], 128),
        ([128, 256, 128], 256),
        ([384, 128, 256], 128),
        ([256, 128], 512),
    ],
)
def test_matches_host_split(split_sections: list[int], N: int) -> None:
    """Graph-safe amax + cast is byte-identical to the host-split row-scaled cast."""
    sum_m = sum(split_sections)
    torch.manual_seed(0)
    x = torch.randn((sum_m, N), dtype=torch.bfloat16, device="cuda")
    tokens_per_expert = torch.tensor(split_sections, dtype=torch.int64, device="cuda")

    grouped = nvfp4_group_row_scaled_cast(x, tokens_per_expert)
    reference = _host_split_reference(x, split_sections)

    _assert_matches_reference(grouped, reference, split_sections, N)


@pytest.mark.skipif(not recipe_available, reason=reason_for_no_recipe)
@pytest.mark.parametrize(
    "split_sections, N, capacity",
    [
        ([128, 128], 256, 512),
        ([256, 128, 128], 128, 768),
    ],
)
def test_paged_stashing_capacity_noop(split_sections: list[int], N: int, capacity: int) -> None:
    """Capacity rows beyond the live token sum are a device-side no-op."""
    sum_m = sum(split_sections)
    assert capacity > sum_m
    torch.manual_seed(1)
    x = torch.randn((capacity, N), dtype=torch.bfloat16, device="cuda")
    x_valid = x[:sum_m, :].clone()
    tokens_per_expert = torch.tensor(split_sections, dtype=torch.int64, device="cuda")

    grouped = nvfp4_group_row_scaled_cast(x, tokens_per_expert)
    reference = _host_split_reference(x_valid, split_sections)

    _assert_matches_reference(grouped, reference, split_sections, N)


@pytest.mark.skipif(not recipe_available, reason=reason_for_no_recipe)
def test_capture_once_replay_changed_routing() -> None:
    """Capture the amax + cast once, then replay it with redistributed routing."""
    N = 256
    num_experts = 3
    capacity = 768
    splits_a = [256, 128, 128]
    splits_b = [128, 256, 128]

    tokens_per_expert = torch.empty(num_experts, dtype=torch.int64, device="cuda")
    x = torch.empty((capacity, N), dtype=torch.bfloat16, device="cuda")

    torch.manual_seed(2)
    x_a = torch.randn((sum(splits_a), N), dtype=torch.bfloat16, device="cuda")
    x_b = torch.randn((sum(splits_b), N), dtype=torch.bfloat16, device="cuda")

    def program_routing(splits: list[int], x_src: torch.Tensor) -> None:
        tokens_per_expert.copy_(torch.tensor(splits, dtype=torch.int64, device="cuda"))
        x.zero_()
        x[: x_src.shape[0], :].copy_(x_src)

    # Warm up outside capture so one-time setup does not land in the captured graph.
    program_routing(splits_a, x_a)
    warmup_stream = torch.cuda.Stream()
    warmup_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup_stream):
        _ = nvfp4_group_row_scaled_cast(x, tokens_per_expert)
    torch.cuda.current_stream().wait_stream(warmup_stream)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        grouped = nvfp4_group_row_scaled_cast(x, tokens_per_expert)

    program_routing(splits_a, x_a)
    graph.replay()
    torch.cuda.synchronize()
    _assert_matches_reference(grouped, _host_split_reference(x_a, splits_a), splits_a, N)

    program_routing(splits_b, x_b)
    graph.replay()
    torch.cuda.synchronize()
    _assert_matches_reference(grouped, _host_split_reference(x_b, splits_b), splits_b, N)
