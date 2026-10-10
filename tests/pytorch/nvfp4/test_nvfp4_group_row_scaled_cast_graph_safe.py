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


@pytest.mark.skipif(not recipe_available, reason=reason_for_no_recipe)
@pytest.mark.parametrize(
    "split_sections, N",
    [
        ([128, 128], 64),
        ([128] * 65, 128),
    ],
)
def test_split_quantize_falls_back_to_per_expert(split_sections: list[int], N: int) -> None:
    """Shapes the grouped kernel rejects (last dim not 128-aligned, >64 experts) cast per-expert."""
    sum_m = sum(split_sections)
    torch.manual_seed(3)
    x = torch.randn((sum_m, N), dtype=torch.bfloat16, device="cuda")

    outputs = tex.split_quantize(
        x, split_sections, [_row_scaled_quantizer() for _ in split_sections]
    )

    assert len(outputs) == len(split_sections)
    for out, m in zip(outputs, split_sections):
        assert tuple(out.shape) == (m, N)


@pytest.mark.skipif(not recipe_available, reason=reason_for_no_recipe)
@pytest.mark.parametrize(
    "input_shape, split_sections",
    [
        ((256, 2, 128), [128, 128]),  # higher-rank: not a 2D [rows, last_dim] input
        ((512, 128), [128, 128]),  # partial: splits cover 256 of 512 rows
    ],
)
def test_split_quantize_non_grouped_shapes_fall_back(
    input_shape, split_sections: list[int]
) -> None:
    """Higher-rank inputs and partial splits bypass the 2D grouped kernel and cast per-expert,
    matching an individual quantize of each split region."""
    torch.manual_seed(6)
    x = torch.randn(input_shape, dtype=torch.bfloat16, device="cuda")

    outputs = tex.split_quantize(
        x, split_sections, [_row_scaled_quantizer() for _ in split_sections]
    )

    assert len(outputs) == len(split_sections)
    offset = 0
    for out, m in zip(outputs, split_sections):
        ref = _row_scaled_quantizer()(x[offset : offset + m])
        torch.testing.assert_close(
            out._rowwise_data.reshape(-1).view(torch.uint8),
            ref._rowwise_data.reshape(-1).view(torch.uint8),
            atol=0,
            rtol=0,
        )
        offset += m


def _tensor_scaled_quantizer() -> NVFP4Quantizer:
    return NVFP4Quantizer(
        fp4_dtype=te.DType.kFloat4E2M1,
        rowwise=True,
        columnwise=False,
        with_rht=False,
        with_post_rht_amax=False,
        row_scaled_nvfp4=False,
    )


@pytest.mark.skipif(not recipe_available, reason=reason_for_no_recipe)
def test_split_quantize_mixed_row_scaled_falls_back() -> None:
    """A split_quantize that mixes tensor-scaled and row-scaled NVFP4 quantizers casts each split
    per-expert rather than taking the grouped row-scaled path (whose bulk amax, sized from the
    first quantizer, would be overrun by the row-scaled kernel)."""
    split_sections = [128, 128]
    N = 128
    torch.manual_seed(7)
    x = torch.randn((sum(split_sections), N), dtype=torch.bfloat16, device="cuda")

    quantizers = [_tensor_scaled_quantizer(), _row_scaled_quantizer()]
    outputs = tex.split_quantize(x, split_sections, quantizers)

    assert len(outputs) == len(split_sections)
    offset = 0
    for out, m, make_q in zip(
        outputs, split_sections, [_tensor_scaled_quantizer, _row_scaled_quantizer]
    ):
        ref = make_q()(x[offset : offset + m])
        torch.testing.assert_close(
            out._rowwise_data.reshape(-1).view(torch.uint8),
            ref._rowwise_data.reshape(-1).view(torch.uint8),
            atol=0,
            rtol=0,
        )
        offset += m


@pytest.mark.skipif(not recipe_available, reason=reason_for_no_recipe)
def test_split_quantize_empty_routing() -> None:
    """Empty row-scaled routing returns without launching grouped work."""
    split_sections = [0, 0]
    x = torch.empty((0, 128), dtype=torch.bfloat16, device="cuda")

    outputs = tex.split_quantize(
        x, split_sections, [_row_scaled_quantizer() for _ in split_sections]
    )

    assert len(outputs) == len(split_sections)
    for out in outputs:
        assert tuple(out.shape) == (0, 128)


@pytest.mark.skipif(not recipe_available, reason=reason_for_no_recipe)
def test_non_contiguous_input_matches_contiguous() -> None:
    """A non-contiguous input is made contiguous before the graph-safe cast."""
    split_sections = [128, 128]
    N = 256
    sum_m = sum(split_sections)
    torch.manual_seed(4)
    x_t = torch.randn((N, sum_m), dtype=torch.bfloat16, device="cuda").t()
    assert not x_t.is_contiguous()
    tokens_per_expert = torch.tensor(split_sections, dtype=torch.int64, device="cuda")

    grouped = nvfp4_group_row_scaled_cast(x_t, tokens_per_expert)
    reference = _host_split_reference(x_t.contiguous(), split_sections)

    _assert_matches_reference(grouped, reference, split_sections, N)


@pytest.mark.skipif(not recipe_available, reason=reason_for_no_recipe)
def test_strided_first_dims_matches_contiguous() -> None:
    """A strided first_dims view is made contiguous before the cast, so it is not misread."""
    split_sections = [256, 128]
    N = 128
    sum_m = sum(split_sections)
    torch.manual_seed(5)
    x = torch.randn((sum_m, N), dtype=torch.bfloat16, device="cuda")

    # [256, 0, 128, 0][::2] == [256, 128] but strided; a raw data_ptr read would see [256, 0].
    strided = torch.tensor([256, 0, 128, 0], dtype=torch.int64, device="cuda")[::2]
    assert not strided.is_contiguous()
    assert strided.tolist() == split_sections

    grouped = nvfp4_group_row_scaled_cast(x, strided)
    reference = _host_split_reference(x, split_sections)

    _assert_matches_reference(grouped, reference, split_sections, N)


@pytest.mark.skipif(not recipe_available, reason=reason_for_no_recipe)
def test_rejects_unsupported_quantizer_settings() -> None:
    """with_2d_quantization is rejected on both the host-split and graph-safe row-scaled paths."""

    def _quantizer_with_2d() -> NVFP4Quantizer:
        return NVFP4Quantizer(
            fp4_dtype=te.DType.kFloat4E2M1,
            rowwise=True,
            columnwise=False,
            row_scaled_nvfp4=True,
            with_2d_quantization=True,
        )

    split_sections = [128, 128]
    N = 128
    x = torch.randn((sum(split_sections), N), dtype=torch.bfloat16, device="cuda")

    with pytest.raises(RuntimeError):
        tex.split_quantize(x, split_sections, [_quantizer_with_2d() for _ in split_sections])

    tokens_per_expert = torch.tensor(split_sections, dtype=torch.int64, device="cuda")
    tensor_offsets = tex.splits_to_offsets(tokens_per_expert, N)
    with pytest.raises(RuntimeError):
        tex.nvfp4_group_row_scaled_cast_graph_safe(
            x, _quantizer_with_2d(), len(split_sections), tokens_per_expert, tensor_offsets
        )
