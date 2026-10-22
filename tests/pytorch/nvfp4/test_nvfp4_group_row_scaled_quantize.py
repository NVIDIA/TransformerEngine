# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Grouped row-scaled NVFP4 quantization through tex.group_quantize."""

from typing import Optional

import pytest
import torch

import transformer_engine.pytorch as te
import transformer_engine_torch as tex
from transformer_engine.pytorch import NVFP4Quantizer

recipe_available, reason_for_no_recipe = te.is_nvfp4_available(return_reason=True)


def _row_scaled_quantizer(**kwargs) -> NVFP4Quantizer:
    return NVFP4Quantizer(
        fp4_dtype=te.DType.kFloat4E2M1,
        rowwise=True,
        columnwise=False,
        with_rht=False,
        with_post_rht_amax=False,
        row_scaled_nvfp4=True,
        **kwargs,
    )


def _tensor_scaled_quantizer() -> NVFP4Quantizer:
    return NVFP4Quantizer(
        fp4_dtype=te.DType.kFloat4E2M1,
        rowwise=True,
        columnwise=False,
        with_rht=False,
        with_post_rht_amax=False,
        row_scaled_nvfp4=False,
    )


def _group_quantize(x: torch.Tensor, num_tensors: int, first_dims: Optional[torch.Tensor]):
    return tex.group_quantize(x, _row_scaled_quantizer(), num_tensors, first_dims)


def _per_expert_reference(x_valid: torch.Tensor, split_sections: list[int]):
    N = x_valid.shape[1]
    data, scale, amax = [], [], []
    offset = 0
    for m in split_sections:
        if m == 0:
            continue
        ref = _row_scaled_quantizer()(x_valid[offset : offset + m])
        data.append(ref._rowwise_data.view(torch.uint8).reshape(m, N // 2))
        scale.append(ref._rowwise_scale_inv.view(torch.uint8).reshape(m, -1)[:, : N // 16])
        amax.append(ref._amax_rowwise.reshape(m))
        offset += m
    return torch.cat(data, dim=0), torch.cat(scale, dim=0), torch.cat(amax, dim=0)


def _assert_matches_reference(grouped, x_valid: torch.Tensor, split_sections: list[int]) -> None:
    N = x_valid.shape[1]
    sum_m = sum(split_sections)
    ref_data, ref_scale, ref_amax = _per_expert_reference(x_valid, split_sections)

    capacity = int(grouped.logical_shape[0])
    scale_stride = grouped.scale_inv.numel() // capacity
    g_data = grouped.rowwise_data.view(torch.uint8).reshape(capacity, N // 2)[:sum_m]
    g_scale = grouped.scale_inv.view(torch.uint8).reshape(capacity, scale_stride)[:sum_m]
    g_amax = grouped.amax.reshape(-1)[:sum_m]

    torch.testing.assert_close(g_amax, ref_amax, atol=0, rtol=0)
    torch.testing.assert_close(g_data, ref_data, atol=0, rtol=0)
    torch.testing.assert_close(g_scale[:, : N // 16], ref_scale, atol=0, rtol=0)


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
def test_matches_per_expert_quantize(split_sections: list[int], N: int) -> None:
    torch.manual_seed(0)
    x = torch.randn((sum(split_sections), N), dtype=torch.bfloat16, device="cuda")
    first_dims = torch.tensor(split_sections, dtype=torch.int64, device="cuda")

    grouped = _group_quantize(x, len(split_sections), first_dims)

    assert grouped.row_scaled_nvfp4
    _assert_matches_reference(grouped, x, split_sections)


@pytest.mark.skipif(not recipe_available, reason=reason_for_no_recipe)
@pytest.mark.parametrize("num_tensors, m, N", [(1, 128, 128), (4, 256, 256)])
def test_uniform_shapes(num_tensors: int, m: int, N: int) -> None:
    torch.manual_seed(1)
    x = torch.randn((num_tensors * m, N), dtype=torch.bfloat16, device="cuda")

    grouped = _group_quantize(x, num_tensors, None)

    _assert_matches_reference(grouped, x, [m] * num_tensors)


@pytest.mark.skipif(not recipe_available, reason=reason_for_no_recipe)
@pytest.mark.parametrize(
    "split_sections, N, capacity",
    [
        ([128, 128], 256, 512),
        ([256, 128, 128], 128, 768),
    ],
)
def test_paged_stashing_capacity_noop(split_sections: list[int], N: int, capacity: int) -> None:
    sum_m = sum(split_sections)
    assert capacity > sum_m
    torch.manual_seed(2)
    x = torch.randn((capacity, N), dtype=torch.bfloat16, device="cuda")
    first_dims = torch.tensor(split_sections, dtype=torch.int64, device="cuda")

    grouped = _group_quantize(x, len(split_sections), first_dims)

    _assert_matches_reference(grouped, x[:sum_m].clone(), split_sections)


@pytest.mark.skipif(not recipe_available, reason=reason_for_no_recipe)
def test_capture_once_replay_changed_routing() -> None:
    N = 256
    num_experts = 3
    capacity = 768
    splits_a = [256, 128, 128]
    splits_b = [128, 256, 128]

    first_dims = torch.empty(num_experts, dtype=torch.int64, device="cuda")
    x = torch.empty((capacity, N), dtype=torch.bfloat16, device="cuda")

    torch.manual_seed(3)
    x_a = torch.randn((sum(splits_a), N), dtype=torch.bfloat16, device="cuda")
    x_b = torch.randn((sum(splits_b), N), dtype=torch.bfloat16, device="cuda")

    def program_routing(splits: list[int], x_src: torch.Tensor) -> None:
        first_dims.copy_(torch.tensor(splits, dtype=torch.int64, device="cuda"))
        x.zero_()
        x[: x_src.shape[0], :].copy_(x_src)

    # Warm up outside capture so one-time setup does not land in the captured graph.
    program_routing(splits_a, x_a)
    warmup_stream = torch.cuda.Stream()
    warmup_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup_stream):
        _ = _group_quantize(x, num_experts, first_dims)
    torch.cuda.current_stream().wait_stream(warmup_stream)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        grouped = _group_quantize(x, num_experts, first_dims)

    program_routing(splits_a, x_a)
    graph.replay()
    torch.cuda.synchronize()
    _assert_matches_reference(grouped, x_a, splits_a)

    program_routing(splits_b, x_b)
    graph.replay()
    torch.cuda.synchronize()
    _assert_matches_reference(grouped, x_b, splits_b)


@pytest.mark.skipif(not recipe_available, reason=reason_for_no_recipe)
def test_non_contiguous_input_matches_contiguous() -> None:
    split_sections = [128, 128]
    N = 256
    torch.manual_seed(4)
    x_t = torch.randn((N, sum(split_sections)), dtype=torch.bfloat16, device="cuda").t()
    assert not x_t.is_contiguous()
    first_dims = torch.tensor(split_sections, dtype=torch.int64, device="cuda")

    grouped = _group_quantize(x_t, len(split_sections), first_dims)

    _assert_matches_reference(grouped, x_t.contiguous(), split_sections)


@pytest.mark.skipif(not recipe_available, reason=reason_for_no_recipe)
def test_rejects_unaligned_uniform_shapes() -> None:
    x = torch.randn((128, 128), dtype=torch.bfloat16, device="cuda")

    with pytest.raises(RuntimeError):
        _group_quantize(x, 2, None)


@pytest.mark.skipif(not recipe_available, reason=reason_for_no_recipe)
def test_rejects_strided_first_dims() -> None:
    x = torch.randn((384, 128), dtype=torch.bfloat16, device="cuda")
    strided = torch.tensor([256, 0, 128, 0], dtype=torch.int64, device="cuda")[::2]
    assert not strided.is_contiguous()

    with pytest.raises(RuntimeError):
        _group_quantize(x, 2, strided)


@pytest.mark.skipif(not recipe_available, reason=reason_for_no_recipe)
def test_rejects_unsupported_quantizer_settings() -> None:
    x = torch.randn((256, 128), dtype=torch.bfloat16, device="cuda")
    first_dims = torch.tensor([128, 128], dtype=torch.int64, device="cuda")

    with pytest.raises(RuntimeError):
        tex.group_quantize(x, _row_scaled_quantizer(with_2d_quantization=True), 2, first_dims)


@pytest.mark.skipif(not recipe_available, reason=reason_for_no_recipe)
@pytest.mark.parametrize(
    "input_shape, split_sections",
    [
        ((384, 128), [256, 128]),
        ((256, 64), [128, 128]),
        ((256, 2, 128), [128, 128]),
        ((512, 128), [128, 128]),
    ],
)
def test_split_quantize_matches_per_expert(input_shape, split_sections: list[int]) -> None:
    torch.manual_seed(5)
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


@pytest.mark.skipif(not recipe_available, reason=reason_for_no_recipe)
def test_split_quantize_mixed_row_scaled_falls_back() -> None:
    split_sections = [128, 128]
    N = 128
    torch.manual_seed(6)
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
    split_sections = [0, 0]
    x = torch.empty((0, 128), dtype=torch.bfloat16, device="cuda")

    outputs = tex.split_quantize(
        x, split_sections, [_row_scaled_quantizer() for _ in split_sections]
    )

    assert len(outputs) == len(split_sections)
    for out in outputs:
        assert tuple(out.shape) == (0, 128)
