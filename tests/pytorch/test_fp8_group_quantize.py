# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Tests for grouped FP8 tensor-scaling quantization."""

from typing import Optional

import pytest
import torch
import transformer_engine.pytorch as te
import transformer_engine.pytorch.ops as te_ops
from transformer_engine.common import recipe
from transformer_engine.pytorch import Float8CurrentScalingQuantizer, Float8Quantizer
from transformer_engine.pytorch.constants import TE_DType_To_Torch
from transformer_engine.pytorch.ops.basic import grouped_linear as grouped_linear_impl
from transformer_engine.pytorch.tensor import GroupedTensor
from transformer_engine.pytorch.utils import is_non_tn_fp8_gemm_supported
import transformer_engine_torch as tex

from references.quantize_scale_calc import scale_from_amax_tensor
from references.ref_per_tensor_cs import ref_per_tensor_cs_cast


fp8_available, reason_for_no_fp8 = te.is_fp8_available(return_reason=True)


def _make_quantizer(mode: str) -> Float8CurrentScalingQuantizer:
    quantizer = Float8CurrentScalingQuantizer(
        fp8_dtype=tex.DType.kFloat8E4M3,
        device="cuda",
        force_pow_2_scales=False,
        amax_epsilon=0.0,
    )
    quantizer.set_usage(
        rowwise=mode in ("rowwise", "both"),
        columnwise=mode in ("columnwise", "both"),
    )
    return quantizer


def _materialized_mode(mode: str) -> str:
    """Mode actually materialized by auto-created FP8 grouped tensors."""
    if is_non_tn_fp8_gemm_supported() and mode in ("columnwise", "both"):
        return "rowwise"
    return mode


def _case_input(case: str, cols: int):
    if case == "uniform":
        num_tensors = 3
        rows_per_tensor = 256
        actual_rows = num_tensors * rows_per_tensor
        tensor = torch.randn(actual_rows, cols, dtype=torch.bfloat16, device="cuda")
        return tensor, num_tensors, None, [rows_per_tensor] * num_tensors

    if case == "empty_split":
        first_dims_list = [512, 0, 384]
    else:
        first_dims_list = [512, 256, 384]
    actual_rows = sum(first_dims_list)
    allocated_rows = actual_rows * 2
    tensor = torch.randn(allocated_rows, cols, dtype=torch.bfloat16, device="cuda")
    tensor[actual_rows:].fill_(10000.0)
    first_dims = torch.tensor(first_dims_list, dtype=torch.int64, device="cuda")
    return tensor, len(first_dims_list), first_dims, first_dims_list


def _expected_quantized_members(input_tensors: list[torch.Tensor], mode: str):
    rowwise_parts = []
    columnwise_parts = []
    scales = []
    scale_invs = []
    amaxes = []
    for group in input_tensors:
        rows = group.shape[0]
        if rows == 0:
            amax = torch.zeros(1, dtype=torch.float32, device=group.device)
            scale, scale_inv, _ = scale_from_amax_tensor(
                torch.float32,
                amax,
                TE_DType_To_Torch[tex.DType.kFloat8E4M3],
                eps=0.0,
                pow_2_scales=False,
            )
            scales.append(scale)
            scale_invs.append(scale_inv)
            amaxes.append(amax)
            if mode in ("rowwise", "both"):
                rowwise_parts.append(torch.empty(0, dtype=torch.uint8, device=group.device))
            if mode in ("columnwise", "both"):
                columnwise_parts.append(torch.empty(0, dtype=torch.uint8, device=group.device))
            continue
        qx, sx, qx_t, _ = ref_per_tensor_cs_cast(
            group,
            fp8_dtype=tex.DType.kFloat8E4M3,
            return_transpose=mode in ("columnwise", "both"),
            force_pow_2_scales=False,
            amax_epsilon=0.0,
        )
        amax = torch.amax(torch.abs(group.float())).reshape(1)
        scale, _, _ = scale_from_amax_tensor(
            torch.float32,
            amax,
            TE_DType_To_Torch[tex.DType.kFloat8E4M3],
            eps=0.0,
            pow_2_scales=False,
        )
        scales.append(scale)
        scale_invs.append(sx.reshape(1))
        amaxes.append(amax)
        if mode in ("rowwise", "both"):
            rowwise_parts.append(qx.contiguous().view(torch.uint8).reshape(-1))
        if mode in ("columnwise", "both"):
            columnwise_parts.append(qx_t.contiguous().view(torch.uint8).reshape(-1))

    rowwise = torch.cat(rowwise_parts) if rowwise_parts else None
    columnwise = torch.cat(columnwise_parts) if columnwise_parts else None
    return rowwise, columnwise, torch.cat(scales), torch.cat(scale_invs), torch.cat(amaxes)


def _expected_quantized_parts(tensor: torch.Tensor, first_dims_list: list[int], mode: str):
    row_offset = 0
    members = []
    for rows in first_dims_list:
        members.append(tensor[row_offset : row_offset + rows])
        row_offset += rows
    return _expected_quantized_members(members, mode)


def _graph_input_with_members(shape_case: str):
    num_tensors = 3
    if shape_case == "varying-first":
        first_dims_list = [128, 64, 96]
        hidden = 256
        tensor = torch.randn(sum(first_dims_list), hidden, dtype=torch.bfloat16, device="cuda")
        first_dims = torch.tensor(first_dims_list, dtype=torch.int64, device="cuda")
        input_tensors = []
        row_offset = 0
        for rows in first_dims_list:
            input_tensors.append(tensor[row_offset : row_offset + rows])
            row_offset += rows
        return tensor, num_tensors, first_dims, None, input_tensors

    if shape_case == "varying-last":
        rows = 128
        last_dims_list = [256, 128, 384]
        flat_input = torch.empty(
            rows * sum(last_dims_list), dtype=torch.bfloat16, device="cuda"
        )
        input_tensors = []
        offset = 0
        for cols in last_dims_list:
            member = torch.randn(rows, cols, dtype=torch.bfloat16, device="cuda")
            input_tensors.append(member)
            flat_input[offset : offset + member.numel()].copy_(member.reshape(-1))
            offset += member.numel()
        last_dims = torch.tensor(last_dims_list, dtype=torch.int64, device="cuda")
        return (
            flat_input.view(rows, sum(last_dims_list)),
            num_tensors,
            None,
            last_dims,
            input_tensors,
        )

    raise ValueError(f"Unknown graph input case: {shape_case}")


def _graph_input(shape_case: str):
    tensor, num_tensors, first_dims, last_dims, _ = _graph_input_with_members(shape_case)
    return tensor, num_tensors, first_dims, last_dims


def _tensor_offsets(
    first_dims: Optional[torch.Tensor],
    last_dims: Optional[torch.Tensor],
    logical_first_dim: int,
    logical_last_dim: int,
) -> Optional[torch.Tensor]:
    if first_dims is None and last_dims is None:
        return None
    split_dims = first_dims if first_dims is not None else last_dims
    common_dim = logical_last_dim if first_dims is not None else logical_first_dim
    return torch.cat(
        (
            torch.zeros(1, dtype=torch.int64, device=split_dims.device),
            torch.cumsum(split_dims * common_dim, dim=0),
        )
    )


def _assert_grouped_outputs_equal(actual, expected) -> None:
    for attr in (
        "rowwise_data",
        "columnwise_data",
        "scale",
        "scale_inv",
        "columnwise_scale_inv",
        "amax",
    ):
        actual_value = getattr(actual, attr)
        expected_value = getattr(expected, attr)
        if actual_value is None or expected_value is None:
            assert actual_value is None and expected_value is None
        else:
            assert torch.equal(actual_value, expected_value)


@pytest.mark.skipif(not fp8_available, reason=reason_for_no_fp8)
@pytest.mark.parametrize("mode", ["rowwise", "columnwise", "both"])
@pytest.mark.parametrize("case", ["uniform", "overallocated", "empty_split"])
def test_group_quantize_fp8_current_scaling_modes(mode: str, case: str) -> None:
    """Grouped FP8 current scaling matches per-tensor references in every direction mode."""
    cols = 1024
    tensor, num_tensors, first_dims, first_dims_list = _case_input(case, cols)
    quantizer = _make_quantizer(mode)
    materialized_mode = _materialized_mode(mode)

    grouped = tex.group_quantize(tensor, quantizer, num_tensors, first_dims)
    expected_rowwise, expected_columnwise, expected_scale, expected_scale_inv, expected_amax = (
        _expected_quantized_parts(tensor, first_dims_list, materialized_mode)
    )

    if expected_rowwise is not None:
        assert torch.equal(grouped.rowwise_data[: expected_rowwise.numel()], expected_rowwise)
        torch.testing.assert_close(grouped.scale_inv, expected_scale_inv, rtol=0, atol=0)
    else:
        assert grouped.rowwise_data is None

    if expected_columnwise is not None:
        assert torch.equal(
            grouped.columnwise_data[: expected_columnwise.numel()],
            expected_columnwise,
        )
        torch.testing.assert_close(grouped.columnwise_scale_inv, expected_scale_inv, rtol=0, atol=0)
    else:
        assert grouped.columnwise_data is None

    torch.testing.assert_close(grouped.scale, expected_scale, rtol=0, atol=0)
    torch.testing.assert_close(grouped.amax, expected_amax, rtol=0, atol=0)


@pytest.mark.skipif(not fp8_available, reason=reason_for_no_fp8)
@pytest.mark.parametrize("mode", ["rowwise", "columnwise", "both"])
@pytest.mark.parametrize("shape_case", ["varying-first", "varying-last"])
def test_group_quantize_fp8_current_scaling_cuda_graph_varying_dims(
    mode: str, shape_case: str
) -> None:
    """Grouped FP8 current scaling is graph capturable for varying grouped dimensions."""
    tensor, num_tensors, first_dims, last_dims = _graph_input(shape_case)
    static_input = tensor.clone()
    quantizer = _make_quantizer(mode)

    static_output = tex.group_quantize(
        static_input, quantizer, num_tensors, first_dims, last_dims=last_dims
    )
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static_output = tex.group_quantize(
            static_input,
            quantizer,
            num_tensors,
            first_dims,
            output=static_output,
            last_dims=last_dims,
        )

    fresh_input = tensor + torch.randn_like(tensor) * 0.01
    static_input.copy_(fresh_input)
    graph.replay()
    torch.cuda.synchronize()

    expected = tex.group_quantize(
        static_input, quantizer, num_tensors, first_dims, last_dims=last_dims
    )
    _assert_grouped_outputs_equal(static_output, expected)


@pytest.mark.skipif(not fp8_available, reason=reason_for_no_fp8)
@pytest.mark.parametrize("mode", ["rowwise", "columnwise", "both"])
@pytest.mark.parametrize("shape_case", ["varying-first", "varying-last"])
def test_group_quantize_fp8_precomputed_cuda_graph_varying_dims(
    mode: str, shape_case: str
) -> None:
    """Grouped FP8 precomputed-scale quantization is graph capturable with varying dims."""
    tensor, num_tensors, first_dims, last_dims = _graph_input(shape_case)
    static_input = tensor.clone()
    current_quantizer = _make_quantizer(mode)
    prepared = tex.group_quantize(
        static_input, current_quantizer, num_tensors, first_dims, last_dims=last_dims
    )
    delayed_quantizer = Float8Quantizer(
        scale=prepared.scale,
        amax=prepared.amax,
        fp8_dtype=tex.DType.kFloat8E4M3,
        rowwise=mode in ("rowwise", "both"),
        columnwise=mode in ("columnwise", "both"),
    )
    static_output = tex.group_quantize(
        static_input, delayed_quantizer, num_tensors, first_dims, last_dims=last_dims
    )
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static_output = tex.group_quantize(
            static_input,
            delayed_quantizer,
            num_tensors,
            first_dims,
            output=static_output,
            last_dims=last_dims,
        )

    fresh_input = tensor + torch.randn_like(tensor) * 0.01
    static_input.copy_(fresh_input)
    graph.replay()
    torch.cuda.synchronize()

    expected = tex.group_quantize(
        static_input, delayed_quantizer, num_tensors, first_dims, last_dims=last_dims
    )
    _assert_grouped_outputs_equal(static_output, expected)


@pytest.mark.skipif(not fp8_available, reason=reason_for_no_fp8)
@pytest.mark.parametrize("mode", ["columnwise", "both"])
@pytest.mark.parametrize("shape_case", ["varying-first", "varying-last"])
def test_group_quantize_fp8_explicit_columnwise_output_varying_dims(
    mode: str, shape_case: str
) -> None:
    """An explicit columnwise grouped FP8 output is materialized on every architecture."""
    tensor, num_tensors, first_dims, last_dims, input_tensors = _graph_input_with_members(
        shape_case
    )
    current_quantizer = _make_quantizer(mode)
    prepared = tex.group_quantize(
        tensor, current_quantizer, num_tensors, first_dims, last_dims=last_dims
    )
    delayed_quantizer = Float8Quantizer(
        scale=prepared.scale,
        amax=prepared.amax,
        fp8_dtype=tex.DType.kFloat8E4M3,
        rowwise=mode == "both",
        columnwise=True,
    )
    rowwise_data = (
        torch.empty(tensor.numel(), dtype=torch.uint8, device="cuda")
        if mode == "both"
        else None
    )
    columnwise_data = torch.empty(tensor.numel(), dtype=torch.uint8, device="cuda")
    scale_inv = (
        prepared.scale_inv
        if prepared.scale_inv is not None
        else prepared.columnwise_scale_inv
    )
    tensor_offsets = _tensor_offsets(
        first_dims, last_dims, tensor.shape[0], tensor.shape[1]
    )
    output = GroupedTensor(
        tuple(tensor.shape),
        TE_DType_To_Torch[tex.DType.kFloat8E4M3],
        num_tensors=num_tensors,
        shapes=None,
        quantizer=delayed_quantizer,
        data=rowwise_data,
        columnwise_data=columnwise_data,
        scale_inv=scale_inv if rowwise_data is not None else None,
        columnwise_scale_inv=scale_inv,
        amax=prepared.amax,
        scale=prepared.scale,
        first_dims=first_dims,
        last_dims=last_dims,
        tensor_offsets=tensor_offsets,
    )

    grouped = tex.group_quantize(
        tensor,
        delayed_quantizer,
        num_tensors,
        first_dims,
        output=output,
        last_dims=last_dims,
    )
    expected_rowwise, expected_columnwise, _, _, _ = _expected_quantized_members(
        input_tensors, mode
    )

    if expected_rowwise is not None:
        assert torch.equal(grouped.rowwise_data[: expected_rowwise.numel()], expected_rowwise)
    else:
        assert grouped.rowwise_data is None
    assert grouped.columnwise_data is not None
    assert torch.equal(
        grouped.columnwise_data[: expected_columnwise.numel()],
        expected_columnwise,
    )


@pytest.mark.skipif(not fp8_available, reason=reason_for_no_fp8)
@pytest.mark.parametrize("mode", ["rowwise", "columnwise", "both"])
def test_group_quantize_fp8_current_scaling_varying_last_dim(mode: str) -> None:
    """Grouped FP8 current scaling supports packed groups with varying last dimensions."""
    num_tensors = 3
    rows = 256
    last_dims_list = [512, 1024, 256]
    total_cols = sum(last_dims_list)
    flat_input = torch.empty(rows * total_cols, dtype=torch.bfloat16, device="cuda")
    input_tensors = []
    offset = 0
    for cols in last_dims_list:
        member = torch.randn(rows, cols, dtype=torch.bfloat16, device="cuda")
        input_tensors.append(member)
        flat_input[offset : offset + member.numel()].copy_(member.reshape(-1))
        offset += member.numel()
    tensor = flat_input.view(rows, total_cols)
    last_dims = torch.tensor(last_dims_list, dtype=torch.int64, device="cuda")
    quantizer = _make_quantizer(mode)
    materialized_mode = _materialized_mode(mode)

    grouped = tex.group_quantize(tensor, quantizer, num_tensors, None, last_dims=last_dims)
    expected_rowwise, expected_columnwise, expected_scale, expected_scale_inv, expected_amax = (
        _expected_quantized_members(input_tensors, materialized_mode)
    )

    assert grouped.first_dims is None
    assert torch.equal(grouped.last_dims, last_dims)
    if expected_rowwise is not None:
        assert torch.equal(grouped.rowwise_data[: expected_rowwise.numel()], expected_rowwise)
        torch.testing.assert_close(grouped.scale_inv, expected_scale_inv, rtol=0, atol=0)
    else:
        assert grouped.rowwise_data is None
    if expected_columnwise is not None:
        assert torch.equal(
            grouped.columnwise_data[: expected_columnwise.numel()],
            expected_columnwise,
        )
        torch.testing.assert_close(grouped.columnwise_scale_inv, expected_scale_inv, rtol=0, atol=0)
    else:
        assert grouped.columnwise_data is None
    torch.testing.assert_close(grouped.scale, expected_scale, rtol=0, atol=0)
    torch.testing.assert_close(grouped.amax, expected_amax, rtol=0, atol=0)


@pytest.mark.skipif(not fp8_available, reason=reason_for_no_fp8)
def test_group_quantize_fp8_columnwise_split_uses_columnwise_scale_inv() -> None:
    """Columnwise-only grouped FP8 tensors split into usable Float8Tensor metadata."""
    if is_non_tn_fp8_gemm_supported():
        pytest.skip("Auto-created grouped FP8 tensors do not materialize columnwise data.")
    cols = 1024
    tensor, num_tensors, first_dims, first_dims_list = _case_input("overallocated", cols)
    quantizer = _make_quantizer("columnwise")

    grouped = tex.group_quantize(tensor, quantizer, num_tensors, first_dims)
    tensors = grouped.split_into_quantized_tensors()

    row_offset = 0
    for idx, (part, rows) in enumerate(zip(tensors, first_dims_list)):
        assert part._data is None
        assert part._transpose is not None
        assert part._scale_inv is not None
        assert part._transpose.shape == (cols, rows)
        torch.testing.assert_close(
            part._scale_inv,
            grouped.columnwise_scale_inv[idx : idx + 1],
            rtol=0,
            atol=0,
        )
        expected = tensor[row_offset : row_offset + rows]
        fp8_dtype = TE_DType_To_Torch[tex.DType.kFloat8E4M3]
        dequantized_from_columnwise = (
            part._transpose.view(fp8_dtype).float().t().contiguous() * part._scale_inv
        )
        torch.testing.assert_close(
            dequantized_from_columnwise,
            expected.float(),
            atol=0.125,
            rtol=0.1,
        )
        row_offset += rows


@pytest.mark.skipif(not fp8_available, reason=reason_for_no_fp8)
def test_grouped_linear_fp8_current_scaling_bias_backward_empty_split(monkeypatch) -> None:
    """FP8 current-scaling grouped-linear bias backward avoids MXFP8-only bgrad fusion."""
    if torch.cuda.get_device_capability() < (10, 0):
        pytest.skip("GroupedTensor grouped GEMM path requires SM100+.")

    def _unexpected_bgrad_group_quantize(*_args, **_kwargs):
        raise AssertionError("FP8 current scaling must not call bgrad_group_quantize")

    monkeypatch.setattr(
        grouped_linear_impl.tex,
        "bgrad_group_quantize",
        _unexpected_bgrad_group_quantize,
    )
    monkeypatch.setattr(
        te_ops.GroupedLinear,
        "_is_graph_safe_path_supported",
        staticmethod(lambda **_kwargs: True),
    )

    num_groups = 3
    in_features = 128
    out_features = 128
    split_sizes = torch.tensor([64, 0, 64], dtype=torch.int64, device="cuda")
    input_ = torch.randn(
        int(split_sizes.sum().item()),
        in_features,
        dtype=torch.bfloat16,
        device="cuda",
        requires_grad=True,
    )
    grad_output = torch.randn(
        int(split_sizes.sum().item()),
        out_features,
        dtype=torch.bfloat16,
        device="cuda",
    )
    op = te_ops.GroupedLinear(
        num_groups,
        in_features,
        out_features,
        bias=True,
        device="cuda",
        dtype=torch.bfloat16,
    )

    with te.autocast(enabled=True, recipe=recipe.Float8CurrentScaling()):
        output = op(input_, split_sizes)
    output.backward(grad_output)

    assert input_.grad is not None
    split_offsets = torch.cat(
        (torch.zeros(1, dtype=split_sizes.dtype, device=split_sizes.device), split_sizes.cumsum(0))
    )
    for idx, rows in enumerate(split_sizes.tolist()):
        weight = getattr(op, f"weight{idx}")
        bias = getattr(op, f"bias{idx}")
        assert weight.grad is not None
        assert bias.grad is not None
        start = int(split_offsets[idx].item())
        end = int(split_offsets[idx + 1].item())
        group_grad = grad_output[start:end]
        expected_dbias = group_grad.float().sum(dim=0).to(dtype=bias.grad.dtype)
        torch.testing.assert_close(bias.grad, expected_dbias, rtol=1e-2, atol=1e-2)
        if rows == 0:
            assert torch.count_nonzero(bias.grad) == 0
