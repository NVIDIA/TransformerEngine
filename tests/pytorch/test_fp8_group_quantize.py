# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Tests for grouped FP8 tensor-scaling quantization."""

import pytest
import torch
import transformer_engine.pytorch as te
from transformer_engine.pytorch import Float8CurrentScalingQuantizer
from transformer_engine.pytorch.constants import TE_DType_To_Torch
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


def _case_input(case: str, cols: int):
    if case == "uniform":
        num_tensors = 3
        rows_per_tensor = 256
        actual_rows = num_tensors * rows_per_tensor
        tensor = torch.randn(actual_rows, cols, dtype=torch.bfloat16, device="cuda")
        return tensor, num_tensors, None, [rows_per_tensor] * num_tensors

    first_dims_list = [512, 256, 384]
    actual_rows = sum(first_dims_list)
    allocated_rows = actual_rows * 2
    tensor = torch.randn(allocated_rows, cols, dtype=torch.bfloat16, device="cuda")
    tensor[actual_rows:].fill_(10000.0)
    first_dims = torch.tensor(first_dims_list, dtype=torch.int64, device="cuda")
    return tensor, len(first_dims_list), first_dims, first_dims_list


def _expected_quantized_parts(tensor: torch.Tensor, first_dims_list: list[int], mode: str):
    rowwise_parts = []
    columnwise_parts = []
    scales = []
    scale_invs = []
    amaxes = []
    row_offset = 0
    for rows in first_dims_list:
        group = tensor[row_offset : row_offset + rows]
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
        row_offset += rows

    rowwise = torch.cat(rowwise_parts) if rowwise_parts else None
    columnwise = torch.cat(columnwise_parts) if columnwise_parts else None
    return rowwise, columnwise, torch.cat(scales), torch.cat(scale_invs), torch.cat(amaxes)


@pytest.mark.skipif(not fp8_available, reason=reason_for_no_fp8)
@pytest.mark.parametrize("mode", ["rowwise", "columnwise", "both"])
@pytest.mark.parametrize("case", ["uniform", "overallocated"])
def test_group_quantize_fp8_current_scaling_modes(mode: str, case: str) -> None:
    """Grouped FP8 current scaling matches per-tensor references in every direction mode."""
    cols = 1024
    tensor, num_tensors, first_dims, first_dims_list = _case_input(case, cols)
    quantizer = _make_quantizer(mode)

    grouped = tex.group_quantize(tensor, quantizer, num_tensors, first_dims)
    expected_rowwise, expected_columnwise, expected_scale, expected_scale_inv, expected_amax = (
        _expected_quantized_parts(tensor, first_dims_list, mode)
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
def test_group_quantize_fp8_columnwise_split_uses_columnwise_scale_inv() -> None:
    """Columnwise-only grouped FP8 tensors split into usable Float8Tensor metadata."""
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
