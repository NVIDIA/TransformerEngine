# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Tests for GroupedTensor class"""

from typing import List, Tuple
import pytest
import torch
import transformer_engine.pytorch as te
from transformer_engine.pytorch.tensor.storage.grouped_tensor import GroupedTensor
from transformer_engine.pytorch import (
    Float8Quantizer,
    Float8CurrentScalingQuantizer,
    Float8BlockQuantizer,
    MXFP8Quantizer,
    NVFP4Quantizer,
)
import transformer_engine_torch as tex

# Check available recipes
fp8_available, reason_for_no_fp8 = te.is_fp8_available(return_reason=True)
fp8_block_scaling_available, reason_for_no_fp8_block_scaling = te.is_fp8_block_scaling_available(
    return_reason=True
)
mxfp8_available, reason_for_no_mxfp8 = te.is_mxfp8_available(return_reason=True)
nvfp4_available, reason_for_no_nvfp4 = te.is_nvfp4_available(return_reason=True)

_quantization_params = [
    pytest.param(
        "fp8_delayed_scaling",
        marks=pytest.mark.skipif(not fp8_available, reason=reason_for_no_fp8),
    ),
    pytest.param(
        "fp8_current_scaling",
        marks=pytest.mark.skipif(not fp8_available, reason=reason_for_no_fp8),
    ),
    pytest.param(
        "fp8_blockwise",
        marks=pytest.mark.skipif(
            not fp8_block_scaling_available, reason=reason_for_no_fp8_block_scaling
        ),
    ),
    pytest.param(
        "mxfp8",
        marks=pytest.mark.skipif(not mxfp8_available, reason=reason_for_no_mxfp8),
    ),
    pytest.param(
        "nvfp4",
        marks=pytest.mark.skipif(not nvfp4_available, reason=reason_for_no_nvfp4),
    ),
]


def make_quantizers(quantization: str, num_tensors: int, shape: List[Tuple[int, int]]):
    """Create quantizers for given quantization scheme"""
    quantizers = []
    for i in range(num_tensors):
        if quantization == "fp8_delayed_scaling":
            quantizer = Float8Quantizer(
                scale=torch.ones(1, dtype=torch.float32, device="cuda"),
                amax=torch.zeros(1, dtype=torch.float32, device="cuda"),
                fp8_dtype=tex.DType.kFloat8E4M3,
            )
        elif quantization == "fp8_current_scaling":
            quantizer = Float8CurrentScalingQuantizer(
                fp8_dtype=tex.DType.kFloat8E4M3,
                device="cuda",
            )
            quantizer.set_usage(rowwise=True, columnwise=False)
        elif quantization == "fp8_blockwise":
            quantizer = Float8BlockQuantizer(
                fp8_dtype=tex.DType.kFloat8E4M3,
                rowwise=True,
                columnwise=False,
                force_pow_2_scales=True,
                amax_epsilon=0.0,
                block_scaling_dim=1,
            )
        elif quantization == "mxfp8":
            quantizer = MXFP8Quantizer(fp8_dtype=tex.DType.kFloat8E4M3)
        elif quantization == "nvfp4":
            quantizer = NVFP4Quantizer(
                with_rht=False,
                with_post_rht_amax=False,
                with_2d_quantization=False,
                stochastic_rounding=False,
                with_random_sign_mask=False,
            )
        else:
            raise ValueError(f"Unknown quantization scheme: {quantization}")

        quantizer.internal = False
        quantizers.append(quantizer)

    return quantizers


def _get_rowwise_data_tensor(qtensor, quantization: str) -> torch.Tensor:
    if quantization in ("fp8_delayed_scaling", "fp8_current_scaling"):
        return qtensor._data
    if quantization in ("fp8_blockwise", "mxfp8", "nvfp4"):
        return qtensor._rowwise_data
    raise ValueError(f"Unknown quantization scheme: {quantization}")


def _rowwise_offset_bytes(numel: int, quantization: str) -> int:
    if quantization == "nvfp4":
        return numel // 2
    return numel


class TestGroupedTensor:
    @staticmethod
    def setup_class(cls) -> None:
        # Configure RNG
        seed = 1234
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

    def test_basic_construction_all_same_shape(self) -> None:
        """Test GroupedTensor construction with all tensors having same shape"""
        num_tensors = 4
        shape = [(256, 512) for _ in range(num_tensors)]

        grouped_tensor = GroupedTensor.make_grouped_tensor(
            num_tensors=num_tensors,
            shape=shape,
            quantizers=None,
            device="cuda",
            dtype=torch.float32,
        )

        assert grouped_tensor.num_tensors == num_tensors
        assert grouped_tensor.all_same_shape()
        assert grouped_tensor.all_same_first_dim()
        assert grouped_tensor.all_same_last_dim()
        assert grouped_tensor.logical_shape == (num_tensors * 256, 512)
        assert grouped_tensor.get_common_first_dim() == 256
        assert grouped_tensor.get_common_last_dim() == 512
        assert grouped_tensor.has_data()

    def test_basic_construction_varying_first_dim(self) -> None:
        """Test GroupedTensor construction with varying first dimension"""
        num_tensors = 3
        shape = [(128, 512), (256, 512), (384, 512)]

        grouped_tensor = GroupedTensor.make_grouped_tensor(
            num_tensors=num_tensors,
            shape=shape,
            quantizers=None,
            device="cuda",
            dtype=torch.float32,
        )

        assert grouped_tensor.num_tensors == num_tensors
        assert not grouped_tensor.all_same_shape()
        assert not grouped_tensor.all_same_first_dim()
        assert grouped_tensor.all_same_last_dim()
        assert grouped_tensor.get_common_last_dim() == shape[0][1]
        assert grouped_tensor.logical_shape == (
            sum(v for v, _ in shape),
            shape[0][1],
        )  # sum of first dims

    def test_basic_construction_varying_last_dim(self) -> None:
        """Test GroupedTensor construction with varying last dimension"""
        num_tensors = 3
        shape = [(512, 128), (512, 256), (512, 384)]

        grouped_tensor = GroupedTensor.make_grouped_tensor(
            num_tensors=num_tensors,
            shape=shape,
            quantizers=None,
            device="cuda",
            dtype=torch.float32,
        )

        assert grouped_tensor.num_tensors == num_tensors
        assert not grouped_tensor.all_same_shape()
        assert grouped_tensor.all_same_first_dim()
        assert not grouped_tensor.all_same_last_dim()
        assert grouped_tensor.get_common_first_dim() == shape[0][0]
        assert grouped_tensor.logical_shape == (
            shape[0][0],
            sum(v for _, v in shape),
        )  # sum of last dims

    def test_basic_construction_varying_both_dims(self) -> None:
        """Test GroupedTensor construction with varying both dimensions"""
        num_tensors = 3
        shape = [(128, 256), (256, 384), (384, 512)]

        grouped_tensor = GroupedTensor.make_grouped_tensor(
            num_tensors=num_tensors,
            shape=shape,
            quantizers=None,
            device="cuda",
            dtype=torch.float32,
        )

        assert grouped_tensor.num_tensors == num_tensors
        assert not grouped_tensor.all_same_shape()
        assert not grouped_tensor.all_same_first_dim()
        assert not grouped_tensor.all_same_last_dim()
        assert grouped_tensor.varying_both_dims()
        total_elements = sum(s[0] * s[1] for s in shape)
        assert grouped_tensor.logical_shape == (1, total_elements)

    def test_split_into_quantized_tensors_no_quantization(self) -> None:
        """Test split_into_quantized_tensors for unquantized tensors"""
        num_tensors = 3
        shape = [(256, 512) for _ in range(num_tensors)]

        grouped_tensor = GroupedTensor.make_grouped_tensor(
            num_tensors=num_tensors,
            shape=shape,
            quantizers=None,
            device="cuda",
            dtype=torch.float32,
        )

        # Get the original data pointer
        original_data_ptr = grouped_tensor.data.data_ptr()

        # Split into tensors
        tensors = grouped_tensor.split_into_quantized_tensors()

        assert len(tensors) == num_tensors

        # Verify each tensor has correct shape and shares storage
        for i, tensor in enumerate(tensors):
            assert tensor.shape == shape[i]
            assert isinstance(tensor, torch.Tensor)
            assert not hasattr(tensor, "_data")  # Not a quantized tensor

            # Verify data pointer is within the original grouped tensor storage
            # The tensor should be a view of the original data
            assert tensor.data_ptr() >= original_data_ptr

            # Calculate expected offset
            expected_offset = i * (shape[i][0] * shape[i][1]) * tensor.element_size()
            assert tensor.data_ptr() == original_data_ptr + expected_offset

    @pytest.mark.parametrize("quantization", _quantization_params)
    def test_split_into_quantized_tensors_quantized(self, quantization: str) -> None:
        """Test split_into_quantized_tensors for quantized tensors"""
        num_tensors = 3
        shape = [(512, 512) for _ in range(num_tensors)]
        quantizers = make_quantizers(quantization, num_tensors, shape)

        grouped_tensor = GroupedTensor.make_grouped_tensor(
            num_tensors=num_tensors,
            shape=shape,
            quantizers=quantizers,
            device="cuda",
        )

        # Get the original data pointer
        original_data_ptr = grouped_tensor.data.data_ptr()

        # Split into tensors
        tensors = grouped_tensor.split_into_quantized_tensors()

        assert len(tensors) == num_tensors

        # Verify each tensor shares storage with the grouped tensor
        for i, tensor in enumerate(tensors):
            rowwise_data = _get_rowwise_data_tensor(tensor, quantization)
            assert rowwise_data is not None
            assert rowwise_data.data_ptr() >= original_data_ptr
            numel = shape[i][0] * shape[i][1]
            expected_offset = _rowwise_offset_bytes(i * numel, quantization)
            assert rowwise_data.data_ptr() == original_data_ptr + expected_offset

    def test_split_varying_shapes(self) -> None:
        """Test split_into_quantized_tensors with varying shapes"""
        num_tensors = 3
        shape = [(128, 512), (256, 512), (384, 512)]

        grouped_tensor = GroupedTensor.make_grouped_tensor(
            num_tensors=num_tensors,
            shape=shape,
            quantizers=None,
            device="cuda",
            dtype=torch.float32,
        )

        original_data_ptr = grouped_tensor.data.data_ptr()
        tensors = grouped_tensor.split_into_quantized_tensors()

        assert len(tensors) == num_tensors

        # Verify shapes and storage
        cumulative_offset = 0
        for i, tensor in enumerate(tensors):
            assert tensor.shape == shape[i]
            expected_offset = cumulative_offset * tensor.element_size()
            assert tensor.data_ptr() == original_data_ptr + expected_offset
            cumulative_offset += shape[i][0] * shape[i][1]

    @pytest.mark.parametrize("quantization", _quantization_params)
    def test_quantize_inplace(self, quantization: str) -> None:
        """Test that quantize is done in-place for all recipes"""
        num_tensors = 3
        shape = [(512, 512) for _ in range(num_tensors)]
        quantizers = make_quantizers(quantization, num_tensors, shape)

        grouped_tensor = GroupedTensor.make_grouped_tensor(
            num_tensors=num_tensors,
            shape=shape,
            quantizers=quantizers,
            device="cuda",
        )

        # Get original data pointers before quantization
        original_data_ptr = grouped_tensor.data.data_ptr()
        original_scale_inv_ptr = grouped_tensor.scale_inv.data_ptr()
        original_scale_ptr = (
            grouped_tensor.scale.data_ptr() if grouped_tensor.scale is not None else None
        )

        # Create input tensors
        input_tensors = [torch.randn(s, dtype=torch.float32, device="cuda") for s in shape]

        # Quantize in place
        quantized_tensors = grouped_tensor.quantize(input_tensors)

        # Verify data pointers haven't changed (in-place operation)
        assert grouped_tensor.data.data_ptr() == original_data_ptr
        assert grouped_tensor.scale_inv.data_ptr() == original_scale_inv_ptr
        if original_scale_ptr is not None:
            assert grouped_tensor.scale.data_ptr() == original_scale_ptr

        # Verify returned tensors point to the same storage
        for i, qtensor in enumerate(quantized_tensors):
            rowwise_data = _get_rowwise_data_tensor(qtensor, quantization)
            numel = shape[i][0] * shape[i][1]
            expected_offset = _rowwise_offset_bytes(i * numel, quantization)
            assert rowwise_data.data_ptr() == original_data_ptr + expected_offset

    @pytest.mark.parametrize("quantization", _quantization_params)
    def test_quantize_varying_shapes(self, quantization: str) -> None:
        """Test quantize with varying shapes"""
        num_tensors = 3
        shape = [(256, 512), (512, 512), (768, 512)]
        quantizers = make_quantizers(quantization, num_tensors, shape)

        grouped_tensor = GroupedTensor.make_grouped_tensor(
            num_tensors=num_tensors,
            shape=shape,
            quantizers=quantizers,
            device="cuda",
        )

        # Get original data pointers
        original_data_ptr = grouped_tensor.data.data_ptr()

        # Create input tensors with varying shapes
        input_tensors = [torch.randn(s, dtype=torch.float32, device="cuda") for s in shape]

        # Quantize in place
        quantized_tensors = grouped_tensor.quantize(input_tensors)

        # Verify data pointer hasn't changed
        assert grouped_tensor.data.data_ptr() == original_data_ptr

        # Verify each tensor points to correct location
        cumulative_numel = 0
        for qtensor, tensor_shape in zip(quantized_tensors, shape):
            rowwise_data = _get_rowwise_data_tensor(qtensor, quantization)
            expected_offset = _rowwise_offset_bytes(cumulative_numel, quantization)
            assert rowwise_data.data_ptr() == original_data_ptr + expected_offset
            cumulative_numel += tensor_shape[0] * tensor_shape[1]

    @pytest.mark.parametrize("quantization", _quantization_params)
    def test_static_quantize_method(self, quantization: str) -> None:
        """Test the static quantize method"""
        num_tensors = 3
        shape = [(512, 512) for _ in range(num_tensors)]
        quantizers = make_quantizers(quantization, num_tensors, shape)

        # Create input tensors
        input_tensors = [torch.randn(s, dtype=torch.float32, device="cuda") for s in shape]

        # Use static quantize method
        grouped_tensor = GroupedTensor.create_and_quantize(
            tensors=input_tensors,
            quantizers=quantizers,
            device="cuda",
        )

        # Verify the grouped tensor was created correctly
        assert grouped_tensor.num_tensors == num_tensors
        assert grouped_tensor.has_data()

        # Verify quantized_tensors were created and point to same storage
        assert grouped_tensor.quantized_tensors is not None
        assert len(grouped_tensor.quantized_tensors) == num_tensors

        original_data_ptr = grouped_tensor.data.data_ptr()
        for i, qtensor in enumerate(grouped_tensor.quantized_tensors):
            rowwise_data = _get_rowwise_data_tensor(qtensor, quantization)
            numel = shape[i][0] * shape[i][1]
            expected_offset = _rowwise_offset_bytes(i * numel, quantization)
            assert rowwise_data.data_ptr() == original_data_ptr + expected_offset

    def test_clear(self) -> None:
        """Test clear method"""
        num_tensors = 3
        shape = [(256, 512) for _ in range(num_tensors)]

        grouped_tensor = GroupedTensor.make_grouped_tensor(
            num_tensors=num_tensors,
            shape=shape,
            quantizers=None,
            device="cuda",
            dtype=torch.float32,
        )

        assert grouped_tensor.has_data()
        assert grouped_tensor.num_tensors == num_tensors

        grouped_tensor.clear()

        assert not grouped_tensor.has_data()
        assert grouped_tensor.num_tensors == 0
        assert grouped_tensor.data is None
        assert grouped_tensor.logical_shape == (0, 0)
