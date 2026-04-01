# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Tests for GroupedTensor class"""

from typing import List, Tuple
import pytest
import torch
import transformer_engine.pytorch as te
from transformer_engine.pytorch.tensor.grouped_tensor import GroupedTensor
from transformer_engine.pytorch import (
    Quantizer,
    Float8Quantizer,
    Float8CurrentScalingQuantizer,
    Float8BlockQuantizer,
    MXFP8Quantizer,
    NVFP4Quantizer,
)
from transformer_engine.pytorch.constants import TE_DType_To_Torch
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


def make_quantizer(quantization: str, num_tensors: int, shape: List[Tuple[int, int]]) -> Quantizer:
    """Create quantizer for given quantization scheme"""

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

    return quantizer


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

        grouped_tensor = GroupedTensor.make_grouped_tensor_with_shapes(
            num_tensors=num_tensors,
            shapes=shape,
            quantizer=None,
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

        grouped_tensor = GroupedTensor.make_grouped_tensor_with_shapes(
            num_tensors=num_tensors,
            shapes=shape,
            quantizer=None,
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

    def test_split_into_quantized_tensors_no_quantization(self) -> None:
        """Test split_into_quantized_tensors for unquantized tensors"""
        num_tensors = 3
        shape = [(256, 512) for _ in range(num_tensors)]

        grouped_tensor = GroupedTensor.make_grouped_tensor_with_shapes(
            num_tensors=num_tensors,
            shapes=shape,
            quantizer=None,
            device="cuda",
            dtype=torch.float32,
        )

        # GroupedTensor is a wrapper; use backing storage buffer pointer.
        storage = grouped_tensor.rowwise_data
        if storage is None:
            storage = grouped_tensor.columnwise_data
        assert storage is not None
        original_data_ptr = storage.data_ptr()

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
        quantizer = make_quantizer(quantization, num_tensors, shape)

        grouped_tensor = GroupedTensor.make_grouped_tensor_with_shapes(
            num_tensors=num_tensors,
            shapes=shape,
            quantizer=quantizer,
            device="cuda",
            dtype=torch.float32,
        )

        # GroupedTensor is a wrapper; use backing storage buffer pointer.
        storage = grouped_tensor.rowwise_data
        if storage is None:
            storage = grouped_tensor.columnwise_data
        assert storage is not None
        original_data_ptr = storage.data_ptr()

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

        grouped_tensor = GroupedTensor.make_grouped_tensor_with_shapes(
            num_tensors=num_tensors,
            shapes=shape,
            quantizer=None,
            device="cuda",
            dtype=torch.float32,
        )

        storage = grouped_tensor.rowwise_data
        if storage is None:
            storage = grouped_tensor.columnwise_data
        assert storage is not None
        original_data_ptr = storage.data_ptr()
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
        quantizer = make_quantizer(quantization, num_tensors, shape)

        grouped_tensor = GroupedTensor.make_grouped_tensor_with_shapes(
            num_tensors=num_tensors,
            shapes=shape,
            quantizer=quantizer,
            device="cuda",
            dtype=torch.float32,
        )

        # Get original data pointers before quantization
        storage = grouped_tensor.rowwise_data
        if storage is None:
            storage = grouped_tensor.columnwise_data
        assert storage is not None
        original_data_ptr = storage.data_ptr()
        original_scale_inv_ptr = grouped_tensor.scale_inv.data_ptr()
        original_scale_ptr = (
            grouped_tensor.scale.data_ptr() if grouped_tensor.scale is not None else None
        )

        # Create input tensors
        input_tensors = [torch.randn(s, dtype=torch.float32, device="cuda") for s in shape]

        # Quantize in place
        quantized_tensors = grouped_tensor.quantize(input_tensors)

        # Verify data pointers haven't changed (in-place operation)
        assert storage.data_ptr() == original_data_ptr
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
        quantizer = make_quantizer(quantization, num_tensors, shape)

        grouped_tensor = GroupedTensor.make_grouped_tensor_with_shapes(
            num_tensors=num_tensors,
            shapes=shape,
            quantizer=quantizer,
            device="cuda",
            dtype=torch.float32,
        )

        # Get original data pointers
        storage = grouped_tensor.rowwise_data
        if storage is None:
            storage = grouped_tensor.columnwise_data
        assert storage is not None
        original_data_ptr = storage.data_ptr()

        # Create input tensors with varying shapes
        input_tensors = [torch.randn(s, dtype=torch.float32, device="cuda") for s in shape]

        # Quantize in place
        quantized_tensors = grouped_tensor.quantize(input_tensors)

        # Verify data pointer hasn't changed
        assert storage.data_ptr() == original_data_ptr

        # Verify each tensor points to correct location
        cumulative_numel = 0
        for qtensor, tensor_shape in zip(quantized_tensors, shape):
            rowwise_data = _get_rowwise_data_tensor(qtensor, quantization)
            expected_offset = _rowwise_offset_bytes(cumulative_numel, quantization)
            assert rowwise_data.data_ptr() == original_data_ptr + expected_offset
            cumulative_numel += tensor_shape[0] * tensor_shape[1]

    @pytest.mark.parametrize(
        "shape",
        [[(256, 512), (512, 512), (768, 512)], [(512, 512), (512, 512), (512, 512)]],
    )
    @pytest.mark.skipif(not mxfp8_available, reason=reason_for_no_mxfp8)
    def test_quantize_grouped_mxfp8(self, shape: List[Tuple[int, int]]) -> None:
        """Test grouped quantization for MXFP8 against per-tensor quantization."""
        # Test wont pass until the grouped quantization PR from Oleg is merged.
        num_tensors = 2
        shape = [(512, 1024) for _ in range(num_tensors)]

        # Create BF16 input tensors and pack into a 2D tensor
        input_tensors = [torch.randn(s, dtype=torch.bfloat16, device="cuda") for s in shape]
        grouped_input = torch.cat(input_tensors, dim=0)

        # Create MXFP8 output grouped tensor (rowwise only for easier validation)
        quantizer = MXFP8Quantizer(fp8_dtype=tex.DType.kFloat8E4M3)
        quantizer.set_usage(rowwise=True, columnwise=False)
        first_dims = torch.tensor(
            [shape[0][0] for _ in range(num_tensors)],
            dtype=torch.int64,
            device="cuda",
        )

        # Quantize using grouped API
        grouped_output = tex.group_quantize(
            grouped_input,
            quantizer,
            num_tensors,
            first_dims,
        )
        # Build expected output by quantizing each tensor independently
        expected_data = []
        expected_scale_inv = []
        for tensor in input_tensors:
            qtensor = quantizer(tensor)
            expected_data.append(qtensor._rowwise_data.reshape(-1))
            expected_scale_inv.append(qtensor._rowwise_scale_inv.reshape(-1))

        expected_data = torch.cat(expected_data)
        expected_scale_inv = torch.cat(expected_scale_inv)

        assert torch.equal(grouped_output.rowwise_data, expected_data)
        assert torch.equal(grouped_output.scale_inv, expected_scale_inv)

    @pytest.mark.skipif(not mxfp8_available, reason=reason_for_no_mxfp8)
    def test_group_quantize_cudagraph_capturable(self) -> None:
        """Ensure group_quantize is CUDA graph capturable."""
        num_tensors = 2
        shape = [(512, 1024) for _ in range(num_tensors)]
        input_tensors = [torch.randn(s, dtype=torch.bfloat16, device="cuda") for s in shape]
        grouped_input = torch.cat(input_tensors, dim=0)

        quantizer = MXFP8Quantizer(fp8_dtype=tex.DType.kFloat8E4M3)
        quantizer.set_usage(rowwise=True, columnwise=False)
        first_dims = torch.tensor(
            [shape[0][0] for _ in range(num_tensors)],
            dtype=torch.int64,
            device="cuda",
        )

        torch.cuda.synchronize()
        static_input = grouped_input.clone()
        static_first_dims = first_dims.clone()

        # Warmup to initialize kernels and allocator state
        _ = tex.group_quantize(static_input, quantizer, num_tensors, static_first_dims)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_output = tex.group_quantize(
                static_input,
                quantizer,
                num_tensors,
                static_first_dims,
            )

        fresh_input = torch.cat(
            [torch.randn(s, dtype=torch.bfloat16, device="cuda") for s in shape],
            dim=0,
        )
        static_input.copy_(fresh_input)
        graph.replay()
        torch.cuda.synchronize()

        expected = tex.group_quantize(static_input, quantizer, num_tensors, static_first_dims)
        assert torch.equal(static_output.rowwise_data, expected.rowwise_data)
        assert torch.equal(static_output.scale_inv, expected.scale_inv)

    def test_clear(self) -> None:
        """Test clear method"""
        num_tensors = 3
        shape = [(256, 512) for _ in range(num_tensors)]

        grouped_tensor = GroupedTensor.make_grouped_tensor_with_shapes(
            num_tensors=num_tensors,
            shapes=shape,
            quantizer=None,
            device="cuda",
            dtype=torch.float32,
        )

        assert grouped_tensor.has_data()
        assert grouped_tensor.num_tensors == num_tensors

        grouped_tensor.clear()

        assert not grouped_tensor.has_data()
        assert grouped_tensor.num_tensors == 0
        assert grouped_tensor.rowwise_data is None
        assert grouped_tensor.logical_shape == (0, 0)

    def test_grouped_linear_load_state_dict_multi_to_single_param(self, tmp_path) -> None:
        """Load per-GEMM checkpoint from disk into single grouped parameter format."""
        num_gemms = 3
        in_features = 64
        out_features = 32
        dtype = torch.float32

        src = te.GroupedLinear(
            num_gemms=num_gemms,
            in_features=in_features,
            out_features=out_features,
            params_dtype=dtype,
            single_grouped_parameter=False,
        ).cuda()
        with torch.no_grad():
            for i in range(num_gemms):
                getattr(src, f"weight{i}").copy_(
                    torch.randn(out_features, in_features, device="cuda", dtype=dtype)
                )
                if src.use_bias:
                    getattr(src, f"bias{i}").copy_(
                        torch.randn(out_features, device="cuda", dtype=dtype)
                    )
        expected_weights = [getattr(src, f"weight{i}").detach().clone() for i in range(num_gemms)]
        ckpt_path = tmp_path / "grouped_linear_per_gemm.pt"
        torch.save(src.state_dict(), ckpt_path)
        del src

        src_state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        dst = te.GroupedLinear(
            num_gemms=num_gemms,
            in_features=in_features,
            out_features=out_features,
            params_dtype=dtype,
            single_grouped_parameter=True,
        ).cuda()
        load_result = dst.load_state_dict(src_state_dict, strict=True)
        assert len(load_result.missing_keys) == 0
        assert len(load_result.unexpected_keys) == 0

        assert getattr(dst, "weight", None) is not None
        loaded_weights = dst.weight.split_into_quantized_tensors()
        assert len(loaded_weights) == num_gemms
        for loaded_weight, expected_weight in zip(loaded_weights, expected_weights):
            assert torch.equal(loaded_weight, expected_weight)

    def test_grouped_linear_load_state_dict_single_to_multi_param(self, tmp_path) -> None:
        """Load grouped-parameter checkpoint from disk into per-GEMM parameter format."""
        num_gemms = 3
        in_features = 64
        out_features = 32
        dtype = torch.float32

        src = te.GroupedLinear(
            num_gemms=num_gemms,
            in_features=in_features,
            out_features=out_features,
            params_dtype=dtype,
            single_grouped_parameter=True,
        ).cuda()
        with torch.no_grad():
            source_weights = src.weight.split_into_quantized_tensors()
            for i in range(num_gemms):
                source_weights[i].copy_(
                    torch.randn(out_features, in_features, device="cuda", dtype=dtype)
                )
        expected_weights = [weight.detach().clone() for weight in source_weights]
        ckpt_path = tmp_path / "grouped_linear_single_param.pt"
        torch.save(src.state_dict(), ckpt_path)
        del src

        src_state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        dst = te.GroupedLinear(
            num_gemms=num_gemms,
            in_features=in_features,
            out_features=out_features,
            params_dtype=dtype,
            single_grouped_parameter=False,
        ).cuda()
        load_result = dst.load_state_dict(src_state_dict, strict=True)
        assert len(load_result.missing_keys) == 0
        assert len(load_result.unexpected_keys) == 0

        for i, expected_weight in enumerate(expected_weights):
            assert torch.equal(getattr(dst, f"weight{i}"), expected_weight)
