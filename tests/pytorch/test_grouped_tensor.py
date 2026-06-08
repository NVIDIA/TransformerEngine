# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Tests for GroupedTensor class"""

from typing import List, Optional, Tuple
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
from transformer_engine.pytorch.utils import is_non_tn_fp8_gemm_supported
import transformer_engine_torch as tex

# Import test utilities
from utils import assert_close

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
            fp8_dtype=te.DType.kFloat8E4M3,
        )
    elif quantization == "fp8_current_scaling":
        quantizer = Float8CurrentScalingQuantizer(
            fp8_dtype=te.DType.kFloat8E4M3,
            device="cuda",
        )
        quantizer.set_usage(rowwise=True, columnwise=False)
    elif quantization == "fp8_blockwise":
        quantizer = Float8BlockQuantizer(
            fp8_dtype=te.DType.kFloat8E4M3,
            rowwise=True,
            columnwise=False,
            force_pow_2_scales=True,
            amax_epsilon=0.0,
            block_scaling_dim=1,
        )
    elif quantization == "mxfp8":
        quantizer = MXFP8Quantizer(fp8_dtype=te.DType.kFloat8E4M3)
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

    @pytest.mark.parametrize(
        "split_sizes_list,logical_last_dim",
        [
            pytest.param([3, 4, 5, 2], 7, id="all_nonzero"),
            pytest.param([3, 0, 5, 2], 7, id="zero_middle"),
            pytest.param([0, 3, 5, 0], 11, id="zero_edges"),
            pytest.param([1], 17, id="single_group"),
            pytest.param([1, 2, 3, 4, 5, 6, 7, 8], 13, id="many_groups"),
            # MoE-style group counts. ``split_points`` (an int32[num_groups]
            # tensor packed into a shared buffer alongside int64 outputs) used
            # to land at an 8-byte-aligned offset for these counts, which
            # tripped cuDNN's 16-byte alignment requirement in grouped GEMM.
            pytest.param([8192] * 8, 2048, id="num_groups_8_uniform"),
            pytest.param([4096] * 16, 4096, id="num_groups_16_uniform"),
            pytest.param([2048] * 32, 7168, id="num_groups_32_uniform"),
            pytest.param([1024] * 64, 7168, id="num_groups_64_uniform"),
            pytest.param([512] * 128, 7168, id="num_groups_128_uniform"),
            # Non-uniform with large totals to also exercise tensor_offsets > 2^31.
            pytest.param(
                [12345, 0, 8192, 1, 65536, 100, 131072, 7],
                7168,
                id="non_uniform_large_totals",
            ),
        ],
    )
    @pytest.mark.parametrize("input_dtype", [torch.int32, torch.int64], ids=["int32", "int64"])
    @pytest.mark.parametrize("input_device", ["cuda", "cpu"], ids=["cuda", "cpu"])
    @pytest.mark.parametrize("bulk_allocate", [False, True], ids=["separate", "bulk"])
    def test_splits_to_offsets_multi(
        self,
        bulk_allocate: bool,
        input_device: str,
        input_dtype: torch.dtype,
        split_sizes_list: List[int],
        logical_last_dim: int,
    ) -> None:
        """Test fused grouped split metadata preparation."""
        device = torch.device("cuda")
        split_sizes = torch.tensor(split_sizes_list, dtype=input_dtype, device=input_device)

        # Exercise the grouped-MLP-shaped call: mix of int32 (no leading zero)
        # and int64 (with leading zero) outputs, several strides.
        strides = [1, 1, logical_last_dim, 0, logical_last_dim + 17]
        include_leading_zero = [False, True, True, True, True]
        dtypes = [torch.int32, torch.int64, torch.int64, torch.int64, torch.int64]
        split_sizes_out, outputs = tex.splits_to_offsets_multi(
            split_sizes,
            device,
            strides=strides,
            include_leading_zero=include_leading_zero,
            dtypes=dtypes,
            bulk_allocate=bulk_allocate,
        )

        # Reference implementation.
        expected_split_sizes_i64 = split_sizes.to(device=device, dtype=torch.int64)
        expected_base_offsets = torch.cat(
            (
                torch.zeros(1, dtype=torch.int64, device=device),
                torch.cumsum(expected_split_sizes_i64, dim=0),
            )
        )

        # Check output split_sizes: always int64, always on the target device.
        assert split_sizes_out.device.type == "cuda"
        assert split_sizes_out.dtype == torch.int64
        assert_close(split_sizes_out, expected_split_sizes_i64)

        # Check output offsets.
        assert len(outputs) == len(strides)
        for output, stride, with_zero, dtype in zip(outputs, strides, include_leading_zero, dtypes):
            assert output.dtype == dtype
            assert output.device.type == "cuda"
            expected_length = split_sizes.numel() + (1 if with_zero else 0)
            assert output.numel() == expected_length
            expected = expected_base_offsets * stride
            if not with_zero:
                expected = expected[1:]
            assert_close(output, expected)

        # Check pointer alignment: cuDNN CuTe-DSL grouped GEMM kernels
        # require 16-byte-aligned data pointers.
        for idx, output in enumerate(outputs):
            assert (
                output.data_ptr() % 16 == 0
            ), f"outputs[{idx}] data_ptr is not 16-byte aligned: {output.data_ptr():#x}"

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
        "shape_case",
        ["varying_first", "varying_last", "varying_both"],
    )
    @pytest.mark.parametrize("output_dbias", [False, True])
    @pytest.mark.skipif(not mxfp8_available, reason=reason_for_no_mxfp8)
    def test_quantize_grouped_mxfp8(self, shape_case: str, output_dbias: bool) -> None:
        """Test grouped MXFP8 quantization against per-tensor quantization for
        varying first/last/both dimensions."""
        if output_dbias and shape_case != "varying_first":
            pytest.skip("bgrad_group_quantize requires constant last dimension")

        # Per-tensor shapes are chosen to satisfy MXFP8 alignment requirements:
        #   - rowwise scale block size 32 -> last dim must be a multiple of 32
        #   - kernel chunk size 128 along the first dim -> first dim must be a
        #     multiple of 128 (per-tensor for VARYING_BOTH_DIMS, otherwise the
        #     logical first dim).
        if shape_case == "varying_first":
            per_tensor_shapes = [(128, 512), (256, 512), (384, 512)]
        elif shape_case == "varying_last":
            per_tensor_shapes = [(512, 128), (512, 256), (512, 384)]
        else:  # varying_both
            per_tensor_shapes = [(128, 256), (256, 512), (384, 384)]

        num_tensors = len(per_tensor_shapes)

        # Each tensor occupies a contiguous chunk of a flat buffer; the kernel
        # locates each chunk via tensor_offsets, so the 2D view below only needs
        # to encode the correct total number of elements.
        input_tensors = [
            torch.randn(s, dtype=torch.bfloat16, device="cuda") for s in per_tensor_shapes
        ]
        flat_buffer = torch.cat([t.reshape(-1) for t in input_tensors])

        first_dims_host: Optional[List[int]]
        last_dims_host: Optional[List[int]]
        if shape_case == "varying_first":
            first_dims_host = [s[0] for s in per_tensor_shapes]
            last_dims_host = None
            common_last = per_tensor_shapes[0][1]
            grouped_input = flat_buffer.view(sum(first_dims_host), common_last)
        elif shape_case == "varying_last":
            first_dims_host = None
            last_dims_host = [s[1] for s in per_tensor_shapes]
            common_first = per_tensor_shapes[0][0]
            grouped_input = flat_buffer.view(common_first, sum(last_dims_host))
        else:  # varying_both
            first_dims_host = [s[0] for s in per_tensor_shapes]
            last_dims_host = [s[1] for s in per_tensor_shapes]
            grouped_input = flat_buffer.view(1, -1)

        first_dims = (
            torch.tensor(first_dims_host, dtype=torch.int64, device="cuda")
            if first_dims_host is not None
            else None
        )
        last_dims = (
            torch.tensor(last_dims_host, dtype=torch.int64, device="cuda")
            if last_dims_host is not None
            else None
        )

        quantizer = MXFP8Quantizer(fp8_dtype=tex.DType.kFloat8E4M3)
        quantizer.set_usage(rowwise=True, columnwise=False)

        if output_dbias:
            grouped_output, dbias = tex.bgrad_group_quantize(
                grouped_input, quantizer, num_tensors, first_dims, last_dims
            )
        else:
            grouped_output = tex.group_quantize(
                grouped_input, quantizer, num_tensors, first_dims, last_dims
            )

        # Reference: quantize each tensor independently and concatenate the
        # rowwise data / scale_inv buffers in tensor order.
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

        if output_dbias:
            expected_dbias = torch.stack([t.sum(dim=0) for t in input_tensors])
            assert torch.allclose(dbias, expected_dbias)

    @pytest.mark.parametrize("output_dbias", [False, True])
    @pytest.mark.skipif(not mxfp8_available, reason=reason_for_no_mxfp8)
    def test_group_quantize_precomputed_offsets(self, output_dbias: bool) -> None:
        """Test grouped quantization can reuse caller-provided tensor offsets."""
        num_tensors = 2
        last_dim = 1024
        split_sizes_list = [512, 512]
        input_tensors = [
            torch.randn(split, last_dim, dtype=torch.bfloat16, device="cuda")
            for split in split_sizes_list
        ]
        grouped_input = torch.cat(input_tensors, dim=0)

        quantizer = MXFP8Quantizer(fp8_dtype=tex.DType.kFloat8E4M3)
        quantizer.set_usage(rowwise=True, columnwise=False)
        split_sizes = torch.tensor(split_sizes_list, dtype=torch.int64, device="cuda")
        split_sizes, (tensor_offsets,) = tex.splits_to_offsets_multi(
            split_sizes,
            torch.device("cuda"),
            strides=[last_dim],
            include_leading_zero=[True],
            dtypes=[torch.int64],
        )

        if output_dbias:
            grouped_output, dbias = tex.bgrad_group_quantize(
                grouped_input,
                quantizer,
                num_tensors,
                split_sizes,
                tensor_offsets=tensor_offsets,
            )
            expected_output, expected_dbias = tex.bgrad_group_quantize(
                grouped_input,
                quantizer,
                num_tensors,
                split_sizes,
            )
            assert torch.allclose(dbias, expected_dbias)
        else:
            grouped_output = tex.group_quantize(
                grouped_input,
                quantizer,
                num_tensors,
                split_sizes,
                tensor_offsets=tensor_offsets,
            )
            expected_output = tex.group_quantize(
                grouped_input,
                quantizer,
                num_tensors,
                split_sizes,
            )

        assert grouped_output.tensor_offsets.data_ptr() == tensor_offsets.data_ptr()
        assert torch.equal(grouped_output.rowwise_data, expected_output.rowwise_data)
        assert torch.equal(grouped_output.scale_inv, expected_output.scale_inv)

    @pytest.mark.skipif(not mxfp8_available, reason=reason_for_no_mxfp8)
    def test_bgrad_group_quantize_zero_size_tensor(self) -> None:
        """Test bgrad_group_quantize handles zero-row input without error."""
        num_tensors = 3
        last_dim = 1024
        grouped_input = torch.empty(0, last_dim, dtype=torch.bfloat16, device="cuda")

        quantizer = MXFP8Quantizer(fp8_dtype=te.DType.kFloat8E4M3)
        quantizer.set_usage(rowwise=True, columnwise=False)
        first_dims = torch.zeros(num_tensors, dtype=torch.int64, device="cuda")

        grouped_output, dbias = tex.bgrad_group_quantize(
            grouped_input,
            quantizer,
            num_tensors,
            first_dims,
        )

        assert dbias.shape == (num_tensors, last_dim)
        assert torch.all(dbias == 0)

    @pytest.mark.parametrize(
        "quantization",
        [
            pytest.param(
                "fp8_current_scaling",
                marks=pytest.mark.skipif(not fp8_available, reason=reason_for_no_fp8),
            ),
            pytest.param(
                "mxfp8",
                marks=pytest.mark.skipif(not mxfp8_available, reason=reason_for_no_mxfp8),
            ),
        ],
    )
    @pytest.mark.parametrize("output_dbias", [False, True])
    @pytest.mark.parametrize("shape_case", ["varying_first", "varying_last"])
    def test_group_quantize_cudagraph_capturable(
        self, quantization: str, output_dbias: bool, shape_case: str
    ) -> None:
        """Ensure group_quantize is CUDA graph capturable."""
        if output_dbias and quantization != "mxfp8":
            pytest.skip("bgrad_group_quantize only supports MXFP8")
        if output_dbias and shape_case == "varying_last":
            pytest.skip("bgrad_group_quantize does not accept last_dims")

        if shape_case == "varying_last":
            rows = 128
            last_dims_host = [256, 128, 384]
            num_tensors = len(last_dims_host)
            total_cols = sum(last_dims_host)
            flat_input = torch.empty(rows * total_cols, dtype=torch.bfloat16, device="cuda")
            offset = 0
            for cols in last_dims_host:
                member = torch.randn(rows, cols, dtype=torch.bfloat16, device="cuda")
                flat_input[offset : offset + member.numel()].copy_(member.reshape(-1))
                offset += member.numel()
            grouped_input = flat_input.view(rows, total_cols)
            first_dims = None
            last_dims = torch.tensor(last_dims_host, dtype=torch.int64, device="cuda")
        else:
            first_dims_host = [256, 128, 384]
            num_tensors = len(first_dims_host)
            hidden = 1024
            shape = [(r, hidden) for r in first_dims_host]
            input_tensors = [
                torch.randn(s, dtype=torch.bfloat16, device="cuda") for s in shape
            ]
            grouped_input = torch.cat(input_tensors, dim=0)
            first_dims = torch.tensor(first_dims_host, dtype=torch.int64, device="cuda")
            last_dims = None

        if quantization == "mxfp8":
            quantizer = MXFP8Quantizer(fp8_dtype=tex.DType.kFloat8E4M3)
        else:
            quantizer = Float8CurrentScalingQuantizer(
                fp8_dtype=tex.DType.kFloat8E4M3,
                device="cuda",
                force_pow_2_scales=False,
                amax_epsilon=0.0,
            )
        quantizer.set_usage(rowwise=True, columnwise=False)

        torch.cuda.synchronize()
        static_input = grouped_input.clone()
        static_first_dims = first_dims.clone() if first_dims is not None else None
        static_last_dims = last_dims.clone() if last_dims is not None else None

        def _run_group_quantize(input_tensor):
            """Return (output, dbias) where dbias is None when output_dbias is False."""
            if output_dbias:
                out, dbias = tex.bgrad_group_quantize(
                    input_tensor, quantizer, num_tensors, static_first_dims
                )
                return out, dbias
            out = tex.group_quantize(
                input_tensor,
                quantizer,
                num_tensors,
                static_first_dims,
                last_dims=static_last_dims,
            )
            return out, None

        # Warmup to initialize kernels and allocator state
        _ = _run_group_quantize(static_input)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_output, static_dbias = _run_group_quantize(static_input)

        fresh_input = torch.randn_like(grouped_input)
        static_input.copy_(fresh_input)
        graph.replay()
        torch.cuda.synchronize()

        expected_out, expected_dbias = _run_group_quantize(static_input)
        assert torch.equal(static_output.rowwise_data, expected_out.rowwise_data)
        assert torch.equal(static_output.scale_inv, expected_out.scale_inv)
        if output_dbias:
            assert torch.allclose(static_dbias, expected_dbias)

    @pytest.mark.parametrize("mode", ["rowwise", "columnwise", "both"])
    @pytest.mark.parametrize(
        "shape_case",
        ["uniform", "varying_first", "empty_split", "varying_last"],
    )
    @pytest.mark.parametrize("overallocated", [False, True])
    @pytest.mark.skipif(not fp8_available, reason=reason_for_no_fp8)
    def test_group_quantize_fp8_current_scaling(
        self,
        mode: str,
        shape_case: str,
        overallocated: bool,
    ) -> None:
        """Test grouped FP8 current scaling matches per-tensor current scaling
        across shape topologies:

        - ``uniform``: no ``first_dims``/``last_dims`` (kernel partitions implicitly).
        - ``varying_first``: ``first_dims`` set, values vary.
        - ``empty_split``: ``first_dims`` set with one zero entry.
        - ``varying_last``: ``last_dims`` set, values vary.

        When ``overallocated`` is True the input is reshaped to a logical_shape whose
        first dim is twice ``sum(first_dims)``, so the kernel sees an active region
        (rows covered by ``first_dims``) followed by an unused tail. The backing
        buffer always matches logical_shape exactly. The unused input tail rows are
        poisoned with a large sentinel value (1e4); since the per-group amax
        assertion compares against amax computed over the active input tensors only,
        any tail read by the kernel would explode the per-group amax and the
        assertion would fail. Overallocation is skipped for ``uniform`` and
        ``varying_last`` because they don't have a varying-first tail to test.
        """
        if overallocated and shape_case in ("uniform", "varying_last"):
            pytest.skip(
                "Overallocation is not meaningful for this shape_case "
                "(implicit partitioning / varying-last semantics)."
            )

        # Per-tensor shapes for each shape_case.
        if shape_case == "uniform":
            per_tensor_shapes = [(128, 256)] * 3
            first_dims_host = None
            last_dims_host = None
        elif shape_case == "varying_first":
            first_dims_host = [64, 128, 96]
            last_dims_host = None
            per_tensor_shapes = [(r, 256) for r in first_dims_host]
        elif shape_case == "empty_split":
            first_dims_host = [128, 0, 96]
            last_dims_host = None
            per_tensor_shapes = [(r, 256) for r in first_dims_host]
        elif shape_case == "varying_last":
            first_dims_host = None
            last_dims_host = [513, 1027, 259]
            per_tensor_shapes = [(256, c) for c in last_dims_host]
        else:
            raise ValueError(f"Unknown shape_case: {shape_case}")

        num_tensors = len(per_tensor_shapes)
        first_dims = (
            torch.tensor(first_dims_host, dtype=torch.int64, device="cuda")
            if first_dims_host is not None
            else None
        )
        last_dims = (
            torch.tensor(last_dims_host, dtype=torch.int64, device="cuda")
            if last_dims_host is not None
            else None
        )

        # Per-tensor data + flat buffer (tensor i occupies a contiguous chunk).
        input_tensors = [
            torch.randn(s, dtype=torch.bfloat16, device="cuda") for s in per_tensor_shapes
        ]
        actual_numel = sum(t.numel() for t in input_tensors)
        allocated_numel = actual_numel * 2 if overallocated else actual_numel
        flat_buffer = torch.empty(allocated_numel, dtype=torch.bfloat16, device="cuda")
        offset = 0
        for t in input_tensors:
            flat_buffer[offset : offset + t.numel()].copy_(t.reshape(-1))
            offset += t.numel()
        if overallocated:
            flat_buffer[actual_numel:].fill_(10000.0)

        # View flat buffer as the 2D shape expected by group_quantize.
        if shape_case == "varying_last":
            common_first = per_tensor_shapes[0][0]
            total_last = sum(last_dims_host)
            grouped_input = flat_buffer.view(common_first, total_last)
        else:
            common_last = per_tensor_shapes[0][1]
            allocated_first = allocated_numel // common_last
            grouped_input = flat_buffer.view(allocated_first, common_last)

        requested_rowwise = mode in ("rowwise", "both")
        requested_columnwise = mode in ("columnwise", "both")
        rowwise = requested_rowwise or (
            requested_columnwise and is_non_tn_fp8_gemm_supported()
        )
        columnwise = requested_columnwise and not is_non_tn_fp8_gemm_supported()

        quantizer = Float8CurrentScalingQuantizer(
            fp8_dtype=tex.DType.kFloat8E4M3,
            device="cuda",
            force_pow_2_scales=False,
            amax_epsilon=0.0,
        )
        quantizer.set_usage(rowwise=requested_rowwise, columnwise=requested_columnwise)

        grouped_output = tex.group_quantize(
            grouped_input, quantizer, num_tensors, first_dims, last_dims=last_dims
        )

        # Metadata validation for the varying-last code path.
        if shape_case == "varying_last":
            assert grouped_output.first_dims is None
            assert torch.equal(grouped_output.last_dims, last_dims)

        # When ``overallocated`` is True, the input has poisoned rows past
        # sum(first_dims) (filled with 1e4). If the kernel were to read those
        # tail rows, per-group amax would explode well above what bf16 N(0,1)
        # produces. We catch that implicitly via the per-group amax assertion in
        # ``_assert_fp8_cs_group_quantize_matches_reference`` below.
        self._assert_fp8_cs_group_quantize_matches_reference(
            grouped_output=grouped_output,
            input_tensors=input_tensors,
            rowwise=rowwise,
            columnwise=columnwise,
        )

    @staticmethod
    def _assert_fp8_cs_group_quantize_matches_reference(
        *,
        grouped_output,
        input_tensors: List[torch.Tensor],
        rowwise: bool,
        columnwise: bool,
    ) -> None:
        """Validate amax/scale/scale_inv/per-tensor data of an FP8 current-scaling
        grouped output against per-tensor Float8CurrentScalingQuantizer references."""
        expected_amax = torch.stack(
            [
                (
                    tensor.abs().max().float()
                    if tensor.numel() > 0
                    else torch.zeros((), dtype=torch.float32, device="cuda")
                )
                for tensor in input_tensors
            ]
        )
        torch.testing.assert_close(grouped_output.amax, expected_amax, rtol=0.0, atol=0.0)

        scale_inv = grouped_output.scale_inv
        if scale_inv is None:
            scale_inv = grouped_output.columnwise_scale_inv
        torch.testing.assert_close(
            scale_inv,
            torch.reciprocal(grouped_output.scale),
            rtol=1e-6,
            atol=1e-6,
        )
        if rowwise and columnwise:
            torch.testing.assert_close(
                grouped_output.columnwise_scale_inv,
                grouped_output.scale_inv,
                rtol=0.0,
                atol=0.0,
            )

        expected_rowwise = []
        expected_columnwise = []
        for tensor in input_tensors:
            if tensor.numel() == 0:
                continue
            ref_quantizer = Float8CurrentScalingQuantizer(
                fp8_dtype=tex.DType.kFloat8E4M3,
                device="cuda",
                rowwise=True,
                columnwise=False,
                force_pow_2_scales=False,
                amax_epsilon=0.0,
            )
            ref = ref_quantizer(tensor)
            ref_rowwise_data = ref._data.reshape(tensor.shape)
            if rowwise:
                expected_rowwise.append(ref_rowwise_data.reshape(-1))
            if columnwise:
                expected_columnwise.append(ref_rowwise_data.T.contiguous().reshape(-1))

        if rowwise and expected_rowwise:
            expected = torch.cat(expected_rowwise)
            assert torch.equal(grouped_output.rowwise_data[: expected.numel()], expected)
        if columnwise and expected_columnwise:
            expected = torch.cat(expected_columnwise)
            assert torch.equal(grouped_output.columnwise_data[: expected.numel()], expected)

    @pytest.mark.parametrize(
        "shape",
        [[(512, 1024), (512, 1024)], [(256, 512), (512, 512), (768, 512)]],
    )
    @pytest.mark.skipif(not mxfp8_available, reason=reason_for_no_mxfp8)
    def test_group_dequantize(self, shape: List[Tuple[int, int]]) -> None:
        """Test grouped dequantization for MXFP8 back to BF16."""
        num_tensors = len(shape)

        # Create BF16 input tensors and quantize them with MXFP8.
        input_tensors = [torch.randn(s, dtype=torch.bfloat16, device="cuda") for s in shape]
        grouped_input = torch.cat(input_tensors, dim=0)

        quantizer = MXFP8Quantizer(fp8_dtype=te.DType.kFloat8E4M3)
        quantizer.set_usage(rowwise=True, columnwise=False)
        first_dims = torch.tensor([s[0] for s in shape], dtype=torch.int64, device="cuda")

        # Quantize.
        quantized = tex.group_quantize(grouped_input, quantizer, num_tensors, first_dims)

        # Dequantize.
        dequantized = tex.group_dequantize(quantized, te.DType.kBFloat16)

        # Verify output metadata.
        assert dequantized.num_tensors == num_tensors
        assert dequantized.logical_shape == quantized.logical_shape
        assert torch.equal(dequantized.first_dims, quantized.first_dims)
        assert torch.equal(dequantized.tensor_offsets, quantized.tensor_offsets)

        # Verify dequantized values are close to original (per-tensor).
        dequantized_tensors = dequantized.split_into_quantized_tensors()
        assert len(dequantized_tensors) == num_tensors
        for orig, deq in zip(input_tensors, dequantized_tensors):
            torch.testing.assert_close(deq, orig, atol=0.125, rtol=0.1)

    @pytest.mark.skipif(not mxfp8_available, reason=reason_for_no_mxfp8)
    def test_group_dequantize_cudagraph_capturable(self) -> None:
        """Ensure group_dequantize is CUDA graph capturable."""
        num_tensors = 2
        shape = [(512, 1024) for _ in range(num_tensors)]
        input_tensors = [torch.randn(s, dtype=torch.bfloat16, device="cuda") for s in shape]
        grouped_input = torch.cat(input_tensors, dim=0)

        quantizer = MXFP8Quantizer(fp8_dtype=te.DType.kFloat8E4M3)
        quantizer.set_usage(rowwise=True, columnwise=False)
        first_dims = torch.tensor(
            [shape[0][0] for _ in range(num_tensors)],
            dtype=torch.int64,
            device="cuda",
        )

        # Quantize to get MXFP8 grouped tensor.
        quantized = tex.group_quantize(grouped_input, quantizer, num_tensors, first_dims)

        # Warmup dequantize.
        torch.cuda.synchronize()
        _ = tex.group_dequantize(quantized, te.DType.kBFloat16)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_output = tex.group_dequantize(quantized, te.DType.kBFloat16)

        # Replay with different input data.
        fresh_input = torch.cat(
            [torch.randn(s, dtype=torch.bfloat16, device="cuda") for s in shape],
            dim=0,
        )
        fresh_quantized = tex.group_quantize(fresh_input, quantizer, num_tensors, first_dims)
        quantized.rowwise_data.copy_(fresh_quantized.rowwise_data)
        quantized.scale_inv.copy_(fresh_quantized.scale_inv)

        graph.replay()
        torch.cuda.synchronize()

        expected = tex.group_dequantize(quantized, te.DType.kBFloat16)
        expected_tensors = expected.split_into_quantized_tensors()
        static_tensors = static_output.split_into_quantized_tensors()
        for exp, got in zip(expected_tensors, static_tensors):
            assert torch.equal(got, exp)

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
            single_grouped_weight=False,
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
        expected_biases = [getattr(src, f"bias{i}").detach().clone() for i in range(num_gemms)]
        ckpt_path = tmp_path / "grouped_linear_per_gemm.pt"
        torch.save(src.state_dict(), ckpt_path)
        del src

        src_state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        dst = te.GroupedLinear(
            num_gemms=num_gemms,
            in_features=in_features,
            out_features=out_features,
            params_dtype=dtype,
            single_grouped_weight=True,
            single_grouped_bias=True,
        ).cuda()
        load_result = dst.load_state_dict(src_state_dict, strict=True)
        assert len(load_result.missing_keys) == 0
        assert len(load_result.unexpected_keys) == 0

        assert getattr(dst, "weight", None) is not None
        loaded_weights = dst.weight.split_into_quantized_tensors()
        assert len(loaded_weights) == num_gemms
        for loaded_weight, expected_weight in zip(loaded_weights, expected_weights):
            assert torch.equal(loaded_weight, expected_weight)

        assert getattr(dst, "bias", None) is not None
        loaded_biases = dst.bias.split_into_quantized_tensors()
        assert len(loaded_biases) == num_gemms
        for loaded_bias, expected_bias in zip(loaded_biases, expected_biases):
            assert torch.equal(loaded_bias.reshape(-1), expected_bias.reshape(-1))

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
            single_grouped_weight=True,
            single_grouped_bias=True,
        ).cuda()
        with torch.no_grad():
            source_weights = src.weight.split_into_quantized_tensors()
            for i in range(num_gemms):
                source_weights[i].copy_(
                    torch.randn(out_features, in_features, device="cuda", dtype=dtype)
                )
        expected_weights = [weight.detach().clone() for weight in source_weights]
        source_biases = src.bias.split_into_quantized_tensors()
        for i in range(num_gemms):
            source_biases[i].copy_(torch.randn(out_features, device="cuda", dtype=dtype))
        expected_biases = [b.detach().clone() for b in source_biases]
        ckpt_path = tmp_path / "grouped_linear_single_param.pt"
        torch.save(src.state_dict(), ckpt_path)
        del src

        src_state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        dst = te.GroupedLinear(
            num_gemms=num_gemms,
            in_features=in_features,
            out_features=out_features,
            params_dtype=dtype,
            single_grouped_weight=False,
        ).cuda()
        load_result = dst.load_state_dict(src_state_dict, strict=True)
        assert len(load_result.missing_keys) == 0
        assert len(load_result.unexpected_keys) == 0

        for i, expected_weight in enumerate(expected_weights):
            assert torch.equal(getattr(dst, f"weight{i}"), expected_weight)
        for i, expected_bias in enumerate(expected_biases):
            assert torch.equal(getattr(dst, f"bias{i}"), expected_bias.reshape(-1))
