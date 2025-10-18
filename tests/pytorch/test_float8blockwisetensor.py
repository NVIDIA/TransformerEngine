# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

from collections.abc import Iterable
import io
import math
from typing import Any, Dict, List, Tuple, Union

import pytest
import torch

import transformer_engine.common.recipe
from transformer_engine.pytorch import (
    Float8BlockQuantizer,
    Float8BlockwiseQTensor,
    get_device_compute_capability,
)
import transformer_engine_torch as tex

# PyTorch tensor dtypes
_dtypes: List[torch.dtype] = [torch.float32, torch.float16, torch.bfloat16]
# TE FP8 dtypes
_fp8_dtypes: List[tex.DType] = [tex.DType.kFloat8E4M3, tex.DType.kFloat8E5M2]

# Numerical tolerances with FP8 types
_tols: Dict[tex.DType, Dict[str, float]] = {
    tex.DType.kFloat8E4M3: dict(rtol=0.125, atol=0.08),
    tex.DType.kFloat8E5M2: dict(rtol=0.25, atol=0.125),
}


def _to_list(x: Union[Iterable, Any]) -> List:
    """Convert to list if iterable, otherwise put in singleton list"""
    if isinstance(x, Iterable):
        return list(x)
    else:
        return [x]


# Types that can be interpreted as tensor dims
DimsType = Union[Iterable[int], int]

# TODO replace with call to fp8.py when recipe added.
recipe_available = get_device_compute_capability() >= (9, 0) and float(torch.version.cuda) >= 12.8
reason_for_no_recipe = "Quantize kernels require TMA and are only relevant with GEMMS."


@pytest.mark.skipif(not recipe_available, reason=reason_for_no_recipe)
class TestFloat8BlockwiseTensor:

    @staticmethod
    def setup_class(cls) -> None:
        # Configure RNG
        seed = 1234
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

    def test_constructor(
        self,
        dims: DimsType = 1,
        fp8_dtype: tex.DType = tex.DType.kFloat8E4M3,
        dtype: torch.dtype = torch.float32,
        is_2D_scaled: bool = True,
    ) -> None:
        """Call constructor and perform sanity checks"""
        dims = _to_list(dims)

        rowwise = True
        columnwise = True
        quantizer = Float8BlockQuantizer(
            fp8_dtype=fp8_dtype,
            rowwise=rowwise,
            columnwise=columnwise,
            block_scaling_dim=2 if is_2D_scaled else 1,
        )

        scale_dims = quantizer.get_scale_shape(dims, columnwise=False)
        columnwise_scale_dims = quantizer.get_scale_shape(dims, columnwise=True)
        columnwise_dims = quantizer.get_columnwise_shape(dims)
        tensor = Float8BlockwiseQTensor(
            shape=dims,
            dtype=dtype,
            rowwise_data=torch.zeros(dims, device="cuda", dtype=torch.uint8),
            rowwise_scale_inv=torch.zeros(scale_dims, device="cuda", dtype=torch.float32),
            columnwise_data=torch.zeros(columnwise_dims, device="cuda", dtype=torch.uint8),
            columnwise_scale_inv=torch.zeros(
                columnwise_scale_dims, device="cuda", dtype=torch.float32
            ),
            fp8_dtype=fp8_dtype,
            is_2D_scaled=is_2D_scaled,
            quantizer=quantizer,
        )
        assert list(tensor.size()) == dims, "Incorrect dims"
        assert tensor.dtype == dtype, "Incorrect nominal dtype"
        assert tensor.is_cuda, "Incorrect device"

    def _test_quantize_dequantize(
        self,
        quantizer: Float8BlockQuantizer,
        dtype: torch.dtype = torch.float32,
        dims: DimsType = (23, 128),
        rtol: float = 0.0,
        atol: float = 0.0,
        dequant_columnwise: bool = False,
        use_cpp_allocation: bool = False,
    ) -> None:
        """Check numerical error when casting to FP8 and back"""
        dims = _to_list(dims)

        # Initialize random data
        # Note: Make sure values are not all close to zero, or else
        # test may pass trivially.
        x_ref = 2 * torch.rand(dims, dtype=dtype, device="cpu") - 1
        x_ref.view(-1)[0] = 0.75
        x_ref_cuda = x_ref.to("cuda")

        # Cast to FP8 and back
        if not use_cpp_allocation:
            x_fp8 = quantizer.make_empty(shape=dims, device="cuda")
            quantizer.update_quantized(x_ref_cuda, x_fp8)
        else:
            # This codepath allows the CPP binding to allocate the output
            # tensor
            x_fp8 = tex.quantize(x_ref_cuda, quantizer, None, None)
        if dequant_columnwise:
            # Strip out rowwise data to verify dequantization of
            # columnwise data.
            x_fp8.update_usage(rowwise_usage=False, columnwise_usage=True)
        x_fp8 = x_fp8.dequantize(dtype=dtype).cpu()

        # Check results
        torch.testing.assert_close(x_fp8, x_ref, rtol=rtol, atol=atol)

        # Make sure we are not trivially passing the test
        with pytest.raises(AssertionError):
            torch.testing.assert_close(x_fp8, -x_ref, rtol=rtol, atol=atol)

    @pytest.mark.parametrize("fp8_dtype", _fp8_dtypes)
    @pytest.mark.parametrize("dtype", _dtypes)
    @pytest.mark.parametrize("block_scaling_dim", [1, 2])
    def test_quantize_dequantize_dtypes(
        self, fp8_dtype: tex.DType, dtype: torch.dtype, block_scaling_dim: int
    ) -> None:
        atol = _tols[fp8_dtype]["atol"]
        rtol = _tols[fp8_dtype]["rtol"]
        quantizer = Float8BlockQuantizer(
            fp8_dtype=fp8_dtype,
            rowwise=True,
            columnwise=False,
            block_scaling_dim=block_scaling_dim,
        )
        self._test_quantize_dequantize(quantizer=quantizer, dtype=dtype, atol=atol, rtol=rtol)

    @pytest.mark.parametrize("fp8_dtype", _fp8_dtypes)
    @pytest.mark.parametrize("dtype", _dtypes)
    @pytest.mark.parametrize("block_scaling_dim", [1])
    def test_quantize_dequantize_columnwise_only(
        self, fp8_dtype: tex.DType, dtype: torch.dtype, block_scaling_dim: int
    ) -> None:
        atol = _tols[fp8_dtype]["atol"]
        rtol = _tols[fp8_dtype]["rtol"]
        quantizer = Float8BlockQuantizer(
            fp8_dtype=fp8_dtype,
            rowwise=False,
            columnwise=True,
            block_scaling_dim=block_scaling_dim,
        )
        self._test_quantize_dequantize(
            quantizer=quantizer, dtype=dtype, atol=atol, rtol=rtol, use_cpp_allocation=True
        )

    @pytest.mark.parametrize(
        "dims", [[], 256, 311, [264], [256, 512], [250, 500], [7, 5, 3], [2, 3, 5, 3]]
    )
    @pytest.mark.parametrize("block_scaling_dim", [1, 2])
    @pytest.mark.parametrize("dq_columnwise", [True, False])
    @pytest.mark.parametrize("all_gather_usage", [True, False])
    def test_quantize_dequantize_dims(
        self,
        dims: DimsType,
        block_scaling_dim: int,
        dq_columnwise: bool,
        all_gather_usage: bool,
    ) -> None:
        if all_gather_usage and block_scaling_dim != 1:
            pytest.skip("all_gather_usage only implemented for 1D block quantization.")
        atol = _tols[tex.DType.kFloat8E4M3]["atol"]
        rtol = _tols[tex.DType.kFloat8E4M3]["rtol"]
        quantizer = Float8BlockQuantizer(
            fp8_dtype=tex.DType.kFloat8E4M3,
            rowwise=True,
            columnwise=dq_columnwise,
            block_scaling_dim=block_scaling_dim,
            all_gather_usage=all_gather_usage,
        )
        self._test_quantize_dequantize(
            quantizer=quantizer,
            dims=dims,
            atol=atol,
            rtol=rtol,
            dequant_columnwise=dq_columnwise,
        )

    @pytest.mark.parametrize(
        "dims", [[], 256, 311, [264], [256, 512], [250, 500], [7, 5, 3], [2, 3, 5, 3]]
    )
    @pytest.mark.parametrize("block_scaling_dim", [1, 2])
    @pytest.mark.parametrize("dq_columnwise", [True, False])
    @pytest.mark.xfail(raises=NotImplementedError)
    def test_quantize_dequantize_compact_format(
        self, dims: DimsType, block_scaling_dim: int, dq_columnwise: bool
    ) -> None:
        atol = _tols[tex.DType.kFloat8E4M3]["atol"]
        rtol = _tols[tex.DType.kFloat8E4M3]["rtol"]
        quantizer = Float8BlockQuantizer(
            fp8_dtype=tex.DType.kFloat8E4M3,
            rowwise=True,
            columnwise=dq_columnwise,
            block_scaling_dim=block_scaling_dim,
            all_gather_usage=(block_scaling_dim == 1),
        )
        self._test_quantize_dequantize(
            quantizer=quantizer,
            dims=dims,
            atol=atol,
            rtol=rtol,
            dequant_columnwise=dq_columnwise,
        )

    @pytest.mark.parametrize(
        "dims", [[], 256, 311, [264], [256, 512], [250, 500], [7, 5, 3], [2, 3, 5, 3]]
    )
    @pytest.mark.parametrize("block_scaling_dim", [1, 2])
    @pytest.mark.parametrize("fp8_dtype", _fp8_dtypes)
    @pytest.mark.parametrize("dq_columnwise", [True, False])
    def test_quantize_dequantize_dims_cpp_allocate_output(
        self, dims: DimsType, block_scaling_dim: int, fp8_dtype: tex.DType, dq_columnwise: bool
    ) -> None:
        atol = _tols[fp8_dtype]["atol"]
        rtol = _tols[fp8_dtype]["rtol"]
        quantizer = Float8BlockQuantizer(
            fp8_dtype=fp8_dtype,
            rowwise=True,
            columnwise=dq_columnwise,
            block_scaling_dim=block_scaling_dim,
        )
        self._test_quantize_dequantize(
            quantizer=quantizer,
            dims=dims,
            atol=atol,
            rtol=rtol,
            dequant_columnwise=dq_columnwise,
            use_cpp_allocation=True,
        )

    @pytest.mark.parametrize("dims", [[256, 512], [250, 500]])
    @pytest.mark.parametrize("block_scaling_dim", [1, 2])
    def test_data_accessors(self, dims: DimsType, block_scaling_dim: int) -> None:
        """Test data accessors of Float8BlockwiseQTensor"""
        device = "cuda"
        dtype = torch.bfloat16
        x_hp = torch.rand(_to_list(dims), dtype=dtype, device=device)
        y_hp = torch.rand(_to_list(dims), dtype=dtype, device=device)

        fp8_dtype = tex.DType.kFloat8E4M3
        quantizer = Float8BlockQuantizer(
            fp8_dtype=fp8_dtype,
            rowwise=True,
            columnwise=True,
            block_scaling_dim=block_scaling_dim,
        )

        # Create FP8 tensor
        x_fp8 = quantizer.quantize(x_hp)

        x_recovered = x_fp8.data
        torch.testing.assert_close(x_recovered, x_hp, **_tols[fp8_dtype])

        x_fp8.data = y_hp
        y_recovered = x_fp8.data
        torch.testing.assert_close(y_recovered, y_hp, **_tols[fp8_dtype])

    @pytest.mark.parametrize("dims", [[256, 512], [250, 500]])
    @pytest.mark.parametrize("block_scaling_dim", [1, 2])
    @pytest.mark.parametrize("all_gather_usage", [True, False])
    def test_serialization(
        self, dims: DimsType, block_scaling_dim: int, all_gather_usage: bool
    ) -> None:
        """Test serialization of Float8BlockwiseQTensor"""
        if all_gather_usage and block_scaling_dim != 1:
            pytest.skip("all_gather_usage only implemented for 1D block quantization.")
        device = "cuda"
        dtype = torch.bfloat16
        x_hp = torch.rand(_to_list(dims), dtype=dtype, device=device)
        quantizer = Float8BlockQuantizer(
            fp8_dtype=tex.DType.kFloat8E5M2,
            rowwise=True,
            columnwise=True,
            block_scaling_dim=block_scaling_dim,
            all_gather_usage=all_gather_usage,
        )

        # Create FP8 tensor
        x_fp8 = quantizer.quantize(x_hp)

        # Save tensor
        buffer = io.BytesIO()
        torch.save(x_fp8, buffer)

        # Load tensor
        buffer.seek(0)
        x_fp8_loaded = torch.load(buffer, weights_only=False)

        # Test that loaded tensor matches original
        assert isinstance(x_fp8_loaded, Float8BlockwiseQTensor)
        torch.testing.assert_close(x_fp8_loaded._rowwise_data, x_fp8._rowwise_data)
        torch.testing.assert_close(x_fp8_loaded._columnwise_data, x_fp8._columnwise_data)
        torch.testing.assert_close(x_fp8_loaded._rowwise_scale_inv, x_fp8._rowwise_scale_inv)
        torch.testing.assert_close(x_fp8_loaded._columnwise_scale_inv, x_fp8._columnwise_scale_inv)
        torch.testing.assert_close(x_fp8_loaded.data, x_fp8.data)
        assert x_fp8_loaded._is_2D_scaled == x_fp8._is_2D_scaled
        assert x_fp8_loaded.dtype == x_fp8.dtype
        assert x_fp8_loaded._fp8_dtype == x_fp8._fp8_dtype
        assert x_fp8_loaded._data_format == x_fp8._data_format

        # Test that dequantized values match
        x_fp8_dequant = x_fp8.dequantize()
        x_fp8_loaded_dequant = x_fp8_loaded.dequantize()
        torch.testing.assert_close(x_fp8_loaded_dequant, x_fp8_dequant)

    @pytest.mark.parametrize("fp8_dtype", [tex.DType.kFloat8E4M3], ids=str)
    @pytest.mark.parametrize("dtype", [torch.bfloat16], ids=str)
    @pytest.mark.parametrize("dims", [[256, 512], [250, 500]])
    @pytest.mark.parametrize("block_scaling_dim", [1, 2])
    def test_inplace_ops(
        self, fp8_dtype: tex.DType, dtype: torch.dtype, dims: DimsType, block_scaling_dim: int
    ) -> None:
        """Test in-place operations"""
        device = "cuda"
        x_hp = torch.rand(_to_list(dims), dtype=dtype, device=device)
        y_hp = torch.rand(_to_list(dims), dtype=dtype, device=device)

        quantizer = Float8BlockQuantizer(
            fp8_dtype=fp8_dtype,
            rowwise=True,
            columnwise=True,
            block_scaling_dim=block_scaling_dim,
        )

        # Test in-place add
        x_fp8 = quantizer.quantize(x_hp.clone())
        y_fp8 = quantizer.quantize(y_hp.clone())
        x_fp8.add_(y_fp8)
        torch.testing.assert_close(x_fp8.dequantize(), x_hp + y_hp, **_tols[fp8_dtype])

        # Test in-place subtract
        x_fp8 = quantizer.quantize(x_hp.clone())
        y_fp8 = quantizer.quantize(y_hp.clone())
        x_fp8.sub_(y_fp8)
        torch.testing.assert_close(x_fp8.dequantize(), x_hp - y_hp, **_tols[fp8_dtype])

        # Test in-place multiply
        x_fp8 = quantizer.quantize(x_hp.clone())
        y_fp8 = quantizer.quantize(y_hp.clone())
        x_fp8.mul_(y_fp8)
        torch.testing.assert_close(x_fp8.dequantize(), x_hp * y_hp, **_tols[fp8_dtype])

    @pytest.mark.parametrize("fp8_dtype", [tex.DType.kFloat8E4M3], ids=str)
    @pytest.mark.parametrize("dtype", [torch.bfloat16], ids=str)
    @pytest.mark.parametrize("dims", [[256, 512], [250, 500]])
    @pytest.mark.parametrize("block_scaling_dim", [1, 2])
    def test_out_of_place_ops(
        self, fp8_dtype: tex.DType, dtype: torch.dtype, dims: DimsType, block_scaling_dim: int
    ) -> None:
        """Test out-of-place operations"""
        device = "cuda"
        x_hp = torch.rand(_to_list(dims), dtype=dtype, device=device)
        y_hp = torch.rand(_to_list(dims), dtype=dtype, device=device)

        quantizer = Float8BlockQuantizer(
            fp8_dtype=fp8_dtype,
            rowwise=True,
            columnwise=True,
            block_scaling_dim=block_scaling_dim,
        )

        x_fp8 = quantizer.quantize(x_hp.clone())
        y_fp8 = quantizer.quantize(y_hp.clone())

        # Test exact operations
        torch.testing.assert_close(-x_fp8, -x_hp, **_tols[fp8_dtype])
        torch.testing.assert_close(x_fp8.abs(), x_hp.abs(), **_tols[fp8_dtype])

        # Test elementwise operations
        torch.testing.assert_close(x_fp8 + y_fp8, x_hp + y_hp, **_tols[fp8_dtype])
        torch.testing.assert_close(x_fp8 - y_fp8, x_hp - y_hp, **_tols[fp8_dtype])
        torch.testing.assert_close(x_fp8 * y_fp8, x_hp * y_hp, **_tols[fp8_dtype])
        torch.testing.assert_close(torch.sin(x_fp8), torch.sin(x_hp), **_tols[fp8_dtype])

        # Make sure we are not trivially passing tests
        with pytest.raises(AssertionError):
            torch.testing.assert_close(x_fp8 + y_fp8, x_hp - y_hp, **_tols[fp8_dtype])

    @pytest.mark.parametrize("fp8_dtype", [tex.DType.kFloat8E4M3], ids=str)
    @pytest.mark.parametrize("dtype", [torch.bfloat16], ids=str)
    @pytest.mark.parametrize("dims", [[256, 512], [250, 500]])
    @pytest.mark.parametrize("block_scaling_dim", [1, 2])
    def test_view_same_shape(
        self, fp8_dtype: tex.DType, dtype: torch.dtype, dims: DimsType, block_scaling_dim: int
    ) -> None:
        """Test view operations that preserve tensor shape"""
        device = "cuda"
        x_hp = torch.rand(_to_list(dims), dtype=dtype, device=device)

        quantizer = Float8BlockQuantizer(
            fp8_dtype=fp8_dtype,
            rowwise=True,
            columnwise=True,
            block_scaling_dim=block_scaling_dim,
        )

        x_fp8 = quantizer.make_empty(x_hp.shape, dtype=dtype, device=device)
        quantizer.update_quantized(x_hp.clone(), x_fp8)

        # Test view with same shape
        x_view = x_fp8.view(*dims)
        torch.testing.assert_close(x_view.dequantize(), x_hp, **_tols[fp8_dtype])
        assert x_view.shape == x_fp8.shape, "Shape changed after view with same dims"

        # Make sure we are not trivially passing tests
        with pytest.raises(AssertionError):
            torch.testing.assert_close(x_view.dequantize(), -x_hp, **_tols[fp8_dtype])

    @pytest.mark.parametrize("fp8_dtype", [tex.DType.kFloat8E4M3, tex.DType.kFloat8E5M2], ids=str)
    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=str)
    @pytest.mark.parametrize(
        "dims", [[16, 16, 512], [16, 16, 512, 16], [12, 7, 11], [13, 14, 16], [2, 3, 5]]
    )
    def test_view_and_reshape_1D(
        self, fp8_dtype: tex.DType, dtype: torch.dtype, dims: List[int]
    ) -> None:
        """Test view operations that preserve tensor shape"""
        device = "cuda"

        def is_bitwise_equal(a, b):
            if a.numel() != b.numel():
                return False
            a_flat = a.reshape(-1).view(torch.uint8)
            b_flat = b.reshape(-1).view(torch.uint8)
            return torch.all((a_flat ^ b_flat) == 0)

        x_hp = torch.rand(dims, dtype=dtype, device=device)
        quantizer = Float8BlockQuantizer(
            fp8_dtype=fp8_dtype,
            rowwise=True,
            columnwise=True,
            block_scaling_dim=1,
        )
        x_fp8 = quantizer.make_empty(x_hp.shape, dtype=dtype, device=device)
        quantizer.update_quantized(x_hp.clone(), x_fp8)

        # Test view, high dimension tensor -> 2D tensor
        x_hp_view = x_hp.view(-1, dims[-1]).contiguous()
        x_fp8_view = x_fp8.view(-1, dims[-1])
        # Check the dequantized result
        torch.testing.assert_close(
            x_fp8_view.dequantize().contiguous(), x_hp_view, **_tols[fp8_dtype]
        )
        # Check the bitwise equality of the inner data
        assert is_bitwise_equal(x_fp8_view._rowwise_data, x_fp8._rowwise_data)
        assert is_bitwise_equal(x_fp8_view._rowwise_scale_inv, x_fp8._rowwise_scale_inv)
        # Check the data ptr
        assert x_fp8_view._rowwise_data.data_ptr() == x_fp8._rowwise_data.data_ptr()
        assert x_fp8_view._rowwise_scale_inv.data_ptr() == x_fp8._rowwise_scale_inv.data_ptr()

        # Test reshape high dimension tensor -> 2D tensor
        x_hp_reshape = x_hp.reshape(-1, dims[-1]).contiguous()
        x_fp8_reshape = x_fp8.reshape(-1, dims[-1])
        # Check the dequantized result
        torch.testing.assert_close(
            x_fp8_reshape.dequantize().contiguous(), x_hp_reshape, **_tols[fp8_dtype]
        )
        # Check the bitwise equality of the inner data
        assert is_bitwise_equal(x_fp8_reshape._rowwise_data, x_fp8._rowwise_data)
        assert is_bitwise_equal(x_fp8_reshape._rowwise_scale_inv, x_fp8._rowwise_scale_inv)

    @pytest.mark.parametrize("fp8_dtype", [tex.DType.kFloat8E4M3, tex.DType.kFloat8E5M2], ids=str)
    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=str)
    @pytest.mark.parametrize("dims", [[16, 16, 512, 16], [2, 512, 512, 128], [3, 13, 14, 16]])
    def test_view_and_reshape_2D(
        self, fp8_dtype: tex.DType, dtype: torch.dtype, dims: List[int]
    ) -> None:
        """Test view operations that preserve tensor shape"""
        device = "cuda"

        def is_bitwise_equal(a, b):
            if a.numel() != b.numel():
                return False
            a_flat = a.reshape(-1).view(torch.uint8)
            b_flat = b.reshape(-1).view(torch.uint8)
            return torch.all((a_flat ^ b_flat) == 0)

        x_hp = torch.rand(dims, dtype=dtype, device=device)
        quantizer = Float8BlockQuantizer(
            fp8_dtype=fp8_dtype,
            rowwise=True,
            columnwise=True,
            block_scaling_dim=2,
        )
        x_fp8 = quantizer.make_empty(x_hp.shape, dtype=dtype, device=device)
        quantizer.update_quantized(x_hp.clone(), x_fp8)

        # Test view, high dimension tensor -> 2D tensor
        x_hp_view = x_hp.view(-1, dims[-2], dims[-1]).contiguous()
        x_fp8_view = x_fp8.view(-1, dims[-2], dims[-1])
        # Check the dequantized result
        torch.testing.assert_close(
            x_fp8_view.dequantize().contiguous(), x_hp_view, **_tols[fp8_dtype]
        )
        # Check the bitwise equality of the inner data
        assert is_bitwise_equal(x_fp8_view._rowwise_data, x_fp8._rowwise_data)
        assert is_bitwise_equal(x_fp8_view._rowwise_scale_inv, x_fp8._rowwise_scale_inv)
        # Check the data ptr
        assert x_fp8_view._rowwise_data.data_ptr() == x_fp8._rowwise_data.data_ptr()
        assert x_fp8_view._rowwise_scale_inv.data_ptr() == x_fp8._rowwise_scale_inv.data_ptr()

        # Test reshape high dimension tensor -> 2D tensor
        x_hp_reshape = x_hp.reshape(-1, dims[-2], dims[-1]).contiguous()
        x_fp8_reshape = x_fp8.reshape(-1, dims[-2], dims[-1])
        # Check the dequantized result
        torch.testing.assert_close(
            x_fp8_reshape.dequantize().contiguous(), x_hp_reshape, **_tols[fp8_dtype]
        )
        # Check the bitwise equality of the inner data
        assert is_bitwise_equal(x_fp8_reshape._rowwise_data, x_fp8._rowwise_data)
        assert is_bitwise_equal(x_fp8_reshape._rowwise_scale_inv, x_fp8._rowwise_scale_inv)

    @pytest.mark.parametrize("fp8_dtype", [tex.DType.kFloat8E4M3], ids=str)
    @pytest.mark.parametrize("dtype", [torch.bfloat16], ids=str)
    @pytest.mark.parametrize("dims", [[256, 512], [250, 500]])
    @pytest.mark.parametrize("block_scaling_dim", [1, 2])
    def test_reshape_same_shape(
        self, fp8_dtype: tex.DType, dtype: torch.dtype, dims: DimsType, block_scaling_dim: int
    ) -> None:
        """Test reshape operations that preserve tensor shape"""
        device = "cuda"
        x_hp = torch.rand(_to_list(dims), dtype=dtype, device=device)

        quantizer = Float8BlockQuantizer(
            fp8_dtype=fp8_dtype,
            rowwise=True,
            columnwise=True,
            block_scaling_dim=block_scaling_dim,
        )

        x_fp8 = quantizer.make_empty(x_hp.shape, dtype=dtype, device=device)
        quantizer.update_quantized(x_hp.clone(), x_fp8)

        # Test reshape with same shape
        x_reshape = x_fp8.reshape(*dims)
        torch.testing.assert_close(x_reshape.dequantize(), x_hp, **_tols[fp8_dtype])
        assert x_reshape.shape == x_fp8.shape, "Shape changed after reshape with same dims"

        # Test reshape with -1 canonicalization
        new_dims = [-1, dims[1]]
        x_reshape = x_fp8.reshape(*new_dims)
        torch.testing.assert_close(x_reshape.dequantize(), x_hp, **_tols[fp8_dtype])
        assert x_reshape.shape == x_fp8.shape, "Shape changed after reshape with -1"

        # Make sure we are not trivially passing tests
        with pytest.raises(AssertionError):
            torch.testing.assert_close(x_reshape.dequantize(), -x_hp, **_tols[fp8_dtype])

    @pytest.mark.parametrize("fp8_dtype", [tex.DType.kFloat8E4M3], ids=str)
    @pytest.mark.parametrize("dtype", [torch.bfloat16], ids=str)
    @pytest.mark.parametrize("dims", [[256, 512], [250, 500]])
    @pytest.mark.parametrize("block_scaling_dim", [1, 2])
    def test_clone_detach(
        self, fp8_dtype: tex.DType, dtype: torch.dtype, dims: DimsType, block_scaling_dim: int
    ) -> None:
        """Test clone and detach operations"""
        device = "cuda"
        x_hp = torch.rand(_to_list(dims), dtype=dtype, device=device)

        quantizer = Float8BlockQuantizer(
            fp8_dtype=fp8_dtype,
            rowwise=True,
            columnwise=True,
            block_scaling_dim=block_scaling_dim,
        )

        x_fp8 = quantizer.quantize(x_hp.clone())

        # Test clone
        x_clone = x_fp8.clone()
        torch.testing.assert_close(x_clone.dequantize(), x_hp, **_tols[fp8_dtype])
        assert x_clone.shape == x_fp8.shape, "Shape changed after clone"

        # Test detach
        x_detach = x_fp8.detach()
        torch.testing.assert_close(x_detach.dequantize(), x_hp, **_tols[fp8_dtype])
        assert x_detach.shape == x_fp8.shape, "Shape changed after detach"

        # Make sure we are not trivially passing tests
        with pytest.raises(AssertionError):
            torch.testing.assert_close(x_clone.dequantize(), -x_hp, **_tols[fp8_dtype])
