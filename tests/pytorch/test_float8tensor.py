# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

from collections.abc import Iterable
import io
from typing import Any, Dict, List, Tuple, Union

import pytest
import torch

import transformer_engine.common.recipe
import transformer_engine.pytorch as te
from transformer_engine.pytorch.fp8 import FP8GlobalStateManager
from transformer_engine.pytorch.tensor.float8_tensor import Float8Quantizer, Float8Tensor
import transformer_engine_torch as tex

# PyTorch tensor dtypes
_dtypes: List[torch.dtype] = [torch.float32, torch.float16, torch.bfloat16]
# TE FP8 dtypes
_fp8_dtypes: List[tex.DType] = [tex.DType.kFloat8E4M3, tex.DType.kFloat8E5M2]

# Numerical tolerances with FP8 types
_tols: Dict[tex.DType, Dict[str, float]] = {
    tex.DType.kFloat8E4M3: dict(rtol=0.125, atol=0.0675),  # epsilon = 0.0625
    tex.DType.kFloat8E5M2: dict(rtol=0.25, atol=0.125),  # epsilon = 0.125
}


def _to_list(x: Union[Iterable, Any]) -> List:
    """Convert to list if iterable, otherwise put in singleton list"""
    if isinstance(x, Iterable):
        return list(x)
    else:
        return [x]


# Types that can be interpreted as tensor dims
DimsType = Union[Iterable[int], int]

# Check if FP8 is supported
fp8_available, reason_for_no_fp8 = FP8GlobalStateManager.is_fp8_available()


def to_float8(
    tensor: torch.Tensor,
    fp8_dtype: tex.DType = tex.DType.kFloat8E4M3,
    scale: float = 1.0,
) -> Float8Tensor:
    """Cast tensor to FP8"""
    quantizer = Float8Quantizer(
        scale=torch.full([1], scale, dtype=torch.float32, device="cuda"),
        amax=torch.empty([1], dtype=torch.float32, device="cuda"),
        fp8_dtype=fp8_dtype,
    )
    return quantizer(tensor.cuda())


@pytest.mark.skipif(not fp8_available, reason=reason_for_no_fp8)
class TestFloat8Tensor:

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
        scale_inv: float = 0.375,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        """Call constructor and perform sanity checks"""
        dims = _to_list(dims)
        tensor = Float8Tensor(
            shape=dims,
            dtype=dtype,
            data=torch.zeros(dims, device="cuda", dtype=torch.uint8),
            fp8_dtype=fp8_dtype,
            fp8_scale_inv=torch.full([1], scale_inv),
        )
        assert list(tensor.size()) == dims, "Incorrect dims"
        assert tensor.dtype == dtype, "Incorrect nominal dtype"
        assert tensor.is_cuda, "Incorrect device"

    def _test_quantize_dequantize(
        self,
        fp8_dtype: tex.DType = tex.DType.kFloat8E4M3,
        scale: float = 3.5,
        dtype: torch.dtype = torch.float32,
        dims: DimsType = 23,
    ) -> None:
        """Check numerical error when casting to FP8 and back"""

        # Initialize random data
        x_ref = 2 * torch.rand(_to_list(dims), dtype=dtype, device="cpu") - 1

        # Cast to FP8 and back
        x_fp8 = to_float8(x_ref, fp8_dtype=fp8_dtype, scale=scale)
        x_fp8 = x_fp8.dequantize().cpu()

        # Check results
        torch.testing.assert_close(x_fp8, x_ref, **_tols[fp8_dtype])

        # Make sure we are not trivially passing the test
        with pytest.raises(AssertionError):
            torch.testing.assert_close(x_fp8, -x_ref, **_tols[fp8_dtype])

    @pytest.mark.parametrize("fp8_dtype", _fp8_dtypes)
    @pytest.mark.parametrize("dtype", _dtypes)
    def test_quantize_dequantize_dtypes(
        self,
        fp8_dtype: tex.DType,
        dtype: torch.dtype,
    ) -> None:
        self._test_quantize_dequantize(fp8_dtype=fp8_dtype, dtype=dtype)

    @pytest.mark.parametrize("scale", [0.375, 1, 3.5])
    def test_quantize_dequantize_scales(self, scale: float) -> None:
        self._test_quantize_dequantize(scale=scale)

    @pytest.mark.parametrize("dims", [[], 1, 311, [7, 11], [7, 5, 3], [2, 3, 5, 3]])
    def test_quantize_dequantize_dims(self, dims: DimsType) -> None:
        self._test_quantize_dequantize(dims=dims)

    def test_basic_ops(
        self,
        dims: DimsType = 23,
        fp8_dtype: tex.DType = tex.DType.kFloat8E4M3,
        scale: float = 3.5,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        """Test basic out-of-place ops"""

        # Initialize random data
        dims = _to_list(dims)
        x_ref = 2 * torch.rand(dims, dtype=dtype, device="cpu") - 1
        y_ref = 2 * torch.rand(dims, dtype=dtype, device="cpu") - 1
        x_fp8 = to_float8(x_ref, fp8_dtype=fp8_dtype, scale=scale)
        y_fp8 = to_float8(y_ref, fp8_dtype=fp8_dtype, scale=scale)
        x_ref = x_fp8.dequantize()
        y_ref = y_fp8.dequantize()

        # Exact operations
        torch.testing.assert_close(-x_fp8, -x_ref, rtol=0, atol=0)
        torch.testing.assert_close(x_fp8.abs(), x_ref.abs(), rtol=0, atol=0)

        # Operations with numerical error
        tols = _tols[fp8_dtype]
        torch.testing.assert_close(x_fp8 + y_fp8, x_ref + y_ref, **tols)
        torch.testing.assert_close(x_fp8 - y_fp8, x_ref - y_ref, **tols)
        torch.testing.assert_close(x_fp8 * y_fp8, x_ref * y_ref, **tols)
        torch.testing.assert_close(x_fp8 + y_ref, x_ref + y_ref, **tols)
        torch.testing.assert_close(x_ref + y_fp8, x_ref + y_ref, **tols)
        torch.testing.assert_close(torch.sin(x_fp8), torch.sin(x_ref), **tols)

        # Make sure we are not trivially passing tests
        with pytest.raises(AssertionError):
            torch.testing.assert_close(x_fp8 + y_fp8, x_ref - y_fp8, **tols)

    def test_inplace_ops(
        self,
        dims: DimsType = 23,
        fp8_dtype: tex.DType = tex.DType.kFloat8E4M3,
        scale: float = 3.5,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        """Test in-place ops"""

        # Initialize random data
        dims = _to_list(dims)
        x_ref = 2 * torch.rand(dims, dtype=dtype, device="cpu") - 1
        y_ref = 2 * torch.rand(dims, dtype=dtype, device="cpu") - 1
        x_fp8 = to_float8(x_ref, fp8_dtype=fp8_dtype, scale=scale)
        y_fp8 = to_float8(y_ref, fp8_dtype=fp8_dtype, scale=scale)
        x_ref = x_fp8.dequantize()
        y_ref = y_fp8.dequantize()

        # In-place operations
        tols = _tols[fp8_dtype]
        x_fp8 += y_ref
        x_ref += y_ref
        torch.testing.assert_close(x_fp8, x_ref, **tols)
        x_ref = x_fp8.dequantize()
        x_fp8 -= y_fp8
        x_ref -= y_fp8
        torch.testing.assert_close(x_fp8, x_ref, **tols)
        x_ref = x_fp8.dequantize()
        x_fp8 *= 2
        x_ref *= 2
        torch.testing.assert_close(x_fp8, x_ref, **tols)
        x_ref = x_fp8.dequantize()

        # Make sure we are not trivially passing tests
        x_ref += 123
        with pytest.raises(AssertionError):
            torch.testing.assert_close(x_fp8, x_ref, **tols)

    def test_serialization(
        self,
        dims: DimsType = [2, 3, 5],
        fp8_dtype: tex.DType = tex.DType.kFloat8E4M3,
        scale: float = 0.5,
        dtype: torch.dtype = torch.float32,
    ):

        # Initialize random data
        dims = _to_list(dims)
        x_ref = 2 * torch.rand(dims, dtype=dtype, device="cpu") - 1
        x_fp8 = to_float8(x_ref, fp8_dtype=fp8_dtype, scale=scale)
        x_ref = x_fp8.dequantize()

        # Serialize tensor
        byte_stream = io.BytesIO()
        torch.save(x_fp8, byte_stream)
        x_bytes = byte_stream.getvalue()

        # Mess up and delete old tensor
        x_fp8._data.zero_()
        x_fp8._scale_inv.zero_()
        del x_fp8, byte_stream

        # Deserialize tensor
        x_fp8 = torch.load(io.BytesIO(x_bytes), weights_only=False)
        del x_bytes

        # Check results
        tols = dict(rtol=0, atol=0)
        torch.testing.assert_close(x_fp8, x_ref, **tols)

        # Make sure we are not trivially passing tests
        x_fp8._data.zero_()
        x_fp8._scale_inv.zero_()
        with pytest.raises(AssertionError):
            torch.testing.assert_close(x_fp8, x_ref, **tols)

    def test_set_data(self):
        """Test directly setting .data attr"""

        # Initialize Float8Tensor
        x0 = torch.zeros(4, dtype=torch.float32)
        x = to_float8(x0)
        assert isinstance(x, Float8Tensor)
        assert x0.size() == x.size() == x._data.size()
        assert x.dtype == torch.float32
        assert x.is_cuda and x._data.is_cuda
        y = x.dequantize()
        assert not isinstance(y, Float8Tensor)
        assert x.size() == y.size()
        assert x.dtype == y.dtype
        assert x.device == y.device

        # Set data to plain tensor
        x0 = torch.zeros((3, 2), dtype=torch.float16, device=x.device)
        x.data = x0
        assert isinstance(x, Float8Tensor)
        assert x0.size() == x.size() == x._data.size()
        assert x0.dtype == x.dtype
        assert x0.device == x.device == x._data.device
        y = x.dequantize()
        assert not isinstance(y, Float8Tensor)
        assert x.size() == y.size()
        assert x.dtype == y.dtype
        assert x.device == y.device

        # Set data to Float8Tensor
        x0 = to_float8(torch.zeros((4, 3, 1), dtype=torch.float32))
        x.data = x0
        assert isinstance(x, Float8Tensor)
        assert x0.size() == x.size() == x._data.size()
        assert x0.dtype == x.dtype
        assert x0.device == x.device == x._data.device
        assert x0._data is x._data
        assert x0._scale_inv is x._scale_inv
        y = x.dequantize()
        assert not isinstance(y, Float8Tensor)
        assert x.size() == y.size()
        assert x.dtype == y.dtype
        assert x.device == y.device
