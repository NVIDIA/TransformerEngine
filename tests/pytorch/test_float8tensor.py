# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

from collections.abc import Iterable
import io
from typing import Any, Dict, List, Tuple, Union

import pytest
import torch

import transformer_engine.common.recipe
import transformer_engine.pytorch as te
from transformer_engine.pytorch.float8_tensor import Float8Tensor
from transformer_engine.pytorch.fp8 import FP8GlobalStateManager
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
            data=torch.zeros(dims, device="cuda", dtype=torch.uint8),
            fp8_dtype=fp8_dtype,
            fp8_scale_inv=torch.full([1], scale_inv),
            dtype=dtype,
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
        x_fp8 = Float8Tensor.to_float8(
            x_ref,
            fp8_dtype=fp8_dtype,
            scale=torch.full([1], scale),
        )
        x_fp8 = x_fp8.from_float8().cpu()

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

    def test_fp8_meta(
        self,
        dtype: torch.dtype = torch.float32,
        dims: DimsType = 23,
    ) -> None:
        """Construct Float8Tensor using FP8 metadata and perform basic checks"""

        # Get FP8 metadata from linear module
        fp8_dtype = tex.DType.kFloat8E4M3
        recipe = transformer_engine.common.recipe.DelayedScaling(
            fp8_format=transformer_engine.common.recipe.Format.E4M3,
        )
        with te.fp8_autocast(enabled=True, fp8_recipe=recipe):
            module = te.Linear(32, 32)
            _ = module(torch.zeros([8, 32], device="cuda"))
        fp8_meta = module.fp8_meta
        fp8_meta_index = tex.FP8FwdTensors.GEMM1_WEIGHT
        fp8_meta_key = FP8GlobalStateManager.get_meta_tensor_key(forward=True)

        # Initialize random data
        dims = _to_list(dims)
        x_ref = 2 * torch.rand(dims, dtype=dtype, device="cpu") - 1

        # Make Float8Tensor
        x_fp8 = Float8Tensor.to_float8(
            x_ref,
            fp8_meta=fp8_meta,
            fp8_meta_index=fp8_meta_index,
        )
        x_ref = x_fp8.from_float8()
        assert list(x_fp8.size()) == dims, "Incorrect dims"
        assert x_fp8.dtype == dtype, "Incorrect nominal dtype"
        assert x_fp8.is_cuda, "Incorrect device"
        assert x_fp8._fp8_dtype == fp8_dtype, "Incorrect FP8 dtype"

        # Change FP8 metadata scale
        fp8_meta[fp8_meta_key].scale[fp8_meta_index] = 2
        fp8_meta[fp8_meta_key].scale_inv.fill_(123)

        # Check results
        torch.testing.assert_close(x_fp8, x_ref, **_tols[fp8_dtype])
        with pytest.raises(AssertionError):
            # Make sure we are not trivially passing the test
            torch.testing.assert_close(x_fp8, -x_ref, **_tols[fp8_dtype])

        # Check if scaling factor is updated after in-place ops
        x_fp8 += 0
        fp8_meta[fp8_meta_key].scale[fp8_meta_index] = 4
        fp8_meta[fp8_meta_key].scale_inv.fill_(321)
        assert x_fp8._scale_inv.item() == 0.5, "Incorrect FP8 scale_inv"
        torch.testing.assert_close(x_fp8, x_ref, **_tols[fp8_dtype])
        y = x_fp8.detach()
        y += 0
        assert x_fp8._scale_inv.item() == 0.25, "Incorrect FP8 scale_inv"
        torch.testing.assert_close(x_fp8, x_ref, **_tols[fp8_dtype])

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
        x_fp8 = Float8Tensor.to_float8(
            x_ref,
            fp8_dtype=fp8_dtype,
            scale=torch.full([1], scale),
        )
        y_fp8 = Float8Tensor.to_float8(
            y_ref,
            fp8_dtype=fp8_dtype,
            scale=torch.full([1], scale),
        )
        x_ref = x_fp8.from_float8()
        y_ref = y_fp8.from_float8()

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
        x_fp8 = Float8Tensor.to_float8(
            x_ref,
            fp8_dtype=fp8_dtype,
            scale=torch.full([1], scale),
        )
        y_fp8 = Float8Tensor.to_float8(
            y_ref,
            fp8_dtype=fp8_dtype,
            scale=torch.full([1], scale),
        )
        x_ref = x_fp8.from_float8()
        y_ref = y_fp8.from_float8()

        # In-place operations
        tols = _tols[fp8_dtype]
        x_fp8 += y_ref
        x_ref += y_ref
        torch.testing.assert_close(x_fp8, x_ref, **tols)
        x_ref = x_fp8.from_float8()
        x_fp8 -= y_fp8
        x_ref -= y_fp8
        torch.testing.assert_close(x_fp8, x_ref, **tols)
        x_ref = x_fp8.from_float8()
        x_fp8 *= 2
        x_ref *= 2
        torch.testing.assert_close(x_fp8, x_ref, **tols)
        x_ref = x_fp8.from_float8()

        # Make sure we are not trivially passing tests
        x_ref += 123
        with pytest.raises(AssertionError):
            torch.testing.assert_close(x_fp8, x_ref, **tols)

    @pytest.mark.parametrize("dims", [[33, 41], [7, 11]])
    def test_transpose(
        self,
        dims: DimsType,
        fp8_dtype: tex.DType = tex.DType.kFloat8E4M3,
        scale: float = 0.5,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        """Test transpose"""

        # Initialize random data
        dims = _to_list(dims)
        x = 2 * torch.rand(dims, dtype=dtype, device="cpu") - 1
        x_fp8 = Float8Tensor.to_float8(
            x,
            fp8_dtype=fp8_dtype,
            scale=torch.full([1], scale),
        )
        x = x_fp8.from_float8()

        # Perform transpose
        x_fp8_t = x_fp8.transpose_2d()
        x_t = x.transpose(0, 1)
        x_fp8_t = Float8Tensor.make_like(x_fp8, data=x_fp8_t)

        # Check results
        tols = dict(rtol=0, atol=0)
        torch.testing.assert_close(x_fp8_t, x_t, **tols)

        # Make sure we are not trivially passing the test
        with pytest.raises(AssertionError):
            torch.testing.assert_close(x_fp8_t, x, **tols)

        # Caching test.
        assert x_fp8._transpose_invalid, "Transpose cache must be invalid when not caching."
        x_fp8 += 0.5
        x = x_fp8.from_float8()
        x_fp8_t = Float8Tensor.make_like(x_fp8, data=x_fp8.transpose_2d(fill_cache=True))
        x_t = x.transpose(0, 1)
        torch.testing.assert_close(x_fp8_t, x_t, **tols)
        assert not x_fp8._transpose_invalid, "Transpose cache reset incorrectly."

        # Inplace update test.
        x_fp8 += 0.5
        assert x_fp8._transpose_invalid, "Transpose cache not invalidated properly."
        x = x_fp8.from_float8()
        x_fp8_t = Float8Tensor.make_like(x_fp8, data=x_fp8.transpose_2d(fill_cache=True))
        x_t = x.transpose(0, 1)
        torch.testing.assert_close(x_fp8_t, x_t, **tols)
        assert not x_fp8._transpose_invalid, "Transpose cache reset incorrectly."

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
        x_fp8 = Float8Tensor.to_float8(
            x_ref,
            fp8_dtype=fp8_dtype,
            scale=torch.full([1], scale),
        )
        x_ref = x_fp8.from_float8()

        # Serialize tensor
        byte_stream = io.BytesIO()
        torch.save(x_fp8, byte_stream)
        x_bytes = byte_stream.getvalue()

        # Mess up and delete old tensor
        x_fp8._data.zero_()
        x_fp8._scale_inv.zero_()
        del x_fp8, byte_stream

        # Deserialize tensor
        x_fp8 = torch.load(io.BytesIO(x_bytes))
        del x_bytes

        # Check results
        tols = dict(rtol=0, atol=0)
        torch.testing.assert_close(x_fp8, x_ref, **tols)

        # Make sure we are not trivially passing tests
        x_fp8._data.zero_()
        x_fp8._scale_inv.zero_()
        with pytest.raises(AssertionError):
            torch.testing.assert_close(x_fp8, x_ref, **tols)
