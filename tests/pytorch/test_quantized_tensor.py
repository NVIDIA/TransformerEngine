# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

from collections.abc import Iterable
import io
from typing import Any, Dict, List, Tuple, Union, Optional

import pytest
import torch

import transformer_engine.common.recipe
import transformer_engine.pytorch as te
from transformer_engine.pytorch import (
    Float8Quantizer,
    Float8CurrentScalingQuantizer,
    Float8BlockQuantizer,
    MXFP8Quantizer,
    NVFP4Quantizer,
    Float8Tensor,
    MXFP8Tensor,
    NVFP4Tensor,
    QuantizedTensor,
)

from transformer_engine.pytorch.utils import is_non_tn_fp8_gemm_supported
import transformer_engine_torch as tex

from references.ref_per_tensor_cs import ref_per_tensor_cs_cast

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

# Supported quantization recipes
fp8_available, reason_for_no_fp8 = te.is_fp8_available(return_reason=True)
fp8_block_scaling_available, reason_for_no_fp8_block_scaling = te.is_fp8_block_scaling_available(
    return_reason=True
)
mxfp8_available, reason_for_no_mxfp8 = te.is_mxfp8_available(return_reason=True)
nvfp4_available, reason_for_no_nvfp4 = te.is_nvfp4_available(return_reason=True)
_quantization_list: List[str] = []
if fp8_available:
    _quantization_list.append("fp8")
if fp8_block_scaling_available:
    _quantization_list.append("fp8_blockwise")
if mxfp8_available:
    _quantization_list.append("mxfp8")
if nvfp4_available:
    _quantization_list.append("nvfp4")


# delayed scaling
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


# current scaling
def to_float8_CS(
    tensor: torch.Tensor,
    fp8_dtype: tex.DType = tex.DType.kFloat8E4M3,
    return_transpose: bool = False,
    force_pow_2_scales: bool = False,
    amax_epsilon: float = 0.0,
) -> Float8Tensor:
    """Cast tensor to FP8"""
    tensor = tensor.cuda()
    quantizer = Float8CurrentScalingQuantizer(
        fp8_dtype=fp8_dtype,
        device=tensor.device,
        force_pow_2_scales=force_pow_2_scales,
        amax_epsilon=amax_epsilon,
    )
    if return_transpose:
        quantizer.set_usage(rowwise=True, columnwise=True)
    else:
        quantizer.set_usage(rowwise=True, columnwise=False)
    return quantizer(tensor)


@torch.no_grad()
def make_reference_and_test_tensors(
    shape: int | Iterable[int],
    quantization: Optional[str] = None,
    ref_dtype: torch.dtype = torch.float64,
    ref_device: torch.device = "cpu",
    test_dtype: torch.dtype = torch.float32,
    test_device: torch.device = "cuda",
    requires_grad: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Construct tensors with the same values

    The reference tensor is intended for use in plain PyTorch
    operations in high precision. The test tensor is intended for use
    in Transformer Engine operations.

    If a quantization scheme is provided, the tensor values are
    quantized so that they are representable.

    """

    # Random reference tensor
    ref = torch.rand(shape, dtype=ref_dtype, device=ref_device)

    # Construct test tensor from reference tensor
    test = ref.to(device=test_device, dtype=test_dtype)
    if quantization is None:
        if test.data_ptr() == ref.data_ptr():
            test = test.clone()
    elif quantization in ("fp8", "fp8_delayed_scaling"):
        quantizer = Float8Quantizer(
            scale=torch.ones(1, dtype=torch.float32, device=test_device).squeeze(),
            amax=torch.zeros(1, dtype=torch.float32, device=test_device),
            fp8_dtype=tex.DType.kFloat8E4M3,
        )
        test = quantizer(test)
    elif quantization == "fp8_current_scaling":
        quantizer = Float8CurrentScalingQuantizer(
            fp8_dtype=tex.DType.kFloat8E4M3,
            device=test_device,
        )
        test = quantizer(test)
    elif quantization == "fp8_blockwise":
        quantizer = Float8BlockQuantizer(
            fp8_dtype=tex.DType.kFloat8E4M3,
            rowwise=True,
            columnwise=True,
            force_pow_2_scales=True,
            amax_epsilon=0.0,
            block_scaling_dim=1,
        )
        test = quantizer(test)
    elif quantization == "mxfp8":
        test = MXFP8Quantizer(fp8_dtype=tex.DType.kFloat8E4M3)(test)
    elif quantization == "nvfp4":
        test = NVFP4Quantizer(
            with_rht=False,
            with_post_rht_amax=False,
            with_2d_quantization=False,
            stochastic_rounding=False,
            with_random_sign_mask=False,
        )(test)
    else:
        raise ValueError(f"Unsupported quantization scheme ({quantization})")

    # Make sure reference and test tensors match each other
    ref.copy_(test)

    ref.requires_grad_(requires_grad)
    test.requires_grad_(requires_grad)
    return ref, test


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

    @pytest.mark.parametrize("fp8_dtype", _fp8_dtypes)
    @pytest.mark.parametrize("dtype", _dtypes)
    @pytest.mark.parametrize("noop", [True, False])
    def test_quantize_dequantize_noop(
        self, fp8_dtype: tex.DType, dtype: torch.dtype, noop: bool
    ) -> None:
        noop_tensor = torch.zeros(1, dtype=torch.float32, device="cuda")
        if noop:
            noop_tensor = torch.ones(1, dtype=torch.float32, device="cuda")
        dims = 23
        scale: float = 3.5

        # Initialize random data
        x_ref = 2 * torch.rand(_to_list(dims), dtype=dtype, device="cpu") - 1

        # Cast to FP8 and back
        x_fp8 = to_float8(x_ref, fp8_dtype=fp8_dtype, scale=scale)
        # if noop, then when we input a different tensor, output should still be x_fp8_orig
        x_ref_noop_test = 2 * x_ref.cuda()
        x_fp8_orig = x_fp8.clone()
        x_fp8.quantize_(x_ref_noop_test, noop_flag=noop_tensor)
        if noop_tensor.item() == 1.0:
            torch.testing.assert_close(x_fp8, x_fp8_orig, atol=0, rtol=0)
        else:
            torch.testing.assert_close(x_fp8, x_ref_noop_test, **_tols[fp8_dtype])

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

    @pytest.mark.parametrize("dims", [2, [4, 4], [8, 5, 3, 3]])
    def test_chunk_op(
        self,
        dims: DimsType,
        fp8_dtype: tex.DType = tex.DType.kFloat8E4M3,
        scale: float = 3.5,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        """Test for ops for which shape of inputs and outputs differ."""

        # Initialize random data
        dims = _to_list(dims)
        x_ref = torch.randn(dims, dtype=dtype, device="cpu")
        x_fp8 = to_float8(x_ref, fp8_dtype=fp8_dtype, scale=1.0)

        # Get chunks.
        chunk1, chunk2 = x_fp8.chunk(2, dim=0)

        # Test chunks.
        torch.testing.assert_close(x_fp8[0 : dims[0] // 2,], chunk1, atol=0, rtol=0)
        torch.testing.assert_close(x_fp8[dims[0] // 2 :,], chunk2, atol=0, rtol=0)

        # Check shapes.
        assert (
            chunk1.shape == torch.Size([x_fp8.shape[0] // 2]) + x_fp8.shape[1:]
        ), "Wrong shape for chunk1"
        assert (
            chunk2.shape == torch.Size([x_fp8.shape[0] // 2]) + x_fp8.shape[1:]
        ), "Wrong shape for chunk2"

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


@pytest.mark.skipif(not fp8_available, reason=reason_for_no_fp8)
class TestCurrentScalingFloat8Tensor:

    @staticmethod
    def setup_class(cls) -> None:
        # Configure RNG
        seed = 1234
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

    @pytest.mark.parametrize("fp8_dtype", _fp8_dtypes)
    @pytest.mark.parametrize("dtype", _dtypes)
    @pytest.mark.parametrize(
        "dims", [[], 1, 311, [7, 11], [7, 5, 3], [2, 3, 5, 3], [128, 128], [611, 782]]
    )
    @pytest.mark.parametrize("return_transpose", [True, False], ids=str)
    @pytest.mark.parametrize("force_pow_2_scales", [True, False], ids=str)
    @pytest.mark.parametrize("amax_epsilon", [0.0, 1e-6], ids=str)
    def test_quantize(
        self,
        fp8_dtype: tex.DType,
        dtype: torch.dtype,
        dims: DimsType,
        return_transpose: bool,
        force_pow_2_scales: bool,
        amax_epsilon: float,
    ) -> None:
        """Check numerical error when casting to FP8"""

        # Skip invalid configurations
        if is_non_tn_fp8_gemm_supported() and return_transpose:
            pytest.skip("FP8 transpose is neither needed nor supported on current system")

        # Initialize random high precision data
        device = "cuda"
        x_hp = 2 * torch.rand(_to_list(dims), dtype=dtype, device=device) - 1

        # Cast to FP8 and back
        x_fp8 = to_float8_CS(
            x_hp,
            fp8_dtype=fp8_dtype,
            return_transpose=return_transpose,
            force_pow_2_scales=force_pow_2_scales,
            amax_epsilon=amax_epsilon,
        )

        # get reference implementation of current scaling
        x_fp8_ref, sx_ref, x_fp8_t_ref, _ = ref_per_tensor_cs_cast(
            x_hp,
            fp8_dtype=fp8_dtype,
            return_transpose=return_transpose,
            force_pow_2_scales=force_pow_2_scales,
            amax_epsilon=amax_epsilon,
        )

        torch.testing.assert_close(x_fp8._data, x_fp8_ref.view(torch.uint8), atol=0.0, rtol=0.0)
        torch.testing.assert_close(x_fp8._scale_inv, sx_ref, atol=0.0, rtol=0.0)
        if return_transpose:
            torch.testing.assert_close(
                x_fp8._transpose, x_fp8_t_ref.view(torch.uint8), atol=0.0, rtol=0.0
            )

    @pytest.mark.parametrize("fp8_dtype", [tex.DType.kFloat8E4M3], ids=str)
    @pytest.mark.parametrize("dtype", [torch.bfloat16], ids=str)
    @pytest.mark.parametrize("dims", [[], 1, 311, [7, 11], [7, 5, 3], [2, 3, 5, 3]])
    def test_quantize_dequantize(
        self, fp8_dtype: tex.DType, dtype: torch.dtype, dims: DimsType
    ) -> None:
        """Check numerical error when casting to FP8 and back"""

        # Initialize random high precision data
        device = "cuda"
        x_hp = 2 * torch.rand(_to_list(dims), dtype=dtype, device=device) - 1

        # Cast to FP8 and back
        x_fp8 = to_float8_CS(x_hp, fp8_dtype=fp8_dtype)
        x_fp8_dequantized = x_fp8.dequantize()

        # Check results
        torch.testing.assert_close(x_fp8_dequantized, x_hp, **_tols[fp8_dtype])

        # Make sure we are not trivially passing the test
        with pytest.raises(AssertionError):
            torch.testing.assert_close(x_fp8_dequantized, -x_hp, **_tols[fp8_dtype])


class TestQuantizedTensor:
    @staticmethod
    def setup_class(cls) -> None:
        # Configure RNG
        seed = 1234
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

    @pytest.mark.parametrize("op", ("clone", "view", "reshape", "contiguous"))
    @pytest.mark.parametrize("quantization", _quantization_list)
    def test_identity_op(
        self,
        *,
        op: str,
        quantization: str,
        shape: Iterable[int] = (128, 128),
        dtype: torch.dtype = torch.bfloat16,
        device: torch.device = "cuda",
    ) -> None:
        """Test operations that do not affect tensor values.

        These operations are must produce outputs that are bit-wise
        equivalent to the inputs. They must support autograd.

        """

        # Create reference and quantized tensor
        x_ref, x_test = make_reference_and_test_tensors(
            shape=shape,
            quantization=quantization,
            test_dtype=dtype,
        )
        dy_ref, dy_test = make_reference_and_test_tensors(
            shape=shape,
            test_dtype=dtype,
            requires_grad=False,
        )

        # Apply identity operation
        if op == "clone":
            y_ref = x_ref.clone()
            y_test = x_test.clone()
        elif op == "view":
            y_ref = x_ref.view(shape)
            y_test = x_test.view(shape)
        elif op == "reshape":
            y_ref = x_ref.reshape(shape)
            y_test = x_test.reshape(shape)
        elif op == "contiguous":
            y_ref = x_ref.contiguous()
            y_test = x_test.contiguous()

        # Check autograd
        y_test.backward(dy_test)
        assert x_test.grad is not None

        # Check values
        tols = dict(rtol=0, atol=0)
        if isinstance(y_test, QuantizedTensor):
            y_test = y_test.dequantize()
        y_test = y_test.to(dtype=torch.float64, device="cpu")
        dx_test = x_test.grad.to(dtype=torch.float64, device="cpu")
        dx_ref = dy_ref
        torch.testing.assert_close(y_test, y_ref, **tols)
        torch.testing.assert_close(dx_test, dx_ref, **tols)

    @pytest.mark.parametrize("quantization", _quantization_list)
    @pytest.mark.parametrize("dim", [0, 1])
    def test_chunk(
        self,
        *,
        quantization: str,
        dim: int,
        shape: Iterable[int] = (128, 128),
        chunks: int = 2,
        dtype: torch.dtype = torch.bfloat16,
        device: torch.device = "cuda",
    ) -> None:

        # Create reference and quantized tensor
        x_ref, x_test = make_reference_and_test_tensors(
            shape=shape,
            quantization=quantization,
            test_dtype=dtype,
        )

        # Chunk tensors
        ys_ref = torch.chunk(x_ref, chunks, dim=dim)
        ys_test = torch.chunk(x_test, chunks, dim=dim)

        # Check splits
        for y_ref, y_test in zip(ys_ref, ys_test):

            # Check split shapes
            assert y_ref.size() == y_test.size()

            # Check that splits are quantized when expected
            if quantization == "fp8":
                assert isinstance(y_test, Float8Tensor)
                y_test = y_test.dequantize()
            elif quantization == "mxfp8" and dim == 0:
                assert isinstance(y_test, MXFP8Tensor)
                y_test = y_test.dequantize()

            # Check values
            tols = dict(rtol=0, atol=0)  # Chunking is exact
            y_test = y_test.to(dtype=torch.float64, device="cpu")
            torch.testing.assert_close(y_test, y_ref, **tols)
