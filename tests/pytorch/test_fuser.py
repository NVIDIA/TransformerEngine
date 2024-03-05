# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

from __future__ import annotations

import math

import pytest
import torch

import transformer_engine.pytorch as te
import transformer_engine.pytorch.fuser as te_fuser
from transformer_engine.pytorch.fuser.ops._common import is_float8_tensor
from transformer_engine.pytorch.fp8 import FP8GlobalStateManager
from transformer_engine.pytorch.float8_tensor import Float8Tensor
from transformer_engine.pytorch.utils import is_bf16_compatible
import transformer_engine_extensions as tex

# Check if FP8 is supported
fp8_available, reason_for_no_fp8 = FP8GlobalStateManager.is_fp8_available()

# Supported data types
_dtypes: list[torch.dtype] = [torch.float32, torch.float16]
if is_bf16_compatible():  # bf16 requires sm_80 or higher
    _dtypes.append(torch.bfloat16)

# Supported devices
_devices: list[torch.device] = [torch.device("cpu"), torch.device("cuda")]

def dtype_tols(dtype: torch.dtype | tex.DType) -> dict[str, float]:
    """Estimated numerical error for a datatype

    Based on tolerances for torch.testing.assert_close.

    """

    # Transformer Engine dtypes
    if isinstance(dtype, tex.DType):
        if dtype == tex.DType.kFloat8E4M3:
            return dict(rtol=0.125, atol=0.0675)  # epsilon = 0.0625
        if dtype == tex.DType.kFloat8E5M2:
            return dict(rtol=0.25, atol=0.125)  # epsilon = 0.152
        dtype = {
            tex.DType.kByte: torch.uint8,
            tex.DType.kInt32: torch.int32,
            tex.DType.kFloat32: torch.float32,
            tex.DType.kFloat16: torch.half,
            tex.DType.kBFloat16: torch.bfloat16,
        }[dtype]

    # PyTorch dtypes
    if dtype == torch.float16:
        return dict(rtol=1e-3, atol=1e-5)
    if dtype == torch.bfloat16:
        return dict(rtol=1.6e-2, atol=1e-5)
    if dtype == torch.float32:
        return dict(rtol=1.3e-6, atol=1e-5)
    if dtype == torch.float64:
        return dict(rtol=1e-7, atol=1e-7)
    raise ValueError(f"Unsupported dtype ({dtype})")


class TestFuserOps:

    @staticmethod
    def setup_class(cls) -> None:
        # Configure RNG
        seed = 1234
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

    @pytest.mark.parametrize(
        "shapes",
        (
            ((1,2,3,4), (2,12)),
            ((5,4,3,2), (-1,6)),
            ((30,), (2, 3, -1)),
            ((6,7), (3, -1, 7)),
        ),
    )
    @pytest.mark.parametrize("dtype", _dtypes)
    @pytest.mark.parametrize("device", ("cuda", "cpu"))
    @pytest.mark.parametrize(
        "memory_format",
        (torch.contiguous_format, torch.channels_last),
    )
    @pytest.mark.parametrize("fp8", (False, True))
    def test_reshape(
        self,
        shapes: tuple[Iterable[int], Iterable[int]],
        dtype: torch.dtype,
        device: torch.device,
        memory_format: torch.memory_format,
        fp8: bool,
    ) -> None:
        """Reshape operation"""
        in_shape, out_shape = shapes

        # Skip invalid configurations
        if memory_format == torch.channels_last and len(in_shape) != 4:
            pytest.skip("torch.channels_last only supports 4D tensors")
        if fp8 and not fp8_available:
            pytest.skip(reason_for_no_fp8)
        if fp8 and torch.device(device).type != "cuda":
            pytest.skip("FP8 is only supported on CUDA devices")

        # Random data
        with torch.no_grad():
            x_ref = torch.rand(in_shape, dtype=dtype, device=device)
            x_ref = x_ref.to(memory_format=memory_format)
            x_test = x_ref.clone()
            if fp8:
                x_test = Float8Tensor.to_float8(x_test)
                x_test = x_test.contiguous(memory_format=memory_format)
                x_ref.copy_(x_test)
        x_ref.requires_grad_()
        x_test.requires_grad_()

        # Plain PyTorch implementation
        y_ref = x_ref.reshape(out_shape)
        dy = torch.rand_like(y_ref)
        y_ref.backward(dy)

        # Implementation with fusable operation
        op = te_fuser.ops.Reshape(out_shape)
        y_test = op(x_test)
        y_test.backward(dy)

        # Check results
        tols = dict(rtol=0, atol=0)  # Reshape is exact
        torch.testing.assert_close(y_test, y_ref, **tols)
        torch.testing.assert_close(x_test.grad, x_ref.grad, **tols)

    @pytest.mark.parametrize("size", (1, 7, 32))
    @pytest.mark.parametrize("in_shape", ([], [1, 3], [2, 3, 4]))
    @pytest.mark.parametrize("dtype", _dtypes)
    @pytest.mark.parametrize("device", _devices)
    @pytest.mark.parametrize("fp8", (False, True))
    def test_bias(
        self,
        size: int,
        in_shape: Iterable[int],
        dtype: torch.dtype,
        device: torch.device,
        fp8: bool,
    ) -> None:
        """Bias operation"""

        # Make input and bias shapes consistent
        in_shape = list(in_shape) + [size]

        # Skip invalid configurations
        if fp8 and not fp8_available:
            pytest.skip(reason_for_no_fp8)
        if fp8 and torch.device(device).type != "cuda":
            pytest.skip("FP8 is only supported on CUDA devices")

        # Random data
        with torch.no_grad():
            x_ref = torch.rand(in_shape, dtype=torch.float32, device="cpu")
            b_ref = torch.rand(size, dtype=torch.float32, device="cpu")
            x_test = x_ref.clone().to(device=device, dtype=dtype)
            if fp8:
                x_test = Float8Tensor.to_float8(x_test)
                x_ref.copy_(x_test)
            dy_ref = torch.rand_like(x_ref)
            dy_test = dy_ref.clone().to(device=device, dtype=dtype)
        x_ref.requires_grad_()
        b_ref.requires_grad_()
        x_test.requires_grad_()

        # Plain PyTorch implementation
        y_ref = x_ref + b_ref.reshape([1] * (len(in_shape) - 1) + [size])
        y_ref.backward(dy_ref)

        # Implementation with fusable operation
        op = te_fuser.ops.Bias(size, device=device, dtype=dtype)
        with torch.no_grad():
            op.bias.copy_(b_ref)
        y_test = op(x_test)
        y_test.backward(dy_test)

        # Check results
        y_ref = y_ref.to(dtype=dtype, device="cpu")
        dx_ref = x_ref.grad.to(dtype=dtype, device="cpu")
        db_ref = b_ref.grad.to(dtype=dtype, device="cpu")
        y_test = y_test.to(dtype=dtype, device="cpu")
        dx_test = x_test.grad.to(dtype=dtype, device="cpu")
        db_test = op.bias.grad.to(dtype=dtype, device="cpu")
        torch.testing.assert_close(y_test, y_ref)
        torch.testing.assert_close(dx_test, dx_ref)
        torch.testing.assert_close(db_test, db_ref)

    @pytest.mark.parametrize("weight_shape", ((48, 16), (3, 5)))
    @pytest.mark.parametrize("in_shape", ([], [5, 1], [2, 4, 4]))
    @pytest.mark.parametrize("dtype", _dtypes)
    @pytest.mark.parametrize("device", ("cuda",))
    @pytest.mark.parametrize("fp8_input", (False, True))
    @pytest.mark.parametrize("fp8_weight", (False, True))
    @pytest.mark.parametrize("fp8_compute", (False, True))
    def test_unfused_linear(
        self,
        weight_shape: tuple[int, int],
        in_shape: Iterable[int],
        dtype: torch.dtype,
        device: torch.device,
        fp8_input: bool,
        fp8_weight: bool,
        fp8_compute: bool,
    ) -> None:

        # Make input and weight shapes consistent
        out_features, in_features = weight_shape
        in_shape = list(in_shape) + [in_features]
        out_shape = in_shape[:-1] + [out_features]

        # Skip invalid configurations
        if fp8_input or fp8_weight or fp8_compute:
            if not fp8_available:
                pytest.skip(reason_for_no_fp8)
            if torch.device(device).type != "cuda":
                pytest.skip("FP8 is only supported on CUDA devices")
        if fp8_compute:
            if (
                math.prod(in_shape[:-1]) % 16 != 0
                or in_features % 16 != 0
                or out_features % 16 != 0
            ):
                pytest.skip("FP8 GEMMs require dims that are divisible by 16")

        # Random data
        with torch.no_grad():
            x_ref = torch.rand(in_shape, dtype=torch.float32, device="cpu")
            w_ref = torch.rand(
                out_features,
                in_features,
                dtype=torch.float32,
                device="cpu",
            )
            x_test = x_ref.clone().to(device=device, dtype=dtype)
            w_test = w_ref.clone().to(device=device, dtype=dtype)
            if fp8_input:
                x_test = Float8Tensor.to_float8(x_test)
                x_ref.copy_(x_test)
            if fp8_weight or fp8_compute:
                w_test = Float8Tensor.to_float8(w_test)
                w_ref.copy_(w_test)
            dy_ref = torch.rand(out_shape, dtype=torch.float32, device="cpu")
            dy_test = dy_ref.clone().to(device=device, dtype=dtype)
        x_ref.requires_grad_()
        w_ref.requires_grad_()
        x_test.requires_grad_()

        # Plain PyTorch implementation
        y_ref = torch.nn.functional.linear(x_ref, w_ref)
        y_ref.backward(dy_ref)

        # Implementation with fusable operation
        with te.fp8_model_init(enabled=fp8_weight):
            op = te_fuser.ops.UnfusedLinear(
                in_features,
                out_features,
                device=device,
                dtype=dtype,
            )
        with torch.no_grad():
            op.weight.copy_(w_test)
            del w_test
        with te.fp8_autocast(enabled=fp8_compute):
            y_test = op(x_test)
        y_test.backward(dy_test)

        # Expected numerical error
        tols = dtype_tols(dtype)
        if dtype == torch.float32:
            tols = dtype_tols(torch.float16)  # TF32 GEMM
        if fp8_compute:
            tols = dtype_tols(
                op.weight._fp8_dtype
                if is_float8_tensor(op.weight)
                else tex.DType.kFloat8E4M3
            )

        # Check results
        y_ref = y_ref.to(dtype=dtype, device="cpu")
        dx_ref = x_ref.grad.to(dtype=dtype, device="cpu")
        dw_ref = w_ref.grad.to(dtype=dtype, device="cpu")
        y_test = y_test.to(dtype=dtype, device="cpu")
        dx_test = x_test.grad.to(dtype=dtype, device="cpu")
        dw_test = op.weight.grad.to(dtype=dtype, device="cpu")
        torch.testing.assert_close(y_test, y_ref, **tols)
        torch.testing.assert_close(dx_test, dx_ref, **tols)
        torch.testing.assert_close(dw_test, dw_ref, **tols)
