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
from transformer_engine.pytorch.fuser.ops.fused_forward import (
    ForwardLinearBiasActivation,
)
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

@torch.no_grad()
def make_reference_and_test_tensors(
    shape: int | Iterable[int],
    ref_dtype: torch.dtype = torch.float64,
    ref_device: torch.device = "cpu",
    test_dtype: torch.dtype = torch.float32,
    test_device: torch.device = "cuda",
    test_is_fp8: bool = False,
    requires_grad: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Construct tensors with the same values

    The reference tensor is intended for use in plain PyTorch
    operations in high precision. The test tensor is intended for use
    in Transformer Engine operations.

    """
    ref = torch.rand(shape, dtype=ref_dtype, device=ref_device)
    if test_is_fp8:
        test = Float8Tensor.to_float8(ref)
    else:
        test = ref.to(device=test_device, dtype=test_dtype)
        if test.data_ptr() == ref.data_ptr():
            test = test.clone()
    ref.copy_(test)
    ref.requires_grad_(requires_grad)
    test.requires_grad_(requires_grad)
    return ref, test


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
        *,
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
        x_ref, x_test = make_reference_and_test_tensors(
            in_shape,
            test_dtype=dtype,
            test_device=device,
            test_is_fp8=fp8,
        )
        x_test = x_test.contiguous(memory_format=memory_format)
        x_test = x_test.detach().requires_grad_()
        dy_ref, dy_test = make_reference_and_test_tensors(
            x_ref.reshape(out_shape).size(),
            test_dtype=dtype,
            test_device=device,
            requires_grad=False,
        )

        # Plain PyTorch implementation
        y_ref = x_ref.reshape(out_shape)
        y_ref.backward(dy_ref)

        # Implementation with fusable operation
        op = te_fuser.ops.Reshape(out_shape)
        y_test = op(x_test)
        y_test.backward(dy_test)

        # Check results
        tols = dict(rtol=0, atol=0)  # Reshape is exact
        y_test = y_test.to(
            dtype=torch.float64,
            device="cpu",
            memory_format=torch.contiguous_format,
        )
        dx_test = x_test.grad.to(
            dtype=torch.float64,
            device="cpu",
            memory_format=torch.contiguous_format,
        )
        torch.testing.assert_close(y_test, y_ref, **tols)
        torch.testing.assert_close(dx_test, x_ref.grad, **tols)

    @pytest.mark.parametrize("size", (1, 7, 32))
    @pytest.mark.parametrize("in_shape", ([], [1, 3], [2, 3, 4]))
    @pytest.mark.parametrize("dtype", _dtypes)
    @pytest.mark.parametrize("device", _devices)
    @pytest.mark.parametrize("fp8", (False, True))
    def test_bias(
        self,
        *,
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
        x_ref, x_test = make_reference_and_test_tensors(
            in_shape,
            test_dtype=dtype,
            test_device=device,
            test_is_fp8=fp8,
        )
        b_ref, b_test = make_reference_and_test_tensors(
            size,
            test_dtype=dtype,
            test_device=device,
        )
        dy_ref, dy_test = make_reference_and_test_tensors(
            in_shape,
            test_dtype=dtype,
            test_device=device,
            requires_grad=False,
        )

        # Plain PyTorch implementation
        y_ref = x_ref + b_ref.reshape([1] * (len(in_shape) - 1) + [size])
        y_ref.backward(dy_ref)

        # Implementation with fusable operation
        op = te_fuser.ops.Bias(size, device=device, dtype=dtype)
        with torch.no_grad():
            op.bias.copy_(b_test)
            del b_test
        y_test = op(x_test)
        y_test.backward(dy_test)

        # Check results
        tols = dtype_tols(dtype)
        y_test = y_test.to(dtype=torch.float64, device="cpu")
        dx_test = x_test.grad.to(dtype=torch.float64, device="cpu")
        db_test = op.bias.grad.to(dtype=torch.float64, device="cpu")
        torch.testing.assert_close(y_test, y_ref, **tols)
        torch.testing.assert_close(dx_test, x_ref.grad, **tols)
        torch.testing.assert_close(db_test, b_ref.grad, **tols)

    @pytest.mark.parametrize("weight_shape", ((48, 16), (3, 5)))
    @pytest.mark.parametrize("in_shape", ([], [5, 1], [2, 2, 4]))
    @pytest.mark.parametrize("dtype", _dtypes)
    @pytest.mark.parametrize("fp8_compute", (False, True))
    @pytest.mark.parametrize("fp8_input", (False, True))
    @pytest.mark.parametrize("fp8_weight", (False, True))
    @pytest.mark.parametrize("fp8_grad_output", (False, True))
    def test_unfused_linear(
        self,
        *,
        weight_shape: tuple[int, int],
        in_shape: Iterable[int],
        dtype: torch.dtype,
        device: torch.device = "cuda",
        fp8_compute: bool,
        fp8_input: bool,
        fp8_weight: bool,
        fp8_grad_output: bool,
    ) -> None:

        # Make input and weight shapes consistent
        out_features, in_features = weight_shape
        in_shape = list(in_shape) + [in_features]
        out_shape = in_shape[:-1] + [out_features]

        # Skip invalid configurations
        if fp8_compute or fp8_input or fp8_weight or fp8_grad_output:
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
        x_ref, x_test = make_reference_and_test_tensors(
            in_shape,
            test_dtype=dtype,
            test_device=device,
            test_is_fp8=(fp8_compute or fp8_input),
        )
        w_ref, w_test = make_reference_and_test_tensors(
            (out_features, in_features),
            test_dtype=dtype,
            test_device=device,
            test_is_fp8=(fp8_compute or fp8_weight),
        )
        dy_ref, dy_test = make_reference_and_test_tensors(
            out_shape,
            test_dtype=dtype,
            test_device=device,
            test_is_fp8=(fp8_compute or fp8_grad_output),
            requires_grad=False,
        )

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
        y_test = y_test.to(dtype=torch.float64, device="cpu")
        dx_test = x_test.grad.to(dtype=torch.float64, device="cpu")
        dw_test = op.weight.grad.to(dtype=torch.float64, device="cpu")
        torch.testing.assert_close(y_test, y_ref, **tols)
        torch.testing.assert_close(dx_test, x_ref.grad, **tols)
        torch.testing.assert_close(dw_test, w_ref.grad, **tols)

    @pytest.mark.parametrize("bias", (False, True))
    @pytest.mark.parametrize("fp8_compute", (False, True))
    @pytest.mark.parametrize("fp8_weight", (False, True))
    def test_linear(
        self,
        *,
        bias: bool,
        weight_shape: tuple[int, int] = (32, 32),
        in_shape: Iterable[int] = (32,),
        dtype: torch.dtype = torch.float32,
        device: torch.device = "cuda",
        fp8_compute: bool,
        fp8_input: bool = False,
        fp8_weight: bool,
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
        x_ref, x_test = make_reference_and_test_tensors(
            in_shape,
            test_dtype=dtype,
            test_device=device,
            test_is_fp8=(fp8_compute or fp8_input),
        )
        w_ref, w_test = make_reference_and_test_tensors(
            (out_features, in_features),
            test_dtype=dtype,
            test_device=device,
            test_is_fp8=(fp8_compute or fp8_weight),
        )
        if bias:
            b_ref, b_test = make_reference_and_test_tensors(
                out_features,
                test_dtype=dtype,
                test_device=device,
            )
        else:
            b_ref, b_test = None, None
        dy_ref, dy_test = make_reference_and_test_tensors(
            out_shape,
            test_dtype=dtype,
            test_device=device,
            requires_grad=False,
        )

        # Plain PyTorch implementation
        y_ref = torch.nn.functional.linear(x_ref, w_ref, bias=b_ref)
        y_ref.backward(dy_ref)

        # Implementation with fusable operation
        with te.fp8_model_init(enabled=fp8_weight):
            op = te_fuser.ops.Linear(
                in_features,
                out_features,
                bias=bias,
                device=device,
                dtype=dtype,
            )
        with torch.no_grad():
            op.weight.copy_(w_test)
            if bias:
                op.bias.copy_(b_test)
            del w_test
            del b_test
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
        y_test = y_test.to(dtype=torch.float64, device="cpu")
        dx_test = x_test.grad.to(dtype=torch.float64, device="cpu")
        dw_test = op.weight.grad.to(dtype=torch.float64, device="cpu")
        torch.testing.assert_close(y_test, y_ref, **tols)
        torch.testing.assert_close(dx_test, x_ref.grad, **tols)
        torch.testing.assert_close(dw_test, w_ref.grad, **tols)
        if bias:
            db_test = op.bias.grad.to(dtype=torch.float64, device="cpu")
            torch.testing.assert_close(db_test, b_ref.grad, **tols)

class TestFuserFusions:

    @staticmethod
    def setup_class(cls) -> None:
        # Configure RNG
        seed = 1234
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

    @pytest.mark.parametrize("weight_shape", ((32, 48), (3, 5)))
    @pytest.mark.parametrize("in_shape", ([], [1, 7], [4, 2, 10]))
    @pytest.mark.parametrize("dtype", _dtypes)
    @pytest.mark.parametrize("fp8_compute", (False, True))
    @pytest.mark.parametrize("fp8_input", (False, True))
    @pytest.mark.parametrize("fp8_weight", (False, True))
    def test_linear_bias_activation(
        self,
        *,
        bias: bool = True,
        weight_shape: tuple[int, int],
        in_shape: Iterable[int],
        dtype: torch.dtype,
        device: torch.device = "cuda",
        fp8_compute: bool,
        fp8_input: bool,
        fp8_weight: bool,
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
        x_ref, x_test = make_reference_and_test_tensors(
            in_shape,
            test_dtype=dtype,
            test_device=device,
            test_is_fp8=(fp8_compute or fp8_input),
        )
        w_ref, w_test = make_reference_and_test_tensors(
            (out_features, in_features),
            test_dtype=dtype,
            test_device=device,
            test_is_fp8=(fp8_compute or fp8_weight),
        )
        if bias:
            b_ref, b_test = make_reference_and_test_tensors(
                out_features,
                test_dtype=dtype,
                test_device=device,
            )
        else:
            b_ref, b_test = None, None
        dy_ref, dy_test = make_reference_and_test_tensors(
            out_shape,
            test_dtype=dtype,
            test_device=device,
            requires_grad=False,
        )

        # Plain PyTorch implementation
        y_ref = torch.nn.functional.linear(x_ref, w_ref, bias=b_ref)
        y_ref.backward(dy_ref)

        # Implementation with fusable operations
        with te.fp8_model_init(enabled=fp8_weight):
            model = te_fuser.Sequential(
                te_fuser.ops.Linear(
                    in_features,
                    out_features,
                    bias=bias,
                    device=device,
                    dtype=dtype,
                ),
            )
        with torch.no_grad():
            model[0].weight.copy_(w_test)
            if bias:
                model[0].bias.copy_(b_test)
            del w_test
            del b_test
        with te.fp8_autocast(enabled=fp8_compute):
            y_test = model(x_test)
        y_test.backward(dy_test)

        # Check that forward operations have been fused
        forward_ops = model._module_groups[0]._forward_ops
        assert len(forward_ops) == 1
        assert isinstance(forward_ops[0][0], ForwardLinearBiasActivation)

        # Expected numerical error
        tols = dtype_tols(dtype)
        if dtype == torch.float32:
            tols = dtype_tols(torch.float16)  # TF32 GEMM
        if fp8_compute:
            tols = dtype_tols(
                model[0].weight._fp8_dtype
                if is_float8_tensor(model[0].weight)
                else tex.DType.kFloat8E4M3
            )

        # Check results
        y_test = y_test.to(dtype=torch.float64, device="cpu")
        dx_test = x_test.grad.to(dtype=torch.float64, device="cpu")
        dw_test = model[0].weight.grad.to(dtype=torch.float64, device="cpu")
        torch.testing.assert_close(y_test, y_ref, **tols)
        torch.testing.assert_close(dx_test, x_ref.grad, **tols)
        torch.testing.assert_close(dw_test, w_ref.grad, **tols)
        if bias:
            db_test = model[0].bias.grad.to(dtype=torch.float64, device="cpu")
            torch.testing.assert_close(db_test, b_ref.grad, **tols)
