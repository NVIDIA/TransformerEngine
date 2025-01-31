# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

from __future__ import annotations

from collections.abc import Iterable
import math
from typing import Optional

import pytest
import torch

import transformer_engine
import transformer_engine.common.recipe
import transformer_engine.pytorch as te
from transformer_engine.pytorch.fp8 import FP8GlobalStateManager
import transformer_engine.pytorch.ops as te_ops
from transformer_engine.pytorch.ops._common import is_float8_tensor
from transformer_engine.pytorch.ops.fused import (
    BackwardLinearAdd,
    ForwardLinearBiasActivation,
    ForwardLinearBiasAdd,
)
from transformer_engine.pytorch.tensor import QuantizedTensor
from transformer_engine.pytorch.tensor.float8_tensor import Float8Tensor, Float8Quantizer
from transformer_engine.pytorch.utils import is_bf16_compatible
import transformer_engine_torch as tex

# Check if FP8 is supported
fp8_available, reason_for_no_fp8 = FP8GlobalStateManager.is_fp8_available()
mxfp8_available, reason_for_no_mxfp8 = FP8GlobalStateManager.is_mxfp8_available()

# Supported data types
_dtypes: list[torch.dtype] = [torch.float32, torch.float16]
if is_bf16_compatible():  # bf16 requires sm_80 or higher
    _dtypes.append(torch.bfloat16)

# Supported devices
_devices: list[torch.device] = [torch.device("cpu"), torch.device("cuda")]


def maybe_skip_quantization(
    quantization: Optional[str],
    *,
    dims: Optional[Iterable[int] | int] = None,
    device: Optional[torch.device | str] = None,
) -> None:

    # Don't skip if there is no quantization
    if quantization is None:
        return

    # Check if quantization scheme is supported
    if quantization == "fp8" and not fp8_available:
        pytest.skip(reason_for_no_fp8)
    if quantization == "mxfp8" and not mxfp8_available:
        pytest.skip(reason_for_no_mxfp8)

    if dims is not None:
        if not isinstance(dims, Iterable):
            dims = (dims,)
        if quantization == "fp8":
            if math.prod(dims[:-1]) % 16 != 0 or dims[-1] % 16 != 0:
                pytest.skip("FP8 GEMMs require dims that are divisible by 16")
        elif quantization == "mxfp8":
            if math.prod(dims[:-1]) % 128 != 0 or dims[-1] % 128 != 0:
                pytest.skip("FP8 GEMMs require dims that are divisible by 128")

    # Check if device is supported
    if device is not None and torch.device(device).type != "cuda":
        pytest.skip("Quantization is only supported on CUDA devices")


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
    test = ref.to(device=test_device, dtype=test_dtype)
    if test_is_fp8:
        quantizer = Float8Quantizer(
            scale=torch.ones(1, dtype=torch.float32, device=test_device).squeeze(),
            amax=torch.zeros(1, dtype=torch.float32, device=test_device),
            fp8_dtype=tex.DType.kFloat8E4M3,
        )
        test = quantizer(test)
    elif test.data_ptr() == ref.data_ptr():
        test = test.clone()
    ref.copy_(test)
    ref.requires_grad_(requires_grad)
    test.requires_grad_(requires_grad)
    return ref, test


def make_recipe(name: Optional[str] = None) -> Optional[Recipe]:
    """Make recipe for quantization scheme"""
    if name is None:
        return None
    if name == "fp8":
        return transformer_engine.common.recipe.DelayedScaling(
            fp8_format=transformer_engine.common.recipe.Format.E4M3,
        )
    if name == "mxfp8":
        return transformer_engine.common.recipe.MXFP8Recipe(
            fp8_format=transformer_engine.common.recipe.Format.E4M3,
        )
    raise ValueError(f"Unsupported quantization scheme ({name})")


class TestSequential:
    """Tests for sequential container"""

    def test_modules(self) -> None:
        """Check that list of modules can be manipulated as expected"""

        # Construct sequential container
        modules = [
            te_ops.Identity(),
            te_ops.Identity(),
            torch.nn.Identity(),
            te_ops.Identity(),
        ]
        model = te_ops.Sequential(*modules)

        # Length
        assert len(model) == len(modules)

        # Iterator
        for module1, module2 in zip(model, modules):
            assert module1 is module2

        # Index by int
        for i, module in enumerate(modules):
            assert model[i] is module
            assert model[i - len(modules)] is module

        # Index by slice
        model_subset = model[1:-1]
        modules_subset = modules[1:-1]
        assert isinstance(model_subset, te_ops.Sequential)
        for module1, module2 in zip(model_subset, modules_subset):
            assert module1 is module2

        # Set element
        new_module = torch.nn.Identity()
        idx = 1
        modules[idx] = new_module
        model[idx] = new_module
        for module1, module2 in zip(model, modules):
            assert module1 is module2

        # Delete element
        idx = 1
        del modules[idx]
        del model[idx]
        for module1, module2 in zip(model, modules):
            assert module1 is module2

        # Append
        new_module = torch.nn.Identity()
        modules.append(new_module)
        model.append(new_module)
        for module1, module2 in zip(model, modules):
            assert module1 is module2

        # Extend
        new_modules = [te_ops.Identity(), te_ops.Identity()]
        modules.extend(new_modules)
        model.extend(new_modules)
        for module1, module2 in zip(model, modules):
            assert module1 is module2

        # Insert
        new_module = te_ops.Identity()
        idx = 2
        modules.insert(idx, new_module)
        model.insert(idx, new_module)
        for module1, module2 in zip(model, modules):
            assert module1 is module2

        # Pop
        idx = 2
        assert model.pop(idx) is modules.pop(idx)
        for module1, module2 in zip(model, modules):
            assert module1 is module2

        # Out-of-place add
        new_modules = [torch.nn.Identity(), te_ops.Identity()]
        added_modules = modules + new_modules
        added_model = model + te_ops.Sequential(*new_modules)
        for module1, module2 in zip(model, modules):
            assert module1 is module2
        for module1, module2 in zip(added_model, added_modules):
            assert module1 is module2

        # In-place add
        new_modules = [te_ops.Identity(), torch.nn.Identity()]
        modules += new_modules
        model += te_ops.Sequential(*new_modules)
        for module1, module2 in zip(model, modules):
            assert module1 is module2

    def test_module_groups(self) -> None:
        """Check that modules are grouped together correctly"""
        model = te_ops.Sequential(
            te_ops.Identity(),
            te_ops.Identity(),
            torch.nn.Identity(),
            torch.nn.Identity(),
            te_ops.Identity(),
            torch.nn.Identity(),
            te_ops.Identity(),
            te_ops.Identity(),
            te_ops.Identity(),
        )
        model(torch.zeros(1))
        assert len(model._module_groups) == 6


class TestFuser:
    """Tests for operation fusion infrastructure"""

    @staticmethod
    def setup_class(cls) -> None:
        # Configure RNG
        seed = 1234
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

    @pytest.mark.skipif(not fp8_available, reason=reason_for_no_fp8)
    def test_fp8_scale_update(
        self,
        size: int = 16,
        dtype: torch.dtype = torch.float32,
        device: torch.device = "cuda",
    ):
        """Test FP8 scaling factors with delayed scaling recipe"""

        # FP8 recipe
        margin = 2
        fp8_format = transformer_engine.common.recipe.Format.HYBRID
        recipe = transformer_engine.common.recipe.DelayedScaling(
            margin=margin,
            interval=1,
            fp8_format=fp8_format,
            amax_history_len=8,
            amax_compute_algo="max",
        )

        # Construct model
        with te.fp8_model_init(recipe=recipe):
            model = te_ops.basic.BasicLinear(
                size,
                size,
                device=device,
                dtype=dtype,
            )

        # Training steps
        w_vals = [2, 5, 3, 11]
        x_vals = [7, 3, 5]
        dy_vals = [1, 2, 1]
        with torch.no_grad():
            model.weight.fill_(w_vals[0])
        for step in range(3):

            # Data tensors
            x = torch.full(
                (size, size),
                x_vals[step],
                dtype=dtype,
                device=device,
                requires_grad=True,
            )
            dy = torch.full(
                (size, size),
                dy_vals[step],
                dtype=dtype,
                device=device,
            )

            # Training step
            with te.fp8_autocast(fp8_recipe=recipe):
                y = model(x)
            y.backward(dy)
            with torch.no_grad():
                model.weight.fill_(w_vals[step + 1])

            # Check that output tensors match expected
            tols = dict(rtol=0, atol=0)
            y_val_ref = w_vals[step] * x_vals[step] * size
            dx_val_ref = w_vals[step] * dy_vals[step] * size
            torch.testing.assert_close(
                y,
                torch.full_like(y, y_val_ref),
                **dtype_tols(tex.DType.kFloat8E4M3),
            )
            torch.testing.assert_close(
                x.grad,
                torch.full_like(x.grad, dx_val_ref),
                **dtype_tols(tex.DType.kFloat8E5M2),
            )

            # Check that scaling factors match expected
            w_amax_ref = max(w_vals[: step + 1])
            x_amax_ref = max(x_vals[: step + 1])
            dy_amax_ref = max(dy_vals[: step + 1])
            w_scale_ref = (fp8_format.value.max_fwd / w_amax_ref) / (2**margin)
            x_scale_ref = (fp8_format.value.max_fwd / x_amax_ref) / (2**margin)
            dy_scale_ref = (fp8_format.value.max_bwd / dy_amax_ref) / (2**margin)
            w_scale = model.get_quantizer("forward", 1).scale
            x_scale = model.get_quantizer("forward", 0).scale
            dy_scale = model.get_quantizer("backward", 0).scale
            torch.testing.assert_close(w_scale, torch.full_like(w_scale, w_scale_ref))
            torch.testing.assert_close(x_scale, torch.full_like(x_scale, x_scale_ref))
            torch.testing.assert_close(dy_scale, torch.full_like(dy_scale, dy_scale_ref))

    @pytest.mark.parametrize("init_dtype", _dtypes)
    @pytest.mark.parametrize("final_dtype", _dtypes)
    @pytest.mark.parametrize("quantization", (None, "fp8", "mxfp8"))
    def test_dtype_cast(
        self,
        *,
        init_dtype: torch.dtype,
        final_dtype: torch.dtype,
        device: torch.device = "cuda",
        quantization: Optional[str],
    ) -> None:
        """Check dtype cast functions"""

        # Skip invalid configurations
        maybe_skip_quantization(quantization, device=device)
        with_quantization = quantization is not None

        # Data dimensions
        size = 16
        if quantization == "mxfp8":
            size = 128

        # Random data
        dtype = torch.float32
        if torch.float16 in (init_dtype, final_dtype):
            dtype = torch.float16
        if torch.bfloat16 in (init_dtype, final_dtype):
            dtype = torch.bfloat16
        w_ref, w_test = make_reference_and_test_tensors(
            (size, size),
            test_dtype=dtype,
            test_device=device,
            test_is_fp8=with_quantization,
        )

        # Construct operation
        with te.fp8_model_init(enabled=with_quantization, recipe=make_recipe(quantization)):
            op = te_ops.Linear(size, size, bias=False, device=device, dtype=init_dtype)
        with torch.no_grad():
            op.weight.copy_(w_test)
            del w_test

        # Cast operation dtype
        if final_dtype == torch.float32:
            op.float()
        elif final_dtype == torch.float16:
            op.half()
        elif final_dtype == torch.bfloat16:
            op.bfloat16()

        # Check weights
        assert isinstance(op.weight, QuantizedTensor) == with_quantization
        assert op.weight.dtype == final_dtype
        w_test = op.weight.to(dtype=torch.float64, device="cpu")
        torch.testing.assert_close(w_test, w_ref, rtol=0, atol=0)

        # Check forward and backward pass
        x = torch.zeros(
            (size, size),
            dtype=init_dtype,
            device=device,
            requires_grad=True,
        )
        y = op(x)
        y.backward(torch.zeros_like(y))
        assert y.dtype == final_dtype
        assert x.grad.dtype == init_dtype
        assert op.weight.grad.dtype == final_dtype

    @pytest.mark.parametrize("model_dtype", _dtypes)
    @pytest.mark.parametrize("autocast_dtype", _dtypes)
    @pytest.mark.parametrize("quantization", (None, "fp8", "mxfp8"))
    def test_pyt_autocast(
        self,
        *,
        model_dtype: torch.dtype,
        autocast_dtype: torch.dtype,
        device: torch.device = "cuda",
        quantization: Optional[str],
        quantized_weights: bool = False,
    ) -> None:
        """Test with PyTorch autocast"""
        device = torch.device(device)

        # Skip invalid configurations
        quantized_compute = quantization is not None
        maybe_skip_quantization(quantization)

        # Data dimensions
        size = 16
        if quantization == "mxfp8":
            size = 128

        # Construct operation
        recipe = make_recipe(quantization)
        with te.fp8_model_init(enabled=quantized_weights, recipe=recipe):
            op = te_ops.Linear(size, size, bias=False, device=device, dtype=model_dtype)

        # Check forward and backward pass
        x = torch.zeros(
            (size, size),
            dtype=model_dtype,
            device=device,
            requires_grad=True,
        )
        with te.fp8_autocast(enabled=quantized_compute, fp8_recipe=recipe):
            with torch.autocast(device_type=device.type, dtype=autocast_dtype):
                y = op(x)
        y.backward(torch.zeros_like(y))
        assert y.dtype == autocast_dtype
        assert x.grad.dtype == model_dtype
        assert op.weight.grad.dtype == model_dtype

        # Check forward and backward pass (swapped context order)
        if quantized_compute:
            x.grad = None
            op.weight.grad = None
            with torch.autocast(device_type=device.type, dtype=autocast_dtype):
                with te.fp8_autocast(enabled=quantized_compute, fp8_recipe=recipe):
                    y = op(x)
            y.backward(torch.zeros_like(y))
            assert y.dtype == autocast_dtype
            assert x.grad.dtype == model_dtype
            assert op.weight.grad.dtype == model_dtype


class TestBasicOps:
    """Tests for individual operations"""

    @staticmethod
    def setup_class(cls) -> None:
        # Configure RNG
        seed = 1234
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

    @pytest.mark.parametrize("dtype", _dtypes)
    @pytest.mark.parametrize("device", ("cuda", "cpu"))
    @pytest.mark.parametrize("fp8", (False, True))
    def test_identity(
        self,
        *,
        in_shape: Iterable[int] = (1,),
        dtype: torch.dtype,
        device: torch.device,
        fp8: bool,
    ) -> None:

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
        dy_ref, dy_test = make_reference_and_test_tensors(
            in_shape,
            test_dtype=dtype,
            test_device=device,
            requires_grad=False,
        )

        # Plain PyTorch implementation
        y_ref = x_ref
        dx_ref = dy_ref

        # Implementation with fusible operation
        op = te_ops.Identity()
        y_test = op(x_test)
        y_test.backward(dy_test)

        # Check results
        tols = dict(rtol=0, atol=0)  # Identity is exact
        y_test = y_test.to(dtype=torch.float64, device="cpu")
        dx_test = x_test.grad.to(dtype=torch.float64, device="cpu")
        torch.testing.assert_close(y_test, y_ref, **tols)
        torch.testing.assert_close(dx_test, dx_ref, **tols)

        # Make sure we are not trivially passing the test
        with pytest.raises(AssertionError):
            torch.testing.assert_close(y_test, -y_ref, **tols)
        with pytest.raises(AssertionError):
            torch.testing.assert_close(dx_test, -dx_ref, **tols)

    @pytest.mark.parametrize(
        "shapes",
        (
            ((1, 2, 3, 4), (2, 12)),
            ((5, 4, 3, 2), (-1, 6)),
            ((30,), (2, 3, -1)),
            ((6, 7), (3, -1, 7)),
        ),
    )
    @pytest.mark.parametrize("dtype", _dtypes)
    @pytest.mark.parametrize("fp8", (False, True))
    def test_reshape(
        self,
        *,
        shapes: tuple[Iterable[int], Iterable[int]],
        dtype: torch.dtype,
        device: torch.device = "cuda",
        memory_format: torch.memory_format = torch.contiguous_format,
        fp8: bool,
    ) -> None:
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

        # Implementation with fusible operation
        op = te_ops.Reshape(out_shape)
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
    @pytest.mark.parametrize("in_shape", ((-1,), (1, 3, -1), (2, 3, 4, -1)))
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

        # Make input and bias shapes consistent
        in_shape = list(in_shape)[:-1] + [size]

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

        # Implementation with fusible operation
        op = te_ops.Bias(size, device=device, dtype=dtype)
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

    @pytest.mark.parametrize("quantization", ("fp8", "mxfp8"))
    @pytest.mark.parametrize("cast_forward", (False, True))
    @pytest.mark.parametrize("cast_backward", (False, True))
    def test_quantize(
        self,
        *,
        in_shape: Iterable[int] = (128, 128),
        dtype: torch.dtype = torch.bfloat16,
        device: torch.device = "cuda",
        quantization: str,
        cast_forward: bool,
        cast_backward: bool,
    ) -> None:
        """Quantize"""

        # Skip invalid configurations
        maybe_skip_quantization(quantization)

        # Random data
        x_ref, x_test = make_reference_and_test_tensors(
            in_shape,
            test_dtype=dtype,
            test_device=device,
            requires_grad=False,
            test_is_fp8=True,
        )
        x_test = x_test.dequantize().requires_grad_()
        dy_ref, dy_test = make_reference_and_test_tensors(
            in_shape,
            test_dtype=dtype,
            test_device=device,
            requires_grad=False,
            test_is_fp8=True,
        )
        dy_test = dy_test.dequantize()

        # Plain PyTorch implementation
        y_ref = x_ref
        dx_ref = dy_ref

        # Implementation with fusible operation
        op = te_ops.Quantize(forward=cast_forward, backward=cast_backward)
        recipe = make_recipe(quantization)
        with te.fp8_autocast(fp8_recipe=recipe):
            y_test = op(x_test)
        y_test.backward(dy_test)

        # Check tensor types
        assert isinstance(y_test, QuantizedTensor) == cast_forward
        assert isinstance(x_test.grad, QuantizedTensor) == cast_backward

        # Check values
        tols = dict(rtol=0, atol=0)
        y_test = y_test.to(dtype=torch.float64, device="cpu")
        dx_test = x_test.grad.to(dtype=torch.float64, device="cpu")
        torch.testing.assert_close(y_test, y_ref, **tols)
        torch.testing.assert_close(dx_test, dx_ref, **tols)

    def _test_basic_linear(
        self,
        *,
        weight_shape: tuple[int, int] = (32, 32),
        in_shape: Iterable[int] = (32, -1),
        dtype: torch.dtype = torch.float32,
        device: torch.device = "cuda",
        quantization: Optional[str] = None,
        quantized_compute: bool = False,
        quantized_input: bool = False,
        quantized_weight: bool = False,
        quantized_output: bool = False,
        quantized_grad_output: bool = False,
        quantized_grad_input: bool = False,
        accumulate_into_main_grad: bool = False,
    ) -> None:
        """Helper function for tests with GEMM"""

        # Make input and weight shapes consistent
        out_features, in_features = weight_shape
        in_shape = list(in_shape)[:-1] + [in_features]
        out_shape = in_shape[:-1] + [out_features]

        # Skip invalid configurations
        maybe_skip_quantization(quantization, dims=in_shape, device=device)
        maybe_skip_quantization(quantization, dims=out_shape)
        if quantization == "fp8" and quantized_output and not quantized_compute:
            pytest.skip("FP8 output is only supported with FP8 GEMMs")
        if quantization == "fp8" and quantized_grad_input and not quantized_compute:
            pytest.skip("FP8 grad input is only supported with FP8 GEMMs")

        # Random data
        x_ref, x_test = make_reference_and_test_tensors(
            in_shape,
            test_dtype=dtype,
            test_device=device,
            test_is_fp8=(quantized_compute or quantized_input),
        )
        if isinstance(x_test, QuantizedTensor):
            with torch.no_grad():
                x_test = x_test.dequantize().requires_grad_()
        w_ref, w_test = make_reference_and_test_tensors(
            (out_features, in_features),
            test_dtype=dtype,
            test_device=device,
            test_is_fp8=(quantized_compute or quantized_weight),
        )
        dy_ref, dy_test = make_reference_and_test_tensors(
            out_shape,
            test_dtype=dtype,
            test_device=device,
            test_is_fp8=(quantized_compute or quantized_grad_output),
            requires_grad=False,
        )
        if isinstance(dy_test, QuantizedTensor):
            dy_test = dy_test.dequantize()

        # Plain PyTorch implementation
        y_ref = torch.nn.functional.linear(x_ref, w_ref)
        y_ref.backward(dy_ref)

        # Implementation with fusible operation
        recipe = make_recipe(quantization)
        with te.fp8_model_init(enabled=quantized_weight, recipe=recipe):
            op = te_ops.BasicLinear(
                in_features,
                out_features,
                device=device,
                dtype=dtype,
                accumulate_into_main_grad=accumulate_into_main_grad,
            )
        with torch.no_grad():
            op.weight.copy_(w_test)
            del w_test
            op.weight.main_grad = torch.full_like(op.weight, 0.5, dtype=torch.float32)
        forward = te_ops.Sequential(
            te_ops.Quantize(forward=quantized_input, backward=quantized_grad_input),
            op,
            te_ops.Quantize(forward=quantized_output, backward=quantized_grad_output),
        )
        with te.fp8_autocast(enabled=quantized_compute, fp8_recipe=recipe):
            y_test = forward(x_test)
        y_test.backward(dy_test)

        # Expected numerical error
        tols = dtype_tols(dtype)
        if dtype == torch.float32:
            tols = dtype_tols(torch.float16)  # TF32 GEMM
        if quantized_compute or quantized_output or quantized_grad_input:
            tols = dtype_tols(tex.DType.kFloat8E4M3)

        # Check results
        y_test = y_test.to(dtype=torch.float64, device="cpu")
        dx_test = x_test.grad.to(dtype=torch.float64, device="cpu")
        torch.testing.assert_close(y_test, y_ref, **tols)
        torch.testing.assert_close(dx_test, x_ref.grad, **tols)
        if accumulate_into_main_grad:
            if op.weight.grad is not None:
                torch.testing.assert_close(
                    op.weight.grad,
                    torch.zeros_like(op.weight.grad),
                    rtol=0,
                    atol=0,
                )
            dw_test = op.weight.main_grad.to(dtype=torch.float64, device="cpu") - 0.5
        else:
            dw_test = op.weight.grad.to(dtype=torch.float64, device="cpu")
            torch.testing.assert_close(
                op.weight.main_grad,
                torch.full_like(op.weight.main_grad, 0.5),
                rtol=0,
                atol=0,
            )
        torch.testing.assert_close(dw_test, w_ref.grad, **tols)

    @pytest.mark.parametrize("weight_shape", ((128, 128), (3, 5)))
    @pytest.mark.parametrize("in_shape", ((-1,), (5, 1, -1), (4, 4, 8, -1)))
    @pytest.mark.parametrize("dtype", _dtypes)
    @pytest.mark.parametrize("quantization", (None, "fp8", "mxfp8"))
    @pytest.mark.parametrize("accumulate_into_main_grad", (False, True))
    def test_basic_linear(
        self,
        *,
        weight_shape: tuple[int, int],
        in_shape: Iterable[int],
        dtype: torch.dtype,
        quantization: Optional[str],
        accumulate_into_main_grad: bool,
    ) -> None:
        """GEMM"""
        self._test_basic_linear(
            weight_shape=weight_shape,
            in_shape=in_shape,
            dtype=dtype,
            quantization=quantization,
            quantized_compute=quantization is not None,
            accumulate_into_main_grad=accumulate_into_main_grad,
        )

    @pytest.mark.skipif(not fp8_available, reason=reason_for_no_fp8)
    @pytest.mark.parametrize("quantization", ("fp8", "mxfp8"))
    @pytest.mark.parametrize("quantized_compute", (False, True))
    @pytest.mark.parametrize("quantized_input", (False, True))
    @pytest.mark.parametrize("quantized_weight", (False, True))
    @pytest.mark.parametrize("quantized_output", (False, True))
    @pytest.mark.parametrize("quantized_grad_output", (False, True))
    @pytest.mark.parametrize("quantized_grad_input", (False, True))
    def test_basic_linear_quantized(
        self,
        *,
        quantization: str,
        quantized_compute: bool,
        quantized_input: bool,
        quantized_weight: bool,
        quantized_output: bool,
        quantized_grad_output: bool,
        quantized_grad_input: bool,
    ) -> None:
        """GEMM with FP8 inputs and outputs"""
        self._test_basic_linear(
            dtype=torch.bfloat16,
            quantization=quantization,
            quantized_compute=quantized_compute,
            quantized_input=quantized_input,
            quantized_weight=quantized_weight,
            quantized_output=quantized_output,
            quantized_grad_output=quantized_grad_output,
            quantized_grad_input=quantized_grad_input,
        )

    @pytest.mark.parametrize("bias", (False, True))
    @pytest.mark.parametrize("quantization", (None, "fp8", "mxfp8"))
    @pytest.mark.parametrize("quantized_weight", (False, True))
    def test_linear(
        self,
        *,
        bias: bool,
        weight_shape: tuple[int, int] = (128, 128),
        in_shape: Iterable[int] = (128, -1),
        dtype: torch.dtype = torch.float32,
        device: torch.device = "cuda",
        quantization: Optional[str],
        quantized_weight: bool,
    ) -> None:
        """GEMM + bias"""

        # Make input and weight shapes consistent
        out_features, in_features = weight_shape
        in_shape = list(in_shape)[:-1] + [in_features]
        out_shape = in_shape[:-1] + [out_features]

        # Skip invalid configurations
        quantized_compute = quantization is not None
        maybe_skip_quantization(quantization, dims=in_shape, device=device)
        maybe_skip_quantization(quantization, dims=out_shape)

        # Random data
        x_ref, x_test = make_reference_and_test_tensors(
            in_shape,
            test_dtype=dtype,
            test_device=device,
            test_is_fp8=quantized_compute,
        )
        if isinstance(x_test, QuantizedTensor):
            with torch.no_grad():
                x_test = x_test.dequantize().requires_grad_()
        w_ref, w_test = make_reference_and_test_tensors(
            (out_features, in_features),
            test_dtype=dtype,
            test_device=device,
            test_is_fp8=(quantized_compute or quantized_weight),
        )
        b_ref, b_test = None, None
        if bias:
            b_ref, b_test = make_reference_and_test_tensors(
                out_features,
                test_dtype=dtype,
                test_device=device,
            )
        dy_ref, dy_test = make_reference_and_test_tensors(
            out_shape,
            test_dtype=dtype,
            test_device=device,
            requires_grad=False,
        )

        # Plain PyTorch implementation
        y_ref = torch.nn.functional.linear(x_ref, w_ref, bias=b_ref)
        y_ref.backward(dy_ref)

        # Implementation with fusible operation
        recipe = make_recipe(quantization)
        with te.fp8_model_init(enabled=quantized_weight, recipe=recipe):
            op = te_ops.Linear(
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
        with te.fp8_autocast(enabled=quantized_compute, fp8_recipe=recipe):
            y_test = op(x_test)
        y_test.backward(dy_test)

        # Expected numerical error
        tols = dtype_tols(dtype)
        if dtype == torch.float32:
            tols = dtype_tols(torch.float16)  # TF32 GEMM
        if quantized_compute:
            tols = dtype_tols(tex.DType.kFloat8E4M3)

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

    @pytest.mark.parametrize("weight_shape", ((7, 2), (128,)))
    @pytest.mark.parametrize("in_shape", ((-1,), (6, 64, -1)))
    @pytest.mark.parametrize("dtype", _dtypes)
    @pytest.mark.parametrize("zero_centered_gamma", (False, True))
    @pytest.mark.parametrize("quantization", (None, "fp8", "mxfp8"))
    def test_layer_norm(
        self,
        *,
        weight_shape: Iterable[int],
        in_shape: Iterable[int],
        dtype: torch.dtype,
        device: torch.device = "cuda",
        eps: float = 0.3,
        zero_centered_gamma: bool,
        quantization: Optional[str],
    ) -> None:
        """Layer norm"""

        # Make input and weight shapes consistent
        in_shape = list(in_shape)[:-1] + list(weight_shape)

        # Skip invalid configurations
        maybe_skip_quantization(quantization, dims=in_shape, device=device)

        # Random data
        x_ref, x_test = make_reference_and_test_tensors(
            in_shape,
            test_dtype=dtype,
            test_device=device,
        )
        w_ref, w_test = make_reference_and_test_tensors(
            weight_shape,
            test_dtype=dtype,
            test_device=device,
        )
        b_ref, b_test = make_reference_and_test_tensors(
            weight_shape,
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
        y_ref = torch.nn.functional.layer_norm(
            x_ref,
            weight_shape,
            weight=(w_ref + 1 if zero_centered_gamma else w_ref),
            bias=b_ref,
            eps=eps,
        )
        y_ref.backward(dy_ref)

        # Implementation with fusible operation
        op = te_ops.LayerNorm(
            weight_shape,
            eps=eps,
            device=device,
            dtype=dtype,
            zero_centered_gamma=zero_centered_gamma,
        )
        with torch.no_grad():
            op.weight.copy_(w_test)
            op.bias.copy_(b_test)
            del w_test
            del b_test
        quantized_compute = quantization is not None
        recipe = make_recipe(quantization)
        forward = te_ops.Sequential(
            op,
            te_ops.Quantize(forward=quantized_compute, backward=False),
        )
        with te.fp8_autocast(enabled=quantized_compute, fp8_recipe=recipe):
            y_test = forward(x_test)
        y_test.backward(dy_test)

        # Expected numerical error
        tols = dtype_tols(dtype)
        if quantized_compute:
            tols = dtype_tols(tex.DType.kFloat8E4M3)

        # Check results
        y_test = y_test.to(dtype=torch.float64, device="cpu")
        dx_test = x_test.grad.to(dtype=torch.float64, device="cpu")
        dw_test = op.weight.grad.to(dtype=torch.float64, device="cpu")
        db_test = op.bias.grad.to(dtype=torch.float64, device="cpu")
        torch.testing.assert_close(y_test, y_ref, **tols)
        torch.testing.assert_close(dx_test, x_ref.grad, **tols)
        torch.testing.assert_close(dw_test, w_ref.grad, **tols)
        torch.testing.assert_close(db_test, b_ref.grad, **tols)

    def test_layer_norm_autocast(
        self,
        *,
        weight_shape: Iterable[int] = (32,),
        in_shape: Iterable[int] = (32,),
        dtype: torch.dtype = torch.float16,
        autocast_dtype: torch.dtype = torch.float32,
        device: torch.device = "cuda",
        eps: float = 0.3,
    ) -> None:
        """Layer norm with PyTorch autocast"""

        # Make input and weight shapes consistent
        in_shape = list(in_shape)[:-1] + list(weight_shape)

        # Random data
        x_ref, x_test = make_reference_and_test_tensors(
            in_shape,
            test_dtype=autocast_dtype,
            test_device=device,
        )
        w_ref, w_test = make_reference_and_test_tensors(
            weight_shape,
            test_dtype=dtype,
            test_device=device,
        )
        b_ref, b_test = make_reference_and_test_tensors(
            weight_shape,
            test_dtype=dtype,
            test_device=device,
        )
        dy_ref, dy_test = make_reference_and_test_tensors(
            in_shape,
            test_dtype=autocast_dtype,
            test_device=device,
            requires_grad=False,
        )

        # Plain PyTorch implementation
        y_ref = torch.nn.functional.layer_norm(
            x_ref,
            weight_shape,
            weight=w_ref,
            bias=b_ref,
            eps=eps,
        )
        y_ref.backward(dy_ref)

        # Implementation with fusible operation
        op = te_ops.LayerNorm(
            weight_shape,
            eps=eps,
            device=device,
            dtype=dtype,
        )
        with torch.no_grad():
            op.weight.copy_(w_test)
            op.bias.copy_(b_test)
            del w_test
            del b_test
        with torch.autocast(device, dtype=autocast_dtype):
            y_test = op(x_test)
        y_test.backward(dy_test)

        # Check results
        assert y_test.dtype == autocast_dtype
        y_test = y_test.to(dtype=torch.float64, device="cpu")
        dx_test = x_test.grad.to(dtype=torch.float64, device="cpu")
        dw_test = op.weight.grad.to(dtype=torch.float64, device="cpu")
        db_test = op.bias.grad.to(dtype=torch.float64, device="cpu")
        torch.testing.assert_close(y_test, y_ref, **dtype_tols(autocast_dtype))
        torch.testing.assert_close(dx_test, x_ref.grad, **dtype_tols(autocast_dtype))
        torch.testing.assert_close(dw_test, w_ref.grad, **dtype_tols(dtype))
        torch.testing.assert_close(db_test, b_ref.grad, **dtype_tols(dtype))

    @pytest.mark.parametrize("weight_shape", ((19,), (128,)))
    @pytest.mark.parametrize("in_shape", ((-1,), (6, 64, -1)))
    @pytest.mark.parametrize("dtype", _dtypes)
    @pytest.mark.parametrize("zero_centered_gamma", (False, True))
    @pytest.mark.parametrize("quantization", (None, "fp8", "mxfp8"))
    def test_rmsnorm(
        self,
        *,
        weight_shape: Iterable[int],
        in_shape: Iterable[int],
        dtype: torch.dtype,
        device: torch.device = "cuda",
        eps: float = 0.3,
        zero_centered_gamma: bool,
        quantization: Optional[str],
    ) -> None:
        """Layer norm"""

        # Make input and weight shapes consistent
        in_shape = list(in_shape)[:-1] + list(weight_shape)

        # Skip invalid configurations
        maybe_skip_quantization(quantization, dims=in_shape, device=device)

        # Random data
        x_ref, x_test = make_reference_and_test_tensors(
            in_shape,
            test_dtype=dtype,
            test_device=device,
        )
        w_ref, w_test = make_reference_and_test_tensors(
            weight_shape,
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
        inner_dims = tuple(range(len(in_shape) - len(weight_shape), len(in_shape)))
        var_ref = x_ref.square().sum(dim=inner_dims, keepdim=True) / math.prod(weight_shape)
        if zero_centered_gamma:
            y_ref = x_ref / torch.sqrt(eps + var_ref) * (1 + w_ref)
        else:
            y_ref = x_ref / torch.sqrt(eps + var_ref) * w_ref
        y_ref.backward(dy_ref)

        # Implementation with fusible operation
        op = te_ops.RMSNorm(
            weight_shape,
            eps=eps,
            device=device,
            dtype=dtype,
            zero_centered_gamma=zero_centered_gamma,
        )
        with torch.no_grad():
            op.weight.copy_(w_test)
            del w_test
        quantized_compute = quantization is not None
        recipe = make_recipe(quantization)
        forward = te_ops.Sequential(
            op,
            te_ops.Quantize(forward=quantized_compute, backward=False),
        )
        with te.fp8_autocast(enabled=quantized_compute, fp8_recipe=recipe):
            y_test = forward(x_test)
        y_test.backward(dy_test)

        # Expected numerical error
        tols = dtype_tols(dtype)
        if quantized_compute:
            tols = dtype_tols(tex.DType.kFloat8E4M3)

        # Check results
        y_test = y_test.to(dtype=torch.float64, device="cpu")
        dx_test = x_test.grad.to(dtype=torch.float64, device="cpu")
        dw_test = op.weight.grad.to(dtype=torch.float64, device="cpu")
        torch.testing.assert_close(y_test, y_ref, **tols)
        torch.testing.assert_close(dx_test, x_ref.grad, **tols)
        torch.testing.assert_close(dw_test, w_ref.grad, **tols)

    @pytest.mark.parametrize("dtype", _dtypes)
    @pytest.mark.parametrize("device", ("cuda", "cpu"))
    @pytest.mark.parametrize("fp8", (False, True))
    def test_add_in_place(
        self,
        *,
        in_shape: Iterable[int] = (1,),
        dtype: torch.dtype,
        device: torch.device,
        fp8: bool,
    ) -> None:
        """Add two tensors

        Join in compute graph.

        """

        # Skip invalid configurations
        if fp8 and not fp8_available:
            pytest.skip(reason_for_no_fp8)
        if fp8 and torch.device(device).type != "cuda":
            pytest.skip("FP8 is only supported on CUDA devices")

        # Random data
        x1_ref, x1_test = make_reference_and_test_tensors(
            in_shape,
            test_dtype=dtype,
            test_device=device,
            test_is_fp8=fp8,
        )
        x2_ref, x2_test = make_reference_and_test_tensors(
            in_shape,
            test_dtype=dtype,
            test_device=device,
            test_is_fp8=fp8,
        )
        dy_ref, dy_test = make_reference_and_test_tensors(
            in_shape,
            test_dtype=dtype,
            test_device=device,
            requires_grad=False,
        )

        # Plain PyTorch implementation
        y_ref = x2_ref.detach()
        y_ref += x1_ref
        dx1_ref = dy_ref
        dx2_ref = dy_ref

        # Implementation with fusible operation
        op = te_ops.AddInPlace()
        y_test = op(x1_test, x2_test)
        y_test.backward(dy_test)

        # Check results
        tols = dtype_tols(dtype)
        if fp8:
            tols = dtype_tols(x1_test._fp8_dtype)
        y_test = y_test.to(dtype=torch.float64, device="cpu")
        dx1_test = x1_test.grad.to(dtype=torch.float64, device="cpu")
        dx2_test = x2_test.grad.to(dtype=torch.float64, device="cpu")
        torch.testing.assert_close(y_test, y_ref, **tols)
        torch.testing.assert_close(dx1_test, dx1_ref, rtol=0, atol=0)
        torch.testing.assert_close(dx2_test, dx2_ref, rtol=0, atol=0)

    @pytest.mark.parametrize("dtype", _dtypes)
    @pytest.mark.parametrize("device", ("cuda", "cpu"))
    @pytest.mark.parametrize("fp8", (False, True))
    def test_make_extra_output(
        self,
        *,
        in_shape: Iterable[int] = (1,),
        dtype: torch.dtype,
        device: torch.device,
        fp8: bool,
    ) -> None:
        """Output tensor twice

        Split in compute graph.

        """

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
        dy1_ref, dy1_test = make_reference_and_test_tensors(
            in_shape,
            test_dtype=dtype,
            test_device=device,
            requires_grad=False,
        )
        dy2_ref, dy2_test = make_reference_and_test_tensors(
            in_shape,
            test_dtype=dtype,
            test_device=device,
            requires_grad=False,
        )

        # Plain PyTorch implementation
        y1_ref = x_ref
        y2_ref = x_ref
        (y1_ref * dy1_ref + y2_ref * dy2_ref).sum().backward()

        # Implementation with fusible operation
        op = te_ops.MakeExtraOutput()
        y1_test, y2_test = op(x_test)
        (y1_test * dy1_test + y2_test * dy2_test).sum().backward()

        # Check results
        tols = dtype_tols(dtype)
        y1_test = y1_test.to(dtype=torch.float64, device="cpu")
        y2_test = y2_test.to(dtype=torch.float64, device="cpu")
        dx_test = x_test.grad.to(dtype=torch.float64, device="cpu")
        torch.testing.assert_close(y1_test, y1_ref, rtol=0, atol=0)
        torch.testing.assert_close(y2_test, y2_ref, rtol=0, atol=0)
        torch.testing.assert_close(dx_test, x_ref.grad, **tols)

    @pytest.mark.parametrize("activation", ("relu", "gelu", "geglu", "reglu", "swiglu"))
    @pytest.mark.parametrize("out_shape", ((37,), (2, 13), (128, 1, 128)))
    @pytest.mark.parametrize("dtype", _dtypes)
    @pytest.mark.parametrize("quantization", (None, "fp8", "mxfp8"))
    def test_activation(
        self,
        *,
        activation: str,
        out_shape: Iterable[int],
        dtype: torch.dtype,
        device: torch.device = "cuda",
        quantization: Optional[str],
    ) -> None:
        """Activation functions"""

        # Tensor dimensions
        in_shape = list(out_shape)
        if activation in ("geglu", "reglu", "swiglu"):
            in_shape[-1] *= 2

        # Skip invalid configurations
        quantized_compute = quantization is not None
        maybe_skip_quantization(quantization, dims=in_shape, device=device)

        # Random data
        x_ref, x_test = make_reference_and_test_tensors(
            in_shape,
            test_dtype=dtype,
            test_device=device,
            test_is_fp8=quantized_compute,
        )
        if quantized_compute:
            with torch.no_grad():
                x_test = x_test.dequantize().requires_grad_()
        dy_ref, dy_test = make_reference_and_test_tensors(
            out_shape,
            test_dtype=dtype,
            test_device=device,
            requires_grad=False,
        )

        # Plain PyTorch implementation
        y_ref: torch.Tensor
        if activation == "gelu":
            y_ref = torch.nn.functional.gelu(x_ref, approximate="tanh")
        elif activation == "relu":
            y_ref = torch.nn.functional.relu(x_ref)
        elif activation == "geglu":
            x1, x2 = x_ref.chunk(2, dim=-1)
            y_ref = torch.nn.functional.gelu(x1, approximate="tanh") * x2
        elif activation == "reglu":
            x1, x2 = x_ref.chunk(2, dim=-1)
            y_ref = torch.nn.functional.relu(x1) * x2
        elif activation == "swiglu":
            x1, x2 = x_ref.chunk(2, dim=-1)
            y_ref = torch.nn.functional.silu(x1) * x2
        else:
            raise ValueError(f"Unexpected activation function ({activation})")
        y_ref.backward(dy_ref)

        # Implementation with fusible operation
        recipe = make_recipe(quantization)
        make_op = dict(
            gelu=te_ops.GELU,
            relu=te_ops.ReLU,
            geglu=te_ops.GEGLU,
            reglu=te_ops.ReGLU,
            swiglu=te_ops.SwiGLU,
        )[activation]
        forward = te_ops.Sequential(
            make_op(),
            te_ops.Quantize(forward=quantized_compute, backward=False),
        )
        with te.fp8_autocast(enabled=quantized_compute, fp8_recipe=recipe):
            y_test = forward(x_test)
        y_test.backward(dy_test)

        # Expected numerical error
        tols = dtype_tols(dtype)
        if quantized_compute:
            tols = dtype_tols(tex.DType.kFloat8E4M3)
        if activation == "relu":
            tols = {"atol": 0, "rtol": 0}

        # Check results
        y_test = y_test.to(dtype=torch.float64, device="cpu")
        dx_test = x_test.grad.to(dtype=torch.float64, device="cpu")
        torch.testing.assert_close(y_test, y_ref, **tols)
        torch.testing.assert_close(dx_test, x_ref.grad, **tols)

    @pytest.mark.parametrize("dtype", _dtypes)
    @pytest.mark.parametrize("quantization", (None, "fp8", "mxfp8"))
    @pytest.mark.parametrize("quantize_forward", (False, True))
    @pytest.mark.parametrize("quantize_backward", (False, True))
    def test_swiglu(
        self,
        *,
        out_shape: Iterable[int] = (128, 128),
        dtype: torch.dtype,
        device: torch.device = "cuda",
        quantization: Optional[str],
        quantize_forward: bool,
        quantize_backward: bool,
    ):

        # Tensor dimensions
        in_shape = list(out_shape)
        in_shape[-1] *= 2

        # Skip invalid configurations
        quantized_compute = quantization is not None
        if not quantized_compute and (quantize_forward or quantize_backward):
            pytest.skip("Quantization scheme has not been provided")
        maybe_skip_quantization(quantization, dims=in_shape, device=device)

        # Random data
        x_ref, x_test = make_reference_and_test_tensors(
            in_shape,
            test_dtype=dtype,
            test_device=device,
        )
        dy_ref, dy_test = make_reference_and_test_tensors(
            out_shape,
            test_dtype=dtype,
            test_device=device,
            requires_grad=False,
        )

        # Plain PyTorch implementation
        x1, x2 = x_ref.chunk(2, dim=-1)
        y_ref = torch.nn.functional.silu(x1) * x2
        y_ref.backward(dy_ref)

        # Implementation with fusible operation
        recipe = make_recipe(quantization)
        forward = te_ops.Sequential(
            te_ops.Quantize(forward=False, backward=quantize_backward),
            te_ops.SwiGLU(),
            te_ops.Quantize(forward=quantize_forward, backward=False),
        )
        with te.fp8_autocast(enabled=quantized_compute, fp8_recipe=recipe):
            y_test = forward(x_test)
        y_test.backward(dy_test)

        # Expected numerical error
        tols = dtype_tols(dtype)
        if quantized_compute:
            tols = dtype_tols(tex.DType.kFloat8E4M3)

        # Check results
        y_test = y_test.to(dtype=torch.float64, device="cpu")
        dx_test = x_test.grad.to(dtype=torch.float64, device="cpu")
        torch.testing.assert_close(y_test, y_ref, **tols)
        torch.testing.assert_close(dx_test, x_ref.grad, **tols)


class TestFusedOps:
    """Tests for fused operations"""

    @staticmethod
    def setup_class(cls) -> None:
        # Configure RNG
        seed = 1234
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

    @pytest.mark.parametrize("weight_shape", ((128, 128), (3, 5)))
    @pytest.mark.parametrize("in_shape", ((-1,), (1, 7, -1), (128, -1)))
    @pytest.mark.parametrize("dtype", _dtypes)
    @pytest.mark.parametrize("quantization", (None, "fp8", "mxfp8"))
    @pytest.mark.parametrize("quantized_weight", (False, True))
    def test_forward_linear_bias_activation(
        self,
        *,
        bias: bool = True,
        weight_shape: tuple[int, int],
        in_shape: Iterable[int],
        dtype: torch.dtype,
        device: torch.device = "cuda",
        quantization: Optional[str],
        quantized_weight: bool,
    ) -> None:
        """Forward GEMM + bias + activation"""

        # Make input and weight shapes consistent
        out_features, in_features = weight_shape
        in_shape = list(in_shape)[:-1] + [in_features]
        out_shape = in_shape[:-1] + [out_features]

        # Skip invalid configurations
        quantized_compute = quantization is not None
        maybe_skip_quantization(quantization, dims=in_shape, device=device)
        maybe_skip_quantization(quantization, dims=out_shape)
        if dtype not in (torch.float16, torch.bfloat16):
            pytest.skip(
                "FP8 fused linear-bias-activation is only supported with FP16 or BF16 output"
            )

        # Random data
        x_ref, x_test = make_reference_and_test_tensors(
            in_shape,
            test_dtype=dtype,
            test_device=device,
            test_is_fp8=quantized_compute,
        )
        if quantized_compute:
            with torch.no_grad():
                x_test = x_test.dequantize().requires_grad_()
        w_ref, w_test = make_reference_and_test_tensors(
            (out_features, in_features),
            test_dtype=dtype,
            test_device=device,
            test_is_fp8=(quantized_compute or quantized_weight),
        )
        b_ref, b_test = None, None
        if bias:
            b_ref, b_test = make_reference_and_test_tensors(
                out_features,
                test_dtype=dtype,
                test_device=device,
            )
        dy_ref, dy_test = make_reference_and_test_tensors(
            out_shape,
            test_dtype=dtype,
            test_device=device,
            requires_grad=False,
        )

        # Plain PyTorch implementation
        y_ref = torch.nn.functional.linear(x_ref, w_ref, bias=b_ref)
        y_ref.backward(dy_ref)

        # Implementation with fusible operations
        recipe = make_recipe(quantization)
        with te.fp8_model_init(enabled=quantized_compute, recipe=recipe):
            model = te_ops.Sequential(
                te_ops.Linear(
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
        with te.fp8_autocast(enabled=quantized_compute, fp8_recipe=recipe):
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
        if quantized_compute:
            tols = dtype_tols(tex.DType.kFloat8E4M3)

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

    @pytest.mark.parametrize("bias", (False, True))
    @pytest.mark.parametrize("dtype", _dtypes)
    @pytest.mark.parametrize("quantization", (None, "fp8", "mxfp8"))
    def test_forward_linear_bias_add(
        self,
        *,
        bias: bool,
        weight_shape: tuple[int, int] = (128, 128),
        in_shape: Iterable[int] = (128, -1),
        dtype: torch.dtype,
        device: torch.device = "cuda",
        quantization: Optional[str],
        quantized_weight: bool = False,
    ) -> None:
        """Forward GEMM + bias + add"""

        # Make input and weight shapes consistent
        out_features, in_features = weight_shape
        in_shape = list(in_shape)[:-1] + [in_features]
        out_shape = in_shape[:-1] + [out_features]

        # Skip invalid configurations
        quantized_compute = quantization is not None
        maybe_skip_quantization(quantization, dims=in_shape, device=device)
        maybe_skip_quantization(quantization, dims=out_shape)
        if quantized_compute and dtype not in (torch.float16, torch.bfloat16):
            pytest.skip("FP8 GEMM is only supported with FP8, FP16, or BF16 output")

        # Random data
        x1_ref, x1_test = make_reference_and_test_tensors(
            in_shape,
            test_dtype=dtype,
            test_device=device,
            test_is_fp8=quantized_compute,
        )
        if isinstance(x1_test, QuantizedTensor):
            with torch.no_grad():
                x1_test = x1_test.dequantize().requires_grad_()
        w_ref, w_test = make_reference_and_test_tensors(
            (out_features, in_features),
            test_dtype=dtype,
            test_device=device,
            test_is_fp8=(quantized_compute or quantized_weight),
        )
        b_ref, b_test = None, None
        if bias:
            b_ref, b_test = make_reference_and_test_tensors(
                out_features,
                test_dtype=dtype,
                test_device=device,
            )
        x2_ref, x2_test = make_reference_and_test_tensors(
            out_shape,
            test_dtype=dtype,
            test_device=device,
        )
        dy_ref, dy_test = make_reference_and_test_tensors(
            out_shape,
            test_dtype=dtype,
            test_device=device,
            requires_grad=False,
        )

        # Plain PyTorch implementation
        y_ref = torch.nn.functional.linear(x1_ref, w_ref, bias=b_ref) + x2_ref
        y_ref.backward(dy_ref)

        # Implementation with fusible operations
        recipe = make_recipe(quantization)
        with te.fp8_model_init(enabled=quantized_weight, recipe=recipe):
            model = te_ops.Sequential(
                te_ops.Linear(
                    in_features,
                    out_features,
                    bias=bias,
                    device=device,
                    dtype=dtype,
                ),
                te_ops.AddInPlace(),
            )
        with torch.no_grad():
            model[0].weight.copy_(w_test)
            if bias:
                model[0].bias.copy_(b_test)
            del w_test
            del b_test
        with te.fp8_autocast(enabled=quantized_compute, fp8_recipe=recipe):
            y_test = model(x1_test, x2_test)
        y_test.backward(dy_test)

        # Check that forward operations have been fused
        forward_ops = model._module_groups[0]._forward_ops
        assert len(forward_ops) == 1
        assert isinstance(forward_ops[0][0], ForwardLinearBiasAdd)

        # Expected numerical error
        tols = dtype_tols(dtype)
        if dtype == torch.float32:
            tols = dtype_tols(torch.float16)  # TF32 GEMM
        if quantized_compute:
            tols = dtype_tols(tex.DType.kFloat8E4M3)

        # Check results
        y_test = y_test.to(dtype=torch.float64, device="cpu")
        dx1_test = x1_test.grad.to(dtype=torch.float64, device="cpu")
        dx2_test = x2_test.grad.to(dtype=torch.float64, device="cpu")
        dw_test = model[0].weight.grad.to(dtype=torch.float64, device="cpu")
        torch.testing.assert_close(y_test, y_ref, **tols)
        torch.testing.assert_close(dx1_test, x1_ref.grad, **tols)
        torch.testing.assert_close(dx2_test, x2_ref.grad, **tols)
        torch.testing.assert_close(dw_test, w_ref.grad, **tols)
        if bias:
            db_test = model[0].bias.grad.to(dtype=torch.float64, device="cpu")
            torch.testing.assert_close(db_test, b_ref.grad, **tols)

    @pytest.mark.parametrize("dtype", _dtypes)
    @pytest.mark.parametrize("quantization", (None, "fp8", "mxfp8"))
    def test_backward_linear_add(
        self,
        *,
        weight_shape: tuple[int, int] = (128, 128),
        in_shape: Iterable[int] = (128, -1),
        dtype: torch.dtype,
        device: torch.device = "cuda",
        quantization: Optional[str],
        quantized_weight: bool = False,
    ) -> None:
        """Backward dgrad GEMM + add"""

        # Make input and weight shapes consistent
        out_features, in_features = weight_shape
        in_shape = list(in_shape)[:-1] + [in_features]
        out_shape = in_shape[:-1] + [out_features]

        # Skip invalid configurations
        quantized_compute = quantization is not None
        maybe_skip_quantization(quantization, dims=in_shape, device=device)
        maybe_skip_quantization(quantization, dims=out_shape)
        if quantized_compute and dtype not in (torch.float16, torch.bfloat16):
            pytest.skip("FP8 GEMM is only supported with FP8, FP16, or BF16 output")

        # Random data
        x_ref, x_test = make_reference_and_test_tensors(
            in_shape,
            test_dtype=dtype,
            test_device=device,
            test_is_fp8=quantized_compute,
        )
        if isinstance(x_test, QuantizedTensor):
            with torch.no_grad():
                x_test = x_test.dequantize().requires_grad_()
        w_ref, w_test = make_reference_and_test_tensors(
            (out_features, in_features),
            test_dtype=dtype,
            test_device=device,
            test_is_fp8=(quantized_compute or quantized_weight),
        )
        dy1_ref, dy1_test = make_reference_and_test_tensors(
            out_shape,
            test_dtype=dtype,
            test_device=device,
            requires_grad=False,
        )
        dy2_ref, dy2_test = make_reference_and_test_tensors(
            out_shape,
            test_dtype=dtype,
            test_device=device,
            requires_grad=False,
        )

        # Plain PyTorch implementation
        y1_ref = torch.nn.functional.linear(x_ref, w_ref)
        y2_ref = x_ref
        (y1_ref * dy1_ref + y2_ref * dy2_ref).sum().backward()

        # Implementation with fusible operations
        recipe = make_recipe(quantization)
        with te.fp8_model_init(enabled=quantized_weight):
            model = te_ops.Sequential(
                te_ops.MakeExtraOutput(),
                te_ops.Linear(
                    in_features,
                    out_features,
                    bias=False,
                    device=device,
                    dtype=dtype,
                ),
            )
        with torch.no_grad():
            model[1].weight.copy_(w_test)
            del w_test
        with te.fp8_autocast(enabled=quantized_compute, fp8_recipe=recipe):
            y1_test, y2_test = model(x_test)
        (y1_test * dy1_test + y2_test * dy2_test).sum().backward()

        # Check that backward operations have been fused
        backward_ops = model._module_groups[0]._backward_ops
        assert len(backward_ops) == 1
        assert isinstance(backward_ops[0][0], BackwardLinearAdd)

        # Expected numerical error
        tols = dtype_tols(dtype)
        if dtype == torch.float32:
            tols = dtype_tols(torch.float16)  # TF32 GEMM
        if quantized_compute:
            tols = dtype_tols(tex.DType.kFloat8E4M3)

        # Check results
        y1_test = y1_test.to(dtype=torch.float64, device="cpu")
        y2_test = y2_test.to(dtype=torch.float64, device="cpu")
        dx_test = x_test.grad.to(dtype=torch.float64, device="cpu")
        dw_test = model[1].weight.grad.to(dtype=torch.float64, device="cpu")
        torch.testing.assert_close(y1_test, y1_ref, **tols)
        torch.testing.assert_close(y2_test, y2_ref, **tols)
        torch.testing.assert_close(dx_test, x_ref.grad, **tols)
        torch.testing.assert_close(dw_test, w_ref.grad, **tols)
