# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

from __future__ import annotations

import math

import pytest
import torch

import transformer_engine.pytorch as te
import transformer_engine.pytorch.fuser as te_fuser
from transformer_engine.pytorch.fp8 import FP8GlobalStateManager
from transformer_engine.pytorch.float8_tensor import Float8Tensor
from transformer_engine.pytorch.utils import is_bf16_compatible

# Check if FP8 is supported
fp8_available, reason_for_no_fp8 = FP8GlobalStateManager.is_fp8_available()

# Supported data types
_dtypes: list[torch.dtype] = [torch.float32, torch.float16]
if is_bf16_compatible():  # bf16 requires sm_80 or higher
    _dtypes.append(torch.bfloat16)

# Supported devices
_devices: list[torch.device] = [torch.device("cpu"), torch.device("cuda")]


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
            x_test = x_ref.detach().clone()
            if fp8:
                x_test = Float8Tensor.to_float8(x_test)
                x_test = x_test.contiguous(memory_format=memory_format)
                x_ref.copy_(x_test.from_float8())
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
    def test_bias(
        self,
        size: int,
        in_shape: Iterable[int],
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        """Bias operation"""

        # Random data
        in_shape = list(in_shape) + [size]
        with torch.no_grad():
            x_ref = torch.rand(in_shape, dtype=dtype, device=device)
            b_ref = torch.rand(size, dtype=dtype, device=device)
            x_test = x_ref.detach().clone()
            dy = torch.rand_like(x_ref)
        x_ref.requires_grad_()
        b_ref.requires_grad_()
        x_test.requires_grad_()

        # Plain PyTorch implementation
        y_ref = x_ref + b_ref.reshape([1] * (len(in_shape) - 1) + [size])
        y_ref.backward(dy)

        # Implementation with fusable operation
        op = te_fuser.ops.Bias(size, device=device, dtype=dtype)
        with torch.no_grad():
            op.bias.copy_(b_ref)
        y_test = op(x_test)
        y_test.backward(dy)

        # Check results
        torch.testing.assert_close(y_test, y_ref)
        torch.testing.assert_close(x_test.grad, x_ref.grad)
        torch.testing.assert_close(op.bias.grad, b_ref.grad)
