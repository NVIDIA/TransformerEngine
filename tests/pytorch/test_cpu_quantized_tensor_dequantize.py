# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Tests for CPU QuantizedTensor dequantize (issue #3294).

Verifies that dequantizing a QuantizedTensor on CPU does not cause
a RecursionError from __torch_dispatch__ re-entering dequantize().
"""

import pytest
import torch

# Skip the entire module when CUDA is not available.
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.fixture(params=["nvfp4", "mxfp8"])
def quantizer(request):
    """Yield an NVFP4 or MXFP8 quantizer."""
    if request.param == "nvfp4":
        from transformer_engine.pytorch import NVFP4Quantizer

        return NVFP4Quantizer()
    else:
        from transformer_engine.pytorch import MXFP8Quantizer

        return MXFP8Quantizer()


def _quantize_cpu(quantizer, shape=(128, 128)):
    """Quantize a random tensor and move the result to CPU."""
    x = torch.randn(shape, device="cuda")
    xq = quantizer(x)
    return xq.cpu()


class TestCPUQuantizedTensorDequantize:
    """Regression tests for issue #3294."""

    def test_dequantize_no_recursion(self, quantizer):
        """Calling dequantize() on a CPU QuantizedTensor must not recurse."""
        xq_cpu = _quantize_cpu(quantizer)
        result = xq_cpu.dequantize()
        assert isinstance(result, torch.Tensor)
        assert result.device.type == "cpu"

    def test_arithmetic_no_recursion(self, quantizer):
        """Arithmetic on a CPU QuantizedTensor triggers dequantize internally."""
        xq_cpu = _quantize_cpu(quantizer)
        result = xq_cpu + 0
        assert isinstance(result, torch.Tensor)
        assert result.device.type == "cpu"

    def test_roundtrip_values(self, quantizer):
        """Dequantized CPU values should match dequantized CUDA values."""
        x = torch.randn(128, 128, device="cuda")
        xq = quantizer(x)
        ref = xq.dequantize().cpu()
        xq_cpu = xq.cpu()
        out = xq_cpu.dequantize()
        torch.testing.assert_close(out, ref)

    def test_float_cast_no_recursion(self, quantizer):
        """.float() on a CPU QuantizedTensor must not recurse."""
        xq_cpu = _quantize_cpu(quantizer)
        result = xq_cpu.float()
        assert result.dtype == torch.float32
        assert result.device.type == "cpu"
