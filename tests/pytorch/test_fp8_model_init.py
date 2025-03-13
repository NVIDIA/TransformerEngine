# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import pytest
import torch

import transformer_engine.pytorch as te
from transformer_engine.pytorch.tensor import QuantizedTensor
from transformer_engine.pytorch.fp8 import FP8GlobalStateManager, fp8_model_init

# Check if FP8 is supported
fp8_available, reason_for_no_fp8 = FP8GlobalStateManager.is_fp8_available()


@pytest.mark.skipif(not fp8_available, reason=reason_for_no_fp8)
class TestFP8ModelInit:

    @staticmethod
    def setup_class(cls) -> None:
        # Configure RNG
        seed = 1234
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

    def test_default(self) -> None:
        """Test default parameters of fp8_model_init"""
        with fp8_model_init():
            model = te.Linear(768, 768)

        weight = model.weight

        assert isinstance(weight, QuantizedTensor), "Weight should be QuantizedTensor"
        assert not hasattr(
            weight, "._high_precision_init_val"
        ), "_high_precision_init_val should not exist"
        assert not hasattr(
            weight, "get_high_precision_init_val"
        ), "get_high_precision_init_val() should not exist"
        assert not hasattr(
            weight, "clear_high_precision_init_val"
        ), "clear_high_precision_init_val() should not exist"

    def test_preserve_high_precision_init_val(self) -> None:
        """Test fp8_model_init with preserve_high_precision_init_val=True"""
        with fp8_model_init(preserve_high_precision_init_val=True):
            model = te.Linear(768, 768)

        weight = model.weight

        assert isinstance(weight, QuantizedTensor), "Weight should be QuantizedTensor"
        assert hasattr(weight, "_high_precision_init_val"), "_high_precision_init_val not found"
        assert hasattr(
            weight, "get_high_precision_init_val"
        ), "get_high_precision_init_val() not found"
        assert hasattr(
            weight, "clear_high_precision_init_val"
        ), "clear_high_precision_init_val() not found"

        high_precision = weight.get_high_precision_init_val()
        assert high_precision.device.type == "cpu", "high_precision_init_val is not on the CPU"

        new_weight = weight._get_quantizer().make_empty(
            shape=weight.shape, dtype=weight.dtype, device=weight.device
        )
        weight._get_quantizer().update_quantized(high_precision.to(weight.device), new_weight)

        torch.testing.assert_close(
            new_weight.dequantize(dtype=weight.dtype),
            weight.dequantize(dtype=weight.dtype),
            rtol=0,
            atol=0,
        )

        weight.clear_high_precision_init_val()
        assert (
            weight.get_high_precision_init_val() is None
        ), "clear_high_precision_init_val() not work"
        assert not hasattr(
            weight, "._high_precision_init_val"
        ), "clear_high_precision_init_val() not work"
