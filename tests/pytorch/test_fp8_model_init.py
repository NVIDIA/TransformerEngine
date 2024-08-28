# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import pytest
import torch

import transformer_engine.pytorch as te
from transformer_engine.pytorch.float8_tensor import Float8Tensor
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

        assert isinstance(model.weight, Float8Tensor), "Weight should be Float8Tensor"
        assert not hasattr(
            model.weight, "._high_precision_init_val"
        ), "_high_precision_init_val should not exist"
        assert not hasattr(
            model.weight, "get_high_precision_init_val"
        ), "get_high_precision_init_val() should not exist"
        assert not hasattr(
            model.weight, "clear_high_precision_init_val"
        ), "clear_high_precision_init_val() should not exist"

    def test_preserve_high_precision_init_val(self) -> None:
        """Test fp8_model_init with preserve_high_precision_init_val=True"""
        with fp8_model_init(preserve_high_precision_init_val=True):
            model = te.Linear(768, 768)

        assert isinstance(model.weight, Float8Tensor), "Weight should be Float8Tensor"
        assert hasattr(
            model.weight, "_high_precision_init_val"
        ), "_high_precision_init_val not found"
        assert hasattr(
            model.weight, "get_high_precision_init_val"
        ), "get_high_precision_init_val() not found"
        assert hasattr(
            model.weight, "clear_high_precision_init_val"
        ), "clear_high_precision_init_val() not found"

        high_precision = model.weight.get_high_precision_init_val()
        assert high_precision.device.type == "cpu", "high_precision_init_val is not on the CPU"

        new_fp8 = Float8Tensor.to_float8(
            high_precision.to(model.weight.device),
            fp8_meta=model.weight._fp8_meta,
            fp8_meta_index=model.weight._fp8_meta_index,
            amax=torch.empty(1, device="cuda"),  # Dummy amax to avoid overwriting history.
        )
        assert torch.all(
            new_fp8._data == model.weight._data
        ), "high_precision_init_val and model.weight are not equal"

        model.weight.clear_high_precision_init_val()
        assert (
            model.weight.get_high_precision_init_val() is None
        ), "clear_high_precision_init_val() not work"
        assert not hasattr(
            model.weight, "._high_precision_init_val"
        ), "clear_high_precision_init_val() not work"
