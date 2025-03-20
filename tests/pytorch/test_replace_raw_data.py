# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import pytest
import torch


from transformer_engine.pytorch.fp8 import FP8GlobalStateManager
from transformer_engine.pytorch.tensor import replace_raw_data
from transformer_engine.pytorch.tensor.float8_tensor import Float8Quantizer
from transformer_engine_torch import DType as TE_DType

fp8_available, reason_for_no_fp8 = FP8GlobalStateManager.is_fp8_available()


@pytest.mark.skipif(not fp8_available, reason=reason_for_no_fp8)
def test_replace_raw_data_for_float8tensor():
    fp8_quantizer = Float8Quantizer(
        scale=torch.empty(1, dtype=torch.float32, device="cuda"),
        amax=torch.empty(1, dtype=torch.float32, device="cuda"),
        fp8_dtype=TE_DType.kFloat8E4M3,
        rowwise=True,
        columnwise=True,
    )
    fp8_tensor = fp8_quantizer.make_empty([128, 128], dtype=torch.bfloat16, device="cuda")

    attrs_to_check = ["_quantizer", "_fp8_dtype", "_scale_inv", "_transpose", "_transpose_invalid"]
    attrs = {}
    for attr in attrs_to_check:
        attrs[attr] = getattr(fp8_tensor, attr)

    old_data = fp8_tensor._data
    new_data = torch.empty_like(old_data)
    replace_raw_data(fp8_tensor, new_data)

    # Make sure the new_data is properly assigned.
    assert fp8_tensor._data.data_ptr() != old_data.data_ptr()
    assert fp8_tensor._data.data_ptr() == new_data.data_ptr()
    # Make sure the values are not changed.
    torch.testing.assert_close(old_data, fp8_tensor._data, atol=0, rtol=0)
    # Make sure other attributes are not changed (totally identical)
    for attr in attrs_to_check:
        assert id(getattr(fp8_tensor, attr)) == id(attrs[attr])
