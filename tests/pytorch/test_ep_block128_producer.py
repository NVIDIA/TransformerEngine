# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# See LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0
"""Compare compact EP quantization with native GEMM-ready blockwise quantization."""

import pytest
import torch
import transformer_engine.pytorch  # Load the common library before the extension.
import transformer_engine_torch as tex

from transformer_engine.common.recipe import Float8BlockScaling
from transformer_engine.pytorch.ep import (
    _quantize_float8_blockwise,
    ep_quantize_float8_blockwise_dual_scale,
)
from transformer_engine.pytorch.tensor.float8_blockwise_tensor import Float8BlockQuantizer


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9,
    reason="EP block128 producer requires Hopper or newer",
)


@pytest.mark.parametrize("rows", [128, 256, 512])
@pytest.mark.parametrize("hidden", [4096, 8192])
@pytest.mark.parametrize("power_of_two", [False, True])
@pytest.mark.parametrize("zero", [False, True])
def test_producer_quantization_matches_native(rows, hidden, power_of_two, zero):
    torch.manual_seed(123)
    source = torch.randn(rows, hidden, device="cuda", dtype=torch.bfloat16)
    if zero:
        source.zero_()
    quantizer = Float8BlockQuantizer(
        tex.DType.kFloat8E4M3,
        rowwise=True,
        columnwise=True,
        block_scaling_dim=1,
        force_pow_2_scales=power_of_two,
    )
    native = quantizer.quantize(source)
    compact_quantizer = Float8BlockQuantizer(
        tex.DType.kFloat8E4M3,
        rowwise=True,
        columnwise=True,
        block_scaling_dim=1,
        force_pow_2_scales=power_of_two,
        rowwise_scale_layout=tex.FP8BlockwiseRowwiseOption.ROWWISE_COMPACT,
    )
    compact = compact_quantizer.quantize(source)
    assert compact._rowwise_scale_inv.shape == (rows, hidden // 128)
    assert compact._rowwise_scale_inv.is_contiguous()
    torch.testing.assert_close(compact._rowwise_data, native._rowwise_data, rtol=0, atol=0)
    torch.testing.assert_close(
        compact._rowwise_scale_inv, native._rowwise_scale_inv[:, :rows].T, rtol=0, atol=0
    )
    torch.testing.assert_close(compact._columnwise_data, native._columnwise_data, rtol=0, atol=0)
    torch.testing.assert_close(
        compact._columnwise_scale_inv, native._columnwise_scale_inv, rtol=0, atol=0
    )


@pytest.mark.parametrize("rows", [0, 128, 129])
@pytest.mark.parametrize("hidden", [4096, 8192])
def test_wire_only_does_not_allocate_columnwise(rows, hidden):
    source = torch.randn(rows, hidden, device="cuda", dtype=torch.bfloat16)
    recipe = Float8BlockScaling()
    wire_only, wire = _quantize_float8_blockwise(source, recipe)
    shared, shared_wire = ep_quantize_float8_blockwise_dual_scale(source, recipe)
    assert wire_only._columnwise_data is None
    assert wire_only._columnwise_scale_inv is None
    assert shared._columnwise_data is not None
    assert shared._columnwise_scale_inv is not None
    torch.testing.assert_close(wire_only._rowwise_data, shared._rowwise_data, rtol=0, atol=0)
    torch.testing.assert_close(
        wire_only._rowwise_scale_inv, shared._rowwise_scale_inv, rtol=0, atol=0
    )
    torch.testing.assert_close(wire, shared_wire, rtol=0, atol=0)
