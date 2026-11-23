# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# See LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0
"""FP8 blockwise rowwise scale layout selection through Float8BlockQuantizer."""

import pytest
import torch
import transformer_engine.pytorch as te
import transformer_engine_torch as tex

from transformer_engine.pytorch.tensor.float8_blockwise_tensor import Float8BlockQuantizer


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9,
    reason="FP8 blockwise quantization requires Hopper or newer",
)


def make_quantizer(option, columnwise):
    return Float8BlockQuantizer(
        te.DType.kFloat8E4M3,
        rowwise=True,
        columnwise=columnwise,
        block_scaling_dim=1,
        force_pow_2_scales=False,
        rowwise_scale_layout=option,
    )


@pytest.mark.parametrize("rows,hidden", [(256, 256), (130, 384), (512, 640)])
def test_rowwise_layout_option_single_tensor(rows, hidden):
    source = torch.randn(rows, hidden, dtype=torch.bfloat16, device="cuda")
    gemm = make_quantizer(tex.FP8BlockwiseRowwiseOption.ROWWISE_GEMM_READY, True)(source)
    compact = make_quantizer(tex.FP8BlockwiseRowwiseOption.ROWWISE_COMPACT, True)(source)
    assert compact._rowwise_scale_inv.shape == (rows, hidden // 128)
    torch.testing.assert_close(compact._rowwise_data, gemm._rowwise_data, rtol=0, atol=0)
    torch.testing.assert_close(
        compact._rowwise_scale_inv,
        gemm._rowwise_scale_inv[:, :rows].T,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(compact._columnwise_data, gemm._columnwise_data, rtol=0, atol=0)
    torch.testing.assert_close(
        compact._columnwise_scale_inv, gemm._columnwise_scale_inv, rtol=0, atol=0
    )
