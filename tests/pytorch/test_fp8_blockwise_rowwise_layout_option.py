# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# See LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0
"""FP8 blockwise rowwise scale layout selection through Float8BlockQuantizer."""

import pickle

import pytest
import torch
import transformer_engine.pytorch as te
import transformer_engine_torch as tex

from transformer_engine.pytorch.dynamo.tensor_spec import TensorSpec
from transformer_engine.pytorch.tensor.float8_blockwise_tensor import Float8BlockQuantizer
from transformer_engine.pytorch.tensor.float8_blockwise_tensor import (
    Float8BlockwiseQTensor,
    _make_float8_blockwise_tensor_in_reduce_ex,
)
from transformer_engine.pytorch.tensor.storage.float8_blockwise_tensor_storage import (
    Float8BlockwiseQTensorStorage,
)


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
    torch.testing.assert_close(compact.dequantize(), gemm.dequantize(), rtol=0, atol=0)


def test_rowwise_layout_is_preserved_by_value_rebuild():
    compact = make_quantizer(tex.FP8BlockwiseRowwiseOption.ROWWISE_COMPACT, False)
    gemm = make_quantizer(tex.FP8BlockwiseRowwiseOption.ROWWISE_GEMM_READY, False)
    assert compact != gemm
    assert hash(compact) != hash(gemm)
    expression, globals_ = compact.__fx_repr__()
    rebuilt = eval(expression, dict(globals_))  # pylint: disable=eval-used
    assert rebuilt.rowwise_scale_layout == tex.FP8BlockwiseRowwiseOption.ROWWISE_COMPACT
    assert rebuilt.get_scale_shape((130, 384), False) == (130, 3)

    source = torch.randn(130, 384, dtype=torch.bfloat16, device="cuda")
    torch.testing.assert_close(
        rebuilt(source).dequantize(), compact(source).dequantize(), rtol=0, atol=0
    )


def test_grouped_quantizer_remains_gemm_ready():
    counts = torch.tensor([128, 256], device="cuda", dtype=torch.int64)
    rows = int(counts.sum())
    hidden = 384
    offsets = torch.cat((counts.new_zeros(1), counts.cumsum(0))) * hidden
    source = torch.randn(rows, hidden, device="cuda", dtype=torch.bfloat16)
    compact_quantizer = make_quantizer(tex.FP8BlockwiseRowwiseOption.ROWWISE_COMPACT, False)
    grouped = tex.group_quantize(
        source, compact_quantizer, counts.numel(), counts, tensor_offsets=offsets
    )
    expected_scales = torch.cat(
        [
            scales.T.contiguous().flatten()
            for scales in compact_quantizer(source)._rowwise_scale_inv.split([128, 256])
        ]
    )
    # Block128 grouped GEMM uses per-expert block-major scales without the
    # MXFP8-style 128x4 swizzle.
    assert not grouped._with_gemm_swizzled_scales
    assert grouped._block_scaling_rowwise_scale_layout == int(
        tex.FP8BlockwiseRowwiseOption.ROWWISE_GEMM_READY
    )
    torch.testing.assert_close(grouped.scale_inv, expected_scales, rtol=0, atol=0)


@pytest.mark.parametrize("rebuild", ["view", "reshape", "flatten", "pickle"])
def test_compact_layout_is_preserved_by_tensor_rebuild(rebuild):
    compact = make_quantizer(tex.FP8BlockwiseRowwiseOption.ROWWISE_COMPACT, False)(
        torch.randn(2, 128, 384, dtype=torch.bfloat16, device="cuda")
    )
    expected = compact.dequantize()
    if rebuild == "view":
        rebuilt = compact.view(256, 384)
        expected = expected.view(256, 384)
    elif rebuild == "reshape":
        rebuilt = compact.reshape(256, 384)
        expected = expected.reshape(256, 384)
    elif rebuild == "flatten":
        names, context = compact.__tensor_flatten__()
        inner = {name: getattr(compact, name) for name in names}
        rebuilt = compact.__tensor_unflatten__(inner, context, compact.shape, compact.stride())
    else:
        rebuilt = pickle.loads(pickle.dumps(compact))
    assert rebuilt._block_scaling_rowwise_scale_layout == int(
        tex.FP8BlockwiseRowwiseOption.ROWWISE_COMPACT
    )
    torch.testing.assert_close(rebuilt.dequantize(), expected, rtol=0, atol=0)


def test_storage_only_compact_dequantize_does_not_need_quantizer():
    rows, hidden = 128, 384
    data = torch.arange(rows * hidden, device="cuda", dtype=torch.int32)
    data = ((data % 32 + 1) * 8).to(torch.uint8).view(rows, hidden)
    scales = torch.linspace(0.5, 2.0, rows * (hidden // 128), device="cuda").reshape(
        rows, hidden // 128
    )
    storage = Float8BlockwiseQTensorStorage(
        data,
        scales,
        None,
        None,
        te.DType.kFloat8E4M3,
        None,
        False,
        block_scaling_rowwise_scale_layout=tex.FP8BlockwiseRowwiseOption.ROWWISE_COMPACT,
        fake_dtype=torch.bfloat16,
    )
    expected = (
        (
            data.view(torch.float8_e4m3fn).float().view(rows, hidden // 128, 128)
            * scales.unsqueeze(-1)
        )
        .reshape(rows, hidden)
        .bfloat16()
    )
    assert storage._quantizer is None
    torch.testing.assert_close(storage.dequantize(), expected, rtol=0, atol=0)


@pytest.mark.parametrize("internal", [False, True])
def test_compact_layout_is_preserved_by_tensor_spec(internal):
    quantizer = make_quantizer(tex.FP8BlockwiseRowwiseOption.ROWWISE_COMPACT, False)
    quantizer.internal = internal
    tensor = TensorSpec(
        shape=(128, 384),
        dtype=torch.bfloat16,
        quantizer=quantizer,
        device=torch.device("cuda"),
    ).create_tensor()
    assert isinstance(
        tensor, Float8BlockwiseQTensorStorage if internal else Float8BlockwiseQTensor
    )
    assert tensor._rowwise_scale_inv.shape == (128, 3)
    assert tensor._block_scaling_rowwise_scale_layout == int(
        tex.FP8BlockwiseRowwiseOption.ROWWISE_COMPACT
    )


def test_legacy_pickle_payload_defaults_to_gemm_ready_layout():
    quantizer = make_quantizer(tex.FP8BlockwiseRowwiseOption.ROWWISE_GEMM_READY, False)
    rows, hidden = 128, 384
    data = torch.zeros(rows, hidden, dtype=torch.uint8, device="cuda")
    scales = torch.ones(hidden // 128, rows, dtype=torch.float32, device="cuda")
    # Before block_scaling_rowwise_scale_layout was added, data_format was the
    # last positional argument in the reduction payload.
    legacy_args = (
        torch.Size((rows, hidden)),
        data,
        scales,
        None,
        None,
        te.DType.kFloat8E4M3,
        torch.bfloat16,
        quantizer,
        False,
        None,
    )
    tensor = _make_float8_blockwise_tensor_in_reduce_ex(*legacy_args)
    assert tensor._block_scaling_rowwise_scale_layout == int(
        tex.FP8BlockwiseRowwiseOption.ROWWISE_GEMM_READY
    )
    torch.testing.assert_close(tensor.dequantize(), torch.zeros_like(data, dtype=torch.bfloat16))


def test_legacy_pickle_reconstruction_classmethod():
    quantizer = make_quantizer(tex.FP8BlockwiseRowwiseOption.ROWWISE_GEMM_READY, False)
    rows, hidden = 128, 384
    tensor = Float8BlockwiseQTensor._make_in_reduce_ex(
        torch.Size((rows, hidden)),
        torch.zeros(rows, hidden, dtype=torch.uint8, device="cuda"),
        torch.ones(hidden // 128, rows, dtype=torch.float32, device="cuda"),
        None,
        None,
        te.DType.kFloat8E4M3,
        torch.bfloat16,
        quantizer,
        False,
        None,
    )
    assert tensor._block_scaling_rowwise_scale_layout == int(
        tex.FP8BlockwiseRowwiseOption.ROWWISE_GEMM_READY
    )
