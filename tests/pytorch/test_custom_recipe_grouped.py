# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Grouped GEMM with custom (reference) quantizers.

These drive the two custom-recipe grouped entry points with the reference NVFP4/MXFP4 quantizer,
which is the only in-tree way to reach the Python `qgemm` implementations: the module and ops
grouped paths only ever build custom operands for the discrete (list) form, and the reference
quantizer is what makes a `qgemm` callable without a compiled custom kernel.

Covered here, because nothing else in the suite reaches it:
  * `_custom_grouped_gemm` (list form) — operand roles, m_splits slicing, per-expert bias, and the
    `qresult_x`/`qresult_w` hand-off that non-pow-2 NVFP4 needs.
  * `_custom_grouped_tensor_gemm` (GroupedTensorStorage form) — per-group offsets and block scales.
  * The refusals: dgrad/wgrad, non-TN layouts, malformed bias lists, and the non-pow-2 recipe that
    this layout cannot serve.
"""

import pytest
import torch

import transformer_engine.pytorch as te
from transformer_engine.pytorch.cpp_extensions import (
    general_grouped_gemm,
    general_grouped_gemm_for_grouped_tensor,
)
from transformer_engine.pytorch.custom_recipes import reference_utils
from transformer_engine.pytorch.custom_recipes.gemm import GEMMType, MMParams
from transformer_engine.pytorch.custom_recipes.reference_nvfp4 import NVFP4QuantizerRef
from transformer_engine.pytorch.tensor.storage.grouped_tensor_storage import GroupedTensorStorage

DEVICE = "cuda"
DTYPE = torch.bfloat16
SPLITS = [16, 32, 8, 24]
K, N = 256, 128
TOTAL_M = sum(SPLITS)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="grouped GEMM requires a CUDA device"
)


def _quantizer(pow_2_scales: bool) -> NVFP4QuantizerRef:
    # MXFP4 (E8M0 scales) is defined on 1x32 tiles, NVFP4 on 1x16.
    return NVFP4QuantizerRef(
        dtype=reference_utils.Fp4Formats.E2M1,
        pow_2_scales=pow_2_scales,
        quant_tile_shape=(1, 32) if pow_2_scales else (1, 16),
    )


def _operands():
    torch.manual_seed(0)
    weights = [torch.randn(N, K, device=DEVICE, dtype=DTYPE) for _ in SPLITS]
    inputs = [torch.randn(m, K, device=DEVICE, dtype=DTYPE) for m in SPLITS]
    biases = [torch.randn(N, device=DEVICE, dtype=DTYPE) for _ in SPLITS]
    return weights, inputs, biases


def _per_group_reference(quantizer, qweights, qinputs, biases=None) -> torch.Tensor:
    """One qgemm per expert, placed at that expert's row offset: what the dispatch must produce."""
    expected = torch.empty(TOTAL_M, N, device=DEVICE, dtype=DTYPE)
    offset = 0
    for i, m in enumerate(SPLITS):
        expected[offset : offset + m] = quantizer.qgemm(
            qinputs[i].data[:m],
            qweights[i].data,
            MMParams(out_dtype=DTYPE, use_split_accumulator=False),
            DTYPE,
            qinputs[i].scale[:m],
            qweights[i].scale,
            biases[i] if biases is not None else None,
            gemm_type=GEMMType.FPROP,
            qresult_x=qinputs[i],
            qresult_w=qweights[i],
        )
        offset += m
    return expected


def _assert_close_to_unquantized(actual: torch.Tensor, weights, inputs, biases=None) -> None:
    """4-bit operands are coarse, so only gross (operand-role) mistakes are ruled out here."""
    offset = 0
    for i, m in enumerate(SPLITS):
        expected = torch.nn.functional.linear(
            inputs[i], weights[i], biases[i] if biases is not None else None
        )
        block = actual[offset : offset + m]
        deviation = (block - expected).norm() / expected.norm().clamp_min(1e-6)
        assert deviation < 0.25, f"expert {i}: quantized result deviates {deviation:.1%}"
        offset += m


@pytest.mark.parametrize("use_bias", [False, True], ids=["no-bias", "bias"])
@pytest.mark.parametrize("pow_2_scales", [True, False], ids=["mxfp4", "nvfp4"])
def test_custom_grouped_gemm_list_form(use_bias: bool, pow_2_scales: bool) -> None:
    """`general_grouped_gemm` dispatches per-expert custom operands to the quantizer's qgemm."""
    quantizer = _quantizer(pow_2_scales)
    weights, inputs, biases = _operands()
    qweights = [quantizer(w) for w in weights]
    qinputs = [quantizer(x) for x in inputs]

    out = torch.empty(TOTAL_M, N, device=DEVICE, dtype=DTYPE)
    general_grouped_gemm(
        qweights,
        qinputs,
        [out],
        [None] * len(SPLITS),
        DTYPE,
        single_output=True,
        m_splits=SPLITS,
        bias=biases if use_bias else None,
        use_bias=use_bias,
    )

    torch.testing.assert_close(
        out, _per_group_reference(quantizer, qweights, qinputs, biases if use_bias else None)
    )
    _assert_close_to_unquantized(out, weights, inputs, biases if use_bias else None)


def test_custom_grouped_tensor_gemm() -> None:
    """`general_grouped_gemm_for_grouped_tensor` slices groups out of GroupedTensorStorage."""
    quantizer = _quantizer(pow_2_scales=True)
    weights, inputs, _ = _operands()
    qweights = [quantizer(w) for w in weights]
    qinputs = [quantizer(x) for x in inputs]

    row_offsets = [0]
    for m in SPLITS:
        row_offsets.append(row_offsets[-1] + m)
    weight_offsets = [i * N for i in range(len(SPLITS) + 1)]

    grouped_weights = GroupedTensorStorage(
        shape=(len(SPLITS) * N, K),
        dtype=DTYPE,
        num_tensors=len(SPLITS),
        quantizer=quantizer,
        data=torch.cat([t.data for t in qweights], dim=0),
        scale_inv=torch.cat([t.scale for t in qweights], dim=0),
        offsets=weight_offsets,
        scale_inv_offsets=weight_offsets,
    )
    grouped_inputs = GroupedTensorStorage(
        shape=(TOTAL_M, K),
        dtype=DTYPE,
        num_tensors=len(SPLITS),
        quantizer=quantizer,
        data=torch.cat([t.data for t in qinputs], dim=0),
        scale_inv=torch.cat([t.scale for t in qinputs], dim=0),
        offsets=row_offsets,
        scale_inv_offsets=row_offsets,
    )
    grouped_out = GroupedTensorStorage(
        shape=(TOTAL_M, N),
        dtype=DTYPE,
        num_tensors=len(SPLITS),
        quantizer=None,
        data=torch.empty(TOTAL_M, N, device=DEVICE, dtype=DTYPE),
        offsets=row_offsets,
    )

    general_grouped_gemm_for_grouped_tensor(
        grouped_weights, grouped_inputs, grouped_out, layout="TN"
    )

    torch.testing.assert_close(
        grouped_out.rowwise_data,
        _per_group_reference(quantizer, qweights, qinputs),
    )
    _assert_close_to_unquantized(grouped_out.rowwise_data, weights, inputs)


def test_custom_grouped_tensor_gemm_refuses_nvfp4() -> None:
    """This layout carries no per-group quantized tensors, so a non-pow-2 recipe must refuse."""
    quantizer = _quantizer(pow_2_scales=False)
    weights, inputs, _ = _operands()
    qweights = [quantizer(w) for w in weights]
    qinputs = [quantizer(x) for x in inputs]
    row_offsets = [0]
    for m in SPLITS:
        row_offsets.append(row_offsets[-1] + m)
    weight_offsets = [i * N for i in range(len(SPLITS) + 1)]

    grouped_weights = GroupedTensorStorage(
        shape=(len(SPLITS) * N, K),
        dtype=DTYPE,
        num_tensors=len(SPLITS),
        quantizer=quantizer,
        data=torch.cat([t.data for t in qweights], dim=0),
        scale_inv=torch.cat([t.scale for t in qweights], dim=0),
        offsets=weight_offsets,
        scale_inv_offsets=weight_offsets,
    )
    grouped_inputs = GroupedTensorStorage(
        shape=(TOTAL_M, K),
        dtype=DTYPE,
        num_tensors=len(SPLITS),
        quantizer=quantizer,
        data=torch.cat([t.data for t in qinputs], dim=0),
        scale_inv=torch.cat([t.scale for t in qinputs], dim=0),
        offsets=row_offsets,
        scale_inv_offsets=row_offsets,
    )
    grouped_out = GroupedTensorStorage(
        shape=(TOTAL_M, N),
        dtype=DTYPE,
        num_tensors=len(SPLITS),
        quantizer=None,
        data=torch.empty(TOTAL_M, N, device=DEVICE, dtype=DTYPE),
        offsets=row_offsets,
    )
    with pytest.raises(ValueError, match="qresult"):
        general_grouped_gemm_for_grouped_tensor(
            grouped_weights, grouped_inputs, grouped_out, layout="TN"
        )


def test_custom_grouped_gemm_refuses_unsupported_contracts() -> None:
    """dgrad/wgrad, non-TN layouts and malformed bias lists are refused before any work."""
    quantizer = _quantizer(pow_2_scales=True)
    weights, inputs, biases = _operands()
    qweights = [quantizer(w) for w in weights]
    qinputs = [quantizer(x) for x in inputs]
    out = torch.empty(TOTAL_M, N, device=DEVICE, dtype=DTYPE)

    def call(**kwargs):
        general_grouped_gemm(
            qweights,
            qinputs,
            [out],
            [None] * len(SPLITS),
            DTYPE,
            single_output=True,
            m_splits=SPLITS,
            **kwargs,
        )

    with pytest.raises(NotImplementedError, match="fprop only"):
        call(grad=True)
    with pytest.raises(NotImplementedError, match="TN layout"):
        call(layout="NN")
    with pytest.raises(ValueError, match="per-group bias"):
        call(use_bias=True)
    with pytest.raises(ValueError, match="expected 4 biases"):
        call(bias=biases[:2], use_bias=True)


def test_reference_quantizer_in_place_update_contract() -> None:
    """`quantize_` refreshes every field a later GEMM reads, honours noop_flag, returns self."""
    quantizer = _quantizer(pow_2_scales=False)
    tensor = torch.randn(32, K, device=DEVICE, dtype=DTYPE)
    qresult = quantizer(tensor)
    before_amax = qresult.global_amax_row.clone()

    assert qresult.quantize_(tensor * 4.0) is qresult
    assert not torch.equal(before_amax, qresult.global_amax_row), "stale global amax after refresh"
    assert tuple(qresult.original_shape) == (32, K)

    snapshot = qresult.data.clone()
    qresult.quantize_(tensor * 0.25, noop_flag=torch.tensor([1.0], device=DEVICE))
    assert torch.equal(snapshot, qresult.data), "noop_flag was ignored"


def test_reference_qgemm_applies_bias_in_fp32_epilogue() -> None:
    """The bias is an epilogue term, so qgemm(bias=x) == qgemm() + x rather than an error."""
    quantizer = _quantizer(pow_2_scales=True)
    activations = quantizer(torch.randn(16, K, device=DEVICE, dtype=DTYPE))
    weights = quantizer(torch.randn(N, K, device=DEVICE, dtype=DTYPE))
    bias = torch.randn(N, device=DEVICE, dtype=DTYPE)
    params = MMParams(out_dtype=DTYPE, use_split_accumulator=False)

    plain = quantizer.qgemm(
        activations.data,
        weights.data,
        params,
        DTYPE,
        activations.scale,
        weights.scale,
        gemm_type=GEMMType.FPROP,
    )
    with_bias = quantizer.qgemm(
        activations.data,
        weights.data,
        params,
        DTYPE,
        activations.scale,
        weights.scale,
        bias,
        gemm_type=GEMMType.FPROP,
    )
    torch.testing.assert_close(with_bias, plain + bias, rtol=2e-2, atol=2e-2)
