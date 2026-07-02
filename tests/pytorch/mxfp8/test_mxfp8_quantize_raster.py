# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import pytest
import torch
import transformer_engine.pytorch as te
import transformer_engine_torch as tex
from transformer_engine.pytorch import MXFP8Quantizer

recipe_available, reason_for_no_recipe = te.is_mxfp8_available(return_reason=True)


def _rowwise_quantize(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    quantizer = MXFP8Quantizer(
        fp8_dtype=te.DType.kFloat8E4M3,
        rowwise=True,
        columnwise=False,
    )
    y = quantizer(x)
    return y._rowwise_data.view(dtype=torch.uint8), y._rowwise_scale_inv


def _rowwise_quantize_padded_reference(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    padded_cols = ((x.size(1) + 127) // 128) * 128
    x_padded = torch.zeros((x.size(0), padded_cols), dtype=x.dtype, device=x.device)
    x_padded[:, : x.size(1)] = x

    q_ref, s_ref = _rowwise_quantize(x_padded)
    valid_scale_cols = x.size(1) // 32
    return q_ref[:, : x.size(1)].contiguous(), s_ref[: x.size(0), :valid_scale_cols].contiguous()


@pytest.mark.skipif(not recipe_available, reason=reason_for_no_recipe)
def test_mxfp8_generic_quantize_reverse_raster_preserves_values() -> None:
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    # N=96 avoids the specialized rowwise cast-only path, exercising the generic TMA path.
    x = torch.randn((320, 96), dtype=torch.bfloat16, device="cuda")

    q, s = _rowwise_quantize(x)
    q_ref, s_ref = _rowwise_quantize_padded_reference(x)

    torch.testing.assert_close(q, q_ref, atol=0.0, rtol=0.0)
    torch.testing.assert_close(s[: x.size(0), : x.size(1) // 32], s_ref, atol=0.0, rtol=0.0)


@pytest.mark.skipif(not recipe_available, reason=reason_for_no_recipe)
def test_mxfp8_grouped_quantize_reverse_raster_preserves_values() -> None:
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)

    split_sections = [128, 256, 128]
    x = torch.randn((sum(split_sections), 96), dtype=torch.bfloat16, device="cuda")
    split_section_tensor = torch.tensor(split_sections, dtype=torch.int64, device="cuda")
    quantizer = MXFP8Quantizer(
        fp8_dtype=te.DType.kFloat8E4M3,
        rowwise=True,
        columnwise=False,
    )

    grouped_output = tex.group_quantize(x, quantizer, len(split_sections), split_section_tensor)
    outputs = grouped_output.split_into_quantized_tensors()

    for x_chunk, output in zip(torch.split(x, split_sections), outputs):
        q_ref, s_ref = _rowwise_quantize_padded_reference(x_chunk)
        q = output._rowwise_data.view(dtype=torch.uint8)
        s = output._rowwise_scale_inv

        torch.testing.assert_close(q, q_ref, atol=0.0, rtol=0.0)
        torch.testing.assert_close(s, s_ref, atol=0.0, rtol=0.0)
