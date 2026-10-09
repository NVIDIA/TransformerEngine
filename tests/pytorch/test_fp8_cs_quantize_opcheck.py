# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import torch

import transformer_engine.pytorch.onnx_extensions  # noqa: F401


def test_fp8_cs_quantize_fake_tensor_metadata():
    x = torch.randn(16, 16, device="cuda", dtype=torch.float32)

    torch.library.opcheck(
        torch.ops.tex.fp8_cs_quantize.default,
        (x,),
        test_utils=("test_faketensor",),
    )
