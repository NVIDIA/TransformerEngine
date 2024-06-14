# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

from typing import Tuple

import pytest
import torch

import transformer_engine.pytorch as te

# Model names for test_torch_dynamo
_model_factory = {
    "Linear": [(lambda: te.Linear(16, 16)), [16, 16]],
    "LayerNorm": [(lambda: te.LayerNorm(16)), [16, 16]],
    "LayerNormLinear": [(lambda: te.LayerNormLinear(16, 16)), [16, 16]],
    "LayerNormMLP": [(lambda: te.LayerNormMLP(16, 16)), [16, 16]],
    "TransformerLayer": [(lambda: te.TransformerLayer(128, 128, 2)), [4, 1, 128]],
}


@pytest.mark.skipif(torch.__version__ < "2", reason="torch.compile not available")
@pytest.mark.parametrize("model_name", list(_model_factory.keys()))
def test_torch_dynamo(model_name: str):
    """Test compatibility with Torch Dynamo

    Construct model, optimize with Torch Dynamo, and perform a single
    forward and backward pass.

    """

    # Helper function to construct tensor with default options
    def make_tensor(
        dims: Tuple[int],
        dtype: torch.dtype = torch.float32,
        device: torch.device = "cuda",
        requires_grad: bool = True,
        **kwargs,
    ):
        return torch.zeros(
            dims,
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
            **kwargs,
        )

    # Construct model and input tensors
    model_builder, input_builder = _model_factory[model_name]
    model = model_builder()
    inputs = [make_tensor(input_builder)]

    # Optimize model with TorchDynamo
    torch.compile(model)

    # Forward and backward pass
    out = model(*inputs)
    out.backward(torch.zeros_like(out))
