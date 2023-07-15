# Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

from typing import Tuple

import pytest
import torch

import transformer_engine.pytorch as te

# Model names for test_torch_dynamo
_model_names = ["Linear", "LayerNorm", "LayerNormLinear", "LayerNormMLP"]


@pytest.mark.skipif(torch.__version__ < "2", reason="torch.compile not available")
@pytest.mark.parametrize("model_name", _model_names)
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
    model = None
    inputs = []
    if model_name == "Linear":
        model = te.Linear(16, 16)
        inputs = [make_tensor([16,16])]
    elif model_name == "LayerNorm":
        model = te.LayerNorm(16)
        inputs = [make_tensor([16,16])]
    elif model_name == "LayerNormLinear":
        model = te.LayerNormLinear(16,16)
        inputs = [make_tensor([16,16])]
    elif model_name == "LayerNormMLP":
        model = te.LayerNormMLP(16,16)
        inputs = [make_tensor([16,16])]
    assert model is not None, f"could not construct {model_name}"

    # Optimize model with TorchDynamo
    torch.compile(model)

    # Forward and backward pass
    out = model(*inputs)
    out.backward(torch.zeros_like(out))
