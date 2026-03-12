# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


def test_lazy_compile():
    """Smoke test to ensure lazy compilation is working."""
    from transformer_engine.pytorch.jit import dgelu_fused_

    dgelu_fused_(torch.randn(10, 10), torch.randn(10, 10))


def test_l2normalization_fused():
    """Smoke test for L2Normalization fusion functions."""
    from transformer_engine.pytorch.jit import (
        l2normalization_fused,
        l2normalization_fwd_fused,
        l2normalization_backward_fused,
    )

    # Basic smoke test like other JIT functions
    x = torch.randn(10, 128, device="cuda", dtype=torch.float32)
    eps = 1e-6

    # Test inference version
    output_inf = l2normalization_fused(x, eps)

    # Test training version with backward
    x_train = torch.randn(10, 128, device="cuda", dtype=torch.float32, requires_grad=True)
    output_train, rsqrt_norm = l2normalization_fwd_fused(x_train, eps)
    grad_output = torch.randn_like(output_train)
    grad_input = l2normalization_backward_fused(grad_output, x_train, rsqrt_norm, eps)


def test_l2normalization_fused_correctness():
    """Simple verification that L2Normalization fusion matches reference implementation."""
    from transformer_engine.pytorch.jit import (
        l2normalization_fwd_fused,
        l2normalization_backward_fused,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.randn(16, 64, device=device, dtype=torch.float32, requires_grad=True)
    eps = 1e-6

    # Test fused forward
    output_fused, rsqrt_norm = l2normalization_fwd_fused(x, eps)

    # Reference implementation
    x_ref = x.clone().detach().requires_grad_(True)
    x_squared = x_ref.pow(2)
    l2_norm_squared = x_squared.sum(dim=-1, keepdim=True)
    rsqrt_norm_ref = torch.rsqrt(l2_norm_squared + eps)
    output_ref = x_ref * rsqrt_norm_ref

    # Check forward pass matches
    torch.testing.assert_close(output_fused, output_ref, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(rsqrt_norm, rsqrt_norm_ref, atol=1e-6, rtol=1e-5)

    # Test fused backward
    grad_output = torch.randn_like(output_fused)
    grad_input_fused = l2normalization_backward_fused(grad_output, x, rsqrt_norm, eps)

    # Reference backward
    output_ref.backward(grad_output)
    grad_input_ref = x_ref.grad

    # Check backward pass matches
    torch.testing.assert_close(grad_input_fused, grad_input_ref, atol=1e-5, rtol=1e-4)
