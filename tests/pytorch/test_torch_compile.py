# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Test torch.compile(fullgraph=True) with Transformer Engine operations."""

import pytest
import torch
import transformer_engine.pytorch as te
from transformer_engine.pytorch.quantization import FP8GlobalStateManager
from transformer_engine.pytorch.ops.compile_compat import TorchCompileCompatibleFuser
from utils import reset_rng_states


@pytest.fixture(autouse=True)
def reset_global_fp8_state():
    yield
    FP8GlobalStateManager.reset()


class TestBasicLinear:
    """Tests for BasicLinear operation."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_forward_backward(self, dtype):
        """Test BasicLinear forward+backward with torch.compile(fullgraph=True)."""
        reset_rng_states()

        # Create operation
        linear = te.ops.BasicLinear(256, 512, device="cuda", dtype=dtype)

        # Create TorchCompileCompatibleFuser OUTSIDE compiled region
        fuser = TorchCompileCompatibleFuser([linear])

        x = torch.randn(32, 64, 256, device="cuda", dtype=dtype, requires_grad=True)

        # Eager reference
        y_eager = fuser(x)
        loss = y_eager.sum()
        loss.backward()
        grad_x_eager = x.grad.clone()
        grad_w_eager = linear.weight.grad.clone()

        # Reset grads
        x.grad = None
        linear.weight.grad = None

        # Compiled with fullgraph=True
        @torch.compile(fullgraph=True)
        def compiled_forward(fuser, x):
            return fuser(x)

        y_compiled = compiled_forward(fuser, x)
        y_compiled.sum().backward()

        # Check numerics
        tols = dict(rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(y_eager, y_compiled, **tols)
        torch.testing.assert_close(grad_x_eager, x.grad, **tols)
        torch.testing.assert_close(grad_w_eager, linear.weight.grad, **tols)


class TestBias:
    """Tests for Bias operation."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_forward_backward(self, dtype):
        """Test Bias forward+backward with torch.compile(fullgraph=True)."""
        reset_rng_states()

        # Create operation
        bias = te.ops.Bias(512, device="cuda", dtype=dtype)

        # Create TorchCompileCompatibleFuser
        fuser = TorchCompileCompatibleFuser([bias])

        x = torch.randn(32, 64, 512, device="cuda", dtype=dtype, requires_grad=True)

        # Eager reference
        y_eager = fuser(x)
        loss = y_eager.sum()
        loss.backward()
        grad_x_eager = x.grad.clone()
        grad_b_eager = bias.bias.grad.clone()

        # Reset grads
        x.grad = None
        bias.bias.grad = None

        # Compiled with fullgraph=True
        @torch.compile(fullgraph=True)
        def compiled_forward(fuser, x):
            return fuser(x)

        y_compiled = compiled_forward(fuser, x)
        y_compiled.sum().backward()

        # Check numerics
        tols = dict(rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(y_eager, y_compiled, **tols)
        torch.testing.assert_close(grad_x_eager, x.grad, **tols)
        torch.testing.assert_close(grad_b_eager, bias.bias.grad, **tols)


class TestLinearWithBias:
    """Tests for Linear (BasicLinear + Bias) operation."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_forward_backward(self, dtype):
        """Test Linear with bias forward+backward with torch.compile(fullgraph=True)."""
        reset_rng_states()

        # Create fused Linear operation (contains BasicLinear + Bias)
        linear = te.ops.Linear(256, 512, bias=True, device="cuda", dtype=dtype)

        # Create TorchCompileCompatibleFuser
        fuser = TorchCompileCompatibleFuser([linear])

        x = torch.randn(32, 64, 256, device="cuda", dtype=dtype, requires_grad=True)

        # Eager reference
        y_eager = fuser(x)
        loss = y_eager.sum()
        loss.backward()
        grad_x_eager = x.grad.clone()
        grad_w_eager = linear.weight.grad.clone()
        grad_b_eager = linear.bias.grad.clone()

        # Reset grads
        x.grad = None
        linear.weight.grad = None
        linear.bias.grad = None

        # Compiled with fullgraph=True
        @torch.compile(fullgraph=True)
        def compiled_forward(fuser, x):
            return fuser(x)

        y_compiled = compiled_forward(fuser, x)
        y_compiled.sum().backward()

        # Check numerics
        tols = dict(rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(y_eager, y_compiled, **tols)
        torch.testing.assert_close(grad_x_eager, x.grad, **tols)
        torch.testing.assert_close(grad_w_eager, linear.weight.grad, **tols)
        torch.testing.assert_close(grad_b_eager, linear.bias.grad, **tols)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_without_bias(self, dtype):
        """Test Linear without bias with torch.compile(fullgraph=True)."""
        reset_rng_states()

        # Create Linear without bias
        linear = te.ops.Linear(256, 512, bias=False, device="cuda", dtype=dtype)

        fuser = TorchCompileCompatibleFuser([linear])

        x = torch.randn(32, 64, 256, device="cuda", dtype=dtype, requires_grad=True)

        # Eager reference
        y_eager = fuser(x)
        y_eager.sum().backward()
        grad_x_eager = x.grad.clone()
        grad_w_eager = linear.weight.grad.clone()

        # Reset grads
        x.grad = None
        linear.weight.grad = None

        # Compiled
        @torch.compile(fullgraph=True)
        def compiled_forward(fuser, x):
            return fuser(x)

        y_compiled = compiled_forward(fuser, x)
        y_compiled.sum().backward()

        # Check numerics
        tols = dict(rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(y_eager, y_compiled, **tols)
        torch.testing.assert_close(grad_x_eager, x.grad, **tols)
        torch.testing.assert_close(grad_w_eager, linear.weight.grad, **tols)


class TestFP8:
    """Tests for FP8 quantization support."""

    @pytest.mark.parametrize("linear_cls", ["BasicLinear", "Linear"])
    def test_fp8_forward_backward(self, linear_cls):
        """Test FP8 forward+backward with torch.compile(fullgraph=True)."""
        reset_rng_states()

        # Create operation
        if linear_cls == "BasicLinear":
            linear = te.ops.BasicLinear(256, 512, device="cuda", dtype=torch.bfloat16)
        else:
            linear = te.ops.Linear(256, 512, bias=True, device="cuda", dtype=torch.bfloat16)

        fuser = TorchCompileCompatibleFuser([linear])

        x = torch.randn(32, 64, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)

        # Enable FP8
        from transformer_engine.common.recipe import DelayedScaling, Format

        fp8_recipe = DelayedScaling(fp8_format=Format.HYBRID)

        # Warmup: run several iterations to stabilize DelayedScaling amax history
        # This ensures eager and compiled use the same scaling factors
        for _ in range(5):
            x.grad = None
            linear.weight.grad = None
            with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
                y_warmup = fuser(x)
            y_warmup.sum().backward()

        # Reset grads after warmup
        x.grad = None
        linear.weight.grad = None

        # Eager reference with FP8 (amax history is now stable)
        with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
            y_eager = fuser(x)
        y_eager.sum().backward()
        grad_x_eager = x.grad.clone()
        grad_w_eager = linear.weight.grad.clone()

        # Reset grads
        x.grad = None
        linear.weight.grad = None

        # Compiled with fullgraph=True and FP8
        @torch.compile(fullgraph=True)
        def compiled_forward(fuser, x):
            return fuser(x)

        with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
            y_compiled = compiled_forward(fuser, x)
        y_compiled.sum().backward()

        # Check numerics (FP8 has lower precision, but with stable scaling should match closely)
        tols = dict(rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(y_eager, y_compiled, **tols)
        torch.testing.assert_close(grad_x_eager, x.grad, **tols)
        torch.testing.assert_close(grad_w_eager, linear.weight.grad, **tols)


class TestFusion:
    """Tests for fused operations (LinearBiasActivation fusion)."""

    def test_linear_bias_fusion_bf16(self):
        """Test BasicLinear + Bias fusion with torch.compile(fullgraph=True).

        This pattern should trigger ForwardLinearBiasActivation fusion when using bf16.
        """
        reset_rng_states()
        dtype = torch.bfloat16

        # Create operations - this pattern triggers fusion in bf16
        linear = te.ops.BasicLinear(256, 512, device="cuda", dtype=dtype)
        bias = te.ops.Bias(512, device="cuda", dtype=dtype)

        fuser = TorchCompileCompatibleFuser([linear, bias])

        # Check that fusion happened
        # After _maybe_fuse_ops, forward_ops should have fewer entries than basic_ops
        x = torch.randn(32, 64, 256, device="cuda", dtype=dtype, requires_grad=True)

        # Trigger fusion by running forward
        _ = fuser(x)

        # Verify fusion: 2 basic ops should become 1 fused forward op
        assert (
            len(fuser.ops_container._forward_ops) == 1
        ), f"Expected 1 fused forward op, got {len(fuser.ops_container._forward_ops)}"

        # Eager reference
        y_eager = fuser(x)
        y_eager.sum().backward()
        grad_x_eager = x.grad.clone()
        grad_w_eager = linear.weight.grad.clone()
        grad_b_eager = bias.bias.grad.clone()

        # Reset grads
        x.grad = None
        linear.weight.grad = None
        bias.bias.grad = None

        # Compiled with fullgraph=True
        @torch.compile(fullgraph=True)
        def compiled_forward(fuser, x):
            return fuser(x)

        y_compiled = compiled_forward(fuser, x)
        y_compiled.sum().backward()

        # Check numerics
        tols = dict(rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(y_eager, y_compiled, **tols)
        torch.testing.assert_close(grad_x_eager, x.grad, **tols)
        torch.testing.assert_close(grad_w_eager, linear.weight.grad, **tols)
        torch.testing.assert_close(grad_b_eager, bias.bias.grad, **tols)

    def test_no_fusion_fp32(self):
        """Test that fusion doesn't happen with fp32 (cuBLAS limitation)."""
        reset_rng_states()
        dtype = torch.float32

        # Create operations - fp32 should NOT trigger fusion
        linear = te.ops.BasicLinear(256, 512, device="cuda", dtype=dtype)
        bias = te.ops.Bias(512, device="cuda", dtype=dtype)

        fuser = TorchCompileCompatibleFuser([linear, bias])

        x = torch.randn(32, 64, 256, device="cuda", dtype=dtype, requires_grad=True)

        # Trigger fusion attempt by running forward
        _ = fuser(x)

        # Verify no fusion: 2 basic ops should remain as 2 forward ops
        assert len(fuser.ops_container._forward_ops) == 2, (
            "Expected 2 forward ops (no fusion for fp32), got"
            f" {len(fuser.ops_container._forward_ops)}"
        )

        # Still test that it works correctly
        y_eager = fuser(x)
        y_eager.sum().backward()
        grad_x_eager = x.grad.clone()

        x.grad = None

        @torch.compile(fullgraph=True)
        def compiled_forward(fuser, x):
            return fuser(x)

        y_compiled = compiled_forward(fuser, x)
        y_compiled.sum().backward()

        tols = dict(rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(y_eager, y_compiled, **tols)
        torch.testing.assert_close(grad_x_eager, x.grad, **tols)


class TestMultipleOps:
    """Tests for multiple operations."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_two_linears(self, dtype):
        """Test two BasicLinear ops with torch.compile(fullgraph=True)."""
        reset_rng_states()

        # Create operations
        linear1 = te.ops.BasicLinear(256, 512, device="cuda", dtype=dtype)
        linear2 = te.ops.BasicLinear(512, 128, device="cuda", dtype=dtype)

        fuser = TorchCompileCompatibleFuser([linear1, linear2])

        x = torch.randn(32, 64, 256, device="cuda", dtype=dtype, requires_grad=True)

        # Eager reference
        y_eager = fuser(x)
        y_eager.sum().backward()
        grad_x_eager = x.grad.clone()
        grad_w1_eager = linear1.weight.grad.clone()
        grad_w2_eager = linear2.weight.grad.clone()

        # Reset grads
        x.grad = None
        linear1.weight.grad = None
        linear2.weight.grad = None

        # Compiled
        @torch.compile(fullgraph=True)
        def compiled_forward(fuser, x):
            return fuser(x)

        y_compiled = compiled_forward(fuser, x)
        y_compiled.sum().backward()

        # Check numerics
        tols = dict(rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(y_eager, y_compiled, **tols)
        torch.testing.assert_close(grad_x_eager, x.grad, **tols)
        torch.testing.assert_close(grad_w1_eager, linear1.weight.grad, **tols)
        torch.testing.assert_close(grad_w2_eager, linear2.weight.grad, **tols)

    def test_linear_bias_linear_with_fusion(self):
        """Test BasicLinear + Bias + BasicLinear with fusion (bf16).

        The first Linear+Bias should fuse into ForwardLinearBiasActivation.
        """
        reset_rng_states()
        dtype = torch.bfloat16

        # Create operations
        linear1 = te.ops.BasicLinear(256, 512, device="cuda", dtype=dtype)
        bias = te.ops.Bias(512, device="cuda", dtype=dtype)
        linear2 = te.ops.BasicLinear(512, 128, device="cuda", dtype=dtype)

        fuser = TorchCompileCompatibleFuser([linear1, bias, linear2])

        x = torch.randn(32, 64, 256, device="cuda", dtype=dtype, requires_grad=True)

        # Trigger fusion
        _ = fuser(x)

        # Verify fusion: 3 basic ops should become 2 forward ops
        # (linear1+bias fused, linear2 separate)
        assert (
            len(fuser.ops_container._forward_ops) == 2
        ), f"Expected 2 forward ops after fusion, got {len(fuser.ops_container._forward_ops)}"

        # Eager reference
        y_eager = fuser(x)
        y_eager.sum().backward()
        grad_x_eager = x.grad.clone()
        grad_w1_eager = linear1.weight.grad.clone()
        grad_b_eager = bias.bias.grad.clone()
        grad_w2_eager = linear2.weight.grad.clone()

        # Reset grads
        x.grad = None
        linear1.weight.grad = None
        bias.bias.grad = None
        linear2.weight.grad = None

        # Compiled
        @torch.compile(fullgraph=True)
        def compiled_forward(fuser, x):
            return fuser(x)

        y_compiled = compiled_forward(fuser, x)
        y_compiled.sum().backward()

        # Check numerics
        tols = dict(rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(y_eager, y_compiled, **tols)
        torch.testing.assert_close(grad_x_eager, x.grad, **tols)
        torch.testing.assert_close(grad_w1_eager, linear1.weight.grad, **tols)
        torch.testing.assert_close(grad_b_eager, bias.bias.grad, **tols)
        torch.testing.assert_close(grad_w2_eager, linear2.weight.grad, **tols)
