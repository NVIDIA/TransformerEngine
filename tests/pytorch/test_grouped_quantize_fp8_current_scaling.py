# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Tests for grouped FP8 current scaling quantization"""

import pytest
import torch
import transformer_engine.pytorch as te
from transformer_engine.pytorch.tensor.grouped_quantize import (
    grouped_quantize_unfused,
    grouped_quantize_current_scaling,
)
from transformer_engine.pytorch import Float8CurrentScalingQuantizer
import transformer_engine_torch as tex

# Check if FP8 is available
fp8_available, reason_for_no_fp8 = te.is_fp8_available(return_reason=True)


@pytest.mark.skipif(not fp8_available, reason=reason_for_no_fp8)
class TestGroupedQuantizeFP8CurrentScaling:
    """Test suite for grouped FP8 current scaling quantization"""
    
    @staticmethod
    def setup_class(cls) -> None:
        """Set up test fixtures"""
        # Configure RNG
        seed = 42
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
    
    def test_unfused_basic(self):
        """Test unfused grouped quantization with simple inputs."""
        num_tensors = 3
        shapes = [(512, 512)] * num_tensors
        device = "cuda"
        
        # Create input tensors
        inputs = [torch.randn(s, dtype=torch.float32, device=device) for s in shapes]
        
        # Create quantizers
        quantizers = [
            Float8CurrentScalingQuantizer(
                fp8_dtype=tex.DType.kFloat8E4M3,
                device=device,
            )
            for _ in range(num_tensors)
        ]
        
        # Set quantizer usage
        for quantizer in quantizers:
            quantizer.set_usage(rowwise=True, columnwise=False)
        
        # Perform unfused quantization
        outputs = grouped_quantize_unfused(inputs, quantizers)
        
        # Validate outputs
        assert len(outputs) == num_tensors
        for i, output in enumerate(outputs):
            assert output.shape == shapes[i]
            assert hasattr(output, '_data')  # Has FP8 data
            assert hasattr(output, '_fp8_scale_inv')  # Has scale inverse
    
    def test_unfused_varying_shapes(self):
        """Test unfused quantization with varying tensor shapes."""
        shapes = [(256, 512), (512, 512), (768, 512)]
        device = "cuda"
        num_tensors = len(shapes)
        
        # Create input tensors with varying shapes
        inputs = [torch.randn(s, dtype=torch.float32, device=device) for s in shapes]
        
        # Create quantizers
        quantizers = [
            Float8CurrentScalingQuantizer(
                fp8_dtype=tex.DType.kFloat8E4M3,
                device=device,
            )
            for _ in range(num_tensors)
        ]
        
        for quantizer in quantizers:
            quantizer.set_usage(rowwise=True, columnwise=False)
        
        # Perform unfused quantization
        outputs = grouped_quantize_unfused(inputs, quantizers)
        
        # Validate outputs
        assert len(outputs) == num_tensors
        for i, output in enumerate(outputs):
            assert output.shape == shapes[i]
    
    def test_unfused_numerical_accuracy(self):
        """Test that unfused quantization produces numerically accurate results."""
        num_tensors = 2
        shapes = [(256, 256)] * num_tensors
        device = "cuda"
        
        # Create input with known values
        inputs = [
            torch.full(shapes[0], 1.0, dtype=torch.float32, device=device),
            torch.full(shapes[1], 2.0, dtype=torch.float32, device=device),
        ]
        
        # Create quantizers
        quantizers = [
            Float8CurrentScalingQuantizer(
                fp8_dtype=tex.DType.kFloat8E4M3,
                device=device,
            )
            for _ in range(num_tensors)
        ]
        
        for quantizer in quantizers:
            quantizer.set_usage(rowwise=True, columnwise=False)
        
        # Perform quantization
        outputs = grouped_quantize_unfused(inputs, quantizers)
        
        # Dequantize and check accuracy
        for i, (input_tensor, output_tensor) in enumerate(zip(inputs, outputs)):
            dequantized = output_tensor.dequantize()
            # FP8 has limited precision, but should be close
            assert torch.allclose(input_tensor, dequantized, rtol=0.02, atol=0.01)
    
    @pytest.mark.xfail(reason="Grouped kernels not yet implemented")
    def test_grouped_basic(self):
        """
        Test grouped (fused) quantization with simple inputs.
        
        NOTE: This test is expected to fail until the C++ kernels are implemented.
        """
        num_tensors = 3
        shapes = [(512, 512)] * num_tensors
        device = "cuda"
        
        # Create input tensors
        inputs = [torch.randn(s, dtype=torch.float32, device=device) for s in shapes]
        
        # Create quantizers
        quantizers = [
            Float8CurrentScalingQuantizer(
                fp8_dtype=tex.DType.kFloat8E4M3,
                device=device,
            )
            for _ in range(num_tensors)
        ]
        
        for quantizer in quantizers:
            quantizer.set_usage(rowwise=True, columnwise=False)
        
        # Perform grouped quantization
        outputs = grouped_quantize_current_scaling(inputs, quantizers)
        
        # Validate outputs
        assert len(outputs) == num_tensors
        for i, output in enumerate(outputs):
            assert output.shape == shapes[i]
    
    @pytest.mark.xfail(reason="Grouped kernels not yet implemented")
    def test_grouped_vs_unfused_equivalence(self):
        """
        Verify that grouped quantization produces equivalent results to unfused.
        
        NOTE: This test is expected to fail until the C++ kernels are implemented.
        """
        num_tensors = 4
        shapes = [(512, 512)] * num_tensors
        device = "cuda"
        
        # Create input tensors
        inputs = [torch.randn(s, dtype=torch.float32, device=device) for s in shapes]
        
        # Create quantizers for unfused approach
        quantizers_unfused = [
            Float8CurrentScalingQuantizer(
                fp8_dtype=tex.DType.kFloat8E4M3,
                device=device,
            )
            for _ in range(num_tensors)
        ]
        
        # Create quantizers for grouped approach
        quantizers_grouped = [
            Float8CurrentScalingQuantizer(
                fp8_dtype=tex.DType.kFloat8E4M3,
                device=device,
            )
            for _ in range(num_tensors)
        ]
        
        for q in quantizers_unfused + quantizers_grouped:
            q.set_usage(rowwise=True, columnwise=False)
        
        # Perform both approaches
        unfused_outputs = grouped_quantize_unfused(inputs, quantizers_unfused)
        grouped_outputs = grouped_quantize_current_scaling(inputs, quantizers_grouped)
        
        # Compare outputs
        for i, (unfused, grouped) in enumerate(zip(unfused_outputs, grouped_outputs)):
            # FP8 data should match exactly
            assert torch.equal(unfused._data, grouped._data), \
                f"FP8 data mismatch for tensor {i}"
            
            # Scales should be close (may have minor differences due to floating point)
            assert torch.allclose(unfused._fp8_scale_inv, grouped._fp8_scale_inv, rtol=1e-5), \
                f"Scale mismatch for tensor {i}"
    
    @pytest.mark.xfail(reason="Grouped kernels not yet implemented")
    def test_grouped_varying_shapes(self):
        """
        Test grouped quantization with tensors of different shapes.
        
        NOTE: This test is expected to fail until the C++ kernels are implemented.
        """
        shapes = [(256, 512), (512, 512), (768, 512), (1024, 512)]
        device = "cuda"
        num_tensors = len(shapes)
        
        # Create input tensors with varying shapes
        inputs = [torch.randn(s, dtype=torch.float32, device=device) for s in shapes]
        
        # Create quantizers
        quantizers = [
            Float8CurrentScalingQuantizer(
                fp8_dtype=tex.DType.kFloat8E4M3,
                device=device,
            )
            for _ in range(num_tensors)
        ]
        
        for quantizer in quantizers:
            quantizer.set_usage(rowwise=True, columnwise=False)
        
        # Perform grouped quantization
        outputs = grouped_quantize_current_scaling(inputs, quantizers)
        
        # Validate outputs
        assert len(outputs) == num_tensors
        for i, output in enumerate(outputs):
            assert output.shape == shapes[i]
    
    def test_error_handling_mismatched_counts(self):
        """Test error handling when tensor and quantizer counts don't match."""
        device = "cuda"
        
        inputs = [torch.randn(256, 256, device=device) for _ in range(3)]
        quantizers = [
            Float8CurrentScalingQuantizer(tex.DType.kFloat8E4M3, device=device)
            for _ in range(2)  # Intentionally mismatched
        ]
        
        # Should raise ValueError
        with pytest.raises(ValueError, match="must match"):
            grouped_quantize_unfused(inputs, quantizers)
    
    def test_error_handling_non_2d_tensors(self):
        """Test error handling for non-2D tensors in grouped approach."""
        device = "cuda"
        
        # Create 3D tensor (not supported)
        inputs = [torch.randn(4, 256, 256, device=device)]
        quantizers = [
            Float8CurrentScalingQuantizer(tex.DType.kFloat8E4M3, device=device)
        ]
        
        quantizers[0].set_usage(rowwise=True, columnwise=False)
        
        # Unfused should work (quantizes any shape)
        outputs = grouped_quantize_unfused(inputs, quantizers)
        assert len(outputs) == 1
        
        # Grouped should raise error (requires 2D for now)
        with pytest.raises(ValueError, match="must be 2D"):
            grouped_quantize_current_scaling(inputs, quantizers)
    
    @pytest.mark.xfail(reason="Performance benchmarking - not a correctness test")
    def test_performance_comparison(self):
        """
        Compare performance of unfused vs grouped quantization.
        
        This is not a correctness test - it's for performance analysis.
        Expected results: Grouped should be ~3x faster for 8 experts.
        """
        num_experts = 8
        shapes = [(512, 1024)] * num_experts
        device = "cuda"
        num_iterations = 100
        
        # Create inputs
        inputs = [torch.randn(s, dtype=torch.float32, device=device) for s in shapes]
        
        # Benchmark unfused
        quantizers_unfused = [
            Float8CurrentScalingQuantizer(tex.DType.kFloat8E4M3, device=device)
            for _ in range(num_experts)
        ]
        for q in quantizers_unfused:
            q.set_usage(rowwise=True, columnwise=False)
        
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        for _ in range(num_iterations):
            _ = grouped_quantize_unfused(inputs, quantizers_unfused)
        end.record()
        torch.cuda.synchronize()
        unfused_time = start.elapsed_time(end) / num_iterations
        
        # Benchmark grouped
        quantizers_grouped = [
            Float8CurrentScalingQuantizer(tex.DType.kFloat8E4M3, device=device)
            for _ in range(num_experts)
        ]
        for q in quantizers_grouped:
            q.set_usage(rowwise=True, columnwise=False)
        
        torch.cuda.synchronize()
        start.record()
        for _ in range(num_iterations):
            _ = grouped_quantize_current_scaling(inputs, quantizers_grouped)
        end.record()
        torch.cuda.synchronize()
        grouped_time = start.elapsed_time(end) / num_iterations
        
        print(f"\nPerformance Results ({num_experts} experts, {shapes[0]}):")
        print(f"  Unfused: {unfused_time:.3f} ms")
        print(f"  Grouped: {grouped_time:.3f} ms")
        print(f"  Speedup: {unfused_time / grouped_time:.2f}x")
        
        # This test always fails - it's just for information
        assert False, "Performance test completed"
