# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""
Grouped quantization utilities for FP8 current scaling.

This module provides functionality to quantize multiple tensors simultaneously,
which is particularly useful for Mixture of Experts (MoE) models where you need
to quantize tensors for each expert independently before GEMM operations.
"""

from typing import List, Optional
import torch
import transformer_engine_torch as tex

from .float8_tensor import Float8CurrentScalingQuantizer
from .storage.grouped_tensor import GroupedTensor
from ..quantized_tensor import QuantizedTensor


def grouped_quantize_unfused(
    tensors: List[torch.Tensor],
    quantizers: List[Float8CurrentScalingQuantizer],
) -> List[QuantizedTensor]:
    """
    Unfused approach for grouped FP8 current scaling quantization.
    
    This function quantizes multiple tensors independently using individual kernel
    launches for each tensor. This approach has significant overhead from:
    - Multiple CPU function calls
    - Multiple kernel launches  
    - CPU-GPU synchronizations
    - Breaking CUDA Graph compatibility
    
    Args:
        tensors: List of input tensors to quantize
        quantizers: List of Float8CurrentScalingQuantizer instances (one per tensor)
        
    Returns:
        List of quantized tensors
        
    Example:
        >>> # For MoE, you might have tensors split by expert
        >>> input_per_expert = [expert_input_1, expert_input_2, expert_input_3, ...]
        >>> quantizers = [quantizer_1, quantizer_2, quantizer_3, ...]
        >>> quantized_tensors = grouped_quantize_unfused(input_per_expert, quantizers)
        
    Note:
        This approach is provided for comparison and educational purposes.
        For production use, prefer the fused grouped quantization approach
        which launches a single multi-tensor kernel.
    """
    if len(tensors) != len(quantizers):
        raise ValueError(
            f"Number of tensors ({len(tensors)}) must match number of "
            f"quantizers ({len(quantizers)})"
        )
    
    quantized_tensors = []
    
    # Process each tensor independently
    # WARNING: This causes multiple kernel launches and potential CPU-GPU synchronizations
    for tensor, quantizer in zip(tensors, quantizers):
        # Each call launches separate kernels for:
        # 1. Computing amax
        # 2. Computing scale from amax
        # 3. Performing FP8 quantization
        quantized = quantizer(tensor)
        quantized_tensors.append(quantized)
    
    return quantized_tensors


def grouped_quantize_current_scaling(
    tensors: List[torch.Tensor],
    quantizers: List[Float8CurrentScalingQuantizer],
    device: Optional[torch.device] = None,
) -> List[QuantizedTensor]:
    """
    Fused grouped FP8 current scaling quantization.
    
    This function implements an optimized grouped quantization approach that:
    1. Computes amax for all tensors in a single grouped kernel
    2. Computes scales from amaxes in a single grouped kernel  
    3. Performs FP8 quantization for all tensors in a single grouped kernel
    
    For FP8 current scaling, the workflow MUST be:
    - Step 1: Compute amax for each tensor (requires scanning input)
    - Step 2: Compute scale from amax (scale = max_fp8 / (amax + epsilon))
    - Step 3: Perform FP8 quantization (output = cast_to_fp8(input * scale))
    
    These steps cannot be fused into a single kernel because we need the amax
    values before computing scales. However, we can process multiple tensors
    simultaneously in each step.
    
    Args:
        tensors: List of input tensors to quantize (all must be 2D)
        quantizers: List of Float8CurrentScalingQuantizer instances (one per tensor)
        device: CUDA device for allocation (defaults to current device)
        
    Returns:
        List of quantized tensors with their storage backed by GroupedTensor
        
    Example:
        >>> # For MoE with N experts
        >>> num_experts = 8
        >>> input_per_expert = [expert_input[i] for i in range(num_experts)]
        >>> quantizers = [Float8CurrentScalingQuantizer(...) for _ in range(num_experts)]
        >>> quantized_tensors = grouped_quantize_current_scaling(input_per_expert, quantizers)
        >>> # Now pass to grouped GEMM
        
    Note:
        This is significantly more efficient than the unfused approach because:
        - Reduces kernel launch overhead (3 launches instead of 3*N)
        - Better CUDA Graph compatibility
        - Improved memory coalescing
        - Lower CPU overhead
    """
    if len(tensors) != len(quantizers):
        raise ValueError(
            f"Number of tensors ({len(tensors)}) must match number of "
            f"quantizers ({len(quantizers)})"
        )
    
    if len(tensors) == 0:
        return []
    
    # Validate that all tensors are 2D
    for i, tensor in enumerate(tensors):
        if tensor.ndim != 2:
            raise ValueError(
                f"All tensors must be 2D for grouped quantization. "
                f"Tensor {i} has shape {tensor.shape}"
            )
    
    # Validate that all quantizers use current scaling
    for i, quantizer in enumerate(quantizers):
        if not isinstance(quantizer, Float8CurrentScalingQuantizer):
            raise TypeError(
                f"All quantizers must be Float8CurrentScalingQuantizer instances. "
                f"Quantizer {i} has type {type(quantizer)}"
            )
    
    # Set device
    if device is None:
        device = tensors[0].device
    
    # Get shapes for all tensors
    shapes = [tuple(t.shape) for t in tensors]
    
    # Create GroupedTensor for input (unquantized, for amax computation)
    # This packs all input tensors into a single contiguous buffer
    input_grouped = GroupedTensor.make_grouped_tensor(
        num_tensors=len(tensors),
        shape=shapes,
        quantizers=None,  # Input is high precision
        device=device,
        dtype=tensors[0].dtype,
    )
    
    # Copy input tensors into grouped storage
    input_splits = input_grouped.split_into_quantized_tensors()
    for input_split, tensor in zip(input_splits, tensors):
        input_split.copy_(tensor)
    
    # Create GroupedTensor for output (quantized, with current scaling metadata)
    output_grouped = GroupedTensor.make_grouped_tensor(
        num_tensors=len(tensors),
        shape=shapes,
        quantizers=quantizers,
        device=device,
    )
    
    # Step 1: Compute grouped amax
    # This launches a single kernel that computes amax for all tensors
    # The amax values are stored in output_grouped.amax
    _grouped_compute_amax(input_grouped, output_grouped)
    
    # Step 2: Compute scales from amaxes
    # This launches a single kernel that computes scale for all tensors
    # scale = max_fp8 / (amax + epsilon)
    # If force_pow_2_scales is enabled, scales are rounded to nearest power of 2
    _grouped_compute_scales(output_grouped, quantizers)
    
    # Step 3: Perform grouped FP8 quantization
    # This launches a single kernel that quantizes all tensors using computed scales
    _grouped_fp8_quantize(input_grouped, output_grouped, quantizers)
    
    # Split the grouped output tensor into individual quantized tensors
    # These tensors share the underlying storage with output_grouped
    quantized_tensors = output_grouped.split_into_quantized_tensors()
    
    return quantized_tensors


def _grouped_compute_amax(
    input_grouped: GroupedTensor,
    output_grouped: GroupedTensor,
) -> None:
    """
    Compute amax for all tensors in a grouped tensor using a single kernel launch.
    
    This function launches the nvte_group_amax_graph_safe kernel which:
    - Processes all tensors in parallel
    - Computes max(abs(tensor)) for each tensor
    - Stores result in output_grouped.amax
    
    Args:
        input_grouped: GroupedTensor containing input data
        output_grouped: GroupedTensor where amax will be stored
    """
    # Use the graph-safe grouped amax kernel
    # This is CUDA Graph compatible and efficient
    tex.group_amax_graph_safe(input_grouped, output_grouped)


def _grouped_compute_scales(
    output_grouped: GroupedTensor,
    quantizers: List[Float8CurrentScalingQuantizer],
) -> None:
    """
    Compute FP8 scales from amaxes for all tensors using a single kernel launch.
    
    For each tensor:
        scale = max_fp8 / (amax + epsilon)
        scale_inv = 1.0 / scale
        
    If force_pow_2_scales is enabled:
        scale = 2^floor(log2(scale))
        
    Args:
        output_grouped: GroupedTensor with amax values; scale/scale_inv will be computed
        quantizers: List of quantizers (used for configuration)
    """
    # Get FP8 dtype and configuration from first quantizer
    # (all quantizers should have the same configuration)
    fp8_dtype = quantizers[0].dtype
    force_pow_2_scales = quantizers[0].force_pow_2_scales
    epsilon = quantizers[0].amax_epsilon
    
    # Get max representable value for FP8 format
    if fp8_dtype == tex.DType.kFloat8E4M3:
        max_fp8 = 448.0  # Max value for E4M3
    elif fp8_dtype == tex.DType.kFloat8E5M2:
        max_fp8 = 57344.0  # Max value for E5M2
    else:
        raise ValueError(f"Unsupported FP8 dtype: {fp8_dtype}")
    
    # Prepare tensor lists for multi-tensor kernel
    # Format: [amax_0, scale_0, scale_inv_0], [amax_1, scale_1, scale_inv_1], ...
    num_tensors = output_grouped.num_tensors
    
    # Create views into the grouped tensor buffers
    amax_list = []
    scale_list = []
    scale_inv_list = []
    
    for i in range(num_tensors):
        # Each tensor has one amax, scale, and scale_inv value
        amax_list.append(output_grouped.amax[i:i+1])
        scale_list.append(output_grouped.scale[i:i+1])
        scale_inv_list.append(output_grouped.scale_inv[i:i+1])
    
    # Launch grouped scale computation kernel
    # This computes scale and scale_inv for all tensors in a single kernel
    tex.multi_tensor_compute_scale_and_scale_inv(
        amax_list,
        scale_list, 
        scale_inv_list,
        max_fp8,
        force_pow_2_scales,
        epsilon,
    )


def _grouped_fp8_quantize(
    input_grouped: GroupedTensor,
    output_grouped: GroupedTensor,
    quantizers: List[Float8CurrentScalingQuantizer],
) -> None:
    """
    Perform FP8 quantization for all tensors using computed scales in a single kernel.
    
    For each element in each tensor:
        fp8_value = saturate(cast_to_fp8(input * scale))
        
    Args:
        input_grouped: GroupedTensor containing high-precision input data
        output_grouped: GroupedTensor where quantized data will be stored (with scales)
        quantizers: List of quantizers (used for configuration)
    """
    # The quantized grouped kernel handles:
    # 1. Reading input from input_grouped.data
    # 2. Reading scales from output_grouped.scale
    # 3. Computing input * scale
    # 4. Casting to FP8 with saturation
    # 5. Writing to output_grouped.data
    # 6. Optionally transposing to output_grouped.columnwise_data
    
    # Determine if we need rowwise and/or columnwise output
    rowwise_usage = quantizers[0].rowwise_usage
    columnwise_usage = quantizers[0].columnwise_usage
    
    if rowwise_usage and not columnwise_usage:
        # Only rowwise quantization
        _grouped_fp8_quantize_rowwise(input_grouped, output_grouped)
    elif columnwise_usage and not rowwise_usage:
        # Only columnwise quantization (transposed)
        _grouped_fp8_quantize_columnwise(input_grouped, output_grouped)
    elif rowwise_usage and columnwise_usage:
        # Both rowwise and columnwise
        # Can potentially be fused, but for now do separately
        _grouped_fp8_quantize_rowwise(input_grouped, output_grouped)
        _grouped_fp8_quantize_columnwise(input_grouped, output_grouped)
    else:
        raise ValueError("At least one of rowwise or columnwise must be enabled")


def _grouped_fp8_quantize_rowwise(
    input_grouped: GroupedTensor,
    output_grouped: GroupedTensor,
) -> None:
    """
    Perform rowwise FP8 quantization for all tensors.
    
    Args:
        input_grouped: GroupedTensor with input data
        output_grouped: GroupedTensor with scales and output buffer
    """
    # Launch grouped quantization kernel for rowwise layout
    # This kernel:
    # - Reads from input_grouped.data (high precision)
    # - Reads scales from output_grouped.scale (or scale_inv)
    # - Writes quantized FP8 to output_grouped.data
    tex.group_fp8_quantize_rowwise(
        input_grouped,
        output_grouped,
    )


def _grouped_fp8_quantize_columnwise(
    input_grouped: GroupedTensor,
    output_grouped: GroupedTensor,
) -> None:
    """
    Perform columnwise (transposed) FP8 quantization for all tensors.
    
    Args:
        input_grouped: GroupedTensor with input data
        output_grouped: GroupedTensor with scales and output buffer
    """
    # Launch grouped quantization kernel for columnwise (transposed) layout
    # This kernel:
    # - Reads from input_grouped.data (high precision)
    # - Reads scales from output_grouped.scale (or scale_inv)
    # - Transposes and writes quantized FP8 to output_grouped.columnwise_data
    tex.group_fp8_quantize_columnwise(
        input_grouped,
        output_grouped,
    )
