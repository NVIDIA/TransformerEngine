"""Runtime GEMM hooks used by AutoswitchGemm."""

from __future__ import annotations

from typing import Optional

import torch

from transformer_engine.debug.pytorch.debug_state import TEDebugState
from transformer_engine.pytorch.quantized_tensor import QuantizedTensorStorage, Quantizer
from transformer_engine.pytorch.utils import cast_if_needed

def _is_fp8_debug_quantizer(quantizer: Optional[Quantizer]) -> bool:
    """Return True for DebugQuantizer objects wrapping an FP8/NVFP4 quantizer."""
    return (
        quantizer is not None
        and quantizer.__class__.__name__ == "DebugQuantizer"
        and getattr(quantizer, "parent_quantizer", None) is not None
    )


def should_resolve_inputs_after_sampling(
    lhs_quantizer: Optional[Quantizer],
    rhs_quantizer: Optional[Quantizer],
) -> bool:
    """Return True when runtime GEMM decision path should be applied."""
    return _is_fp8_debug_quantizer(lhs_quantizer) or _is_fp8_debug_quantizer(rhs_quantizer)


def _to_high_precision_gemm_input(tensor, dtype: torch.dtype):
    """Convert GEMM input to high precision tensor if needed."""
    if hasattr(tensor, "get_tensor") and hasattr(tensor, "rowwise_gemm_tensor"):
        rowwise_tensor = _to_high_precision_gemm_input(tensor.get_tensor(False), dtype)
        columnwise_src = tensor.get_tensor(True)
        if columnwise_src is tensor.get_tensor(False):
            columnwise_tensor = rowwise_tensor
        else:
            columnwise_tensor = _to_high_precision_gemm_input(columnwise_src, dtype)
        tensor.rowwise_gemm_tensor = rowwise_tensor
        tensor.columnwise_gemm_tensor = columnwise_tensor
        return tensor

    if dtype is None:
        dtype = getattr(tensor, "dtype", None)
    if isinstance(tensor, QuantizedTensorStorage):
        if dtype is None:
            return tensor.dequantize()
        try:
            return tensor.dequantize(dtype=dtype)
        except TypeError:
            return cast_if_needed(tensor.dequantize(), dtype)
    if dtype is None:
        return tensor
    return cast_if_needed(tensor, dtype)


def resolve_gemm_inputs_after_sampling(
    gemm_name: str,
    lhs,
    rhs,
    lhs_quantizer: Optional[Quantizer],
    rhs_quantizer: Optional[Quantizer],
    target_dtype: torch.dtype,
):
    """
    Make post-sampling GEMM precision decision and enforce OR logic across inputs.

    If any sampled input for this GEMM triggers high precision, both GEMM inputs are
    converted to high precision tensors before kernel launch.
    """
    layer_name = (
        getattr(lhs_quantizer, "layer_name", None) or getattr(rhs_quantizer, "layer_name", None)
    )
    if layer_name is None:
        return lhs, rhs

    try:
        import nvdlfw_inspect.api as debug_api
    except ImportError:
        return lhs, rhs

    iteration = TEDebugState.get_iteration()
    enabled_ret = debug_api.transformer_engine.fp8_gemm_enabled(
        layer_name=layer_name,
        gemm=gemm_name,
        iteration=iteration,
        final_decision=True,
    )
    quantized_enabled = enabled_ret[0] if isinstance(enabled_ret, tuple) else enabled_ret
    if quantized_enabled:
        return lhs, rhs

    return (
        _to_high_precision_gemm_input(lhs, target_dtype),
        _to_high_precision_gemm_input(rhs, target_dtype),
    )
