"""Runtime GEMM hooks used by AutoswitchGemm."""

from __future__ import annotations

import copy
import os
from datetime import datetime
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
    if tensor is None:
        return None

    if hasattr(tensor, "get_tensor") and hasattr(tensor, "rowwise_gemm_tensor"):
        # Clone wrapper before replacing internals to avoid mutating cached/reused
        # DebugQuantizedTensor objects across multiple GEMM calls in one step.
        tensor_copy = copy.copy(tensor)

        # Convert both GEMM views explicitly once autoswitch requests high precision.
        # This avoids mixed rowwise/columnwise dtypes when a later GEMM selects
        # the opposite view.
        rowwise_src = getattr(tensor, "rowwise_gemm_tensor", None)
        columnwise_src = getattr(tensor, "columnwise_gemm_tensor", None)
        rowwise_tensor = _to_high_precision_gemm_input(rowwise_src, dtype)
        columnwise_tensor = _to_high_precision_gemm_input(columnwise_src, dtype)
        if rowwise_tensor is None:
            rowwise_tensor = columnwise_tensor
        if columnwise_tensor is None:
            columnwise_tensor = rowwise_tensor
        tensor_copy.rowwise_gemm_tensor = rowwise_tensor
        tensor_copy.columnwise_gemm_tensor = columnwise_tensor
        return tensor_copy

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


def _parent_quantizer(quantizer: Optional[Quantizer]) -> Optional[Quantizer]:
    """Return the quantizer that performs real quantization."""
    if quantizer is None:
        return None
    parent = getattr(quantizer, "parent_quantizer", None)
    return parent if parent is not None else quantizer


def _to_quantized_gemm_input(tensor, quantizer: Optional[Quantizer], dtype: torch.dtype):
    """Convert GEMM input to a quantized DebugQuantizedTensor-compatible object."""
    if tensor is None:
        return None

    if hasattr(tensor, "get_tensor") and hasattr(tensor, "rowwise_gemm_tensor"):
        tensor_copy = copy.copy(tensor)
        rowwise_src = getattr(tensor, "rowwise_gemm_tensor", None)
        columnwise_src = getattr(tensor, "columnwise_gemm_tensor", None)
        rowwise_tensor = _to_quantized_gemm_input(rowwise_src, quantizer, dtype)
        if columnwise_src is rowwise_src:
            columnwise_tensor = rowwise_tensor
        else:
            columnwise_tensor = _to_quantized_gemm_input(columnwise_src, quantizer, dtype)
        if rowwise_tensor is None:
            rowwise_tensor = columnwise_tensor
        if columnwise_tensor is None:
            columnwise_tensor = rowwise_tensor
        tensor_copy.rowwise_gemm_tensor = rowwise_tensor
        tensor_copy.columnwise_gemm_tensor = columnwise_tensor
        return tensor_copy

    if isinstance(tensor, QuantizedTensorStorage):
        return tensor

    quantizer = _parent_quantizer(quantizer)
    if quantizer is None or not isinstance(tensor, torch.Tensor):
        return tensor

    # Use an isolated quantizer copy so runtime coercion does not perturb module state.
    quantizer = quantizer.copy() if hasattr(quantizer, "copy") else copy.copy(quantizer)
    quantizer.set_usage(rowwise=True, columnwise=True)
    return quantizer(cast_if_needed(tensor, dtype))


def _selected_gemm_tensor(tensor, transpose: bool):
    """Return the tensor view that general_gemm will pass to the backend."""
    if hasattr(tensor, "get_tensor") and hasattr(tensor, "rowwise_gemm_tensor"):
        return tensor.get_tensor(transpose)
    return tensor


def _is_quantized_gemm_tensor(tensor) -> bool:
    """Return True if the selected GEMM operand is a quantized tensor."""
    return isinstance(tensor, QuantizedTensorStorage)


def _selected_transposes_for_gemm(gemm_name: str) -> tuple[bool, bool]:
    """Return DebugQuantizedTensor view selection for known TE GEMM layouts."""
    # general_gemm selects A.get_tensor(not transa) and B.get_tensor(transb).
    # Linear fprop uses TN, dgrad uses NN, and wgrad uses NT.
    if gemm_name == "fprop":
        return False, False
    if gemm_name == "dgrad":
        return True, False
    if gemm_name == "wgrad":
        return True, True
    return False, False


def _selected_gemm_quantization_state(gemm_name: str, lhs, rhs) -> tuple[bool, bool]:
    """Return whether the actual selected GEMM operands are quantized."""
    lhs_transpose, rhs_transpose = _selected_transposes_for_gemm(gemm_name)
    lhs_tensor = _selected_gemm_tensor(lhs, lhs_transpose)
    rhs_tensor = _selected_gemm_tensor(rhs, rhs_transpose)
    return _is_quantized_gemm_tensor(lhs_tensor), _is_quantized_gemm_tensor(rhs_tensor)


def _log_final_gemm_decision(
    layer_name: str,
    gemm_name: str,
    iteration: int,
    quantized_enabled: bool,
    lhs_quantized: bool,
    rhs_quantized: bool,
) -> None:
    """Write final AutoswitchGemm decision to the autoswitch rank-local log."""
    rank = os.getenv("RANK", "0")
    if rank != "0":
        return
    try:
        from nvdlfw_inspect.logging import get_logger

        root_log_dir = getattr(get_logger(), "root_log_dir", None)
    except Exception:  # pylint: disable=broad-except
        root_log_dir = None
    if not root_log_dir:
        return

    log_dir = os.path.join(root_log_dir, "nvdlfw_inspect_autoswitchgemm_logs")
    log_file = os.path.join(log_dir, f"nvdlfw_inspect_globalrank-{rank}.log")
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]
    message = (
        f"{timestamp} - INFO - {layer_name}_{gemm_name}_final_decision "
        f"\t\t\t\t iteration={iteration:06d} "
        f"\t\t\t\t quantized_enabled={int(bool(quantized_enabled))} "
        f"lhs_quantized={int(lhs_quantized)} "
        f"rhs_quantized={int(rhs_quantized)}"
    )
    with open(log_file, mode="a", encoding="utf-8") as log:
        log.write(message + "\n")


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
        lhs_out = _to_quantized_gemm_input(lhs, lhs_quantizer, target_dtype)
        rhs_out = _to_quantized_gemm_input(rhs, rhs_quantizer, target_dtype)
    else:
        lhs_out = _to_high_precision_gemm_input(lhs, target_dtype)
        rhs_out = _to_high_precision_gemm_input(rhs, target_dtype)

    lhs_quantized, rhs_quantized = _selected_gemm_quantization_state(gemm_name, lhs_out, rhs_out)
    _log_final_gemm_decision(
        layer_name,
        gemm_name,
        iteration,
        bool(quantized_enabled),
        lhs_quantized,
        rhs_quantized,
    )
    return lhs_out, rhs_out
