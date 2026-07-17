"""Runtime GEMM hooks used by AutoswitchGemm."""

from __future__ import annotations

import copy
import os
from typing import Optional

import torch

from transformer_engine.debug.pytorch.debug_state import TEDebugState
from transformer_engine.pytorch.quantized_tensor import QuantizedTensorStorage, Quantizer
from transformer_engine.pytorch.utils import cast_if_needed

_AUTOSWITCH_LOGGING_ENV = "NVTE_AUTOSWITCH_GEMM_LOGGING"


def _env_flag_enabled(name: str, default: bool = False) -> bool:
    """Interpret common boolean environment flag values."""
    default_value = "1" if default else "0"
    return os.getenv(name, default_value).strip().lower() in {"1", "true", "yes", "on"}


def _autoswitch_logging_enabled() -> bool:
    """Return True when verbose AutoswitchGemm runtime logging is enabled."""
    try:
        from transformer_engine.debug.features.autoswitch_gemm import (
            autoswitch_gemm_logging_enabled,
        )

        return bool(autoswitch_gemm_logging_enabled())
    except Exception:  # pylint: disable=broad-except
        return _env_flag_enabled(_AUTOSWITCH_LOGGING_ENV, False)


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
        if rowwise_tensor is None and columnwise_tensor is None:
            return tensor
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
        try:
            if dtype is None:
                return tensor.dequantize()
            return tensor.dequantize(dtype=dtype)
        except TypeError:
            return cast_if_needed(tensor.dequantize(), dtype)
        except NotImplementedError as err:
            if "column-wise NVFP4" in str(err):
                return None
            raise
    if dtype is None:
        return tensor
    return cast_if_needed(tensor, dtype)


def _parent_quantizer(quantizer: Optional[Quantizer]) -> Optional[Quantizer]:
    """Return the quantizer that performs real quantization."""
    if quantizer is None:
        return None
    parent = getattr(quantizer, "parent_quantizer", None)
    return parent if parent is not None else quantizer


def _can_quantize(tensor, quantizer: Optional[Quantizer]) -> bool:
    """Return whether a tensor can be quantized by this quantizer."""
    if quantizer is None or not isinstance(tensor, torch.Tensor):
        return False
    is_quantizable = getattr(quantizer, "is_quantizable", None)
    if callable(is_quantizable):
        try:
            return bool(is_quantizable(tensor))
        except Exception:  # pylint: disable=broad-except
            return False
    return True


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
    tensor = cast_if_needed(tensor, dtype)
    if not _can_quantize(tensor, quantizer):
        return tensor

    # Use an isolated quantizer copy so runtime coercion does not perturb module state.
    quantizer = quantizer.copy() if hasattr(quantizer, "copy") else copy.copy(quantizer)
    quantizer.set_usage(rowwise=True, columnwise=True)
    return quantizer(tensor)


def _selected_gemm_tensor(tensor, transpose: bool):
    """Return the tensor view that general_gemm will pass to the backend."""
    if hasattr(tensor, "get_tensor") and hasattr(tensor, "rowwise_gemm_tensor"):
        return tensor.get_tensor(transpose)
    return tensor


def _is_quantized_gemm_tensor(tensor) -> bool:
    """Return True if the selected GEMM operand is a quantized tensor."""
    return isinstance(tensor, QuantizedTensorStorage)


def _precision_name_from_class_name(class_name: str) -> str:
    """Map quantizer/tensor class names to user-facing precision labels."""
    lowered = class_name.lower()
    if "mxfp8" in lowered:
        return "mxfp8"
    if "nvfp4" in lowered:
        return "nvfp4"
    if "float8blockwise" in lowered or "blockwise" in lowered:
        return "fp8_blockwise"
    if "float8" in lowered or "fp8" in lowered:
        return "fp8"
    return "quantized"


def _precision_name_from_quantizer(quantizer: Optional[Quantizer]) -> str:
    """Return requested quantized precision based on the underlying quantizer."""
    quantizer = _parent_quantizer(quantizer)
    if quantizer is None:
        return "quantized"
    return _precision_name_from_class_name(quantizer.__class__.__name__)


def _precision_name_from_tensor(tensor) -> str:
    """Return actual precision based on the selected GEMM operand."""
    if isinstance(tensor, QuantizedTensorStorage):
        return _precision_name_from_class_name(tensor.__class__.__name__)
    dtype = getattr(tensor, "dtype", None)
    if dtype is None:
        return "unknown"
    if dtype == torch.bfloat16:
        return "bf16"
    if dtype == torch.float16:
        return "fp16"
    if dtype == torch.float32:
        return "fp32"
    return str(dtype).replace("torch.", "")


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


def _selected_gemm_precision(gemm_name: str, lhs, rhs) -> str:
    """Return the actual precision label for selected GEMM operands."""
    lhs_transpose, rhs_transpose = _selected_transposes_for_gemm(gemm_name)
    lhs_precision = _precision_name_from_tensor(_selected_gemm_tensor(lhs, lhs_transpose))
    rhs_precision = _precision_name_from_tensor(_selected_gemm_tensor(rhs, rhs_transpose))
    if lhs_precision == rhs_precision:
        return lhs_precision
    return f"{lhs_precision}+{rhs_precision}"


def _log_final_gemm_decision(
    layer_name: str,
    gemm_name: str,
    iteration: int,
    quantized_enabled: bool,
    lhs_quantized: bool,
    rhs_quantized: bool,
    requested_precision: str,
    actual_precision: str,
) -> None:
    """Write final AutoswitchGemm decision to the autoswitch rank-local log."""
    if not _autoswitch_logging_enabled():
        return
    try:
        from transformer_engine.debug.features.autoswitch_gemm import (
            _get_autoswitch_metric_logger,
            autoswitch_gemm_should_log_final_decision,
            autoswitch_gemm_log_iteration,
        )
    except Exception:  # pylint: disable=broad-except
        return

    try:
        should_log_final_decision = autoswitch_gemm_should_log_final_decision(iteration)
    except Exception:  # pylint: disable=broad-except
        return
    if not should_log_final_decision:
        return
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

    log_iteration = autoswitch_gemm_log_iteration(iteration)
    metric_logger = _get_autoswitch_metric_logger()
    if not metric_logger.ensure_initialized(root_log_dir):
        return
    if metric_logger.logger is None:
        return
    metric_logger.logger.info(
        f"{layer_name}_{gemm_name}_final_decision "
        f"\t\t\t\t iteration={log_iteration:06d} "
        f"\t\t\t\t quantized_enabled={int(bool(quantized_enabled))} "
        f"requested_precision={requested_precision} "
        f"precision={actual_precision} "
        f"lhs_quantized={int(lhs_quantized)} "
        f"rhs_quantized={int(rhs_quantized)}"
    )


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
    try:
        enabled_ret = debug_api.transformer_engine.fp8_gemm_enabled(
            layer_name=layer_name,
            gemm=gemm_name,
            iteration=iteration,
            final_decision=True,
        )
    except TypeError as err:
        if "final_decision" not in str(err):
            raise
        enabled_ret = debug_api.transformer_engine.fp8_gemm_enabled(
            layer_name=layer_name,
            gemm=gemm_name,
            iteration=iteration,
        )
    quantized_enabled = enabled_ret[0] if isinstance(enabled_ret, tuple) else enabled_ret
    requested_precision = (
        _precision_name_from_quantizer(lhs_quantizer) if quantized_enabled else "bf16"
    )
    if quantized_enabled:
        lhs_out = _to_quantized_gemm_input(lhs, lhs_quantizer, target_dtype)
        rhs_out = _to_quantized_gemm_input(rhs, rhs_quantizer, target_dtype)
    else:
        lhs_out = _to_high_precision_gemm_input(lhs, target_dtype)
        rhs_out = _to_high_precision_gemm_input(rhs, target_dtype)

    lhs_quantized, rhs_quantized = _selected_gemm_quantization_state(gemm_name, lhs_out, rhs_out)
    if quantized_enabled and not (lhs_quantized and rhs_quantized):
        lhs_out = _to_high_precision_gemm_input(lhs_out, target_dtype)
        rhs_out = _to_high_precision_gemm_input(rhs_out, target_dtype)
        lhs_quantized, rhs_quantized = _selected_gemm_quantization_state(
            gemm_name, lhs_out, rhs_out
        )
    actual_precision = _selected_gemm_precision(gemm_name, lhs_out, rhs_out)
    _log_final_gemm_decision(
        layer_name,
        gemm_name,
        iteration,
        bool(quantized_enabled),
        lhs_quantized,
        rhs_quantized,
        requested_precision,
        actual_precision,
    )
    return lhs_out, rhs_out
