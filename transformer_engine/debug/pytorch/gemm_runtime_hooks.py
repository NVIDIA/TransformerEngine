"""Runtime GEMM hooks used by AutoswitchGemm."""

from __future__ import annotations

from typing import Any, Optional

import torch

from transformer_engine.debug.pytorch.debug_state import TEDebugState
from transformer_engine.pytorch.quantized_tensor import QuantizedTensorStorage, Quantizer
from transformer_engine.pytorch.utils import cast_if_needed

_AUTOSWITCH_FEATURE_NAME = "AutoswitchGemm"
_AUTOSWITCH_ENABLED_CACHE = {}


def _is_fp8_debug_quantizer(quantizer: Optional[Quantizer]) -> bool:
    """Return True for DebugQuantizer objects wrapping an FP8/NVFP4 quantizer."""
    return (
        quantizer is not None
        and quantizer.__class__.__name__ == "DebugQuantizer"
        and getattr(quantizer, "parent_quantizer", None) is not None
    )


def _feature_block_enabled(feature_config: Any) -> bool:
    """Return whether an Autoswitch feature block is enabled."""
    if isinstance(feature_config, dict):
        return bool(feature_config.get("enabled", True))
    if isinstance(feature_config, bool):
        return feature_config
    return feature_config is not None


def _contains_enabled_autoswitch(config: Any, visited: Optional[set] = None) -> bool:
    """Recursively check whether config contains enabled AutoswitchGemm feature."""
    if visited is None:
        visited = set()
    obj_id = id(config)
    if obj_id in visited:
        return False
    visited.add(obj_id)

    if isinstance(config, dict):
        for key, value in config.items():
            if key == _AUTOSWITCH_FEATURE_NAME and _feature_block_enabled(value):
                return True
        for value in config.values():
            if _contains_enabled_autoswitch(value, visited):
                return True
        return False

    if isinstance(config, (list, tuple, set)):
        for item in config:
            if _contains_enabled_autoswitch(item, visited):
                return True
        return False

    return False


def _autoswitch_feature_enabled() -> bool:
    """Best-effort detection for whether AutoswitchGemm is enabled in debug config."""
    try:
        import nvdlfw_inspect.api as debug_api
    except ImportError:
        return False

    manager = getattr(debug_api, "DEBUG_MANAGER", None)
    if manager is None:
        return False

    manager_id = id(manager)
    cached = _AUTOSWITCH_ENABLED_CACHE.get(manager_id)
    if cached is not None:
        return cached

    candidate_configs = []
    for attr in (
        "config",
        "_config",
        "debug_config",
        "_debug_config",
        "user_config",
        "_user_config",
        "raw_config",
        "_raw_config",
    ):
        value = getattr(manager, attr, None)
        if value is not None:
            candidate_configs.append(value)

    for attr_name, value in getattr(manager, "__dict__", {}).items():
        if "config" in attr_name.lower() and value is not None:
            candidate_configs.append(value)

    if not candidate_configs:
        # Keep previous behavior if manager internals cannot be introspected.
        _AUTOSWITCH_ENABLED_CACHE[manager_id] = True
        return True

    enabled = any(_contains_enabled_autoswitch(config) for config in candidate_configs)
    _AUTOSWITCH_ENABLED_CACHE[manager_id] = enabled
    return enabled


def should_resolve_inputs_after_sampling(
    lhs_quantizer: Optional[Quantizer],
    rhs_quantizer: Optional[Quantizer],
) -> bool:
    """Return True when runtime GEMM decision path should be applied."""
    if not (_is_fp8_debug_quantizer(lhs_quantizer) or _is_fp8_debug_quantizer(rhs_quantizer)):
        return False
    return _autoswitch_feature_enabled()


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
