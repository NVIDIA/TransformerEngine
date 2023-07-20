# Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""FP8 utilities for TransformerEngine"""

from contextlib import contextmanager
from typing import Tuple, Optional, Dict, Any

import paddle
import transformer_engine_paddle as tex
from transformer_engine.common.recipe import DelayedScaling, Format

# FP8 support
_is_fp8_available = None
_reason_for_no_fp8 = ""
# FP8 status
_FP8_ENABLED = False
_FP8_CALIBRATION = False
_FP8_RECIPE = None


def _check_fp8_support() -> Tuple[bool, str]:
    """Return if fp8 support is available"""

    # Check GPU arch
    arch = paddle.device.cuda.get_device_capability()
    if arch >= (9, 0):    # hopper and above
        return True, ""
    if arch < (8, 9):    # pre-ada
        return False, "Device compute capability 8.9 or higher required for FP8 execution."

    # Special handling for Ada
    if tex.get_cublasLt_version() < 120103:
        return False, "CublasLt version 12.1.3.x or higher required for FP8 execution on Ada."
    if not paddle.version.cuda():
        return False, "Cuda version 12.1 or higher required for FP8 execution on Ada."
    if tuple(int(v) for v in paddle.version.cuda().split(".")) < (12, 1):
        return False, "Cuda version 12.1 or higher required for FP8 execution on Ada."
    return True, ""


def is_fp8_available() -> Tuple[bool, str]:
    """Return if fp8 support is available"""
    global _is_fp8_available, _reason_for_no_fp8
    if _is_fp8_available is None:
        _is_fp8_available, _reason_for_no_fp8 = _check_fp8_support()
    return _is_fp8_available, _reason_for_no_fp8


# Functions used to access fp8 status
def is_fp8_enabled() -> bool:
    """Is FP8 enabled"""
    return _FP8_ENABLED


def is_fp8_calibration() -> bool:
    """Is FP8 calibration"""
    return _FP8_CALIBRATION


def get_fp8_recipe() -> DelayedScaling:
    """Return the fp8 recipe"""
    return _FP8_RECIPE


def get_default_fp8_recipe() -> DelayedScaling:
    """FP8 recipe if not provided by user
    Margin = 0, interval = 1, E4M3
    """
    return DelayedScaling()


@contextmanager
def fp8_autocast(
    enabled: bool = False,
    calibrating: bool = False,
    fp8_recipe: Optional[DelayedScaling] = None,
) -> None:
    """
    Context manager for FP8 usage.
    """

    global _FP8_ENABLED, _FP8_CALIBRATION, _FP8_RECIPE
    fp8_state = (_FP8_ENABLED, _FP8_CALIBRATION, _FP8_RECIPE)
    try:
        _FP8_ENABLED = enabled
        _FP8_CALIBRATION = calibrating
        _FP8_RECIPE = get_default_fp8_recipe() if fp8_recipe is None else fp8_recipe

        if enabled:
            fp8_available, reason_for_no_fp8 = is_fp8_available()
            assert fp8_available, reason_for_no_fp8
        yield
    finally:
        (_FP8_ENABLED, _FP8_CALIBRATION, _FP8_RECIPE) = fp8_state


def get_fp8_te_dtype(fp8_recipe: DelayedScaling, fprop_tensor: bool = True) -> tex.DType:
    """Get fp8 data type according to recipe and tensor"""
    if fp8_recipe.fp8_format == Format.E4M3 or (fp8_recipe.fp8_format == Format.HYBRID
                                                and fprop_tensor):
        return tex.DType.kFloat8E4M3
    return tex.DType.kFloat8E5M2


def update_amax_history(amax_history: paddle.Tensor) -> paddle.Tensor:
    """Update amax history and set next amax to zero."""
    if amax_history.shape[0] > 1:
        amax_history = paddle.roll(amax_history, -1, 0)
    amax_history[0] = 0.0
    return amax_history


def _default_get_amax(
    amax_history: paddle.Tensor,
    amax_compute_algo: str,
) -> Tuple[paddle.Tensor, paddle.Tensor]:
    """Default function to obtain amax from history."""
    if amax_compute_algo == "max":
        amax = paddle.max(amax_history, axis=0)
    else:    # amax_compute_algo == "most_recent"
        amax = amax_history[0]

    amax_history = update_amax_history(amax_history)
    return amax_history, amax


def _default_sf_compute(
    amax: paddle.Tensor,
    scale: paddle.Tensor,
    fp8_max: float,
    margin: int,
) -> paddle.Tensor:
    """Default function to convert amax to scaling factor."""
    return tex.update_scale(amax, scale, fp8_max, float(margin))


def _compute_scaling_factor_inverse(
    scale: paddle.Tensor,
    scale_inv: paddle.Tensor,
    non_weight_mask: paddle.Tensor,
    update_weight_scale_inv: bool,
) -> paddle.Tensor:
    """Compute inverse of scaling factor."""
    if update_weight_scale_inv:
        return 1.0 / scale
    return paddle.where(non_weight_mask, 1.0 / scale, scale_inv)


def fused_amax_and_scale_update(
    amax_history: paddle.Tensor,
    scale: paddle.Tensor,
    scale_inv: paddle.Tensor,
    fp8_max: float,
    margin: int,
    amax_compute_algo: str,
    non_weight_mask: paddle.Tensor,
    update_weight_scale_inv: bool,
) -> Tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    """Amax to scale conversion."""

    # Get amax from history.
    amax_history, amax = _default_get_amax(
        amax_history,
        amax_compute_algo,
    )

    # Calculate new scaling factor.
    scale = _default_sf_compute(
        amax,
        scale,
        fp8_max,
        margin,
    )

    # Calculate new inverse of scaling factor.
    scale_inv = _compute_scaling_factor_inverse(
        scale,
        scale_inv,
        non_weight_mask,
        update_weight_scale_inv,
    )

    return amax_history, scale, scale_inv


def amax_and_scale_update(
    fp8_meta: Dict[str, Any],
    fwd_update: bool,
    update_weight_scale_inv: bool = True,
) -> None:
    """Updates fp8 amaxes/scales for fwd | bwd."""
    amax_compute = fp8_meta["recipe"].amax_compute_algo
    sf_compute = fp8_meta["recipe"].scaling_factor_compute_algo
    fp8_meta_tensor_key = "scaling_fwd" if fwd_update else "scaling_bwd"
    fp8_max_key = "fp8_max_fwd" if fwd_update else "fp8_max_bwd"

    if not callable(amax_compute) and sf_compute is None:
        (
            fp8_meta[fp8_meta_tensor_key].amax_history,
            fp8_meta[fp8_meta_tensor_key].scale,
            fp8_meta[fp8_meta_tensor_key].scale_inv,
        ) = fused_amax_and_scale_update(
            fp8_meta[fp8_meta_tensor_key].amax_history,
            fp8_meta[fp8_meta_tensor_key].scale,
            fp8_meta[fp8_meta_tensor_key].scale_inv,
            fp8_meta[fp8_max_key],
            fp8_meta["recipe"].margin,
            fp8_meta["recipe"].amax_compute_algo,
            fp8_meta[fp8_meta_tensor_key + "_non_weight_mask"],
            update_weight_scale_inv,
        )
    else:
        raise ValueError("We only support the fp8 recipe with 'max' or 'most_recent' "
                         "amax_compute_algo and default scaling_factor_compute_algo at this "
                         "moment.")
