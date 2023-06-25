# Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""FP8 utilities for TransformerEngine"""

from typing import Tuple, Optional
from contextlib import contextmanager

import paddle
import transformer_engine_paddle as tex
from transformer_engine.common.recipe import DelayedScaling

_FP8_ENABLED = False
_FP8_CALIBRATION = False
_FP8_RECIPE = None
_is_fp8_available = None
_reason_for_no_fp8 = ""


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
    fp8_group=None,
) -> None:
    """
    Context manager for FP8 usage.
    """
    if enabled:
        raise NotImplementedError("FP8 not implemented.")

    if (fp8_recipe is not None and fp8_recipe.reduce_amax) or fp8_group is not None:
        raise NotImplementedError("Distributed not implemented.")

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


def is_fp8_enabled() -> bool:
    """Is FP8 enabled"""
    return _FP8_ENABLED


def is_fp8_calibration() -> bool:
    """Is FP8 calibration"""
    return _FP8_CALIBRATION


def get_fp8_recipe() -> DelayedScaling:
    """Return the fp8 recipe"""
    return _FP8_RECIPE
