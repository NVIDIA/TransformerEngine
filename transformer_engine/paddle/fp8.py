# Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""FP8 utilities for TransformerEngine"""

import copy
from contextlib import contextmanager
from typing import Tuple, Optional, Dict, Any

import numpy as np

import paddle
import transformer_engine_paddle as tex
from transformer_engine.common.recipe import DelayedScaling, Format

# FP8 support
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


class FP8State:
    """Stores FP8 state"""

    def __init__(self):
        self.fp8_enabled = False
        self.fp8_calibration = False
        self.fp8_recipe = None

    def is_fp8_enabled(self) -> bool:
        """Is FP8 enabled"""
        return self.fp8_enabled

    def is_fp8_calibration(self) -> bool:
        """Is FP8 calibration"""
        return self.fp8_calibration

    def get_fp8_recipe(self) -> DelayedScaling:
        """Return the fp8 recipe"""
        return self.fp8_recipe

    @staticmethod
    def get_default_fp8_recipe() -> DelayedScaling:
        """FP8 recipe if not provided by user
        Margin = 0, interval = 1, E4M3
        """
        return DelayedScaling()


_global_fp8_state = FP8State()


def get_global_fp8_state() -> FP8State:
    """Get global fp8 state"""
    return _global_fp8_state


@contextmanager
def fp8_autocast(
    enabled: bool = False,
    calibrating: bool = False,
    fp8_recipe: Optional[DelayedScaling] = None,
) -> None:
    """
    Context manager for FP8 usage.
    """

    global _global_fp8_state
    saved_fp8_state = copy.deepcopy(_global_fp8_state)
    try:
        _global_fp8_state.fp8_enabled = enabled
        _global_fp8_state.fp8_calibration = calibrating
        _global_fp8_state.fp8_recipe = FP8State.get_default_fp8_recipe(
        ) if fp8_recipe is None else fp8_recipe

        if enabled:
            fp8_available, reason_for_no_fp8 = is_fp8_available()
            assert fp8_available, reason_for_no_fp8
        yield
    finally:
        _global_fp8_state = saved_fp8_state


def get_fp8_te_dtype(fp8_recipe: DelayedScaling, fprop_tensor: bool = True) -> tex.DType:
    """Get fp8 data type according to recipe and tensor"""
    if fp8_recipe.fp8_format == Format.E4M3 or (fp8_recipe.fp8_format == Format.HYBRID
                                                and fprop_tensor):
        return tex.DType.kFloat8E4M3
    return tex.DType.kFloat8E5M2


def amax_and_scale_update(
    fp8_meta: Dict[str, Any],
    fwd_update: bool,
) -> None:
    """Updates fp8 amaxes/scales for fwd | bwd."""
    amax_compute = fp8_meta["recipe"].amax_compute_algo
    sf_compute = fp8_meta["recipe"].scaling_factor_compute_algo
    fp8_meta_tensor_key = "scaling_fwd" if fwd_update else "scaling_bwd"
    fp8_max_key = "fp8_max_fwd" if fwd_update else "fp8_max_bwd"

    if not callable(amax_compute) and sf_compute is None:
        # Obtain amax from history
        amax_history = fp8_meta[fp8_meta_tensor_key].amax_history
        if amax_compute == "max":
            amax = paddle.max(amax_history, axis=0)
        else:    # amax_compute_algo == "most_recent"
            amax = amax_history[0]

        # Update amax history and set next amax to zero
        if amax_history.shape[0] > 1:
            amax_history = paddle.roll(amax_history, -1, 0)
        amax_history[0] = 0.0
        fp8_meta[fp8_meta_tensor_key].amax_history = amax_history

        # Update scaling factor
        fp8_meta[fp8_meta_tensor_key].scale = tex.update_scale(
            amax=amax,
            scale=fp8_meta[fp8_meta_tensor_key].scale,
            fp8_max=fp8_meta[fp8_max_key],
            margin=float(fp8_meta["recipe"].margin))

        # Update scale_inv
        fp8_meta[fp8_meta_tensor_key].scale_inv = \
                    1.0 / fp8_meta[fp8_meta_tensor_key].scale

    else:
        raise ValueError("We only support the fp8 recipe with 'max' or 'most_recent' "
                         "amax_compute_algo and default scaling_factor_compute_algo at this "
                         "moment.")


class FP8TensorMeta():
    """Holds FP8 scaling and amax history for FP8 layers"""

    def __init__(self, is_forward: bool):
        self.scale = paddle.Tensor()
        self.scale_inv = paddle.Tensor()
        self.amax_history = paddle.Tensor()
        self.is_initialized = False
        self.is_forward = is_forward

    def prepare(self, num_gemms: bool, amax_history_len: int) -> None:
        """Prepare scales and amax tensors. It is called during fprop in each iteration.
        If the meta tensors are not initialized yet, initialization is performed. If already
        initialized, resize the meta tensors if amax_history_len has changed."""

        if self.is_initialized:
            # Handle changed amax history size.
            curr_len = self.amax_history.shape[0]
            num_fp8_tensors = self.amax_history.shape[1]
            if amax_history_len < curr_len:
                self.amax_history = (self.amax_history[:amax_history_len])
            elif amax_history_len > curr_len:
                extra_rows = amax_history_len - curr_len
                self.amax_history = paddle.concat([
                    self.amax_history,
                    paddle.zeros((extra_rows, num_fp8_tensors), dtype='float32')
                ],
                                                  axis=0)
            return

        # Max. number of fp8 tensors per GEMM = 3 (input, weight, output) for fwd and
        # 2 (grad_output and grad_input) for bwd
        num_fp8_tensors = (num_gemms * 3 if self.is_forward else num_gemms * 2)

        self.scale = paddle.ones(num_fp8_tensors, dtype='float32')
        self.scale_inv = paddle.ones(num_fp8_tensors, dtype='float32')
        self.amax_history = paddle.zeros([amax_history_len, num_fp8_tensors], dtype='float32')
        self.is_initialized = True

    def to_numpy(self):
        """Convert FP8 meta tensors to numpy."""
        assert self.is_initialized, "FP8TensorMeta is not initialized yet."
        return {
            'scale': self.scale.numpy(),
            'scale_inv': self.scale_inv.numpy(),
            'amax_history': self.amax_history.numpy(),
        }

    def from_numpy(self, data: Dict[str, np.array]):
        """Set FP8 meta tensors from numpy"""
        self.scale = paddle.to_tensor(data['scale'])
        self.scale_inv = paddle.to_tensor(data['scale_inv'])
        self.amax_history = paddle.to_tensor(data['amax_history'])
        self.is_initialized = True
