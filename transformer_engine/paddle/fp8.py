# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""FP8 utilities for TransformerEngine"""

from contextlib import contextmanager
from typing import Tuple, Optional, Dict, Any, Union

import numpy as np

import paddle
from transformer_engine import transformer_engine_paddle as tex
from transformer_engine.common.recipe import DelayedScaling, Format

from .constants import dist_group_type
from .fp8_buffer import FP8MetaFwdBuffer, FP8MetaBwdBuffer, FP8RecomputeBuffer

__all__ = ["fp8_autocast"]

# FP8 support
_is_fp8_available = None
_reason_for_no_fp8 = ""


def _check_fp8_support() -> Tuple[bool, str]:
    """Return if fp8 support is available"""

    # Check GPU arch
    arch = paddle.device.cuda.get_device_capability()
    if arch >= (9, 0):  # hopper and above
        return True, ""
    if arch < (8, 9):  # pre-ada
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
        self._fp8_enabled = False
        self._fp8_calibration = False
        self._fp8_recipe = None
        self._fp8_distributed_group = None
        self._is_first_fp8_module = False
        self._fp8_autocast_counter = 0
        self._fp8_autocast_depth = 0
        self._fp8_recompute_enabled = False
        self._fp8_fwd_buffer = FP8MetaFwdBuffer()
        self._fp8_bwd_buffer = FP8MetaBwdBuffer()
        self._fp8_recompute_buffer = FP8RecomputeBuffer()

    def is_fp8_enabled(self) -> bool:
        """Is FP8 enabled"""
        return self._fp8_enabled

    def is_fp8_calibration(self) -> bool:
        """Is FP8 calibration"""
        return self._fp8_calibration

    def get_fp8_recipe(self) -> DelayedScaling:
        """Return the fp8 recipe"""
        return self._fp8_recipe

    @staticmethod
    def get_default_fp8_recipe() -> DelayedScaling:
        """FP8 recipe with default args."""
        return DelayedScaling()

    def get_autocast_id(self) -> int:
        """Returns the number of times of entering the `fp8_autocast` context.
        as a unique ID for different training steps."""
        return self._fp8_autocast_counter

    def is_first_fp8_module(self):
        """Returns `True` only the first time when called multiple
        times from within the same `fp8_autocast` context.
        """
        tmp = self._is_first_fp8_module
        self._is_first_fp8_module = False
        return tmp

    def get_fp8_group(self) -> Union[dist_group_type, None]:
        """Return the fp8 group for scale/amax comm"""
        return self._fp8_distributed_group

    def get_fp8_fwd_buffer(self) -> FP8MetaFwdBuffer:
        """Returns global fp8 forward buffer."""
        return self._fp8_fwd_buffer

    def get_fp8_bwd_buffer(self) -> FP8MetaBwdBuffer:
        """Returns global fp8 backward buffer."""
        return self._fp8_bwd_buffer

    def is_fp8_recompute_enabled(self) -> bool:
        """Is FP8 recompute enabled"""
        return self._fp8_recompute_enabled

    def get_fp8_recompute_buffer(self) -> FP8RecomputeBuffer:
        """Returns global fp8 recompute buffer."""
        return self._fp8_recompute_buffer

    def enter(
        self,
        enabled: bool,
        calibrating: bool,
        fp8_recipe: Optional[DelayedScaling],
        fp8_group: Optional[dist_group_type],
    ) -> None:
        """Called when entering 'fp8_autocast'"""
        self.saved_states = (
            self._fp8_enabled,
            self._fp8_calibration,
            self._fp8_recipe,
            self._fp8_distributed_group,
            self._is_first_fp8_module,
        )

        self._fp8_enabled = enabled
        self._fp8_calibration = calibrating
        self._fp8_recipe = self.get_default_fp8_recipe() if fp8_recipe is None else fp8_recipe
        self._fp8_distributed_group = fp8_group

        if self._fp8_autocast_depth == 0:
            self._is_first_fp8_module = True
            self._fp8_autocast_counter += 1
        self._fp8_autocast_depth += 1

    def exit(self):
        """Called when exiting 'fp8_autocast'"""
        # Restore saved states
        (
            self._fp8_enabled,
            self._fp8_calibration,
            self._fp8_recipe,
            self._fp8_distributed_group,
            self._is_first_fp8_module,
        ) = self.saved_states

        self._fp8_autocast_depth -= 1

        if self._fp8_autocast_depth == 0:
            self._fp8_fwd_buffer.finalize()


_global_fp8_state = FP8State()


def get_global_fp8_state() -> FP8State:
    """Get global fp8 state"""
    return _global_fp8_state


@contextmanager
def fp8_autocast(
    enabled: bool = False,
    calibrating: bool = False,
    fp8_recipe: Optional[DelayedScaling] = None,
    fp8_group: Optional[dist_group_type] = None,
) -> None:
    """
    Context manager for FP8 usage.

    .. code-block:: python

        with fp8_autocast(enabled=True):
            out = model(inp)

    .. note::

        Support for FP8 in the Linear layer of Transformer Engine is currently limited to tensors
        with shapes where both dimensions are divisible by 16. In terms of the input to the full
        Transformer network, this typically requires padding sequence length to be multiple of 16.

    .. note::

        When :attr:`fp8_recipe.reduce_amax==True`, any module must not be invoked more than once
        inside a single `fp8_autocast` region. This is unsupported behavior because the amax
        reduction is handled during the exit of the `fp8_autocast` context. Calling the same
        module more than once inside an `fp8_autocast` region overrides the amax tensors
        before reduction can occur.

    Parameters
    ----------
    enabled: bool, default = `False`
             whether or not to enable fp8
    calibrating: bool, default = `False`
                 calibration mode allows collecting statistics such as amax and scale
                 data of fp8 tensors even when executing without fp8 enabled. This is
                 useful for saving an inference ready fp8 checkpoint while training
                 using a higher precision.
    fp8_recipe: recipe.DelayedScaling, default = `None`
                recipe used for FP8 training.
    fp8_group: paddle.distributed.collective.Group, default = `None`
               distributed group over which amaxes for the fp8 tensors
               are reduced at the end of each training step.
    """
    try:
        _global_fp8_state.enter(enabled, calibrating, fp8_recipe, fp8_group)

        if enabled:
            fp8_available, reason_for_no_fp8 = is_fp8_available()
            assert fp8_available, reason_for_no_fp8
        yield
    finally:
        _global_fp8_state.exit()


def get_fp8_te_dtype(fp8_recipe: DelayedScaling, fprop_tensor: bool = True) -> tex.DType:
    """Get fp8 data type according to recipe and tensor"""
    if fp8_recipe.fp8_format == Format.E4M3 or (
        fp8_recipe.fp8_format == Format.HYBRID and fprop_tensor
    ):
        return tex.DType.kFloat8E4M3
    return tex.DType.kFloat8E5M2


def amax_and_scale_update(
    fp8_meta: Dict[str, Any],
    fwd_update: bool,
    update_weight_scale_inv: bool = True,
) -> None:
    """Updates fp8 amaxes/scales for fwd | bwd."""
    amax_compute = fp8_meta["recipe"].amax_compute_algo
    sf_compute = fp8_meta["recipe"].scaling_factor_compute_algo
    fp8_meta_tensor_key = "scaling_fwd" if fwd_update else "scaling_bwd"

    if not callable(amax_compute) and sf_compute is None:
        non_weight_mask = fp8_meta[fp8_meta_tensor_key].non_weight_mask
        if update_weight_scale_inv:
            non_weight_mask = paddle.empty([0])
        tex.amax_and_scale_update_inplace(
            _amax_history=fp8_meta[fp8_meta_tensor_key].amax_history,
            _scale=fp8_meta[fp8_meta_tensor_key].scale,
            _scale_inv=fp8_meta[fp8_meta_tensor_key].scale_inv,
            non_weight_mask=non_weight_mask,
            fp8_dtype=int(get_fp8_te_dtype(fp8_meta["recipe"], fwd_update)),
            margin=float(fp8_meta["recipe"].margin),
            amax_compute=amax_compute,
        )
    else:
        raise ValueError(
            "We only support the fp8 recipe with 'max' or 'most_recent' "
            "amax_compute_algo and default scaling_factor_compute_algo at this "
            "moment."
        )


class FP8TensorMeta:
    """Holds FP8 scaling and amax history for FP8 layers"""

    def __init__(self, is_forward: bool):
        self.scale = paddle.Tensor()
        self.scale_inv = paddle.Tensor()
        self.amax_history = paddle.Tensor()
        self.non_weight_mask = paddle.Tensor()
        self.is_initialized = False
        self.is_forward = is_forward

    def get_non_weight_mask(self, num_gemms: int):
        """Needed for calculation of scale inverses to
        preserve scale_inv when caching FP8 weights"""
        if self.is_forward:
            # [True, False, True]: -> [input, weight, output]
            return paddle.to_tensor([True, False, True] * num_gemms)
        # [True, True]: -> [grad_output, grad_input]
        return paddle.to_tensor([True, True] * num_gemms)

    def prepare(self, num_gemms: int, amax_history_len: int) -> None:
        """Prepare scales and amax tensors. It is called during fprop in each iteration.
        If the meta tensors are not initialized yet, initialization is performed. If already
        initialized, resize the meta tensors if amax_history_len has changed."""

        if self.is_initialized:
            # Handle changed amax history size.
            curr_len = self.amax_history.shape[0]
            num_fp8_tensors = self.amax_history.shape[1]
            if amax_history_len < curr_len:
                self.amax_history = self.amax_history[:amax_history_len]
            elif amax_history_len > curr_len:
                extra_rows = amax_history_len - curr_len
                self.amax_history = paddle.concat(
                    [
                        self.amax_history,
                        paddle.zeros((extra_rows, num_fp8_tensors), dtype="float32"),
                    ],
                    axis=0,
                )
            return

        # Max. number of fp8 tensors per GEMM = 3 (input, weight, output) for fwd and
        # 2 (grad_output and grad_input) for bwd
        num_fp8_tensors = num_gemms * 3 if self.is_forward else num_gemms * 2

        self.scale = paddle.ones(num_fp8_tensors, dtype="float32")
        self.scale_inv = paddle.ones(num_fp8_tensors, dtype="float32")
        self.amax_history = paddle.zeros([amax_history_len, num_fp8_tensors], dtype="float32")
        self.non_weight_mask = self.get_non_weight_mask(num_gemms=num_gemms)

        self.is_initialized = True

    def to_numpy(self):
        """Convert FP8 meta tensors to numpy."""
        assert self.is_initialized, "FP8TensorMeta is not initialized yet."
        return {
            "scale": self.scale.numpy(),
            "scale_inv": self.scale_inv.numpy(),
            "amax_history": self.amax_history.numpy(),
        }

    def from_numpy(self, data: Dict[str, np.array]):
        """Set FP8 meta tensors from numpy"""
        self.scale = paddle.to_tensor(data["scale"])
        self.scale_inv = paddle.to_tensor(data["scale_inv"])
        self.amax_history = paddle.to_tensor(data["amax_history"])

        num_fp8_tensors = self.scale.shape[0]
        num_gemms = num_fp8_tensors // 3 if self.is_forward else num_fp8_tensors // 2
        self.non_weight_mask = self.get_non_weight_mask(num_gemms=num_gemms)

        self.is_initialized = True
