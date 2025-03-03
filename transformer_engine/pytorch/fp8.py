# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""FP8 utilities for TransformerEngine"""
from __future__ import annotations

import abc
import os
from contextlib import contextmanager
from collections import deque
from typing import Callable, List, Optional, Dict, Any, Tuple, Union

import torch
import transformer_engine_torch as tex
from transformer_engine.common.recipe import Recipe, DelayedScaling, Format, MXFP8BlockScaling

from .constants import dist_group_type
from .utils import get_device_compute_capability
from .jit import jit_fuser


__all__ = ["fp8_autocast", "fp8_model_init"]


def check_fp8_support() -> Tuple[bool, str]:
    """Return if fp8 support is available"""
    if get_device_compute_capability() >= (9, 0):  # hopper and above
        return True, ""
    if get_device_compute_capability() < (8, 9):  # pre-ada
        return False, "Device compute capability 8.9 or higher required for FP8 execution."
    if tex.get_cublasLt_version() < 120103:
        return False, "CublasLt version 12.1.3.x or higher required for FP8 execution on Ada."
    if float(torch.version.cuda) < 12.1:
        return False, "Cuda version 12.1 or higher required for FP8 execution on Ada."
    return True, ""


def check_mxfp8_support() -> Tuple[bool, str]:
    """Return if fp8 support is available"""
    if get_device_compute_capability() >= (10, 0):  # blackwell and above
        return True, ""
    return False, "Device compute capability 10.0 or higher required for MXFP8 execution."


def get_default_fp8_recipe() -> Recipe:
    """FP8 recipe with default args."""
    if get_device_compute_capability() >= (10, 0):  # blackwell and above
        return MXFP8BlockScaling()
    return DelayedScaling()


def get_fp8_torch_dtype(fp8_recipe: Recipe, fprop_tensor: bool = True) -> torch.dtype:
    """Get fp8 data type according to recipe and tensor"""
    if fp8_recipe.fp8_format == Format.E4M3 or (
        fp8_recipe.fp8_format == Format.HYBRID and fprop_tensor
    ):
        return torch.float8_e4m3fn
    return torch.float8_e5m2


def get_fp8_te_dtype(fp8_recipe: Recipe, fprop_tensor: bool = True) -> tex.DType:
    """Get fp8 data type according to recipe and tensor"""
    if fp8_recipe.fp8_format == Format.E4M3 or (
        fp8_recipe.fp8_format == Format.HYBRID and fprop_tensor
    ):
        return tex.DType.kFloat8E4M3
    return tex.DType.kFloat8E5M2


def get_fp8_max(fp8_recipe: Recipe, fprop_tensor: bool = True) -> tex.DType:
    """Get max representible FP8 value."""
    if fp8_recipe.fp8_format == Format.E4M3 or (
        fp8_recipe.fp8_format == Format.HYBRID and fprop_tensor
    ):
        return Format.E4M3.value.max_fwd
    return Format.E5M2.value.max_fwd


class FP8GlobalStateManager:
    """Class to keep track of and manipulate the global
    FP8 state at different stages of execution.
    """

    FP8_ENABLED = False
    FP8_CALIBRATION = False
    FP8_RECIPE = None
    FP8_DISTRIBUTED_GROUP = None
    FP8_PARAMETERS = False
    IS_FIRST_FP8_MODULE = False
    FP8_GRAPH_CAPTURING = False
    FP8_AUTOCAST_DEPTH = 0
    global_amax_buffer = {}
    global_amax_history_buffer = {}
    global_scale_buffer = {}
    fp8_tensors_recompute_buffer = []
    fp8_available = None
    reason_for_no_fp8 = ""
    autocast_arguments = {}
    autocast_to_fp8_params = {}
    fp8_param_to_autocast = {}
    skip_fp8_weight_update_tensor = None
    mxfp8_available = None
    reason_for_no_mxfp8 = ""

    @classmethod
    def reset(cls) -> None:
        """Reset the global state"""
        cls.FP8_ENABLED = False
        cls.FP8_CALIBRATION = False
        cls.FP8_RECIPE = None
        cls.FP8_DISTRIBUTED_GROUP = None
        cls.FP8_PARAMETERS = False
        cls.IS_FIRST_FP8_MODULE = False
        cls.FP8_GRAPH_CAPTURING = False
        cls.FP8_AUTOCAST_DEPTH = 0
        cls.global_amax_buffer = {}
        cls.global_amax_history_buffer = {}
        cls.global_scale_buffer = {}
        cls.fp8_tensors_recompute_buffer = []
        cls.fp8_available = None
        cls.reason_for_no_fp8 = ""
        cls.autocast_arguments = {}
        cls.autocast_to_fp8_params = {}
        cls.fp8_param_to_autocast = {}
        cls.skip_fp8_weight_update_tensor = None
        cls.mxfp8_available = None
        cls.reason_for_no_mxfp8 = ""

    @classmethod
    def set_skip_fp8_weight_update_tensor(cls, skip: bool) -> None:
        """`skip_fp8_weight_update_tensor` inplace setter."""
        if cls.skip_fp8_weight_update_tensor is None:
            cls.skip_fp8_weight_update_tensor = torch.empty(1, dtype=torch.float32, device="cuda")
        cls.skip_fp8_weight_update_tensor.fill_(skip)

    @classmethod
    def get_skip_fp8_weight_update_tensor(cls) -> None:
        """`skip_fp8_weight_update_tensor` getter."""
        return cls.skip_fp8_weight_update_tensor

    @classmethod
    def is_fp8_available(cls) -> Tuple[bool, str]:
        """Return if fp8 support is available"""
        if cls.fp8_available is None:
            cls.fp8_available, cls.reason_for_no_fp8 = check_fp8_support()
        return cls.fp8_available, cls.reason_for_no_fp8

    @classmethod
    def is_mxfp8_available(cls) -> Tuple[bool, str]:
        """Return if MXFP8/current scaling support is available."""
        if cls.mxfp8_available is None:
            cls.mxfp8_available, cls.reason_for_no_mxfp8 = check_mxfp8_support()
        return cls.mxfp8_available, cls.reason_for_no_mxfp8

    @staticmethod
    def get_meta_tensor_key(forward: bool = True) -> str:
        """Returns scaling key in `fp8_meta`."""
        if forward:
            return "scaling_fwd"
        return "scaling_bwd"

    @staticmethod
    def get_fwd_bwd_key(forward: bool = True) -> str:
        """Convert bool `forward` to string."""
        return "forward" if forward else "backward"

    @classmethod
    def get_buffer_info(cls) -> str:
        """
        Returns a key for `fp8_meta` that stores the module's index
        in the global buffers along with autocast information.
        """
        return "buffer_index_and_autocast_key"

    @classmethod
    def get_key_in_buffer(
        cls,
        forward: bool,
        fp8_recipe: Recipe,
        fp8_group: dist_group_type,
    ) -> str:
        """Returns a key into the global FP8 buffers."""
        autocast_key = cls.get_unique_autocast_key(fp8_recipe, fp8_group)
        fwd_bwd_key = cls.get_fwd_bwd_key(forward)
        return f"{fwd_bwd_key}_{autocast_key}"

    @classmethod
    def split_key_in_buffer(cls, key: str) -> Tuple[bool, str]:
        """Splits buffer key into relevant parts."""
        forward, autocast_key = key.split("_", 1)
        forward = forward == "forward"
        return forward, autocast_key

    @classmethod
    def add_fp8_tensors_to_global_buffer(
        cls,
        fp8_meta: Dict[str, Any],
    ) -> None:
        """
        The amax reduction process happens completely outside the FP8 modules.
        To participate in the reduction, the only role played by a module is
        to call this function in order to append it's FP8 tensor into a global
        buffer. There are 5 global buffers maintained, one each for amax, amax
        history, scale, scale-inverse, and non-weight-mask. Each buffer has
        keys that hold FP8 tensors. Keys have a `forward_` or `backward_` prefix
        to indicate the type of FP8 tensor, since the forward and backward
        reductions happen separately.

        Note: For CG capture, this method is called from the graphed
        wrapper. For non CG case, it's called from within the module.
        """

        if fp8_meta["recipe"].mxfp8():
            return

        # Every module must call this function exactly once since
        # the amax tensors are static. Ensures that compatibility
        # with non-graphed modules is maintained.
        index_in_buffer = cls.get_buffer_info()  # Same index for fwd/bwd fp8 tensors.
        if index_in_buffer in fp8_meta:
            return

        fp8_meta[index_in_buffer] = []
        for forward in (True, False):
            fp8_meta_tensor_key = cls.get_meta_tensor_key(forward=forward)
            if fp8_meta_tensor_key not in fp8_meta:
                # Handles non-parameter FP8 modules, e.g. DPA.
                continue

            key = cls.get_key_in_buffer(forward, fp8_meta["recipe"], fp8_meta["fp8_group"])

            if key not in cls.global_amax_buffer:
                cls.global_amax_buffer[key] = [fp8_meta[fp8_meta_tensor_key].amax_history[0]]
                cls.global_amax_history_buffer[key] = [fp8_meta[fp8_meta_tensor_key].amax_history]
                cls.global_scale_buffer[key] = [fp8_meta[fp8_meta_tensor_key].scale]
            else:
                cls.global_amax_buffer[key].append(fp8_meta[fp8_meta_tensor_key].amax_history[0])
                cls.global_amax_history_buffer[key].append(
                    fp8_meta[fp8_meta_tensor_key].amax_history
                )
                cls.global_scale_buffer[key].append(fp8_meta[fp8_meta_tensor_key].scale)
            fp8_meta[index_in_buffer].append(len(cls.global_amax_buffer[key]) - 1)
            fp8_meta[index_in_buffer].append(key)

    @classmethod
    def is_fp8_enabled(cls) -> bool:
        """Is FP8 enabled"""
        return cls.FP8_ENABLED

    @classmethod
    def is_fp8_calibration(cls) -> bool:
        """Is FP8 calibration"""
        return cls.FP8_CALIBRATION

    @classmethod
    def with_fp8_parameters(cls) -> bool:
        """Should the parameters be stored as FP8"""
        return cls.FP8_PARAMETERS

    @classmethod
    def fp8_graph_capturing(cls) -> bool:
        """Is CUDA graph capture under way?"""
        return cls.FP8_GRAPH_CAPTURING or torch.cuda.is_current_stream_capturing()

    @classmethod
    def is_first_fp8_module(cls):
        """Returns `True` only the first time when called multiple
        times from within the same `fp8_autocast` context.
        """
        tmp = cls.IS_FIRST_FP8_MODULE
        cls.IS_FIRST_FP8_MODULE = False
        return tmp

    @classmethod
    def get_fp8_recipe(cls) -> Recipe:
        """Return the fp8 recipe"""
        if cls.FP8_RECIPE is not None:
            return cls.FP8_RECIPE
        return get_default_fp8_recipe()

    @classmethod
    def get_fp8_group(cls) -> Union[dist_group_type, None]:
        """Return the fp8 group for scale/amax comm"""
        return cls.FP8_DISTRIBUTED_GROUP

    @classmethod
    def get_fp8_autocast_state(cls) -> Tuple[bool, bool, Recipe, dist_group_type, bool]:
        """FP8 autocast state getter"""
        return (
            cls.FP8_ENABLED,
            cls.FP8_CALIBRATION,
            cls.FP8_RECIPE,
            cls.FP8_DISTRIBUTED_GROUP,
            cls.IS_FIRST_FP8_MODULE,
            cls.FP8_GRAPH_CAPTURING,
        )

    @classmethod
    def set_fp8_autocast_state(
        cls, fp8_state: Tuple[bool, bool, DelayedScaling, dist_group_type, bool]
    ) -> None:
        """FP8 autocast state setter"""
        (
            cls.FP8_ENABLED,
            cls.FP8_CALIBRATION,
            cls.FP8_RECIPE,
            cls.FP8_DISTRIBUTED_GROUP,
            cls.IS_FIRST_FP8_MODULE,
            cls.FP8_GRAPH_CAPTURING,
        ) = fp8_state

    @staticmethod
    def reduce_tensor_across_group_op_max(tensor: torch.Tensor, group: dist_group_type) -> None:
        """Reduce tensor across given group."""
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(
                tensor,
                op=torch.distributed.ReduceOp.MAX,
                group=group,
                async_op=False,
            )

    @classmethod
    def reduce_and_update_fp8_tensors(
        cls,
        forward: bool = True,
    ) -> None:
        """Concatenate, reduce, and split amaxes in the global buffer."""
        for buffer_key, amax_buffer in cls.global_amax_buffer.items():
            # Check for forward or backward reduction.
            fwd_update, autocast_key = cls.split_key_in_buffer(buffer_key)
            if fwd_update != forward:
                continue
            if len(amax_buffer) == 0:
                continue

            # Retrieve autocast specific args and concat amaxes.
            recipe, group = cls.autocast_arguments[autocast_key]
            contiguous_amax = torch.cat(amax_buffer)

            # Reduction.
            if (
                recipe.reduce_amax
                and torch.distributed.is_initialized()
                and torch.distributed.get_world_size(group=group) > 1
            ):
                cls.reduce_tensor_across_group_op_max(contiguous_amax, group)

            # Amax and scale update.
            unfused_update = (
                bool(int(os.getenv("NVTE_UNFUSED_FP8_UPDATE", "0")))
                or callable(recipe.amax_compute_algo)
                or callable(recipe.scaling_factor_compute_algo)
            )

            if not unfused_update:
                tex.fused_amax_and_scale_update_after_reduction(
                    contiguous_amax,
                    cls.global_amax_history_buffer[buffer_key],
                    cls.global_scale_buffer[buffer_key],
                    recipe.amax_compute_algo,
                    get_fp8_te_dtype(recipe, forward),
                    recipe.margin,
                )
            else:
                split_and_copy(contiguous_amax, amax_buffer, [x.numel() for x in amax_buffer])

                for amax_history, scale in zip(
                    cls.global_amax_history_buffer[buffer_key],
                    cls.global_scale_buffer[buffer_key],
                ):
                    _amax_and_scale_update(
                        amax_history, scale, get_fp8_max(recipe, forward), recipe
                    )

    @classmethod
    def get_unique_autocast_key(
        cls,
        recipe: Optional[Recipe] = None,
        group: Optional[dist_group_type] = None,
    ):
        """
        For FP8, each autocast can be uniquely identified by the recipe and fp8 group.
        Safely using `hash` as we never cross checkpoint boundaries.
        """
        return f"{str(recipe)}:{hash(group)}"

    @classmethod
    def fp8_autocast_enter(
        cls,
        enabled: bool = False,
        calibrating: bool = False,
        fp8_recipe: Optional[Recipe] = None,
        fp8_group: Optional[dist_group_type] = None,
        _graph: bool = False,
    ) -> None:
        """Set state and tracking variables for entry into FP8 region."""

        fp8_recipe = get_default_fp8_recipe() if fp8_recipe is None else fp8_recipe
        autocast_key = cls.get_unique_autocast_key(fp8_recipe, fp8_group)
        cls.autocast_arguments[autocast_key] = (fp8_recipe, fp8_group)

        cls.FP8_ENABLED = enabled
        cls.FP8_CALIBRATION = calibrating
        cls.FP8_RECIPE = fp8_recipe
        cls.FP8_DISTRIBUTED_GROUP = fp8_group
        cls.FP8_GRAPH_CAPTURING = _graph

        if cls.FP8_AUTOCAST_DEPTH == 0:
            cls.IS_FIRST_FP8_MODULE = True
        cls.FP8_AUTOCAST_DEPTH += 1

        if enabled:
            fp8_available, reason_for_no_fp8 = cls.is_fp8_available()
            assert fp8_available, reason_for_no_fp8
            if isinstance(fp8_recipe, MXFP8BlockScaling):
                mxfp8_available, reason_for_no_mxfp8 = cls.is_mxfp8_available()
                assert mxfp8_available, reason_for_no_mxfp8

    @classmethod
    def fp8_autocast_exit(cls, enabled: bool, _graph: bool) -> None:
        """Set state and tracking variables for exit from FP8 region."""
        cls.FP8_AUTOCAST_DEPTH -= 1
        # Reduce only the non-FP8 weight modules here.
        # FP8 weight modules are reduced at the end of the optimizer
        # step after the weight amax is populated.
        if enabled and cls.FP8_AUTOCAST_DEPTH == 0 and not _graph and torch.is_grad_enabled():
            cls.reduce_and_update_fp8_tensors(forward=True)

    @classmethod
    def copy_forward_fp8_meta_tensors_for_recompute(cls, fp8_meta: Dict[str, Any]) -> None:
        """Copy the scaling factors and amaxes for recompute forward phase
        to ensure both forward steps are numerically same.
        """

        if fp8_meta["recipe"].mxfp8():
            return

        buffer_position_key = "global_fp8_buffer_pos_fwd_recompute"

        to_copy = [
            fp8_meta["scaling_fwd"].amax_history.clone(),
            fp8_meta["scaling_fwd"].scale.clone(),
        ]

        if buffer_position_key in fp8_meta:
            cls.fp8_tensors_recompute_buffer[fp8_meta[buffer_position_key]].append(to_copy)
        else:
            if len(cls.fp8_tensors_recompute_buffer) == 0:
                cls.fp8_tensors_recompute_buffer = [deque()]
            else:
                cls.fp8_tensors_recompute_buffer.append(deque())
            cls.fp8_tensors_recompute_buffer[-1].append(to_copy)
            fp8_meta[buffer_position_key] = len(cls.fp8_tensors_recompute_buffer) - 1

    @classmethod
    def get_old_fp8_meta_tensors_for_recompute(cls, fp8_meta: Dict[str, Any]) -> None:
        """Switch to the copied scaling factors and amaxes from phase
        1 forward for indentical numerical outputs.
        """

        if fp8_meta["recipe"].mxfp8():
            return

        # Store updated amaxes and scales from phase 1 post forward.
        fp8_meta["updated_amax_history_fwd"] = fp8_meta["scaling_fwd"].amax_history
        fp8_meta["updated_scale_fwd"] = fp8_meta["scaling_fwd"].scale

        # Retrieve stashed amaxes and scales from phase 1 pre forward.
        buffer_position_key = "global_fp8_buffer_pos_fwd_recompute"
        stashed_fp8_meta = cls.fp8_tensors_recompute_buffer[fp8_meta[buffer_position_key]].popleft()

        # Replace amaxes and scales with stashed values for phase 2 forward
        fp8_meta["scaling_fwd"].amax_history.copy_(stashed_fp8_meta[0])
        fp8_meta["scaling_fwd"].scale.copy_(stashed_fp8_meta[1])

    @staticmethod
    def restore_fp8_meta_tensors(fp8_meta: Dict[str, Any]) -> None:
        """Restore latest scaling factors and amaxes after recompute forward run."""

        if fp8_meta["recipe"].mxfp8():
            return

        fp8_meta["scaling_fwd"].amax_history.copy_(fp8_meta["updated_amax_history_fwd"])
        fp8_meta["scaling_fwd"].scale.copy_(fp8_meta["updated_scale_fwd"])


@contextmanager
def fp8_model_init(enabled: bool = True, recipe: Optional[Recipe] = None) -> None:
    """
    Context manager for FP8 initialization of parameters.

    Example usage:

    .. code-block:: python

        with fp8_model_init(enabled=True):
            model = transformer_engine.pytorch.Linear(768, 768)

    Parameters
    ----------
    enabled: bool, default = `True`
             when enabled, Transformer Engine modules created inside this `fp8_model_init`
             region will hold only FP8 copies of its parameters, as opposed to the default
             behavior where both higher precision and FP8 copies are present. Setting this
             option to `True` may result in lower memory consumption and is especially
             useful for scenarios like:

             * full model training using optimizer with master weights, where the high
               precision copies of weights are already present in the optimizer.
             * inference, where only the FP8 copies of the parameters are used.
             * LoRA-like fine-tuning, where the main parameters of the model do not change.
    recipe: transformer_engine.common.recipe.Recipe, default = `None`
            Recipe used to create the parameters. If left to None, it uses the default FP8 recipe.

             This functionality is *EXPERIMENTAL*.
    """
    _fp8_parameters = FP8GlobalStateManager.FP8_PARAMETERS
    _fp8_recipe = FP8GlobalStateManager.FP8_RECIPE
    FP8GlobalStateManager.FP8_PARAMETERS = enabled
    FP8GlobalStateManager.FP8_RECIPE = get_default_fp8_recipe() if recipe is None else recipe
    try:
        yield
    finally:
        FP8GlobalStateManager.FP8_PARAMETERS = _fp8_parameters
        FP8GlobalStateManager.FP8_RECIPE = _fp8_recipe


@contextmanager
def fp8_autocast(
    enabled: bool = True,
    calibrating: bool = False,
    fp8_recipe: Optional[Recipe] = None,
    fp8_group: Optional[dist_group_type] = None,
    _graph: bool = False,
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
    enabled: bool, default = `True`
             whether or not to enable fp8
    calibrating: bool, default = `False`
                 calibration mode allows collecting statistics such as amax and scale
                 data of fp8 tensors even when executing without fp8 enabled. This is
                 useful for saving an inference ready fp8 checkpoint while training
                 using a higher precision.
    fp8_recipe: recipe.Recipe, default = `None`
                recipe used for FP8 training.
    fp8_group: torch._C._distributed_c10d.ProcessGroup, default = `None`
               distributed group over which amaxes for the fp8 tensors
               are reduced at the end of each training step.
    """
    fp8_state = FP8GlobalStateManager.get_fp8_autocast_state()
    FP8GlobalStateManager.fp8_autocast_enter(
        enabled=enabled,
        calibrating=calibrating,
        fp8_recipe=fp8_recipe,
        fp8_group=fp8_group,
        _graph=_graph,
    )
    try:
        yield
    finally:
        FP8GlobalStateManager.set_fp8_autocast_state(fp8_state)
        FP8GlobalStateManager.fp8_autocast_exit(enabled, _graph=_graph)


def _update_amax_history(amax_history: torch.Tensor) -> torch.Tensor:
    """Update amax history and set next amax to zero."""
    if amax_history.shape[0] > 1:
        new_amax_history = torch.roll(amax_history, -1, 0)
        amax_history.copy_(new_amax_history)
    amax_history[0].fill_(0.0)
    return amax_history


@torch.jit.script
def _default_get_amax_and_update_history(
    amax_history: torch.Tensor,
    amax_compute_algo: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Default function to obtain amax from history."""
    if amax_compute_algo == "max":
        amax = torch.max(amax_history, dim=0).values
    else:  # amax_compute_algo == "most_recent"
        amax = amax_history[0].clone()

    amax_history = _update_amax_history(amax_history)
    return amax_history, amax


@jit_fuser
def _default_sf_compute(
    amax: torch.Tensor,
    scale: torch.Tensor,
    fp8_max: float,
    margin: int,
    _fp32_max: float = torch.finfo(torch.float32).max,  # finfo not available in jitter
) -> torch.Tensor:
    """Default function to convert amax to scaling factor.
    Computing the scaling factor requires consideration of the following scenarios:
    1. amax == 0:
       No action is possible, set scale to the previous scale (or 1).
    2. 0 < amax < tiny_amax
       The amax is too tiny that the scale becomes infinite in FP32.
       Set scale = FP32_max
    3. tiny_amax <= amax < FP32_max:
       Set scale = FP8_max (or scaled_max) / amax
    4. When amax == inf or amax == nan:
       No action is possible, set scale to the previous scale (or 1).
    """
    sf = (fp8_max / amax) / (2**margin)
    sf = torch.where(amax > 0.0, sf, scale)
    sf = torch.where(torch.isfinite(amax), sf, scale)
    sf = torch.where(torch.isinf(sf), torch.full_like(sf, _fp32_max), sf)
    scale.copy_(sf)
    return scale


def _compute_amax_and_update_history(
    amax_history: torch.Tensor,
    amax_compute_algo: Union[Callable, str],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Obtain the amax from the history."""

    if callable(amax_compute_algo):
        amax = amax_compute_algo(amax_history)
        amax_history = _update_amax_history(amax_history)
        return amax_history, amax
    return _default_get_amax_and_update_history(
        amax_history,
        amax_compute_algo,
    )


def _compute_scaling_factor(
    amax: torch.Tensor,
    scale: torch.Tensor,
    fp8_max: float,
    recipe: DelayedScaling,
) -> torch.Tensor:
    """Convert amax to scaling factor."""

    if recipe.scaling_factor_compute_algo is None:
        return _default_sf_compute(
            amax,
            scale,
            fp8_max,
            recipe.margin,
        )
    return recipe.scaling_factor_compute_algo(amax, scale, fp8_max, recipe)


def _amax_and_scale_update(
    amax_history: torch.Tensor,
    scale: torch.Tensor,
    fp8_max: float,
    recipe: DelayedScaling,
) -> None:
    """Updates FP8 meta tensors."""
    new_amax_history, amax = _compute_amax_and_update_history(
        amax_history,
        recipe.amax_compute_algo,
    )
    new_scale = _compute_scaling_factor(amax, scale, fp8_max, recipe)
    scale.copy_(new_scale)
    amax_history.copy_(new_amax_history)


def split_and_copy(
    buffer: torch.Tensor,
    outputs: List[torch.Tensor],
    chunk_sizes: List[int],
) -> None:
    """Split `buffer` by `chunk_sizes` and copy into `outputs`."""
    splits = buffer.split(chunk_sizes)
    torch._foreach_copy_(outputs, splits)


class RecipeState(abc.ABC):
    """Configuration and state for a quantization recipe.

    This is a builder class for quantizers, which are in turn builder
    classes for quantized tensors.

    This class may pack together the state for multiple quantizers,
    which is helpful for applying fused kernels with less overhead.

    """

    @staticmethod
    def create(
        recipe: Recipe,
        *,
        mode: str,
        num_quantizers: int = 1,
        device: Optional[torch.device] = None,
    ) -> RecipeState:
        """Factory method to create the state for a quantization recipe

        Parameters
        ----------
        recipe: Recipe
            Quantization recipe.
        mode: {"forward", "backward"}
            Training stage where quantization will be performed.
        num_quantizers: int, default = 1
            Number of quantizers to create state for.
        device: torch.device, default = default CUDA device
            Device for quantized tensors.

        Returns
        -------
        RecipeState:
            Quantization recipe state.

        """

        cls = None
        if recipe.delayed():
            cls = DelayedScalingRecipeState
        elif recipe.mxfp8():
            cls = MXFP8BlockScalingRecipeState
        else:
            raise ValueError("{recipe.__class__.__name__} is not supported")
        return cls(
            recipe,
            mode=mode,
            num_quantizers=num_quantizers,
            device=device,
        )

    @abc.abstractmethod
    def make_quantizers(self) -> list:
        """Convert recipe state to quantizers.

        Quantizers are builder classes for quantized tensors. They are
        typically used to convert a high-precision tensor (e.g. in
        FP32 or BF16) into a quantized tensor (e.g. in FP8).

        """


class DelayedScalingRecipeState(RecipeState):
    """State for FP8 quantization with per-tensor delayed scaling.

    Delayed scaling recipe requires a scaling factor (applied when
    casting to FP8) and a history of max-abs values ("amax") from
    recent FP8 casts for updating the scaling factor. The scale update
    is handled externally by `FP8GlobalStateManager`.

    """

    recipe: DelayedScaling
    mode: str
    dtype: tex.DType
    scale: torch.Tensor
    amax_history: torch.Tensor

    def __init__(
        self,
        recipe: DelayedScaling,
        *,
        mode: str,
        num_quantizers: int = 1,
        device: Optional[torch.device] = None,
    ) -> None:
        self.recipe = recipe
        self.mode = mode
        self.num_quantizers = num_quantizers
        self.dtype = get_fp8_te_dtype(recipe, mode == "forward")

        # Allocate buffers
        if device is None:
            device = torch.device("cuda")
        self.scale = torch.ones(num_quantizers, dtype=torch.float32, device=device)
        self.amax_history = torch.zeros(
            recipe.amax_history_len,
            num_quantizers,
            dtype=torch.float32,
            device=device,
        )

    def make_quantizers(self) -> list:
        # TODO(ksivamani); Find better design for this, adding here to avoid circular import.
        from .tensor.float8_tensor import Float8Quantizer

        return [
            Float8Quantizer(self.scale[i], self.amax_history[0][i].reshape((1,)), self.dtype)
            for i in range(self.num_quantizers)
        ]


class MXFP8BlockScalingRecipeState(RecipeState):
    """Configuration for MXFP8 quantization.

    MXFP8 quantization does not require state.

    """

    recipe: MXFP8BlockScaling
    mode: str
    dtype: tex.DType

    def __init__(
        self,
        recipe: MXFP8BlockScaling,
        *,
        mode: str,
        num_quantizers: int = 1,
        device: Optional[torch.device] = None,
    ) -> None:
        self.recipe = recipe
        self.mode = mode
        self.num_quantizers = num_quantizers
        self.dtype = get_fp8_te_dtype(recipe, mode == "forward")

        # Allocate buffers
        if device is None:
            device = torch.device("cuda")

    def make_quantizers(self) -> list:
        # TODO(ksivamani); Find better design for this, adding here to avoid circular import.
        from .tensor.mxfp8_tensor import MXFP8Quantizer

        return [MXFP8Quantizer(self.dtype) for i in range(self.num_quantizers)]
