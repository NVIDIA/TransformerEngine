# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""FP8 utilities for TransformerEngine"""
import os
from contextlib import contextmanager
from collections import deque
from typing import Callable, List, Optional, Dict, Any, Tuple, Union

import torch
import transformer_engine_extensions as tex
from transformer_engine.common.recipe import DelayedScaling, Format

from .constants import dist_group_type
from .utils import get_device_compute_capability
from .jit import jit_fuser


__all__ = ["fp8_autocast", "fp8_model_init"]


def check_fp8_support() -> Tuple[bool, str]:
    """Return if fp8 support is available"""
    if get_device_compute_capability() >= (9, 0): # hopper and above
        return True, ""
    if get_device_compute_capability() < (8, 9): # pre-ada
        return False, "Device compute capability 8.9 or higher required for FP8 execution."
    if tex.get_cublasLt_version() < 120103:
        return False, "CublasLt version 12.1.3.x or higher required for FP8 execution on Ada."
    if float(torch.version.cuda) < 12.1:
        return False, "Cuda version 12.1 or higher required for FP8 execution on Ada."
    return True, ""


def get_default_fp8_recipe() -> DelayedScaling:
    """FP8 recipe if not provided by user
    Margin = 0, interval = 1, E4M3
    """
    return DelayedScaling()


def get_fp8_te_dtype(
    fp8_recipe: DelayedScaling, fprop_tensor: bool = True
) -> tex.DType:
    """Get fp8 data type according to recipe and tensor"""
    if fp8_recipe.fp8_format == Format.E4M3 or (
        fp8_recipe.fp8_format == Format.HYBRID and fprop_tensor
    ):
        return tex.DType.kFloat8E4M3
    return tex.DType.kFloat8E5M2


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
    FP8_AUTOCAST_COUNTER = 0
    FP8_CURRENT_CONTEXT_ID = 0
    FP8_AUTOCAST_DEPTH = 0
    global_fp8_buffer = {}
    fp8_tensors_recompute_buffer = []
    amax_forward_global_reduce_func = None
    buffer_delete_key_fwd = None
    buffer_delete_key_bwd = None
    amax_reduce_handle_fwd = None
    fp8_available = None
    reason_for_no_fp8 = ""
    dp_amax_reduce_interval = None
    dp_amax_reduce_forward_idx = 0
    dp_amax_reduce_backward_idx = 0

    @classmethod
    def reset(cls) -> None:
        """Reset the global state"""
        cls.FP8_ENABLED = False
        cls.FP8_CALIBRATION = False
        cls.FP8_RECIPE = None
        cls.FP8_DISTRIBUTED_GROUP = None
        cls.IS_FIRST_FP8_MODULE = False
        cls.FP8_AUTOCAST_COUNTER = 0
        cls.FP8_CURRENT_CONTEXT_ID = 0
        cls.FP8_AUTOCAST_DEPTH = 0
        cls.global_fp8_buffer = {}
        cls.fp8_tensors_recompute_buffer = []
        cls.amax_forward_global_reduce_func = None
        cls.buffer_delete_key_fwd = None
        cls.buffer_delete_key_bwd = None
        cls.amax_reduce_handle_fwd = None
        cls.fp8_available = None
        cls.reason_for_no_fp8 = ""
        cls.dp_amax_reduce_interval = None
        cls.dp_amax_reduce_forward_idx = 0
        cls.dp_amax_reduce_backward_idx = 0

    @classmethod
    def is_fp8_available(cls) -> Tuple[bool, str]:
        """Return if fp8 support is available"""
        if cls.fp8_available is None:
            cls.fp8_available, cls.reason_for_no_fp8 = check_fp8_support()
        return cls.fp8_available, cls.reason_for_no_fp8

    @classmethod
    def get_global_fp8_state_checkpoint(cls) -> Dict[str, Union[int, str]]:
        """Returns global fp8 state variables."""
        # Convert attributes to dictionary to make future proof against
        # changes in global state variables in order to make setting the
        # checkpoint backwards compatible.
        global_fp8_state = {}
        global_fp8_state["FP8_AUTOCAST_COUNTER"] = cls.FP8_AUTOCAST_COUNTER
        global_fp8_state["FP8_CURRENT_CONTEXT_ID"] = cls.FP8_CURRENT_CONTEXT_ID
        global_fp8_state["FP8_AUTOCAST_DEPTH"] = cls.FP8_AUTOCAST_DEPTH
        global_fp8_state["buffer_delete_key_fwd"] = cls.buffer_delete_key_fwd
        global_fp8_state["buffer_delete_key_bwd"] = cls.buffer_delete_key_bwd
        global_fp8_state["dp_amax_reduce_interval"] = cls.dp_amax_reduce_interval
        global_fp8_state["dp_amax_reduce_forward_idx"] = cls.dp_amax_reduce_forward_idx
        global_fp8_state["dp_amax_reduce_backward_idx"] = cls.dp_amax_reduce_backward_idx
        return global_fp8_state

    @classmethod
    def set_global_fp8_state_checkpoint(cls, state: Dict[str, Union[int, str]]) -> None:
        """Sets global fp8 state variables."""
        for k, v in state.items():
            if hasattr(cls, k):
                setattr(cls, k, v)

    @classmethod
    def get_global_fp8_buffer_checkpoint(cls) -> Dict[str, List[torch.Tensor]]:
        """Returns global fp8 amax buffer."""
        return cls.global_fp8_buffer

    @classmethod
    def set_global_fp8_buffer_checkpoint(cls, buffer: Dict[str, List[torch.Tensor]]) -> None:
        """Sets global fp8 amax buffer."""
        # Map all tensors back to GPU.
        for k, v in buffer.items():
            buffer[k] = [tensor.cuda() for tensor in v]

        cls.global_fp8_buffer = buffer

    @staticmethod
    def get_meta_tensor_key(forward: bool = True) -> str:
        """Returns scaling key in `fp8_meta`."""
        if forward:
            return "scaling_fwd"
        return "scaling_bwd"

    @staticmethod
    def get_buffer_position_key(forward: bool = True) -> str:
        """Returns module position key in `fp8_meta`."""
        if forward:
            return "global_fp8_buffer_pos_fwd"
        return "global_fp8_buffer_pos_bwd"

    @staticmethod
    def get_autocast_key(forward: bool = True) -> str:
        """Returns module position key in `fp8_meta`."""
        if forward:
            return "autocast_id_fwd"
        return "autocast_id_bwd"

    @staticmethod
    def get_amax_buffer_key(fp8_meta: Dict[str, Any], forward: bool = True) -> str:
        """Return a key in `_global_fp8_buffer` for the AMAX storage."""
        if forward:
            return f"FWD_AMAX_{fp8_meta['autocast_id_fwd']}"
        return f"BWD_AMAX_{fp8_meta['autocast_id_bwd']}"

    @classmethod
    def get_amax_reduce_handle_fwd(cls) -> Union[bool, None]:
        """Return AMAX reduction wait handle of forward prop."""
        return cls.amax_reduce_handle_fwd

    @classmethod
    def setup_amax_forward_global_reduce_func(cls, f: Callable) -> None:
        """Sets up the function to call during autocast exit."""
        cls.amax_forward_global_reduce_func = f

    @classmethod
    def add_amax_to_global_buffer(cls, fp8_meta: Dict[str, Any], forward: bool = True) -> None:
        """Append 1D tensor `amax` to global buffer."""
        buffer_key = cls.get_amax_buffer_key(fp8_meta, forward=forward)
        fp8_meta_tensor_key = cls.get_meta_tensor_key(forward=forward)
        buffer_position_key = cls.get_buffer_position_key(forward=forward)

        if buffer_key not in cls.global_fp8_buffer:
            cls.global_fp8_buffer[buffer_key] = [fp8_meta[fp8_meta_tensor_key].amax_history[0]]
        else:
            cls.global_fp8_buffer[buffer_key].append(
                fp8_meta[fp8_meta_tensor_key].amax_history[0]
            )

        if buffer_position_key not in fp8_meta:
            fp8_meta[buffer_position_key] = len(cls.global_fp8_buffer[buffer_key]) - 1

        # Catch incorrect fp8_autocast usage.
        assert fp8_meta[buffer_position_key] == len(cls.global_fp8_buffer[buffer_key]) - 1, \
            "Same module is being invoked more than once inside an `fp8_autocast` " \
            "region when using FP8 with amax reduction. This behavior is currently" \
            " unsupported. For more details and correct usage, please see " \
            "https://github.com/NVIDIA/TransformerEngine/pull/93."

    @classmethod
    def copy_amax_from_global_buffer(
        cls, fp8_meta: Dict[str, Any], forward: bool = True
    ) -> None:
        """Populate current amax with the correct location from buffer."""
        fp8_meta_tensor_key = cls.get_meta_tensor_key(forward=forward)
        buffer_position_key = cls.get_buffer_position_key(forward=forward)
        if buffer_position_key not in fp8_meta:
            return

        amax_buffer_key = cls.get_amax_buffer_key(fp8_meta, forward=forward)
        assert amax_buffer_key in cls.global_fp8_buffer, "TE internal error."

        fp8_meta[fp8_meta_tensor_key].amax_history[0] = cls.global_fp8_buffer[amax_buffer_key][
            fp8_meta[buffer_position_key]
        ]

    @classmethod
    def set_amax_buffer_key_deletion(
        cls, fp8_meta: Dict[str, Any], forward: bool = True
    ) -> None:
        """Delete this amax key from global buffer during autocast end."""
        if cls.get_autocast_key(forward=forward) not in fp8_meta:
            return
        if forward:
            cls.buffer_delete_key_fwd = cls.get_amax_buffer_key(fp8_meta, forward=forward)
        else:
            cls.buffer_delete_key_bwd = cls.get_amax_buffer_key(fp8_meta, forward=forward)

    @classmethod
    def delete_key_from_amax_buffer(cls, forward: bool = True) -> None:
        """Delete the key from global amax buffer."""
        if forward:
            if (
                cls.buffer_delete_key_fwd is not None
                and cls.buffer_delete_key_fwd in cls.global_fp8_buffer
            ):
                del cls.global_fp8_buffer[cls.buffer_delete_key_fwd]
        else:
            if (
                cls.buffer_delete_key_bwd is not None
                and cls.buffer_delete_key_bwd in cls.global_fp8_buffer
            ):
                del cls.global_fp8_buffer[cls.buffer_delete_key_bwd]

    @classmethod
    def get_fp8_context_id(cls) -> int:
        """Returns an ID for the current FP8 context."""
        return cls.FP8_CURRENT_CONTEXT_ID

    @classmethod
    def set_fp8_context_id(cls, ctx_id: int) -> None:
        """Sets the current FP8 context."""
        cls.FP8_CURRENT_CONTEXT_ID = ctx_id

    @classmethod
    def new_fp8_context_id(cls) -> int:
        """Returns global autocast counter as a proxy to be used
        as the autocast ID for FP8 modules.
        """
        return cls.FP8_AUTOCAST_COUNTER

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
    def is_first_fp8_module(cls):
        """Returns `True` only the first time when called multiple
        times from within the same `fp8_autocast` context.
        """
        tmp = cls.IS_FIRST_FP8_MODULE
        cls.IS_FIRST_FP8_MODULE = False
        return tmp

    @classmethod
    def get_fp8_recipe(cls) -> DelayedScaling:
        """Return the fp8 recipe"""
        return cls.FP8_RECIPE

    @classmethod
    def get_fp8_group(cls) -> Union[dist_group_type, None]:
        """Return the fp8 group for scale/amax comm"""
        return cls.FP8_DISTRIBUTED_GROUP

    @classmethod
    def get_fp8_autocast_state(cls) -> Tuple[bool, bool, DelayedScaling, dist_group_type, bool]:
        """FP8 autocast state getter"""
        return (
            cls.FP8_ENABLED,
            cls.FP8_CALIBRATION,
            cls.FP8_RECIPE,
            cls.FP8_DISTRIBUTED_GROUP,
            cls.IS_FIRST_FP8_MODULE)

    @classmethod
    def set_fp8_autocast_state(
        cls,
        fp8_state: Tuple[bool, bool, DelayedScaling, dist_group_type, bool]
    ) -> None:
        """FP8 autocast state setter"""
        (cls.FP8_ENABLED,
         cls.FP8_CALIBRATION,
         cls.FP8_RECIPE,
         cls.FP8_DISTRIBUTED_GROUP,
         cls.IS_FIRST_FP8_MODULE) = fp8_state

    @staticmethod
    def reduce_tensor_across_group_op_max(
        tensor: torch.Tensor, group: dist_group_type, async_op: bool
    ) -> None:
        """Reduce tensor across given group."""
        if torch.distributed.is_initialized():
            wait_handle = torch.distributed.all_reduce(
                tensor,
                op=torch.distributed.ReduceOp.MAX,
                group=group,
                async_op=async_op,
            )
            return wait_handle
        return None

    @classmethod
    def global_amax_reduction(
        cls,
        fp8_meta: Dict[str, Any],
        tp_group: dist_group_type,
        tp_size: int,
        forward: bool = True,
    ) -> None:
        """Concatenate, reduce, and split amaxes in the global buffer."""
        amax_buffer_key = cls.get_amax_buffer_key(fp8_meta, forward=forward)

        # Key already deleted.
        if amax_buffer_key not in cls.global_fp8_buffer:
            return None

        # Reduce AMAX in DP-domain at an interval.
        # `NVTE_DP_AMAX_REDUCE_INTERVAL` should be set as an integer value larger than 0. If
        # `NVTE_DP_AMAX_REDUCE_INTERVAL` is set to 0, AMAX is reduced only in TP domain.
        if cls.dp_amax_reduce_interval is None:
            cls.dp_amax_reduce_interval = int(os.getenv("NVTE_DP_AMAX_REDUCE_INTERVAL", "1"))

        if cls.dp_amax_reduce_interval == 0:
            tp_amax_reduce = True
        else:
            tp_amax_reduce = False
            if forward:
                if cls.dp_amax_reduce_forward_idx == 0:
                    reduce_group = fp8_meta["fp8_group"]
                else:
                    tp_amax_reduce = True
                cls.dp_amax_reduce_forward_idx = (
                    (cls.dp_amax_reduce_forward_idx + 1) % cls.dp_amax_reduce_interval)
            else:
                if cls.dp_amax_reduce_backward_idx == 0:
                    reduce_group = fp8_meta["fp8_group"]
                else:
                    tp_amax_reduce = True
                cls.dp_amax_reduce_backward_idx = (
                    (cls.dp_amax_reduce_backward_idx + 1) % cls.dp_amax_reduce_interval)

        if tp_amax_reduce:
            if tp_size > 1:
                reduce_group = tp_group
            else:
                return None

        chunk_sizes = [x.numel() for x in cls.global_fp8_buffer[amax_buffer_key]]
        contiguous_amax = torch.cat(cls.global_fp8_buffer[amax_buffer_key])

        wait_handle = cls.reduce_tensor_across_group_op_max(
            contiguous_amax,
            reduce_group,
            fp8_meta["async_amax_reduction"],
        )

        cls.global_fp8_buffer[amax_buffer_key] = list(contiguous_amax.split(chunk_sizes))
        return wait_handle

    @classmethod
    def fp8_autocast_enter(
        cls,
        enabled: bool = False,
        calibrating: bool = False,
        fp8_recipe: Optional[DelayedScaling] = None,
        fp8_group: Optional[dist_group_type] = None,
    ) -> None:
        """Set state and tracking variables for entry into FP8 region."""
        if cls.FP8_AUTOCAST_DEPTH == 0:
            if callable(cls.amax_forward_global_reduce_func):
                cls.amax_reduce_handle_fwd = cls.amax_forward_global_reduce_func() # pylint: disable=not-callable
            cls.delete_key_from_amax_buffer(forward=True)

        cls.FP8_ENABLED = enabled
        cls.FP8_CALIBRATION = calibrating
        cls.FP8_RECIPE = get_default_fp8_recipe() if fp8_recipe is None else fp8_recipe
        cls.FP8_DISTRIBUTED_GROUP = fp8_group

        if cls.FP8_AUTOCAST_DEPTH == 0:
            cls.IS_FIRST_FP8_MODULE = True
            cls.FP8_AUTOCAST_COUNTER += 1
        cls.FP8_AUTOCAST_DEPTH += 1

        if enabled:
            fp8_available, reason_for_no_fp8 = cls.is_fp8_available()
            assert fp8_available, reason_for_no_fp8

    @classmethod
    def fp8_autocast_exit(cls):
        """Set state and tracking variables for exit from FP8 region."""
        cls.FP8_AUTOCAST_DEPTH -= 1

    @classmethod
    def copy_forward_fp8_meta_tensors_for_recompute(cls, fp8_meta: Dict[str, Any]) -> None:
        """Copy the scaling factors and amaxes for recompute forward phase
        to ensure both forward steps are numerically same.
        """
        buffer_position_key = "global_fp8_buffer_pos_fwd_recompute"

        to_copy = [
            fp8_meta["scaling_fwd"].amax_history.clone(),
            fp8_meta["scaling_fwd"].scale.clone(),
            fp8_meta["scaling_fwd"].scale_inv.clone(),
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

        # Store updated amaxes and scales from phase 1 post forward.
        fp8_meta["updated_amax_history_fwd"] = fp8_meta["scaling_fwd"].amax_history
        fp8_meta["updated_scale_fwd"] = fp8_meta["scaling_fwd"].scale
        fp8_meta["updated_scale_inv_fwd"] = fp8_meta["scaling_fwd"].scale_inv

        # Retrieve stashed amaxes and scales from phase 1 pre forward.
        buffer_position_key = "global_fp8_buffer_pos_fwd_recompute"
        stashed_fp8_meta = cls.fp8_tensors_recompute_buffer[
            fp8_meta[buffer_position_key]
        ].popleft()

        # Replace amaxes and scales with stashed values for phase 2 forward
        fp8_meta["scaling_fwd"].amax_history = stashed_fp8_meta[0]
        fp8_meta["scaling_fwd"].scale = stashed_fp8_meta[1]
        fp8_meta["scaling_fwd"].scale_inv = stashed_fp8_meta[2]

    @staticmethod
    def restore_fp8_meta_tensors(fp8_meta: Dict[str, Any]) -> None:
        """Restore latest scaling factors and amaxes after recompute forward run."""
        fp8_meta["scaling_fwd"].amax_history = fp8_meta["updated_amax_history_fwd"]
        fp8_meta["scaling_fwd"].scale = fp8_meta["updated_scale_fwd"]
        fp8_meta["scaling_fwd"].scale_inv = fp8_meta["updated_scale_inv_fwd"]


@contextmanager
def fp8_model_init(enabled: bool = True) -> None:
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

             This functionality is *EXPERIMENTAL*.
    """
    try:
        _fp8_parameters = FP8GlobalStateManager.FP8_PARAMETERS
        FP8GlobalStateManager.FP8_PARAMETERS = enabled
        yield
    finally:
        FP8GlobalStateManager.FP8_PARAMETERS = _fp8_parameters # pylint: disable=used-before-assignment


@contextmanager
def fp8_autocast(
    enabled: bool = True,
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
    enabled: bool, default = `True`
             whether or not to enable fp8
    calibrating: bool, default = `False`
                 calibration mode allows collecting statistics such as amax and scale
                 data of fp8 tensors even when executing without fp8 enabled. This is
                 useful for saving an inference ready fp8 checkpoint while training
                 using a higher precision.
    fp8_recipe: recipe.DelayedScaling, default = `None`
                recipe used for FP8 training.
    fp8_group: torch._C._distributed_c10d.ProcessGroup, default = `None`
               distributed group over which amaxes for the fp8 tensors
               are reduced at the end of each training step.
    """
    try:
        fp8_state = FP8GlobalStateManager.get_fp8_autocast_state()
        FP8GlobalStateManager.fp8_autocast_enter(enabled=enabled,
                                                 calibrating=calibrating,
                                                 fp8_recipe=fp8_recipe,
                                                 fp8_group=fp8_group)
        yield
    finally:
        FP8GlobalStateManager.set_fp8_autocast_state(fp8_state) # pylint: disable=used-before-assignment
        FP8GlobalStateManager.fp8_autocast_exit()


def _update_amax_history(amax_history: torch.Tensor) -> torch.Tensor:
    """Update amax history and set next amax to zero."""
    if amax_history.shape[0] > 1:
        amax_history = torch.roll(amax_history, -1, 0)
    amax_history[0].fill_(0.0)
    return amax_history


@torch.jit.script
def _default_get_amax(
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
) -> torch.Tensor:
    """Default function to convert amax to scaling factor."""
    sf = (fp8_max / amax) / (2 ** margin)
    sf = torch.where(amax > 0.0, sf, scale)
    sf = torch.where(torch.isfinite(amax), sf, scale)
    return sf


@jit_fuser
def _compute_scaling_factor_inverse(
    scale: torch.Tensor,
    scale_inv: torch.Tensor,
    non_weight_mask: torch.Tensor,
    update_weight_scale_inv: bool,
) -> torch.Tensor:
    """Compute inverse of scaling factor."""
    if update_weight_scale_inv:
        return 1.0 / scale
    return torch.where(non_weight_mask, 1.0 / scale, scale_inv)


def _fused_amax_and_scale_update(
    amax_history: torch.Tensor,
    scale: torch.Tensor,
    scale_inv: torch.Tensor,
    fp8_dtype: tex.DType,
    margin: int,
    amax_compute_algo: str,
    non_weight_mask: torch.Tensor,
    update_weight_scale_inv: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Update amax history and FP8 scaling factors"""
    if update_weight_scale_inv:
        non_weight_mask = torch.Tensor()
    tex.fused_amax_and_scale_update(
        amax_history,
        scale,
        scale_inv,
        non_weight_mask,
        amax_history,
        scale,
        scale_inv,
        amax_compute_algo,
        fp8_dtype,
        margin,
    )
    return amax_history, scale, scale_inv


def _compute_amax(
    amax_history: torch.Tensor,
    recipe: DelayedScaling,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Obtain the amax from the history."""

    if callable(recipe.amax_compute_algo):
        amax = recipe.amax_compute_algo(amax_history)
        amax_history = _update_amax_history(amax_history)
        return amax_history, amax
    return _default_get_amax(
        amax_history,
        recipe.amax_compute_algo,
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
        ) = _fused_amax_and_scale_update(
            fp8_meta[fp8_meta_tensor_key].amax_history,
            fp8_meta[fp8_meta_tensor_key].scale,
            fp8_meta[fp8_meta_tensor_key].scale_inv,
            get_fp8_te_dtype(fp8_meta["recipe"], fwd_update),
            fp8_meta["recipe"].margin,
            fp8_meta["recipe"].amax_compute_algo,
            fp8_meta[fp8_meta_tensor_key + "_non_weight_mask"],
            update_weight_scale_inv,
        )
    else:
        fp8_meta[fp8_meta_tensor_key].amax_history, amax = _compute_amax(
            fp8_meta[fp8_meta_tensor_key].amax_history,
            fp8_meta["recipe"],
        )
        fp8_meta[fp8_meta_tensor_key].scale = _compute_scaling_factor(
            amax,
            fp8_meta[fp8_meta_tensor_key].scale,
            fp8_meta[fp8_max_key],
            fp8_meta["recipe"],
        )
        fp8_meta[fp8_meta_tensor_key].scale_inv = _compute_scaling_factor_inverse(
            fp8_meta[fp8_meta_tensor_key].scale,
            fp8_meta[fp8_meta_tensor_key].scale_inv,
            fp8_meta[fp8_meta_tensor_key + "_non_weight_mask"],
            update_weight_scale_inv,
        )
