# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""FP8 utilies for TransformerEngine"""
from contextlib import contextmanager
from collections import deque
from typing import Callable, List, Optional, Dict, Any, Tuple, Union, Deque

import torch
import transformer_engine_extensions as tex
from transformer_engine.common.recipe import DelayedScaling, Format

from .constants import dist_group_type

_FP8_ENABLED = False
_FP8_RECIPE = None
_FP8_DISTRIBUTED_GROUP = None
_IS_FIRST_FP8_MODULE = False
_FP8_AUTOCAST_COUNTER = 0
_FP8_CURRENT_CONTEXT_ID = 0
_FP8_AUTOCAST_DEPTH = 0
_global_fp8_buffer = {}
_fp8_tensors_recompute_buffer = []
_amax_forward_global_reduce_func = None
_buffer_delete_key_fwd = None
_buffer_delete_key_bwd = None


def get_meta_tensor_key(forward: bool = True) -> str:
    """Returns scaling key in `fp8_meta`."""
    if forward:
        return "scaling_fwd"
    return "scaling_bwd"


def get_buffer_position_key(forward: bool = True) -> str:
    """Returns module position key in `fp8_meta`."""
    if forward:
        return "global_fp8_buffer_pos_fwd"
    return "global_fp8_buffer_pos_bwd"


def get_autocast_key(forward: bool = True) -> str:
    """Returns module position key in `fp8_meta`."""
    if forward:
        return "autocast_id_fwd"
    return "autocast_id_bwd"


def get_global_fp8_buffer() -> Dict[str, List[torch.Tensor]]:
    """Returns global fp8 buffer."""
    return _global_fp8_buffer


def set_global_fp8_buffer(buffer: Dict[str, List[torch.Tensor]]) -> None:
    """Sets global fp8 buffer."""
    global _global_fp8_buffer

    # Map all tensors back to GPU.
    for k, v in buffer.items():
        buffer[k] = [tensor.cuda() for tensor in v]

    _global_fp8_buffer = buffer


def get_global_fp8_recompute_buffer() -> Dict[str, List[torch.Tensor]]:
    """Returns global fp8 recompute buffer."""
    return _fp8_tensors_recompute_buffer


def set_global_fp8_recompute_buffer(buffer: List[Deque[List[torch.Tensor]]]) -> None:
    """Sets global fp8 recompute buffer."""
    global _fp8_tensors_recompute_buffer

    # Map all tensors back to GPU.
    for index, deck in enumerate(buffer):
        buffer[index] = deque([[t.cuda() for t in tensors] for tensors in deck])

    _fp8_tensors_recompute_buffer = buffer


def setup_amax_forward_global_reduce_func(f: Callable) -> None:
    """Sets up the function to call during autocast exit."""
    global _amax_forward_global_reduce_func
    _amax_forward_global_reduce_func = f


def get_amax_buffer_key(fp8_meta: Dict[str, Any], forward: bool = True) -> str:
    """Return a key in `_global_fp8_buffer` for the AMAX storage."""
    if forward:
        return f"FWD_AMAX_{fp8_meta['autocast_id_fwd']}"
    return f"BWD_AMAX_{fp8_meta['autocast_id_bwd']}"


def add_amax_to_global_buffer(fp8_meta: Dict[str, Any], forward: bool = True) -> None:
    """Append 1D tensor `amax` to global buffer."""
    global _global_fp8_buffer
    buffer_key = get_amax_buffer_key(fp8_meta, forward=forward)
    fp8_meta_tensor_key = get_meta_tensor_key(forward=forward)
    buffer_position_key = get_buffer_position_key(forward=forward)

    if buffer_key not in _global_fp8_buffer:
        _global_fp8_buffer[buffer_key] = [fp8_meta[fp8_meta_tensor_key].amax_history[0]]
    else:
        _global_fp8_buffer[buffer_key].append(fp8_meta[fp8_meta_tensor_key].amax_history[0])

    if buffer_position_key not in fp8_meta:
        fp8_meta[buffer_position_key] = len(_global_fp8_buffer[buffer_key]) - 1


def copy_forward_fp8_meta_tensors_for_recompute(fp8_meta: Dict[str, Any]) -> None:
    """Copy the scaling factors and amaxes for recompute forward phase
    to ensure both forward steps are numerically same.
    """
    global _fp8_tensors_recompute_buffer
    buffer_position_key = "global_fp8_buffer_pos_fwd_recompute"

    to_copy = [
        fp8_meta["scaling_fwd"].amax_history.clone(),
        fp8_meta["scaling_fwd"].scale.clone(),
        fp8_meta["scaling_fwd"].scale_inv.clone(),
    ]

    if buffer_position_key in fp8_meta:
        _fp8_tensors_recompute_buffer[fp8_meta[buffer_position_key]].append(to_copy)
    else:
        if len(_fp8_tensors_recompute_buffer) == 0:
            _fp8_tensors_recompute_buffer = [deque()]
        else:
            _fp8_tensors_recompute_buffer.append(deque())
        _fp8_tensors_recompute_buffer[-1].append(to_copy)
        fp8_meta[buffer_position_key] = len(_fp8_tensors_recompute_buffer) - 1


def get_old_fp8_meta_tensors_for_recompute(fp8_meta: Dict[str, Any]) -> None:
    """Switch to the copied scaling factors and amaxes from phase
    1 forward for indentical numerical outputs.
    """

    # Store updated amaxes and scales from phase 1 post forward.
    fp8_meta["updated_amax_history_fwd"] = fp8_meta["scaling_fwd"].amax_history
    fp8_meta["updated_scale_fwd"] = fp8_meta["scaling_fwd"].scale
    fp8_meta["updated_scale_inv_fwd"] = fp8_meta["scaling_fwd"].scale_inv

    # Retrieve stashed amaxes and scales from phase 1 pre forward.
    buffer_position_key = "global_fp8_buffer_pos_fwd_recompute"
    stashed_fp8_meta = _fp8_tensors_recompute_buffer[fp8_meta[buffer_position_key]].popleft()

    # Replace amaxes and scales with stashed values for phase 2 forward
    fp8_meta["scaling_fwd"].amax_history = stashed_fp8_meta[0]
    fp8_meta["scaling_fwd"].scale = stashed_fp8_meta[1]
    fp8_meta["scaling_fwd"].scale_inv = stashed_fp8_meta[2]


def restore_fp8_meta_tensors(fp8_meta: Dict[str, Any]) -> None:
    """Restore latest scaling factors and amaxes after recompute forward run."""
    fp8_meta["scaling_fwd"].amax_history = fp8_meta["updated_amax_history_fwd"]
    fp8_meta["scaling_fwd"].scale = fp8_meta["updated_scale_fwd"]
    fp8_meta["scaling_fwd"].scale_inv = fp8_meta["updated_scale_inv_fwd"]


def copy_amax_from_global_buffer(fp8_meta: Dict[str, Any], forward: bool = True) -> None:
    """Populate current amax with the correct location from buffer."""
    fp8_meta_tensor_key = get_meta_tensor_key(forward=forward)
    buffer_position_key = get_buffer_position_key(forward=forward)
    if buffer_position_key not in fp8_meta:
        return
    amax_buffer_key = get_amax_buffer_key(fp8_meta, forward=forward)
    fp8_meta[fp8_meta_tensor_key].amax_history[0] = _global_fp8_buffer[amax_buffer_key][
        fp8_meta[buffer_position_key]
    ]


def set_amax_buffer_key_deletion(fp8_meta: Dict[str, Any], forward: bool = True) -> None:
    """Delete this amax key from global buffer during autocast end."""
    if get_autocast_key(forward=forward) not in fp8_meta:
        return
    global _buffer_delete_key_fwd, _buffer_delete_key_bwd
    if forward:
        _buffer_delete_key_fwd = get_amax_buffer_key(fp8_meta, forward=forward)
    else:
        _buffer_delete_key_bwd = get_amax_buffer_key(fp8_meta, forward=forward)


def get_default_fp8_recipe() -> DelayedScaling:
    """FP8 recipe if not provided by user
    Margin = 0, interval = 1, E4M3
    """
    return DelayedScaling()


@contextmanager
def fp8_autocast(
    enabled: bool = False,
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

    Parameters
    ----------
    enabled: bool, default = `False`
             whether or not to enable fp8
    fp8_recipe: recipe.DelayedScaling, default = `None`
                recipe used for FP8 training.
    fp8_group: torch._C._distributed_c10d.ProcessGroup, default = `None`
               distributed group over which amaxes for the fp8 tensors
               are reduced at the end of each training step.
    """

    global _FP8_ENABLED, _FP8_RECIPE, _FP8_DISTRIBUTED_GROUP, _FP8_AUTOCAST_DEPTH
    global _IS_FIRST_FP8_MODULE, _FP8_AUTOCAST_COUNTER
    global _global_fp8_buffer, _buffer_delete_key_fwd
    fp8_state = (_FP8_ENABLED, _FP8_RECIPE, _FP8_DISTRIBUTED_GROUP)
    try:
        _FP8_ENABLED = enabled
        _FP8_RECIPE = get_default_fp8_recipe() if fp8_recipe is None else fp8_recipe
        _FP8_DISTRIBUTED_GROUP = fp8_group

        if _FP8_AUTOCAST_DEPTH == 0:
            _IS_FIRST_FP8_MODULE = True
            _FP8_AUTOCAST_COUNTER += 1
        _FP8_AUTOCAST_DEPTH += 1

        if enabled:
            assert (
                torch.cuda.get_device_properties(torch.cuda.current_device()).major >= 9
            ), "Device compute capability 9.x required for FP8 execution."
        yield
    finally:
        _FP8_ENABLED, _FP8_RECIPE, _FP8_DISTRIBUTED_GROUP = fp8_state
        _IS_FIRST_FP8_MODULE = False
        _FP8_AUTOCAST_DEPTH -= 1

        if _FP8_AUTOCAST_DEPTH == 0:
            if callable(_amax_forward_global_reduce_func):
                _amax_forward_global_reduce_func()
            delete_key_from_amax_buffer(forward=True)


def get_fp8_context_id() -> int:
    """Returns an ID for the current FP8 context."""
    return _FP8_CURRENT_CONTEXT_ID


def set_fp8_context_id(ctx_id: int) -> None:
    """Sets the current FP8 context."""
    global _FP8_CURRENT_CONTEXT_ID
    _FP8_CURRENT_CONTEXT_ID = ctx_id


def new_fp8_context_id() -> int:
    """Returns global autocast counter as a proxy to be used
    as the autocast ID for FP8 modules.
    """
    return _FP8_AUTOCAST_COUNTER


def is_fp8_enabled() -> bool:
    """Is FP8 enabled"""
    return _FP8_ENABLED


def is_first_fp8_module():
    """Returns `True` only the first time when called multiple
    times from within the same `fp8_autocast` context.
    """
    global _IS_FIRST_FP8_MODULE
    tmp = _IS_FIRST_FP8_MODULE
    _IS_FIRST_FP8_MODULE = False
    return tmp


def get_fp8_recipe() -> DelayedScaling:
    """Return the fp8 recipe"""
    return _FP8_RECIPE


def get_fp8_group() -> Union[dist_group_type, None]:
    """Return the fp8 group for scale/amax comm"""
    return _FP8_DISTRIBUTED_GROUP


def update_amax_history(amax_history: torch.Tensor) -> torch.Tensor:
    """Update amax history and set next amax to zero."""
    amax_history = torch.roll(amax_history, -1, 0)
    amax_history[0].fill_(0.0)
    return amax_history


@torch.jit.script
def _default_get_amax(
    amax_history: torch.Tensor, amax_compute_algo: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Default function to obtain amax from history."""
    if amax_compute_algo == "max":
        amax = torch.max(amax_history, dim=0).values
    else:  # amax_compute_algo == "most_recent"
        amax = amax_history[0]

    amax_history = update_amax_history(amax_history)
    return amax_history, amax


@torch.jit.script
def _default_sf_compute(
    amax: torch.Tensor, scale: torch.Tensor, fp8_max: float, margin: int,
) -> torch.Tensor:
    """Default function to convert amax to scaling factor."""
    exp = torch.floor(torch.log2(fp8_max / amax)) - margin
    sf = torch.round(torch.pow(2, torch.abs(exp)))
    sf = torch.where(amax > 0.0, sf, scale)
    sf = torch.where(torch.isfinite(amax), sf, scale)
    sf = torch.where(exp < 0, 1 / sf, sf)

    return sf


@torch.jit.script
def fused_amax_and_scale_update(
    amax_history: torch.Tensor,
    scale: torch.Tensor,
    fp8_max: float,
    margin: int,
    amax_compute_algo: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Amax to scale conversion."""

    # Get amax from history.
    amax_history, amax = _default_get_amax(amax_history, amax_compute_algo,)

    # Calculate new scaling factor.
    return amax_history, _default_sf_compute(amax, scale, fp8_max, margin,)


def _compute_amax(
    amax_history: torch.Tensor, recipe: DelayedScaling,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Obtain the amax from the history."""

    if callable(recipe.amax_compute_algo):
        amax = recipe.amax_compute_algo(amax_history)
        amax_history = update_amax_history(amax_history)
        return amax_history, amax
    return _default_get_amax(amax_history, recipe.amax_compute_algo,)


def _compute_scaling_factor(
    amax: torch.Tensor, scale: torch.Tensor, fp8_max: float, recipe: DelayedScaling,
) -> torch.Tensor:
    """Convert amax to scaling factor."""

    if recipe.scaling_factor_compute_algo is None:
        return _default_sf_compute(amax, scale, fp8_max, recipe.margin,)
    return recipe.scaling_factor_compute_algo(amax, scale, fp8_max, recipe)


def amax_and_scale_update(fp8_meta: Dict[str, Any], fwd_update: bool,) -> None:
    """Updates fp8 amaxes/scales for fwd | bwd."""
    amax_compute = fp8_meta["recipe"].amax_compute_algo
    sf_compute = fp8_meta["recipe"].scaling_factor_compute_algo
    fp8_meta_tensor_key = "scaling_fwd" if fwd_update else "scaling_bwd"
    fp8_max_key = "fp8_max_fwd" if fwd_update else "fp8_max_bwd"

    if not callable(amax_compute) and sf_compute is None:
        (
            fp8_meta[fp8_meta_tensor_key].amax_history,
            fp8_meta[fp8_meta_tensor_key].scale,
        ) = fused_amax_and_scale_update(
            fp8_meta[fp8_meta_tensor_key].amax_history,
            fp8_meta[fp8_meta_tensor_key].scale,
            fp8_meta[fp8_max_key],
            fp8_meta["recipe"].margin,
            fp8_meta["recipe"].amax_compute_algo,
        )
    else:
        fp8_meta[fp8_meta_tensor_key].amax_history, amax = _compute_amax(
            fp8_meta[fp8_meta_tensor_key].amax_history, fp8_meta["recipe"],
        )
        fp8_meta[fp8_meta_tensor_key].scale = _compute_scaling_factor(
            amax, fp8_meta[fp8_meta_tensor_key].scale, fp8_meta[fp8_max_key], fp8_meta["recipe"],
        )


def get_fp8_te_dtype(fp8_recipe: DelayedScaling, fprop_tensor: bool = True) -> tex.DType:
    """Get fp8 data type according to recipe and tensor"""
    if fp8_recipe.fp8_format == Format.E4M3 or (
        fp8_recipe.fp8_format == Format.HYBRID and fprop_tensor
    ):
        return tex.DType.kFloat8E4M3
    return tex.DType.kFloat8E5M2


def reduce_tensor_across_group_op_max(tensor: torch.Tensor, group: dist_group_type) -> None:
    """Reduce tensor across given group."""
    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(
            tensor, op=torch.distributed.ReduceOp.MAX, group=group, async_op=False,
        )


def global_amax_reduction(
    fp8_meta: Dict[str, Any],
    reduce_amax_across_tp_group: bool = False,
    tp_group: Optional[dist_group_type] = None,
    forward: bool = True,
) -> None:
    """Concatenate, reduce, and split amaxes in the global buffer."""
    global _global_fp8_buffer
    amax_buffer_key = get_amax_buffer_key(fp8_meta, forward=forward)

    # Key already deleted.
    if amax_buffer_key not in _global_fp8_buffer:
        return

    chunk_sizes = [x.numel() for x in _global_fp8_buffer[amax_buffer_key]]
    contiguous_amax = torch.cat(_global_fp8_buffer[amax_buffer_key])

    reduce_tensor_across_group_op_max(contiguous_amax, fp8_meta["fp8_group"])
    if reduce_amax_across_tp_group:
        reduce_tensor_across_group_op_max(contiguous_amax, tp_group)

    _global_fp8_buffer[amax_buffer_key] = list(contiguous_amax.split(chunk_sizes))


def delete_key_from_amax_buffer(forward: bool = True) -> None:
    """Delete the key from global amax buffer."""

    global _global_fp8_buffer, _buffer_delete_key_fwd, _buffer_delete_key_bwd
    if forward:
        if _buffer_delete_key_fwd is not None and _buffer_delete_key_fwd in _global_fp8_buffer:
            del _global_fp8_buffer[_buffer_delete_key_fwd]
    else:
        if _buffer_delete_key_bwd is not None and _buffer_delete_key_bwd in _global_fp8_buffer:
            del _global_fp8_buffer[_buffer_delete_key_bwd]
