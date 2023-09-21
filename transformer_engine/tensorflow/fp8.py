# Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""FP8 utilies for TransformerEngine"""
from contextlib import contextmanager
from typing import Generator, Optional, Dict, Any

import tensorflow as tf
import transformer_engine_tensorflow as tex

from transformer_engine.common.recipe import DelayedScaling, Format

_FP8_ENABLED = False
_FP8_RECIPE = None
_FP8_DISTRIBUTED_GROUP = None
_IS_FIRST_FP8_MODULE = False
_FP8_AUTOCAST_COUNTER = 0
_FP8_CURRENT_CONTEXT_ID = 0
_FP8_AUTOCAST_DEPTH = 0
_global_fp8_buffer = {}
_amax_forward_global_reduce_func = lambda: None
_buffer_delete_key_fwd = None
_buffer_delete_key_bwd = None


def get_meta_tensor_key(forward: bool = True) -> str:
    """Returns scaling key in `fp8_meta`."""
    if forward:
        return "scaling_fwd"
    return "scaling_bwd"


def get_autocast_key(forward: bool = True) -> str:
    """Returns module position key in `fp8_meta`."""
    if forward:
        return "autocast_id_fwd"
    return "autocast_id_bwd"


def get_amax_buffer_key(fp8_meta: Dict[str, Any], forward: bool = True) -> str:
    """Return a key in `_global_fp8_buffer` for the AMAX storage."""
    if forward:
        return f"FWD_AMAX_{fp8_meta['autocast_id_fwd']}"
    return f"BWD_AMAX_{fp8_meta['autocast_id_bwd']}"


def set_amax_buffer_key_deletion(
    fp8_meta: Dict[str, Any], forward: bool = True
) -> None:
    """Delete this amax key from global buffer during autocast end."""
    if get_autocast_key(forward=forward) not in fp8_meta:
        return
    global _buffer_delete_key_fwd, _buffer_delete_key_bwd
    if forward:
        _buffer_delete_key_fwd = get_amax_buffer_key(fp8_meta, forward=forward)
    else:
        _buffer_delete_key_bwd = get_amax_buffer_key(fp8_meta, forward=forward)


def get_default_fp8_recipe():
    """FP8 recipe if not provided by user
    Margin = 0, interval = 1, E4M3
    """
    return DelayedScaling()


@contextmanager
def fp8_autocast(
    enabled: bool = False,
    fp8_recipe: Optional[DelayedScaling] = None,
) -> Generator[None, None, None]:
    """
    Context manager for FP8 usage.

    .. code-block:: python

        with fp8_autocast(enabled=True):
            out = model(inp)

    .. note::

        Support for FP8 in the Dense layer of Transformer Engine is currently
        limited to tensors with shapes where both dimensions are divisible by 16.
        In terms of the input to the full Transformer network, this typically
        requires padding sequence length to be multiple of 16.

    Parameters
    ----------
    enabled: bool, default = `False`
             whether or not to enable fp8
    fp8_recipe: recipe.DelayedScaling, default = `None`
                recipe used for FP8 training.
    """
    global _FP8_ENABLED, _FP8_RECIPE, _FP8_DISTRIBUTED_GROUP, _FP8_AUTOCAST_DEPTH
    global _IS_FIRST_FP8_MODULE, _FP8_AUTOCAST_COUNTER
    global _global_fp8_buffer, _buffer_delete_key_fwd
    fp8_state = (_FP8_ENABLED, _FP8_RECIPE, _FP8_DISTRIBUTED_GROUP)
    try:
        _FP8_ENABLED = enabled
        _FP8_RECIPE = get_default_fp8_recipe() if fp8_recipe is None else fp8_recipe

        if _FP8_AUTOCAST_DEPTH == 0:
            _IS_FIRST_FP8_MODULE = True
            _FP8_AUTOCAST_COUNTER += 1
        _FP8_AUTOCAST_DEPTH += 1

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


def is_fp8_enabled():
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


def get_fp8_recipe():
    """Return the fp8 recipe"""
    return _FP8_RECIPE


def _default_sf_compute(amax, scale, fp8_max, margin):
    """Default function to convert amax to scaling factor."""
    sf = (fp8_max / amax) / (2 ** margin)
    sf = tf.where(amax > 0.0, sf, scale)
    sf = tf.where(tf.math.is_finite(amax), sf, scale)
    return sf


def _roll_and_zero_out(amax_history):
    """Update amax history and set next amax to zero."""
    amax_history = tf.roll(amax_history, -1, 0)
    zeros = tf.zeros(shape=amax_history[0].shape)
    updated = tf.tensor_scatter_nd_update(amax_history, [[0]], [zeros])
    return updated


@tf.function(jit_compile=True)
def _reduce_max_and_default_sf_compute(amax_history, scale, fp8_max, margin):
    """Get amax using max algorithm and compute scaling factor."""
    amax = tf.reduce_max(amax_history, axis=0)
    sf = _default_sf_compute(amax, scale, fp8_max, margin)
    updated = _roll_and_zero_out(amax_history)
    return updated, sf


@tf.function(jit_compile=True)
def _most_recent_and_default_sf_compute(amax_history, scale, fp8_max, margin):
    """Get amax using most-recent algorithm and compute scaling factor."""
    amax = amax_history[0]
    sf = _default_sf_compute(amax, scale, fp8_max, margin)
    updated = _roll_and_zero_out(amax_history)
    return updated, sf


def fused_amax_and_scale_update(
    amax_history: tf.Variable,
    scale: tf.Variable,
    scale_inv: tf.Variable,
    fp8_max: float,
    margin: int,
    amax_compute_algo: str,
):
    """Amax to scale conversion."""

    if amax_compute_algo == "max":
        updated, sf = _reduce_max_and_default_sf_compute(
            amax_history, scale, fp8_max, margin
        )
    else:
        assert amax_compute_algo == "most_recent"
        updated, sf = _most_recent_and_default_sf_compute(
            amax_history, scale, fp8_max, margin
        )
    amax_history.assign(updated)
    scale.assign(sf)
    scale_inv.assign(1.0 / sf)


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
        fused_amax_and_scale_update(
            fp8_meta[fp8_meta_tensor_key]["amax_history"],
            fp8_meta[fp8_meta_tensor_key]["scale"],
            fp8_meta[fp8_meta_tensor_key]["scale_inv"],
            fp8_meta[fp8_max_key],
            fp8_meta["recipe"].margin,
            fp8_meta["recipe"].amax_compute_algo,
        )
    else:
        raise ValueError(
            "We only support the fp8 recipe with 'max' or 'most_recent' "
            "amax_compute_algo and default scaling_factor_compute_algo at this "
            "moment."
        )


def get_fp8_te_dtype(fp8_recipe: DelayedScaling, fprop_tensor: bool = True):
    """Get fp8 data type according to recipe and tensor"""
    if fp8_recipe.fp8_format == Format.E4M3 or (
        fp8_recipe.fp8_format == Format.HYBRID and fprop_tensor
    ):
        return tex.DType.kFloat8E4M3
    return tex.DType.kFloat8E5M2


def delete_key_from_amax_buffer(forward: bool = True) -> None:
    """Delete the key from global amax buffer."""
    global _global_fp8_buffer, _buffer_delete_key_fwd, _buffer_delete_key_bwd
    if forward:
        if (
            _buffer_delete_key_fwd is not None
            and _buffer_delete_key_fwd in _global_fp8_buffer
        ):
            del _global_fp8_buffer[_buffer_delete_key_fwd]
    else:
        if (
            _buffer_delete_key_bwd is not None
            and _buffer_delete_key_bwd in _global_fp8_buffer
        ):
            del _global_fp8_buffer[_buffer_delete_key_bwd]
