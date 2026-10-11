# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Structural callback keys shared by score-modified attention frontends."""

from collections.abc import Mapping
import inspect


class UncacheableKey:
    """Unique static identity for callbacks whose topology may change."""


def freeze_cache_key(value, is_tensor):
    """Freeze nested structural keys while rejecting framework runtime tensors."""
    if is_tensor(value):
        raise TypeError(
            "score_mod_graph_cache_key() must not include tensors. Pass runtime tensors "
            "through score_mod_tensors or score_mod_bprop_tensors instead."
        )
    if isinstance(value, Mapping):
        return tuple(
            sorted(
                (
                    (freeze_cache_key(key, is_tensor), freeze_cache_key(val, is_tensor))
                    for key, val in value.items()
                ),
                key=repr,
            )
        )
    if isinstance(value, (list, tuple)):
        return tuple(freeze_cache_key(item, is_tensor) for item in value)
    if isinstance(value, (set, frozenset)):
        return tuple(sorted((freeze_cache_key(item, is_tensor) for item in value), key=repr))
    try:
        hash(value)
    except TypeError as exc:
        raise TypeError(
            "score_mod_graph_cache_key() must return a hashable value or a nested "
            "combination of mapping/list/tuple/set values."
        ) from exc
    return value


def explicit_cache_key(owner, is_tensor):
    """Read an optional structural key from a callback's owner."""
    key = getattr(owner, "score_mod_graph_cache_key", None)
    if key is None:
        return None
    return freeze_cache_key(key() if callable(key) else key, is_tensor)


def callback_cache_key(callback, is_tensor, uncacheable):
    """Cache stable functions and explicitly keyed owners; distinguish anonymous code."""
    if callback is None:
        return None
    owner = getattr(callback, "__self__", None)
    function = getattr(callback, "__func__", None)
    if owner is not None and function is not None:
        key = explicit_cache_key(owner, is_tensor)
        if key is None:
            return uncacheable()
        return ("bound_method", type(owner), function.__module__, function.__qualname__, key)
    key = explicit_cache_key(callback, is_tensor)
    if key is not None:
        return (
            "callable",
            type(callback),
            getattr(callback, "__module__", None),
            getattr(callback, "__qualname__", None),
            key,
        )
    if (
        inspect.isfunction(callback)
        and callback.__closure__ is None
        and "<locals>" not in callback.__qualname__
    ):
        identity = (
            callback.__code__
            if callback.__name__ == "<lambda>" or not callback.__qualname__
            else callback.__qualname__
        )
        return ("function", callback.__module__, identity)
    return uncacheable()


def wrap_score_mod(callback, graph_tensors):
    """Adapt TE's three-argument callback to cuDNN's two-argument callback."""
    if callback is None:
        return None

    def wrapped(graph, scores):
        return callback(graph, scores, graph_tensors)

    return wrapped
