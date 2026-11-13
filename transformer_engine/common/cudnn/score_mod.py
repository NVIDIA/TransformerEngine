# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Structural callback keys shared by score-modified attention frontends."""

from collections.abc import Mapping
import inspect
import struct


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


def _freeze_function_default(value, is_tensor):
    """Snapshot builtin defaults without trusting opaque objects' identity hashes."""
    value_type = type(value)
    if is_tensor(value):
        raise TypeError("Runtime tensors cannot be implicit callback cache keys.")
    if value_type in (type(None), bool, int, str, bytes):
        return value_type, value
    if value_type is float:
        return value_type, struct.pack("!d", value)
    if value_type is complex:
        return value_type, struct.pack("!dd", value.real, value.imag)
    if value_type is dict:
        return value_type, tuple(
            (_freeze_function_default(key, is_tensor), _freeze_function_default(item, is_tensor))
            for key, item in value.items()
        )
    if value_type in (list, tuple, set, frozenset):
        return value_type, tuple(_freeze_function_default(item, is_tensor) for item in value)
    raise TypeError("Opaque callback defaults require an explicit structural cache key.")


def callback_cache_key(callback, is_tensor, uncacheable):
    """Cache stateless functions by code/defaults and explicitly keyed callback owners."""
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
        try:
            defaults = _freeze_function_default(callback.__defaults__, is_tensor)
            kwdefaults = _freeze_function_default(callback.__kwdefaults__, is_tensor)
        except (TypeError, RecursionError):
            return uncacheable()
        return (
            "function",
            callback.__module__,
            callback.__qualname__,
            callback.__code__,
            defaults,
            kwdefaults,
        )
    return uncacheable()


def wrap_score_mod(callback, graph_tensors):
    """Adapt TE's three-argument callback to cuDNN's two-argument callback."""
    if callback is None:
        return None

    def wrapped(graph, scores):
        return callback(graph, scores, graph_tensors)

    return wrapped
