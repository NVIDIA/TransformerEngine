# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Framework-independent cache-key policy for cuDNN score-modification graphs."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping
from typing import Any


class UncacheableScoreModKey:
    """Identity key for callbacks whose graph topology cannot be cached safely."""

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return self is other


UNCACHEABLE_SCORE_MOD = UncacheableScoreModKey()


def is_uncacheable_score_mod_key(key: Any) -> bool:
    """Return whether a score-modification graph key disables caching."""

    return isinstance(key, UncacheableScoreModKey)


def freeze_score_mod_cache_key(value: Any, *, is_array: Callable[[Any], bool]) -> Any:
    """Convert a user-provided score-modification key into a hashable structure."""

    if is_array(value):
        raise TypeError(
            "score_mod_graph_cache_key() must not include tensors. Pass runtime tensors "
            "through score_mod_tensors or score_mod_bprop_tensors instead."
        )
    if isinstance(value, Mapping):
        items = (
            (
                freeze_score_mod_cache_key(key, is_array=is_array),
                freeze_score_mod_cache_key(item, is_array=is_array),
            )
            for key, item in value.items()
        )
        return tuple(sorted(items, key=repr))
    if isinstance(value, (list, tuple)):
        return tuple(
            freeze_score_mod_cache_key(item, is_array=is_array) for item in value
        )
    if isinstance(value, (set, frozenset)):
        items = (freeze_score_mod_cache_key(item, is_array=is_array) for item in value)
        return tuple(sorted(items, key=repr))
    try:
        hash(value)
    except TypeError as exc:
        raise TypeError(
            "score_mod_graph_cache_key() must return a hashable value or a nested "
            "combination of mapping/list/tuple/set values."
        ) from exc
    return value


def _explicit_cache_key(
    callback_owner: Any, *, is_array: Callable[[Any], bool]
) -> Any | None:
    explicit_key = getattr(callback_owner, "score_mod_graph_cache_key", None)
    if explicit_key is None:
        return None
    explicit_key = explicit_key() if callable(explicit_key) else explicit_key
    return freeze_score_mod_cache_key(explicit_key, is_array=is_array)


def score_mod_callback_cache_key(
    callback: Callable | None,
    *,
    is_array: Callable[[Any], bool],
    uncacheable_key_factory: Callable[[], Any] | None = None,
) -> Any:
    """Create a stable graph key for a score-modification callable.

    Stateful callables must provide ``score_mod_graph_cache_key``. Stateless named
    functions use their qualified name, while lambdas use their code object so two
    lambdas in the same module cannot collide.
    """

    def uncacheable_key():
        if uncacheable_key_factory is None:
            return UNCACHEABLE_SCORE_MOD
        return uncacheable_key_factory()

    if callback is None:
        return None
    self_obj = getattr(callback, "__self__", None)
    func_obj = getattr(callback, "__func__", None)
    if self_obj is not None and func_obj is not None:
        explicit_key = _explicit_cache_key(self_obj, is_array=is_array)
        if explicit_key is None:
            return uncacheable_key()
        return (
            "bound_method",
            type(self_obj),
            func_obj.__module__,
            func_obj.__qualname__,
            explicit_key,
        )

    explicit_key = _explicit_cache_key(callback, is_array=is_array)
    if explicit_key is not None:
        return (
            "callable",
            type(callback),
            getattr(callback, "__module__", None),
            getattr(callback, "__qualname__", None),
            explicit_key,
        )

    if (
        inspect.isfunction(callback)
        and callback.__closure__ is None
        and "<locals>" not in callback.__qualname__
    ):
        if callback.__name__ == "<lambda>" or not callback.__qualname__:
            return ("function", callback.__module__, callback.__code__)
        return ("function", callback.__module__, callback.__qualname__)

    return uncacheable_key()
