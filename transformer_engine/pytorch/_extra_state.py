# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Helpers for Transformer Engine PyTorch extra-state checkpoint handling."""

from __future__ import annotations

from enum import Enum
import os
import pickletools
from typing import Optional

from ..common.recipe import Recipe


UNSAFE_PICKLE_EXTRA_STATE_ENV = "NVTE_ALLOW_UNSAFE_PICKLE_EXTRA_STATE"

_RECIPE_MODULE = "transformer_engine.common.recipe"
_RECIPE_KEY = "recipe"

# Pickle keys that FP8 delayed scaling stores in its ``_extra_state``. This is
# purely a backward-compatibility hack for that recipe; any future stateful
# recipe is expected to checkpoint without pickling.
_FLOAT8_DELAYED_SCALING_STATE_KEYS = {
    "scale_fwd",
    "amax_history_fwd",
    "scale_bwd",
    "amax_history_bwd",
}


class CheckpointExtraStatePolicy(Enum):
    """How pickled PyTorch ``_extra_state`` should be handled in ``set_extra_state``.

    Pickling of ``_extra_state`` is a PyTorch-specific backward-compatibility
    concern, so the recipe-to-policy map lives here in ``te.pytorch`` rather than
    in ``te.common``.

    ``STATELESS`` recipes carry no checkpoint state and never need unpickling.
    ``STATEFUL_FP8_DELAYED_SCALING`` is the only recipe that still relies on
    unsafe pickling (a legacy of FP8 delayed scaling). ``DYNAMIC`` means the
    recipe class alone is not enough; callers must inspect the checkpoint payload
    shape before deciding whether the pickle can be ignored.
    """

    STATELESS = "stateless"
    STATEFUL_FP8_DELAYED_SCALING = "stateful_fp8_delayed_scaling"
    DYNAMIC = "dynamic"


# Map of first-party recipes to their checkpoint policy. When a new stateful
# recipe is added, update this map (and any associated checkpoint handling)
# here instead of adding PyTorch-specific logic to ``te.common``.
_RECIPE_POLICIES: dict[tuple[str, str], CheckpointExtraStatePolicy] = {
    (_RECIPE_MODULE, "DelayedScaling"): CheckpointExtraStatePolicy.STATEFUL_FP8_DELAYED_SCALING,
    (_RECIPE_MODULE, "Float8CurrentScaling"): CheckpointExtraStatePolicy.STATELESS,
    (_RECIPE_MODULE, "MXFP8BlockScaling"): CheckpointExtraStatePolicy.STATELESS,
    (_RECIPE_MODULE, "Float8BlockScaling"): CheckpointExtraStatePolicy.STATELESS,
    (_RECIPE_MODULE, "NVFP4BlockScaling"): CheckpointExtraStatePolicy.STATELESS,
    (_RECIPE_MODULE, "CustomRecipe"): CheckpointExtraStatePolicy.DYNAMIC,
}


def recipe_extra_state_policy(recipe: Recipe) -> Optional[CheckpointExtraStatePolicy]:
    """Return the checkpoint policy for a recipe instance, if known."""
    cls = type(recipe)
    return _RECIPE_POLICIES.get((cls.__module__, cls.__name__))


def is_stateless_recipe(recipe: Recipe) -> bool:
    """Return whether a recipe carries no extra state to checkpoint."""
    return recipe_extra_state_policy(recipe) is CheckpointExtraStatePolicy.STATELESS


class _PickledExtraStateAction(Enum):
    """Action to take for a pickled extra-state payload."""

    IGNORE = "ignore"
    UNSAFE_LOAD = "unsafe_load"


def unsafe_pickle_extra_state_enabled() -> bool:
    """Return whether unsafe extra-state pickle loading is enabled."""
    return os.getenv(UNSAFE_PICKLE_EXTRA_STATE_ENV, "0") == "1"


def extra_state_pickle_advisory(context: str) -> str:
    """Security advisory for pickled extra state."""
    return (
        f"Refusing to load pickled Transformer Engine extra state for {context}. "
        "Delayed-scaling FP8 metadata can be stored as a Python pickle, and loading it "
        "can execute arbitrary code. Only enable unsafe loading if this checkpoint is from "
        f"a trusted source. To load it anyway, set {UNSAFE_PICKLE_EXTRA_STATE_ENV}=1."
    )


def should_load_extra_state_pickle(data: bytes, context: str) -> bool:
    """Return whether callers should use the unsafe pickle loader.

    ``False`` means the payload was identified as empty/stateless and should be
    ignored. ``True`` means the caller may unpickle because the unsafe opt-in is
    enabled. Otherwise this raises with the security advisory.
    """
    action = _classify_extra_state_pickle(data)
    if action is _PickledExtraStateAction.IGNORE:
        return False
    if unsafe_pickle_extra_state_enabled():
        return True
    raise RuntimeError(extra_state_pickle_advisory(context))


def _classify_extra_state_pickle(data: bytes) -> _PickledExtraStateAction:
    """Classify a pickled extra-state payload without executing it."""
    if not data:
        return _PickledExtraStateAction.IGNORE

    try:
        return _classify_extra_state_pickle_impl(data)
    except Exception:  # pylint: disable=broad-except
        return _PickledExtraStateAction.UNSAFE_LOAD


def _classify_extra_state_pickle_impl(data: bytes) -> _PickledExtraStateAction:
    strings: list[str] = []
    has_recipe_key = False
    has_delayed_state_keys = False
    has_global = False
    policies: set[CheckpointExtraStatePolicy] = set()

    for opcode, arg, _pos in pickletools.genops(data):
        if opcode.name in {
            "STRING",
            "BINSTRING",
            "SHORT_BINSTRING",
            "UNICODE",
            "BINUNICODE",
            "BINUNICODE8",
            "SHORT_BINUNICODE",
        }:
            text = _string_opcode_arg(arg)
            if text is not None:
                strings.append(text)
                has_recipe_key = has_recipe_key or text == _RECIPE_KEY
                has_delayed_state_keys = (
                    has_delayed_state_keys or text in _FLOAT8_DELAYED_SCALING_STATE_KEYS
                )
            continue

        if opcode.name == "GLOBAL":
            has_global = True
            global_ref = _global_opcode_arg(arg)
        elif opcode.name == "STACK_GLOBAL":
            has_global = True
            global_ref = _stack_global_args(strings)
        else:
            continue

        if global_ref is None:
            continue
        policy = _RECIPE_POLICIES.get(global_ref)
        if policy is not None:
            policies.add(policy)

    # A payload that never resolves a global cannot construct an arbitrary
    # callable, so unpickling it cannot execute code (e.g. the empty dict that
    # older stateless checkpoints serialized). It carries no state worth
    # loading, so treat it as safe to ignore. A genuine TE 1.x delayed-scaling
    # checkpoint always serializes torch tensors and thus contains globals.
    if not has_global and not has_delayed_state_keys:
        return _PickledExtraStateAction.IGNORE

    # TE 1.x checkpoints did not store a recipe and only supported delayed scaling.
    if not has_recipe_key:
        return _PickledExtraStateAction.UNSAFE_LOAD

    if CheckpointExtraStatePolicy.STATEFUL_FP8_DELAYED_SCALING in policies:
        return _PickledExtraStateAction.UNSAFE_LOAD

    if has_delayed_state_keys:
        return _PickledExtraStateAction.UNSAFE_LOAD

    # Unknown/newer payload shape. Give trusted users the explicit opt-in path.
    if not policies:
        return _PickledExtraStateAction.UNSAFE_LOAD

    return _PickledExtraStateAction.IGNORE


def _string_opcode_arg(arg: object) -> Optional[str]:
    if isinstance(arg, str):
        return arg
    if isinstance(arg, bytes):
        try:
            return arg.decode("utf-8")
        except UnicodeDecodeError:
            return None
    return None


def _global_opcode_arg(arg: object) -> Optional[tuple[str, str]]:
    if not isinstance(arg, str):
        return None
    parts = arg.split()
    if len(parts) != 2:
        return None
    return parts[0], parts[1]


def _stack_global_args(strings: list[str]) -> Optional[tuple[str, str]]:
    if len(strings) < 2:
        return None
    return strings[-2], strings[-1]
