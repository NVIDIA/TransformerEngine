# Copyright (c) 2025, BAAI. All rights reserved.
#
# See LICENSE for license information.

"""
KunLunXin backend operator registrations.

This module registers all KunLunXin PyTorch implementations.
"""

from __future__ import annotations

import functools

from transformer_engine.plugin.core.types import OpImpl, BackendImplKind


def _bind_is_available(fn, is_available_fn):
    """Wrap a function and bind _is_available attribute for OpImpl.is_available() check."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)
    wrapper._is_available = is_available_fn
    return wrapper


def register_builtins(registry) -> None:
    """
    Register all KunLunXin PyTorch operator implementations.

    Args:
        registry: Registry to register into
    """
    from .kunlunxin import KunLunXinBackend

    # Create a backend instance to access the methods
    backend = KunLunXinBackend()

    # Bind is_available to all methods
    is_avail = backend.is_available

    impls = [
        # FlashAttention class getter
        OpImpl(op_name="get_flash_attention_class", impl_id="vendor.kunlunxin", kind=BackendImplKind.VENDOR, fn=_bind_is_available(backend.get_flash_attention_class, is_avail), vendor="KUNLUNXIN", priority=100),

    ]

    registry.register_many(impls)
