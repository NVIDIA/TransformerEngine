# Copyright (c) 2025, BAAI. All rights reserved.
#
# See LICENSE for license information.

"""
FlagOS backend operator registrations.

This module registers all DEFAULT (FlagOS) implementations.
"""

from __future__ import annotations

import functools

from ...types import OpImpl, BackendImplKind


def _bind_is_available(fn, is_available_fn):
    """Wrap a function and bind _is_available attribute for OpImpl.is_available() check."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)
    wrapper._is_available = is_available_fn
    return wrapper


def register_builtins(registry) -> None:
    """
    Register all FlagOS (DEFAULT) operator implementations.

    Args:
        registry: Registry to register into
    """
    from .flagos import FlagOSBackend

    # Create a backend instance to access the methods
    backend = FlagOSBackend()

    # Bind is_available to all methods
    is_avail = backend.is_available

    impls = [
        OpImpl(op_name="rmsnorm_fwd", impl_id="default.flagos", kind=BackendImplKind.DEFAULT, fn=_bind_is_available(backend.rmsnorm_fwd, is_avail), vendor=None, priority=150),
        OpImpl(op_name="rmsnorm_bwd", impl_id="default.flagos", kind=BackendImplKind.DEFAULT, fn=_bind_is_available(backend.rmsnorm_bwd, is_avail), vendor=None, priority=150),
        OpImpl(op_name="generic_gemm", impl_id="default.flagos", kind=BackendImplKind.DEFAULT, fn=_bind_is_available(backend.generic_gemm, is_avail), vendor=None, priority=150),
        OpImpl(op_name="multi_tensor_scale", impl_id="default.flagos", kind=BackendImplKind.DEFAULT, fn=_bind_is_available(backend.multi_tensor_scale, is_avail), vendor=None, priority=150),
        OpImpl(op_name="multi_tensor_adam", impl_id="default.flagos", kind=BackendImplKind.DEFAULT, fn=_bind_is_available(backend.multi_tensor_adam, is_avail), vendor=None, priority=150),
        OpImpl(op_name="multi_tensor_l2norm", impl_id="default.flagos", kind=BackendImplKind.DEFAULT, fn=_bind_is_available(backend.multi_tensor_l2norm, is_avail), vendor=None, priority=150),

        # FlashAttention class getter
        OpImpl(op_name="get_flash_attention_class", impl_id="default.flagos", kind=BackendImplKind.DEFAULT, fn=_bind_is_available(backend.get_flash_attention_class, is_avail), vendor=None, priority=150),
    ]

    registry.register_many(impls)
