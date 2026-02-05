# Copyright (c) 2025, BAAI. All rights reserved.
#
# See LICENSE for license information.

"""
Built-in operator implementations registration.

This module registers DEFAULT (FlagOS) and REFERENCE (PyTorch) implementations
for all supported operators by calling register_builtins from each backend.
"""

from __future__ import annotations

from .registry import OpRegistry


def register_builtins(registry: OpRegistry) -> None:
    """
    Register all built-in operator implementations.

    This function registers:
    - DEFAULT implementations (FlagOS/flag_gems)
    - REFERENCE implementations (PyTorch)
    - VENDOR implementations (CUDA, if available)

    Args:
        registry: Registry to register into
    """
    # Register FlagOS (DEFAULT) implementations
    try:
        from .backends.flagos.register_ops import register_builtins as register_flagos
        register_flagos(registry)
    except Exception as e:
        print(f"[WARNING] Failed to register FlagOS operators: {e}")
    
    # Register PyTorch (REFERENCE) implementations
    try:
        from .backends.reference.register_ops import register_builtins as register_reference
        register_reference(registry)
    except Exception as e:
        print(f"[WARNING] Failed to register Reference operators: {e}")
    
    # Register CUDA (VENDOR) implementations
    try:
        from .backends.vendor.cuda.register_ops import register_builtins as register_cuda
        register_cuda(registry)
    except Exception as e:
        # CUDA may not be available, this is expected
        pass

    # Register HYGON (VENDOR) implementations
    try:
        from .backends.vendor.hygon.register_ops import register_builtins as register_hygon
        register_hygon(registry)
    except Exception as e:
        # HYGON may not be available, this is expected
        pass

    # Register Metax (VENDOR) implementations
    try:
        from .backends.vendor.metax.register_ops import register_builtins as register_metax
        register_metax(registry)
    except Exception as e:
        # Metax may not be available, this is expected
        pass

    # Register KUNLUNXIN (VENDOR) implementations
    try:
        from .backends.vendor.kunlunxin.register_ops import register_builtins as register_kunlunxin
        register_kunlunxin(registry)
    except Exception as e:
        # KunLunXin may not be available, this is expected
        pass
    
    # Register Iluvatar (VENDOR) implementations
    try:
        from .backends.vendor.iluvatar.register_ops import register_builtins as register_iluvatar
        register_iluvatar(registry)
    except Exception as e:
        # Iluvatar may not be available, this is expected
        pass