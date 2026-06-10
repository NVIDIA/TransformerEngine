"""Python-side compatibility patches for the MUSA vendor backend."""

from __future__ import annotations

from collections.abc import Callable

import torch


def _noop(*args, **kwargs):
    return None


# Patches: (parent_object, attribute_name, replacement_callable)
_PATCH_CALLS: list[tuple[object, str, Callable[..., object]]] = [
    # We do not recommend replace is_available, due to its device-related behavior.
    # (torch.cuda, "is_available", torch.musa.is_available),
    (torch.cuda, "get_device_properties", torch.musa.get_device_properties),
    (torch.cuda, "device", torch.musa.device),
    (torch.cuda, "current_device", torch.musa.current_device),
    (torch.cuda, "synchronize", torch.musa.synchronize),
    (torch.cuda, "is_current_stream_capturing", torch.musa.is_current_stream_capturing),
    # TODO: Add NVTX patches for MUSA.
    # NVTX is CUDA-specific; make it a no-op on MUSA.
    (torch.cuda.nvtx, "range_push", _noop),
    (torch.cuda.nvtx, "range_pop", _noop),
    # TODO: Add other patches for MUSA.
]


def apply_patch() -> None:
    """Apply MUSA Python-side patches (idempotent, best-effort)."""
    try:
        from .musa import MUSABackend

        if not MUSABackend().is_available():
            return
    except Exception as e:
        print(f"[TE-FL] MUSA backend not available: {e}")
        # If backend availability can't be determined, don't patch.
        return

    # Mark TE global device type for Python-side callers.
    # IMPORTANT: do not import `transformer_engine` here, because TE's `__init__.py`
    # imports this module to run patches and that would cause a circular import.
    try:
        import transformer_engine

        transformer_engine.TE_DEVICE_TYPE = "musa"
        transformer_engine.TE_PLATFORM = torch.musa
    except Exception as e:
        print(f"[TE-FL Musa Patches] Error setting TE device type or platform: {e}")
        # Best-effort: don't fail patching if we can't set the global.
        pass

    # Only patch when torch.musa exists and is usable.
    if not hasattr(torch, "musa"):
        return
    try:
        if not torch.musa.is_available():
            return
    except Exception:
        return

    for parent, attr, replacement in _PATCH_CALLS:
        if not hasattr(parent, attr):
            continue
        try:
            setattr(parent, attr, replacement)
        except Exception:
            # Best-effort: patching should never crash import/initialization.
            continue
    print(f"[TE-FL] MUSA backend patches applied")
