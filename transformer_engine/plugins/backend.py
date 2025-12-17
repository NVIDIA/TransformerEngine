# Copyright (c) 2025, BAAI. All rights reserved.
#
# See LICENSE for license information.

import os
import torch
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from .register import get_backend, get_selected_backend, register_backend
from .logger import get_logger
logger = get_logger()

from .import_utils import have_flag_gems

HAVE_FLAG_GEMS = have_flag_gems()

class BackendDispatch:
    """
    Transformer Engine Backend that routes operations to appropriate implementations.
    
    Uses caching to avoid repeated flag checks and backend lookups for the same operation.
    """
    
    def __init__(self):
        """Initialize the backend with an empty implementation cache."""
        # Cache for operation implementations: {operation: impl}
        self._impl_cache: Dict[str, Any] = {}
    
    def _get_impl(self, operation: str):
        """
        Get the implementation for an operation based on flags.
        Falls back to native if the selected backend doesn't have the operation.
        Uses caching to avoid repeated lookups.
        
        Args:
            operation: Name of the operation (e.g., "gemm", "rmsnorm_fwd")
        
        Returns:
            The implementation function/class to use
        
        Raises:
            RuntimeError: If native backend doesn't have the operation
        """
        # Check cache first
        if operation in self._impl_cache:
            return self._impl_cache[operation]
        
        # Get selected backend based on global environment variable
        selected_backend = get_selected_backend()
        native_backend = get_backend("native")
        
        # Try to get implementation from selected backend, fallback to native if not found
        impl = selected_backend.get(operation)
        if impl is None:
            logger.debug(
                f"Backend '{selected_backend.name}' doesn't have '{operation}', "
                f"falling back to native"
            )
            impl = native_backend.get(operation)
            if impl is None:
                raise RuntimeError(
                    f"Operation '{operation}' is not registered in native backend. "
                    f"Available operations: {sorted(native_backend._implementations.keys())}"
                )
        
        # Cache the implementation for future use
        logger.info(f"Backend '{selected_backend.name}' use implementation of '{operation}' for training")
        self._impl_cache[operation] = impl
        
        return impl
    
    def _reset_cache_to_native(self, operation: str):
        # Check cache first
        if operation in self._impl_cache:
            # Get native backend
            native_backend = get_backend("native")
            impl = native_backend.get(operation)
            if impl is None:
                raise RuntimeError(
                    f"Operation '{operation}' is not registered in native backend. "
                    f"Available operations: {sorted(native_backend._implementations.keys())}"
                )
            # Cache the implementation for future use
            self._impl_cache[operation] = impl

    def clear_cache(self):
        """Clear the implementation cache. Useful if flags change at runtime."""
        self._impl_cache.clear()
        logger.debug("Cleared implementation cache")

    def gemm(self, *args, **kwargs):
        """GEMM operation with automatic fallback to native."""
        impl = self._get_impl("gemm")
        try:
            return impl(*args, **kwargs)
        except Exception as e:
            logger.warning(f"GEMM implementation failed, falling back to native: {e}")
            self._reset_cache_to_native("gemm")
            native_backend = get_backend("native")
            return native_backend.get("gemm")(*args, **kwargs)
    
    def apply_normalization(self, *args, **kwargs):
        """Apply normalization with automatic fallback to native."""
        impl = self._get_impl("apply_normalization")
        try:
            return impl(*args, **kwargs)
        except Exception as e:
            logger.warning(f"Apply Normalization implementation failed, falling back to native: {e}")
            self._reset_cache_to_native("apply_normalization")
            native_backend = get_backend("native")
            return native_backend.get("apply_normalization")(*args, **kwargs)
    
    def rmsnorm_fwd(self, *args, **kwargs):
        """RMSNorm forward pass with automatic fallback to native."""
        impl = self._get_impl("rmsnorm_fwd")
        try:
            return impl(*args, **kwargs)
        except Exception as e:
            logger.warning(f"RmsNorm FWD implementation failed, falling back to native: {e}")
            self._reset_cache_to_native("rmsnorm_fwd")
            native_backend = get_backend("native")
            return native_backend.get("rmsnorm_fwd")(*args, **kwargs)
    
    def rmsnorm_bwd(self, *args, **kwargs):
        """RMSNorm backward pass with automatic fallback to native."""
        impl = self._get_impl("rmsnorm_bwd")
        try:
            return impl(*args, **kwargs)
        except Exception as e:
            logger.warning(f"RmsNorm BWD implementation failed, falling back to native: {e}")
            self._reset_cache_to_native("rmsnorm_bwd")
            native_backend = get_backend("native")
            trimmed_args = args[:-1]  # cut eps
            return native_backend.get("rmsnorm_bwd")(*trimmed_args, **kwargs)
    
    def multi_tensor_adam(self):
        """Multi-tensor Adam optimizer with automatic fallback to native."""
        impl = self._get_impl("adam")
        try:
            return impl
        except Exception as e:
            logger.warning(f"Adam implementation failed, falling back to native: {e}")
            self._reset_cache_to_native("adam")
            native_backend = get_backend("native")
            return native_backend.get("adam")
    
    def flash_attention(self, *args, **kwargs):
        """Flash Attention with automatic fallback to native."""
        flash_attention_instance = args[0]
        trimmed_args = args[1:]
        native_impl = get_backend("native").get("flash_attention")
        try:
            selected_impl = self._get_impl("flash_attention")
            flash_attention_instance.forward = selected_impl.forward.__get__(flash_attention_instance, native_impl)
            return flash_attention_instance(*trimmed_args, **kwargs)
        except Exception as e:
            logger.warning(f"Flash Attention Forward implementation failed, falling back to native: {e}")
            self._reset_cache_to_native("flash_attention")
            flash_attention_instance.forward = native_impl.forward.__get__(flash_attention_instance, native_impl)
            return flash_attention_instance(*trimmed_args, **kwargs)


# Backend initialization state
_backends_initialized = False
_backend_instance = None

def _initialize_backends():
    """
    Initialize all backend registrations.
    This function is called automatically on first use.
    """
    global _backends_initialized, _backend_instance
    
    if _backends_initialized:
        return
    
    from .backend_native import register_backend_native
    register_backend_native()
    if HAVE_FLAG_GEMS:
        from .backend_fl import register_backend_fl
        register_backend_fl()
    
    _backend_instance = BackendDispatch()
    _backends_initialized = True
    
    logger.info("Backend system initialized successfully")

# Create backend instance on module import
_initialize_backends()
backend = _backend_instance
