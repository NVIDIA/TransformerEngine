# Copyright (c) 2025, BAAI. All rights reserved.
#
# See LICENSE for license information.

"""Backend registry for managing multiple backend implementations."""
import os
from typing import Any, Dict, Optional

from .logger import get_logger
logger = get_logger()


class Backend:
    """
    A backend that can register and provide implementations for various operations.
    
    Each backend can register its own implementations for operations like gemm,
    rmsnorm_fwd, etc. If an operation is not registered, it will fallback to
    the native backend.
    
    Usage:
        backend = Backend("my_backend")
        backend.register("gemm", my_gemm_function)
        backend.register("rmsnorm_fwd", my_rmsnorm_fwd)
        
        # Use the backend
        result = backend.gemm(...)
    """
    
    def __init__(self, name: str):
        """
        Initialize a backend.
        
        Args:
            name: Name of the backend (e.g., "native", "te_fl", "custom")
        """
        self.name = name
        self._implementations: Dict[str, Any] = {}
    
    def register(self, operation: str, implementation: Any) -> None:
        """
        Register an implementation for an operation.
        
        Args:
            operation: Name of the operation (e.g., "gemm", "rmsnorm_fwd")
            implementation: Function or class to register
        """
        self._implementations[operation] = implementation
        logger.info(f"Backend '{self.name}' registered implementation for '{operation}'")
    
    def has(self, operation: str) -> bool:
        """Check if this backend has an implementation for the operation."""
        return operation in self._implementations
    
    def get(self, operation: str, default: Optional[Any] = None) -> Optional[Any]:
        """Get the implementation for an operation, or return default if not found."""
        return self._implementations.get(operation, default)
    
    def __getattr__(self, operation: str) -> Any:
        """
        Allow accessing operations as attributes (e.g., backend.gemm).
        Returns the registered implementation if available.
        """
        if operation.startswith("_") or operation in ("name", "register", "has", "get"):
            return super().__getattribute__(operation)
        
        if operation in self._implementations:
            return self._implementations[operation]
        
        raise AttributeError(
            f"Backend '{self.name}' does not have implementation for '{operation}'. "
            f"Available operations: {list(self._implementations.keys())}"
        )


def get_selected_backend() -> Backend:
    """
    Get the selected backend instance based on global environment variable.
    No longer depends on operation-specific flags.
    
    Returns:
        Backend instance to use
    """
    global_flag = os.environ.get("USE_TRANSFORMER_ENGINE_FL", "0")
    if global_flag.lower() in ("1", "true", "yes", "on"):
        backend_name = "te_fl"
    else:
        backend_name = "native"
    return get_backend(backend_name)


# Global backends registry
_backends: Dict[str, Backend] = {}


def get_backend(name: str) -> Backend:
    """
    Get a backend by name. Creates it if it doesn't exist.
    
    Args:
        name: Name of the backend
    
    Returns:
        Backend instance
    """
    if name not in _backends:
        _backends[name] = Backend(name)
    return _backends[name]


def register_backend(backend_name: str, implementations: Dict[str, Any]):
    """
    Register backend implementations.
    
    Args:
        backend_name: Name of the backend (e.g., "native", "te_fl", "custom")
        implementations: Dictionary mapping operation names to their implementations.
                        Example: {"gemm": native_gemm, "flash_attention": native_flash_attn}
    
    Usage:
        # Register native backend
        register_backend("native", {
            "gemm": gemm_native,
            "rmsnorm_fwd": rmsnorm_fwd_native,
            "flash_attention": flash_attn_native,
        })
        
        # Register TE-FL backend
        register_backend("te_fl", {
            "gemm": gemm_fl,
            "rmsnorm_fwd": rmsnorm_fwd_fl,
            "flash_attention": flash_attn_fl,
        })
        
        # Register custom backend
        register_backend("custom", {
            "gemm": custom_gemm,
            "custom_op": custom_function,
        })
    """
    backend = get_backend(backend_name)
    
    for operation, implementation in implementations.items():
        backend.register(operation, implementation)
