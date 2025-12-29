# Copyright (c) 2025, BAAI. All rights reserved.
#
# See LICENSE for license information.

import torch
import numpy as np
from typing import List, Dict, Callable, Any, Optional


def get_available_backends() -> List[str]:
    """
    Get list of available backends by extracting unique impl_ids from OpRegistry.

    Returns impl_id prefixes (e.g., "default.flagos" -> "flagos")
    """
    try:
        from transformer_engine.plugin.core import get_registry

        registry = get_registry()
        all_impls = []
        for op_name in registry.list_operators():
            all_impls.extend(registry.get_implementations(op_name))

        # Extract unique impl_id prefixes (e.g., "default.flagos" -> "flagos")
        impl_ids = set()
        for impl in all_impls:
            # impl_id format: "kind.name" (e.g., "default.flagos", "vendor.cuda")
            parts = impl.impl_id.split('.', 1)
            if len(parts) == 2:
                impl_ids.add(parts[1])  # Get the "name" part
            else:
                impl_ids.add(impl.impl_id)

        return sorted(impl_ids)
    except Exception as e:
        print(f"Warning: Could not load backends: {e}")
        import traceback
        traceback.print_exc()
        return []


def get_backend(name: str):
    """
    Get a backend-like object that dispatches to a specific implementation.

    Args:
        name: Backend name (e.g., "cuda", "flagos", "torch")

    Returns:
        A wrapper object that calls the specific backend implementation
    """
    from transformer_engine.plugin.core import get_registry
    from transformer_engine.plugin.core.logger_manager import get_logger
    import functools

    logger = get_logger()

    class BackendWrapper:
        """Wrapper that calls specific backend implementations"""

        def __init__(self, backend_name: str):
            self.backend_name = backend_name
            self.registry = get_registry()
            self._called_ops = set()  # Track which ops have been called (for logging)

        def _find_impl(self, op_name: str):
            """Find implementation matching the backend name"""
            impls = self.registry.get_implementations(op_name)

            # Try to find implementation matching backend_name
            # Match against impl_id suffix (e.g., "vendor.cuda" matches "cuda")
            for impl in impls:
                if impl.impl_id.endswith(f".{self.backend_name}") or impl.impl_id == self.backend_name:
                    if impl.is_available():
                        return impl
                    else:
                        raise RuntimeError(
                            f"Implementation '{impl.impl_id}' for op '{op_name}' is not available"
                        )

            raise NotImplementedError(
                f"No implementation found for op '{op_name}' with backend '{self.backend_name}'"
            )

        def __getattr__(self, op_name: str):
            """Dynamically resolve operator to specific backend implementation"""
            impl = self._find_impl(op_name)

            # Log on first call to this op for this backend
            if op_name not in self._called_ops:
                self._called_ops.add(op_name)
                logger.info(
                    f"[Test] Op '{op_name}' using '{impl.impl_id}' "
                    f"(kind={impl.kind.value}, vendor={impl.vendor})"
                )

            return impl.fn

    return BackendWrapper(name)


def allclose(a: torch.Tensor, b: torch.Tensor, rtol: float = 1e-5, atol: float = 1e-8) -> bool:
    return torch.allclose(a, b, rtol=rtol, atol=atol)


def compute_relative_error(output: torch.Tensor, reference: torch.Tensor) -> float:
    diff = (output - reference).abs()
    relative_error = (diff / (reference.abs() + 1e-10)).mean().item()
    return relative_error


def compute_max_error(output: torch.Tensor, reference: torch.Tensor) -> float:
    return (output - reference).abs().max().item()


class TestCase:
    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self.passed = 0
        self.failed = 0
        self.skipped = 0
        self.errors: List[str] = []

    def setup(self):
        pass

    def teardown(self):
        pass

    def assert_close(
        self,
        output: torch.Tensor,
        reference: torch.Tensor,
        rtol: float = 1e-5,
        atol: float = 1e-8,
        msg: str = "",
    ):
        if not allclose(output, reference, rtol, atol):
            max_err = compute_max_error(output, reference)
            rel_err = compute_relative_error(output, reference)
            error_msg = f"{msg}\n  Max error: {max_err:.6e}, Relative error: {rel_err:.6e}"
            self.errors.append(error_msg)
            self.failed += 1
            raise AssertionError(error_msg)
        self.passed += 1

    def report(self):
        total = self.passed + self.failed + self.skipped
        print(f"\n{'='*60}")
        print(f"Test: {self.name}")
        if self.description:
            print(f"Description: {self.description}")
        print(f"{'='*60}")
        print(f"Total: {total}, Passed: {self.passed}, Failed: {self.failed}, Skipped: {self.skipped}")
        if self.errors:
            print(f"\nErrors:")
            for i, error in enumerate(self.errors, 1):
                print(f"  {i}. {error}")
        print(f"{'='*60}")
        return self.failed == 0


def generate_random_tensor(
    shape: tuple,
    dtype: torch.dtype = torch.float32,
    device: str = "cpu",
    requires_grad: bool = False,
) -> torch.Tensor:
    if dtype in (torch.bfloat16, torch.float16):
        tensor = torch.randn(shape, dtype=torch.float32, device=device)
        tensor = tensor.to(dtype=dtype)
        if requires_grad:
            tensor.requires_grad_(True)
    else:
        tensor = torch.randn(shape, dtype=dtype, device=device, requires_grad=requires_grad)
    return tensor


def generate_test_shapes() -> List[tuple]:
    return [
        (2, 4),
        (8, 16),
        (32, 64),
        (2, 4, 8),
        (4, 8, 16),
        (2, 4, 8, 16),
    ]


def run_test_on_backends(
    test_func: Callable,
    backends: Optional[List[str]] = None,
    reference_backend: str = "reference",
) -> Dict[str, bool]:
    if backends is None:
        backends = get_available_backends()

    results = {}
    for backend_name in backends:
        try:
            test_func(backend_name)
            results[backend_name] = True
            print(f"  ✓ {backend_name}")
        except Exception as e:
            results[backend_name] = False
            print(f"  ✗ {backend_name}: {e}")

    return results


def skip_if_backend_unavailable(backend_name: str) -> bool:
    available = get_available_backends()
    return backend_name not in available
