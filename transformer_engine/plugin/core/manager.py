# Copyright (c) 2025, BAAI. All rights reserved.
#
# See LICENSE for license information.

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

from .discovery import discover_plugin
from .registry import OpRegistry
from .policy import SelectionPolicy, get_policy
from .types import OpImpl, BackendImplKind, match_token
from .logger_manager import get_logger

logger = get_logger()


@dataclass
class _OpManagerState:
    """Internal state for OpManager"""
    init_pid: int = -1
    initialized: bool = False
    policy_epoch: int = 0


class OpManager:
    """
    Main manager for operator dispatching and selection.

    Responsibilities:
    - Lazy initialization and plugin discovery
    - Multi-process safety (PID detection + at_fork)
    - Policy-based operator selection
    - Dispatch caching with invalidation
    """

    def __init__(self, registry: Optional[OpRegistry] = None) -> None:
        self._lock = threading.RLock()
        self._registry = registry or OpRegistry()
        self._state = _OpManagerState()
        self._dispatch_cache: Dict[Tuple[str, str, int], Callable] = {}
        self._called_ops: Dict[str, str] = {}  # Map op_name -> last_used_impl_id (for logging)

        # Register at_fork handler for multi-process safety
        try:
            os.register_at_fork(after_in_child=self._reset_after_fork)
        except AttributeError:
            # os.register_at_fork not available (Windows)
            pass

    @property
    def registry(self) -> OpRegistry:
        """Get the underlying operator registry"""
        return self._registry

    def _reset_after_fork(self) -> None:
        """Reset state after process fork"""
        with self._lock:
            self._state.initialized = False
            self._state.init_pid = -1
            self._state.policy_epoch += 1
            self._dispatch_cache.clear()
            self._called_ops.clear()
            logger.debug("OpManager reset after fork")

    def bump_policy_epoch(self) -> None:
        """
        Increment policy epoch to invalidate dispatch cache.

        Call this when policy changes at runtime.
        """
        with self._lock:
            self._state.policy_epoch += 1
            self._dispatch_cache.clear()
            logger.debug(f"Policy epoch bumped to {self._state.policy_epoch}")

    def ensure_initialized(self) -> None:
        """
        Ensure the manager is initialized in the current process.

        Performs:
        1. PID check (multi-process safety)
        2. Register built-in operator implementations
        3. Discover and register plugin
        """
        with self._lock:
            pid = os.getpid()

            # Check if already initialized in this process
            if self._state.initialized and self._state.init_pid == pid:
                return

            logger.debug(f"Initializing OpManager in PID {pid}")

            # Mark as initialized
            self._state.initialized = True
            self._state.init_pid = pid

            # Register built-in operators
            from . import builtin_ops
            builtin_ops.register_builtins(self._registry)

            # Discover and register plugin
            discover_plugin(self._registry)

            # Invalidate cache
            self._state.policy_epoch += 1
            self._dispatch_cache.clear()

            # Print initialization summary
            snap = self._registry.snapshot()
            total_ops = len(snap.impls_by_op)
            total_impls = sum(len(impls) for impls in snap.impls_by_op.values())

            logger.info(f"OpManager initialized: {total_ops} ops with {total_impls} implementations")

            # Group implementations by kind for summary
            vendor_count = sum(1 for impls in snap.impls_by_op.values()
                             for impl in impls if impl.kind == BackendImplKind.VENDOR)
            reference_count = sum(1 for impls in snap.impls_by_op.values()
                                for impl in impls if impl.kind == BackendImplKind.REFERENCE)
            default_count = sum(1 for impls in snap.impls_by_op.values()
                              for impl in impls if impl.kind == BackendImplKind.DEFAULT)

            logger.debug(f"  Vendor: {vendor_count}, Default: {default_count}, Reference: {reference_count}")

            # List all registered impl_ids
            if logger.logger.isEnabledFor(logger.logger.level):
                impl_ids = sorted(set(impl.impl_id for impls in snap.impls_by_op.values() for impl in impls))
                logger.info(f"Registered impl_ids: {impl_ids}")

    def _matches_vendor_filters(self, impl: OpImpl, policy: SelectionPolicy) -> bool:
        """Check if implementation matches policy vendor filters"""
        if impl.kind != BackendImplKind.VENDOR:
            return True

        if impl.vendor is None:
            return False

        # Check deny list
        if impl.vendor in policy.deny_vendors:
            return False

        # Check allow list (if specified)
        if policy.allow_vendors is not None and impl.vendor not in policy.allow_vendors:
            return False

        return True

    def _default_order(self, policy: SelectionPolicy) -> list[str]:
        """Get default selection order based on policy"""
        return policy.get_default_order()

    def resolve(self, op_name: str) -> Callable:
        """
        Resolve and return the best implementation for an operator.

        Selection process:
        1. Check dispatch cache
        2. Get all registered implementations
        3. Filter by policy (vendor allow/deny)
        4. Filter by availability (is_available())
        5. Select best match using per-op order or default order
        6. Cache the result

        Args:
            op_name: Name of the operator to resolve

        Returns:
            Callable implementation function

        Raises:
            RuntimeError: If no implementation found
        """
        self.ensure_initialized()

        policy = get_policy()
        policy_fp = policy.fingerprint()
        epoch = self._state.policy_epoch

        # Check cache
        cache_key = (op_name, policy_fp, epoch)
        cached = self._dispatch_cache.get(cache_key)
        if cached is not None:
            return cached

        # Get all implementations for this operator
        snap = self._registry.snapshot()
        candidates = list(snap.impls_by_op.get(op_name, []))

        # Filter by vendor policy
        candidates = [c for c in candidates if self._matches_vendor_filters(c, policy)]

        # Filter by availability
        available: list[OpImpl] = []
        for c in candidates:
            try:
                if c.is_available():
                    available.append(c)
                else:
                    logger.debug(f"Implementation {c.impl_id} not available for op={op_name}")
            except Exception as e:
                logger.warning(f"Error checking availability of {c.impl_id}: {e}")
                continue

        candidates = available

        if not candidates:
            raise RuntimeError(
                f"No available implementation for op='{op_name}'. "
                f"Registered: {[impl.impl_id for impl in snap.impls_by_op.get(op_name, [])]}"
            )

        # Get selection order (per-op or default)
        order = policy.per_op_order_dict.get(op_name) or self._default_order(policy)

        # Select best implementation
        chosen: Optional[OpImpl] = None
        for token in order:
            matches = [c for c in candidates if match_token(c, token)]
            if not matches:
                continue

            # Sort by priority (higher first), then by impl_id for stability
            matches.sort(key=lambda x: (x.priority, x.impl_id), reverse=True)
            chosen = matches[0]
            break

        if chosen is None:
            if policy.strict:
                raise RuntimeError(
                    f"No implementation available for op='{op_name}' under strict policy. "
                    f"Candidates: {[c.impl_id for c in candidates]}"
                )
            raise RuntimeError(
                f"No implementation selected for op='{op_name}'. "
                f"Candidates: {[c.impl_id for c in candidates]}, Order: {order}"
            )

        # Cache the result
        self._dispatch_cache[cache_key] = chosen.fn
        return chosen.fn

    def resolve_candidates(self, op_name: str) -> list[OpImpl]:
        """
        Resolve and return all available implementations for an operator,
        sorted by priority (highest first).

        This is similar to resolve() but returns all viable candidates
        instead of just the best one. Useful for fallback mechanisms.

        Args:
            op_name: Name of the operator to resolve

        Returns:
            List of OpImpl sorted by priority (highest first)

        Raises:
            RuntimeError: If no implementation found
        """
        self.ensure_initialized()

        policy = get_policy()

        # Get all implementations for this operator
        snap = self._registry.snapshot()
        candidates = list(snap.impls_by_op.get(op_name, []))

        # Filter by vendor policy
        candidates = [c for c in candidates if self._matches_vendor_filters(c, policy)]

        # Filter by availability
        available: list[OpImpl] = []
        for c in candidates:
            try:
                if c.is_available():
                    available.append(c)
                else:
                    logger.debug(f"Implementation {c.impl_id} not available for op={op_name}")
            except Exception as e:
                logger.warning(f"Error checking availability of {c.impl_id}: {e}")
                continue

        candidates = available

        if not candidates:
            raise RuntimeError(
                f"No available implementation for op='{op_name}'. "
                f"Registered: {[impl.impl_id for impl in snap.impls_by_op.get(op_name, [])]}"
            )

        # Get selection order (per-op or default)
        order = policy.per_op_order_dict.get(op_name) or self._default_order(policy)

        # Sort candidates by order tokens, then by priority
        sorted_candidates: list[OpImpl] = []
        for token in order:
            matches = [c for c in candidates if match_token(c, token)]
            if matches:
                # Sort by priority (higher first), then by impl_id for stability
                matches.sort(key=lambda x: (x.priority, x.impl_id), reverse=True)
                sorted_candidates.extend(matches)

        # Remove duplicates while preserving order
        seen = set()
        unique_candidates = []
        for c in sorted_candidates:
            if c.impl_id not in seen:
                seen.add(c.impl_id)
                unique_candidates.append(c)

        if not unique_candidates:
            raise RuntimeError(
                f"No implementation selected for op='{op_name}'. "
                f"Candidates: {[c.impl_id for c in candidates]}, Order: {order}"
            )

        return unique_candidates

    def call(self, op_name: str, *args, **kwargs):
        """
        Resolve and call an operator implementation with optional fallback support.

        When TE_FL_STRICT=1, this method will try alternative implementations
        if the primary one fails. Otherwise, it behaves like the original implementation.

        Logs on first call or when the implementation changes (e.g., backend switch).

        Args:
            op_name: Name of the operator
            *args, **kwargs: Arguments passed to the implementation

        Returns:
            Result from the implementation

        Raises:
            RuntimeError: If all implementations fail (when fallback enabled) or
                         if the primary implementation fails (when fallback disabled)
        """
        enable_fallback = os.getenv("TE_FL_STRICT", "1") != "0"

        if not enable_fallback:
            # Original behavior: use cached resolve() and fast-fail
            fn = self.resolve(op_name)

            # Get current impl_id to check if it changed
            impl_id = self.get_selected_impl_id(op_name)
            last_impl_id = self._called_ops.get(op_name)

            # Log if first call or implementation changed
            if last_impl_id != impl_id:
                with self._lock:
                    # Double-check after acquiring lock
                    if self._called_ops.get(op_name) != impl_id:
                        snap = self._registry.snapshot()
                        for impl in snap.impls_by_op.get(op_name, []):
                            if impl.impl_id == impl_id:
                                if last_impl_id is None:
                                    logger.info(
                                        f"Op '{op_name}' using '{impl_id}' "
                                        f"(kind={impl.kind.value}, vendor={impl.vendor})"
                                    )
                                else:
                                    logger.info(
                                        f"Op '{op_name}' switched from '{last_impl_id}' to '{impl_id}' "
                                        f"(kind={impl.kind.value}, vendor={impl.vendor})"
                                    )
                                break
                        self._called_ops[op_name] = impl_id

            return fn(*args, **kwargs)

        # Fallback mode: try candidates in priority order
        candidates = self.resolve_candidates(op_name)
        last_error = None

        for idx, impl in enumerate(candidates):
            try:
                # Log primary implementation or fallback attempts
                if idx == 0:
                    # Primary implementation
                    last_impl_id = self._called_ops.get(op_name)
                    if last_impl_id != impl.impl_id:
                        with self._lock:
                            if self._called_ops.get(op_name) != impl.impl_id:
                                if last_impl_id is None:
                                    logger.info(
                                        f"Op '{op_name}' using '{impl.impl_id}' "
                                        f"(kind={impl.kind.value}, vendor={impl.vendor})"
                                    )
                                else:
                                    logger.info(
                                        f"Op '{op_name}' switched from '{last_impl_id}' to '{impl.impl_id}' "
                                        f"(kind={impl.kind.value}, vendor={impl.vendor})"
                                    )
                                self._called_ops[op_name] = impl.impl_id
                else:
                    # Always log fallback attempts (these are important runtime events)
                    logger.info(
                        f"Op '{op_name}' fallback to '{impl.impl_id}' "
                        f"(kind={impl.kind.value}, vendor={impl.vendor})"
                    )

                result = impl.fn(*args, **kwargs)

                # Update tracked impl_id on success (for fallback case)
                if idx > 0:
                    with self._lock:
                        self._called_ops[op_name] = impl.impl_id

                return result

            except Exception as e:
                last_error = e
                if idx < len(candidates) - 1:
                    # Not the last candidate, log warning and try next
                    logger.warning(
                        f"Implementation '{impl.impl_id}' failed for op '{op_name}': {e}"
                    )
                else:
                    # Last candidate failed, log error
                    logger.error(
                        f"Last implementation '{impl.impl_id}' failed for op '{op_name}': {e}"
                    )

        # All implementations failed
        raise RuntimeError(
            f"All {len(candidates)} implementation(s) failed for op='{op_name}'. "
            f"Last error: {last_error}"
        ) from last_error

    def get_selected_impl_id(self, op_name: str) -> str:
        """
        Get the impl_id of the currently selected implementation.

        Args:
            op_name: Name of the operator

        Returns:
            Implementation ID string
        """
        fn = self.resolve(op_name)

        # Try to find the impl by function identity
        snap = self._registry.snapshot()
        for impl in snap.impls_by_op.get(op_name, []):
            if impl.fn is fn:
                return impl.impl_id

        return "unknown"


# Global default instance
_default_manager: Optional[OpManager] = None
_manager_lock = threading.RLock()


def get_default_manager() -> OpManager:
    """Get or create the global default OpManager instance"""
    global _default_manager

    if _default_manager is None:
        with _manager_lock:
            if _default_manager is None:
                _default_manager = OpManager()

    return _default_manager


def reset_default_manager() -> None:
    """Reset the global default OpManager (useful for testing)"""
    global _default_manager

    with _manager_lock:
        _default_manager = None
