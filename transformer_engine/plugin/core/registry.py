# Copyright (c) 2025, BAAI. All rights reserved.
#
# See LICENSE for license information.

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Dict, List, Sequence

from .types import OpImpl


@dataclass
class OpRegistrySnapshot:
    """Immutable snapshot of operator registry state"""
    impls_by_op: Dict[str, List[OpImpl]]


class OpRegistry:
    """
    Thread-safe registry for operator implementations.

    This registry stores operator implementations indexed by op_name and impl_id.
    Each operator can have multiple implementations from different backends/vendors.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        # Structure: {op_name: {impl_id: OpImpl}}
        self._impls_by_op: Dict[str, Dict[str, OpImpl]] = {}

    def register_impl(self, impl: OpImpl) -> None:
        """
        Register a single operator implementation.

        Args:
            impl: OpImpl instance to register

        Raises:
            ValueError: If impl_id is already registered for this op_name
        """
        with self._lock:
            by_id = self._impls_by_op.setdefault(impl.op_name, {})
            if impl.impl_id in by_id:
                raise ValueError(
                    f"Duplicate impl_id '{impl.impl_id}' for op='{impl.op_name}'. "
                    f"Existing: {by_id[impl.impl_id]}, New: {impl}"
                )
            by_id[impl.impl_id] = impl

    def register_many(self, impls: Sequence[OpImpl]) -> None:
        """
        Register multiple operator implementations.

        Args:
            impls: Sequence of OpImpl instances to register
        """
        for impl in impls:
            self.register_impl(impl)

    def snapshot(self) -> OpRegistrySnapshot:
        """
        Create an immutable snapshot of current registry state.

        Returns:
            OpRegistrySnapshot with all registered implementations
        """
        with self._lock:
            impls_by_op = {
                op: list(by_id.values())
                for op, by_id in self._impls_by_op.items()
            }
        return OpRegistrySnapshot(impls_by_op=impls_by_op)

    def get_implementations(self, op_name: str) -> List[OpImpl]:
        """
        Get all implementations for a specific operator.

        Args:
            op_name: Name of the operator

        Returns:
            List of OpImpl for the operator (empty if not found)
        """
        with self._lock:
            by_id = self._impls_by_op.get(op_name, {})
            return list(by_id.values())

    def get_implementation(self, op_name: str, impl_id: str) -> OpImpl | None:
        """
        Get a specific implementation by op_name and impl_id.

        Args:
            op_name: Name of the operator
            impl_id: Implementation ID

        Returns:
            OpImpl if found, None otherwise
        """
        with self._lock:
            by_id = self._impls_by_op.get(op_name, {})
            return by_id.get(impl_id)

    def list_operators(self) -> List[str]:
        """
        List all registered operator names.

        Returns:
            List of operator names
        """
        with self._lock:
            return list(self._impls_by_op.keys())

    def clear(self) -> None:
        """Clear all registered implementations"""
        with self._lock:
            self._impls_by_op.clear()
