# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Sequential container for fusible operations."""

from __future__ import annotations
from collections.abc import Iterable, Iterator
from typing import Optional

import torch

from transformer_engine.pytorch.ops.op import FusibleOperation
from transformer_engine.pytorch.ops.fuser import OperationFuser


class Sequential(torch.nn.Module):
    """Sequential container for fusible operations

    This is a drop-in replacement for `torch.nn.Sequential`, with
    support for fusing `FusibleOperation`s.

    Parameters
    ----------
    *args: FusibleOperation or torch.nn.Module
        Neural network modules

    """

    def __init__(
        self,
        *args: FusibleOperation | torch.nn.Module,
    ) -> None:
        super().__init__()

        # List of modules, with fusible operations grouped together
        self._module_groups: Optional[list[OperationFuser | torch.nn.Module]]
        self._module_groups = None

        # Add modules
        if len(args) == 1 and isinstance(args[0], dict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for module in args:
                self.append(module)

    def add_module(self, name: str, module: Optional[torch.nn.Module]) -> None:
        # pylint: disable=missing-function-docstring
        self._module_groups = None
        super().add_module(name, module)

    def _get_keys_by_idx(self, idx: int | slice) -> list[str]:
        """Get module keys corresponding to indices"""
        if isinstance(idx, slice):
            return list(self._modules.keys())[idx]
        size = len(self._modules)
        if not -size <= idx < size:
            raise IndexError(f"Attempted to access index {idx}, but there are {size} entries")
        if idx < 0:
            idx += size
        for i, key in enumerate(self._modules.keys()):
            if i == idx:
                return [key]
        raise RuntimeError(f"Could not access index {idx}")

    def _next_key(self) -> str:
        """Key for a newly added module"""
        idx = 0
        for key in self._modules.keys():
            try:
                key_idx = int(key)
            except (ValueError, TypeError):
                pass
            else:
                idx = max(idx, key_idx + 1)
        return str(idx)

    def __getitem__(
        self,
        idx: slice | int,
    ) -> Sequential | torch.nn.Module:
        keys = self._get_keys_by_idx(idx)
        if isinstance(idx, slice):
            out = Sequential()
            out.extend(self._modules[key] for key in keys)
            return out
        return self._modules[keys[0]]

    def __setitem__(self, idx: int, module: torch.nn.Module) -> None:
        self._module_groups = None
        key = self._get_keys_by_idx(idx)[0]
        self._modules[key] = module

    def __delitem__(self, idx: slice | int) -> None:
        self._module_groups = None
        for key in self._get_keys_by_idx(idx):
            del self._modules[key]

    def __len__(self) -> int:
        return len(self._modules)

    def __iter__(self) -> Iterator[torch.nn.Module]:
        return iter(self._modules.values())

    def append(self, module: torch.nn.Module) -> Sequential:
        """Add module at the end of the container"""
        self.add_module(self._next_key(), module)
        return self

    def extend(self, modules: Iterable[torch.nn.Module]) -> Sequential:
        """Add modules at the end of the container"""
        for module in modules:
            self.append(module)
        return self

    def insert(self, idx: int, module: torch.nn.Module) -> Sequential:
        """Add modules at a position in the container"""
        self._module_groups = None
        keys = self._get_keys_by_idx(slice(idx, None))
        keys.append(self._next_key())
        for i in reversed(range(1, len(keys))):
            self._modules[keys[i]] = self._modules[keys[i - 1]]
        self._modules[keys[0]] = module
        return self

    def pop(self, idx: slice | int) -> torch.nn.Module:
        """Remove module at a position in the container"""
        out = self[idx]
        del self[idx]
        return out

    def __iadd__(self, modules: Iterable[torch.nn.Modules]) -> Sequential:
        return self.extend(modules)

    def __add__(self, modules: Iterable[torch.nn.Modules]) -> Sequential:
        out = Sequential()
        out.extend(self)
        out.extend(modules)
        return out

    @classmethod
    def _make_module_groups(
        cls,
        modules: Iterable[torch.nn.Module],
    ) -> list[OperationFuser | torch.nn.Module]:
        """Make list of modules, with fusible operations grouped together"""

        # Group fusible operations together
        groups = []
        for module in modules:
            if isinstance(module, FusibleOperation):
                if not groups or not isinstance(groups[-1], list):
                    groups.append([])
                groups[-1].append(module)
            else:
                groups.append(module)
        for idx, group in enumerate(groups):
            if isinstance(group, list):
                groups[idx] = OperationFuser(group, fuse_ops=True)

        # Check if operations expect extra input or output tensors
        # Note: If any op has extra inputs or outputs, then the entire
        # Sequential must be made up of TE ops.
        if len(groups) > 1:
            ops = []
            for group in groups:
                if isinstance(group, OperationFuser):
                    ops.extend(group._basic_ops)
            num_extra_inputs = sum(op.num_extra_inputs for op in ops)
            num_extra_outputs = sum(op.num_extra_outputs for op in ops)
            if num_extra_inputs > 0 or num_extra_outputs > 0:
                raise RuntimeError(
                    f"`Sequential` expects {num_extra_inputs} extra inputs "
                    f"and {num_extra_outputs} extra outputs, "
                    "but it contains non-fusible operations"
                )

        return groups

    def forward(
        self,
        input: torch.Tensor,  # pylint: disable=redefined-builtin
        *extra_inputs: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """Forward pass"""

        # Create module groups if needed
        if self._module_groups is None:
            self._module_groups = self._make_module_groups(self._modules.values())

        # Forward pass for each module group
        x = input
        for module_group in self._module_groups:
            x = module_group(x, *extra_inputs)
        return x
