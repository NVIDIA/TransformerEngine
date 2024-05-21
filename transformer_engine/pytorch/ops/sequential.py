# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

from __future__ import annotations

from collections import OrderedDict

import torch

from transformer_engine.pytorch.ops import FusableOperation
from transformer_engine.pytorch.ops.fuser import OperationFuser


class Sequential(torch.nn.Module):
    """Sequential container for fusable operations

    This is a drop-in replacement for `torch.nn.Sequential`, with
    support for fusing `FusableOperation`s.

    Parameters
    ----------
    *args: FusableOperation or torch.nn.Module
        Neural network modules

    """

    def __init__(
        self,
        *args: FusableOperation | torch.nn.Module,
    ) -> None:
        super().__init__()

        # Add modules
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)

        # List of modules, with fusable operations grouped together
        self._module_groups: Optional[list[OperationFuser | torch.nn.Module]]
        self._module_groups = None

    def add_module(self, *args, **kwargs) -> None:
        super().add_module(*args, **kwargs)
        self._module_groups = None

    def __getitem__(
        self,
        idx: slice | int,
    ) -> list[torch.nn.Module] | torch.nn.Module:
        if isinstance(idx, slice):
            return list(self._modules.values())[idx]
        else:
            size = len(self)
            if not -size <= idx < size:
                raise IndexError(
                    f"Attempted to access index {idx}, "
                    f"but there are {size} entries"
                )
            idx %= size
            for i, op in enumerate(self._modules.values()):
                if i == idx:
                    break
            return op

    def __len__(self) -> int:
        return len(self._modules)

    def _make_module_groups(
        self,
        modules: Iterable[torch.nn.Module],
    ) -> list[OperationFuser | torch.nn.Module]:
        """Make list of modules, with fusable operations grouped together"""
        module_groups = []
        fusable_ops = []
        def maybe_add_fuser():
            nonlocal fusable_ops
            if fusable_ops:
                module_groups.append(OperationFuser(fusable_ops, fuse_ops=True))
                fusable_ops = []
        for module in modules:
            if isinstance(module, FusableOperation):
                fusable_ops.append(module)
            else:
                maybe_add_fuser()
                module_groups.append(module)
        maybe_add_fuser()
        return module_groups

    def forward(self, input: torch.Tensor) -> torch.Tensor:

        # Create module groups if needed
        if self._module_groups is None:
            self._module_groups = self._make_module_groups(self._modules.values())

        # Forward pass for each module group
        x = input
        for module_group in self._module_groups:
            x = module_group(x)
        return x

    ### TODO Dunder functions
