# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Tensor proxy class"""

from __future__ import annotations
import abc
from typing import Optional, Tuple

import torch
from torch.utils._pytree import tree_map


class _DecodeFunc(torch.autograd.Function):
    """Autograd function to convert tensor proxy to standard tensor"""

    @staticmethod
    def forward(
        _ctx: torch.autograd.function.FunctionCtx,  # unused
        tensor: ProxyTensor,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        return tensor.proxy_decode(dtype=dtype)

    @staticmethod
    def backward(
        _ctx: torch.autograd.function.FunctionCtx,  # unused
        grad: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], ...]:
        return grad, None

class _IdentityFunc(torch.autograd.Function):
    """Autograd function to create tensor proxy with same data"""

    @staticmethod
    def forward(
        _ctx: torch.autograd.function.FunctionCtx,  # unused
        tensor: ProxyTensor,
    ) -> ProxyTensor:
        return tensor.proxy_detach()

    @staticmethod
    def backward(
        _ctx: torch.autograd.function.FunctionCtx,  # unused
        grad: torch.Tensor,
    ) -> torch.Tensor:
        return grad


class ProxyTensor(torch.Tensor):
    """Proxy class for a tensor

    Tensor proxies do not store data like standard PyTorch tensors,
    i.e. in a memory address that can be accessed with the data_ptr
    method. Rather, they implement functions to encode/decode data in
    some other data format. Otherwise they have the same interface as
    standard PyTorch tensors, including support for PyTorch operations
    that do not involve sharing data (e.g. view).

    """

    def proxy_decode(self, *, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        """Create standard PyTorch tensor with values from tensor proxy"""
        raise NotImplementedError(
            f"{self.__class__.__name__} class does not implement proxy_decode function"
        )

    def proxy_encode_(self, tensor: torch.Tensor) -> Self:
        """Update values in tensor proxy"""
        raise NotImplementedError(
            f"{self.__class__.__name__} class does not implement proxy_encode_ function"
        )

    def proxy_detach(self) -> ProxyTensor:
        """Create new tensor proxy with same encoded data

        Output tensor must be detached from the current autograd
        graph.

        """
        raise NotImplementedError(
            f"{self.__class__.__name__} class does not implement proxy_detach function"
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(data={self.proxy_decode(dtype=self.dtype)})"

    def float(self) -> torch.Tensor:
        return _DecodeFunc.apply(tensor, dtype=torch.float32)

    def bfloat16(self) -> torch.Tensor:
        return _DecodeFunc.apply(tensor, dtype=torch.bfloat16)

    def half(self) -> torch.Tensor:
        return _DecodeFunc.apply(tensor, dtype=torch.float16)

    def cpu(self) -> torch.Tensor:
        return _DecodeFunc.apply(tensor).cpu()

    def expand_as(self, other: torch.Tensor) -> torch.Tensor:
        if other is self:
            # Note: expand_as is hackily used to create dummy autograd nodes
            # and access the backward graph (see
            # https://github.com/pytorch/pytorch/blob/238fb660851268f44ff88127887041fea352fe48/torch/nn/parallel/distributed.py#L1026).
            # We hackily add a dummy function to handle this case.
            return _IdentityFunc.apply(self)
        return super().expand_as(other)

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs=None):

        # Detach op
        if func == torch.ops.aten.detach.default:
            return args[0].proxy_detach()

        # In-place copy op
        if func == torch.ops.aten.copy_.default:
            dst = args[0]
            src = args[1]
            if isinstance(dst, ProxyTensor):
                dst.proxy_encode_(src)
            else:
                if isinstance(src, ProxyTensor):
                    src = src.proxy_decode()
                dst.copy_(src)
            return None

        # View op
        if func == torch.ops.aten.view.default:
            raise NotImplementedError("{cls.__name__} class does not support tensor views")

        def maybe_unwrap(arg):
            if isinstance(arg, ProxyTensor):
                return arg.proxy_decode(dtype=arg.dtype)
            return arg

        def maybe_update_inplace(arg, new_arg, schema_arg):
            if (
                    isinstance(arg, ProxyTensor)
                    and isinstance(new_arg, torch.Tensor)
                    and hasattr(schema_arg, "alias_info")
                    and hasattr(schema_arg.alias_info, "is_write")
                    and schema_arg.alias_info.is_write
            ):
                arg.proxy_encode_(new_arg)

        # In-place op: decode proxy, perform op, and encode proxy
        if func._schema.is_mutable:
            new_args = tree_map(maybe_unwrap, args)
            new_kwargs = tree_map(maybe_unwrap, kwargs)
            schema_args = func._schema.arguments
            args_len = len(args)
            super().__torch_dispatch__(func, types, new_args, new_kwargs)
            for arg, new_arg, schema_arg in zip(args, new_args, schema_args):
                maybe_update_inplace(arg, new_arg, schema_arg)
            for kwarg, new_kwarg, schema_arg in zip(kwargs, new_kwargs, schema_args[args_len:]):
                assert kwarg == new_kwarg == schema_arg.name, "name of the kw argument should match"
                maybe_update_inplace(kwargs[kwarg], new_kwargs[new_kwarg], schema_arg)
            return None

        # Default op: decode proxy and perform op
        args = tree_map(maybe_unwrap, args)
        if kwargs is not None:
            kwargs = tree_map(maybe_unwrap, kwargs)
        out = super().__torch_dispatch__(func, types, args, kwargs)
        return out

    # Do not force the ProxyTensor type on the returned tensor
    __torch_function__ = classmethod(torch._C._disabled_torch_function_impl)
