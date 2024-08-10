# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Proxy class for tensor"""

from __future__ import annotations
import abc
from typing import Any, Dict, Optional, Tuple

import torch
from torch.utils._pytree import tree_map

class _DecodeFunc(torch.autograd.Function):

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

def _torch_dispatch_maybe_unwrap(arg: Any) -> Any:
    """Decode tensor proxies if needed

    Helper function in __torch_dispatch__.

    """
    if isinstance(arg, ProxyTensor):
        return arg.proxy_decode(dtype=arg.dtype)
    return arg

def _torch_dispatch_maybe_update_inplace(arg: Any, new_arg: Any, schema_arg: Any) -> None:
    """
    """

    if (
        isinstance(arg, ProxyTensor)
        and isinstance(new_arg, torch.Tensor)
        and hasattr(schema_arg, "alias_info")
        and hasattr(schema_arg.alias_info, "is_write")
        and schema_arg.alias_info.is_write
    ):
        arg.proxy_encode_(new_arg)



class ProxyTensor(torch.Tensor):

    def proxy_decode(self, *, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        raise NotImplementedError(
            f"{self.__class__.__name__} class does not implement proxy_decode function"
        )

    @abc.abstractmethod
    def proxy_encode_(self, tensor: torch.Tensor) -> Self:
        raise NotImplementedError(
            f"{self.__class__.__name__} class does not implement proxy_encode function"
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(decoded data={self.proxy_decode(dtype=self.dtype)})"

    def float(self) -> torch.Tensor:
        return _DecodeFunc.apply(tensor, dtype=torch.float32)

    def bfloat16(self) -> torch.Tensor:
        return _DecodeFunc.apply(tensor, dtype=torch.bfloat16)

    def half(self) -> torch.Tensor:
        return _DecodeFunc.apply(tensor, dtype=torch.float16)

    def cpu(self) -> torch.Tensor:
        return _DecodeFunc.apply(tensor).cpu()

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs=None):

        # In-place copy op
        if func == torch.ops.aten.copy_.default:
            dst = args[0]
            src = args[1]
            if isinstance(dst, ProxyTensor):
                dst.proxy_encode_(src)
            elif isinstance(src, ProxyTensor):
                dst.copy_(src.proxy_decode())
            else:
                dst.copy_(src)

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
                maybe_update_inplace(
                    kwargs[kwarg],
                    new_kwargs[new_kwarg],
                    schema_arg,
                )
            return None

        # Default op: decode proxy and perform op
        args = tree_map(maybe_unwrap, args)
        if kwargs is not None:
            kwargs = tree_map(maybe_unwrap, kwargs)
        out = super().__torch_dispatch__(func, types, args, kwargs)
        return out

    # Do not force the ProxyTensor type on the returned tensor
    __torch_function__ = classmethod(torch._C._disabled_torch_function_impl)
