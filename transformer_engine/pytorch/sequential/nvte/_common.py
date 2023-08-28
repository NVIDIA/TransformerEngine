from __future__ import annotations
from collections import namedtuple
from typing import Any, Callable, Sequence
from types import GenericAlias
import typing
import warnings
from enum import Enum

import torch
from .. import cpp_extensions as _nvte
from ..utils import (
    PS,
    T,
    get_arg_names,
    get_arg_types,
    get_return_type,
    exec_saving_source,
    reinterpret_cast,
    is_generic,
)


def _type_name(t: type) -> str:
    if is_generic(t):
        result = str(t)
    else:
        result = f"{t.__module__}.{t.__name__}"

    return (
        result.replace("builtins.", "")
        .replace("transformer_engine.pytorch.sequential.", "")
        .replace("collections.abc", "typing")
        .replace("__init__.pyi", "cpp_extensions")
    )


def _wrap_type(
    type_wrap_func: Callable[[type], type],
    arg_type_: type | GenericAlias,
) -> Any:
    if is_generic(arg_type_):
        arg_type_ = reinterpret_cast(arg_type_, GenericAlias)
        origin = arg_type_.__origin__
        while hasattr(origin, "__origin__"):
            origin = getattr(origin, "__origin__")
        args: tuple[type | GenericAlias, ...] = typing.get_args(arg_type_)
        new_args = [_wrap_type(type_wrap_func, arg) for arg in args]
        return origin.__class_getitem__(new_args)  # type: ignore
    else:
        arg_type_ = reinterpret_cast(arg_type_, type)
        return type_wrap_func(arg_type_)


def _arg_type_wrap_func(arg_type: type):
    if arg_type is _nvte.Tensor:
        return Sequence[torch.Tensor]
    elif issubclass(arg_type, Enum):
        return int
    elif issubclass(arg_type, (int, float, bool, str, torch.Tensor)):
        return arg_type
    else:
        raise NotImplementedError(arg_type)


def _wrap_arg_type(arg_type: type | GenericAlias) -> Any:
    return _wrap_type(_arg_type_wrap_func, arg_type)


def _result_type_wrap_func(result_type: type):
    if result_type is _nvte.Tensor:
        return tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    else:
        return _arg_type_wrap_func(result_type)


def _wrap_result_type(result_type: type | GenericAlias) -> Any:
    wrapped_type = _wrap_type(_result_type_wrap_func, result_type)
    # Flatten tuple of tuples of tensors
    if issubclass(wrapped_type, tuple):
        arg_types = typing.get_args(wrapped_type)
        if any(issubclass(arg_type, tuple) for arg_type in arg_types):
            assert all(
                issubclass(arg_type, tuple)
                and typing.get_args(arg_type)
                == (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor)
                for arg_type in arg_types
            )
            tensors = len(arg_types)
            types = (torch.Tensor,) * 4 * tensors
            return tuple.__class_getitem__(types)
    return wrapped_type  # type: ignore


def _wrap_unwrap_code(
    arg_name: str,
    arg_type: type,
    arg_type_name: str,
    wrapped_arg_type_name: str,
):
    if arg_type is _nvte.Tensor:
        w = f"    {arg_name}_: {wrapped_arg_type_name} = te_to_torch_tensor({arg_name})\n"
        u = f"    {arg_name}: {arg_type_name} = torch_to_te_tensor({arg_name}_)\n"
    elif issubclass(arg_type, tuple) and all(
        sub_type is _nvte.Tensor for sub_type in typing.get_args(arg_type)
    ):
        w = f"    {arg_name}_: {wrapped_arg_type_name} = tuple(t for tensor in {arg_name} for t in te_to_torch_tensor(tensor))\n"
        u = f"    {arg_name}: {arg_type_name} = tuple(torch_to_te_tensor(*({arg_name}_[j] for j in range(i, i + 4, 1))) for i in range(0, len({arg_name}_), 4))\n"
    elif issubclass(arg_type, Enum):
        w = f"    {arg_name}_: {wrapped_arg_type_name} = {arg_name}.value\n"
        u = f"    {arg_name}: {arg_type_name} = {arg_type_name}({arg_name}_)\n"
    else:
        w = f"    {arg_name}_: {wrapped_arg_type_name} = {arg_name}\n"
        u = f"    {arg_name}: {arg_type_name} = {arg_name}_\n"
    return (w, u)


def _arg_wrap_unwrap_code(arg_name: str, arg_type: type, arg_type_name: str):
    wrapped_arg_type_name = _type_name(_wrap_arg_type(arg_type))
    return _wrap_unwrap_code(arg_name, arg_type, arg_type_name, wrapped_arg_type_name)


def _result_wrap_unwrap_code(result_type: type, result_type_name: str):
    wrapped_result_type_name = _type_name(_wrap_result_type(result_type))
    return _wrap_unwrap_code(
        "result", result_type, result_type_name, wrapped_result_type_name
    )


def _register_op(func: Callable[..., Any], abstract_impl: Callable[..., Any]):
    name = f"nvte::{func.__name__}"
    # Different versions of PyTorch have different ways of registering custom ops
    try:
        decl, impl, aimp = (  # type: ignore
            torch._custom_ops.custom_op,  # type: ignore
            torch._custom_ops.impl,  # type: ignore
            torch._custom_ops.impl_abstract,  # type: ignore
        )
        decl(name)(func)
        impl(name)(func)
        aimp(name)(abstract_impl)
        return
    except AttributeError:
        pass
    try:
        decl = torch._custom_op.impl.custom_op  # type: ignore
        declared = decl(name)(func)  # type: ignore
        declared.impl("cuda")(func)  # type: ignore
        declared.impl_abstract()(abstract_impl)  # type: ignore
        return
    except AttributeError:
        pass
    if not hasattr(_register_op, "warned"):  # type: ignore
        _register_op.warned = True  # type: ignore
        warnings.warn("Unable to find custom_op, decorator has no effect")


def torch_op(func: Callable[PS, T]) -> Callable[PS, T]:
    def make_wrapper(func: Callable[..., Any]):
        # Dynamically generate code of the wrappers
        arg_types = get_arg_types(func)
        arg_names = get_arg_names(func)
        arg_type_names = list(map(_type_name, arg_types))
        return_type = get_return_type(func)
        return_type_name = _type_name(return_type)
        outer_sig = f"""({ ','.join(
            f'{arg_name}: {arg_type_name}'
            for arg_name, arg_type_name in zip(arg_names, arg_type_names)
        ) }) -> {return_type_name}"""
        arg_wrapping_code = ""
        arg_unwrapping_code = ""
        for arg_name, arg_type, arg_type_name in zip(
            arg_names, arg_types, arg_type_names
        ):
            w, u = _arg_wrap_unwrap_code(arg_name, arg_type, arg_type_name)
            arg_wrapping_code += w
            arg_unwrapping_code += u
        wrapped_args = ",".join(f"{arg_name}_" for arg_name in arg_names)

        result_wrapping_code, result_unwrapping_code = _result_wrap_unwrap_code(
            return_type, return_type_name
        )

        wrapped_arg_names = [f"{arg_name}_" for arg_name in arg_names]
        wrapped_arg_types = [_wrap_arg_type(t) for t in arg_types]
        wrapped_arg_type_names = [_type_name(t) for t in wrapped_arg_types]
        wrapped_return_type = _wrap_result_type(return_type)
        wrapped_return_type_name = _type_name(wrapped_return_type)
        inner_sig = f"""({ ','.join(
            f'{arg_name}: {arg_type_name}'
            for arg_name, arg_type_name in zip(wrapped_arg_names, wrapped_arg_type_names)
        ) }) -> {wrapped_return_type_name}"""
        unwrapped_args = ",".join(f"{arg_name}" for arg_name in arg_names)

        arg_unwrapping_code = arg_unwrapping_code.lstrip()
        arg_wrapping_code = arg_wrapping_code.lstrip()
        result_wrapping_code = result_wrapping_code.lstrip()
        result_unwrapping_code = result_unwrapping_code.lstrip()

        source = f"""\
import torch
from .. import cpp_extensions
import typing

raw_handles: list[cpp_extensions.RawTensor] = []

def te_to_torch_tensor(t: cpp_extensions.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    raw_handles.append(t._raw)
    return (t.data, t.amax, t.scale, t.scale_inv)

def torch_to_te_tensor(t: typing.Sequence[torch.Tensor]) -> cpp_extensions.Tensor:
    _raw = raw_handles.pop(0)
    return cpp_extensions.Tensor(_raw, *t)

def {func.__name__}_aimp{inner_sig}:
    {arg_unwrapping_code}
    func.__globals__["_nvte"] = impostor
    result: {return_type_name} = func({unwrapped_args})
    func.__globals__["_nvte"] = cpp_extensions
    {result_wrapping_code}
    return result_

def {func.__name__}{inner_sig}:
    {arg_unwrapping_code}
    result: {return_type_name} = func({unwrapped_args})
    {result_wrapping_code}
    return result_

def {func.__name__}_wrap{outer_sig}:
    {arg_wrapping_code}
    result_: {wrapped_return_type_name} = torch.ops.nvte.{func.__name__}({wrapped_args})
    {result_unwrapping_code}
    return result
"""
        try:
            # Swap real cpp_extensions (_nvte) for impostor that does nothing
            # This is needed so the abstract implementation is traceable by PyTorch Dynamo
            class NVTEImpostor:
                def __getattr__(self, attr_name: str) -> Any:
                    if attr_name == "Tensor":
                        return namedtuple("Tensor", ["data", "amax", "scale", "scale_inv"])  # type: ignore
                    else:
                        attr = getattr(_nvte, attr_name)
                        if callable(attr):
                            return lambda *args, **kwargs: None  # type: ignore
                        else:
                            return attr

            # Create op
            ns = dict(func=func, __name__=__name__, impostor=NVTEImpostor())
            exec_saving_source(source, ns)
            op_impl = reinterpret_cast(ns[func.__name__], Callable[..., Any])
            op_wrap = reinterpret_cast(ns[f"{func.__name__}_wrap"], Callable[PS, T])
            op_aimp = reinterpret_cast(ns[f"{func.__name__}_aimp"], Callable[..., Any])
            _register_op(op_impl, op_aimp)

            return op_wrap
        except Exception as e:
            raise RuntimeError(
                f"Failed to compile wrapper for {func.__name__}. Generated code: \n```\n{source}```"
            ) from e

    return make_wrapper(func)


def make_nvte_tensor(t: torch.Tensor) -> _nvte.Tensor:
    return _nvte.Tensor(
        t.data,
        torch.Tensor().cuda(),
        torch.Tensor().cuda(),
        torch.Tensor().cuda(),
    )
