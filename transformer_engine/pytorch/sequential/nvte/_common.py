from __future__ import annotations
from collections import namedtuple
from typing import Any, Callable
import warnings
from enum import Enum
from types import GenericAlias
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
)


def torch_op(func: Callable[PS, T]) -> Callable[PS, T]:
    def make_wrapper(func: Callable[..., Any]):
        def type_name(t: type) -> str:
            if t.__module__ == "builtins":
                if isinstance(t, GenericAlias):
                    return str(t)
                else:
                    return t.__name__
            elif t.__module__ == "transformer_engine.pytorch.sequential.cpp_extensions":
                return f"cpp_extensions.{t.__name__}"
            else:
                return f"{t.__module__}.{t.__name__}"

        def wrap_unwrap_code(arg_name: str, arg_type: type, arg_type_name: str):
            wrapped_arg_type_name = type_name(wrap_type(arg_type))
            if arg_type is _nvte.Tensor:
                w = f"    {arg_name}_: {wrapped_arg_type_name} = te_to_torch_tensor({arg_name})\n"
                u = f"    {arg_name}: {arg_type_name} = torch_to_te_tensor({arg_name}_)\n"
            elif issubclass(arg_type, Enum):
                w = f"    {arg_name}_: {wrapped_arg_type_name} = {arg_name}.value\n"
                u = f"    {arg_name}: {arg_type_name} = {arg_type_name}({arg_name}_)\n"
            elif arg_type in [int, float, bool, str, torch.Tensor]:
                w = f"    {arg_name}_: {wrapped_arg_type_name} = {arg_name}\n"
                u = f"    {arg_name}: {arg_type_name} = {arg_name}_\n"
            else:
                raise NotImplementedError(arg_type_name)
            return (w, u)

        def wrap_type(arg_type: type):
            if arg_type is _nvte.Tensor:
                return tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
            elif issubclass(arg_type, Enum):
                return int
            elif arg_type in [int, float, bool, str, torch.Tensor]:
                return arg_type
            else:
                raise NotImplementedError(arg_type_name)

        def register_op(func: Callable[..., Any], abstract_impl: Callable[..., Any]):
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
            if not hasattr(register_op, "warned"):  # type: ignore
                register_op.warned = True  # type: ignore
                warnings.warn("Unable to find custom_op, decorator has no effect")

        # Dynamically generate code of the wrappers
        arg_types = get_arg_types(func)
        arg_names = get_arg_names(func)
        arg_type_names = list(map(type_name, arg_types))
        return_type = get_return_type(func)
        return_type_name = type_name(return_type)
        outer_sig = f"""({ ','.join(
            f'{arg_name}: {arg_type_name}'
            for arg_name, arg_type_name in zip(arg_names, arg_type_names)
        ) }) -> {return_type_name}"""
        arg_wrapping_code = ""
        arg_unwrapping_code = ""
        for arg_name, arg_type, arg_type_name in zip(
            arg_names, arg_types, arg_type_names
        ):
            w, u = wrap_unwrap_code(arg_name, arg_type, arg_type_name)
            arg_wrapping_code += w
            arg_unwrapping_code += u
        wrapped_args = ",".join(f"{arg_name}_" for arg_name in arg_names)

        result_wrapping_code, result_unwrapping_code = wrap_unwrap_code(
            "result", return_type, return_type_name
        )

        wrapped_arg_names = [f"{arg_name}_" for arg_name in arg_names]
        wrapped_arg_types = [wrap_type(t) for t in arg_types]
        wrapped_arg_type_names = [type_name(t) for t in wrapped_arg_types]
        wrapped_return_type = wrap_type(return_type)
        wrapped_return_type_name = type_name(wrapped_return_type)
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

raw_handles: list[cpp_extensions.RawTensor] = []

def te_to_torch_tensor(t: cpp_extensions.Tensor):
    raw_handles.append(t._raw)
    return (t.data, t.amax, t.scale, t.scale_inv)

def torch_to_te_tensor(t: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]):
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
            register_op(op_impl, op_aimp)

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
