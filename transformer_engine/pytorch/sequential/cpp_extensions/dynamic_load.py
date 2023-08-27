import functools
import inspect
from types import ModuleType
from typing import Any, Callable, TypeVar
from ..utils import import_file_as_module, get_arg_types
import torch
import re
import ast
import transformer_engine_cuda  # type: ignore

_T1 = TypeVar("_T1")
_T2 = TypeVar("_T2")


def _to_dict(l: list[tuple[_T1, _T2]], /) -> dict[_T1, _T2]:
    return {t[0]: t[1] for t in l}


def _get_stub_module():
    return import_file_as_module("__init__.pyi")


def _get_real_module() -> ModuleType:
    return transformer_engine_cuda


def _this_module():
    import sys

    return sys.modules[__name__]


def _name_resolution(name: str) -> Any:
    try:
        return ast.literal_eval(name)
    except ValueError:
        pass

    namespaces = name.split(".")
    result = _this_module()
    for name in namespaces:
        result = getattr(result, name)
    return result


def _get_real_func_arg_types(func: Callable[..., Any]):
    assert func.__doc__ is not None
    type_names: list[str] = re.split(r"[\(\),: ]", func.__doc__)[3:-2:4]
    types = [_name_resolution(name) for name in type_names]
    assert all(isinstance(t, type) for t in types)
    types: list[type]
    return types


def _wrap_function(real_func: Callable[..., Any]):
    Tensor = transformer_engine_cuda.Tensor  # type: ignore

    @functools.wraps(real_func)
    def wrapper(*args: Any):
        real_args = [arg if not isinstance(arg, Tensor) else arg.__raw for arg in args]
        return real_func(*real_args, torch.cuda.current_stream().cuda_stream)

    return wrapper


def inject_real(namespace: dict[str, Any]):
    stub = _get_stub_module()
    real = _get_real_module()

    stub_functions = _to_dict(inspect.getmembers(stub, inspect.isfunction))
    real_functions = _to_dict(inspect.getmembers(real, inspect.isroutine))

    for func_name, func_obj in stub_functions.items():
        if func_name not in real_functions:
            raise RuntimeError(
                f"Function {func_name} declared in {stub} not found in {real}"
            )
        stub_arg_types = get_arg_types(func_obj)
        real_arg_types = _get_real_func_arg_types(real_functions[func_name])
        if stub_arg_types != real_arg_types:
            raise RuntimeError(
                f"Function {func_name} implementation in {real} inconsistent with stub in {stub}"
            )
        namespace[func_name] = _wrap_function(real_functions[func_name])

    stub_types = _to_dict(inspect.getmembers(stub, inspect.isclass))
    real_types = _to_dict(inspect.getmembers(real, inspect.isclass))

    for type_name, _ in stub_types.items():
        if type_name not in real_types:
            raise RuntimeError(
                f"Type {type_name} declared in {stub} not found in {real}"
            )
        if type_name == "Tensor":
            namespace["RawTensor"] = real_types["Tensor"]
        else:
            namespace[type_name] = real_types[type_name]
