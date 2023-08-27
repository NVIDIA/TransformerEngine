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


def _wrap_function(real_func: Callable[..., Any]):
    Tensor = transformer_engine_cuda.Tensor  # type: ignore

    @functools.wraps(real_func)
    def wrapper(*args: Any):
        real_args = [arg if not isinstance(arg, Tensor) else arg.__raw for arg in args]
        return real_func(*real_args, torch.cuda.current_stream().cuda_stream)

    return wrapper


def inject_real(namespace: dict[str, Any]):
    stub = import_file_as_module("__init__.pyi")
    real = transformer_engine_cuda

    stub_functions = _to_dict(inspect.getmembers(stub, inspect.isfunction))
    real_functions = _to_dict(inspect.getmembers(real, inspect.isroutine))

    for func_name, _ in stub_functions.items():
        if func_name not in real_functions:
            raise RuntimeError(
                f"Function {func_name} declared in {stub} not found in {real}"
            )
        namespace[func_name] = _wrap_function(real_functions[func_name])

    stub_types = _to_dict(inspect.getmembers(stub, inspect.isclass))
    real_types = _to_dict(inspect.getmembers(real, inspect.isclass))

    for type_name, _ in stub_types.items():
        if type_name == "RawTensor":
            namespace["RawTensor"] = real_types["Tensor"]
        elif type_name == "Tensor":
            continue
        else:
            if type_name not in real_types:
                raise RuntimeError(
                    f"Type {type_name} declared in {stub} not found in {real}"
                )
            namespace[type_name] = real_types[type_name]
