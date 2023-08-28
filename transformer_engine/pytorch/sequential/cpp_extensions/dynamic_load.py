from enum import Enum
import functools
import inspect
from typing import Any, Callable, TypeVar
from ..utils import import_file_as_module
import torch
import transformer_engine_cuda  # type: ignore

_T1 = TypeVar("_T1")
_T2 = TypeVar("_T2")


def _to_dict(l: list[tuple[_T1, _T2]], /) -> dict[_T1, _T2]:
    return {t[0]: t[1] for t in l}


def _wrap_function(real_func: Callable[..., Any]):
    @functools.wraps(real_func)
    def wrapper(*args: Any):
        real_args: list[Any] = []
        for arg in args:
            if arg.__class__.__name__ == "Tensor":
                real_args.append(arg._raw)
            elif isinstance(arg, Enum):
                real_args.append(getattr(type(arg), "__orig_type__")(arg.value))
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

    for type_name, type_obj in stub_types.items():
        if type_name not in real_types:
            raise RuntimeError(
                f"Type {type_name} declared in {stub} not found in {real}"
            )
        if issubclass(type_obj, Enum):
            setattr(type_obj, "__orig_type__", real_types[type_name])
            namespace[type_name] = type_obj
        else:
            namespace[type_name] = real_types[type_name]
