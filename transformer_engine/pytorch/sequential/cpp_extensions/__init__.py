from ..utils import import_file_as_module, get_arg_types

import_file_as_module("printing", only_for_side_effects=True)

from enum import Enum
import inspect
import transformer_engine_cuda  # type: ignore

stub = import_file_as_module("__init__.pyi")
from typing import TypeVar, Any

T1 = TypeVar("T1")
T2 = TypeVar("T2")


def to_dict(l: list[tuple[T1, T2]], /) -> dict[T1, T2]:
    return {t[0]: t[1] for t in l}


stub_functions = to_dict(inspect.getmembers(stub, inspect.isfunction))
stub_types = to_dict(inspect.getmembers(stub, inspect.isclass))
enum_names = {
    type_name
    for type_name, type_obj in stub_types.items()
    if issubclass(type_obj, Enum)
}

real_functions = to_dict(
    inspect.getmembers(transformer_engine_cuda, inspect.isfunction)
)
real_types = to_dict(inspect.getmembers(transformer_engine_cuda, inspect.isclass))

for enum_name in enum_names:
    globals()[enum_name] = stub_types[enum_name]

for class_name in stub_types.keys() - enum_names:
    stub_type = stub_types[class_name]
    real_type = real_types[class_name]
    real_type.__annotations__ = stub_type.__annotations__
    for attr_name, attr_obj in real_type.__dict__.items():
        attr_obj.__annotations__ = stub_type.__dict__[attr_name].__annotations__
    globals()[class_name] = real_type

for func_name, func_obj in stub_functions.items():
    stub_arg_types = tuple(get_arg_types(func_obj))

    def wrapper(*args: Any):
        real_args = ()
        for arg in args:
            if isinstance(arg, Enum):
                real_args += (arg.value,)
            else:
                real_args += (arg,)
        func_obj(*real_args)

    wrapper.__name__ = func_name
    wrapper.__annotations__ = func_obj.__annotations__
    globals()[func_name] = wrapper
