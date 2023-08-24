from enum import Enum
import inspect
from typing import Any
import transformer_engine_cuda  # type: ignore
from ..utils import import_file_as_module, get_return_type


def inject_real(namespace: dict[str, Any]):
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
        namespace[enum_name] = stub_types[enum_name]

    for class_name in stub_types.keys() - enum_names:
        stub_type = stub_types[class_name]
        real_type = real_types[class_name]
        real_type.__annotations__ = stub_type.__annotations__
        for attr_name, attr_obj in real_type.__dict__.items():
            attr_obj.__annotations__ = stub_type.__dict__[attr_name].__annotations__
        namespace[class_name] = real_type

    for func_name, func_obj in stub_functions.items():
        real_func = real_functions[func_name]
        exposed_return_type: type = get_return_type(func_obj)

        def wrapper(*args: Any) -> Any:
            real_args = ()
            for arg in args:
                if isinstance(arg, Enum):
                    real_args += (arg.value,)
                else:
                    real_args += (arg,)
            result = real_func(*real_args)
            if issubclass(exposed_return_type, Enum):
                assert isinstance(result, int)
                return exposed_return_type(result)  # type: ignore
            else:
                return result

        wrapper.__name__ = func_name
        wrapper.__annotations__ = func_obj.__annotations__
        namespace[func_name] = wrapper
