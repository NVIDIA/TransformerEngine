from enum import Enum
import inspect
from typing import Any, Callable
from ..utils import import_file_as_module, get_return_type
import torch
from torch._ops import OpOverloadPacket, _OpNamespace  # type: ignore
from torch._classes import _ClassNamespace  # type: ignore
from torch._C import ScriptClass  # type: ignore

try:
    # Normally, torch.classes.load_library would be used
    # to load the classes from the module.
    # However, that requires knowing where the module is.
    # A simpler way is to just import it.
    import transformer_engine_cuda  # type: ignore
except:
    # The import will always fail, as torch libraries
    # are not supposed to be imported directly.
    # However, it does achieve the effect of loading the classes.
    pass


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

    function_ns = torch.ops.transformer_engine_cuda  # type: ignore
    assert isinstance(function_ns, _OpNamespace)
    type_ns = torch.classes.transformer_engine_cuda  # type: ignore
    assert isinstance(type_ns, _ClassNamespace)

    real_function: Callable[[str], OpOverloadPacket] = lambda name: getattr(
        function_ns, name
    )
    real_type: Callable[[str], ScriptClass] = lambda name: getattr(type_ns, name)  # type: ignore

    for enum_name in enum_names:
        namespace[enum_name] = stub_types[enum_name]

    for class_name in stub_types.keys() - enum_names:
        namespace[class_name] = real_type(class_name)

    for func_name, func_obj in stub_functions.items():
        exposed_return_type: type = get_return_type(func_obj)

        def make_wrapper(real_func: Any):
            def wrapper(*args: Any) -> Any:
                real_args = ()
                for arg in args:
                    if isinstance(arg, Enum):
                        real_args += (arg.value,)
                    else:
                        real_args += (arg,)
                result: Any = real_func(*real_args)
                if issubclass(exposed_return_type, Enum):
                    assert isinstance(result, int)
                    return exposed_return_type(result)  # type: ignore
                else:
                    return result

            return wrapper

        wrapper = make_wrapper(real_function(func_name))

        wrapper.__name__ = func_name
        wrapper.__annotations__ = func_obj.__annotations__
        namespace[func_name] = wrapper


inject_real(globals())
