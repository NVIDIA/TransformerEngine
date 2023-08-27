from enum import Enum
from typing import (
    Any,
    Callable,
    Generic,
    Generator,
    Literal,
    Protocol,
    TypeVar,
    overload,
)
from types import GenericAlias, TracebackType, ModuleType
from typing_extensions import ParamSpec
import warnings

PS = ParamSpec("PS")
T = TypeVar("T")
ExcT = TypeVar("ExcT")


class _Context(Generic[PS, T]):
    def __init__(
        self,
        func: Callable[PS, Generator[T, None, None]],
        *args: PS.args,
        **kwargs: PS.kwargs,
    ):
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def __enter__(self):
        gen = self.func(*self.args, **self.kwargs)
        self.gen = gen
        return next(gen)

    def __exit__(
        self,
        exc_type: type[ExcT],
        exc_value: ExcT,
        exc_traceback: TracebackType,
    ):
        try:
            next(self.gen)
        except StopIteration:
            # Discard exception, it is expected
            pass


class contextmanager(Generic[PS, T]):
    def __init__(self, func: Callable[PS, Generator[T, None, None]]):
        self.func = func

    def __call__(self, *args: PS.args, **kwargs: PS.kwargs):
        return _Context(self.func, *args, **kwargs)


def cache(func: Callable[[], T]) -> Callable[[], T]:
    result = func()

    def wrapper():
        return result

    return wrapper


@overload
def import_file_as_module(
    file_path: str,
    run_module: bool = True,
    *,
    only_for_side_effects: Literal[False] = False,
) -> ModuleType:
    ...


@overload
def import_file_as_module(
    file_path: str,
    run_module: bool = True,
    *,
    only_for_side_effects: Literal[True] = True,
) -> None:
    ...


def import_file_as_module(
    file_path: str, run_module: bool = True, *, only_for_side_effects: bool = False
):
    if only_for_side_effects and not run_module:
        raise ValueError("Cannot import file for side effects only without running it!")

    from importlib.util import spec_from_loader, module_from_spec
    from importlib.machinery import SourceFileLoader
    from pathlib import Path
    import inspect
    import sys
    import os

    try:
        caller_path = Path(inspect.getframeinfo(sys._getframe(1))[0]).resolve(
            strict=True
        )
        old_cwd = os.getcwd()
        os.chdir(caller_path.parent)
    except:
        old_cwd = None

    try:
        path = Path(file_path)
        if not path.suffix:
            path = path.with_suffix(".py")
        path = path.resolve(strict=True)

        spec = spec_from_loader(path.name, SourceFileLoader(path.name, str(path)))
        if spec is None:
            raise ImportError(
                f'Failed to load file "{path}" as module: spec_from_loader returned None'
            )
        mod = module_from_spec(spec)
        if run_module:
            if spec.loader is None:
                raise ImportError(
                    f'Failed to run file "{path}" as module: spec_from_loader returned spec with a None loader'
                )
            spec.loader.exec_module(mod)
        if only_for_side_effects:
            return None
        else:
            return mod
    finally:
        if old_cwd is not None:
            os.chdir(old_cwd)


def get_arg_types(f: Callable[..., Any]) -> list[type]:
    import typing
    import ast

    annotations = typing.get_type_hints(f)
    annotations.pop("return", None)
    arg_type_annotations = tuple(annotations.values())

    arg_types = [
        ast.literal_eval(val) if isinstance(val, str) else val
        for val in arg_type_annotations
    ]

    return arg_types


def get_arg_names(f: Callable[..., Any]) -> list[str]:
    import typing

    annotations = typing.get_type_hints(f)
    annotations.pop("return", None)
    return list(annotations.keys())


def get_return_type(f: Callable[..., T]) -> type[T]:
    import typing
    import ast

    return_annotation = typing.get_type_hints(f)["return"]

    return_type = (
        ast.literal_eval(return_annotation)
        if isinstance(return_annotation, str)
        else return_annotation
    )

    return return_type  # type: ignore


class Decorator(Protocol):
    def __call__(self, f: Callable[PS, T]) -> Callable[PS, T]:
        ...


def cast(x: Any, _: type[T], /) -> T:
    return x


def set_name(name: str) -> Callable[..., Any]:
    def decorator(func: Callable[..., Any]):
        func.__name__ = name
        return func

    return decorator


def recursive_apply(
    func: Callable[[Any], Any],
    x: Any,
    pred: Callable[[Any], bool],
    on_false: Callable[[Any], Any] = lambda x: x,
) -> Any:
    if pred(x):
        return func(x)
    elif isinstance(x, list):
        return [func(y) for y in x]  # type: ignore
    elif isinstance(x, tuple):
        return tuple(func(y) for y in x)  # type: ignore
    elif isinstance(x, dict):
        return {k: func(v) for k, v in x.items()}  # type: ignore
    else:
        return on_false(x)


def torch_op(func: Callable[..., Any]):
    import torch
    from . import cpp_extensions

    version1: bool
    custom_ops = None
    try:
        custom_ops = torch._custom_ops  # type: ignore
        decl = custom_ops.custom_op  # type: ignore
        impl = custom_ops.impl  # type: ignore
        version1 = False
    except AttributeError:
        pass
    if custom_ops is None:
        try:
            custom_ops = torch._custom_op.impl  # type: ignore
            decl = custom_ops.custom_op  # type: ignore
            impl = custom_ops.CustomOp.impl  # type: ignore
            version1 = True
        except AttributeError:
            pass
    if custom_ops is None:
        if not hasattr(torch_op, "warned"):  # type: ignore
            torch_op.warned = True  # type: ignore
            warnings.warn("Unable to find custom_op, torch_op decorator has no effect")
        return func

    decl = cast(decl, Callable[[str], Decorator])  # type: ignore
    impl = cast(impl, Callable[[str], Decorator])  # type: ignore
    name = f"nvte::{func.__name__}"

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
            if arg_type is cpp_extensions.Tensor:
                w = f"{arg_name}_ = ({arg_name}.data, {arg_name}.amax, {arg_name}.scale, {arg_name}.scale_inv)\n"
                u = f"{arg_name} = {arg_type_name}(*{arg_name}_)\n"
            elif issubclass(arg_type, Enum):
                w = f"{arg_name}_ = {arg_name}.value\n"
                u = f"{arg_name} = {arg_type_name}({arg_name}_)\n"
            elif arg_type in [int, float, bool, str, torch.Tensor]:
                w = f"{arg_name}_ = {arg_name}\n"
                u = f"{arg_name} = {arg_name}_\n"
            else:
                raise NotImplementedError(arg_type_name)
            return (w, u)

        def wrap_type(arg_type: type):
            if arg_type is cpp_extensions.Tensor:
                return tuple[
                    int, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
                ]
            elif issubclass(arg_type, Enum):
                return int
            elif arg_type in [int, float, bool, str, torch.Tensor]:
                return arg_type
            else:
                raise NotImplementedError(arg_type_name)

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

        source = f"""\
import torch
from . import cpp_extensions

def {func.__name__}{inner_sig}:
    {arg_unwrapping_code}
    result = func({unwrapped_args})
    {result_wrapping_code}
    return result_

def outer_wrapper{outer_sig}:
    {arg_wrapping_code}
    result_ = {func.__name__}({wrapped_args})
    {result_unwrapping_code}
    return result
"""
        ns = dict(func=func, __name__=__name__)
        try:
            exec(source, ns)
            declared = decl(name)(ns[func.__name__])
            if version1:
                declared.impl("cuda")(ns[func.__name__])  # type: ignore
            else:
                impl(name)(ns[func.__name__])  # type: ignore
        except Exception as e:
            raise RuntimeError(
                f"Failed to compile wrapper for {func.__name__}. Generated code: \n{source}"
            ) from e

        outer_wrapper = ns["outer_wrapper"]
        return outer_wrapper

    return make_wrapper(func)
