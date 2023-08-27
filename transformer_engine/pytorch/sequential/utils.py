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
from types import TracebackType, ModuleType
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

    dec = None
    try:
        dec = torch._custom_ops.custom_op  # type: ignore
    except AttributeError:
        pass
    if dec is None:
        try:
            dec = torch._custom_op.impl.custom_op  # type: ignore
        except AttributeError:
            pass

    if dec is None:
        if not hasattr(torch_op, "warned"):  # type: ignore
            torch_op.warned = True  # type: ignore
            warnings.warn("Unable to find custom_op, torch_op decorator has no effect")
        return func

    dec = cast(dec, Callable[[str], Decorator])  # type: ignore
    name = f"nvte::{func.__name__}"

    def make_wrapper(func: Callable[..., Any]):
        storage: dict[int, Any] = {}

        def wrap(x: Any) -> Any:
            def _(x: cpp_extensions.Tensor | Enum | Any):
                if isinstance(x, cpp_extensions.Tensor):
                    result = (x.data, x.amax, x.scale, x.scale_inv)
                elif isinstance(x, Enum):
                    result = x.value
                else:
                    result = x
                storage[id(result)] = x
                return result

            return recursive_apply(
                _,
                x,
                lambda x: isinstance(
                    x,
                    cpp_extensions.Tensor
                    | cpp_extensions.DType
                    | cpp_extensions.BiasType
                    | cpp_extensions.FusedAttnBackend
                    | cpp_extensions.QKVLayout
                    | cpp_extensions.MaskType,
                ),
            )

        def wrap_type(x: Any) -> str:
            return recursive_apply(
                lambda _: "tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]",
                x,
                lambda x: x is cpp_extensions.Tensor,
                lambda x: f"{x.__module__}.{x.__name__}",
            )

        def unwrap(x: Any) -> Any:
            return recursive_apply(
                lambda x: storage[id(x)],
                x,
                lambda x: id(x) in storage,  # type: ignore
            )

        arg_types = get_arg_types(func)
        return_type = get_return_type(func)

        wrapped_arg_types = [wrap_type(t) for t in arg_types]

        template = f"""\
import torch
def {func.__name__}({",".join(f"{arg_name}: {arg_type_name}" for arg_name, arg_type_name in zip(get_arg_names(func), wrapped_arg_types))}) -> {wrap_type(return_type)}:
    unwrapped = unwrap(({",".join(f"{arg_name}" for arg_name in get_arg_names(func))}))
    result = func(*unwrapped)
    return wrap(result)
"""

        ns = dict(func=func, wrap=wrap, unwrap=unwrap)
        exec(template, ns)
        wrapper1 = dec(name)(ns[func.__name__])

        def wrapper2(*args: Any):
            storage.clear()
            wrapped = wrap(args)
            result = wrapper1(*wrapped)
            return unwrap(result)

        return wrapper2

    return make_wrapper(func)
