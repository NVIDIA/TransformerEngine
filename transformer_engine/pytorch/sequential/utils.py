from __future__ import annotations
from typing import Any, Callable, Generic, Generator, Literal, TypeVar, overload
from types import TracebackType, ModuleType
from typing_extensions import ParamSpec

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
        next(self.gen)


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
        os.chdir(caller_path)
    except:
        old_cwd = None

    try:
        path = Path(file_path).resolve(strict=True)
        if not path.suffix:
            path = path.with_suffix(".py")

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
    if not all(isinstance(val, (str, type)) for val in arg_type_annotations):
        raise ValueError("Unsupported function (type annotations not supported)")
    else:
        arg_types = [
            ast.literal_eval(val) if isinstance(val, str) else val
            for val in arg_type_annotations
        ]
        if not all(isinstance(val, type) for val in arg_types):
            raise ValueError("Unsupported function (type annotations not supported)")

        return arg_types
