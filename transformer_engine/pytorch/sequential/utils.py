from __future__ import annotations
from typing import (
    Any,
    Callable,
    Generic,
    Generator,
    Literal,
    Mapping,
    Protocol,
    Sized,
    TypeVar,
    overload,
    Iterable,
)
from types import NoneType, TracebackType, ModuleType, GenericAlias
from typing_extensions import ParamSpec, TypeVarTuple, Unpack
from .exec_saving_source import exec_saving_source

PS = ParamSpec("PS")
T = TypeVar("T")
Ts = TypeVarTuple("Ts")
Ts2 = TypeVarTuple("Ts2")
CT = TypeVar("CT", covariant=True)
ExcT = TypeVar("ExcT")
SomeDict = TypeVar("SomeDict", bound=Mapping[Any, Any], covariant=True)


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


class SizedIterable(Sized, Iterable[CT], Protocol):
    pass


class enumerate(enumerate[T]):
    def __init__(self, iterable: Iterable[T], start: int = 0) -> None:
        if isinstance(iterable, Sized):
            self.__len__ = lambda: len(iterable)
        super().__init__(iterable, start)

    def __len__(self) -> int:
        ...


def unrolled_for(
    iterations: int,
) -> Callable[
    [Callable[[Unpack[Ts], SomeDict], SomeDict]],
    Callable[[Iterable[tuple[Unpack[Ts]]], SomeDict], None],
]:
    if not hasattr(unrolled_for, "memo"):
        setattr(unrolled_for, "memo", {})
    memo: dict[tuple[int, bool, bool], Callable[..., Any]] = getattr(
        unrolled_for, "memo"
    )

    def decorator(
        f: Callable[[Unpack[Ts], SomeDict], SomeDict]
    ) -> Callable[[Iterable[tuple[Unpack[Ts]]], SomeDict], None]:
        import inspect

        unpack = len(inspect.getfullargspec(f).args) > 1
        INDENT = " " * 4
        pref_code = f"def unrolled_{iterations}(f, iterable, loop_state):\n"
        pref_code += INDENT + "iterator = iter(iterable)\n"
        iter_code = INDENT + "item = next(iterator)\n"
        return_type = get_return_type(f)
        if unpack:
            if return_type is NoneType:
                iter_code += INDENT + "f(*item)\n"
            else:
                iter_code += INDENT + "loop_state = f(*item, **loop_state)\n"
        else:
            if return_type is NoneType:
                iter_code += INDENT + "f(item)\n"
            else:
                iter_code += INDENT + "loop_state = f(item, **loop_state)\n"
        sufx_code = "\n"
        namespace: dict[str, Any] = {}
        full_code = pref_code + iter_code * iterations + sufx_code
        exec_saving_source(full_code, namespace)
        unrolled_loop = namespace[f"unrolled_{iterations}"]
        memo[(iterations, unpack, return_type is not NoneType)] = unrolled_loop
        return lambda iterable, loop_state: unrolled_loop(f, iterable, loop_state)

    return decorator


class Decorator(Protocol[Unpack[Ts], T]):
    def __call__(self, f: Callable[[Unpack[Ts]], T]) -> Callable[[Unpack[Ts]], T]:
        ...


@overload
def is_generic(t: type) -> Literal[False]:
    ...


@overload
def is_generic(t: GenericAlias) -> Literal[True]:
    ...


def is_generic(t: type | GenericAlias):
    from types import GenericAlias
    from typing import _SpecialGenericAlias, _GenericAlias  # type: ignore

    return isinstance(t, GenericAlias | _SpecialGenericAlias | _GenericAlias)


__all__ = [
    "contextmanager",
    "cache",
    "import_file_as_module",
    "exec_saving_source",
    "unrolled_for",
    "is_generic",
    "get_arg_names",
    "get_arg_types",
    "get_return_type",
]
