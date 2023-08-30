from typing import (
    Any,
    Callable,
    Generic,
    Generator,
    Literal,
    Protocol,
    TypeVar,
    Union,
    overload,
    Iterable,
)
from types import TracebackType, ModuleType, GenericAlias
from typing_extensions import ParamSpec, TypeVarTuple, Unpack

PS = ParamSpec("PS")
T = TypeVar("T")
Ts = TypeVarTuple("Ts")
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


def exec_saving_source(source: str, globals: dict[str, Any]):
    import ast
    import linecache

    if not hasattr(exec_saving_source, "sources"):
        old_getlines = linecache.getlines
        sources: list[str] = []

        def patched_getlines(filename: str, module_globals: Any = None):
            if "<exec#" in filename:
                index = int(filename.split("#")[1].split(">")[0])
                return sources[index].splitlines(True)
            else:
                return old_getlines(filename, module_globals)

        linecache.getlines = patched_getlines
        setattr(exec_saving_source, "sources", sources)
    sources: list[str] = getattr(exec_saving_source, "sources")
    exec(
        compile(ast.parse(source), filename=f"<exec#{len(sources)}>", mode="exec"),
        globals,
    )
    sources.append(source)


@overload
def unrolled_for(
    iterable_: Iterable[tuple[Unpack[Ts]]],
) -> Callable[[Callable[[Unpack[Ts]], None | dict[str, Any]]], None]:
    ...


@overload
def unrolled_for(
    iterable_: Iterable[T],
) -> Callable[[Callable[[T], None | dict[str, Any]]], None]:
    ...


def unrolled_for(
    iterable_: Iterable[T] | Iterable[tuple[Unpack[Ts]]],
) -> (
    Callable[[Callable[[T], None | dict[str, Any]]], None]
    | Callable[[Callable[[Unpack[Ts]], None | dict[str, Any]]], None]
):
    def decorator(
        f: Callable[[T], None | dict[str, Any]]
        | Callable[[Unpack[Ts]], None | dict[str, Any]]
    ):
        loop_state: None | dict[str, Any] = None
        for item in iterable_:
            if isinstance(item, tuple):
                if loop_state is None:
                    loop_state = f(*item)  # type: ignore
                else:
                    loop_state = f(*item, **loop_state)  # type: ignore
            else:
                if loop_state is None:
                    loop_state = f(item)  # type: ignore
                else:
                    loop_state = f(item, **loop_state)  # type: ignore

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
