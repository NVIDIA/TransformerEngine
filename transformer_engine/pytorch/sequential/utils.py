from __future__ import annotations
import ast
from typing import (
    Any,
    Callable,
    Generic,
    Generator,
    Literal,
    Mapping,
    Protocol,
    Sequence,
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
    "TorchDynamo-compatible replacement for `contextlib.contextmanager`"

    def __init__(self, func: Callable[PS, Generator[T, None, None]]):
        self.func = func

    def __call__(self, *args: PS.args, **kwargs: PS.kwargs):
        return _Context(self.func, *args, **kwargs)


def cache(func: Callable[[], T]) -> Callable[[], T]:
    "TorchDynamo-compatible replacement for `functools.cache`"
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

    try:
        return_annotation = typing.get_type_hints(f)["return"]
    except KeyError as e:
        raise ValueError(f"{f} must have an annotated return type") from e

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
        pref_code = f"\ndef unrolled_{iterations}(f, iterable, loop_state):\n"
        pref_code += INDENT + "iterator = iter(iterable)\n"
        iter_code = INDENT + "item = next(iterator)\n"
        return_type = get_return_type(f)
        if unpack:
            if return_type is NoneType:
                iter_code += INDENT + "f(*item)\n"
            else:
                iter_code += INDENT + "loop_state = f(*item, loop_state)\n"
        else:
            if return_type is NoneType:
                iter_code += INDENT + "f(item)\n"
            else:
                iter_code += INDENT + "loop_state = f(item, loop_state)\n"
        sufx_code = ""
        namespace: dict[str, Any] = {}
        full_code = pref_code + iter_code * iterations + sufx_code
        exec_saving_source(full_code, namespace)
        unrolled_loop = namespace[f"unrolled_{iterations}"]
        memo[(iterations, unpack, return_type is not NoneType)] = unrolled_loop
        return lambda iterable, loop_state: unrolled_loop(f, iterable, loop_state)

    return decorator


def get_globals(o: object) -> dict[str, Any]:
    "Returns the same object that `globals()` would return inside the provided object."
    try:
        return o.__globals__  # type: ignore
    except:
        pass
    try:
        import sys

        return sys.modules[o.__module__].__dict__
    except:
        pass
    raise ValueError(f"Cannot get globals for {o}")


class MacroVar(Generic[T]):
    "A `TypeVar`-like object representing a `macro`'s parameter."

    def __new__(cls, name: str, type_: type[T] = object) -> T:
        return (name, type_)  # type: ignore


class _MacroTransformer(ast.NodeTransformer):
    def __init__(self, names: Sequence[str], values: Sequence[Any]) -> None:
        if not len(names) == len(values):
            raise ValueError(f"Length mismatch: {len(names)} != {len(values)}")
        for name in names:
            if not name.isidentifier():
                raise ValueError(f"Invalid identifier: {name}")
            if not names.count(name) == 1:
                raise ValueError(f"Duplicate identifier: {name}")
        for value in values:
            try:
                constant = ast.Constant(value=value)
                source = ast.unparse(constant)
                reconstructed = ast.literal_eval(source)
                if not reconstructed == value:
                    raise ValueError(
                        f"Cannot reconstruct value after serialization: {value}"
                    )
            except Exception as e:
                raise ValueError(f"Cannot serialize value: {value}") from e
        self.names = names
        self.values = values

    def visit_Name(self, node: ast.Name):
        if node.id in self.names:
            idx = self.names.index(node.id)
            value = self.values[idx]
            return ast.Constant(value=value)
        else:
            return node


def macro(
    *substitutions: Unpack[Ts], textual: bool = True
) -> Callable[[T], Callable[[Unpack[Ts]], T]]:
    """
    This decorator functions like a C-like macro definition.
    It can be applied to a function or class definition.
    It is to be used together with `MacroVar`s - `TypeVar`-like
    objects representing the macro's parameters.

    Example declaration:
    ```
    X = MacroVar("X", int)
    @macro(X)
    def f():
        return X
    ```
    The above macro can then be used like this:
    ```
    f1 = f(1)
    assert f1() == 1
    ```

    The `textual` (default `True`) argument controls if the
    instantiations of the macro should have their ASTs modified
    in place (textual) or if they should have the same source,
    and instead have the provided values injected as constant
    globals into their namespace.

    For example, with `textual=True`, `inspect.getsource(f1)` outputs:
    ```
    def f():
        return 1
    ```
    With `textual=False`, `inspect.getsource(f1)` outputs:
    ```
    def f():
        return X
    ```
    The `textual=False` mode is needed when `eval(repr(X))` fails.
    In this mode, the substituted values `are` (Python `is`) the
    original provided values.

    It can be used, for example, to sidestep some limitations of TorchDynamo.
    """
    names: list[str] = [name for name, _ in substitutions]  # type: ignore
    for name in names:
        assert name.isidentifier()
        assert names.count(name) == 1

    if textual:

        def textual_decorator(definition: T) -> Callable[[Unpack[Ts]], T]:
            import inspect
            import ast

            try:
                source = inspect.getsource(definition)  # type: ignore
                # Source includes the decorator, remove it
                source = source[source.find("\n") + 1 :].strip()
                ast_tree = ast.parse(source)
            except OSError:
                raise ValueError("Cannot get source code of definition")

            def macro_impl(*values: Unpack[Ts]) -> T:
                _MacroTransformer(names, values).visit(ast_tree)
                ast.fix_missing_locations(ast_tree)
                source = ast.unparse(ast_tree)
                try:
                    globals_ = get_globals(definition).copy()
                    del globals_[definition.__name__]  # type: ignore
                except:
                    globals_ = {}
                exec_saving_source(source, globals_)
                return globals_[definition.__name__]  # type: ignore

            return macro_impl

        return textual_decorator
    else:

        def injection_decorator(definition: T) -> Callable[[Unpack[Ts]], T]:
            import inspect

            try:
                source = inspect.getsource(definition)  # type: ignore
                # Source includes the decorator, remove it
                source = source[source.find("\n") + 1 :].strip()
            except OSError:
                raise ValueError("Cannot get source code of definition")

            def macro_impl(*values: Unpack[Ts]) -> T:
                try:
                    globals_ = get_globals(definition).copy()
                    del globals_[definition.__name__]  # type: ignore
                except:
                    globals_ = {}
                for name, value in zip(names, values):
                    globals_[name] = value
                exec_saving_source(source, globals_)
                return globals_[definition.__name__]  # type: ignore

            return macro_impl

        return injection_decorator


def prevent_import(*names: str):
    prev_import = __builtins__.__dict__["__import__"]

    def restricted_import(
        name: str,
        globals: dict[str, Any] | None = None,
        locals: dict[str, Any] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ):
        assert (
            name not in names
        ), f"Cannot import {name} from this module. See `ARCHITECTURE.md` for more information."

        return prev_import(name, globals, locals, fromlist, level)

    __builtins__.__dict__["__import__"] = restricted_import


@overload
def is_generic(t: type) -> Literal[False]:
    ...


@overload
def is_generic(t: GenericAlias) -> Literal[True]:
    ...


def is_generic(t: type | GenericAlias):
    """
    Returns True if the type is a generic type, False otherwise.
    This is useful for checking if `isinstance` would fail with
    a `TypeError` when called with a generic type.
    """
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
    "macro",
    "MacroVar",
]
