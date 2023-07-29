from dataclasses import dataclass
import functools
from typing import TYPE_CHECKING, Any, ParamSpec, TypeVar, Callable, Sequence, overload
import typing
from pydantic import ValidationError, parse_obj_as


DispatchSet = dict[tuple[type, ...], Callable[..., Any]]
RT = TypeVar("RT")


@dataclass
class _Metadata:
    dispatch_sets = dict[str, DispatchSet]()
    wrappers = dict[str, Callable[..., Any]]()
    memo = dict[str, dict[tuple[type, ...], Callable[..., Any]]]()


def _get_metadata(func: Callable[..., Any]) -> _Metadata:
    del func  # use main module globals to work accross modules
    module_dict = __import__("__main__").__dict__
    if "__multiple_dispatch__" not in module_dict:
        module_dict["__multiple_dispatch__"] = _Metadata()
    assert isinstance(module_dict["__multiple_dispatch__"], _Metadata)
    return module_dict["__multiple_dispatch__"]


def _get_dispatch_set(func: Callable[..., Any]):
    dispatch_set_dict = _get_metadata(func).dispatch_sets

    if func.__qualname__ not in dispatch_set_dict:
        dispatch_set_dict[func.__qualname__] = DispatchSet()
    assert isinstance(dispatch_set_dict[func.__qualname__], dict)
    return dispatch_set_dict[func.__qualname__]


def _get_memo(func: Callable[..., Any]):
    memos = _get_metadata(func).memo
    if func.__qualname__ not in memos:
        memos[func.__qualname__] = {}
    return memos[func.__qualname__]


def _generic_isinstance(obj: Any, type_: type) -> bool:
    try:
        return isinstance(obj, type_)
    except TypeError:
        try:
            parse_obj_as(type_, obj)
            return True
        except ValidationError:
            return False
        except TypeError:
            print((obj, type_))
            raise


def _make_wrapper(func: Callable[..., Any]):
    @functools.wraps(func)
    def wrapper(*args: Any):
        memo = _get_memo(func)
        types = tuple(type(arg) for arg in args)
        if types not in memo:
            dispatch_set = _get_dispatch_set(func)
            candidates = [
                (arg_types, candidate)
                for arg_types, candidate in dispatch_set.items()
                if len(tuple(arg_types)) == len(args)
                and all(
                    _generic_isinstance(arg, type_)
                    for arg, type_ in zip(args, arg_types)
                )
            ]

            scores = list[int]()
            for arg_types, _ in candidates:
                max_index = 0
                for arg, type_ in zip(args, arg_types):
                    if TYPE_CHECKING:  # checker bug
                        ancestors: Sequence[type] = ...  # type: ignore
                    else:
                        ancestors: Sequence[type] = type(arg).__mro__

                    type_index = 0
                    for i, ancestor in enumerate(ancestors):
                        if ancestor is type_:
                            type_index = i
                            break
                    max_index = max(max_index, type_index)
                scores.append(max_index)
            scored_candidates = sorted(zip(scores, candidates))

            if len(scored_candidates) == 0:

                def error(*_: Any):
                    raise TypeError(f"no dispatch for {func.__qualname__} with {types}")

                memo[types] = error
            elif len(scored_candidates) == 1:
                memo[types] = scored_candidates[0][1][1]
            elif scored_candidates[0][0] == scored_candidates[1][0]:

                def error(*_: Any):
                    raise TypeError(
                        f"ambiguous dispatch for {func.__qualname__} with {types}"
                    )

                memo[types] = error
            else:
                memo[types] = scored_candidates[0][1][1]

        return memo[types](*args)

    return wrapper


def _get_wrapper(func: Callable[..., Any]):
    wrapper_dict = _get_metadata(func).wrappers
    if func.__qualname__ not in wrapper_dict:
        wrapper_dict[func.__qualname__] = _make_wrapper(func)
    return wrapper_dict[func.__qualname__]


PS = ParamSpec("PS")


def _decorator(register: bool, func: Callable[PS, RT]) -> Callable[PS, RT]:
    if register:
        disset = _get_dispatch_set(func)
        annotations = typing.get_type_hints(func)
        annotations.pop("return", None)
        arg_types = tuple(annotations.values())
        if arg_types in disset:
            raise ValueError(f"Duplicate dispatch for {func.__qualname__}")
        disset[arg_types] = func

    return _get_wrapper(func)  # type: ignore


@overload
def multiple_dispatch(func: Callable[PS, RT], /) -> Callable[PS, RT]:
    ...


@overload
def multiple_dispatch(
    register: bool, /
) -> Callable[[Callable[PS, RT]], Callable[PS, RT]]:
    ...


def multiple_dispatch(
    x: Callable[PS, RT] | bool, /
) -> Callable[PS, RT] | Callable[[Callable[PS, RT]], Callable[PS, RT]]:
    if isinstance(x, bool):
        return functools.partial(_decorator, x)
    else:
        return _decorator(True, x)
