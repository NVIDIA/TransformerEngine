from typing import Any, Callable, Generic, Generator, TypeVar
from types import TracebackType
from typing_extensions import ParamSpec

PS = ParamSpec("PS")
T = TypeVar("T")
ExcT = TypeVar("ExcT")


class __Context:
    def __init__(
        self,
        func: Callable[PS, Generator[T, None, None]],
        *args: PS.args,
        **kwargs: PS.kwargs
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
        return __Context(self.func, *args, **kwargs)


def cache(func: Callable[[], T]) -> Callable[[], T]:
    result = func()

    def wrapper():
        return result

    return wrapper


@contextmanager
def set_attribute(obj: object, attr: str, value: Any):
    """Set an attribute on an object, and reset it to its original value when the context manager exits."""
    had_value = hasattr(obj, attr)
    if had_value:
        old_value = getattr(obj, attr)
    setattr(obj, attr, value)
    try:
        yield
    finally:
        if had_value:
            setattr(obj, attr, old_value)  # type:ignore
        else:
            delattr(obj, attr)
