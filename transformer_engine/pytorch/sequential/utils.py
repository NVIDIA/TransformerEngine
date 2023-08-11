from contextlib import contextmanager
from typing import Any


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
            setattr(obj, attr, old_value)
        else:
            delattr(obj, attr)
