import inspect
import warnings
from enum import Enum

warnings.simplefilter('default')

def deprecate_wrapper(obj, msg):

    if inspect.isclass(obj):
        if issubclass(obj, Enum):
            class Deprecated:
                def __init__(self, enum_cls):
                    self.enum_cls = enum_cls

                def __getattr__(self, name):
                    if name in self.enum_cls.__members__:
                        warnings.warn(msg, DeprecationWarning)
                        return self.enum_cls.__members__[name]
                    raise AttributeError(f"{self.enum_cls} does not contain {name}")

            return Deprecated(obj)
        else:
            class Deprecated(obj):
                def __init__(self, *args, **kwargs):
                    warnings.warn(msg, DeprecationWarning)
                    super().__init__(*args, **kwargs)

            return Deprecated
    elif inspect.isfunction(obj):

        def deprecated(*args, **kwargs):
            warnings.warn(msg, DeprecationWarning)
            return obj(*args, **kwargs)

        return deprecated
    else:
        raise NotImplementedError(
        f"deprecate_cls_wrapper only support Class and Function, but got {type(obj)}.")