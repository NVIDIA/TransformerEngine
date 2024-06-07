# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""The utilities for Transformer Engine"""
import inspect
import warnings
import functools
import transformer_engine.pytorch.cpp_extensions as ext
from enum import Enum
from typing import Tuple

warnings.filterwarnings(
    "module", category=DeprecationWarning, module="transformer_engine.common.utils")


class DeprecatedEnum:    # pylint: disable=too-few-public-methods
    """DeprecatedEnum"""

    def __init__(self, enum_cls, msg):
        self.enum_cls = enum_cls
        self.msg = msg

    def __iter__(self):
        return iter(list(self.enum_cls.__members__.values()))

    def __getattr__(self, name):
        if name in self.enum_cls.__members__:
            warnings.warn(self.msg, DeprecationWarning)
            return self.enum_cls.__members__[name]
        raise AttributeError(f"{self.enum_cls} does not contain {name}")


def deprecate_wrapper(obj, msg):
    """Deprecate wrapper"""
    if inspect.isclass(obj):
        if issubclass(obj, Enum):
            return DeprecatedEnum(obj, msg)

        class DeprecatedCls(obj):    # pylint: disable=too-few-public-methods
            """DeprecatedCls"""

            def __init__(self, *args, **kwargs):
                warnings.warn(msg, DeprecationWarning)
                super().__init__(*args, **kwargs)

        return DeprecatedCls

    if inspect.isfunction(obj):

        def deprecated(*args, **kwargs):
            warnings.warn(msg, DeprecationWarning)
            return obj(*args, **kwargs)

        return deprecated

    raise NotImplementedError(
        f"deprecate_cls_wrapper only support Class and Function, but got {type(obj)}.")


@functools.cache
def get_cudnn_version() -> Tuple[int, int, int]:
    """Runtime cuDNN version (major, minor, patch)"""
    encoded_version = ext.get_cudnn_version()
    major_version_magnitude = 1000 if encoded_version < 90000 else 10000
    major, encoded_version = divmod(encoded_version, major_version_magnitude)
    minor, patch = divmod(encoded_version, 100)
    return (major, minor, patch)
