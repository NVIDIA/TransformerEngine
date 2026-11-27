# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Python representation of Transformer Engine's C++ data types."""

from enum import IntEnum


class DType(IntEnum):
    """Transformer Engine data types, exposed as ``transformer_engine.pytorch.DType``.

    Values match ``transformer_engine::DType``. The PyTorch extension validates
    every name and value when it loads and caches these enum members for C++
    return values. Comparisons and hashing use the native integer operations.

    Use ``DType(value)`` to explicitly convert a serialized integer tag.
    Dtype arguments to C++ bindings require a member of this enum.
    """

    kByte = 0
    kInt16 = 1
    kInt32 = 2
    kInt64 = 3
    kFloat32 = 4
    kFloat16 = 5
    kBFloat16 = 6
    kFloat8E4M3 = 7
    kFloat8E5M2 = 8
    kFloat8E8M0 = 9
    kFloat4E2M1 = 10
    kFloat8UE5M3 = 11
