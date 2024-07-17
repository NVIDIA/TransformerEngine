# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Fusible operations.

This operation-based API is experimental and subject to change.

"""

from transformer_engine.pytorch.ops.basic import (
    AddInPlace,
    AllGather,
    AllReduce,
    BasicLinear,
    Bias,
    Identity,
    MakeExtraOutput,
    ReduceScatter,
    Reshape,
)
from transformer_engine.pytorch.ops.linear import Linear
from transformer_engine.pytorch.ops.op import FusibleOperation
from transformer_engine.pytorch.ops.sequential import Sequential
