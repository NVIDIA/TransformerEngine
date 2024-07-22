# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Single tensor operations supported by the operation fuser."""

from .all_gather import AllGather
from .all_reduce import AllReduce
from .basic_linear import BasicLinear
from .bias import Bias
from .cast_float8 import CastFloat8
from .identity import Identity
from .layer_norm import LayerNorm
from .reduce_scatter import ReduceScatter
from .reshape import Reshape
from .rmsnorm import RMSNorm
