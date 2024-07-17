# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Single tensor operations supported by the operation fuser."""

from .add_in_place import AddInPlace
from .all_gather import AllGather
from .all_reduce import AllReduce
from .basic_linear import BasicLinear
from .bias import Bias
from .identity import Identity
from .make_extra_output import MakeExtraOutput
from .reduce_scatter import ReduceScatter
from .reshape import Reshape
