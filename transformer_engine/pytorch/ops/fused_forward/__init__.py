# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Compound tensor operation supported by the operation fuser."""

from .linear_bias_activation import (
    ForwardLinearBiasActivation,
    fuse_forward_linear_bias_activation,
)
