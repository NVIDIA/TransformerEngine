# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Fusible operations.

This operation-based API is experimental and subject to change.

"""

from .basic import *
from .fuser import register_backward_fusion, register_forward_fusion
from .linear import Linear
from .op import BasicOperation, FusedOperation, FusibleOperation
from .sequential import Sequential

import transformer_engine.pytorch.ops.fused
