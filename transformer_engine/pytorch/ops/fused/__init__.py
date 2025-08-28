# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Compound tensor operation supported by the operation fuser."""

from .backward_activation_bias import (
    BackwardActivationBias,
    fuse_backward_activation_bias,
)
from .backward_add_rmsnorm import (
    BackwardAddRMSNorm,
    fuse_backward_add_rmsnorm,
)
from .backward_linear_add import (
    BackwardLinearAdd,
    fuse_backward_linear_add,
)
from .backward_linear_scale import (
    BackwardLinearScale,
    fuse_backward_linear_scale,
)
from .forward_linear_bias_activation import (
    ForwardLinearBiasActivation,
    fuse_forward_linear_bias_activation,
)
from .forward_linear_bias_add import (
    ForwardLinearBiasAdd,
    fuse_forward_linear_bias_add,
)
from .forward_linear_scale_add import (
    ForwardLinearScaleAdd,
    fuse_forward_linear_scale_add,
)
from .userbuffers_backward_linear import (
    UserbuffersBackwardLinear,
    fuse_userbuffers_backward_linear,
)
from .userbuffers_forward_linear import (
    UserbuffersForwardLinear,
    fuse_userbuffers_forward_linear,
)
