# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Compound tensor operation supported by the operation fuser."""

from ..fuser import register_backward_fusion, register_forward_fusion
from .backward_activation_bias import BackwardActivationBias
from .backward_add_rmsnorm import BackwardAddRMSNorm
from .backward_linear_add import BackwardLinearAdd
from .backward_linear_scale import BackwardLinearScale
from .forward_linear_bias_activation import ForwardLinearBiasActivation
from .forward_linear_bias_add import ForwardLinearBiasAdd
from .forward_linear_scale_add import ForwardLinearScaleAdd
from .userbuffers_backward_linear import UserbuffersBackwardLinear
from .userbuffers_forward_linear import UserbuffersForwardLinear


# Register forward fusions
register_forward_fusion(UserbuffersForwardLinear.fuse_forward_ops)
register_forward_fusion(ForwardLinearBiasAdd.fuse_forward_ops)
register_forward_fusion(ForwardLinearBiasActivation.fuse_forward_ops)
register_forward_fusion(ForwardLinearScaleAdd.fuse_forward_ops)

# Register backward fusions
register_backward_fusion(UserbuffersBackwardLinear.fuse_backward_ops)
register_backward_fusion(BackwardLinearAdd.fuse_backward_ops)
register_backward_fusion(BackwardLinearScale.fuse_backward_ops)
register_backward_fusion(BackwardActivationBias.fuse_backward_ops)
register_backward_fusion(BackwardAddRMSNorm.fuse_backward_ops)

# Import experimental fusions
# Note: Registration logic is non-trivial, so submodule handles it internally.
from .forward_grouped_mlp import (  # pylint: disable=wrong-import-position
    ForwardGroupedMLP_CuTeGEMMSwiGLU_MXFP8,
)
from .backward_grouped_mlp import (  # pylint: disable=wrong-import-position
    BackwardGroupedMLP_CuTeGEMMDSwiGLU_MXFP8,
)
