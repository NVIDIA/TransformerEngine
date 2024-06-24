# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Fused optimizers and multi-tensor kernels."""
from transformer_engine_torch import (
    multi_tensor_scale,
    multi_tensor_l2norm,
    multi_tensor_unscale_l2norm,
    multi_tensor_adam,
    multi_tensor_adam_capturable,
    multi_tensor_adam_capturable_master,
    multi_tensor_sgd,
)
from .fused_adam import FusedAdam
from .fused_sgd import FusedSGD
from .multi_tensor_apply import MultiTensorApply, multi_tensor_applier
