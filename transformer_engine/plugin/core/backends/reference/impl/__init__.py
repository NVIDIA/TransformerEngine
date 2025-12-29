# Copyright (c) 2025, BAAI. All rights reserved.
#
# See LICENSE for license information.

from .gemm import general_gemm_torch

from .rmsnorm import rmsnorm_fwd_torch, rmsnorm_bwd_torch
from .normalization import layernorm_fwd_torch, layernorm_bwd_torch

from .activation import (
    gelu_torch, geglu_torch, qgelu_torch, qgeglu_torch,
    relu_torch, reglu_torch, srelu_torch, sreglu_torch,
    silu_torch, swiglu_torch, clamped_swiglu_torch,
    dgelu_torch, dgeglu_torch, dqgelu_torch, dqgeglu_torch,
    drelu_torch, dreglu_torch, dsrelu_torch, dsreglu_torch,
    dsilu_torch, dswiglu_torch, clamped_dswiglu_torch,
    dbias_dgelu_torch, dbias_dsilu_torch, dbias_drelu_torch,
    dbias_dqgelu_torch, dbias_dsrelu_torch,
)

from .softmax import (
    scaled_softmax_forward_torch,
    scaled_softmax_backward_torch,
    scaled_masked_softmax_forward_torch,
    scaled_masked_softmax_backward_torch,
    scaled_upper_triang_masked_softmax_forward_torch,
    scaled_upper_triang_masked_softmax_backward_torch,
    scaled_aligned_causal_masked_softmax_forward_torch,
    scaled_aligned_causal_masked_softmax_backward_torch,
)

from .dropout import dropout_fwd_torch, dropout_bwd_torch

from .optimizer import (
    multi_tensor_scale_torch,
    multi_tensor_l2norm_torch,
    multi_tensor_adam_torch,
    multi_tensor_sgd_torch,
    multi_tensor_compute_scale_and_scale_inv_torch,
)

__all__ = [
    "general_gemm_torch",
    "rmsnorm_fwd_torch",
    "rmsnorm_bwd_torch",
    "layernorm_fwd_torch",
    "layernorm_bwd_torch",
    "gelu_torch",
    "geglu_torch",
    "qgelu_torch",
    "qgeglu_torch",
    "relu_torch",
    "reglu_torch",
    "srelu_torch",
    "sreglu_torch",
    "silu_torch",
    "swiglu_torch",
    "clamped_swiglu_torch",
    "dgelu_torch",
    "dgeglu_torch",
    "dqgelu_torch",
    "dqgeglu_torch",
    "drelu_torch",
    "dreglu_torch",
    "dsrelu_torch",
    "dsreglu_torch",
    "dsilu_torch",
    "dswiglu_torch",
    "clamped_dswiglu_torch",
    "dbias_dgelu_torch",
    "dbias_dsilu_torch",
    "dbias_drelu_torch",
    "dbias_dqgelu_torch",
    "dbias_dsrelu_torch",
    "scaled_softmax_forward_torch",
    "scaled_softmax_backward_torch",
    "scaled_masked_softmax_forward_torch",
    "scaled_masked_softmax_backward_torch",
    "scaled_upper_triang_masked_softmax_forward_torch",
    "scaled_upper_triang_masked_softmax_backward_torch",
    "scaled_aligned_causal_masked_softmax_forward_torch",
    "scaled_aligned_causal_masked_softmax_backward_torch",
    "dropout_fwd_torch",
    "dropout_bwd_torch",
    "multi_tensor_scale_torch",
    "multi_tensor_l2norm_torch",
    "multi_tensor_adam_torch",
    "multi_tensor_sgd_torch",
    "multi_tensor_compute_scale_and_scale_inv_torch",
]
