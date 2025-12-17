# Copyright (c) 2025, BAAI. All rights reserved.
#
# See LICENSE for license information.

import os
import torch
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from ..import_utils import safety_import

### RMSNORM
rmsnorm_fwd_fl = safety_import('transformer_engine.plugins.cpp_extensions', 'rmsnorm_fwd_fl')

def apply_normalization_fl(
    inputmat: torch.Tensor,
    ln_out: torch.Tensor,
    ln_weight: torch.Tensor,
    ln_bias: Union[torch.Tensor, None],
    eps: float,
    output_quantizer,
    output_dtype,
    normalization: str,
    fwd_ln_sm_margin: int,
    zero_centered_gamma: bool,
):
    assert normalization == "RMSNorm", "Triton-based LayerNorm is not supported in TE-FL"
    assert ln_bias is None, "Triton-Based RMSNorm do not support bias"
    normalization_func = rmsnorm_fwd_fl
    return normalization_func(
        inputmat,
        ln_weight,
        eps,
        ln_out,
        output_quantizer,
        output_dtype,
        fwd_ln_sm_margin,
        zero_centered_gamma,
    )
