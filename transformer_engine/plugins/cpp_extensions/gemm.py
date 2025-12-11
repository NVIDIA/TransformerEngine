# Copyright (c) 2025, BAAI. All rights reserved.
#
# See LICENSE for license information.

from typing import Iterable, Optional, Tuple, Union, List
import os
import functools
import torch
import transformer_engine_torch as tex
from transformer_engine.pytorch.constants import TE_DType

from transformer_engine.pytorch.quantized_tensor import Quantizer

from ..import_utils import have_flag_gems

HAVE_FLAG_GEMS = have_flag_gems()
if HAVE_FLAG_GEMS:
    import flag_gems

__all__ = [
    "general_gemm_fl",
]


def validate_gemm_scale(scale: Optional[float], required: bool) -> float:
    """Validate whether a GEMM scaling factor is consistent with its usage"""
    if required:
        return scale if scale is not None else 1.0
    if scale not in (0.0, None):
        raise ValueError("scale must be zero")
    return 0.0


def general_gemm_fl(
    A: torch.Tensor,
    B: torch.Tensor,
    out_dtype: Optional[torch.dtype] = None,
    quantization_params: Optional[Quantizer] = None,
    gelu: bool = False,
    gelu_in: torch.Tensor = None,
    alpha: float = 1.0,
    beta: Optional[float] = None,
    accumulate: bool = False,
    layout: str = "TN",
    out: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    use_split_accumulator: bool = False,
    grad: bool = False,
    ub: Union[tex.CommOverlap, tex.CommOverlapP2P] = None,
    ub_type: tex.CommOverlapType = None,
    extra_output: Optional[torch.Tensor] = None,
    bulk_overlap: bool = False,
) -> Iterable[Optional[torch.Tensor]]:

    assert HAVE_FLAG_GEMS, "Triton-Based General Gemm needs FlagGems"
    assert not gelu and gelu_in is None, "Triton-Based General Gemm do not support gelu now"
    assert ub is None and ub_type is None, "Triton-Based General Gemm do not support ub comm in kernels"
    assert quantization_params is None, "Triton-Based General Gemm do not support quantization now"
    assert bias is None, "Triton-Based General Gemm do not support bias now"
    assert layout in ("TN", "NN", "NT"), f"GEMM layout {layout} not supported."

    transa = layout[0] == "T"
    transb = layout[1] == "T"

    alpha = validate_gemm_scale(alpha, True)
    beta = validate_gemm_scale(beta, accumulate)

    if out is not None:
        if not out.is_contiguous():
            raise ValueError("Output tensor is not contiguous.")

    # Use bfloat16 as default bias_dtype
    bias_dtype = TE_DType[torch.bfloat16 if bias is None else bias.dtype]

    s = -1
    b = -1
    orig_A_shape = A.shape
    orig_B_shape = B.shape
    shape_a_changed = False
    shape_b_changed = False

    if A.ndim == 3:
        A = A.view(-1, A.shape[-1])
        shape_a_changed = True

    if B.ndim == 3:
        s, b, _ = B.shape
        B = B.view(-1, B.shape[-1])
        shape_b_changed = True

    A_comp = A.T if transa else A
    B_comp = B.T if transb else B

    out1 = flag_gems.mm(B_comp, A_comp)

    if shape_b_changed:
        out1 = out1.view(s, b, -1)

    if out_dtype is not None and out1.dtype != out_dtype:
        out1 = out1.to(out_dtype)

    bias_grad = None
    gelu_input = None
    extra_output = None
    if out is not None:
        out.add_(out1)
        return out, bias_grad, gelu_input, extra_output
    else:
        return out1, bias_grad, gelu_input, extra_output
