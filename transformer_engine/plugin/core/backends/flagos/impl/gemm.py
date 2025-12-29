# Copyright (c) 2025, BAAI. All rights reserved.
#
# See LICENSE for license information.

from typing import Any, Dict, List, Optional, Tuple, Union
import torch

import flag_gems

__all__ = [
    "generic_gemm_fl",
]

_DTYPE_TO_TORCH = {
    0: torch.uint8,
    2: torch.int32,
    4: torch.float32,
    5: torch.float16,
    6: torch.bfloat16,
    7: torch.float8_e4m3fn,
    8: torch.float8_e5m2,
}

def validate_gemm_scale(scale: Optional[float], required: bool) -> float:
    if required:
        return scale if scale is not None else 1.0
    if scale not in (0.0, None):
        raise ValueError("scale must be zero")
    return 0.0

def _convert_dtype(dtype: Union[int, torch.dtype, None]) -> Optional[torch.dtype]:
    if dtype is None:
        return None
    if isinstance(dtype, torch.dtype):
        return dtype
    if isinstance(dtype, int):
        return _DTYPE_TO_TORCH.get(dtype, None)
    if hasattr(dtype, 'value'):
        return _DTYPE_TO_TORCH.get(dtype.value, None)
    return None

def generic_gemm_fl(
    A: torch.Tensor,
    transA: bool,
    B: torch.Tensor,
    transB: bool,
    D: Optional[torch.Tensor],
    quantizer: Any,
    output_dtype: Any,
    bias: Optional[torch.Tensor],
    bias_type: Any,
    gelu: bool,
    gelu_in: Optional[torch.Tensor],
    grad: bool,
    workspace: torch.Tensor,
    workspace_size: int,
    accumulate: bool,
    use_split_accumulator: bool,
    comm_overlap: Optional[Any] = None,
    comm_type: Optional[Any] = None,
    extra_output: Optional[torch.Tensor] = None,
    bulk_overlap: bool = False,
    alpha: float = 1.0,
    beta: Optional[float] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    with flag_gems.use_gems():
        assert not gelu and gelu_in is None, "Triton-Based General Gemm do not support gelu now"
        assert quantizer is None, "Triton-Based General Gemm do not support quantization now"
        assert bias is None, "Triton-Based General Gemm do not support bias now"

        alpha = validate_gemm_scale(alpha, True)
        beta = validate_gemm_scale(beta, accumulate)

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

        A_comp = A.T if transA else A
        B_comp = B.T if transB else B

        out1 = flag_gems.mm(B_comp, A_comp)

        if shape_b_changed:
            out1 = out1.view(s, b, -1)

        torch_out_dtype = _convert_dtype(output_dtype)
        if torch_out_dtype is not None and out1.dtype != torch_out_dtype:
            out1 = out1.to(torch_out_dtype)

        bias_grad = None
        gelu_input = None
        extra_output_ret = None

        if D is not None:
            if accumulate:
                D.add_(out1)
            else:
                D.copy_(out1)
            return D, bias_grad, gelu_input, extra_output_ret
        else:
            return out1, bias_grad, gelu_input, extra_output_ret
