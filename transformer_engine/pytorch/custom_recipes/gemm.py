# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""GEMM API that enables custom GEMM logic for custom quantization recipes."""

from typing import Iterable, Optional

import torch

from transformer_engine.pytorch.custom_recipes.quantization import (
    MMParams,
    GEMMType,
)
from transformer_engine.pytorch.quantized_tensor import QuantizedTensorStorage, Quantizer
from transformer_engine.pytorch.tensor.utils import is_custom


def custom_gemm(
    A: QuantizedTensorStorage,
    B: QuantizedTensorStorage,
    workspace: torch.Tensor,  # pylint: disable=unused-argument
    out_dtype: Optional[torch.dtype] = None,
    quantization_params: Optional[Quantizer] = None,  # pylint: disable=unused-argument
    gelu: bool = False,  # pylint: disable=unused-argument
    gelu_in: torch.Tensor = None,  # pylint: disable=unused-argument
    accumulate: bool = False,  # pylint: disable=unused-argument
    layout: str = "TN",
    out: Optional[torch.Tensor] = None,  # pylint: disable=unused-argument
    bias: Optional[torch.Tensor] = None,
    use_split_accumulator: bool = False,
    grad: bool = False,
) -> Iterable[Optional[torch.Tensor]]:
    """Dispatch GEMM to quantizer's qgemm method."""
    assert is_custom(A) and is_custom(B), "A and B must be custom tensors"

    A, B = B, A

    # Determine GEMM type based on grad flag and layout
    if not grad:
        gemm_type = GEMMType.FPROP
    else:
        if layout == "NN":
            gemm_type = GEMMType.DGRAD
        elif layout == "NT":
            gemm_type = GEMMType.WGRAD
        else:
            # Default to FPROP for other layouts
            gemm_type = GEMMType.FPROP

    # Extract quantizer from QuantizedTensor to get qgemm logic
    # TODO(etsykunov): make it more flexible, what if we might want to use gemm logic from B._quantizer?
    quantizer = None
    if hasattr(A, "_quantizer") and A._quantizer is not None:
        quantizer = A._quantizer
    elif hasattr(B, "_quantizer") and B._quantizer is not None:
        quantizer = B._quantizer
    else:
        raise ValueError("No quantizer found in QuantizedTensor objects")

    # Create MMParams
    m_params = MMParams(
        out_dtype=out_dtype,
        use_split_accumulator=use_split_accumulator,
    )
    out_dtype = A.dtype if m_params.out_dtype is None else m_params.out_dtype

    if gemm_type == GEMMType.FPROP:
        qx, sx = A.data, A.scale
        qw, sw = B.data, B.scale
        assert qx is not None
        assert sx is not None
        assert qw is not None
        assert sw is not None
        assert A.original_shape is not None

        # Call quantizer's qgemm method
        result = quantizer.qgemm(
            qx,
            qw,
            m_params,
            out_dtype,
            sx,
            sw,
            bias,
            gemm_type=GEMMType.FPROP,
            qresult_x=A,
            qresult_w=B,
        )
        if len(A.original_shape) > 2:
            # Original input was 3D, so we need to reshape result back to 3D
            batch_size = A.original_shape[0]
            seq_len = A.original_shape[1]
            result = result.view(batch_size, seq_len, result.shape[-1])
    elif gemm_type == GEMMType.DGRAD:
        qdy, sdy = A.data, A.scale
        qw_t, sw_t = B.data_t, B.scale_t
        assert qdy is not None
        assert sdy is not None
        assert qw_t is not None
        assert sw_t is not None

        result = quantizer.qgemm(
            qdy,
            qw_t,
            m_params,
            out_dtype,
            sdy,
            sw_t,
            None,
            gemm_type=GEMMType.DGRAD,
            qresult_x=A,
            qresult_w=B,
        )
    elif gemm_type == GEMMType.WGRAD:
        qdy_t, sdy_t = A.data_t, A.scale_t
        qx_t, sx_t = B.data_t, B.scale_t
        assert qdy_t is not None
        assert sdy_t is not None
        assert qx_t is not None
        assert sx_t is not None

        result = quantizer.qgemm(
            qdy_t,
            qx_t,
            m_params,
            out_dtype,
            sdy_t,
            sx_t,
            None,
            gemm_type=GEMMType.WGRAD,
            qresult_x=A,
            qresult_w=B,
        )

    # Return in the same format as general_gemm
    return result, None, None, None
