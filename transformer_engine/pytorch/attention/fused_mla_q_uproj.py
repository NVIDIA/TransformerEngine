# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Fused MLA Q up-projection + per-head RoPE + MXFP8 quantize."""

from __future__ import annotations
import functools
import os
from importlib.metadata import PackageNotFoundError, version as get_pkg_version

import torch
import transformer_engine_torch as tex
from packaging.version import Version as PkgVersion

from ..constants import MXFP8_BLOCK_SCALING_SIZE
from ..quantized_tensor import QuantizedTensor
from ..tensor.mxfp8_tensor import MXFP8Quantizer, MXFP8Tensor
from ..utils import get_device_compute_capability

_CUDNN_FRONTEND_MIN_VERSION = "1.27.0"


def _cudnn_frontend_version_supported() -> bool:
    """Check that the installed nvidia-cudnn-frontend meets the minimum version."""
    try:
        return PkgVersion(get_pkg_version("nvidia-cudnn-frontend")) >= PkgVersion(
            _CUDNN_FRONTEND_MIN_VERSION
        )
    except PackageNotFoundError:
        return False


class FusedMLAQUpProjRopeQuant:
    """Wrapper for the cuDNN fused MLA Q up-proj + per-head RoPE + MXFP8 quantize kernel.

    - If w is already a QuantizedTensor (primary FP8 parameter in MXFP8BlockScaling recipe),
      this performs an MXFP8 GEMM within the fusion (and quantizes the input if necessary)
    - Otherwise (plain BF16 weight), x and w are passed as-is to the BF16 kernel variant.
    """

    @classmethod
    @functools.lru_cache(maxsize=None)
    def _kernel(cls):
        # Import directly from the subpackage to avoid depending on cudnn/__init__.py
        # lazy-import registration (which would require overlaying cudnn/__init__.py and
        # could revert atomicrmw fixes present in the container's version).
        try:
            from cudnn import gemm_proj_rope_mxfp8_wrapper_sm100

            return gemm_proj_rope_mxfp8_wrapper_sm100
        except ImportError:
            return None

    @classmethod
    @functools.lru_cache(maxsize=None)
    def is_supported(cls) -> bool:
        """Whether the cuDNN FE fused gemm rope quant wrapper is available"""
        if int(os.environ.get("NVTE_FUSED_MLA_Q_UPROJ", "1")) <= 0:
            return False
        if not _cudnn_frontend_version_supported():
            return False
        if get_device_compute_capability()[0] < 10:
            return False
        if cls._kernel() is None:
            return False
        return True

    @classmethod
    def run(
        cls,
        x: torch.Tensor,
        w,  # MXFP8Tensor (primary FP8 param) or bf16 torch.Tensor
        cos: torch.Tensor,
        sin: torch.Tensor,
        s: int,
        b: int,
    ) -> "tuple[MXFP8Tensor, torch.Tensor]":
        """Run the fused kernel; return (Q MXFP8Tensor, activation saved for the wgrad backward).

        The kernel precision is selected by the weight precision.
        """

        from cuda.bindings import driver as cuda

        stream = cuda.CUstream(torch.cuda.current_stream(x.device).cuda_stream)
        wrapper = cls._kernel()

        if isinstance(w, QuantizedTensor):
            assert isinstance(w, MXFP8Tensor), (
                "FusedMLAQUpProjRopeQuant expects an MXFP8Tensor weight (MXFP8BlockScaling"
                f" recipe), got {type(w).__name__}. Use the unfused path for other quantization"
                " recipes."
            )
            # ---- FP8 projection: MXFP8-cast x (both usages) + reuse w's fp8 codes -> mxfp8in ----
            # Quantize x with both rowwise (for the forward GEMM) and columnwise (for the FP8
            # wgrad in backward, matching the unfused path).
            x_quantizer = MXFP8Quantizer(
                fp8_dtype=tex.DType.kFloat8E4M3, rowwise=True, columnwise=True
            )
            x_mxfp8 = x_quantizer(x)
            x_code = x_mxfp8._rowwise_data.view(torch.float8_e4m3fn)  # [tokens, K]
            x_scale = x_mxfp8._rowwise_scale_inv  # [tokens, K//32] uint8

            # Primary FP8 parameter: already quantized; use its rowwise FP8 codes + E8M0 scales.
            w.update_usage(rowwise_usage=True, columnwise_usage=None)
            w_code = w._rowwise_data.view(torch.float8_e4m3fn)  # [N, K]
            w_scale = w._rowwise_scale_inv  # [N, K//32] uint8

            out = wrapper(
                x_code,
                w_code,
                cos,
                sin,
                x_scale=x_scale,
                w_scale=w_scale,
                w_out_in=True,
                stream=stream,
            )

            # Drop rowwise data now.
            # Only columnwise x is needed for the FP8 wgrad in backward.
            x_mxfp8.update_usage(rowwise_usage=False, columnwise_usage=True)
            x_saved = x_mxfp8
        else:
            # ---- 16-bit projection: bf16 GEMM inputs -> bf16in (the projection stays bf16) ----
            out = wrapper(x, w, cos, sin, w_out_in=True, stream=stream)
            x_saved = x

        nh = out["out_fp8_row"].shape[1]
        d = out["out_fp8_row"].shape[2]
        query = cls.wrap_mxfp8(
            out["out_fp8_row"],
            out["out_scales_row"],
            out["out_fp8_col"],
            out["out_scales_col"],
            s,
            b,
            nh,
            d,
        )
        # 2nd return is the activation to save for wgrad: MXFP8 (fp8 path) or bf16 (16-bit path).
        return query, x_saved

    @classmethod
    def wrap_mxfp8(
        cls,
        fp8_row: torch.Tensor,
        scales_row: torch.Tensor,
        fp8_col: torch.Tensor,
        scales_col: torch.Tensor,
        s: int,
        b: int,
        nh: int,
        d: int,
    ) -> MXFP8Tensor:
        """Wrap raw data and scale tensors into an MXFP8Tensor"""

        blk = MXFP8_BLOCK_SCALING_SIZE
        # Both rowwise and columnwise Q are required:
        # - Forward QK^T uses rowwise
        # - cuDNN backward (fused_attn_fp8_bwd_impl) requires columnwise for dK gradient
        quantizer = MXFP8Quantizer(fp8_dtype=tex.DType.kFloat8E4M3, rowwise=True, columnwise=True)
        return MXFP8Tensor(
            shape=(s, b, nh, d),
            dtype=torch.bfloat16,
            rowwise_data=fp8_row.view(s, b, nh, d),
            rowwise_scale_inv=scales_row.view(s, b, nh, d // blk),
            columnwise_data=fp8_col.view(s, b, nh, d),
            columnwise_scale_inv=scales_col.view(s // blk, b, nh, d),
            quantizer=quantizer,
            requires_grad=False,
            fp8_dtype=tex.DType.kFloat8E4M3,
            with_gemm_swizzled_scales=False,
        )
