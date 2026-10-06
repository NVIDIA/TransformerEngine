# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Experimental quantize/dequantize quantizers with high-precision storage."""

from functools import lru_cache
import os

import torch
import transformer_engine_torch as tex

from ..constants import DType
from ..quantized_tensor import Quantizer
from ..tensor.identity_tensor import IdentityTensor
from ..tensor.storage.identity_tensor_storage import IdentityTensorStorage
from ..tensor.nvfp4_tensor import NVFP4Quantizer


@lru_cache(maxsize=None)
def _supports_fused_device(device_index):
    """Device capabilities are fixed for the process lifetime."""
    return torch.cuda.get_device_capability(device_index) == (10, 0)


class NVFP4QDQQuantizer(Quantizer):
    """Store native rowwise NVFP4 Q+DQ values in an IdentityTensor.

    ``quantizer`` supplies native NVFP4 options and is copied with rowwise-only
    usage. Output defaults to the source dtype; ``dtype`` overrides only the
    decode dtype, never the quantization input. The high-precision result can
    serve either GEMM orientation.

    ``backend`` is ``auto``, ``reference``, or ``fused``. Auto falls back to
    native Q+DQ outside plain contiguous BF16/FP16 1D E4M3 scaling on SM100.
    In particular, 4over6 and 2D scaling currently use the reference path.
    Forced fused execution raises for unsupported inputs/configurations.
    This experimental API is subject to change.
    """

    # Unknown to native fused quantization producers; outputs use native HP GEMM.
    custom = True

    supports_output_dtype = True

    def __init__(
        self,
        quantizer=None,
        *,
        dtype=None,
        backend="auto",
        rowwise=True,
        columnwise=True,
    ):
        super().__init__(rowwise=rowwise, columnwise=columnwise)
        if backend not in ("auto", "reference", "fused"):
            raise ValueError("backend must be auto, reference, or fused")
        if quantizer is not None and not isinstance(quantizer, NVFP4Quantizer):
            raise TypeError("quantizer must be an NVFP4Quantizer")
        self.nvfp4_quantizer = quantizer.copy() if quantizer is not None else NVFP4Quantizer()
        self.nvfp4_quantizer.set_usage(rowwise=True, columnwise=False)
        self.nvfp4_quantizer.internal = True
        self.dtype = dtype
        self.backend = backend

    def copy(self):
        """Copy options without sharing mutable native quantizer state."""
        result = object.__new__(type(self))
        result.__dict__ = self.__dict__.copy()
        # Native NVFP4 state consists of options and an immutable cached RHT
        # matrix. A shallow copy isolates option updates without rebuilding
        # derived CUDA state for every high-precision result.
        # Avoid the native __getstate__ hook: serialization intentionally drops
        # process groups, whereas a runtime copy must preserve them.
        native = object.__new__(type(self.nvfp4_quantizer))
        native.__dict__ = self.nvfp4_quantizer.__dict__.copy()
        result.nvfp4_quantizer = native
        return result

    @property
    def with_amax_reduction(self):
        """Forward distributed amax policy to the native fallback."""
        return self.nvfp4_quantizer.with_amax_reduction

    @with_amax_reduction.setter
    def with_amax_reduction(self, value):
        self.nvfp4_quantizer.with_amax_reduction = value

    @property
    def amax_reduction_group(self):
        """Native amax reduction group."""
        return self.nvfp4_quantizer.amax_reduction_group

    @amax_reduction_group.setter
    def amax_reduction_group(self, value):
        self.nvfp4_quantizer.amax_reduction_group = value

    def selected_backend(self, tensor, dtype=None):
        """Return dispatch selection, or reject a forced unsupported fast path."""
        q = self.nvfp4_quantizer
        output_dtype = dtype or self.dtype or tensor.dtype
        eligible = (
            tensor.is_cuda
            and tensor.is_contiguous()
            and tensor.ndim >= 2
            and tensor.numel() > 0
            and tensor.shape[-1] % 16 == 0
            and (tensor.numel() // tensor.shape[-1]) % 16 == 0
            and tensor.data_ptr() % 16 == 0
            and tensor.dtype in (torch.bfloat16, torch.float16)
            and output_dtype == tensor.dtype
            and _supports_fused_device(tensor.device.index)
            and q.dtype == DType.kFloat4E2M1
            and q.scale_dtype == DType.kFloat8E4M3
            and q.nvfp4_e4m3_max in (0, 448)
            and not any(
                (
                    q.with_rht,
                    q.with_post_rht_amax,
                    q.with_2d_quantization,
                    q.stochastic_rounding,
                    q.row_scaled_nvfp4,
                    q.nvfp4_use_4over6,
                    q.disable_second_level_scale,
                    q.with_amax_reduction,
                )
            )
            and os.getenv("NVTE_USE_FAST_MATH", "0") == "0"
        )
        if self.backend == "fused" and not eligible:
            raise ValueError("Forced fused NVFP4 QDQ does not support this input/configuration")
        return "fused" if eligible and self.backend != "reference" else "reference"

    def _reference(self, tensor, dtype):
        return self.nvfp4_quantizer.quantize(tensor).dequantize(dtype=dtype)

    def _compute(self, tensor, output, noop_flag=None, *, backend=None):
        if (backend or self.selected_backend(tensor, output.dtype)) == "fused":
            # Framework owns both allocations; amax reduction and QDQ use the
            # current stream and may be captured together.
            amax = torch.empty(1, dtype=torch.float32, device=tensor.device)
            tex.nvfp4_qdq(tensor, output, amax, noop_flag)
        else:
            data = self._reference(tensor, output.dtype)
            if noop_flag is None:
                output.copy_(data)
            else:
                torch.where(noop_flag != 1, data, output, out=output)

    def quantize_impl(self, tensor, *, dtype=None):
        dtype = dtype or self.dtype or tensor.dtype
        if self.selected_backend(tensor, dtype) == "reference":
            data = self._reference(tensor, dtype)
            return self._wrap(data)
        result = self._wrap(torch.empty(tensor.shape, dtype=dtype, device=tensor.device))
        self._compute(tensor, result._hp_data, backend="fused")
        return result

    def make_empty(
        self,
        shape,
        *,
        dtype=torch.float32,
        device=None,
        requires_grad=False,
        pin_memory=False,
    ):
        data = torch.empty(
            tuple(shape),
            dtype=self.dtype or dtype,
            device=device or "cuda",
            pin_memory=pin_memory,
        )
        return self._wrap(data, requires_grad=requires_grad)

    def _wrap(self, data, requires_grad=False):
        if self.internal:
            return IdentityTensorStorage(hp_data=data, fake_dtype=data.dtype, quantizer=self)
        return IdentityTensor(
            data.shape,
            data.dtype,
            hp_data=data,
            quantizer=self,
            requires_grad=requires_grad,
            device=data.device,
        )

    def update_quantized(self, src, dst, *, noop_flag=None):
        if not isinstance(dst, IdentityTensorStorage) or dst._hp_data is None:
            raise TypeError("NVFP4 QDQ requires allocated IdentityTensorStorage")
        if src.shape != dst._hp_data.shape or src.device != dst._hp_data.device:
            raise ValueError("NVFP4 QDQ source and destination must have matching shape/device")
        self._compute(src, dst._hp_data, noop_flag)
        return dst

    def is_requantization_safe(self):
        return self.nvfp4_quantizer.is_requantization_safe()

    def _get_compatible_recipe(self):
        from transformer_engine.common.recipe import CustomRecipe

        return CustomRecipe
