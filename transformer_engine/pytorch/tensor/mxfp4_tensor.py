# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Tensor class with FP4 data and MX (E8M0, block-32) scaling"""

from __future__ import annotations
from collections.abc import Iterable
import math
from typing import Any, Dict, Optional, Tuple, Union

import torch
import transformer_engine_torch as tex

from transformer_engine.common.recipe import Recipe
from ..constants import MXFP8_BLOCK_SCALING_SIZE, DType
from ..utils import devices_match, round_up_to_nearest_multiple
from .storage.mxfp4_tensor_storage import MXFP4TensorStorage, _FromMXFP4Func
from ..quantized_tensor import QuantizedTensor, Quantizer
from ..dynamo import register_value_opaque_quantizer
from ._quantization_helpers import safe_quantized_repr

aten = torch.ops.aten


class MXFP4Quantizer(Quantizer):
    """Builder class for FP4 E2M1 tensors with MX block scaling

    High-precision tensors are divided into groups of 32 consecutive
    elements, each with an E8M0 scale. Row-wise and column-wise data are
    quantized independently from the input.

    MXFP4 tensors are only consumed as the weight operand of
    MXFP4 x MXFP8 GEMMs.

    """

    dtype: DType

    def __init__(
        self,
        *,
        rowwise: bool = True,
        columnwise: bool = True,
    ) -> None:
        super().__init__(rowwise=rowwise, columnwise=columnwise)
        self.dtype = DType.kFloat4E2M1

    def copy(self) -> MXFP4Quantizer:
        """Create shallow copy"""
        quantizer = MXFP4Quantizer(rowwise=self.rowwise_usage, columnwise=self.columnwise_usage)
        quantizer.internal = self.internal
        quantizer.optimize_for_gemm = self.optimize_for_gemm
        return quantizer

    def storage_metadata(self, fake_dtype: torch.dtype) -> Dict[str, Any]:
        return {
            "cls": MXFP4TensorStorage if self.internal else MXFP4Tensor,
            "nontensor_kwargs": {
                "quantizer": self,
                "with_gemm_swizzled_scales": self.optimize_for_gemm,
                "fake_dtype": fake_dtype,
            },
        }

    def inner_tensor_specs(
        self, shape: Tuple[int, ...]
    ) -> Dict[str, Tuple[Tuple[int, ...], torch.dtype]]:
        shape = tuple(shape)
        data_shape = (*shape[:-1], shape[-1] // 2)
        specs: Dict[str, Tuple[Tuple[int, ...], torch.dtype]] = {}
        if self.rowwise_usage:
            specs["_rowwise_data"] = (data_shape, torch.uint8)
            specs["_rowwise_scale_inv"] = (
                tuple(self.get_scale_shape(shape, columnwise=False)),
                torch.uint8,
            )
        if self.columnwise_usage:
            specs["_columnwise_data"] = (data_shape, torch.uint8)
            specs["_columnwise_scale_inv"] = (
                tuple(self.get_scale_shape(shape, columnwise=True)),
                torch.uint8,
            )
        return specs

    def update_quantized(
        self,
        src: torch.Tensor,
        dst: QuantizedTensor,
        *,
        noop_flag: Optional[torch.Tensor] = None,
    ) -> QuantizedTensor:
        if not isinstance(dst, MXFP4TensorStorage):
            raise TypeError(f"Cannot store quantized MXFP4 in {type(dst)} type.")
        if not devices_match(src.device, dst.device):
            src = src.to(device=dst.device)
        if not src.is_contiguous():
            src = src.contiguous()
        tex.quantize(src, self, dst, noop_flag)
        return dst

    def quantize_impl(self, tensor: torch.Tensor) -> QuantizedTensor:
        """Quantize tensor implementation"""
        return tex.quantize(tensor, self)

    def is_quantizable(self, inp: torch.Tensor) -> bool:
        """Returns whether or not given inp can be quantized"""
        if inp.ndim < 2:
            return False
        if inp.shape[-1] % MXFP8_BLOCK_SCALING_SIZE != 0:
            return False
        if math.prod(inp.shape[:-1]) % MXFP8_BLOCK_SCALING_SIZE != 0:
            return False
        return True

    def get_scale_shape(
        self,
        shape: Iterable[int],
        columnwise: bool,
    ) -> Tuple[int, int]:
        """Shape of the E8M0 scale tensor, padded like MXFP8 scales."""
        shape = tuple(shape)
        if columnwise:
            return (
                round_up_to_nearest_multiple(math.prod(shape[:-1]) // MXFP8_BLOCK_SCALING_SIZE, 4),
                round_up_to_nearest_multiple(shape[-1], 128),
            )
        return (
            round_up_to_nearest_multiple(math.prod(shape[:-1]), 128),
            round_up_to_nearest_multiple(shape[-1] // MXFP8_BLOCK_SCALING_SIZE, 4),
        )

    def _get_compatible_recipe(self) -> Union[type[Recipe], None]:
        return None

    def is_requantization_safe(self) -> bool:
        """MXFP4 block scales are derived deterministically from each input."""
        return True


register_value_opaque_quantizer(MXFP4Quantizer)


class MXFP4Tensor(MXFP4TensorStorage, QuantizedTensor):
    """Tensor class with FP4 E2M1 data and MX block scaling

    The tensor presents as having a standard, higher-precision dtype,
    but the data itself is packed FP4. Operations other than GEMM
    dequantize to the nominal dtype.

    """

    def __new__(
        cls,
        *args,
        rowwise_data: Optional[torch.Tensor],
        rowwise_scale_inv: Optional[torch.Tensor],
        columnwise_data: Optional[torch.Tensor],
        columnwise_scale_inv: Optional[torch.Tensor],
        quantizer: Optional[Quantizer],
        with_gemm_swizzled_scales: bool,
        **kwargs,
    ):
        return super().__new__(
            cls,
            rowwise_data,
            rowwise_scale_inv,
            columnwise_data,
            columnwise_scale_inv,
            quantizer,
            with_gemm_swizzled_scales,
            *args,
            **kwargs,
        )

    def __repr__(self, *, tensor_contents=None):
        try:
            return f"MXFP4Tensor(data={self.dequantize()})"
        except Exception as exc:  # pylint: disable=broad-except
            return safe_quantized_repr(self, "MXFP4Tensor", error=exc)

    def dequantize(self, *, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        """Construct plain PyTorch tensor from MXFP4Tensor"""
        if dtype is None:
            dtype = self.dtype
        if torch.is_grad_enabled():
            return _FromMXFP4Func.apply(self, dtype)
        return _FromMXFP4Func.forward(None, self, dtype)

    def _build_default_quantizer(self) -> Optional[Quantizer]:
        return MXFP4Quantizer()

    def quantize_(
        self,
        tensor: torch.Tensor,
        *,
        noop_flag: Optional[torch.Tensor] = None,
    ) -> MXFP4Tensor:
        """Update FP4 data in-place"""
        if isinstance(tensor, QuantizedTensor):
            return self.quantize_(tensor.dequantize())
        return super().quantize_(tensor, noop_flag=noop_flag)

    def view(self, *shape: Tuple[int]) -> MXFP4Tensor:
        # pylint: disable=missing-function-docstring
        return _ViewFunc.apply(self, shape)

    def reshape(self, *shape: Tuple[int]) -> MXFP4Tensor:
        # pylint: disable=missing-function-docstring
        return _ViewFunc.apply(self, shape)

    def contiguous(
        self,
        memory_format: torch.memory_format = torch.contiguous_format,
    ) -> MXFP4Tensor:
        """Returns self, since data is always contiguous"""
        if memory_format != torch.contiguous_format:
            raise ValueError("MXFP4Tensor does not support different memory formats!")
        return self

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs=None):
        if func == aten.clone.default:
            tensor = args[0]
            metadata = tensor.get_metadata()
            for key in (
                "rowwise_data",
                "rowwise_scale_inv",
                "columnwise_data",
                "columnwise_scale_inv",
            ):
                if metadata[key] is not None:
                    metadata[key] = metadata[key].detach().clone()
            return MXFP4Tensor(
                shape=tensor.shape,
                dtype=tensor.dtype,
                device=tensor.device,
                requires_grad=False,
                **metadata,
            )

        if func == aten.view.default:
            tensor, shape = args[0], args[1]
            if list(shape) == list(tensor.size()):
                return tensor.detach()
            return tensor.view(shape)

        if func == aten.copy_.default:
            dst, src = args[0], args[1]
            if (
                isinstance(dst, MXFP4Tensor)
                and isinstance(src, MXFP4Tensor)
                and dst._with_gemm_swizzled_scales != src._with_gemm_swizzled_scales
            ):
                dst.quantize_(src)
                return None

        return super().__torch_dispatch__(func, types, args, kwargs)

    @property
    def device(self):
        """Return the device of the tensor. Define this to avoid expensive PyObject lookups."""
        if self._rowwise_data is not None:
            return self._rowwise_data.device
        if self._columnwise_data is not None:
            return self._columnwise_data.device
        raise RuntimeError("MXFP4Tensor has no data!")

    @property
    def is_cuda(self):
        """Return whether the tensor is on a CUDA device."""
        return self.device.type == "cuda"


class _ViewFunc(torch.autograd.Function):
    """View function

    View the MXFP4Tensor using the provided shape. The inner
    dimension must be unchanged.

    """

    @staticmethod
    def forward(
        ctx,
        tensor: MXFP4Tensor,
        shape: Tuple[Any, ...],
    ) -> MXFP4Tensor:
        # pylint: disable=missing-function-docstring
        ctx.shape = tensor.shape
        if len(shape) == 1 and isinstance(shape[0], Iterable):
            shape = shape[0]
        storage = MXFP4TensorStorage.view(tensor, shape if shape else None)
        if storage is tensor:
            return tensor
        return MXFP4Tensor(
            shape=storage.size(),
            dtype=tensor.dtype,
            device=tensor.device,
            requires_grad=tensor.requires_grad,
            **storage.get_metadata(),
        )

    @staticmethod
    def backward(
        ctx,
        grad: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], ...]:
        # pylint: disable=missing-function-docstring
        return grad.reshape(ctx.shape), None
