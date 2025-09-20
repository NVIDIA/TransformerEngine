# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Quantization API for experimental middleware between Transformer Engine and Kitchen."""

from __future__ import annotations
import abc
import dataclasses
import enum
from typing import Iterable, Optional, Tuple, Union

import torch

from transformer_engine.common.recipe import Recipe
from transformer_engine.pytorch.tensor.quantized_tensor import QuantizedTensorBase, Quantizer
from transformer_engine.pytorch.experimental import utils


@enum.unique
class GEMMType(enum.Enum):
    """Type of GEMM operation being performed."""

    FPROP = "fprop"
    DGRAD = "dgrad"
    WGRAD = "wgrad"


@dataclasses.dataclass(frozen=True)
class MMParams:
    """Matrix multiplication parameters."""

    out_dtype: torch.dtype | None = None
    # Use split accumulator for more accurate FP8 GEMM
    use_split_accumulator: bool = True


@dataclasses.dataclass
class ExperimentalQuantizedTensor(QuantizedTensorBase):
    """Base class for experimental quantized tensor containers.

    An experimental container to hold quantization result, including quantized tensor, optional
    transposed quantized tensor, and corresponding decoding scales.

    data: torch.Tensor
        the quantized tensor.
    scale: torch.Tensor
        the decoding scale for the quantized tensor. Shape depends on the scaling granularity.
        - if scaling type is PER_TENSOR, it should be a 1D scalar tensor.
    data_t: torch.Tensor
        the transposed quantized tensor (computed lazily if needed).
    scale_t: torch.Tensor
        the decoding scale for the transposed quantized tensor.
    dtype: torch.dtype
        nominal tensor datatype.
    device: torch.device
        device of the tensor.
    quant_dtype: Union[utils.Fp4Formats, torch.dtype]
        low precision tensor datatype.
    original_shape: Tuple[int, ...]
        original shape of the tensor.
    quantizer: ExperimentalQuantizer
        Builder class for quantized tensor.
    """

    data: Optional[torch.Tensor] = None
    scale: Optional[torch.Tensor] = None
    data_t: Optional[torch.Tensor] = None
    scale_t: Optional[torch.Tensor] = None
    global_amax_row: Optional[torch.Tensor] = None
    global_amax_col: Optional[torch.Tensor] = None

    dtype: Optional[torch.dtype] = None
    device: Optional[torch.device] = None
    quant_dtype: Optional[Union[utils.Fp4Formats, torch.dtype]] = None
    original_shape: Optional[Tuple[int, ...]] = None
    quantizer: Optional[ExperimentalQuantizer] = None

    @property
    def experimental(self) -> bool:
        """Flag to indicate this quantizer is using experimental Kitchen middleware."""
        return True

    def get_quantizer(self) -> ExperimentalQuantizer:
        """Get builder for QuantizedExperimentalTensor

        Quantizer can be used for in-place operations.

        """
        if self.quantizer is not None:
            return self.quantizer
        raise ValueError("Quantizer is not set")

    def prepare_for_saving(
        self,
    ) -> Tuple[list[Optional[torch.Tensor]], ExperimentalQuantizedTensor]:
        """Prepare the quantization result for saving for backward"""
        tensors = [self.data, self.data_t, self.scale, self.scale_t]
        self.data = None
        self.data_t = None
        self.scale = None
        self.scale_t = None
        return tensors, self

    def restore_from_saved(
        self, tensors: list[Optional[torch.Tensor]]
    ) -> list[Optional[torch.Tensor]]:
        """Restore the quantization result from the saved tensors"""
        self.data = tensors[0]
        self.data_t = tensors[1]
        self.scale = tensors[2]
        self.scale_t = tensors[3]
        return tensors[4:]

    def dequantize(self, *args, **kwargs) -> torch.Tensor:
        """Dequantize the quantized tensor"""
        raise NotImplementedError(
            f"{self.__class__.__name__} class does not implement dequantize function"
        )

    # Compatibility
    @property
    def _data(self):
        return self.data

    @_data.setter
    def _data(self, value):
        self.data = value

    @property
    def _scale_inv(self):
        return self.scale

    @_scale_inv.setter
    def _scale_inv(self, value):
        self.scale = value


class ExperimentalQuantizer(Quantizer):
    """Experimental Quantizer class

    Defines the interface for experimental quantizers.
    """

    def __init__(self, *, rowwise: bool, columnwise: bool) -> None:
        super().__init__(rowwise=rowwise, columnwise=columnwise)
        self.internal = True

    @property
    def experimental(self) -> bool:
        """Flag to indicate this quantizer is using experimental Kitchen middleware"""
        return True

    @abc.abstractmethod
    def qgemm(
        self,
        qx: torch.Tensor,
        qw: torch.Tensor,
        m_params: MMParams,
        out_dtype: torch.dtype,
        sx: torch.Tensor,
        sw: torch.Tensor,
        bias: torch.Tensor | None = None,
        out: torch.Tensor | None = None,
        accumulate: bool = False,
        gemm_type: GEMMType = GEMMType.FPROP,
        qresult_x: ExperimentalQuantizedTensor | None = None,
        qresult_w: ExperimentalQuantizedTensor | None = None,
    ) -> torch.Tensor:
        """Quantized GEMM interface."""

    def dequantize(self, *args, **kwargs) -> torch.Tensor:
        """Dequantize the quantized tensor"""
        raise NotImplementedError(
            f"{self.__class__.__name__} class does not implement dequantize function"
        )

    def update_quantized(self, *args, **kwargs) -> torch.Tensor:
        """Update the quantized tensor with the given tensor in-place"""
        raise NotImplementedError(
            f"{self.__class__.__name__} class does not implement update_quantized function"
        )

    def make_empty(
        self,
        shape: Iterable[int],
        *,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
    ) -> QuantizedTensorBase:
        raise NotImplementedError(
            f"{self.__class__.__name__} class does not implement make_empty function"
        )

    def calibrate(self, tensor: torch.Tensor) -> None:
        raise NotImplementedError(
            f"{self.__class__.__name__} class does not implement calibrate function"
        )

    def _get_compatible_recipe(self) -> Union[type[Recipe], None]:
        raise NotImplementedError(
            f"{self.__class__.__name__} class does not implement _get_compatible_recipe function"
        )
