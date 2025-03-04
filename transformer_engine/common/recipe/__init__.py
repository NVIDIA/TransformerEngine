# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""This module provides predefined FP8 recipes."""
from __future__ import annotations
import warnings
from enum import Enum
from typing import Literal, Optional, Union, Callable, NamedTuple
from pydantic.dataclasses import dataclass


class _FormatHelper(NamedTuple):
    """
    Stores max FP8 values for fprop and bprop a `Format`.
    """

    max_fwd: float
    max_bwd: float


class Format(Enum):
    """
    Supported FP8 formats.

    Values
    ------
    E4M3 :
          All FP8 tensors are in e4m3 format
    E5M2 :
          All FP8 tensors are in e5m2 format
    HYBRID :
            FP8 tensors in the forward pass are in e4m3 format,
            FP8 tensors in the backward pass are in e5m2 format
    """

    E4M3 = _FormatHelper(max_fwd=448, max_bwd=448)
    E5M2 = _FormatHelper(max_fwd=57344, max_bwd=57344)
    HYBRID = _FormatHelper(max_fwd=E4M3.max_fwd, max_bwd=E5M2.max_bwd)


class Recipe:
    """
    Base recipe class.
    """

    def mxfp8(self):
        """Whether the given recipe is MXFP8 block scaling."""
        return isinstance(self, MXFP8BlockScaling)

    def delayed(self):
        """Whether the given recipe is delayed scaling."""
        return isinstance(self, DelayedScaling)


@dataclass()
class DelayedScaling(Recipe):
    """
    Use the delayed scaling factor strategy. Use scale factor from previous
    iteration and record amax history of `amax_history_len` steps.

    Parameters
    ----------
    margin : int, default = 0
            Margin for the scaling factor computation.
    fp8_format : {Format.E4M3, Format.HYBRID}, default = Format.HYBRID
                Controls the FP8 data format used during forward and backward
                pass.
    amax_history_len : int, default = 1024
                      The length of the amax history window used for
                      scaling factor computation.
    amax_compute_algo : {'max', 'most_recent', Callable}, default = 'max'
                       Algorithm used for choosing the `amax` value for the
                       scaling factor computation. There are 2 predefined
                       choices: `max` chooses the largest `amax` in the history
                       window, while `most_recent` always chooses the most recently
                       seen value. Alternatively, one may pass a function of the
                       signature:

                       .. code-block:: python

                         def amax_compute(amax_history: Tensor) -> Tensor

                       where `Tensor` is a framework tensor type.
    scaling_factor_compute_algo : Callable, default = None
                                 Algorithm used for computing the new scaling
                                 factor based on the value of `amax`. It should
                                 be a function of the signature:

                                 .. code-block:: python

                                   def scaling_factor_compute(amax: Tensor,
                                                              old_scaling_factor: Tensor,
                                                              fp8_max: Tensor,
                                                              recipe: DelayedScaling) -> Tensor

                                 where `Tensor` is a framework tensor type.
    reduce_amax: bool, default = `True`
                By default, if `torch.distributed` is initialized, the `amax` value for FP8
                tensors is reduced across the `fp8_group` (specified in the `fp8_autocast`
                call). This keeps the amaxes and scaling factors synced across the given
                distributed group. If set to `False`, this reduction is skipped and every
                GPU maintains local amaxes and scaling factors. To ensure results are
                numerically identical across checkpointing boundaries in this case, all
                ranks must checkpoint in order to store the local tensors.
    fp8_dpa: bool, default = `False`
             Whether to enable FP8 dot product attention (DPA). When the model is placed in an
             `fp8_autocast(enabled=True)` region and `fp8_dpa` is set to `True`, DPA casts the
             inputs from higher precision to FP8, performs attention in FP8, and casts tensors
             back to higher precision as outputs. FP8 DPA currently is only supported in the
             `FusedAttention` backend.
    fp8_mha: bool, default = `False`
            Whether to enable FP8 multi-head attention (MHA). When `True`, it removes the casting
            operations mentioned above at the DPA boundaries. Currently only standard MHA modules
            i.e. `LayerNormLinear/Linear + DPA + Linear`, are supported for this feature. When
            `fp8_mha = False, fp8_dpa = True`, a typical MHA module works as
            `LayerNormLinear (BF16 output) -> (cast to FP8 ) FP8 DPA (cast to BF16) -> Linear`.
            When `fp8_mha = True, fp8_dpa = True`, it becomes
            `LayerNormLinear (FP8 output) -> FP8 DPA -> Linear`.

    Notes
    -----
    * By default (when `scaling_factor_compute_algo` is left as `None`) the scaling
      factor is computed from the final `amax` value using the formula:

      .. code-block:: python

          FP8_MAX = maximum_representable_value(fp8_format)
          new_scaling_factor = (FP8_MAX / amax) / (2 ^ margin)

    * `fp8_dpa` and `fp8_mha` are Beta features, and their API and functionality are
      subject to change in future Transformer Engine releases.
    """

    margin: int = 0
    interval: int = -1
    fp8_format: Format = Format.HYBRID
    amax_history_len: int = 1024
    amax_compute_algo: Union[Literal["max", "most_recent"], Callable] = "max"
    scaling_factor_compute_algo: Optional[Callable] = None
    reduce_amax: bool = True
    fp8_dpa: bool = False
    fp8_mha: bool = False

    def __post_init__(self) -> None:
        assert self.fp8_format != Format.E5M2, "Pure E5M2 training is not supported."
        if self.interval >= 0:
            warnings.warn(
                "`interval` argument is deprecated and unused. "
                "It will be removed in an upcoming release.",
                DeprecationWarning,
            )

    def __repr__(self) -> str:
        return (
            f"margin={self.margin}, "
            f"format={str(self.fp8_format).split('.')[1]}, "
            f"amax_history_len={self.amax_history_len}, "
            f"fp8_dpa={self.fp8_dpa}, "
            f"fp8_mha={self.fp8_mha}"
        )


@dataclass()
class MXFP8BlockScaling(Recipe):
    """
    Use the MXFP8 scaling factor strategy.

    In this strategy, tensors are scaled in blockwise fashion. Each group
    of 32 consecutive values is scaled together using their own scaling
    factor. The type of the scaling factor is E8M0 (8 bits of exponent,
    0 bits of mantissa), equivalent to scaling by a power of 2.

    Since the scaling happens in a particular direction (either rowwise
    or columnwise), in this recipe the quantized tensor and its transpose
    are not numerically equivalent. Due to this, when Transformer Engine
    needs both the MXFP8 tensor and its transpose (e.g. to calculate both
    forward and backward pass), during the quantization both versions are
    computed from the high precision input to avoid double quantization
    errors.

    Parameters
    ----------
    fp8_format : {Format.E4M3, Format.HYBRID}, default = Format.E4M3
                Controls the FP8 data format used during forward and backward
                pass.
    """

    margin: int = 0
    fp8_format: Format = Format.E4M3
    fp8_dpa: bool = False
    fp8_mha: bool = False

    def __post_init__(self) -> None:
        assert self.fp8_format != Format.E5M2, "Pure E5M2 training is not supported."

    def __repr__(self) -> str:
        return f"margin={self.margin}, format={str(self.fp8_format).split('.')[1]},"
